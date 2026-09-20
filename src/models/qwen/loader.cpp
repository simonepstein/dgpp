#include "models/qwen/loader.hpp"

#include "loaders/fp8_quant.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the gated
// residual sites and the final mixers, routers and shared gates, the whole
// indexer, the PLE's norms/conv/buffers and the table's scale, the head
// norms of GDN and QSA, the draft head's own tensors, and the embedding
// (a row gather; sharding it buys nothing in v1). Everything else is a
// slice: GDN/QSA projections, both expert classes, the PLE projections,
// the n-gram shards, the lm head under VocabSharded.
bool is_replicated(const QwenExpectedTensor& e) {
  // ModelOpt stores one scalar weight scale and one scalar activation scale
  // beside every NVFP4 matrix. The payload and per-group scales are sliced,
  // but these two scalars are rank-invariant metadata read by every rank.
  if (e.role == QwenTensorRole::Fp4Global || e.role == QwenTensorRole::InputScale)
    return true;
  switch (e.cls) {
    case QwenWeightClass::Embed:
    case QwenWeightClass::Mixer:
    case QwenWeightClass::Gr:
    case QwenWeightClass::Router:
    case QwenWeightClass::QsaIndexer:
    case QwenWeightClass::Ple:
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::PleTable:
      return e.role == QwenTensorRole::NgramScale;
    case QwenWeightClass::Gdn:
      return ends_with(e.name, "linear_attn.norm.weight");
    case QwenWeightClass::Qsa:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case QwenWeightClass::LmHead:
    case QwenWeightClass::SharedExpert:
    case QwenWeightClass::RoutedExpert:
      return false;
  }
  return false;
}

// auto_gptq packs 32/bits codes per I32 word and so does the engine's
// packed form, which is what makes the repack a transpose. Only int4 is
// implemented (models/qwen/config.cpp refuses the rest).
constexpr int64_t kGptqCodesPerWord = 8;

int gcd_int(int a, int b) { return std::gcd(a, b); }

std::pair<int, int> lm_head_slice(const QwenTextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

// The n-gram table's residency: process-wide, set before
// any stream is built (the memory plan reads it too).
bool g_ngram_table_mmap = false;
// The dense stack's form (engine.dense_weights = "fp8", 2026-09-10).
bool g_dense_weights_fp8 = false;
std::string g_ngram_table_dir;
}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct QwenLoaderFamily::Builder : WeightBuilder<QwenExpectedTensor> {
  const QwenTextConfig& cfg;
  const QwenLocalGeometry& geo;
  QwenLayerResident& out;

  Builder(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
           const std::vector<QwenExpectedTensor>& table_,
           const std::unordered_map<std::string, const QwenExpectedTensor*>& by_name_,
           LayerBump& bump_, QwenLayerResident& out_,
           const std::unordered_map<std::string, const TensorInfo*>& tensors_,
           std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_,
                                          copy_, geo_.rank, geo_.world, "qwen loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const QwenExpectedTensor& e) const override { return is_replicated(e); }

  // A contiguous BF16 row range of a source matrix into a running
  // destination (the GDN's segmented q|k|v merge): rows [src_row, +rows)
  // of `name` land at `dst` + dst_row * width. Returns the source's row
  // width in elements.
  int64_t copy_rows_into(const std::string& name, int64_t src_row, int64_t rows,
                         uint16_t* dst, int64_t dst_row) {
    const QwenExpectedTensor& e = expected(name);
    const int64_t width = static_cast<int64_t>(e.numel()) / e.shape[0];
    check_range(name, src_row, rows, e.shape[0]);
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst) + static_cast<size_t>(dst_row) * width,
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(src_row) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
      // Not consumed: a later segment of the same tensor may follow.
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
    return width;
  }

  void build_gr(const std::string& p, QwenGrResident& g, bool combine) {
    g.hc_norm = load_bf16(p + "hc_norm.weight");
    if (g_dense_weights_fp8) {
      g.down_fp8 = load_bf16_fp8(p + "input_mix_weight_down.weight");
      g.up_fp8 = load_bf16_fp8(p + "input_mix_weight_up.weight");
    } else {
      g.down = load_bf16(p + "input_mix_weight_down.weight");
      g.up = load_bf16(p + "input_mix_weight_up.weight");
    }
    g.inject = combine ? load_bf16(p + "block_inject_weight.weight") : nullptr;
  }

  void build_gdn(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // in_proj_qkv rows [q | k | v] at the local geometry: this rank's key
    // heads' q and k rows, its value heads' v rows. world=1: the whole
    // matrix, byte for byte.
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    const std::string qkv_name = p + "in_proj_qkv.weight";
    if (dense_fp8()) {
      g.in_proj_qkv_fp8 = load_quant_rows_fused(
          qkv_name, {{r * lk * dk, lk * dk}, {K + r * lk * dk, lk * dk},
                     {2 * K + r * lv * dv, lv * dv}});
    } else if (g_dense_weights_fp8) {
      // The three segments assembled on the host, encoded into the bump.
      std::vector<uint16_t> merged;
      if (copy) {
        merged.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(H));
        const uint16_t* src = static_cast<const uint16_t*>(source(qkv_name).data);
        const auto seg = [&](int64_t src_row, int64_t rows, int64_t dst_row) {
          std::memcpy(merged.data() + static_cast<size_t>(dst_row) * H, src + static_cast<size_t>(src_row) * H,
                      static_cast<size_t>(rows) * H * 2);
        };
        seg(r * lk * dk, lk * dk, 0);
        seg(K + r * lk * dk, lk * dk, lk * dk);
        seg(2 * K + r * lv * dv, lv * dv, 2 * lk * dk);
      }
      note_read(expected(qkv_name), static_cast<size_t>(local_rows) * H * 2);
      g.in_proj_qkv_fp8 = encode_fp8(merged.data(), static_cast<size_t>(H), local_rows, H);
      if (copy) consumed(source(qkv_name));
    } else {
      uint16_t* qkv = static_cast<uint16_t*>(
          bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(H) * 2));
      copy_rows_into(qkv_name, r * lk * dk, lk * dk, qkv, 0);
      copy_rows_into(qkv_name, K + r * lk * dk, lk * dk, qkv, lk * dk);
      copy_rows_into(qkv_name, 2 * K + r * lv * dv, lv * dv, qkv, 2 * lk * dk);
      if (copy) consumed(source(qkv_name));
      g.in_proj_qkv = qkv;
    }
    // The conv channels follow the same three segments ([C, 1, w] rows).
    const int64_t w = cfg.gdn_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(w) * 2));
    const std::string conv_name = p + "conv1d.weight";
    copy_rows_into(conv_name, r * lk * dk, lk * dk, conv, 0);
    copy_rows_into(conv_name, K + r * lk * dk, lk * dk, conv, lk * dk);
    copy_rows_into(conv_name, 2 * K + r * lv * dv, lv * dv, conv, 2 * lk * dk);
    if (copy) consumed(source(conv_name));
    g.conv = conv;
    if (dense_fp8())
      g.in_proj_z_fp8 = load_quant_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv,
                                        slice_scale_block(lv * dv));
    else if (g_dense_weights_fp8)
      g.in_proj_z_fp8 = load_bf16_rows_fp8(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    else
      g.in_proj_z = load_bf16_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    g.in_proj_a = load_bf16_rows(p + "in_proj_a.weight", r * lv, lv);
    g.in_proj_b = load_bf16_rows(p + "in_proj_b.weight", r * lv, lv);
    g.a_log = load_bf16_as_f32(p + "A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "norm.weight");
    if (dense_fp8())
      g.out_proj_fp8 = load_quant_cols(p + "out_proj.weight", r * lv * dv, lv * dv,
                                       slice_scale_block(lv * dv));
    else if (g_dense_weights_fp8)
      g.out_proj_fp8 = load_bf16_cols_fp8(p + "out_proj.weight", r * lv * dv, lv * dv);
    else
      g.out_proj = load_bf16_cols(p + "out_proj.weight", r * lv * dv, lv * dv);
  }

  void build_qsa(const std::string& p) {
    const int64_t d = cfg.head_dim;
    QwenQsaResident& a = out.qsa;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * 2 * d, qn = static_cast<int64_t>(geo.local_heads) * 2 * d;
    const int64_t kv0 = static_cast<int64_t>(geo.kv_head_begin) * d, kvn = static_cast<int64_t>(geo.local_kv_heads) * d;
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * d, on = static_cast<int64_t>(geo.local_heads) * d;
    if (dense_fp8()) {
      a.q_proj_fp8 = load_quant_rows(p + "q_proj.weight", q0, qn, slice_scale_block(qn));
      a.k_proj_fp8 = load_quant_rows(p + "k_proj.weight", kv0, kvn, slice_scale_block(kvn));
      a.v_proj_fp8 = load_quant_rows(p + "v_proj.weight", kv0, kvn, slice_scale_block(kvn));
      a.o_proj_fp8 = load_quant_cols(p + "o_proj.weight", o0, on, slice_scale_block(on));
    } else if (g_dense_weights_fp8) {
      a.q_proj_fp8 = load_bf16_rows_fp8(p + "q_proj.weight", q0, qn);
      a.k_proj_fp8 = load_bf16_rows_fp8(p + "k_proj.weight", kv0, kvn);
      a.v_proj_fp8 = load_bf16_rows_fp8(p + "v_proj.weight", kv0, kvn);
      a.o_proj_fp8 = load_bf16_cols_fp8(p + "o_proj.weight", o0, on);
    } else {
      a.q_proj = load_bf16_rows(p + "q_proj.weight", q0, qn);
      a.k_proj = load_bf16_rows(p + "k_proj.weight", kv0, kvn);
      a.v_proj = load_bf16_rows(p + "v_proj.weight", kv0, kvn);
      a.o_proj = load_bf16_cols(p + "o_proj.weight", o0, on);
    }
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.k_norm = load_bf16(p + "k_norm.weight");
    if (g_dense_weights_fp8)
      a.index_qk_proj_fp8 = load_bf16_fp8(p + "indexer.index_qk_proj.weight");
    else
      a.index_qk_proj = load_bf16(p + "indexer.index_qk_proj.weight");
    a.index_q_norm = load_bf16(p + "indexer.q_layernorm.weight");
    a.index_k_norm = load_bf16(p + "indexer.k_layernorm.weight");
  }

  void build_moe(const std::string& p) {
    QwenMoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    m.shared_gate = load_bf16(p + "shared_expert_gate.weight");
    const int64_t S = geo.local_shared_inter, I = geo.local_inter;
    const int64_t r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.scale_block = geo.scale_block;
    const std::string sp = p + "shared_expert.";
    if (dense_fp8()) {
      const int sb = slice_scale_block(S);
      m.shared_fp8[0] = load_quant_rows(sp + "gate_proj.weight", r * S, S, sb);
      m.shared_fp8[1] = load_quant_rows(sp + "up_proj.weight", r * S, S, sb);
      m.shared_fp8[2] = load_quant_cols(sp + "down_proj.weight", r * S, S, sb);
    } else if (g_dense_weights_fp8) {
      m.shared_fp8[0] = load_bf16_rows_fp8(sp + "gate_proj.weight", r * S, S);
      m.shared_fp8[1] = load_bf16_rows_fp8(sp + "up_proj.weight", r * S, S);
      m.shared_fp8[2] = load_bf16_cols_fp8(sp + "down_proj.weight", r * S, S);
    } else {
      m.shared[0] = load_bf16_rows(sp + "gate_proj.weight", r * S, S);
      m.shared[1] = load_bf16_rows(sp + "up_proj.weight", r * S, S);
      m.shared[2] = load_bf16_cols(sp + "down_proj.weight", r * S, S);
    }
    const int E = cfg.num_experts;
    // The NVFP4 release's backbone experts (the MTP layer's stay FP8): the
    // modelopt triple per matrix, sliced on the intermediate axis like the
    // FP8 form — rows for gate/up, whole 16-blocks of columns for down.
    if (cfg.experts_nvfp4 && p.rfind("mtp.", 0) != 0) {
      m.experts_fp4.resize(static_cast<size_t>(E) * 3);
      m.expert_globals = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * 3 * sizeof(float)));
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        float* g = m.expert_globals + static_cast<size_t>(e) * 3;
        m.experts_fp4[static_cast<size_t>(e) * 3 + 0] = load_fp4_rows_mo(ep + "gate_proj", r * I, I, g + 0);
        m.experts_fp4[static_cast<size_t>(e) * 3 + 1] = load_fp4_rows_mo(ep + "up_proj", r * I, I, g + 1);
        m.experts_fp4[static_cast<size_t>(e) * 3 + 2] = load_fp4_cols_mo(ep + "down_proj", r * I, I, g + 2);
      }
      return;
    }
    // The AutoRound release's backbone experts: the auto_gptq triple
    // transposed into the engine's packed form, sliced on the intermediate
    // axis like the other two — rows for gate/up, whole 64-groups of
    // columns for down.
    if (cfg.experts_packed && p.rfind("mtp.", 0) != 0) {
      m.experts_packed.resize(static_cast<size_t>(E) * 3);
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        m.experts_packed[static_cast<size_t>(e) * 3 + 0] =
            load_packq_rows_gptq(ep + "gate_proj", r * I, I);
        m.experts_packed[static_cast<size_t>(e) * 3 + 1] =
            load_packq_rows_gptq(ep + "up_proj", r * I, I);
        m.experts_packed[static_cast<size_t>(e) * 3 + 2] =
            load_packq_cols_gptq(ep + "down_proj", r * I, I);
      }
      return;
    }
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      // The AutoRound release leaves the draft layer's experts in BF16;
      // they take the same load-time FP8 encoding as every other BF16
      // dense matrix, so the draft layer stays on the FP8 expert path.
      if (cfg.experts_packed) {
        m.experts[static_cast<size_t>(e) * 3 + 0] =
            load_bf16_rows_fp8(ep + "gate_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 1] =
            load_bf16_rows_fp8(ep + "up_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 2] =
            load_bf16_cols_fp8(ep + "down_proj.weight", r * I, I);
        continue;
      }
      m.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", r * I, I, geo.scale_block);
    }
  }

  // ---- the checkpoint's own dense FP8 -------------------------------------
  // The AutoRound release ships the QSA and GDN projections and the shared
  // experts as e4m3 + an F32 128x128 grid rather than BF16, so those sites
  // slice the checkpoint instead of encoding it (the draft layer, which no
  // release quantizes, stays on the BF16 path).
  bool dense_fp8() const { return cfg.dense_stack_fp8 && out.layer != cfg.mtp_layer(); }
  // A slice at `rows` (start r * rows) re-blocks the scale grid at the
  // largest block that divides both, exactly as the routed experts do.
  static int slice_scale_block(int64_t rows) {
    return static_cast<int>(std::gcd<int64_t>(128, rows));
  }
  // Several row ranges of one quantized matrix as a single resident matrix
  // (the GDN's [q | k | v] stack). Payload rows are contiguous, so each
  // segment is a memcpy; the scale grid comes across a block at a time,
  // which needs every segment's start AND its destination row to be
  // aligned to the block — checked here rather than assumed.
  GlmQuantMatrix load_quant_rows_fused(const std::string& name,
                                       const std::vector<std::pair<int64_t, int64_t>>& segs) {
    const QwenExpectedTensor& e = expected(name);
    const QwenExpectedTensor& es = expected(name + "_scale_inv");
    const int64_t cols = e.shape[1];
    const int64_t sb = (cols + 127) / 128;
    int64_t rows = 0;
    int64_t block = 128;
    for (const auto& seg : segs) {
      rows += seg.second;
      block = std::gcd(block, seg.first);
      block = std::gcd(block, seg.second);
    }
    const int sbr = static_cast<int>(block);
    const int64_t scale_rows = (rows + sbr - 1) / sbr;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = sbr;
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(scale_rows) * sb * 4));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      int64_t dst = 0;
      for (const auto& seg : segs) {
        if (seg.first % sbr != 0 || dst % sbr != 0)
          fail("fused quantized rows of '" + name + "' cross a scale block");
        std::memcpy(hp + dst * cols,
                    static_cast<const uint8_t*>(tp.data) + seg.first * cols,
                    static_cast<size_t>(seg.second) * cols);
        for (int64_t r = 0; r < (seg.second + sbr - 1) / sbr; ++r)
          copy_scale_row(ts, ((seg.first + r * sbr) / 128) * sb, sb,
                         hs + (dst / sbr + r) * sb);
        dst += seg.second;
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(rows) * cols +
                     static_cast<size_t>(scale_rows) * sb * dtype_size(es.dtype));
    return q;
  }

  // ---- auto_gptq int4 slices, repacked ------------------------------------
  // `base` names the matrix ("...gate_proj"): base.qweight I32 [K/8, N],
  // base.scales F16 [K/group, N], base.qzeros I32 [K/group, N/8].
  //
  // The checkpoint and the engine pack a code the same way — 8 per I32
  // along K, the lowest k in the low nibble — so the 32-bit words are
  // already the engine's and THE REPACK IS A TRANSPOSE: qweight is
  // k-major, the engine's weight_packed is n-major. Two things change
  // around it: the checkpoint's 128-element group becomes two of the
  // engine's 64-element groups carrying the same scale (exact), and the
  // F16 scale is rounded once to BF16 (measured: 0.16% rms, 0.39% worst,
  // the 2^-8 bound — tools/gptq_repack.py verify).
  //
  // Symmetric quantization writes a constant zero point of 2^(bits-1),
  // which is the offset the engine's codes already carry, so qzeros is
  // checked and dropped rather than assumed away.
  void check_gptq_zeros(const std::string& base) {
    const QwenExpectedTensor& ez = expected(base + ".qzeros");
    if (copy) {
      const TensorInfo& t = source(ez.name);
      const uint32_t* z = static_cast<const uint32_t*>(t.data);
      const size_t n = ez.numel();
      for (size_t i = 0; i < n; ++i)
        if (z[i] != 0x77777777u)
          fail("'" + ez.name + "' is not the symmetric constant: the loader implements only "
               "sym=true, whose zero point is 2^(bits-1) and is already in the codes");
      consumed(t);
    }
    note_read(ez, ez.numel() * 4);
  }
  // The geometry the three tensors agree on, or a failure naming the base.
  struct GptqSource {
    const QwenExpectedTensor* words;
    const QwenExpectedTensor* scales;
    int64_t N, K;
  };
  GptqSource gptq_source(const std::string& base) {
    GptqSource s;
    s.words = &expected(base + ".qweight");
    s.scales = &expected(base + ".scales");
    if (s.words->shape.size() != 2 || s.scales->shape.size() != 2)
      fail("'" + base + "' is not a packed matrix");
    s.N = s.words->shape[1];
    s.K = s.words->shape[0] * kGptqCodesPerWord;
    packed_check_cols(s.K, cfg.packed_bits, who.c_str());
    if (s.scales->shape[0] != s.K / cfg.packed_group || s.scales->shape[1] != s.N)
      fail("packed scale geometry mismatch on " + base);
    return s;
  }
  GlmPackedMatrix alloc_packed(int64_t rows, int64_t cols) {
    GlmPackedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.bits = cfg.packed_bits;
    q.packed = static_cast<const uint32_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(q.packed_cols()) * 4));
    q.scales = static_cast<const uint16_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(q.scale_cols()) * 2));
    return q;
  }
  // One row of the engine's scales from the checkpoint's column of them:
  // `src` steps by N, each value covering group/64 of the engine's groups.
  void repack_scale_row(const uint16_t* src, int64_t N, int64_t k_begin, int64_t cols,
                        uint16_t* dst) {
    for (int64_t g = 0; g < cols / kPackedGroup; ++g) {
      const int64_t src_g = (k_begin + g * kPackedGroup) / cfg.packed_group;
      dst[g] = float_to_bf16_bits(f16_bits_to_float(src[src_g * N]));
    }
  }
  // Rows [row_start, +rows) of the [N, K] matrix: a slice of qweight's
  // SECOND axis, gathered into contiguous engine rows.
  GlmPackedMatrix load_packq_rows_gptq(const std::string& base, int64_t row_start, int64_t rows) {
    const GptqSource s = gptq_source(base);
    check_range(base, row_start, rows, s.N);
    GlmPackedMatrix q = alloc_packed(rows, s.K);
    const int64_t kw = s.K / kGptqCodesPerWord;
    if (copy) {
      const TensorInfo& tw = source(s.words->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint32_t* sw = static_cast<const uint32_t*>(tw.data);
      const uint16_t* ss = static_cast<const uint16_t*>(ts.data);
      uint32_t* dw = bump.host(const_cast<uint32_t*>(q.packed));
      uint16_t* ds = bump.host(const_cast<uint16_t*>(q.scales));
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t n = row_start + r;
        for (int64_t w = 0; w < kw; ++w) dw[r * kw + w] = sw[w * s.N + n];
        repack_scale_row(ss + n, s.N, 0, s.K, ds + r * q.scale_cols());
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*s.words, static_cast<size_t>(rows) * static_cast<size_t>(kw) * 4);
    note_read(*s.scales, static_cast<size_t>(rows) * static_cast<size_t>(s.K / cfg.packed_group) * 2);
    check_gptq_zeros(base);
    return q;
  }
  // Columns [col_start, +cols) of every row: a slice of qweight's FIRST
  // axis, so whole words. The start must be on one of the engine's
  // 64-element groups, which is inside one of the checkpoint's.
  GlmPackedMatrix load_packq_cols_gptq(const std::string& base, int64_t col_start, int64_t cols) {
    const GptqSource s = gptq_source(base);
    packed_check_cols(cols, cfg.packed_bits, who.c_str());
    if (col_start % kPackedGroup != 0)
      fail("packed column slice of '" + base + "' must start on a " +
           std::to_string(kPackedGroup) + "-element group boundary");
    check_range(base, col_start, cols, s.K);
    GlmPackedMatrix q = alloc_packed(s.N, cols);
    const int64_t kw = cols / kGptqCodesPerWord, w0 = col_start / kGptqCodesPerWord;
    if (copy) {
      const TensorInfo& tw = source(s.words->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint32_t* sw = static_cast<const uint32_t*>(tw.data);
      const uint16_t* ss = static_cast<const uint16_t*>(ts.data);
      uint32_t* dw = bump.host(const_cast<uint32_t*>(q.packed));
      uint16_t* ds = bump.host(const_cast<uint16_t*>(q.scales));
      for (int64_t r = 0; r < s.N; ++r) {
        for (int64_t w = 0; w < kw; ++w) dw[r * kw + w] = sw[(w0 + w) * s.N + r];
        repack_scale_row(ss + r, s.N, col_start, cols, ds + r * q.scale_cols());
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*s.words, static_cast<size_t>(s.N) * static_cast<size_t>(kw) * 4);
    note_read(*s.scales, static_cast<size_t>(s.N) * static_cast<size_t>(cols / cfg.packed_group) * 2);
    check_gptq_zeros(base);
    return q;
  }

  // ---- NVFP4 slices in the modelopt layout (as models/glm4/loader.cpp) ------
  // `base` names the matrix ("...gate_proj"): base.weight U8 [N, K/2],
  // base.weight_scale e4m3 [N, K/16], base.weight_scale_2 F32 [] — stored
  // in `global` as its reciprocal (the kernels divide by it once), and the
  // unused base.input_scale consumed so the checkpoint reconciles.
  void load_global_reciprocal_into(const std::string& base, float* slot) {
    const QwenExpectedTensor& eg = expected(base + ".weight_scale_2");
    if (copy) {
      const TensorInfo& t = source(eg.name);
      float ws2;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2))
        fail("'" + eg.name + "' is not a positive finite scale");
      const float inv = 1.0f / ws2;
      std::memcpy(bump.host(slot), &inv, 4);
      consumed(t);
    }
    note_read(eg, 4);
    (void)load_raw(base + ".input_scale");
  }
  GlmFp4Matrix load_fp4_rows_mo(const std::string& base, int64_t row_start, int64_t rows, float* global) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const QwenExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0];
    const int64_t cols = ep.shape[1] * 2;
    fp4_check_cols(cols, who.c_str());
    check_range(base, row_start, rows, N);
    if (es.shape[0] != N || es.shape[1] != cols / kFp4Group) fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(rows) * pc);
    note_read(es, static_cast<size_t>(rows) * sc);
    load_global_reciprocal_into(base, global);
    q.global_scale = global;
    return q;
  }
  GlmFp4Matrix load_fp4_cols_mo(const std::string& base, int64_t col_start, int64_t cols, float* global) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const QwenExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0];
    const int64_t full = ep.shape[1] * 2;
    fp4_check_cols(full, who.c_str());
    fp4_check_cols(cols, who.c_str());
    if (col_start % kFp4Group != 0) fail("NVFP4 column slice of " + base + " is not 16-aligned");
    check_range(base, col_start, cols, full);
    if (es.shape[0] != N || es.shape[1] != full / kFp4Group) fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc_full = static_cast<size_t>(full / 2), sc_full = static_cast<size_t>(full / kFp4Group);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == full) {
        std::memcpy(hp, sp, static_cast<size_t>(N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(N) * sc);
      } else {
        for (int64_t row = 0; row < N; ++row) {
          std::memcpy(hp + row * pc, sp + row * pc_full + col_start / 2, pc);
          std::memcpy(hs + row * sc, ss + row * sc_full + col_start / kFp4Group, sc);
        }
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(N) * pc);
    note_read(es, static_cast<size_t>(N) * sc);
    load_global_reciprocal_into(base, global);
    q.global_scale = global;
    return q;
  }

  void build_ple(const std::string& p) {
    QwenPleResident& l = out.ple;
    const QwenNgramGeometry g = cfg.ngram_geometry();
    const int64_t hd = g.head_dim;
    l.hash_heads = geo.hash_heads;
    l.hash_head_begin = geo.hash_head_begin;
    l.row_begin = geo.table_row_begin;
    l.rows = geo.table_rows;
    const int64_t c0 = static_cast<int64_t>(geo.hash_head_begin) * hd;
    const int64_t cn = static_cast<int64_t>(geo.hash_heads) * hd;
    if (g_dense_weights_fp8) {
      l.key_proj_fp8 = load_bf16_cols_fp8(p + "key_proj.weight", c0, cn);
      l.value_proj_fp8 = load_bf16_cols_fp8(p + "value_proj.weight", c0, cn);
    } else {
      l.key_proj = load_bf16_cols(p + "key_proj.weight", c0, cn);
      l.value_proj = load_bf16_cols(p + "value_proj.weight", c0, cn);
    }
    l.norm_key = load_bf16(p + "norm_key.weight");
    l.norm_query = load_bf16(p + "norm_query.weight");
    l.norm_conv = load_bf16(p + "norm_conv.weight");
    l.conv = load_bf16(p + "conv1d.weight");
    // The hash buffers ride along (replicated; the stream verified them
    // against the config at construction), and the table's scale.
    const std::string ep = p + "ple_embedding.";
    (void)load_raw(ep + "layer_multipliers");
    (void)load_raw(ep + "ngram_heads_offsets");
    (void)load_raw(ep + "ngram_heads_vocab_sizes");
    const std::string scale_name = ep + "ngram_embedding.weight_scale";
    (void)load_raw(scale_name);
    if (copy) {
      uint16_t bits;
      std::memcpy(&bits, tensors.at(scale_name)->data, 2);
      l.table_scale = bf16_bits_to_float(bits);
    }
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      fail("layer index out of range: " + std::to_string(layer));
    const bool is_mtp = layer == cfg.mtp_layer();
    const std::string p = qwen_layer_prefix(cfg, layer);
    out.layer = layer;
    out.kind = is_mtp ? QwenLayerKind::Qsa : cfg.layers[layer];
    out.has_ple = !is_mtp && layer == cfg.ple_layer();
    if (out.has_ple) build_ple(p + "ple.");
    build_gr(p + "attn_hyper_connection.", out.attn_gr, true);
    if (out.kind == QwenLayerKind::Gdn)
      build_gdn(p + "linear_attn.");
    else
      build_qsa(p + "self_attn.");
    build_gr(p + "mlp_hyper_connection.", out.mlp_gr, true);
    build_moe(p + "mlp.");
  }
};

// ---------------------------------------------------------------------------

QwenLocalGeometry QwenLocalGeometry::from_config(const QwenTextConfig& cfg, int rank, int world,
                                                 QwenHeadSharding head) {
  if (world > 1) qwen_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("qwen loader: rank/world out of range");
  QwenLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_key_heads = cfg.gdn_key_heads / world;
  g.local_value_heads = cfg.gdn_value_heads / world;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  // The rank's query heads must belong to its kv head(s).
  const int q_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
  if (g.head_begin / q_per_kv != g.kv_head_begin ||
      (g.head_begin + g.local_heads - 1) / q_per_kv != g.kv_head_begin + g.local_kv_heads - 1)
    throw std::invalid_argument("qwen loader: the query heads of a rank straddle kv heads");
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_intermediate_size / world;
  g.scale_block = gcd_int(128, static_cast<int>(g.local_inter));
  if (!cfg.ple_layer_ids.empty()) {
    const QwenNgramGeometry ng = cfg.ngram_geometry();
    g.hash_heads = ng.heads / world;
    g.hash_head_begin = g.hash_heads * rank;
    g.table_row_begin = ng.head_offset[static_cast<size_t>(g.hash_head_begin)];
    const int last = g.hash_head_begin + g.hash_heads;  // one past
    const int64_t end = last < ng.heads ? ng.head_offset[static_cast<size_t>(last)] : ng.total_rows;
    g.table_rows = end - g.table_row_begin;
  }
  if (head == QwenHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void QwenLoaderFamily::validate_binding(const QwenTextConfig& cfg, const PresentMap& present) {
  const QwenBindReport rep = qwen_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "qwen loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The checkpoint's stored hash buffers must equal the config's derivation
// (docs §1.7): a disagreement means a different hash, i.e. a different
// model, and is refused before a byte of weights moves.
void QwenLoaderFamily::check_sources(const QwenTextConfig& cfg_, const LoaderTensorMap& tensors_) {
  if (cfg_.ple_layer_ids.empty()) return;
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  auto read_i64 = [&](const std::string& name, size_t n) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.numel() != n) throw std::runtime_error("qwen loader: '" + name + "' has the wrong length");
    std::vector<int64_t> v(n);
    std::memcpy(v.data(), t.data, n * 8);
    return v;
  };
  const std::vector<int64_t> mult = read_i64(p + "layer_multipliers", static_cast<size_t>(cfg_.ngram_size));
  const std::vector<int64_t> sizes = read_i64(p + "ngram_heads_vocab_sizes", static_cast<size_t>(g.heads));
  const std::vector<int64_t> offs = read_i64(p + "ngram_heads_offsets", static_cast<size_t>(g.heads));
  if (mult != g.multipliers || sizes != g.head_vocab || offs != g.head_offset)
    throw std::runtime_error(
        "qwen loader: the checkpoint's n-gram hash buffers (layer_multipliers, "
        "ngram_heads_vocab_sizes, ngram_heads_offsets) disagree with the config's "
        "derivation — a different hash, refused");
}

bool QwenLoaderFamily::digest_included(const QwenExpectedTensor& e) { return is_replicated(e); }

// The packed column slices (a pack reads its source after the builder
// returns): the classes whose sources the one-pass load drops afterwards.
bool QwenLoaderFamily::discard_after_pack(const QwenExpectedTensor& e) {
  return e.cls == QwenWeightClass::Ple || e.cls == QwenWeightClass::Gdn ||
         e.cls == QwenWeightClass::Qsa || e.cls == QwenWeightClass::SharedExpert;
}

size_t QwenLoaderFamily::globals_bytes(const QwenTextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t W = static_cast<size_t>(cfg.hyper_width());
  const size_t r = static_cast<size_t>(cfg.hc_lowrank);
  // A dense [n, k] matrix's resident bytes: BF16, or block FP8 (codes + the
  // fp32 scale grid) under engine.dense_weights = "fp8".
  const auto dense = [](size_t n, size_t k) -> size_t {
    if (!g_dense_weights_fp8) return align_up_256(n * k * 2);
    return align_up_256(n * k) +
           align_up_256(static_cast<size_t>(fp8_quant::scale_rows(static_cast<int64_t>(n))) *
                        static_cast<size_t>(fp8_quant::scale_cols(static_cast<int64_t>(k))) * 4);
  };
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);                  // embed
  b += dense(static_cast<size_t>(QwenLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count), H);
  b += align_up_256(W * 2) + dense(r, W) + dense(W, r);                             // mixer
  if (cfg.mtp_layer() >= 0) {
    b += 2 * align_up_256(H * H * 2);                          // fc_embedding, fc_hidden
    b += align_up_256(H * 2) + align_up_256(W * 2);            // pre_fc norms
    b += align_up_256(W * 2) + dense(r, W) + dense(W, r);                        // mtp mixer
  }
  return b;
}

size_t QwenLoaderFamily::extra_resident_bytes(const QwenTextConfig& cfg, int rank, int world) {
  return QwenLayerStream::ngram_table_bytes(cfg, rank, world);
}

// The restored PLE layer's host-side scale: from the source when mapped.
void QwenLoaderFamily::after_restore(const QwenTextConfig& cfg, int layer,
                                     const LoaderTensorMap& tensors, QwenLayerResident& out) {
  if (!out.has_ple || tensors.empty()) return;
  const std::string name = qwen_layer_prefix(cfg, layer) + "ple.ple_embedding.ngram_embedding.weight_scale";
  uint16_t bits;
  std::memcpy(&bits, tensors.at(name)->data, 2);
  out.ple.table_scale = bf16_bits_to_float(bits);
}

void QwenLoaderFamily::build_globals(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
                                     const LoaderTensorMap& tensors_, LayerBump& bump,
                                     QwenGlobalsResident& globals_, uint64_t& source_bytes_,
                                     uint64_t& verbatim_bytes_, LoaderHeadSharding head_) {
  LayerBump* globals_bump_ = &bump;
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(t.nbytes()));
    std::memcpy(globals_bump_->host(dst), t.data, t.nbytes());
    source_bytes_ += t.nbytes();
    verbatim_bytes_ += t.nbytes();
    return dst;
  };
  // A BF16 global encoded to block FP8 into the globals bump (dense_weights fp8).
  auto encode_global_fp8 = [&](const std::string& name) -> GlmQuantMatrix {
    const TensorInfo& t = lookup(name);
    if (t.shape.size() != 2) throw std::runtime_error("qwen loader: '" + name + "' is not a matrix");
    const int64_t rows = t.shape[0], cols = t.shape[1];
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(globals_bump_->alloc(static_cast<size_t>(rows) * cols));
    q.scales = static_cast<const float*>(globals_bump_->alloc(
        static_cast<size_t>(fp8_quant::scale_rows(rows)) * fp8_quant::scale_cols(cols) * 4));
    fp8_quant::encode_block128(static_cast<const uint16_t*>(t.data), static_cast<size_t>(cols), rows, cols,
                               globals_bump_->host(const_cast<uint8_t*>(q.payload)),
                               globals_bump_->host(const_cast<float*>(q.scales)));
    source_bytes_ += t.nbytes();
    return q;
  };
  auto copy_gr = [&](const std::string& p, QwenGrResident& g) {
    g.hc_norm = copy_global(p + "hc_norm.weight");
    if (g_dense_weights_fp8) {
      g.down_fp8 = encode_global_fp8(p + "input_mix_weight_down.weight");
      g.up_fp8 = encode_global_fp8(p + "input_mix_weight_up.weight");
    } else {
      g.down = copy_global(p + "input_mix_weight_down.weight");
      g.up = copy_global(p + "input_mix_weight_up.weight");
    }
    g.inject = nullptr;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  {
    const size_t row_bytes = static_cast<size_t>(cfg_.hidden_size) * 2;
    const int begin = geo_.lm_vocab_begin, count = geo_.lm_vocab_count;
    const int64_t H = cfg_.hidden_size;
    // The AutoRound release quantizes the head and ships no BF16 copy of
    // it, so its rows are dequantized here — into the BF16 head, or a
    // 128-row band at a time through the FP8 encoder (the band is the
    // scale grid's own row block, so the encoding is the whole matrix's).
    // Both head forms below are otherwise untouched.
    const bool packed_head = cfg_.lm_head_packed;
    const TensorInfo* t = packed_head ? nullptr : &lookup("lm_head.weight");
    const auto dequant_head_rows = [&](int64_t row_start, int64_t rows, uint16_t* dst) {
      const TensorInfo& tw = lookup("lm_head.qweight");
      const TensorInfo& ts = lookup("lm_head.scales");
      const int64_t N = cfg_.vocab_size, group = cfg_.packed_group;
      const uint32_t* sw = static_cast<const uint32_t*>(tw.data);
      const uint16_t* ss = static_cast<const uint16_t*>(ts.data);
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t n = row_start + r;
        for (int64_t k = 0; k < H; ++k) {
          const uint32_t word = sw[(k / kGptqCodesPerWord) * N + n];
          const int code =
              static_cast<int>((word >> (4 * (k % kGptqCodesPerWord))) & 0xFu) - 8;
          const float scale = f16_bits_to_float(ss[(k / group) * N + n]);
          dst[r * H + k] = float_to_bf16_bits(static_cast<float>(code) * scale);
        }
      }
    };
    if (packed_head) {
      // The same symmetric-constant check the experts get: a head with real
      // zero points would dequantize to the wrong numbers, silently.
      const TensorInfo& tz = lookup("lm_head.qzeros");
      const uint32_t* z = static_cast<const uint32_t*>(tz.data);
      for (size_t i = 0; i < tz.nbytes() / 4; ++i)
        if (z[i] != 0x77777777u)
          throw std::runtime_error(
              "qwen loader: 'lm_head.qzeros' is not the symmetric constant (the loader "
              "implements only sym=true, whose zero point is already in the codes)");
    }
    if (g_dense_weights_fp8) {
      GlmQuantMatrix q;
      q.rows = count;
      q.cols = H;
      q.payload = static_cast<const uint8_t*>(globals_bump_->alloc(static_cast<size_t>(count) * H));
      q.scales = static_cast<const float*>(globals_bump_->alloc(
          static_cast<size_t>(fp8_quant::scale_rows(count)) * fp8_quant::scale_cols(H) * 4));
      uint8_t* hp = globals_bump_->host(const_cast<uint8_t*>(q.payload));
      float* hs = globals_bump_->host(const_cast<float*>(q.scales));
      if (packed_head) {
        const int64_t band = 128, sc = fp8_quant::scale_cols(H);
        std::vector<uint16_t> rows_bf16(static_cast<size_t>(band) * H);
        for (int64_t r0 = 0; r0 < count; r0 += band) {
          const int64_t n = std::min<int64_t>(band, count - r0);
          dequant_head_rows(begin + r0, n, rows_bf16.data());
          fp8_quant::encode_block128(rows_bf16.data(), static_cast<size_t>(H), n, H,
                                     hp + static_cast<size_t>(r0) * H,
                                     hs + static_cast<size_t>(r0 / band) * sc);
        }
      } else {
        fp8_quant::encode_block128(static_cast<const uint16_t*>(t->data) + static_cast<size_t>(begin) * H,
                                   static_cast<size_t>(H), count, H, hp, hs);
      }
      globals_.lm_head_fp8 = q;
    } else {
      uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(static_cast<size_t>(count) * row_bytes));
      if (packed_head) {
        dequant_head_rows(begin, count, globals_bump_->host(dst));
      } else {
        std::memcpy(globals_bump_->host(dst),
                    static_cast<const uint8_t*>(t->data) + static_cast<size_t>(begin) * row_bytes,
                    static_cast<size_t>(count) * row_bytes);
        if (head_ == LoaderHeadSharding::Full) verbatim_bytes_ += static_cast<size_t>(count) * row_bytes;
      }
      globals_.lm_head = dst;
    }
    // The int4 head's source is half a byte per weight plus its scales.
    source_bytes_ += packed_head
                         ? static_cast<size_t>(count) * H / 2 +
                               static_cast<size_t>(count) * H / cfg_.packed_group * 2
                         : static_cast<size_t>(count) * row_bytes;
    globals_.lm_vocab_begin = begin;
    globals_.lm_vocab_count = count;
  }
  copy_gr("model.language_model.hyper_connection_mixer.", globals_.mixer);
  if (cfg_.mtp_layer() >= 0) {
    globals_.mtp_fc_embedding = copy_global("mtp.fc_embedding.weight");
    globals_.mtp_fc_hidden = copy_global("mtp.fc_hidden.weight");
    globals_.mtp_pre_fc_norm_embedding = copy_global("mtp.pre_fc_norm_embedding.weight");
    globals_.mtp_pre_fc_norm_hidden = copy_global("mtp.pre_fc_norm_hidden.weight");
    copy_gr("mtp.hyper_connection_mixer.", globals_.mtp_mixer);
  }
}

template class ResidentLayerStream<QwenLoaderFamily>;

// ---------------------------------------------------------------------------

// The table's tensors are the ones under `ple_embedding.ngram_embedding.`
// — the shards and the per-tensor scale. Everything else the companion
// carries (this release's ships a second copy of layer 1) is left behind.
void QwenLayerStream::set_ngram_table_dir(const std::string& dir) {
  g_ngram_table_dir = dir;
  clear_companion_dirs();
  if (!dir.empty()) add_companion_dir(dir, "ple_embedding.ngram_embedding.");
}
const std::string& QwenLayerStream::ngram_table_dir() { return g_ngram_table_dir; }

QwenLayerStream::QwenLayerStream(const QwenTextConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, QwenResidency residency,
                                 QwenHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<QwenLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  // A release whose dense stack is already e4m3 has no BF16 to give back:
  // engine.dense_weights = "bf16" would need a dequantizing load, which is
  // not implemented. Refuse by name rather than quietly serving FP8.
  if (cfg.dense_stack_fp8 && !g_dense_weights_fp8)
    throw std::runtime_error(
        "qwen loader: engine.dense_weights = \"bf16\" but this release ships the dense "
        "stack in block FP8 (no BF16 copy exists in the checkpoint); set it to \"fp8\"");
  open_resident_image();
}

QwenLayerStream::~QwenLayerStream() {
  if (table_device_) cudaFree(table_device_);
}

size_t QwenLayerStream::ngram_table_bytes(const QwenTextConfig& cfg, int rank, int world) {
  if (g_ngram_table_mmap) return 0;
  if (cfg.ple_layer_ids.empty()) return 0;
  const QwenLocalGeometry g = QwenLocalGeometry::from_config(cfg, rank, world, QwenHeadSharding::Full);
  return static_cast<size_t>(g.table_rows) * static_cast<size_t>(cfg.ngram_geometry().head_dim);
}

void QwenLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& QwenLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

// ---- the n-gram table ----------------------------------------------------------

void QwenLayerStream::set_ngram_table_mmap(bool on) { g_ngram_table_mmap = on; }
bool QwenLayerStream::ngram_table_mmap() { return g_ngram_table_mmap; }

// ---- the dense stack's form (engine.dense_weights) ------------------------------

void QwenLayerStream::set_dense_weights_fp8(bool on) { g_dense_weights_fp8 = on; }
bool QwenLayerStream::dense_weights_fp8() { return g_dense_weights_fp8; }
uint64_t QwenLoaderFamily::loader_format() { return g_dense_weights_fp8 ? 2 : 1; }

// ---- QwenNgramTableMmap ---------------------------------------------------------

QwenNgramTableMmap::QwenNgramTableMmap(std::vector<Part> parts, int64_t capacity, int head_dim)
    : parts_(std::move(parts)), capacity_(capacity), head_dim_(head_dim) {
  if (capacity_ <= 0 || head_dim_ <= 0 || parts_.empty())
    throw std::invalid_argument("QwenNgramTableMmap: empty geometry");
  std::unordered_map<std::string, size_t> by_path;
  part_base_.resize(parts_.size(), nullptr);
  for (size_t s = 0; s < parts_.size(); ++s) {
    const Part& part = parts_[s];
    auto it = by_path.find(part.path);
    if (it == by_path.end()) {
      Mapping m;
      m.fd = ::open(part.path.c_str(), O_RDONLY | O_CLOEXEC);
      if (m.fd < 0) throw std::runtime_error("QwenNgramTableMmap: cannot open " + part.path);
      struct stat st {};
      if (fstat(m.fd, &st) != 0) { ::close(m.fd); throw std::runtime_error("QwenNgramTableMmap: fstat " + part.path); }
      m.len = static_cast<size_t>(st.st_size);
      void* map = mmap(nullptr, m.len, PROT_READ, MAP_SHARED, m.fd, 0);
      if (map == MAP_FAILED) { ::close(m.fd); throw std::runtime_error("QwenNgramTableMmap: mmap " + part.path); }
      m.base = static_cast<uint8_t*>(map);
      // Rows are read one 160-byte record at a time from anywhere in 48 GB:
      // no readahead, or every fault pulls 128 KB for 160 bytes.
      madvise(m.base, m.len, MADV_RANDOM);
      mapped_bytes_ += m.len;
      maps_.push_back(m);
      it = by_path.emplace(part.path, maps_.size() - 1).first;
    }
    const Mapping& m = maps_[it->second];
    if (part.data_begin + static_cast<uint64_t>(part.rows) * head_dim_ > m.len)
      throw std::runtime_error("QwenNgramTableMmap: part " + std::to_string(s) + " past the end of " + part.path);
    part_base_[s] = m.base + part.data_begin;
  }
}

QwenNgramTableMmap::~QwenNgramTableMmap() {
  for (Mapping& m : maps_) {
    if (m.base) munmap(m.base, m.len);
    if (m.fd >= 0) ::close(m.fd);
  }
}

const uint8_t* QwenNgramTableMmap::row(int64_t row) const {
  const int64_t s = row / capacity_, r = row - s * capacity_;
  if (row < 0 || s >= static_cast<int64_t>(parts_.size()) || r >= parts_[static_cast<size_t>(s)].rows)
    throw std::out_of_range("QwenNgramTableMmap: row " + std::to_string(row) + " outside the table");
  return part_base_[static_cast<size_t>(s)] + static_cast<size_t>(r) * head_dim_;
}

void QwenNgramTableMmap::gather(const int32_t* ids, int n, int heads, int head_begin, int heads_local,
                                uint8_t* dst) const {
  if (n <= 0) return;
  const int64_t total = static_cast<int64_t>(n) * heads_local;
  // Every row's page asked for up front (the kernel issues the reads in
  // parallel), then the copies: a decode step's 32 rows in one thread, a
  // prefill chunk's tens of thousands across a few.
  auto copy_range = [&](int64_t lo, int64_t hi) {
    for (int64_t pair = lo; pair < hi; ++pair) {
      const int64_t t = pair / heads_local, hl = pair - t * heads_local;
      const int32_t id = ids[t * heads + head_begin + hl];
      std::memcpy(dst + static_cast<size_t>(pair) * head_dim_, row(id), static_cast<size_t>(head_dim_));
    }
  };
  for (int64_t pair = 0; pair < total; ++pair) {
    const int64_t t = pair / heads_local, hl = pair - t * heads_local;
    const uint8_t* p = row(ids[t * heads + head_begin + hl]);
    const uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{4095};
    madvise(reinterpret_cast<void*>(page), 4096 + static_cast<size_t>(head_dim_), MADV_WILLNEED);
  }
  constexpr int64_t kPerThread = 256;
  if (total <= kPerThread) {
    copy_range(0, total);
    return;
  }
  const int threads = static_cast<int>(std::min<int64_t>(16, (total + kPerThread - 1) / kPerThread));
  const int64_t span = (total + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w) {
    const int64_t lo = w * span, hi = std::min(total, lo + span);
    if (lo < hi) pool.emplace_back(copy_range, lo, hi);
  }
  for (auto& th : pool) th.join();
}

const QwenNgramTableResident& QwenLayerStream::load_ngram_table() {
  if (table_.rows_e4m3 || mmap_table_ || cfg_.ple_layer_ids.empty()) return table_;
  if (sources_released_)
    throw std::runtime_error("qwen loader: load_ngram_table after the checkpoint sources were released");
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const int64_t hd = g.head_dim;
  const int64_t row_begin = geo_.table_row_begin, rows = geo_.table_rows;
  if (g_ngram_table_mmap) {
    // The table stays in its shard(s): the parts' file offsets, mapped by
    // the table object; the resident view keeps the geometry and the scale
    // (the staged gather's), rows_e4m3 null.
    const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
    std::vector<QwenNgramTableMmap::Part> parts;
    for (int s = 0; s < cfg_.split_ngram_parts; ++s) {
      const std::string name = p + "ngram_embedding.shard_" + std::to_string(s) + ".weight";
      auto it = tensors_.find(name);
      if (it == tensors_.end() || !it->second || !it->second->owner)
        throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
      const TensorInfo& t = *it->second;
      parts.push_back(QwenNgramTableMmap::Part{t.owner->path(), t.data_begin, t.shape[0]});
    }
    mmap_table_ = std::make_unique<QwenNgramTableMmap>(std::move(parts), qwen_ngram_shard_capacity(cfg_),
                                                       static_cast<int>(hd));
    {
      const std::string name = p + "ngram_embedding.weight_scale";
      auto it = tensors_.find(name);
      if (it == tensors_.end() || !it->second) throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
      uint16_t bits;
      std::memcpy(&bits, it->second->data, 2);
      table_.scale = bf16_bits_to_float(bits);
    }
    table_.rows_e4m3 = nullptr;
    table_.mmap = mmap_table_.get();
    table_.row_begin = row_begin;
    table_.rows = rows;
    table_.head_dim = static_cast<int>(hd);
    table_.bytes = 0;
    DGPP_LOG_INFO("qwen loader: rank {} n-gram table rows [{}, {}) mmap'ed from the checkpoint ({:.2f} GiB mapped, "
                  "scale {})",
                  rank_, row_begin, row_begin + rows, mmap_table_->mapped_bytes() / (1024.0 * 1024.0 * 1024.0),
                  table_.scale);
    return table_;
  }
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(hd);
  DGPP_CUDA_OK(cudaMalloc(&table_device_, bytes));
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  const int64_t cap = qwen_ngram_shard_capacity(cfg_);
  const int64_t row_end = row_begin + rows;
  uint8_t* dst = static_cast<uint8_t*>(table_device_);
  size_t done = 0;
  for (int s = 0; s < cfg_.split_ngram_parts; ++s) {
    const int64_t s_begin = static_cast<int64_t>(s) * cap;
    const int64_t s_end = s_begin + qwen_ngram_shard_rows(cfg_, s);
    const int64_t lo = std::max(row_begin, s_begin), hi = std::min(row_end, s_end);
    if (lo >= hi) continue;
    const std::string name = p + "ngram_embedding.shard_" + std::to_string(s) + ".weight";
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.owner) t.owner->prefetch(t);
    const uint8_t* src = static_cast<const uint8_t*>(t.data) + static_cast<size_t>(lo - s_begin) * hd;
    size_t remaining = static_cast<size_t>(hi - lo) * static_cast<size_t>(hd);
    while (remaining > 0) {
      const size_t chunk = std::min(remaining, staging_bytes_);
      std::memcpy(staging_, src, chunk);
      DGPP_CUDA_OK(cudaMemcpyAsync(dst + done, staging_, chunk, cudaMemcpyHostToDevice, stream_));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      src += chunk;
      done += chunk;
      remaining -= chunk;
    }
    if (residency_ == QwenResidency::Resident && t.owner) t.owner->discard(t);
  }
  if (done != bytes)
    throw std::runtime_error("qwen loader: the n-gram shards did not cover this rank's rows");
  source_bytes_ += bytes;
  {
    const std::string name = p + "ngram_embedding.weight_scale";
    uint16_t bits;
    std::memcpy(&bits, tensors_.at(name)->data, 2);
    table_.scale = bf16_bits_to_float(bits);
  }
  table_.rows_e4m3 = dst;
  table_.row_begin = row_begin;
  table_.rows = rows;
  table_.head_dim = static_cast<int>(hd);
  table_.bytes = bytes;
  DGPP_LOG_INFO("qwen loader: rank {} n-gram table rows [{}, {}) resident ({:.2f} GiB, scale {})",
                rank_, row_begin, row_end, bytes / (1024.0 * 1024.0 * 1024.0), table_.scale);
  return table_;
}

// ---- sources and digests --------------------------------------------------------

}  // namespace dgpp

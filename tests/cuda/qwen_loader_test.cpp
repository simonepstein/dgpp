// The Qwen3.8-Flash-Next resident loader on the synthetic
// fixture: every class lands byte-exact at worlds 1, 2 and 4 as the slice
// formulas of docs/qwen38_flash_next_plan.md §2.1 say (GDN segments, QSA
// heads and the kv pairing, the experts' inter slices on the re-blocked
// scale grid, the PLE's column slices, the replicated set verbatim); the
// byte formulas equal actual usage and the source-byte plan equals the
// bytes read; the globals and the n-gram table slice; resident cache hits
// and the image round trip; and a tampered hash buffer is refused.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/loader.hpp"
#include "qwen_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::QwenExpectedTensor;
using dgpp::QwenLayerStream;
using dgpp::QwenTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  QwenTextConfig cfg;
  std::string dir;
  std::vector<QwenExpectedTensor> table;
  const QwenExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return qwenfx::tensor_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_config();
  fx.dir = (fs::current_path() / "qwen_loader_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

Fixture write_nvfp4_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_nvfp4_config();
  fx.dir = (fs::current_path() / "qwen_loader_nvfp4_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir, qwenfx::tiny_text_json(),
                        qwenfx::tiny_nvfp4_quant_json());
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

Fixture write_gptq_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_gptq_config();
  fx.dir = (fs::current_path() / "qwen_loader_gptq_fixture").string();
  const std::string text = qwenfx::tiny_gptq_text_json();
  qwenfx::write_fixture(fx.cfg, fx.dir, text.c_str(), qwenfx::tiny_gptq_quant_json());
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

// A one-tensor safetensors file, for the companion-directory tests.
void write_one_tensor(const fs::path& path, const std::vector<std::string>& names,
                      const std::vector<int64_t>& shape) {
  size_t elems = 1;
  for (auto d : shape) elems *= static_cast<size_t>(d);
  const size_t bytes = elems * 2;
  std::string shape_json = "[";
  for (size_t i = 0; i < shape.size(); ++i)
    shape_json += (i ? "," : "") + std::to_string(shape[i]);
  shape_json += "]";
  std::string hdr = "{";
  for (size_t i = 0; i < names.size(); ++i)
    hdr += (i ? ",\"" : "\"") + names[i] + "\":{\"dtype\":\"BF16\",\"shape\":" + shape_json +
           ",\"data_offsets\":[" + std::to_string(i * bytes) + "," +
           std::to_string((i + 1) * bytes) + "]}";
  hdr += "}";
  while (hdr.size() % 8) hdr += ' ';
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path.string());
  const uint64_t n = hdr.size();
  std::fwrite(&n, 8, 1, f);
  std::fwrite(hdr.data(), 1, hdr.size(), f);
  const std::vector<uint8_t> zeros(bytes * names.size(), 0);
  std::fwrite(zeros.data(), 1, zeros.size(), f);
  std::fclose(f);
}

// The companion registry is process-wide: every test that touches it clears
// it again, or the next stream in this binary inherits it.
struct CompanionScope {
  ~CompanionScope() { QwenLayerStream::set_ngram_table_dir(""); }
};

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}

// Rows [r0, +rows) of a [N, W-byte] host tensor.
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
// Columns [c0, +w) bytes of every row of a [rows, full] host matrix, packed.
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t> out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
std::vector<uint8_t> widened_f32(const std::vector<uint8_t>& bf16, int64_t start, int64_t count) {
  std::vector<uint8_t> out(count * 4);
  for (int64_t i = 0; i < count; ++i) {
    uint16_t bits;
    std::memcpy(&bits, bf16.data() + (start + i) * 2, 2);
    const float f = dgpp::bf16_bits_to_float(bits);
    std::memcpy(out.data() + i * 4, &f, 4);
  }
  return out;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}

// The expected re-blocked F32 scale grid of a sliced quantized matrix.
std::vector<uint8_t> reblocked_scales(const std::vector<uint8_t>& src_bf16, int64_t src_cols,
                                      bool row_slice, int64_t start, int64_t count, int sb) {
  std::vector<uint8_t> out;
  auto at = [&](int64_t r, int64_t c) {
    uint16_t bits;
    std::memcpy(&bits, src_bf16.data() + (r * src_cols + c) * 2, 2);
    const float f = dgpp::bf16_bits_to_float(bits);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&f);
    out.insert(out.end(), p, p + 4);
  };
  const int64_t src_rows = static_cast<int64_t>(src_bf16.size() / 2) / src_cols;
  if (row_slice) {
    const int64_t n = (count + sb - 1) / sb;
    for (int64_t i = 0; i < n; ++i)
      for (int64_t c = 0; c < src_cols; ++c) at((start + i * sb) / 128, c);
  } else {
    const int64_t n = (count + sb - 1) / sb;
    for (int64_t r = 0; r < src_rows; ++r)
      for (int64_t j = 0; j < n; ++j) at(r, (start + j * sb) / 128);
  }
  return out;
}

void check_layer(const Fixture& fx, const QwenLayerStream& s, const dgpp::QwenLayerResident& r) {
  const QwenTextConfig& cfg = fx.cfg;
  const dgpp::QwenLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::qwen_layer_prefix(cfg, r.layer);
  const size_t H2 = static_cast<size_t>(cfg.hidden_size) * 2;
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) +
                          " layer " + std::to_string(r.layer) + " ";
  // GR sites: verbatim.
  for (const auto& [gp, gr] : {std::pair{p + "attn_hyper_connection.", &r.attn_gr},
                               std::pair{p + "mlp_hyper_connection.", &r.mlp_gr}}) {
    expect_device_equals(gr->hc_norm, fx.bytes(gp + "hc_norm.weight"), tag + "hc_norm");
    expect_device_equals(gr->down, fx.bytes(gp + "input_mix_weight_down.weight"), tag + "gr down");
    expect_device_equals(gr->up, fx.bytes(gp + "input_mix_weight_up.weight"), tag + "gr up");
    expect_device_equals(gr->inject, fx.bytes(gp + "block_inject_weight.weight"), tag + "gr inject");
  }
  if (r.kind == dgpp::QwenLayerKind::Gdn) {
    const std::string gp = p + "linear_attn.";
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = g.local_key_heads, lv = g.local_value_heads;
    const std::vector<uint8_t> qkv = fx.bytes(gp + "in_proj_qkv.weight");
    std::vector<uint8_t> want = host_rows(qkv, H2, rank * lk * dk, lk * dk);
    auto k = host_rows(qkv, H2, K + rank * lk * dk, lk * dk);
    auto v = host_rows(qkv, H2, 2 * K + rank * lv * dv, lv * dv);
    want.insert(want.end(), k.begin(), k.end());
    want.insert(want.end(), v.begin(), v.end());
    expect_device_equals(r.gdn.in_proj_qkv, want, tag + "gdn qkv segments");
    const size_t cw = static_cast<size_t>(cfg.gdn_conv_width) * 2;
    const std::vector<uint8_t> conv = fx.bytes(gp + "conv1d.weight");
    want = host_rows(conv, cw, rank * lk * dk, lk * dk);
    k = host_rows(conv, cw, K + rank * lk * dk, lk * dk);
    v = host_rows(conv, cw, 2 * K + rank * lv * dv, lv * dv);
    want.insert(want.end(), k.begin(), k.end());
    want.insert(want.end(), v.begin(), v.end());
    expect_device_equals(r.gdn.conv, want, tag + "gdn conv segments");
    expect_device_equals(r.gdn.in_proj_z, host_rows(fx.bytes(gp + "in_proj_z.weight"), H2, rank * lv * dv, lv * dv), tag + "gdn z");
    expect_device_equals(r.gdn.in_proj_a, host_rows(fx.bytes(gp + "in_proj_a.weight"), H2, rank * lv, lv), tag + "gdn a");
    expect_device_equals(r.gdn.in_proj_b, host_rows(fx.bytes(gp + "in_proj_b.weight"), H2, rank * lv, lv), tag + "gdn b");
    expect_device_equals(r.gdn.a_log, widened_f32(fx.bytes(gp + "A_log"), rank * lv, lv), tag + "gdn A_log");
    expect_device_equals(r.gdn.dt_bias, widened_f32(fx.bytes(gp + "dt_bias"), rank * lv, lv), tag + "gdn dt_bias");
    expect_device_equals(r.gdn.norm, fx.bytes(gp + "norm.weight"), tag + "gdn norm");
    const int64_t V = static_cast<int64_t>(cfg.gdn_value_heads) * dv;
    expect_device_equals(r.gdn.out_proj, host_cols(fx.bytes(gp + "out_proj.weight"), cfg.hidden_size, V * 2, rank * lv * dv * 2, lv * dv * 2), tag + "gdn out_proj cols");
    require(r.gdn.local_key_heads == lk && r.gdn.local_value_heads == lv, tag + "gdn head counts");
  } else {
    const std::string ap = p + "self_attn.";
    const int64_t d = cfg.head_dim;
    expect_device_equals(r.qsa.q_proj, host_rows(fx.bytes(ap + "q_proj.weight"), H2, g.head_begin * 2 * d, g.local_heads * 2 * d), tag + "qsa q rows");
    expect_device_equals(r.qsa.k_proj, host_rows(fx.bytes(ap + "k_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "qsa k rows");
    expect_device_equals(r.qsa.v_proj, host_rows(fx.bytes(ap + "v_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "qsa v rows");
    const int64_t Q = static_cast<int64_t>(cfg.num_attention_heads) * d;
    expect_device_equals(r.qsa.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, Q * 2, g.head_begin * d * 2, g.local_heads * d * 2), tag + "qsa o cols");
    expect_device_equals(r.qsa.q_norm, fx.bytes(ap + "q_norm.weight"), tag + "q_norm");
    expect_device_equals(r.qsa.k_norm, fx.bytes(ap + "k_norm.weight"), tag + "k_norm");
    expect_device_equals(r.qsa.index_qk_proj, fx.bytes(ap + "indexer.index_qk_proj.weight"), tag + "indexer proj");
    expect_device_equals(r.qsa.index_q_norm, fx.bytes(ap + "indexer.q_layernorm.weight"), tag + "indexer q norm");
    expect_device_equals(r.qsa.index_k_norm, fx.bytes(ap + "indexer.k_layernorm.weight"), tag + "indexer k norm");
    // The kv pairing: at world 4 ranks 0,1 share head 0 and 2,3 head 1.
    if (world > cfg.num_key_value_heads)
      require(r.qsa.kv_head_begin == rank / (world / cfg.num_key_value_heads), tag + "kv pairing");
  }
  // MoE: router and shared gate verbatim; shared expert BF16 slices; experts
  // e4m3 slices with the re-blocked scale grid.
  const std::string mp = p + "mlp.";
  expect_device_equals(r.moe.router, fx.bytes(mp + "gate.weight"), tag + "router");
  expect_device_equals(r.moe.shared_gate, fx.bytes(mp + "shared_expert_gate.weight"), tag + "shared gate");
  const int64_t S = g.local_shared_inter, I = g.local_inter;
  expect_device_equals(r.moe.shared[0], host_rows(fx.bytes(mp + "shared_expert.gate_proj.weight"), H2, rank * S, S), tag + "shared gate_proj rows");
  expect_device_equals(r.moe.shared[1], host_rows(fx.bytes(mp + "shared_expert.up_proj.weight"), H2, rank * S, S), tag + "shared up rows");
  expect_device_equals(r.moe.shared[2], host_cols(fx.bytes(mp + "shared_expert.down_proj.weight"), cfg.hidden_size, cfg.shared_expert_intermediate_size * 2, rank * S * 2, S * 2), tag + "shared down cols");
  require(r.moe.scale_block == std::gcd(128, static_cast<int>(I)), tag + "scale block");
  const int64_t sb_cols = (cfg.hidden_size + 127) / 128;
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = mp + "experts." + std::to_string(e) + ".";
    const dgpp::GlmQuantMatrix& gate = r.moe.expert(e, 0);
    const dgpp::GlmQuantMatrix& up = r.moe.expert(e, 1);
    const dgpp::GlmQuantMatrix& down = r.moe.expert(e, 2);
    require(gate.rows == I && gate.cols == cfg.hidden_size && gate.scale_block_rows == r.moe.scale_block && gate.scale_block_cols == 128, tag + "gate geometry");
    require(down.rows == cfg.hidden_size && down.cols == I && down.scale_block_cols == r.moe.scale_block && down.scale_block_rows == 128, tag + "down geometry");
    expect_device_equals(gate.payload, host_rows(fx.bytes(ep + "gate_proj.weight"), cfg.hidden_size, rank * I, I), tag + "expert gate payload");
    expect_device_equals(up.payload, host_rows(fx.bytes(ep + "up_proj.weight"), cfg.hidden_size, rank * I, I), tag + "expert up payload");
    expect_device_equals(down.payload, host_cols(fx.bytes(ep + "down_proj.weight"), cfg.hidden_size, cfg.moe_intermediate_size, rank * I, I), tag + "expert down payload");
    expect_device_equals(gate.scales, reblocked_scales(fx.bytes(ep + "gate_proj.weight_scale_inv"), sb_cols, true, rank * I, I, r.moe.scale_block), tag + "expert gate scales");
    expect_device_equals(up.scales, reblocked_scales(fx.bytes(ep + "up_proj.weight_scale_inv"), sb_cols, true, rank * I, I, r.moe.scale_block), tag + "expert up scales");
    expect_device_equals(down.scales, reblocked_scales(fx.bytes(ep + "down_proj.weight_scale_inv"), (cfg.moe_intermediate_size + 127) / 128, false, rank * I, I, r.moe.scale_block), tag + "expert down scales");
  }
  if (r.has_ple) {
    const std::string lp = p + "ple.";
    const dgpp::QwenNgramGeometry ng = cfg.ngram_geometry();
    const size_t E2 = static_cast<size_t>(cfg.ple_embed_dim) * 2;
    const size_t c0 = static_cast<size_t>(g.hash_head_begin) * ng.head_dim * 2;
    const size_t cw = static_cast<size_t>(g.hash_heads) * ng.head_dim * 2;
    expect_device_equals(r.ple.key_proj, host_cols(fx.bytes(lp + "key_proj.weight"), cfg.hyper_width(), E2, c0, cw), tag + "ple key cols");
    expect_device_equals(r.ple.value_proj, host_cols(fx.bytes(lp + "value_proj.weight"), cfg.hidden_size, E2, c0, cw), tag + "ple value cols");
    expect_device_equals(r.ple.norm_key, fx.bytes(lp + "norm_key.weight"), tag + "ple norm_key");
    expect_device_equals(r.ple.norm_query, fx.bytes(lp + "norm_query.weight"), tag + "ple norm_query");
    expect_device_equals(r.ple.norm_conv, fx.bytes(lp + "norm_conv.weight"), tag + "ple norm_conv");
    expect_device_equals(r.ple.conv, fx.bytes(lp + "conv1d.weight"), tag + "ple conv");
    require(r.ple.table_scale == dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(0.02f)), tag + "ple table scale");
    require(r.ple.row_begin == ng.head_offset[static_cast<size_t>(g.hash_head_begin)], tag + "ple row begin");
  }
  require(r.bytes == QwenLayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(qwen_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers + 1;
  for (const int world : {1, 2, 4}) {
    uint64_t sum_source = 0;
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        dgpp::QwenHeadSharding::Full, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before ==
                    QwenLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
      sum_source += s.source_bytes_read();
    }
    std::printf("world %d: %.2f MB of source read across the ranks\n", world,
                static_cast<double>(sum_source) / 1048576.0);
  }
}

DGPP_TEST(qwen_loader_globals_and_ngram_table_slices) {
  const Fixture fx = write_fixture();
  const dgpp::QwenNgramGeometry ng = fx.cfg.ngram_geometry();
  // The whole table's bytes in shard order.
  std::vector<uint8_t> table;
  const std::string tp = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.ngram_embedding.";
  for (int sh = 0; sh < fx.cfg.split_ngram_parts; ++sh) {
    const auto b = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight");
    table.insert(table.end(), b.begin(), b.end());
  }
  require(static_cast<int64_t>(table.size()) == ng.padded_rows * ng.head_dim, "fixture table size");
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.language_model.embed_tokens.weight"), "embed");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == QwenLayerStream::lm_vocab_count(fx.cfg, rank, world, world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full), "lm head count");
      expect_device_equals(g.mixer.down, fx.bytes("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"), "mixer down");
      require(g.mixer.inject == nullptr, "the mixer has no inject");
      expect_device_equals(g.mtp_fc_hidden, fx.bytes("mtp.fc_hidden.weight"), "mtp fc_hidden");
      expect_device_equals(g.mtp_pre_fc_norm_hidden, fx.bytes("mtp.pre_fc_norm_hidden.weight"), "mtp pre_fc_norm_hidden");
      expect_device_equals(g.mtp_mixer.up, fx.bytes("mtp.hyper_connection_mixer.input_mix_weight_up.weight"), "mtp mixer up");
      require(g.bytes == QwenLayerStream::globals_bytes(fx.cfg, rank, world, world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full), "globals formula");
      const auto& t = s.load_ngram_table();
      const int64_t hb = s.geometry().hash_head_begin, hn = s.geometry().hash_heads;
      const int64_t rb = ng.head_offset[static_cast<size_t>(hb)];
      const int64_t re = hb + hn < ng.heads ? ng.head_offset[static_cast<size_t>(hb + hn)] : ng.total_rows;
      require(t.row_begin == rb && t.rows == re - rb, "table slice range");
      expect_device_equals(t.rows_e4m3, host_rows(table, static_cast<size_t>(ng.head_dim), rb, re - rb), "table rows");
      require(t.bytes == QwenLayerStream::ngram_table_bytes(fx.cfg, rank, world), "table bytes formula");
    }
  }
}

// The mmap'ed table: the stream maps the shard(s) instead of
// copying the rows; every row read through the mapping is the shard's,
// the gather lays a rank's heads out as the device gather would, the
// plan's table bytes are zero, and the mode is off unless asked for.
// The dense stack encoded to block FP8 at load (2026-09-10,
// engine.dense_weights = "fp8"): off by default; on, a rows slice, a
// columns slice and the merged GDN qkv carry the recipe's codes — scale
// amax / 448 per 128 x 128 block, round-to-nearest-even e4m3 of w / scale
// — bit for bit against a scalar reference, dequantize back within the
// format's precision, and the layer's resident bytes shrink.
namespace {
void reference_fp8(const std::vector<uint16_t>& w, int64_t rows, int64_t cols, std::vector<uint8_t>* codes,
                   std::vector<float>* scales) {
  const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
  codes->assign(static_cast<size_t>(rows) * cols, 0);
  scales->assign(static_cast<size_t>(sr) * sc, 0.f);
  for (int64_t br = 0; br < sr; ++br)
    for (int64_t bc = 0; bc < sc; ++bc) {
      float amax = 0.f;
      for (int64_t r = br * 128; r < std::min(rows, br * 128 + 128); ++r)
        for (int64_t c = bc * 128; c < std::min(cols, bc * 128 + 128); ++c)
          amax = std::max(amax, std::fabs(dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * cols + c])));
      const float s = amax > 0.f ? amax / 448.f : 1.f;
      (*scales)[static_cast<size_t>(br) * sc + bc] = s;
      for (int64_t r = br * 128; r < std::min(rows, br * 128 + 128); ++r)
        for (int64_t c = bc * 128; c < std::min(cols, bc * 128 + 128); ++c)
          (*codes)[static_cast<size_t>(r) * cols + c] =
              dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * cols + c]) / s);
    }
}
void expect_fp8_matches(const dgpp::GlmQuantMatrix& q, const std::vector<uint16_t>& w, const std::string& what) {
  std::vector<uint8_t> codes;
  std::vector<float> scales;
  reference_fp8(w, q.rows, q.cols, &codes, &scales);
  const std::vector<uint8_t> got = device_bytes(q.payload, codes.size());
  require(got == codes, what + ": the codes differ from the reference recipe");
  const std::vector<uint8_t> sb = device_bytes(q.scales, scales.size() * 4);
  std::vector<float> got_s(scales.size());
  std::memcpy(got_s.data(), sb.data(), sb.size());
  require(got_s == scales, what + ": the scales differ from the reference recipe");
  // Dequantized back: within the e4m3 grid's half ulp (2^-4 relative;
  // the subnormal zone's absolute 2^-10 of the scale).
  const int64_t sc = (q.cols + 127) / 128;
  for (int64_t r = 0; r < q.rows; ++r)
    for (int64_t c = 0; c < q.cols; ++c) {
      const float s = scales[static_cast<size_t>(r / 128) * sc + c / 128];
      const float x = dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * q.cols + c]);
      const float dq = dgpp::fp8_e4m3_bits_to_float(got[static_cast<size_t>(r) * q.cols + c]) * s;
      const float tol = std::fabs(x) / 16.f + s / 1024.f + 1e-30f;
      require(std::fabs(dq - x) <= tol, what + ": dequantized value outside the format's precision at (" +
                                             std::to_string(r) + ", " + std::to_string(c) + ")");
    }
}
}  // namespace

DGPP_TEST(qwen_loader_dense_fp8_at_load_is_the_reference_recipe) {
  const Fixture fx = write_fixture();
  require(!QwenLayerStream::dense_weights_fp8(), "the dense stack is the checkpoint's by default");
  const std::string p = dgpp::qwen_layer_prefix(fx.cfg, 0);
  const size_t bf16_bytes = QwenLayerStream::layer_bytes(fx.cfg, 0, 0, 1);
  QwenLayerStream::set_dense_weights_fp8(true);
  struct Reset { ~Reset() { QwenLayerStream::set_dense_weights_fp8(false); } } reset;
  require(QwenLayerStream::layer_bytes(fx.cfg, 0, 0, 1) < bf16_bytes, "the FP8 layer is smaller in the plan");
  QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::Full);
  const auto& l = s.load_layer(0);
  require(l.kind == dgpp::QwenLayerKind::Gdn, "layer 0 of the fixture is a GDN layer");
  require(l.gdn.in_proj_z == nullptr && l.gdn.in_proj_z_fp8.payload != nullptr, "z is FP8 only");
  require(l.gdn.in_proj_a != nullptr, "a stays BF16");
  // A rows slice (the whole matrix at world 1).
  {
    const auto w = fx.bytes(p + "linear_attn.in_proj_z.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.in_proj_z_fp8, h, "in_proj_z");
  }
  // A columns slice (the whole matrix at world 1) and the merged qkv.
  {
    const auto w = fx.bytes(p + "linear_attn.out_proj.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.out_proj_fp8, h, "out_proj");
  }
  {
    const auto w = fx.bytes(p + "linear_attn.in_proj_qkv.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.in_proj_qkv_fp8, h, "in_proj_qkv");
  }
  require(l.attn_gr.down == nullptr && l.attn_gr.down_fp8.payload != nullptr, "the GR site is FP8");
  require(l.moe.shared[0] == nullptr && l.moe.shared_fp8[0].payload != nullptr, "the shared expert is FP8");
  const auto& g = s.load_globals();
  require(g.lm_head == nullptr && g.lm_head_fp8.payload != nullptr && g.lm_head_fp8.rows == g.lm_vocab_count,
          "lm_head is FP8");
  require(g.bytes == QwenLayerStream::globals_bytes(fx.cfg, 0, 1, dgpp::QwenHeadSharding::Full), "globals formula (fp8)");
}

DGPP_TEST(qwen_loader_mmap_ngram_table_reads_the_shards_rows) {
  const Fixture fx = write_fixture();
  const dgpp::QwenNgramGeometry ng = fx.cfg.ngram_geometry();
  std::vector<uint8_t> table;
  const std::string tp = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.ngram_embedding.";
  for (int sh = 0; sh < fx.cfg.split_ngram_parts; ++sh) {
    const auto b = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight");
    table.insert(table.end(), b.begin(), b.end());
  }
  require(!QwenLayerStream::ngram_table_mmap(), "the mmap mode is off by default");
  QwenLayerStream::set_ngram_table_mmap(true);
  struct Reset { ~Reset() { QwenLayerStream::set_ngram_table_mmap(false); } } reset;
  const size_t hd = static_cast<size_t>(ng.head_dim);
  for (const int world : {1, 2}) {
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
      require(QwenLayerStream::ngram_table_bytes(fx.cfg, rank, world) == 0, "no table bytes under mmap");
      const auto& t = s.load_ngram_table();
      require(t.rows_e4m3 == nullptr && t.mmap != nullptr && t.bytes == 0, "the mmap'ed view");
      require(t.mmap->capacity() == dgpp::qwen_ngram_shard_capacity(fx.cfg) && t.mmap->head_dim() == ng.head_dim, "geometry");
      for (int64_t r : {int64_t{0}, int64_t{1}, ng.total_rows / 2, ng.total_rows - 1})
        require(std::memcmp(t.mmap->row(r), table.data() + static_cast<size_t>(r) * hd, hd) == 0, "row " + std::to_string(r));
      // A gather of the rank's heads: ids [n, heads] → dst [n, heads_local, hd].
      const int hb = s.geometry().hash_head_begin, hn = s.geometry().hash_heads;
      const int n = 5;
      std::vector<int32_t> ids(static_cast<size_t>(n) * ng.heads);
      uint64_t x = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(rank * 7 + world);
      for (int tkn = 0; tkn < n; ++tkn)
        for (int h = 0; h < ng.heads; ++h) {
          x ^= x << 13; x ^= x >> 7; x ^= x << 17;
          ids[static_cast<size_t>(tkn) * ng.heads + h] = static_cast<int32_t>(ng.head_offset[h] + static_cast<int64_t>(x % static_cast<uint64_t>(ng.head_vocab[h])));
        }
      std::vector<uint8_t> dst(static_cast<size_t>(n) * hn * hd);
      t.mmap->gather(ids.data(), n, ng.heads, hb, hn, dst.data());
      for (int tkn = 0; tkn < n; ++tkn)
        for (int hl = 0; hl < hn; ++hl) {
          const int32_t id = ids[static_cast<size_t>(tkn) * ng.heads + hb + hl];
          require(std::memcmp(dst.data() + (static_cast<size_t>(tkn) * hn + hl) * hd, table.data() + static_cast<size_t>(id) * hd, hd) == 0,
                  "gathered row");
        }
    }
  }
}

DGPP_TEST(qwen_loader_nvfp4_metadata_is_replicated_at_world_two) {
  const Fixture fx = write_nvfp4_fixture();
  require(fx.cfg.experts_nvfp4, "the fixture selects NVFP4 experts");
  for (int rank = 0; rank < 2; ++rank) {
    QwenLayerStream s(fx.cfg, fx.dir, rank, 2, dgpp::QwenResidency::Streaming,
                      dgpp::QwenHeadSharding::VocabSharded);
    const auto& layer = s.load_layer(0);
    require(layer.moe.nvfp4(), "the backbone experts stay NVFP4");
    require(layer.moe.experts_fp4.size() == static_cast<size_t>(fx.cfg.num_experts) * 3,
            "every local NVFP4 expert matrix is present");
    require(layer.moe.expert_globals != nullptr, "the replicated weight scales are loaded");
  }
}

DGPP_TEST(qwen_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "qwen_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = QwenLayerStream::resident_image_dir();
  QwenLayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    QwenLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident,
                      dgpp::QwenHeadSharding::VocabSharded, /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    (void)s.load_ngram_table();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    s.release_sources();
    require(s.sources_released(), "sources released");
    require(s.staging_released(), "the pinned staging mirror goes with the sources");
    bool refused = false;
    try {
      (void)s.hash_replicated();
    } catch (const std::runtime_error&) {
      refused = true;
    }
    require(refused, "the digest is a boot check");
  }
  {
    QwenLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident,
                      dgpp::QwenHeadSharding::VocabSharded, /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)],
              "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    // The digest note round trip.
    const dgpp::QwenReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  QwenLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(qwen_loader_refuses_tampered_hash_buffers) {
  const Fixture fx = write_fixture();
  // Flip one byte of layer_multipliers inside the shard.
  const fs::path shard = fs::path(fx.dir) / "model.safetensors";
  const std::string name = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.layer_multipliers";
  std::FILE* f = std::fopen(shard.c_str(), "r+b");
  require(f != nullptr, "open shard");
  uint64_t hlen = 0;
  require(std::fread(&hlen, 8, 1, f) == 1, "header length");
  std::string header(hlen, '\0');
  require(std::fread(header.data(), 1, hlen, f) == hlen, "header");
  const size_t at = header.find("\"" + name + "\"");
  require(at != std::string::npos, "tensor in header");
  const size_t off_at = header.find("\"data_offsets\":[", at);
  const uint64_t begin = std::stoull(header.substr(off_at + 16));
  std::fseek(f, static_cast<long>(8 + hlen + begin), SEEK_SET);
  uint8_t byte = 0;
  require(std::fread(&byte, 1, 1, f) == 1, "read byte");
  byte ^= 1;
  std::fseek(f, static_cast<long>(8 + hlen + begin), SEEK_SET);
  require(std::fwrite(&byte, 1, 1, f) == 1, "write byte");
  std::fclose(f);
  bool refused = false;
  try {
    QwenLayerStream s(fx.cfg, fx.dir, 0, 1);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("hash buffers") != std::string::npos;
  }
  require(refused, "a tampered multiplier is refused by name");
}

int main() { return dgpp::test::run_all(); }

// The repack, against the checkpoint's own bytes: the engine's word for
// (n, k-block) must be the checkpoint's word for the same pair — the two
// layouts differ by a transpose and nothing else — and the engine's BF16
// group scale must be the checkpoint's F16 one, rounded once and written
// to both of the 64-element groups inside the checkpoint's 128.
DGPP_TEST(qwen_loader_repacks_the_autoround_int4_experts) {
  const Fixture fx = write_gptq_fixture();
  require(fx.cfg.experts_packed && fx.cfg.lm_head_packed, "the fixture selects the int4 release");
  const int64_t H = fx.cfg.hidden_size, I = fx.cfg.moe_intermediate_size;
  const int64_t group = fx.cfg.packed_group;
  QwenLayerStream::set_dense_weights_fp8(true);
  struct Reset { ~Reset() { QwenLayerStream::set_dense_weights_fp8(false); } } reset;
  for (int world : {1, 2}) {
    for (int rank = 0; rank < world; ++rank) {
      dgpp::QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                              dgpp::QwenHeadSharding::VocabSharded);
      const auto& layer = s.load_layer(0);
      require(layer.moe.packq(), "the backbone experts are packed");
      require(layer.moe.experts_packed.size() == static_cast<size_t>(fx.cfg.num_experts) * 3,
              "every local packed expert matrix is present");
      const int64_t local_I = I / world;
      // gate_proj: rows [rank * local_I, +local_I) of the [I, H] matrix.
      const std::string ep = "model.language_model.layers.0.mlp.experts.3.";
      {
        const dgpp::GlmPackedMatrix& m = layer.moe.experts_packed[3 * 3 + 0];
        require(m.rows == local_I && m.cols == H && m.bits == 4, "gate_proj geometry");
        const auto words = fx.bytes(ep + "gate_proj.qweight");
        const auto scales = fx.bytes(ep + "gate_proj.scales");
        const auto* sw = reinterpret_cast<const uint32_t*>(words.data());
        const auto* ss = reinterpret_cast<const uint16_t*>(scales.data());
        const auto got_w = device_bytes(m.packed, static_cast<size_t>(m.rows) * m.packed_cols() * 4);
        const auto got_s = device_bytes(m.scales, m.scale_bytes());
        const auto* gw = reinterpret_cast<const uint32_t*>(got_w.data());
        const auto* gs = reinterpret_cast<const uint16_t*>(got_s.data());
        for (int64_t r = 0; r < m.rows; ++r) {
          const int64_t n = rank * local_I + r;
          for (int64_t w = 0; w < m.packed_cols(); ++w)
            require(gw[r * m.packed_cols() + w] == sw[w * I + n],
                    "gate_proj word (" + std::to_string(r) + ", " + std::to_string(w) + ")");
          for (int64_t g = 0; g < m.scale_cols(); ++g) {
            const uint16_t want = dgpp::float_to_bf16_bits(
                dgpp::f16_bits_to_float(ss[(g * dgpp::kPackedGroup / group) * I + n]));
            require(gs[r * m.scale_cols() + g] == want,
                    "gate_proj scale (" + std::to_string(r) + ", " + std::to_string(g) + ")");
          }
        }
      }
      // down_proj: columns [rank * local_I, +local_I) of the [H, I] matrix,
      // which is a slice of qweight's FIRST axis — whole words.
      {
        const dgpp::GlmPackedMatrix& m = layer.moe.experts_packed[3 * 3 + 2];
        require(m.rows == H && m.cols == local_I, "down_proj geometry");
        const auto words = fx.bytes(ep + "down_proj.qweight");
        const auto scales = fx.bytes(ep + "down_proj.scales");
        const auto* sw = reinterpret_cast<const uint32_t*>(words.data());
        const auto* ss = reinterpret_cast<const uint16_t*>(scales.data());
        const auto got_w = device_bytes(m.packed, static_cast<size_t>(m.rows) * m.packed_cols() * 4);
        const auto got_s = device_bytes(m.scales, m.scale_bytes());
        const auto* gw = reinterpret_cast<const uint32_t*>(got_w.data());
        const auto* gs = reinterpret_cast<const uint16_t*>(got_s.data());
        const int64_t w0 = rank * local_I / 8;
        for (int64_t r = 0; r < m.rows; ++r) {
          for (int64_t w = 0; w < m.packed_cols(); ++w)
            require(gw[r * m.packed_cols() + w] == sw[(w0 + w) * H + r], "down_proj word");
          for (int64_t g = 0; g < m.scale_cols(); ++g) {
            const int64_t k = rank * local_I + g * dgpp::kPackedGroup;
            const uint16_t want =
                dgpp::float_to_bf16_bits(dgpp::f16_bits_to_float(ss[(k / group) * H + r]));
            require(gs[r * m.scale_cols() + g] == want, "down_proj scale");
          }
        }
      }
    }
  }
}

// The draft layer is outside the int4 pass: its experts are BF16 in the
// checkpoint and take the same load-time FP8 encoding as the dense stack,
// so the layer stays on the FP8 expert path.
DGPP_TEST(qwen_loader_keeps_the_autoround_draft_layer_off_the_packed_path) {
  const Fixture fx = write_gptq_fixture();
  QwenLayerStream::set_dense_weights_fp8(true);
  struct Reset { ~Reset() { QwenLayerStream::set_dense_weights_fp8(false); } } reset;
  dgpp::QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                          dgpp::QwenHeadSharding::VocabSharded);
  const auto& draft = s.load_layer(fx.cfg.mtp_layer());
  require(!draft.moe.packq(), "the draft layer's experts are not packed");
  require(draft.moe.experts.size() == static_cast<size_t>(fx.cfg.num_experts) * 3,
          "the draft layer's experts took the FP8 path");
  require(draft.moe.experts[0].payload != nullptr, "the draft layer's experts are resident");
}

// The dense stack of the int4 release is the checkpoint's own block FP8,
// sliced rather than encoded — and the draft layer, which no release
// quantizes, stays BF16 beside it.
DGPP_TEST(qwen_loader_reads_the_autoround_dense_stack_as_checkpoint_fp8) {
  const Fixture fx = write_gptq_fixture();
  require(fx.cfg.dense_stack_fp8, "the fixture's dense stack is quantized");
  QwenLayerStream::set_dense_weights_fp8(true);
  struct Reset { ~Reset() { QwenLayerStream::set_dense_weights_fp8(false); } } reset;
  QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                    dgpp::QwenHeadSharding::VocabSharded);
  {
    const auto& l = s.load_layer(0);  // a GDN layer
    require(l.gdn.in_proj_qkv == nullptr && l.gdn.in_proj_qkv_fp8.payload != nullptr,
            "the GDN stack is FP8");
    require(l.gdn.in_proj_z_fp8.payload && l.gdn.out_proj_fp8.payload, "z and out_proj are FP8");
    require(l.gdn.in_proj_a != nullptr && l.gdn.norm != nullptr, "a and the norm stay BF16");
    require(l.moe.shared_fp8[0].payload && l.moe.shared[0] == nullptr, "the shared expert is FP8");
    // The fused [q | k | v] stack carries the source's payload bytes, in
    // segment order, and the source's scale grid with it.
    const std::string p = dgpp::qwen_layer_prefix(fx.cfg, 0);
    const auto src = fx.bytes(p + "linear_attn.in_proj_qkv.weight");
    const auto& m = l.gdn.in_proj_qkv_fp8;
    const auto got = device_bytes(m.payload, static_cast<size_t>(m.rows) * m.cols);
    require(std::memcmp(got.data(), src.data(), got.size()) == 0,
            "world 1 takes the qkv stack byte for byte");
  }
  {
    // The draft layer is BF16 IN THE CHECKPOINT — the binding says so — and
    // takes the ordinary load-time encoding from there, unlike the backbone
    // whose bytes are already e4m3.
    const std::string dp = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.mtp_layer());
    const std::string bp = dgpp::qwen_layer_prefix(fx.cfg, 2);  // a QSA backbone layer
    require(fx.expected(dp + "self_attn.q_proj.weight").dtype == dgpp::DType::BF16,
            "the draft layer's q_proj is BF16 in the checkpoint");
    require(fx.expected(bp + "self_attn.q_proj.weight").dtype == dgpp::DType::F8_E4M3,
            "the backbone's q_proj is e4m3 in the checkpoint");
    const auto& d = s.load_layer(fx.cfg.mtp_layer());
    require(d.qsa.q_proj == nullptr && d.qsa.q_proj_fp8.payload != nullptr,
            "and is encoded to FP8 at load, like every other BF16 dense matrix");
  }
}

// engine.dense_weights = "bf16" against a checkpoint with no BF16 dense
// stack is refused, by name, rather than quietly served as FP8.
DGPP_TEST(qwen_loader_refuses_bf16_dense_weights_on_a_quantized_dense_stack) {
  const Fixture fx = write_gptq_fixture();
  require(!QwenLayerStream::dense_weights_fp8(), "the default is the checkpoint's form");
  std::string msg;
  try {
    QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                      dgpp::QwenHeadSharding::VocabSharded);
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  require(msg.find("dense_weights") != std::string::npos, "refused, naming the setting");
}

// The n-gram table in a companion directory: only the table's own tensors
// come across. The real `ple-table/` also carries its own copy of another
// layer's weights, which would collide with the checkpoint's under the
// duplicate guard — so the filter is what makes the companion usable, and
// these two refusals pin it. A non-matching duplicate is SKIPPED (the
// directory then has nothing of the table's, which is the error we get);
// a matching duplicate is REFUSED by name.
DGPP_TEST(qwen_loader_takes_only_the_table_from_a_companion_directory) {
  const Fixture fx = write_fixture();
  const auto refusal_with = [&](const std::string& sub, const std::vector<std::string>& names,
                                const std::vector<int64_t>& shape) {
    const fs::path dir = fs::path(fx.dir) / sub;
    fs::remove_all(dir);
    fs::create_directories(dir);
    write_one_tensor(dir / "companion.safetensors", names, shape);
    CompanionScope scope;
    QwenLayerStream::set_ngram_table_dir(dir.string());
    try {
      QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                        dgpp::QwenHeadSharding::VocabSharded);
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  // A name the checkpoint already has, outside the table: skipped, so the
  // companion is reported as carrying no table rather than as a duplicate.
  const std::string skipped =
      refusal_with("companion_other", {"model.language_model.layers.0.linear_attn.norm.weight"},
                   {fx.cfg.gdn_value_head_dim});
  require(skipped.find("no tensor matching") != std::string::npos,
          "a non-table duplicate is filtered out, not collided with: " + skipped);
  // One of the table's own, which the fixture keeps in the checkpoint: the
  // duplicate guard catches it and names it.
  const std::string p = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer());
  const std::string table = p + "ple.ple_embedding.ngram_embedding.shard_0.weight";
  const std::string collided = refusal_with("companion_dup", {table}, {8, 8});
  require(collided.find("already in the checkpoint") != std::string::npos &&
              collided.find("shard_0") != std::string::npos,
          "a table tensor the checkpoint already has is refused by name: " + collided);
}

DGPP_TEST(qwen_loader_refuses_a_companion_directory_that_carries_no_table) {
  const Fixture fx = write_fixture();
  CompanionScope scope;
  {
    QwenLayerStream::set_ngram_table_dir((fs::path(fx.dir) / "nowhere").string());
    std::string msg;
    try {
      QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                        dgpp::QwenHeadSharding::VocabSharded);
    } catch (const std::runtime_error& e) {
      msg = e.what();
    }
    require(msg.find("companion directory does not exist") != std::string::npos,
            "a missing directory is refused");
  }
  {
    const fs::path dir = fs::path(fx.dir) / "empty_companion";
    fs::remove_all(dir);
    fs::create_directories(dir);
    write_one_tensor(dir / "other.safetensors", {"something.else.weight"}, {8});
    QwenLayerStream::set_ngram_table_dir(dir.string());
    std::string msg;
    try {
      QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming,
                        dgpp::QwenHeadSharding::VocabSharded);
    } catch (const std::runtime_error& e) {
      msg = e.what();
    }
    require(msg.find("no tensor matching") != std::string::npos,
            "a directory with nothing of the table's is refused, naming the filter");
  }
}

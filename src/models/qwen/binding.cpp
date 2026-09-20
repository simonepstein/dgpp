#include "models/qwen/binding.hpp"

#include <cstdlib>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<QwenExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         QwenWeightClass cls, int layer, int expert = -1,
         QwenTensorRole role = QwenTensorRole::Plain) {
  out.push_back(QwenExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer,
                                   expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              QwenWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// The e4m3 payload + its BF16 block-scale partner, as one call.
void add_quantized(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
                   QwenWeightClass cls, int layer, int expert, bool nvfp4 = false,
                   DType scale_dtype = DType::BF16) {
  if (nvfp4) {
    // The modelopt NVFP4 triple (plus its unused activation scale): `name`
    // is "...proj.weight".
    const std::string base = name.substr(0, name.size() - std::string(".weight").size());
    add(out, name, DType::U8, {rows, cols / 2}, cls, layer, expert, QwenTensorRole::Fp4Payload);
    add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
        QwenTensorRole::Fp4Scale);
    add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert, QwenTensorRole::Fp4Global);
    add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert, QwenTensorRole::InputScale);
    return;
  }
  add(out, name, DType::F8_E4M3, {rows, cols}, cls, layer, expert, QwenTensorRole::Fp8Payload);
  add(out, name + "_scale_inv", scale_dtype, qwen_scale_shape({rows, cols}), cls, layer, expert,
      QwenTensorRole::Fp8Scale);
}

// The auto_gptq int4 triple, as one call. `name` is "...proj.weight" for
// symmetry with add_quantized; the checkpoint's tensors hang off the base.
// Note the transposed shapes: [K/8, N] and [K/group, N] against the
// engine's row-major [N, K].
void add_packed(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
                QwenWeightClass cls, int layer, int expert, int group) {
  const std::string base = name.substr(0, name.size() - std::string(".weight").size());
  add(out, base + ".qweight", DType::I32, {cols / 8, rows}, cls, layer, expert,
      QwenTensorRole::PackedWords);
  add(out, base + ".scales", DType::F16, {cols / group, rows}, cls, layer, expert,
      QwenTensorRole::PackedScale);
  add(out, base + ".qzeros", DType::I32, {cols / group, rows / 8}, cls, layer, expert,
      QwenTensorRole::PackedZeros);
}

// One dense projection: BF16, or — on a release that ships the dense stack
// quantized — the e4m3 payload with an F32 128x128 scale grid.
void add_dense(TensorList& out, bool fp8, const std::string& name,
               int64_t rows, int64_t cols, QwenWeightClass cls, int layer) {
  if (fp8)
    add_quantized(out, name, rows, cols, cls, layer, -1, false, DType::F32);
  else
    add_bf16(out, name, {rows, cols}, cls, layer);
}

// A gated-residual site: hc_norm [W], down [r, W], up [W, r], inject [n, W]
// (the model-level mixer has no inject).
void expect_gr(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer,
               QwenWeightClass cls, bool combine) {
  const int64_t W = cfg.hyper_width();
  add_bf16(out, p + "hc_norm.weight", {W}, cls, layer);
  add_bf16(out, p + "input_mix_weight_down.weight", {cfg.hc_lowrank, W}, cls, layer);
  add_bf16(out, p + "input_mix_weight_up.weight", {W, cfg.hc_lowrank}, cls, layer);
  if (combine) add_bf16(out, p + "block_inject_weight.weight", {cfg.hc_count, W}, cls, layer);
}

void expect_gdn(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
  const int64_t vh = cfg.gdn_value_heads;
  const QwenWeightClass c = QwenWeightClass::Gdn;
  add_bf16(out, p + "A_log", {vh}, c, layer);
  add_bf16(out, p + "dt_bias", {vh}, c, layer);
  add_bf16(out, p + "conv1d.weight", {2 * kdim + vdim, 1, cfg.gdn_conv_width}, c, layer);
  add_bf16(out, p + "in_proj_a.weight", {vh, H}, c, layer);
  add_bf16(out, p + "in_proj_b.weight", {vh, H}, c, layer);
  // The quantized release keeps these three in the checkpoint's block FP8
  // (the draft layer, which no release quantizes, stays BF16).
  const bool fp8 = cfg.dense_stack_fp8 && layer != cfg.mtp_layer();
  add_dense(out, fp8, p + "in_proj_qkv.weight", 2 * kdim + vdim, H, c, layer);
  add_dense(out, fp8, p + "in_proj_z.weight", vdim, H, c, layer);
  add_bf16(out, p + "norm.weight", {cfg.gdn_value_head_dim}, c, layer);
  add_dense(out, fp8, p + "out_proj.weight", H, vdim, c, layer);
}

void expect_qsa(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::Qsa;
  const bool fp8 = cfg.dense_stack_fp8 && layer != cfg.mtp_layer();
  add_dense(out, fp8, p + "q_proj.weight", qh * d * 2, H, c, layer);  // [q | gate] per head
  add_dense(out, fp8, p + "k_proj.weight", kvh * d, H, c, layer);
  add_dense(out, fp8, p + "v_proj.weight", kvh * d, H, c, layer);
  add_dense(out, fp8, p + "o_proj.weight", H, qh * d, c, layer);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
  const int64_t id = cfg.indexer_head_dim;
  add_bf16(out, p + "indexer.index_qk_proj.weight",
           {(cfg.indexer_n_heads + cfg.indexer_kv_heads) * id, H}, QwenWeightClass::QsaIndexer,
           layer);
  add_bf16(out, p + "indexer.q_layernorm.weight", {id}, QwenWeightClass::QsaIndexer, layer);
  add_bf16(out, p + "indexer.k_layernorm.weight", {id}, QwenWeightClass::QsaIndexer, layer);
}

void expect_moe(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  add_bf16(out, p + "gate.weight", {cfg.num_experts, H}, QwenWeightClass::Router, layer);
  add_bf16(out, p + "shared_expert_gate.weight", {1, H}, QwenWeightClass::Router, layer);
  const int64_t S = cfg.shared_expert_intermediate_size;
  const bool dense_fp8 = cfg.dense_stack_fp8 && layer != cfg.mtp_layer();
  const QwenWeightClass sc = QwenWeightClass::SharedExpert;
  add_dense(out, dense_fp8, p + "shared_expert.gate_proj.weight", S, H, sc, layer);
  add_dense(out, dense_fp8, p + "shared_expert.up_proj.weight", S, H, sc, layer);
  add_dense(out, dense_fp8, p + "shared_expert.down_proj.weight", H, S, sc, layer);
  const int64_t I = cfg.moe_intermediate_size;
  // Neither quantized release touches the MTP layer: the NVFP4 one leaves
  // its experts in the FP8 block form, the AutoRound one in BF16 (the
  // loader encodes those to FP8, as it does every other BF16 dense matrix).
  const bool backbone = layer != cfg.mtp_layer();
  const bool nvfp4 = cfg.experts_nvfp4 && backbone;
  const bool packed = cfg.experts_packed && backbone;
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    if (packed) {
      add_packed(out, ep + "gate_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e,
                 cfg.packed_group);
      add_packed(out, ep + "up_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e,
                 cfg.packed_group);
      add_packed(out, ep + "down_proj.weight", H, I, QwenWeightClass::RoutedExpert, layer, e,
                 cfg.packed_group);
      continue;
    }
    if (cfg.experts_packed) {  // the MTP layer of an AutoRound release
      add(out, ep + "gate_proj.weight", DType::BF16, {I, H}, QwenWeightClass::RoutedExpert, layer, e);
      add(out, ep + "up_proj.weight", DType::BF16, {I, H}, QwenWeightClass::RoutedExpert, layer, e);
      add(out, ep + "down_proj.weight", DType::BF16, {H, I}, QwenWeightClass::RoutedExpert, layer, e);
      continue;
    }
    add_quantized(out, ep + "gate_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, nvfp4);
    add_quantized(out, ep + "up_proj.weight", I, H, QwenWeightClass::RoutedExpert, layer, e, nvfp4);
    add_quantized(out, ep + "down_proj.weight", H, I, QwenWeightClass::RoutedExpert, layer, e, nvfp4);
  }
}

void expect_ple(TensorList& out, const std::string& p, const QwenTextConfig& cfg, int layer) {
  const int64_t W = cfg.hyper_width();
  const int64_t E = cfg.ple_embed_dim;
  const QwenWeightClass c = QwenWeightClass::Ple;
  add_bf16(out, p + "conv1d.weight", {W, 1, cfg.ple_conv_kernel_size}, c, layer);
  add_bf16(out, p + "key_proj.weight", {W, E}, c, layer);
  add_bf16(out, p + "value_proj.weight", {cfg.hidden_size, E}, c, layer);
  add_bf16(out, p + "norm_conv.weight", {W}, c, layer);
  add_bf16(out, p + "norm_key.weight", {W}, c, layer);
  add_bf16(out, p + "norm_query.weight", {W}, c, layer);
  const QwenNgramGeometry g = cfg.ngram_geometry();
  const std::string ep = p + "ple_embedding.";
  add(out, ep + "layer_multipliers", DType::I64, {cfg.ngram_size}, c, layer);
  add(out, ep + "ngram_heads_offsets", DType::I64, {g.heads}, c, layer);
  add(out, ep + "ngram_heads_vocab_sizes", DType::I64, {g.heads}, c, layer);
  add(out, ep + "ngram_embedding.weight_scale", DType::BF16, {1}, QwenWeightClass::PleTable, layer,
      -1, QwenTensorRole::NgramScale);
  for (int s = 0; s < cfg.split_ngram_parts; ++s)
    add(out, ep + "ngram_embedding.shard_" + std::to_string(s) + ".weight", DType::F8_E4M3,
        {qwen_ngram_shard_rows(cfg, s), g.head_dim}, QwenWeightClass::PleTable, layer, s,
        QwenTensorRole::NgramShard);
}

int max_layer(const QwenTextConfig& cfg) {
  return cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
}

}  // namespace

std::vector<int64_t> qwen_scale_shape(const std::vector<int64_t>& payload) {
  if (payload.size() != 2) throw std::invalid_argument("qwen_scale_shape: payload must be 2-D");
  return {(payload[0] + 127) / 128, (payload[1] + 127) / 128};
}

int64_t qwen_ngram_shard_capacity(const QwenTextConfig& cfg) {
  const int64_t padded = cfg.ngram_geometry().padded_rows;
  return (padded + cfg.split_ngram_parts - 1) / cfg.split_ngram_parts;
}

int64_t qwen_ngram_shard_rows(const QwenTextConfig& cfg, int shard) {
  const int64_t padded = cfg.ngram_geometry().padded_rows;
  const int64_t cap = qwen_ngram_shard_capacity(cfg);
  const int64_t begin = static_cast<int64_t>(shard) * cap;
  if (shard < 0 || begin >= padded)
    throw std::invalid_argument("qwen_ngram_shard_rows: shard out of range");
  return std::min(cap, padded - begin);
}

std::string qwen_layer_prefix(const QwenTextConfig& cfg, int layer) {
  if (layer == cfg.mtp_layer()) return "mtp.layers.0.";
  return "model.language_model.layers." + std::to_string(layer) + ".";
}

std::vector<QwenExpectedTensor> qwen_expected_layer_tensors(const QwenTextConfig& cfg,
                                                            int layer) {
  if (layer < 0 || layer >= max_layer(cfg))
    throw std::invalid_argument("qwen_expected_layer_tensors: layer out of range");
  const bool is_mtp = layer == cfg.mtp_layer();
  const QwenLayerKind kind = is_mtp ? QwenLayerKind::Qsa : cfg.layers[layer];
  const std::string p = qwen_layer_prefix(cfg, layer);
  TensorList out;
  if (!is_mtp && layer == cfg.ple_layer()) expect_ple(out, p + "ple.", cfg, layer);
  expect_gr(out, p + "attn_hyper_connection.", cfg, layer, QwenWeightClass::Gr, true);
  if (kind == QwenLayerKind::Gdn)
    expect_gdn(out, p + "linear_attn.", cfg, layer);
  else
    expect_qsa(out, p + "self_attn.", cfg, layer);
  expect_gr(out, p + "mlp_hyper_connection.", cfg, layer, QwenWeightClass::Gr, true);
  expect_moe(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<QwenExpectedTensor> qwen_expected_global_tensors(const QwenTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.language_model.embed_tokens.weight", {cfg.vocab_size, H},
           QwenWeightClass::Embed, -1);
  if (cfg.lm_head_packed)
    add_packed(out, "lm_head.weight", cfg.vocab_size, H, QwenWeightClass::LmHead, -1, -1,
               cfg.packed_group);
  else
    add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, QwenWeightClass::LmHead, -1);
  expect_gr(out, "model.language_model.hyper_connection_mixer.", cfg, -1, QwenWeightClass::Mixer,
            false);
  if (cfg.mtp_layer() >= 0) {
    add_bf16(out, "mtp.fc_embedding.weight", {H, H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.fc_hidden.weight", {H, H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_embedding.weight", {H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_hidden.weight", {cfg.hyper_width()}, QwenWeightClass::Mtp,
             cfg.mtp_layer());
    expect_gr(out, "mtp.hyper_connection_mixer.", cfg, cfg.mtp_layer(), QwenWeightClass::Mtp,
              false);
  }
  return out;
}

std::vector<QwenExpectedTensor> qwen_expected_text_tensors(const QwenTextConfig& cfg) {
  TensorList out = qwen_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer(cfg); ++l) {
    TensorList layer = qwen_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

QwenBindReport qwen_validate_text_binding(
    const QwenTextConfig& cfg, const std::unordered_map<std::string, QwenTensorDesc>& present,
    size_t max_errors) {
  QwenBindReport rep;
  const auto expected = qwen_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name,
                             dtype_name(it->second.dtype), dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name,
                             shape_str(it->second.shape), shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.quantized()) ++rep.quantized_matrices;
    if (e.role == QwenTensorRole::NgramShard) ++rep.ngram_shards;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (name.rfind("model.visual.", 0) == 0) {
      ++rep.vision;
      continue;
    }
    // A truncated config (the check apps' --layers N): the layers past it
    // are out of scope, not unexpected.
    {
      static const std::string kLayers = "model.language_model.layers.";
      if (name.rfind(kLayers, 0) == 0) {
        const size_t dot = name.find('.', kLayers.size());
        const int64_t idx = dot == std::string::npos ? -1 : std::atoll(name.substr(kLayers.size(), dot - kLayers.size()).c_str());
        if (idx >= cfg.num_hidden_layers) {
          ++rep.out_of_scope;
          continue;
        }
      }
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void qwen_tp_validate_geometry(const QwenTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("qwen tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.gdn_key_heads % world != 0) fail("linear_num_key_heads must divide by world");
  if (cfg.gdn_value_heads % world != 0) fail("linear_num_value_heads must divide by world");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 && world % cfg.num_key_value_heads != 0)
    fail("num_key_value_heads must divide world or be divided by it");
  if (cfg.num_attention_heads / cfg.num_key_value_heads * cfg.num_key_value_heads !=
      cfg.num_attention_heads)
    fail("query heads per kv head");
  if (!cfg.ple_layer_ids.empty() && cfg.ngram_geometry().heads % world != 0)
    fail("the n-gram hash heads must divide by world");
  for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size}) {
    if (inter % world != 0) fail("an expert intermediate size must divide by world");
    if ((inter / world) % 16 != 0) fail("an expert intermediate slice must be a multiple of 16");
  }
}

}  // namespace dgpp

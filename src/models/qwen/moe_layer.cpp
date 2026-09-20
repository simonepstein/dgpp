#include "models/qwen/moe_layer.hpp"

#include "kernels/fp8_dequant.hpp"
#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/qwen_moe.hpp"

namespace dgpp {

GlmMoeConfig QwenMoeLayer::routed_config(int hidden, int inter, int n_experts,
                                         int top_k, bool norm_topk_prob) {
  GlmMoeConfig c;
  c.hidden = hidden;
  c.inter = inter;
  c.n_experts = n_experts;
  c.top_k = top_k;
  c.n_shared_experts = 0;
  c.routed_scaling_factor = 1.0f;
  c.norm_topk_prob = norm_topk_prob;
  c.swiglu_limit = std::numeric_limits<float>::infinity();
  c.router_mode = MoeRouterMode::SoftmaxTopk;
  GlmMoeConfig::validate_config(c);
  return c;
}

GlmMoeWeights QwenMoeLayer::routed_view(const QwenMoeWeights& w) {
  GlmMoeWeights g;
  g.router_gate = w.router;
  g.router_bias = nullptr;
  g.experts = w.experts;
  g.experts_fp4 = w.experts_fp4;
  g.experts_packed = w.experts_packed;
  return g;
}

QwenMoeLayer::QwenMoeLayer(const QwenMoeWeights& weights, const GlmMoeConfig& cfg,
                           IGemm& gemm, int max_tokens, int decode_slots,
                           int graph_table_slots)
    : w_(weights), cfg_(cfg), gemm_(gemm), max_tokens_(max_tokens),
      routed_(routed_view(weights), cfg, max_tokens, decode_slots, graph_table_slots) {
  if (cfg_.n_shared_experts != 0 || cfg_.router_mode != MoeRouterMode::SoftmaxTopk)
    throw std::invalid_argument(
        "QwenMoeLayer: the routed config must be routed_config()'s (softmax "
        "router, no shared expert in the chain)");
  check_weights();
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t H = static_cast<size_t>(cfg_.hidden);
  const size_t S = static_cast<size_t>(w_.shared_inter);
  DGPP_CUDA_OK(cudaMalloc(&d_acc_, M * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_sgate_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sup_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sact_, M * S * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_sdown_, M * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_sw_, M * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_rows_, M * sizeof(int32_t)));
  std::vector<int32_t> identity(M);
  for (size_t i = 0; i < M; ++i) identity[i] = static_cast<int32_t>(i);
  DGPP_CUDA_OK(cudaMemcpy(d_rows_, identity.data(), M * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  // The GEMM interface's workspace for the three shared-expert shapes.
  const int Hn = cfg_.hidden, Sn = static_cast<int>(w_.shared_inter);
  gemm_ws_bytes_ = std::max({gemm_.query_workspace_bytes(max_tokens_, Sn, Hn, DType::BF16),
                             gemm_.query_workspace_bytes(max_tokens_, Hn, Sn, DType::BF16),
                             static_cast<size_t>(1) << 20});
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));
  if (w_.shared_fp8) DGPP_CUDA_OK(cudaMalloc(&d_shared_bridge_, S * H * 2));
}

QwenMoeLayer::~QwenMoeLayer() {
  if (d_shared_bridge_) cudaFree(d_shared_bridge_);
  cudaFree(d_acc_);
  cudaFree(d_sgate_);
  cudaFree(d_sup_);
  cudaFree(d_sact_);
  cudaFree(d_sdown_);
  cudaFree(d_sw_);
  cudaFree(d_rows_);
  cudaFree(gemm_ws_);
}

void QwenMoeLayer::check_weights() const {
  const bool shared_ok = w_.shared_fp8 ? (w_.shared_fp8[0].payload && w_.shared_fp8[1].payload && w_.shared_fp8[2].payload)
                                       : (w_.shared_gate_proj && w_.shared_up_proj && w_.shared_down_proj);
  const int formats = (w_.experts != nullptr) + (w_.experts_fp4 != nullptr) +
                      (w_.experts_packed != nullptr);
  if (!w_.router || !w_.shared_gate || !shared_ok || formats == 0)
    throw std::invalid_argument("QwenMoeLayer: null weight pointer");
  if (formats != 1)
    throw std::invalid_argument(
        "QwenMoeLayer: exactly one of the fp8, nvfp4 and packed-int expert forms must be bound");
  if (w_.shared_inter <= 0 || w_.shared_inter % 8 != 0)
    throw std::invalid_argument("QwenMoeLayer: shared_inter must be a positive multiple of 8");
  if (cfg_.hidden % 8 != 0)
    throw std::invalid_argument("QwenMoeLayer: hidden must be a multiple of 8");
}

void QwenMoeLayer::rebind(const QwenMoeWeights& w) {
  if (w.shared_inter != w_.shared_inter)
    throw std::invalid_argument("QwenMoeLayer: rebind changes the shared slice width");
  w_ = w;
  check_weights();
  routed_.rebind(routed_view(w_));
}

void QwenMoeLayer::enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
                           cudaStream_t stream, MoeExpertKernel kernel) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");

  // 1. The routed experts: the fp32 chain in ascending expert order, left
  //    unrounded in d_acc_.
  routed_.enqueue_f32(hidden, d_acc_, tokens, stream, kernel);
  shared_tail(hidden, out, tokens, stream);
}

void QwenMoeLayer::enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                                  cudaStream_t stream, int table_slot) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");
  routed_.enqueue_decode_f32(hidden, d_acc_, tokens, nullptr, stream, table_slot);
  // Preserve the small-row fused tail, including C1 MTP. Wider batches
  // use the dense interface (Lt for BF16, streaming MMA for FP8) so the
  // shared weights are not re-read by another four-row GEMV per chunk.
  if (tokens > 8) {
    shared_tail(hidden, out, tokens, stream);
    return;
  }
  // The fused two-launch tail (bitwise the chain; qwen_moe_test pins it),
  // in the weights' form: BF16, or block FP8.
  const int H = cfg_.hidden;
  const int S = static_cast<int>(w_.shared_inter);
  if (w_.shared_fp8) {
    qwen_moe_shared_tail_decode_fp8(hidden, static_cast<size_t>(H), w_.shared_fp8[0].payload, w_.shared_fp8[0].scales,
                                    w_.shared_fp8[1].payload, w_.shared_fp8[1].scales, w_.shared_fp8[2].payload,
                                    w_.shared_fp8[2].scales, w_.shared_gate, d_sact_, d_sw_, d_acc_, out, tokens, H,
                                    S, stream);
    return;
  }
  qwen_moe_shared_tail_decode(hidden, static_cast<size_t>(H), w_.shared_gate_proj,
                              w_.shared_up_proj, w_.shared_down_proj, w_.shared_gate, d_sact_,
                              d_sw_, d_acc_, out, tokens, H, S, stream);
}

void QwenMoeLayer::enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                                   cudaStream_t stream, MoeTraceStaging* trace) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("QwenMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out) throw std::invalid_argument("QwenMoeLayer: null pointer");
  // The tensor-core chain on the grids it takes (the real slices: 160 on a
  // 32 grid at TP=4, 320 on 64 at TP=2, 640 on 128 at TP=1); the GEMV core
  // on any other (a small fixture's 16-wide slice).
  routed_.enqueue_prefill_f32(hidden, d_acc_, tokens, trace, stream,
                              routed_.mma_takes_grid() ? MoeExpertKernel::kMma
                                                       : MoeExpertKernel::kGemv);
  shared_tail(hidden, out, tokens, stream);
}

void QwenMoeLayer::shared_tail(const uint16_t* hidden, uint16_t* out, int tokens,
                               cudaStream_t stream) {
  const int H = cfg_.hidden;
  const int S = static_cast<int>(w_.shared_inter);
  // 2. The BF16 shared expert: gate/up (bf16 out), silu * up with the
  //    chain's two roundings and no clamps, the down projection unrounded.
  if (w_.shared_fp8 && tokens > 128) {
    // Prefill-shaped: each FP8 matrix dequantized into the bridge (the
    // GEMV core's values) and run through the BF16 interface.
    launch_fp8_dequant_blocks(w_.shared_fp8[0].payload, w_.shared_fp8[0].scales, d_shared_bridge_, S, H, stream);
    gemm_.matmul(hidden, d_shared_bridge_, d_sgate_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    launch_fp8_dequant_blocks(w_.shared_fp8[1].payload, w_.shared_fp8[1].scales, d_shared_bridge_, S, H, stream);
    gemm_.matmul(hidden, d_shared_bridge_, d_sup_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
  } else if (w_.shared_fp8) {
    // The FP8 form (engine.dense_weights): the scale GEMM's chunked GEMV at
    // decode rows; the same roundings.
    launch_scale_gemm_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[0].payload, w_.shared_fp8[0].scales,
                           d_sgate_, tokens, S, H, stream, static_cast<size_t>(S), mma_from_rows_);
    launch_scale_gemm_bf16(hidden, static_cast<size_t>(H), w_.shared_fp8[1].payload, w_.shared_fp8[1].scales,
                           d_sup_, tokens, S, H, stream, static_cast<size_t>(S), mma_from_rows_);
  } else {
    gemm_.matmul(hidden, w_.shared_gate_proj, d_sgate_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden, w_.shared_up_proj, d_sup_, tokens, S, H, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
  }
  launch_moe_swiglu_clamp(d_sgate_, d_sup_, d_sact_,
                          static_cast<int64_t>(tokens) * S,
                          std::numeric_limits<float>::infinity(), stream);
  if (w_.shared_fp8 && tokens > 128) {
    launch_fp8_dequant_blocks(w_.shared_fp8[2].payload, w_.shared_fp8[2].scales, d_shared_bridge_, H, S, stream);
    gemm_.matmul(d_sact_, d_shared_bridge_, d_sdown_, tokens, H, S, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(S), gemm_ws_, gemm_ws_bytes_, stream);
  } else if (w_.shared_fp8)
    launch_scale_gemm_f32(d_sact_, static_cast<size_t>(S), w_.shared_fp8[2].payload, w_.shared_fp8[2].scales,
                          d_sdown_, tokens, H, S, stream, static_cast<size_t>(H), mma_from_rows_);
  else
    gemm_.matmul(d_sact_, w_.shared_down_proj, d_sdown_, tokens, H, S, DType::BF16,
                 GemmOut::F32, static_cast<size_t>(S), gemm_ws_, gemm_ws_bytes_, stream);

  // 3. Its weight, sigma(x . g) — bf16 as the reference's — then the
  //    chain's last fma and its one rounding.
  qwen_moe_shared_gate_bf16(hidden, w_.shared_gate, d_sw_, tokens, H, stream);
  launch_moe_accum(d_acc_, d_sdown_, d_rows_, d_sw_, tokens, H, stream);
  launch_moe_round_bf16(out, d_acc_, static_cast<int64_t>(tokens) * H, stream);
}

size_t QwenMoeLayer::scratch_bytes(const GlmMoeConfig& cfg, int64_t shared_inter,
                                   int max_tokens, size_t* pinned_bytes, int decode_slots,
                                   int graph_table_slots) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t H = static_cast<size_t>(cfg.hidden);
  const size_t S = static_cast<size_t>(std::max<int64_t>(shared_inter, 0));
  size_t dev = GlmMoeLayer::scratch_bytes(cfg, max_tokens, decode_slots, graph_table_slots, pinned_bytes);
  dev += M * H * sizeof(float) * 2;  // d_acc_, d_sdown_
  dev += M * S * 2 * 3;              // d_sgate_, d_sup_, d_sact_
  dev += M * sizeof(float);          // d_sw_
  dev += M * sizeof(int32_t);        // d_rows_
  dev += static_cast<size_t>(1) << 20;  // the GEMM workspace floor
  return dev;
}

}  // namespace dgpp

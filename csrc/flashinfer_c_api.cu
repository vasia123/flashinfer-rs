/*
 * FlashInfer C API Implementation
 *
 * This file provides C API wrappers over FlashInfer's C++ template kernels.
 * It includes explicit template instantiations for supported dtype/head_dim combinations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_c_api.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstring>
#include <string>
#include <mutex>

// FlashInfer headers
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/norm.cuh>
#include <flashinfer/sampling.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

// Enable FP16 and BF16 support
#define FLASHINFER_ENABLE_F16
#define FLASHINFER_ENABLE_BF16

/*
 * NOTE: Attention kernel variant selection
 *
 * FlashInfer provides several pre-defined attention variants in variants.cuh:
 * - DefaultAttention<use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>
 *
 * We use DefaultAttention<false, false, false, false> for standard attention
 * without custom masks, sliding windows, soft cap, or ALiBi.
 *
 * For more advanced configurations, additional variants would need to be
 * instantiated with the appropriate template parameters.
 */

namespace {

// Thread-local error message storage
thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_error() {
    g_last_error.clear();
}

// Convert FlashInfer status to our status codes
FlashInferStatus from_cuda_error(cudaError_t err) {
    if (err == cudaSuccess) {
        return FLASHINFER_SUCCESS;
    }
    set_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    return FLASHINFER_CUDA_ERROR;
}

// Data type size in bytes
size_t dtype_size(FlashInferDType dtype) {
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
        case FLASHINFER_DTYPE_BFLOAT16:
            return 2;
        case FLASHINFER_DTYPE_FLOAT32:
            return 4;
        case FLASHINFER_DTYPE_FLOAT8_E4M3:
        case FLASHINFER_DTYPE_FLOAT8_E5M2:
            return 1;
        default:
            return 0;
    }
}

// Plan info storage
struct BatchDecodePlan {
    flashinfer::DecodePlanInfo plan_info;
    int32_t batch_size;
    int32_t num_qo_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t page_size;
    FlashInferDType dtype;
    FlashInferPosEncoding pos_encoding;
    float logits_soft_cap;
    void* float_workspace;
    void* int_workspace;
    size_t float_workspace_size;
    size_t int_workspace_size;
    bool enable_cuda_graph;
    float sm_scale;
    int32_t window_left;
    float* alibi_slopes;      // GPU-allocated ALiBi slopes [num_qo_heads], nullptr if not using ALiBi
    bool owns_alibi_slopes;   // Whether we should free alibi_slopes on destroy
};

struct BatchPrefillPlan {
    flashinfer::PrefillPlanInfo plan_info;
    int32_t batch_size;
    int32_t num_qo_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t page_size;
    FlashInferDType dtype;
    FlashInferPosEncoding pos_encoding;
    float logits_soft_cap;
    int causal;
    void* float_workspace;
    void* int_workspace;
    size_t float_workspace_size;
    size_t int_workspace_size;
    bool enable_cuda_graph;
    float sm_scale;
    int32_t window_left;
    uint32_t total_num_rows;  // Sum of all query lengths
    float* alibi_slopes;      // GPU-allocated ALiBi slopes [num_qo_heads], nullptr if not using ALiBi
    bool owns_alibi_slopes;   // Whether we should free alibi_slopes on destroy
};

// Compute ALiBi slopes for a given number of heads
// Formula: slope[i] = 2^(-8 * (i + 1) / n) where n = num_heads
// This follows the original ALiBi paper: https://arxiv.org/abs/2108.12409
cudaError_t compute_alibi_slopes(float** d_slopes, uint32_t num_heads, cudaStream_t stream) {
    // Allocate host memory for slopes
    std::vector<float> h_slopes(num_heads);

    // Compute slopes: ratio = 2^(-8/n), slope[i] = ratio^(i+1)
    float ratio = std::pow(2.0f, -8.0f / static_cast<float>(num_heads));
    for (uint32_t i = 0; i < num_heads; ++i) {
        h_slopes[i] = std::pow(ratio, static_cast<float>(i + 1));
    }

    // Allocate device memory
    float* slopes = nullptr;
    cudaError_t err = cudaMalloc(&slopes, num_heads * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }

    // Copy to device
    err = cudaMemcpyAsync(slopes, h_slopes.data(), num_heads * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(slopes);
        return err;
    }

    *d_slopes = slopes;
    return cudaSuccess;
}

// Convert our enums to FlashInfer enums
flashinfer::QKVLayout to_qkv_layout(FlashInferKVLayout layout) {
    return layout == FLASHINFER_KV_LAYOUT_HND
        ? flashinfer::QKVLayout::kHND
        : flashinfer::QKVLayout::kNHD;
}

flashinfer::PosEncodingMode to_pos_encoding(FlashInferPosEncoding enc) {
    switch (enc) {
        case FLASHINFER_POS_ENCODING_ROPE_LLAMA:
            return flashinfer::PosEncodingMode::kRoPELlama;
        case FLASHINFER_POS_ENCODING_ALIBI:
            return flashinfer::PosEncodingMode::kALiBi;
        default:
            return flashinfer::PosEncodingMode::kNone;
    }
}

// Template dispatch helpers
template <typename DType, int HEAD_DIM>
struct KernelDispatcher {
    static cudaError_t batch_decode(
        const BatchDecodePlan* plan,
        const void* q,
        const void* k_cache,
        const void* v_cache,
        const int32_t* kv_indptr,
        const int32_t* kv_indices,
        const int32_t* kv_last_page_len,
        void* output,
        float* lse,
        FlashInferKVLayout kv_layout,
        cudaStream_t stream
    );

    static cudaError_t batch_prefill(
        const BatchPrefillPlan* plan,
        const void* q,
        const void* k_cache,
        const void* v_cache,
        const int32_t* kv_indptr,
        const int32_t* kv_indices,
        const int32_t* kv_last_page_len,
        const int32_t* qo_indptr,
        void* output,
        float* lse,
        FlashInferKVLayout kv_layout,
        cudaStream_t stream
    );
};

// Dispatch macro for dtype and head_dim combinations
#define DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DTYPE_T, HEAD_DIM_V, ...)   \
    do {                                                                      \
        if (dtype == FLASHINFER_DTYPE_FLOAT16) {                             \
            using DTYPE_T = half;                                            \
            if (head_dim == 64) {                                            \
                constexpr int HEAD_DIM_V = 64;                               \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 128) {                                    \
                constexpr int HEAD_DIM_V = 128;                              \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 256) {                                    \
                constexpr int HEAD_DIM_V = 256;                              \
                { __VA_ARGS__ }                                              \
            } else {                                                         \
                set_error("Unsupported head_dim for float16");               \
                return FLASHINFER_UNSUPPORTED;                               \
            }                                                                \
        } else if (dtype == FLASHINFER_DTYPE_BFLOAT16) {                     \
            using DTYPE_T = nv_bfloat16;                                     \
            if (head_dim == 64) {                                            \
                constexpr int HEAD_DIM_V = 64;                               \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 128) {                                    \
                constexpr int HEAD_DIM_V = 128;                              \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 256) {                                    \
                constexpr int HEAD_DIM_V = 256;                              \
                { __VA_ARGS__ }                                              \
            } else {                                                         \
                set_error("Unsupported head_dim for bfloat16");              \
                return FLASHINFER_UNSUPPORTED;                               \
            }                                                                \
        } else {                                                             \
            set_error("Unsupported dtype");                                  \
            return FLASHINFER_UNSUPPORTED;                                   \
        }                                                                    \
    } while (0)

}  // namespace

/* ============================================================================
 * C API Implementation
 * ============================================================================ */

extern "C" {

const char* flashinfer_get_last_error(void) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}

void flashinfer_clear_last_error(void) {
    clear_error();
}

const char* flashinfer_version(void) {
    return "0.1.0-rs";  // Version of our Rust bindings
}

FlashInferStatus flashinfer_check_gpu_support(int* supported, int* sm_version) {
    clear_error();

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    *sm_version = props.major * 10 + props.minor;

    // FlashInfer requires SM75+ (Turing or newer)
    *supported = (props.major >= 7 && props.minor >= 5) || props.major >= 8;

    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_batch_decode_workspace_size(
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    int32_t max_seq_len
) {
    clear_error();

    if (batch_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0 || max_seq_len <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Estimate workspace sizes based on FlashInfer requirements
    // These are conservative estimates

    int32_t max_num_pages = (max_seq_len + page_size - 1) / page_size;
    int32_t gqa_ratio = num_qo_heads / num_kv_heads;

    // Float workspace: tmp_v for split-k, tmp_s for softmax
    size_t tmp_v_size = batch_size * num_qo_heads * head_dim * sizeof(float);
    size_t tmp_s_size = batch_size * num_qo_heads * sizeof(float);
    *float_workspace_size = tmp_v_size + tmp_s_size + 4096;  // padding

    // Int workspace: request_indices, kv_tile_indices, o_indptr, etc.
    size_t partition_info_size = batch_size * num_kv_heads * 2 * sizeof(int32_t);
    size_t page_indices_size = batch_size * max_num_pages * sizeof(int32_t);
    *int_workspace_size = partition_info_size + page_indices_size + 4096;  // padding

    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_batch_prefill_workspace_size(
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    int32_t batch_size,
    int32_t total_tokens,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    int32_t max_seq_len
) {
    clear_error();

    if (batch_size <= 0 || total_tokens <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Prefill needs larger workspace due to larger attention matrices
    size_t tmp_size = total_tokens * num_qo_heads * head_dim * sizeof(float);
    *float_workspace_size = tmp_size + 8192;  // padding

    size_t request_info_size = batch_size * 4 * sizeof(int32_t);
    size_t tile_info_size = total_tokens * sizeof(int32_t);
    *int_workspace_size = request_info_size + tile_info_size + 4096;  // padding

    return FLASHINFER_SUCCESS;
}

}  // extern "C"

// Template helpers (must be outside extern "C")
namespace {

// Standard attention variant: no custom mask, no sliding window, no soft cap, no alibi
using StandardAttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;

// Sliding window attention variant: no custom mask, WITH sliding window, no soft cap, no alibi
// Used when window_left >= 0 for models like Mistral
using SlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, false>;

// Soft cap attention variant: no custom mask, no sliding window, WITH soft cap, no alibi
// Used when logits_soft_cap > 0 for models like Gemma 2
using SoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, false>;

// Combined sliding window + soft cap variant
// Used when both window_left >= 0 AND logits_soft_cap > 0
using SlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, false>;

// ALiBi attention variants: use_alibi = true
// ALiBi (Attention with Linear Biases) is used by models like BLOOM and MPT

// ALiBi only
using ALiBiAttentionVariant = flashinfer::DefaultAttention<false, false, false, true>;

// ALiBi + sliding window
using ALiBiSlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, true>;

// ALiBi + soft cap
using ALiBiSoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, true>;

// ALiBi + sliding window + soft cap (all features)
using ALiBiSlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, true>;

// Template helper to call DecodePlan with the right parameters
template <typename DType, uint32_t HEAD_DIM>
cudaError_t call_decode_plan(
    flashinfer::DecodePlanInfo& plan_info,
    void* float_buffer,
    size_t float_workspace_size,
    void* int_buffer,
    void* page_locked_int_buffer,
    size_t int_workspace_size,
    const int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t page_size,
    bool enable_cuda_graph,
    cudaStream_t stream
) {
    using Params = flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;
    using AttentionVariant = StandardAttentionVariant;
    constexpr flashinfer::PosEncodingMode POS_ENCODING_MODE = flashinfer::PosEncodingMode::kNone;

    uint32_t num_kv_heads = num_qo_heads;  // Will be overridden by GQA dispatch

    auto work_estimation_func = [&](bool& split_kv, uint32_t& max_grid_size,
                                    uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
                                    uint32_t& gdy, uint32_t batch_size, int32_t* kv_indptr_h,
                                    const uint32_t num_qo_heads, const uint32_t page_size,
                                    bool enable_cuda_graph, cudaStream_t stream) {
        return flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            1, HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>(
            split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
            batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
    };

    return flashinfer::DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>(
        float_buffer, float_workspace_size,
        int_buffer, page_locked_int_buffer, int_workspace_size,
        plan_info, const_cast<int32_t*>(indptr_h), batch_size, num_qo_heads,
        page_size, enable_cuda_graph, stream, work_estimation_func);
}

// Template helper to call BatchDecodeWithPagedKVCacheDispatched
// AttentionVariant is a template parameter to support both standard and sliding window attention
template <typename DType, uint32_t HEAD_DIM, typename AttentionVariant>
cudaError_t call_batch_decode_run_impl(
    const BatchDecodePlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    using Params = flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;
    constexpr flashinfer::PosEncodingMode POS_ENCODING_MODE = flashinfer::PosEncodingMode::kNone;

    // Create paged_kv structure (contiguous layout version)
    // For HND layout: [num_pages, num_heads, page_size, head_dim]
    // For NHD layout: [num_pages, page_size, num_heads, head_dim]
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        plan->num_kv_heads,
        plan->page_size,
        HEAD_DIM,
        plan->batch_size,
        kv_layout,
        const_cast<DType*>(static_cast<const DType*>(k_cache)),
        const_cast<DType*>(static_cast<const DType*>(v_cache)),
        const_cast<int32_t*>(kv_indices),
        const_cast<int32_t*>(kv_indptr),
        const_cast<int32_t*>(kv_last_page_len)
    );

    // Create params
    Params params;
    params.q = const_cast<DType*>(static_cast<const DType*>(q));
    params.q_rope_offset = nullptr;  // No RoPE
    params.paged_kv = paged_kv;
    params.o = static_cast<DType*>(output);
    params.lse = lse;
    params.maybe_alibi_slopes = plan->alibi_slopes;  // ALiBi slopes (nullptr if not using ALiBi)
    params.padded_batch_size = plan->plan_info.padded_batch_size;
    params.num_qo_heads = plan->num_qo_heads;
    params.q_stride_n = plan->num_qo_heads * HEAD_DIM;  // Contiguous layout
    params.q_stride_h = HEAD_DIM;
    params.window_left = plan->window_left;
    params.logits_soft_cap = plan->logits_soft_cap;
    params.sm_scale = plan->sm_scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    // Set workspace pointers from plan info
    void* int_buffer = plan->int_workspace;
    void* float_buffer = plan->float_workspace;
    const auto& plan_info = plan->plan_info;

    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.request_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.kv_chunk_size_ptr_offset);

    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;

    if (plan_info.split_kv) {
        tmp_v = flashinfer::GetPtrFromBaseOffset<DType>(float_buffer, plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask = flashinfer::GetPtrFromBaseOffset<bool>(
                int_buffer, plan_info.block_valid_mask_offset);
        }
    }
    params.partition_kv = plan_info.split_kv;

    return flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                             AttentionVariant>(
        params, tmp_v, tmp_s, false /* enable_pdl */, stream);
}

// Wrapper that dispatches to the correct attention variant based on window_left, logits_soft_cap, and ALiBi
template <typename DType, uint32_t HEAD_DIM>
cudaError_t call_batch_decode_run(
    const BatchDecodePlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    // Dispatch based on window_left, logits_soft_cap, and ALiBi
    // window_left: -1 means full attention, >= 0 means sliding window
    // logits_soft_cap: 0.0 means no soft cap, > 0.0 means apply soft cap (Gemma 2)
    // alibi_slopes: nullptr means no ALiBi, non-null means use ALiBi (BLOOM, MPT)
    const bool use_sliding_window = (plan->window_left >= 0);
    const bool use_soft_cap = (plan->logits_soft_cap > 0.0f);
    const bool use_alibi = (plan->alibi_slopes != nullptr);

    // Dispatch to appropriate variant based on feature combination
    // 8 combinations: 2^3 (sliding_window × soft_cap × alibi)
    if (use_alibi) {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, ALiBiSlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, ALiBiSlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, ALiBiSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else {
            return call_batch_decode_run_impl<DType, HEAD_DIM, ALiBiAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        }
    } else {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, SlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, SlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_decode_run_impl<DType, HEAD_DIM, SoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else {
            return call_batch_decode_run_impl<DType, HEAD_DIM, StandardAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        }
    }
}

// Template helper to call BatchPrefillWithPagedKVCacheDispatched
// AttentionVariant is a template parameter to support both standard and sliding window attention
template <typename DType, uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename AttentionVariant>
cudaError_t call_batch_prefill_run_impl(
    const BatchPrefillPlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;
    constexpr flashinfer::PosEncodingMode POS_ENCODING_MODE = flashinfer::PosEncodingMode::kNone;
    constexpr bool USE_FP16_QK_REDUCTION = false;
    constexpr uint32_t CTA_TILE_Q = 64;  // Default CTA tile size

    // Create paged_kv structure
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        plan->num_kv_heads,
        plan->page_size,
        HEAD_DIM,
        plan->batch_size,
        kv_layout,
        const_cast<DType*>(static_cast<const DType*>(k_cache)),
        const_cast<DType*>(static_cast<const DType*>(v_cache)),
        const_cast<int32_t*>(kv_indices),
        const_cast<int32_t*>(kv_indptr),
        const_cast<int32_t*>(kv_last_page_len)
    );

    // Create params
    Params params;
    params.q = const_cast<DType*>(static_cast<const DType*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;  // No custom mask
    params.q_indptr = const_cast<int32_t*>(qo_indptr);
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;  // No RoPE
    params.o = static_cast<DType*>(output);
    params.lse = lse;
    params.maybe_alibi_slopes = plan->alibi_slopes;  // ALiBi slopes (nullptr if not using ALiBi)
    params.group_size = flashinfer::uint_fastdiv(plan->num_qo_heads / plan->num_kv_heads);
    params.num_qo_heads = plan->num_qo_heads;
    params.q_stride_n = plan->num_qo_heads * HEAD_DIM;  // Contiguous layout
    params.q_stride_h = HEAD_DIM;
    params.window_left = plan->window_left;
    params.logits_soft_cap = plan->logits_soft_cap;
    params.sm_scale = plan->sm_scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    // Set workspace pointers from plan info
    void* int_buffer = plan->int_workspace;
    void* float_buffer = plan->float_workspace;
    const auto& plan_info = plan->plan_info;

    params.request_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.request_indices_offset);
    params.qo_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.kv_tile_indices_offset);
    params.merge_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.merge_indptr_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<int32_t>(
        int_buffer, plan_info.kv_chunk_size_ptr_offset);

    params.max_total_num_rows = plan->total_num_rows;
    params.total_num_rows = flashinfer::GetPtrFromBaseOffset<uint32_t>(
        int_buffer, plan_info.total_num_rows_offset);
    params.padded_batch_size = plan_info.padded_batch_size;
    params.partition_kv = plan_info.split_kv;

    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;

    if (plan_info.split_kv) {
        tmp_v = flashinfer::GetPtrFromBaseOffset<DType>(float_buffer, plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask = flashinfer::GetPtrFromBaseOffset<bool>(
                int_buffer, plan_info.block_valid_mask_offset);
        }
    }

    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                                                              POS_ENCODING_MODE, USE_FP16_QK_REDUCTION,
                                                              MASK_MODE, AttentionVariant>(
        params, tmp_v, tmp_s, false /* enable_pdl */, stream);
}

// Wrapper that dispatches to the correct attention variant based on window_left, logits_soft_cap, and ALiBi
template <typename DType, uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
cudaError_t call_batch_prefill_run(
    const BatchPrefillPlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    // Dispatch based on window_left, logits_soft_cap, and ALiBi
    // window_left: -1 means full attention, >= 0 means sliding window
    // logits_soft_cap: 0.0 means no soft cap, > 0.0 means apply soft cap (Gemma 2)
    // alibi_slopes: nullptr means no ALiBi, non-null means use ALiBi (BLOOM, MPT)
    const bool use_sliding_window = (plan->window_left >= 0);
    const bool use_soft_cap = (plan->logits_soft_cap > 0.0f);
    const bool use_alibi = (plan->alibi_slopes != nullptr);

    // Dispatch to appropriate variant based on feature combination
    // 8 combinations: 2^3 (sliding_window × soft_cap × alibi)
    if (use_alibi) {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        }
    } else {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        } else {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, StandardAttentionVariant>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, kv_layout, stream);
        }
    }
}

}  // namespace (template helpers)

extern "C" {

FlashInferStatus flashinfer_batch_decode_plan(
    FlashInferBatchDecodePlanHandle* plan_handle,
    void* float_workspace,
    size_t float_workspace_size,
    void* int_workspace,
    size_t int_workspace_size,
    void* page_locked_int_workspace,
    size_t page_locked_int_size,
    const int32_t* kv_indptr,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    FlashInferDType dtype,
    FlashInferPosEncoding pos_encoding,
    float logits_soft_cap,
    int32_t window_left,
    int enable_cuda_graph,
    void* stream
) {
    clear_error();

    if (!plan_handle || !float_workspace || !int_workspace || !kv_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Validate parameters
    if (batch_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Check for unsupported position encoding modes (RoPE modes not yet implemented in this API)
    if (pos_encoding == FLASHINFER_POS_ENCODING_ROPE_LLAMA ||
        pos_encoding == FLASHINFER_POS_ENCODING_ROPE_LLAMA_FREQ_SCALE) {
        set_error("RoPE position encoding is not yet supported. Use FLASHINFER_POS_ENCODING_NONE or FLASHINFER_POS_ENCODING_ALIBI");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    BatchDecodePlan* plan = new BatchDecodePlan();
    plan->batch_size = batch_size;
    plan->num_qo_heads = num_qo_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype = dtype;
    plan->pos_encoding = pos_encoding;
    plan->logits_soft_cap = logits_soft_cap;
    plan->float_workspace = float_workspace;
    plan->int_workspace = int_workspace;
    plan->float_workspace_size = float_workspace_size;
    plan->int_workspace_size = int_workspace_size;
    plan->enable_cuda_graph = (enable_cuda_graph != 0);
    plan->sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    plan->window_left = window_left;  // Sliding window size (-1 = full attention)
    plan->alibi_slopes = nullptr;
    plan->owns_alibi_slopes = false;

    // Compute ALiBi slopes if using ALiBi position encoding
    if (pos_encoding == FLASHINFER_POS_ENCODING_ALIBI) {
        cudaError_t alibi_err = compute_alibi_slopes(&plan->alibi_slopes, num_qo_heads, cuda_stream);
        if (alibi_err != cudaSuccess) {
            delete plan;
            return from_cuda_error(alibi_err);
        }
        plan->owns_alibi_slopes = true;
    }
    cudaError_t err = cudaSuccess;

    // Dispatch based on dtype and head_dim
    DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DType, HEAD_DIM_V, {
        err = call_decode_plan<DType, HEAD_DIM_V>(
            plan->plan_info,
            float_workspace, float_workspace_size,
            int_workspace, page_locked_int_workspace, int_workspace_size,
            kv_indptr, batch_size, num_qo_heads, page_size,
            plan->enable_cuda_graph, cuda_stream);
    });

    if (err != cudaSuccess) {
        delete plan;
        return from_cuda_error(err);
    }

    *plan_handle = reinterpret_cast<FlashInferBatchDecodePlanHandle>(plan);
    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_batch_decode_run(
    FlashInferBatchDecodePlanHandle plan_handle,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    FlashInferKVLayout kv_layout,
    void* stream
) {
    clear_error();

    if (!plan_handle || !q || !k_cache || !v_cache || !output ||
        !kv_indptr || !kv_indices || !kv_last_page_len) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    BatchDecodePlan* plan = reinterpret_cast<BatchDecodePlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout fi_kv_layout = to_qkv_layout(kv_layout);
    cudaError_t err = cudaSuccess;

    FlashInferDType dtype = plan->dtype;
    int32_t head_dim = plan->head_dim;

    // Dispatch based on dtype and head_dim
    DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DType, HEAD_DIM_V, {
        err = call_batch_decode_run<DType, HEAD_DIM_V>(
            plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
            output, lse, fi_kv_layout, cuda_stream);
    });

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_batch_decode_plan_destroy(
    FlashInferBatchDecodePlanHandle plan_handle
) {
    if (plan_handle) {
        BatchDecodePlan* plan = reinterpret_cast<BatchDecodePlan*>(plan_handle);
        // Free ALiBi slopes if we allocated them
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_batch_prefill_plan(
    FlashInferBatchPrefillPlanHandle* plan_handle,
    void* float_workspace,
    size_t float_workspace_size,
    void* int_workspace,
    size_t int_workspace_size,
    void* page_locked_int_workspace,
    size_t page_locked_int_size,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    FlashInferDType dtype,
    FlashInferPosEncoding pos_encoding,
    float logits_soft_cap,
    int32_t window_left,
    int causal,
    int enable_cuda_graph,
    void* stream
) {
    clear_error();

    if (!plan_handle || !float_workspace || !int_workspace ||
        !qo_indptr || !kv_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Validate parameters
    if (batch_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Check for unsupported position encoding modes (RoPE modes not yet implemented in this API)
    if (pos_encoding == FLASHINFER_POS_ENCODING_ROPE_LLAMA ||
        pos_encoding == FLASHINFER_POS_ENCODING_ROPE_LLAMA_FREQ_SCALE) {
        set_error("RoPE position encoding is not yet supported. Use FLASHINFER_POS_ENCODING_NONE or FLASHINFER_POS_ENCODING_ALIBI");
        return FLASHINFER_UNSUPPORTED;
    }

    // Calculate total_num_rows from qo_indptr (sum of all query lengths)
    // qo_indptr is assumed to be on host memory for planning
    uint32_t total_num_rows = qo_indptr[batch_size] - qo_indptr[0];

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    BatchPrefillPlan* plan = new BatchPrefillPlan();
    plan->batch_size = batch_size;
    plan->num_qo_heads = num_qo_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype = dtype;
    plan->pos_encoding = pos_encoding;
    plan->logits_soft_cap = logits_soft_cap;
    plan->causal = causal;
    plan->float_workspace = float_workspace;
    plan->int_workspace = int_workspace;
    plan->float_workspace_size = float_workspace_size;
    plan->int_workspace_size = int_workspace_size;
    plan->enable_cuda_graph = (enable_cuda_graph != 0);
    plan->sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    plan->window_left = window_left;  // Sliding window size (-1 = full attention)
    plan->total_num_rows = total_num_rows;
    plan->alibi_slopes = nullptr;
    plan->owns_alibi_slopes = false;

    // Compute ALiBi slopes if using ALiBi position encoding
    if (pos_encoding == FLASHINFER_POS_ENCODING_ALIBI) {
        cudaError_t alibi_err = compute_alibi_slopes(&plan->alibi_slopes, num_qo_heads, cuda_stream);
        if (alibi_err != cudaSuccess) {
            delete plan;
            return from_cuda_error(alibi_err);
        }
        plan->owns_alibi_slopes = true;
    }

    // Call PrefillPlan to compute workspace layout
    cudaError_t err = flashinfer::PrefillPlan<int32_t>(
        float_workspace, float_workspace_size,
        int_workspace, page_locked_int_workspace, int_workspace_size,
        plan->plan_info,
        const_cast<int32_t*>(qo_indptr),
        const_cast<int32_t*>(kv_indptr),
        total_num_rows,
        batch_size, num_qo_heads, num_kv_heads,
        head_dim, head_dim,  // head_dim_qk, head_dim_vo (same for most models)
        page_size,
        plan->enable_cuda_graph,
        sizeof(half),  // sizeof_dtype_o (use half for f16/bf16)
        plan->window_left,
        -1,  // fixed_split_size (-1 = auto)
        false,  // disable_split_kv
        0,  // num_colocated_ctas
        cuda_stream);

    if (err != cudaSuccess) {
        delete plan;
        return from_cuda_error(err);
    }

    *plan_handle = reinterpret_cast<FlashInferBatchPrefillPlanHandle>(plan);
    return FLASHINFER_SUCCESS;
}

// Macro for prefill dispatch with mask mode
#define DISPATCH_PREFILL_MASK_MODE(causal, MASK_MODE_V, ...)  \
    do {                                                       \
        if (causal) {                                         \
            constexpr flashinfer::MaskMode MASK_MODE_V = flashinfer::MaskMode::kCausal; \
            { __VA_ARGS__ }                                   \
        } else {                                              \
            constexpr flashinfer::MaskMode MASK_MODE_V = flashinfer::MaskMode::kNone; \
            { __VA_ARGS__ }                                   \
        }                                                     \
    } while (0)

FlashInferStatus flashinfer_batch_prefill_run(
    FlashInferBatchPrefillPlanHandle plan_handle,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* qo_indptr,
    void* output,
    float* lse,
    FlashInferKVLayout kv_layout,
    void* stream
) {
    clear_error();

    if (!plan_handle || !q || !k_cache || !v_cache || !output ||
        !kv_indptr || !kv_indices || !kv_last_page_len || !qo_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    BatchPrefillPlan* plan = reinterpret_cast<BatchPrefillPlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout fi_kv_layout = to_qkv_layout(kv_layout);
    cudaError_t err = cudaSuccess;

    FlashInferDType dtype = plan->dtype;
    int32_t head_dim = plan->head_dim;
    bool causal = (plan->causal != 0);

    // Dispatch based on dtype, head_dim, and mask mode
    DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DType, HEAD_DIM_V, {
        DISPATCH_PREFILL_MASK_MODE(causal, MASK_MODE_V, {
            err = call_batch_prefill_run<DType, HEAD_DIM_V, MASK_MODE_V>(
                plan, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                output, lse, fi_kv_layout, cuda_stream);
        });
    });

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_batch_prefill_plan_destroy(
    FlashInferBatchPrefillPlanHandle plan_handle
) {
    if (plan_handle) {
        BatchPrefillPlan* plan = reinterpret_cast<BatchPrefillPlan*>(plan_handle);
        // Free ALiBi slopes if we allocated them
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

}  // extern "C"

// Template helpers for KV cache append (must be outside extern "C")
namespace {

// Template helper for append decode (single token per sequence)
template <typename DType>
cudaError_t call_append_decode(
    flashinfer::paged_kv_t<DType, int32_t>& paged_kv,
    DType* k,
    DType* v,
    cudaStream_t stream
) {
    return flashinfer::AppendPagedKVCacheDecode<DType, int32_t>(paged_kv, k, v, stream);
}

// Template helper for append prefill (multiple tokens)
template <typename DType>
cudaError_t call_append_prefill(
    flashinfer::paged_kv_t<DType, int32_t>& paged_kv,
    DType* k,
    DType* v,
    int32_t* batch_indices,
    int32_t* positions,
    uint32_t nnz,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    cudaStream_t stream
) {
    return flashinfer::AppendPagedKVCache<DType, int32_t>(
        paged_kv, k, v, batch_indices, positions, nnz,
        k_stride_n, k_stride_h, v_stride_n, v_stride_h, stream);
}

// CUDA kernel to compute batch indices and positions from indptr
__global__ void compute_batch_indices_positions_kernel(
    const int32_t* append_indptr,
    const int32_t* seq_lens,
    int32_t* batch_indices,
    int32_t* positions,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread finds its batch based on indptr
    // This is a simple O(batch_size) search per thread, could be optimized with binary search
    uint32_t total_tokens = append_indptr[batch_size];
    if (idx >= total_tokens) return;

    // Find which batch this token belongs to
    for (uint32_t b = 0; b < batch_size; b++) {
        if (idx >= static_cast<uint32_t>(append_indptr[b]) &&
            idx < static_cast<uint32_t>(append_indptr[b + 1])) {
            batch_indices[idx] = static_cast<int32_t>(b);
            // Position = seq_len + offset within this batch's append
            positions[idx] = seq_lens[b] + static_cast<int32_t>(idx - append_indptr[b]);
            return;
        }
    }
}

// ==========================================================================
// RoPE Template Helpers
// ==========================================================================

template <typename DType>
cudaError_t call_batch_qk_apply_rotary(
    DType* q, DType* k, DType* q_out, DType* k_out,
    int32_t* indptr, int32_t* offsets,
    uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim,
    bool interleave, float rope_scale, float rope_theta,
    cudaStream_t stream
) {
    // Compute strides for NHD layout: [total_tokens, num_heads, head_dim]
    size_t q_stride_n = num_qo_heads * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = num_kv_heads * head_dim;
    size_t k_stride_h = head_dim;

    return flashinfer::BatchQKApplyRotary<DType, int32_t>(
        q, k, q_out, k_out, indptr, offsets,
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,  // output strides same as input
        interleave, rope_scale, rope_theta, stream
    );
}

template <typename DType>
cudaError_t call_batch_qk_apply_rotary_pos_ids(
    DType* q, DType* k, DType* q_out, DType* k_out,
    int32_t* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim,
    bool interleave, float rope_scale, float rope_theta,
    cudaStream_t stream
) {
    // Compute strides for NHD layout: [total_tokens, num_heads, head_dim]
    size_t q_stride_n = num_qo_heads * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = num_kv_heads * head_dim;
    size_t k_stride_h = head_dim;

    return flashinfer::BatchQKApplyRotaryPosIds<DType, int32_t>(
        q, k, q_out, k_out, pos_ids, nnz,
        num_qo_heads, num_kv_heads, rotary_dim, head_dim,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,  // output strides same as input
        interleave, rope_scale, rope_theta, stream
    );
}

template <typename DType>
cudaError_t call_batch_qk_apply_rotary_cos_sin_cache(
    DType* q, DType* k, DType* q_out, DType* k_out,
    float* cos_sin_cache, int32_t* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim,
    bool interleave, cudaStream_t stream
) {
    // Compute strides for NHD layout: [total_tokens, num_heads, head_dim]
    size_t q_stride_n = num_qo_heads * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = num_kv_heads * head_dim;
    size_t k_stride_h = head_dim;

    return flashinfer::BatchQKApplyRotaryPosIdsCosSinCache<DType, int32_t>(
        q, k, q_out, k_out, cos_sin_cache, pos_ids, nnz,
        num_qo_heads, num_kv_heads, rotary_dim, head_dim,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,
        q_stride_n, q_stride_h, k_stride_n, k_stride_h,  // output strides same as input
        interleave, stream
    );
}

// ==========================================================================
// Sampling Helpers (float32 only)
// ==========================================================================
// NOTE: FlashInfer's sampling functions have type compatibility issues with half/bf16.
// Sampling operations typically work on float32 probabilities after softmax.
// Users should convert logits to float32, apply softmax, then sample.

cudaError_t call_top_k_sampling_f32(
    float* probs, int32_t* output, float* top_k_arr,
    uint32_t batch_size, uint32_t top_k_val, uint32_t vocab_size,
    bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
    cudaStream_t stream
) {
    return flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        probs, output, nullptr, top_k_arr,
        batch_size, top_k_val, vocab_size,
        deterministic, philox_seed, philox_offset, stream
    );
}

cudaError_t call_top_p_sampling_f32(
    float* probs, int32_t* output, float* top_p_arr,
    uint32_t batch_size, float top_p_val, uint32_t vocab_size,
    bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
    cudaStream_t stream
) {
    return flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        probs, output, nullptr, top_p_arr,
        batch_size, top_p_val, vocab_size,
        deterministic, philox_seed, philox_offset, stream
    );
}

cudaError_t call_min_p_sampling_f32(
    float* probs, float* min_p_arr, int32_t* output,
    uint32_t batch_size, float min_p_val, uint32_t vocab_size,
    bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
    cudaStream_t stream
) {
    return flashinfer::sampling::MinPSamplingFromProb<float, int32_t>(
        probs, min_p_arr, output, nullptr,
        batch_size, min_p_val, vocab_size,
        deterministic, philox_seed, philox_offset, stream
    );
}

cudaError_t call_top_k_top_p_sampling_f32(
    float* probs, int32_t* top_k_arr, float* top_p_arr, int32_t* output,
    uint32_t batch_size, int32_t top_k_val, float top_p_val, uint32_t vocab_size,
    bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
    cudaStream_t stream
) {
    return flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        probs, top_k_arr, top_p_arr, output, nullptr,
        batch_size, top_k_val, top_p_val, vocab_size,
        deterministic, philox_seed, philox_offset, stream
    );
}

cudaError_t call_online_softmax_f32(
    float* logits, float* output, uint32_t batch_size, uint32_t vocab_size,
    float* temperature_arr, float temperature_val, cudaStream_t stream
) {
    return flashinfer::sampling::OnlineSoftmax<float>(
        logits, output, batch_size, vocab_size,
        temperature_arr, temperature_val,
        nullptr, 0,  // No workspace (uses fused path)
        false,       // enable_pdl = false
        stream
    );
}

cudaError_t call_top_p_renorm_prob_f32(
    float* probs, float* renormed_probs, float* top_p_arr,
    uint32_t batch_size, float top_p_val, uint32_t vocab_size,
    cudaStream_t stream
) {
    return flashinfer::sampling::TopPRenormProb<float>(
        probs, renormed_probs, top_p_arr,
        batch_size, top_p_val, vocab_size, stream
    );
}

}  // namespace

extern "C" {

FlashInferStatus flashinfer_append_paged_kv_cache(
    const void* k,
    const void* v,
    void* k_cache,
    void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* append_indptr,
    int32_t batch_size,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    FlashInferDType dtype,
    FlashInferKVLayout kv_layout,
    void* stream
) {
    clear_error();

    if (!k || !v || !k_cache || !v_cache ||
        !kv_indptr || !kv_indices || !kv_last_page_len || !append_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size <= 0 || num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout fi_kv_layout = to_qkv_layout(kv_layout);

    // Copy append_indptr to host to determine total tokens and check if decode mode
    std::vector<int32_t> append_indptr_h(batch_size + 1);
    cudaError_t err = cudaMemcpyAsync(append_indptr_h.data(), append_indptr,
                                       (batch_size + 1) * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }
    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    uint32_t total_tokens = append_indptr_h[batch_size] - append_indptr_h[0];

    // Check if this is decode mode (each sequence appends exactly 1 token)
    bool is_decode = true;
    for (int32_t b = 0; b < batch_size; b++) {
        if (append_indptr_h[b + 1] - append_indptr_h[b] != 1) {
            is_decode = false;
            break;
        }
    }

    // Strides for NHD layout: [total_tokens, num_heads, head_dim]
    size_t k_stride_n = num_kv_heads * head_dim;
    size_t k_stride_h = head_dim;
    size_t v_stride_n = num_kv_heads * head_dim;
    size_t v_stride_h = head_dim;

    if (is_decode) {
        // Decode mode: use optimized single-token append
        switch (dtype) {
            case FLASHINFER_DTYPE_FLOAT16: {
                using DType = half;
                flashinfer::paged_kv_t<DType, int32_t> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, fi_kv_layout,
                    static_cast<DType*>(k_cache), static_cast<DType*>(v_cache),
                    const_cast<int32_t*>(kv_indices), const_cast<int32_t*>(kv_indptr),
                    const_cast<int32_t*>(kv_last_page_len));
                err = call_append_decode<DType>(
                    paged_kv,
                    const_cast<DType*>(static_cast<const DType*>(k)),
                    const_cast<DType*>(static_cast<const DType*>(v)),
                    cuda_stream);
                break;
            }
            case FLASHINFER_DTYPE_BFLOAT16: {
                using DType = nv_bfloat16;
                flashinfer::paged_kv_t<DType, int32_t> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, fi_kv_layout,
                    static_cast<DType*>(k_cache), static_cast<DType*>(v_cache),
                    const_cast<int32_t*>(kv_indices), const_cast<int32_t*>(kv_indptr),
                    const_cast<int32_t*>(kv_last_page_len));
                err = call_append_decode<DType>(
                    paged_kv,
                    const_cast<DType*>(static_cast<const DType*>(k)),
                    const_cast<DType*>(static_cast<const DType*>(v)),
                    cuda_stream);
                break;
            }
            default:
                set_error("Unsupported dtype for append KV cache");
                return FLASHINFER_UNSUPPORTED;
        }
    } else {
        // Prefill mode: need to compute batch_indices and positions
        // Allocate temporary buffers
        int32_t* batch_indices_d = nullptr;
        int32_t* positions_d = nullptr;
        int32_t* seq_lens_d = nullptr;

        err = cudaMallocAsync(&batch_indices_d, total_tokens * sizeof(int32_t), cuda_stream);
        if (err != cudaSuccess) {
            return from_cuda_error(err);
        }
        err = cudaMallocAsync(&positions_d, total_tokens * sizeof(int32_t), cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            return from_cuda_error(err);
        }
        err = cudaMallocAsync(&seq_lens_d, batch_size * sizeof(int32_t), cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            return from_cuda_error(err);
        }

        // Compute seq_lens from kv_indptr and kv_last_page_len on host
        std::vector<int32_t> kv_indptr_h(batch_size + 1);
        std::vector<int32_t> kv_last_page_len_h(batch_size);
        err = cudaMemcpyAsync(kv_indptr_h.data(), kv_indptr,
                              (batch_size + 1) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            cudaFreeAsync(seq_lens_d, cuda_stream);
            return from_cuda_error(err);
        }
        err = cudaMemcpyAsync(kv_last_page_len_h.data(), kv_last_page_len,
                              batch_size * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            cudaFreeAsync(seq_lens_d, cuda_stream);
            return from_cuda_error(err);
        }
        err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            cudaFreeAsync(seq_lens_d, cuda_stream);
            return from_cuda_error(err);
        }

        // Compute seq_lens: (num_pages - 1) * page_size + last_page_len
        std::vector<int32_t> seq_lens_h(batch_size);
        for (int32_t b = 0; b < batch_size; b++) {
            int32_t num_pages = kv_indptr_h[b + 1] - kv_indptr_h[b];
            if (num_pages > 0) {
                seq_lens_h[b] = (num_pages - 1) * page_size + kv_last_page_len_h[b];
            } else {
                seq_lens_h[b] = 0;
            }
        }

        err = cudaMemcpyAsync(seq_lens_d, seq_lens_h.data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            cudaFreeAsync(seq_lens_d, cuda_stream);
            return from_cuda_error(err);
        }

        // Launch kernel to compute batch_indices and positions
        uint32_t block_size = 256;
        uint32_t num_blocks = (total_tokens + block_size - 1) / block_size;
        compute_batch_indices_positions_kernel<<<num_blocks, block_size, 0, cuda_stream>>>(
            append_indptr, seq_lens_d, batch_indices_d, positions_d, batch_size);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFreeAsync(batch_indices_d, cuda_stream);
            cudaFreeAsync(positions_d, cuda_stream);
            cudaFreeAsync(seq_lens_d, cuda_stream);
            return from_cuda_error(err);
        }

        // Call append prefill
        switch (dtype) {
            case FLASHINFER_DTYPE_FLOAT16: {
                using DType = half;
                flashinfer::paged_kv_t<DType, int32_t> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, fi_kv_layout,
                    static_cast<DType*>(k_cache), static_cast<DType*>(v_cache),
                    const_cast<int32_t*>(kv_indices), const_cast<int32_t*>(kv_indptr),
                    const_cast<int32_t*>(kv_last_page_len));
                err = call_append_prefill<DType>(
                    paged_kv,
                    const_cast<DType*>(static_cast<const DType*>(k)),
                    const_cast<DType*>(static_cast<const DType*>(v)),
                    batch_indices_d, positions_d, total_tokens,
                    k_stride_n, k_stride_h, v_stride_n, v_stride_h,
                    cuda_stream);
                break;
            }
            case FLASHINFER_DTYPE_BFLOAT16: {
                using DType = nv_bfloat16;
                flashinfer::paged_kv_t<DType, int32_t> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, fi_kv_layout,
                    static_cast<DType*>(k_cache), static_cast<DType*>(v_cache),
                    const_cast<int32_t*>(kv_indices), const_cast<int32_t*>(kv_indptr),
                    const_cast<int32_t*>(kv_last_page_len));
                err = call_append_prefill<DType>(
                    paged_kv,
                    const_cast<DType*>(static_cast<const DType*>(k)),
                    const_cast<DType*>(static_cast<const DType*>(v)),
                    batch_indices_d, positions_d, total_tokens,
                    k_stride_n, k_stride_h, v_stride_n, v_stride_h,
                    cuda_stream);
                break;
            }
            default:
                cudaFreeAsync(batch_indices_d, cuda_stream);
                cudaFreeAsync(positions_d, cuda_stream);
                cudaFreeAsync(seq_lens_d, cuda_stream);
                set_error("Unsupported dtype for append KV cache");
                return FLASHINFER_UNSUPPORTED;
        }

        // Free temporary buffers
        cudaFreeAsync(batch_indices_d, cuda_stream);
        cudaFreeAsync(positions_d, cuda_stream);
        cudaFreeAsync(seq_lens_d, cuda_stream);
    }

    return from_cuda_error(err);
}

}  // extern "C"

/* ============================================================================
 * Normalization API Implementation
 * ============================================================================ */

FlashInferStatus flashinfer_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    // Stride is hidden_dim for contiguous tensors
    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::RMSNorm<half>(
                const_cast<half*>(static_cast<const half*>(input)),
                const_cast<half*>(static_cast<const half*>(weight)),
                static_cast<half*>(output),
                batch_size, hidden_dim, stride, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::RMSNorm<nv_bfloat16>(
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(input)),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(weight)),
                static_cast<nv_bfloat16*>(output),
                batch_size, hidden_dim, stride, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for RMSNorm");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_fused_add_rmsnorm(
    void* input,
    const void* residual,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !residual || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::FusedAddRMSNorm<half>(
                static_cast<half*>(input),
                const_cast<half*>(static_cast<const half*>(residual)),
                const_cast<half*>(static_cast<const half*>(weight)),
                batch_size, hidden_dim, stride, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            // Copy input to output after fused operation
            if (err == cudaSuccess && input != output) {
                err = cudaMemcpyAsync(output, input,
                    batch_size * hidden_dim * sizeof(half),
                    cudaMemcpyDeviceToDevice, cuda_stream);
            }
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::FusedAddRMSNorm<nv_bfloat16>(
                static_cast<nv_bfloat16*>(input),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(residual)),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(weight)),
                batch_size, hidden_dim, stride, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            // Copy input to output after fused operation
            if (err == cudaSuccess && input != output) {
                err = cudaMemcpyAsync(output, input,
                    batch_size * hidden_dim * sizeof(nv_bfloat16),
                    cudaMemcpyDeviceToDevice, cuda_stream);
            }
            break;
        default:
            set_error("Unsupported dtype for FusedAddRMSNorm");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_rmsnorm_quant(
    const void* input,
    const void* weight,
    void* output,
    float* scale,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType input_dtype,
    FlashInferDType output_dtype,
    void* stream
) {
    clear_error();

    // TODO: Implement RMSNorm with quantized output
    // This requires FP8 support which needs SM89+
    set_error("RMSNorm quantization not yet implemented");
    return FLASHINFER_UNSUPPORTED;
}

FlashInferStatus flashinfer_layernorm(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // NOTE: bias can be NULL for LayerNorm without bias
    // NOTE: Only float16 is supported due to type compatibility issues with bf16 in FlashInfer.
    // This matches the typical use case for LLM inference.

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            // FlashInfer uses "gemma" for gamma (weight) and "beta" for bias
            err = flashinfer::norm::LayerNorm<half, half>(
                const_cast<half*>(static_cast<const half*>(input)),
                const_cast<half*>(static_cast<const half*>(weight)),
                const_cast<half*>(static_cast<const half*>(bias)),
                static_cast<half*>(output),
                batch_size, hidden_dim, eps, cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            // NOTE: bf16 has type compatibility issues in FlashInfer LayerNorm kernel.
            // Use RMSNorm for bf16 workloads or convert to float16.
            set_error("LayerNorm does not support bfloat16 due to FlashInfer type issues. Use float16 or RMSNorm.");
            return FLASHINFER_UNSUPPORTED;
        default:
            set_error("Unsupported dtype for LayerNorm (use float16)");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_qk_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t head_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    // For contiguous [batch_size, num_heads, head_dim] layout:
    // stride_input_n = num_heads * head_dim (stride between batches)
    // stride_input_h = head_dim (stride between heads)
    uint32_t stride_n = num_heads * head_dim;
    uint32_t stride_h = head_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::QKRMSNorm<half>(
                const_cast<half*>(static_cast<const half*>(input)),
                const_cast<half*>(static_cast<const half*>(weight)),
                static_cast<half*>(output),
                batch_size, num_heads, head_dim,
                stride_n, stride_h,  // input strides
                stride_n, stride_h,  // output strides
                eps,
                false,  // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::QKRMSNorm<nv_bfloat16>(
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(input)),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(weight)),
                static_cast<nv_bfloat16*>(output),
                batch_size, num_heads, head_dim,
                stride_n, stride_h,
                stride_n, stride_h,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for QK RMSNorm (use float16 or bfloat16)");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_gemma_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    // Stride is hidden_dim for contiguous tensors
    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::GemmaRMSNorm<half>(
                const_cast<half*>(static_cast<const half*>(input)),
                const_cast<half*>(static_cast<const half*>(weight)),
                static_cast<half*>(output),
                batch_size, hidden_dim, stride, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::GemmaRMSNorm<nv_bfloat16>(
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(input)),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(weight)),
                static_cast<nv_bfloat16*>(output),
                batch_size, hidden_dim, stride, stride, eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for Gemma RMSNorm (use float16 or bfloat16)");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_gemma_fused_add_rmsnorm(
    void* input,
    const void* residual,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!input || !residual || !weight || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::GemmaFusedAddRMSNorm<half>(
                static_cast<half*>(input),
                const_cast<half*>(static_cast<const half*>(residual)),
                const_cast<half*>(static_cast<const half*>(weight)),
                batch_size, hidden_dim, stride, eps,
                false,  // enable_pdl
                cuda_stream
            );
            // Copy input to output since FusedAdd modifies input in-place
            if (input != output && err == cudaSuccess) {
                err = cudaMemcpyAsync(output, input,
                    batch_size * hidden_dim * sizeof(half),
                    cudaMemcpyDeviceToDevice, cuda_stream);
            }
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::GemmaFusedAddRMSNorm<nv_bfloat16>(
                static_cast<nv_bfloat16*>(input),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(residual)),
                const_cast<nv_bfloat16*>(static_cast<const nv_bfloat16*>(weight)),
                batch_size, hidden_dim, stride, eps,
                false,
                cuda_stream
            );
            if (input != output && err == cudaSuccess) {
                err = cudaMemcpyAsync(output, input,
                    batch_size * hidden_dim * sizeof(nv_bfloat16),
                    cudaMemcpyDeviceToDevice, cuda_stream);
            }
            break;
        default:
            set_error("Unsupported dtype for Gemma Fused Add RMSNorm (use float16 or bfloat16)");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

/* ============================================================================
 * Sampling API Implementation
 * ============================================================================ */

FlashInferStatus flashinfer_top_k_sampling(
    const void* probs,
    int32_t* output,
    const int32_t* top_k_arr,
    uint32_t top_k_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    uint64_t philox_seed,
    uint64_t philox_offset,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0 || top_k_val == 0) {
        set_error("Invalid argument: batch_size, vocab_size, and top_k must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Sampling only supports float32 dtype. Convert probs to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    // NOTE: top_k_arr must be converted to float* if non-null (but FlashInfer expects T* for top_k)
    // For simplicity, we pass nullptr and use top_k_val uniformly
    cudaError_t err = call_top_k_sampling_f32(
        const_cast<float*>(static_cast<const float*>(probs)),
        output,
        nullptr,  // Use uniform top_k_val
        batch_size, top_k_val, vocab_size,
        deterministic != 0, philox_seed, philox_offset, cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_top_p_sampling(
    const void* probs,
    int32_t* output,
    const float* top_p_arr,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    uint64_t philox_seed,
    uint64_t philox_offset,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0) {
        set_error("Invalid argument: batch_size and vocab_size must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Sampling only supports float32 dtype. Convert probs to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = call_top_p_sampling_f32(
        const_cast<float*>(static_cast<const float*>(probs)),
        output,
        const_cast<float*>(top_p_arr),
        batch_size, top_p_val, vocab_size,
        deterministic != 0, philox_seed, philox_offset, cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_min_p_sampling(
    const void* probs,
    int32_t* output,
    const float* min_p_arr,
    float min_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    uint64_t philox_seed,
    uint64_t philox_offset,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0) {
        set_error("Invalid argument: batch_size and vocab_size must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Sampling only supports float32 dtype. Convert probs to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = call_min_p_sampling_f32(
        const_cast<float*>(static_cast<const float*>(probs)),
        const_cast<float*>(min_p_arr),
        output,
        batch_size, min_p_val, vocab_size,
        deterministic != 0, philox_seed, philox_offset, cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_top_k_top_p_sampling(
    const void* probs,
    int32_t* output,
    const int32_t* top_k_arr,
    const float* top_p_arr,
    uint32_t top_k_val,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    uint64_t philox_seed,
    uint64_t philox_offset,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !output) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0 || top_k_val == 0) {
        set_error("Invalid argument: batch_size, vocab_size, and top_k must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Sampling only supports float32 dtype. Convert probs to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = call_top_k_top_p_sampling_f32(
        const_cast<float*>(static_cast<const float*>(probs)),
        const_cast<int32_t*>(top_k_arr),
        const_cast<float*>(top_p_arr),
        output,
        batch_size, static_cast<int32_t>(top_k_val), top_p_val, vocab_size,
        deterministic != 0, philox_seed, philox_offset, cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_softmax(
    const void* logits,
    void* probs,
    const float* temperature_arr,
    float temp_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!logits || !probs) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0) {
        set_error("Invalid argument: batch_size and vocab_size must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // NOTE: OnlineSoftmax only supports float32 due to type conversion issues with half/bf16.
    // For half/bf16 inputs, users should convert to float32 first, apply softmax, then convert back.
    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Softmax only supports float32 dtype. Convert logits to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = call_online_softmax_f32(
        const_cast<float*>(static_cast<const float*>(logits)),
        static_cast<float*>(probs),
        batch_size, vocab_size,
        const_cast<float*>(temperature_arr),
        temp_val, cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_top_k_mask_logits(
    void* logits,
    const int32_t* top_k_arr,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    // NOTE: FlashInfer does not provide a direct TopKMaskLogits function.
    // This would need to be implemented separately or use a different approach.
    // For now, we return unsupported.
    set_error("Top-K mask logits not yet implemented - use top_k_sampling directly");
    return FLASHINFER_UNSUPPORTED;
}

FlashInferStatus flashinfer_top_p_renorm_probs(
    const void* probs,
    void* renormed_probs,
    const float* top_p_arr,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !renormed_probs) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || vocab_size == 0) {
        set_error("Invalid argument: batch_size and vocab_size must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;

    // NOTE: Sampling operations only support float32 due to type compatibility issues
    // in FlashInfer's internal arithmetic with half/bf16.
    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("Sampling only supports float32 dtype. Convert probs to float32 before calling.");
        return FLASHINFER_UNSUPPORTED;
    }

    err = call_top_p_renorm_prob_f32(
        const_cast<float*>(static_cast<const float*>(probs)),
        static_cast<float*>(renormed_probs),
        const_cast<float*>(top_p_arr),
        batch_size, top_p_val, vocab_size, cuda_stream
    );

    return from_cuda_error(err);
}

/* ============================================================================
 * RoPE API Implementation
 * ============================================================================ */

FlashInferStatus flashinfer_apply_rope(
    const void* q,
    const void* k,
    void* q_out,
    void* k_out,
    const int32_t* indptr,
    const int32_t* offsets,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    const FlashInferRoPEConfig* config,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!q || !k || !q_out || !k_out || !indptr || !offsets || !config) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;
    float rope_scale = config->scale;
    float rope_theta = config->theta;

    cudaError_t err = cudaSuccess;
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16: {
            using DType = half;
            err = call_batch_qk_apply_rotary<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, rope_scale, rope_theta, cuda_stream
            );
            break;
        }
        case FLASHINFER_DTYPE_BFLOAT16: {
            using DType = nv_bfloat16;
            err = call_batch_qk_apply_rotary<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, rope_scale, rope_theta, cuda_stream
            );
            break;
        }
        default:
            set_error("Unsupported dtype for RoPE");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_apply_rope_inplace(
    void* q,
    void* k,
    const int32_t* indptr,
    const int32_t* offsets,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    const FlashInferRoPEConfig* config,
    FlashInferDType dtype,
    void* stream
) {
    // In-place: output same as input
    return flashinfer_apply_rope(
        q, k, q, k,
        indptr, offsets, batch_size,
        num_qo_heads, num_kv_heads, head_dim,
        config, dtype, stream
    );
}

FlashInferStatus flashinfer_apply_rope_pos_ids(
    const void* q,
    const void* k,
    void* q_out,
    void* k_out,
    const int32_t* pos_ids,
    uint32_t total_tokens,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    const FlashInferRoPEConfig* config,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!q || !k || !q_out || !k_out || !pos_ids || !config) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (total_tokens == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;
    float rope_scale = config->scale;
    float rope_theta = config->theta;

    cudaError_t err = cudaSuccess;
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16: {
            using DType = half;
            err = call_batch_qk_apply_rotary_pos_ids<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<int32_t*>(pos_ids),
                total_tokens, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, rope_scale, rope_theta, cuda_stream
            );
            break;
        }
        case FLASHINFER_DTYPE_BFLOAT16: {
            using DType = nv_bfloat16;
            err = call_batch_qk_apply_rotary_pos_ids<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<int32_t*>(pos_ids),
                total_tokens, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, rope_scale, rope_theta, cuda_stream
            );
            break;
        }
        default:
            set_error("Unsupported dtype for RoPE");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_apply_rope_with_cos_sin_cache(
    const void* q,
    const void* k,
    void* q_out,
    void* k_out,
    const void* cos_cache,
    const void* sin_cache,
    const int32_t* pos_ids,
    uint32_t total_tokens,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    const FlashInferRoPEConfig* config,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!q || !k || !q_out || !k_out || !cos_cache || !sin_cache || !pos_ids || !config) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (total_tokens == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;

    // FlashInfer expects cos_sin_cache as interleaved [max_seq_len, rotary_dim]
    // where each row contains [cos_0, sin_0, cos_1, sin_1, ...]
    // If the caller provides separate cos/sin caches, we need to handle that.
    // For now, we assume cos_cache is actually the combined cos_sin_cache.
    // NOTE: The caller should pass the combined cache, or we need to combine them here.

    cudaError_t err = cudaSuccess;
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16: {
            using DType = half;
            err = call_batch_qk_apply_rotary_cos_sin_cache<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<float*>(static_cast<const float*>(cos_cache)),  // Combined cache
                const_cast<int32_t*>(pos_ids),
                total_tokens, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, cuda_stream
            );
            break;
        }
        case FLASHINFER_DTYPE_BFLOAT16: {
            using DType = nv_bfloat16;
            err = call_batch_qk_apply_rotary_cos_sin_cache<DType>(
                const_cast<DType*>(static_cast<const DType*>(q)),
                const_cast<DType*>(static_cast<const DType*>(k)),
                static_cast<DType*>(q_out),
                static_cast<DType*>(k_out),
                const_cast<float*>(static_cast<const float*>(cos_cache)),  // Combined cache
                const_cast<int32_t*>(pos_ids),
                total_tokens, num_qo_heads, num_kv_heads,
                rotary_dim, head_dim,
                interleave, cuda_stream
            );
            break;
        }
        default:
            set_error("Unsupported dtype for RoPE");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

/* ============================================================================
 * Page Management API Implementation
 * ============================================================================ */

FlashInferStatus flashinfer_get_batch_indices_positions(
    const int32_t* append_indptr,
    const int32_t* seq_lens,
    int32_t* batch_indices,
    int32_t* positions,
    uint32_t batch_size,
    uint32_t total_tokens,
    void* stream
) {
    clear_error();

    if (!append_indptr || !seq_lens || !batch_indices || !positions) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (batch_size == 0 || total_tokens == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Launch kernel to compute batch_indices and positions
    uint32_t block_size = 256;
    uint32_t num_blocks = (total_tokens + block_size - 1) / block_size;
    compute_batch_indices_positions_kernel<<<num_blocks, block_size, 0, cuda_stream>>>(
        append_indptr, seq_lens, batch_indices, positions, batch_size);

    cudaError_t err = cudaGetLastError();
    return from_cuda_error(err);
}

/* ============================================================================
 * Additional Utility API Implementation
 * ============================================================================ */

FlashInferStatus flashinfer_get_workspace_size(
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    uint32_t batch_size,
    uint32_t max_seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size
) {
    clear_error();

    // Use the same logic as batch_decode_workspace_size
    return flashinfer_batch_decode_workspace_size(
        float_workspace_size, int_workspace_size,
        batch_size, num_heads, num_heads, head_dim, page_size, max_seq_len
    );
}

FlashInferStatus flashinfer_get_device_info(
    int device_id,
    int* compute_major,
    int* compute_minor,
    int* sm_count,
    int* supports_pdl,
    int* supports_fp8
) {
    clear_error();

    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    if (compute_major) *compute_major = props.major;
    if (compute_minor) *compute_minor = props.minor;
    if (sm_count) *sm_count = props.multiProcessorCount;

    int sm_version = props.major * 10 + props.minor;
    if (supports_pdl) *supports_pdl = (sm_version >= 90) ? 1 : 0;
    if (supports_fp8) *supports_fp8 = (sm_version >= 89) ? 1 : 0;

    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_set_stream(void* stream) {
    // This is a no-op - we pass streams to each function
    return FLASHINFER_SUCCESS;
}

/* ============================================================================
 * Template instantiations
 * ============================================================================
 *
 * These explicit instantiations ensure the kernels are compiled for
 * the supported dtype/head_dim combinations:
 *
 * - half (fp16) × {64, 128, 256}
 * - nv_bfloat16 (bf16) × {64, 128, 256}
 *
 * Adding more combinations increases compile time and binary size.
 * ============================================================================ */

// Normalization kernel instantiations are implicit (header-only templates)
// The compiler will instantiate them based on the dtype switch in the functions above.

// Attention kernel instantiations will be added when the full Plan-Run API is implemented.

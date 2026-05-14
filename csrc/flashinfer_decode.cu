/*
 * FlashInfer Batch Decode Implementation
 *
 * Contains batch decode attention kernels with 8 attention variants.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>

using namespace flashinfer_rs;

namespace {

// Standard attention variant: no custom mask, no sliding window, no soft cap, no alibi
using StandardAttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;
using SlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, false>;
using SoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, false>;
using SlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, false>;
using ALiBiAttentionVariant = flashinfer::DefaultAttention<false, false, false, true>;
using ALiBiSlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, true>;
using ALiBiSoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, true>;
using ALiBiSlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, true>;

// Template helper to call DecodePlan
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

    uint32_t num_kv_heads = num_qo_heads;

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
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = static_cast<DType*>(output);
    params.lse = lse;
    params.maybe_alibi_slopes = plan->alibi_slopes;
    params.padded_batch_size = plan->plan_info.padded_batch_size;
    params.num_qo_heads = plan->num_qo_heads;
    params.q_stride_n = plan->num_qo_heads * HEAD_DIM;
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

// Wrapper that dispatches to correct attention variant
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
    const bool use_sliding_window = (plan->window_left >= 0);
    const bool use_soft_cap = (plan->logits_soft_cap > 0.0f);
    const bool use_alibi = (plan->alibi_slopes != nullptr);

    // 8-way dispatch
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

// =============================================================================
// FP8 KV-cache decode helpers
//
// Upstream BatchDecodeWithPagedKVCacheDispatched is template-based but not
// scale-aware (BatchDecodeParams has only sm_scale). We add per-tensor scaling
// at the FFI boundary:
//
//   softmax(Q_real @ K_real^T * sm_scale_base)
//     = softmax(Q_bits @ K_bits^T * sm_scale_base * q_scale * k_scale)
// so we set params.sm_scale = plan->sm_scale * q_scale * k_scale.
//
//   out_real = Σ softmax_weights * V_real
//            = Σ softmax_weights * V_bits * v_scale
// so we multiply the output by v_scale post-launch (V is linear in attention).
//
//   LSE = log Σ exp(scores * sm_scale)   — already correct after sm_scale
// baking. With `correct_lse_for_v_scale`, we additionally shift LSE by
// log(v_scale) for callers that compose this into a fused merge.
// =============================================================================

template <typename T>
__global__ void output_post_scale_kernel(T* output, float scale, size_t n) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid < n) {
        output[tid] = static_cast<T>(static_cast<float>(output[tid]) * scale);
    }
}

__global__ void lse_post_shift_kernel(float* lse, float shift, size_t n) {
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid < n) {
        lse[tid] += shift;
    }
}

// Single instantiation of the FP8 decode path. Mirrors call_batch_decode_run_impl
// but allows DTypeQ/DTypeKV/DTypeO to differ, and bakes Q/K scales into sm_scale,
// then post-scales the output by v_scale.
template <typename DTypeQ, typename DTypeKV, typename DTypeO,
          uint32_t HEAD_DIM, typename AttentionVariant>
cudaError_t call_batch_decode_run_fp8_impl(
    const BatchDecodePlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    float q_scale,
    float k_scale,
    float v_scale,
    bool correct_lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    using Params = flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, int32_t>;
    constexpr flashinfer::PosEncodingMode POS_ENCODING_MODE = flashinfer::PosEncodingMode::kNone;

    flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv(
        plan->num_kv_heads,
        plan->page_size,
        HEAD_DIM,
        plan->batch_size,
        kv_layout,
        // KV buffers are FP8 bytes; cast to the appropriate FP8 type.
        const_cast<DTypeKV*>(static_cast<const DTypeKV*>(k_cache)),
        const_cast<DTypeKV*>(static_cast<const DTypeKV*>(v_cache)),
        const_cast<int32_t*>(kv_indices),
        const_cast<int32_t*>(kv_indptr),
        const_cast<int32_t*>(kv_last_page_len)
    );

    Params params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = static_cast<DTypeO*>(output);
    params.lse = lse;
    params.maybe_alibi_slopes = plan->alibi_slopes;
    params.padded_batch_size = plan->plan_info.padded_batch_size;
    params.num_qo_heads = plan->num_qo_heads;
    params.q_stride_n = plan->num_qo_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.window_left = plan->window_left;
    params.logits_soft_cap = plan->logits_soft_cap;
    // Scale baking: Q/K dequant lifted into the softmax scale.
    params.sm_scale = plan->sm_scale * q_scale * k_scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

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

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;

    if (plan_info.split_kv) {
        tmp_v = flashinfer::GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask = flashinfer::GetPtrFromBaseOffset<bool>(
                int_buffer, plan_info.block_valid_mask_offset);
        }
    }
    params.partition_kv = plan_info.split_kv;

    cudaError_t err = flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM,
                                                                         POS_ENCODING_MODE,
                                                                         AttentionVariant>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    if (err != cudaSuccess) return err;

    // V dequant: scale the output by v_scale.
    if (v_scale != 1.0f) {
        const size_t total = static_cast<size_t>(plan->batch_size) *
                             static_cast<size_t>(plan->num_qo_heads) * HEAD_DIM;
        const uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>((total + threads - 1) / threads);
        output_post_scale_kernel<DTypeO><<<blocks, threads, 0, stream>>>(
            static_cast<DTypeO*>(output), v_scale, total);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // Optional LSE shift so it reflects log of post-scaled (real) output magnitudes.
    if (correct_lse && lse != nullptr && v_scale != 1.0f && v_scale > 0.0f) {
        const size_t n = static_cast<size_t>(plan->batch_size) *
                         static_cast<size_t>(plan->num_qo_heads);
        const uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>((n + threads - 1) / threads);
        lse_post_shift_kernel<<<blocks, threads, 0, stream>>>(
            lse, std::log(v_scale), n);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

// Dispatcher over the 8 AttentionVariants for the FP8 path. Mirrors
// call_batch_decode_run() above.
template <typename DTypeQ, typename DTypeKV, typename DTypeO, uint32_t HEAD_DIM>
cudaError_t call_batch_decode_run_fp8(
    const BatchDecodePlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    float q_scale,
    float k_scale,
    float v_scale,
    bool correct_lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    const bool use_sliding_window = (plan->window_left >= 0);
    const bool use_soft_cap = (plan->logits_soft_cap > 0.0f);
    const bool use_alibi = (plan->alibi_slopes != nullptr);

#define CALL_FP8_VARIANT(VARIANT)                                                 \
    return call_batch_decode_run_fp8_impl<DTypeQ, DTypeKV, DTypeO, HEAD_DIM,      \
                                          VARIANT>(                               \
        plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,       \
        output, lse, q_scale, k_scale, v_scale, correct_lse, kv_layout, stream)

    if (use_alibi) {
        if (use_sliding_window && use_soft_cap) {
            CALL_FP8_VARIANT(ALiBiSlidingWindowSoftCapAttentionVariant);
        } else if (use_sliding_window) {
            CALL_FP8_VARIANT(ALiBiSlidingWindowAttentionVariant);
        } else if (use_soft_cap) {
            CALL_FP8_VARIANT(ALiBiSoftCapAttentionVariant);
        } else {
            CALL_FP8_VARIANT(ALiBiAttentionVariant);
        }
    } else {
        if (use_sliding_window && use_soft_cap) {
            CALL_FP8_VARIANT(SlidingWindowSoftCapAttentionVariant);
        } else if (use_sliding_window) {
            CALL_FP8_VARIANT(SlidingWindowAttentionVariant);
        } else if (use_soft_cap) {
            CALL_FP8_VARIANT(SoftCapAttentionVariant);
        } else {
            CALL_FP8_VARIANT(StandardAttentionVariant);
        }
    }
#undef CALL_FP8_VARIANT
}

// (q_dtype, kv_dtype) → (DTypeQ, DTypeKV, DTypeO) dispatch.
// O matches Q for the FP16/BF16-Q paths; for FP8-Q it falls back to BF16 since
// FP8 output requires per-tensor output scaling/clamping that we do not plumb
// through here — callers can re-quantize the BF16 output if they need FP8 logits.
//
// `head_dim` is hand-dispatched to avoid duplicating the FP16-only head_dim
// table from DISPATCH_DTYPE_HEAD_DIM.
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_fp8_head_dim(
    int32_t head_dim,
    const BatchDecodePlan* plan,
    const void* q, const void* k_cache, const void* v_cache,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output, float* lse,
    float q_scale, float k_scale, float v_scale,
    bool correct_lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream,
    bool* dispatched_out
) {
    *dispatched_out = true;
    if (head_dim == 64) {
        return call_batch_decode_run_fp8<DTypeQ, DTypeKV, DTypeO, 64>(
            plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
            output, lse, q_scale, k_scale, v_scale, correct_lse, kv_layout, stream);
    } else if (head_dim == 128) {
        return call_batch_decode_run_fp8<DTypeQ, DTypeKV, DTypeO, 128>(
            plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
            output, lse, q_scale, k_scale, v_scale, correct_lse, kv_layout, stream);
    } else if (head_dim == 256) {
        return call_batch_decode_run_fp8<DTypeQ, DTypeKV, DTypeO, 256>(
            plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
            output, lse, q_scale, k_scale, v_scale, correct_lse, kv_layout, stream);
    }
    *dispatched_out = false;
    return cudaErrorInvalidValue;
}

} // anonymous namespace

extern "C" {

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

    int32_t max_num_pages = (max_seq_len + page_size - 1) / page_size;

    // Float workspace: tmp_v for split-k, tmp_s for softmax
    size_t tmp_v_size = batch_size * num_qo_heads * head_dim * sizeof(float);
    size_t tmp_s_size = batch_size * num_qo_heads * sizeof(float);
    *float_workspace_size = tmp_v_size + tmp_s_size + 4096;

    // Int workspace: request_indices, kv_tile_indices, o_indptr, etc.
    size_t partition_info_size = batch_size * num_kv_heads * 2 * sizeof(int32_t);
    size_t page_indices_size = batch_size * max_num_pages * sizeof(int32_t);
    *int_workspace_size = partition_info_size + page_indices_size + 4096;

    return FLASHINFER_SUCCESS;
}

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

    if (batch_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

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
    plan->window_left = window_left;
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
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
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

    DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DType, HEAD_DIM_V, {
        err = call_batch_decode_run<DType, HEAD_DIM_V>(
            plan, q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
            output, lse, fi_kv_layout, cuda_stream);
    });

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_batch_decode_run_fp8(
    FlashInferBatchDecodePlanHandle plan_handle,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    float q_scale,
    float k_scale,
    float v_scale,
    int correct_lse_for_v_scale,
    FlashInferDType q_dtype,
    FlashInferDType kv_dtype,
    FlashInferKVLayout kv_layout,
    void* stream
) {
    clear_error();

    if (!plan_handle || !q || !k_cache || !v_cache || !output ||
        !kv_indptr || !kv_indices || !kv_last_page_len) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (!std::isfinite(q_scale) || !std::isfinite(k_scale) || !std::isfinite(v_scale)) {
        set_error("q_scale, k_scale, v_scale must be finite");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (kv_dtype != FLASHINFER_DTYPE_FLOAT8_E4M3 &&
        kv_dtype != FLASHINFER_DTYPE_FLOAT8_E5M2) {
        set_error("flashinfer_batch_decode_run_fp8: kv_dtype must be FLOAT8_E4M3 or FLOAT8_E5M2");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    BatchDecodePlan* plan = reinterpret_cast<BatchDecodePlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout fi_kv_layout = to_qkv_layout(kv_layout);
    const bool correct_lse = (correct_lse_for_v_scale != 0);
    const int32_t head_dim = plan->head_dim;

    cudaError_t err = cudaSuccess;
    bool dispatched = false;

    // (q_dtype, kv_dtype) → (DTypeQ, DTypeKV, DTypeO). DTypeO matches Q for
    // FP16/BF16 Q; FP8 Q falls back to BF16 output (see docstring in the header).
#define DISPATCH_FP8(DQ, DKV, DO)                                                  \
    err = dispatch_fp8_head_dim<DQ, DKV, DO>(                                      \
        head_dim, plan, q, k_cache, v_cache, kv_indptr, kv_indices,                \
        kv_last_page_len, output, lse, q_scale, k_scale, v_scale, correct_lse,     \
        fi_kv_layout, cuda_stream, &dispatched)

    if (q_dtype == FLASHINFER_DTYPE_FLOAT16) {
        if (kv_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_FP8(half, __nv_fp8_e4m3, half);
        } else {
            DISPATCH_FP8(half, __nv_fp8_e5m2, half);
        }
    } else if (q_dtype == FLASHINFER_DTYPE_BFLOAT16) {
        if (kv_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_FP8(nv_bfloat16, __nv_fp8_e4m3, nv_bfloat16);
        } else {
            DISPATCH_FP8(nv_bfloat16, __nv_fp8_e5m2, nv_bfloat16);
        }
    } else if (q_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
        if (kv_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_FP8(__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16);
        } else {
            DISPATCH_FP8(__nv_fp8_e4m3, __nv_fp8_e5m2, nv_bfloat16);
        }
    } else if (q_dtype == FLASHINFER_DTYPE_FLOAT8_E5M2) {
        if (kv_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_FP8(__nv_fp8_e5m2, __nv_fp8_e4m3, nv_bfloat16);
        } else {
            DISPATCH_FP8(__nv_fp8_e5m2, __nv_fp8_e5m2, nv_bfloat16);
        }
    } else {
        set_error("flashinfer_batch_decode_run_fp8: q_dtype must be FLOAT16, BFLOAT16, FLOAT8_E4M3, or FLOAT8_E5M2");
        return FLASHINFER_INVALID_ARGUMENT;
    }
#undef DISPATCH_FP8

    if (!dispatched) {
        set_error("flashinfer_batch_decode_run_fp8: unsupported head_dim (only 64, 128, 256)");
        return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_batch_decode_plan_destroy(
    FlashInferBatchDecodePlanHandle plan_handle
) {
    if (plan_handle) {
        BatchDecodePlan* plan = reinterpret_cast<BatchDecodePlan*>(plan_handle);
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

} // extern "C"

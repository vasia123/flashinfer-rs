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

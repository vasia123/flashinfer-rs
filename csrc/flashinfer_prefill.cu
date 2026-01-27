/*
 * FlashInfer Batch Prefill Implementation
 *
 * Contains batch prefill attention kernels with 8 attention variants.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/allocator.h>

using namespace flashinfer_rs;
using flashinfer::GetPtrFromBaseOffset;

namespace {

// Attention variant type aliases
// Standard variants (no ALiBi)
using StandardAttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;
using SlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, false>;
using SoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, false>;
using SlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, false>;

// ALiBi variants
using ALiBiAttentionVariant = flashinfer::DefaultAttention<false, false, false, true>;
using ALiBiSlidingWindowAttentionVariant = flashinfer::DefaultAttention<false, true, false, true>;
using ALiBiSoftCapAttentionVariant = flashinfer::DefaultAttention<false, false, true, true>;
using ALiBiSlidingWindowSoftCapAttentionVariant = flashinfer::DefaultAttention<false, true, true, true>;

// Batch prefill run implementation for specific dtype, head_dim, mask_mode, and attention variant
template<typename DType, unsigned int HEAD_DIM, flashinfer::MaskMode MASK_MODE, typename AttentionVariant>
cudaError_t call_batch_prefill_run_impl(
    const BatchPrefillPlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* kv_last_page_lens,
    const int32_t* qo_indptr,
    void* o,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    // Create paged KV cache descriptor
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(plan->num_kv_heads),
        static_cast<uint32_t>(plan->page_size),
        static_cast<uint32_t>(HEAD_DIM),
        static_cast<uint32_t>(plan->batch_size),
        kv_layout,
        const_cast<DType*>(static_cast<const DType*>(k_cache)),
        const_cast<DType*>(static_cast<const DType*>(v_cache)),
        const_cast<int32_t*>(kv_indices),
        const_cast<int32_t*>(kv_indptr),
        const_cast<int32_t*>(kv_last_page_lens),
        static_cast<int32_t*>(nullptr) // rope_pos_offset
    );

    // Compute strides for Q tensor layout [total_tokens, num_qo_heads, head_dim]
    int32_t q_stride_n = plan->num_qo_heads * HEAD_DIM;
    int32_t q_stride_h = HEAD_DIM;

    // Create prefill params with correct constructor signature:
    // BatchPrefillPagedParams(q, paged_kv, maybe_custom_mask, q_indptr, maybe_mask_indptr,
    //                         maybe_q_rope_offset, o, lse, maybe_alibi_slopes,
    //                         num_qo_heads, q_stride_n, q_stride_h,
    //                         window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta)
    flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t> params(
        const_cast<DType*>(static_cast<const DType*>(q)),  // q
        paged_kv,                                           // paged_kv
        static_cast<uint8_t*>(nullptr),                    // maybe_custom_mask
        const_cast<int32_t*>(qo_indptr),                   // q_indptr
        static_cast<int32_t*>(nullptr),                    // maybe_mask_indptr
        static_cast<int32_t*>(nullptr),                    // maybe_q_rope_offset
        static_cast<DType*>(o),                            // o (output)
        lse,                                               // lse
        plan->alibi_slopes,                                // maybe_alibi_slopes
        static_cast<uint32_t>(plan->num_qo_heads),         // num_qo_heads
        q_stride_n,                                        // q_stride_n
        q_stride_h,                                        // q_stride_h
        plan->window_left,                                 // window_left
        plan->logits_soft_cap,                             // logits_soft_cap
        plan->sm_scale,                                    // sm_scale
        1.0f,                                              // rope_scale
        1e4f                                               // rope_theta
    );

    // Compute pointers from int_workspace using offsets from plan_info
    void* int_buffer_ptr = plan->int_workspace;
    void* float_buffer_ptr = plan->float_workspace;
    const flashinfer::PrefillPlanInfo& plan_info = plan->plan_info;

    params.request_indices = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.request_indices_offset);
    params.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);

    // Handle split_kv mode
    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = GetPtrFromBaseOffset<int32_t>(int_buffer_ptr, plan_info.merge_indptr_offset);
        tmp_v = GetPtrFromBaseOffset<DType>(float_buffer_ptr, plan_info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask = GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
        }
    }

    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.partition_kv = plan_info.split_kv;

    if (plan_info.enable_cuda_graph) {
        params.total_num_rows = GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
    }

    // Call the prefill kernel
    // enable_pdl (programmatic dependent launch) disabled for compatibility
    constexpr bool enable_pdl = false;

    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        64,  // CTA_TILE_Q
        HEAD_DIM, HEAD_DIM,
        flashinfer::PosEncodingMode::kNone,
        false, // USE_FP16_QK_REDUCTION
        MASK_MODE,
        AttentionVariant,
        flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>
    >(params, tmp_v, tmp_s, enable_pdl, stream);
}

// Batch prefill run with variant selection based on plan configuration
template<typename DType, unsigned int HEAD_DIM, flashinfer::MaskMode MASK_MODE>
cudaError_t call_batch_prefill_run(
    const BatchPrefillPlan* plan,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* kv_last_page_lens,
    const int32_t* qo_indptr,
    void* o,
    float* lse,
    flashinfer::QKVLayout kv_layout,
    cudaStream_t stream
) {
    const bool use_alibi = (plan->alibi_slopes != nullptr);
    const bool use_sliding_window = (plan->window_left >= 0);
    const bool use_soft_cap = (plan->logits_soft_cap != 0.0f);

    // 8-way dispatch based on feature combination
    if (use_alibi) {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, ALiBiAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        }
    } else {
        if (use_sliding_window && use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SlidingWindowSoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else if (use_sliding_window) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SlidingWindowAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else if (use_soft_cap) {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, SoftCapAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        } else {
            return call_batch_prefill_run_impl<DType, HEAD_DIM, MASK_MODE, StandardAttentionVariant>(
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_lens, qo_indptr, o, lse, kv_layout, stream);
        }
    }
}

} // anonymous namespace

extern "C" {

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

    if (!float_workspace_size || !int_workspace_size) {
        set_error("float_workspace_size and int_workspace_size must not be null");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // FlashInfer prefill workspace estimation
    // Float workspace for split tensors
    size_t split_kv_workspace = static_cast<size_t>(batch_size) * num_qo_heads * head_dim * sizeof(float) * 16;
    // Int workspace for indices
    size_t split_idx_workspace = static_cast<size_t>(batch_size) * sizeof(int32_t) * 8;

    // Align to 256 bytes
    *float_workspace_size = ((split_kv_workspace + 255) / 256) * 256;
    *int_workspace_size = ((split_idx_workspace + 255) / 256) * 256;

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

    // Validate inputs
    if (!plan_handle || !float_workspace || !int_workspace || !qo_indptr || !kv_indptr) {
        set_error("plan_handle, workspaces, qo_indptr, and kv_indptr must not be null");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Position encoding validation
    if (pos_encoding != FLASHINFER_POS_ENCODING_NONE &&
        pos_encoding != FLASHINFER_POS_ENCODING_ALIBI) {
        set_error("Only NONE and ALIBI position encoding supported for batch prefill plan");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Read qo_indptr from GPU to compute total_num_rows
    std::vector<int32_t> h_qo_indptr(batch_size + 1);
    cudaError_t copy_err = cudaMemcpyAsync(h_qo_indptr.data(), qo_indptr,
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (copy_err != cudaSuccess) {
        return from_cuda_error(copy_err);
    }

    // Read kv_indptr from GPU
    std::vector<int32_t> h_kv_indptr(batch_size + 1);
    copy_err = cudaMemcpyAsync(h_kv_indptr.data(), kv_indptr,
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (copy_err != cudaSuccess) {
        return from_cuda_error(copy_err);
    }

    // Synchronize to get the values
    copy_err = cudaStreamSynchronize(cuda_stream);
    if (copy_err != cudaSuccess) {
        return from_cuda_error(copy_err);
    }

    uint32_t total_num_rows = static_cast<uint32_t>(h_qo_indptr[batch_size]);

    // Create plan
    auto* plan = new BatchPrefillPlan();
    plan->batch_size = batch_size;
    plan->num_qo_heads = num_qo_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype = dtype;
    plan->pos_encoding = pos_encoding;
    plan->logits_soft_cap = logits_soft_cap;
    plan->window_left = window_left;
    plan->causal = causal;
    plan->float_workspace = float_workspace;
    plan->int_workspace = int_workspace;
    plan->float_workspace_size = float_workspace_size;
    plan->int_workspace_size = int_workspace_size;
    plan->enable_cuda_graph = (enable_cuda_graph != 0);
    plan->sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    plan->alibi_slopes = nullptr;
    plan->owns_alibi_slopes = false;

    // Compute ALiBi slopes if needed
    if (pos_encoding == FLASHINFER_POS_ENCODING_ALIBI) {
        cudaError_t alibi_err = compute_alibi_slopes(&plan->alibi_slopes, num_qo_heads, cuda_stream);
        if (alibi_err != cudaSuccess) {
            delete plan;
            return from_cuda_error(alibi_err);
        }
        plan->owns_alibi_slopes = true;
    }

    // sizeof_dtype_o depends on dtype
    uint32_t sizeof_dtype_o = (dtype == FLASHINFER_DTYPE_FLOAT32) ? 4 : 2;

    // Call FlashInfer's PrefillPlan
    // Note: qo_indptr and kv_indptr must be host pointers for PrefillPlan
    cudaError_t err = flashinfer::PrefillPlan<int32_t>(
        float_workspace,
        float_workspace_size,
        int_workspace,
        page_locked_int_workspace,
        int_workspace_size,
        plan->plan_info,
        h_qo_indptr.data(),      // host pointer
        h_kv_indptr.data(),      // host pointer
        total_num_rows,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),  // head_dim_qk
        static_cast<uint32_t>(head_dim),  // head_dim_vo
        static_cast<uint32_t>(page_size),
        (enable_cuda_graph != 0),
        sizeof_dtype_o,
        window_left,
        -1,    // fixed_split_size (-1 = auto)
        false, // disable_split_kv
        0,     // num_colocated_ctas
        cuda_stream
    );

    if (err != cudaSuccess) {
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
        delete plan;
        return from_cuda_error(err);
    }

    *plan_handle = reinterpret_cast<FlashInferBatchPrefillPlanHandle>(plan);
    return FLASHINFER_SUCCESS;
}

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

    if (!plan_handle || !q || !k_cache || !v_cache || !kv_indptr || !kv_indices || !kv_last_page_len || !qo_indptr || !output) {
        set_error("Null pointer passed to batch_prefill_run");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    const BatchPrefillPlan* plan = reinterpret_cast<const BatchPrefillPlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout layout = (kv_layout == FLASHINFER_KV_LAYOUT_NHD) ?
        flashinfer::QKVLayout::kNHD : flashinfer::QKVLayout::kHND;

    cudaError_t err;

    // Determine mask mode from causal flag
    const bool use_causal = (plan->causal != 0);

    // Dispatch based on dtype, head_dim, and mask mode
    #define DISPATCH_PREFILL_CAUSAL(DType, HEAD_DIM_V) \
        if (use_causal) { \
            err = call_batch_prefill_run<DType, HEAD_DIM_V, flashinfer::MaskMode::kCausal>( \
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_len, qo_indptr, \
                output, lse, layout, cuda_stream); \
        } else { \
            err = call_batch_prefill_run<DType, HEAD_DIM_V, flashinfer::MaskMode::kNone>( \
                plan, q, k_cache, v_cache, kv_indices, kv_indptr, kv_last_page_len, qo_indptr, \
                output, lse, layout, cuda_stream); \
        }

    switch (plan->dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            switch (plan->head_dim) {
                case 64: DISPATCH_PREFILL_CAUSAL(__half, 64); break;
                case 128: DISPATCH_PREFILL_CAUSAL(__half, 128); break;
                case 256: DISPATCH_PREFILL_CAUSAL(__half, 256); break;
                default:
                    set_error("Unsupported head_dim: " + std::to_string(plan->head_dim));
                    return FLASHINFER_INVALID_ARGUMENT;
            }
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            switch (plan->head_dim) {
                case 64: DISPATCH_PREFILL_CAUSAL(__nv_bfloat16, 64); break;
                case 128: DISPATCH_PREFILL_CAUSAL(__nv_bfloat16, 128); break;
                case 256: DISPATCH_PREFILL_CAUSAL(__nv_bfloat16, 256); break;
                default:
                    set_error("Unsupported head_dim: " + std::to_string(plan->head_dim));
                    return FLASHINFER_INVALID_ARGUMENT;
            }
            break;
        default:
            set_error("Unsupported dtype for batch prefill");
            return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_PREFILL_CAUSAL

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_batch_prefill_plan_destroy(
    FlashInferBatchPrefillPlanHandle plan_handle
) {
    if (plan_handle) {
        BatchPrefillPlan* plan = reinterpret_cast<BatchPrefillPlan*>(plan_handle);
        if (plan->alibi_slopes && plan->owns_alibi_slopes) {
            cudaFree(plan->alibi_slopes);
        }
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

} // extern "C"

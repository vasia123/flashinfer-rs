/*
 * FlashInfer MLA (Multi-head Latent Attention) Operations
 *
 * Support for DeepSeek v2/v3 MLA architecture with:
 * - Separate ckv_cache and kpe_cache (compressed KV + K position embedding)
 * - Fixed dimensions: head_dim_ckv=512, head_dim_kpe=64, num_heads=128
 * - Softmax scale: 1/sqrt(192) instead of 1/sqrt(head_dim)
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <cuda_fp8.h>
#include <flashinfer/page.cuh>
#include <flashinfer/concat_mla.cuh>
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>

using namespace flashinfer_rs;

// MLA fixed dimensions (DeepSeek v2/v3)
constexpr uint32_t MLA_HEAD_DIM_CKV = 512;
constexpr uint32_t MLA_HEAD_DIM_KPE = 64;
constexpr uint32_t MLA_NUM_HEADS = 128;
constexpr uint32_t MLA_QK_NOPE_HEAD_DIM = 128;
constexpr uint32_t MLA_QK_ROPE_HEAD_DIM = 64;

// MLA Plan storage
struct MLAPlan {
    flashinfer::MLAPlanInfo plan_info;
    int32_t batch_size;
    int32_t num_heads;
    int32_t head_dim_ckv;
    int32_t head_dim_kpe;
    int32_t page_size;
    FlashInferDType dtype;
    bool causal;
    void* float_workspace;
    void* int_workspace;
    size_t float_workspace_size;
    size_t int_workspace_size;
    float sm_scale;
};

extern "C" {

FlashInferStatus flashinfer_append_paged_mla_kv_cache(
    const void* append_ckv,
    const void* append_kpe,
    void* ckv_cache,
    void* kpe_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* batch_indices,
    const int32_t* positions,
    uint32_t nnz,
    uint32_t batch_size,
    uint32_t page_size,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!append_ckv || !append_kpe || !ckv_cache || !kpe_cache ||
        !kv_indptr || !kv_indices || !kv_last_page_len ||
        !batch_indices || !positions) {
        set_error("Null pointer passed to append_paged_mla_kv_cache");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (nnz == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // MLA uses fixed dimensions
    constexpr uint32_t head_dim_ckv = MLA_HEAD_DIM_CKV;
    constexpr uint32_t head_dim_kpe = MLA_HEAD_DIM_KPE;

    // Strides for contiguous layout [nnz, head_dim]
    size_t append_ckv_stride_n = head_dim_ckv;
    size_t append_kpe_stride_n = head_dim_kpe;

    #define DISPATCH_APPEND_MLA(DType) \
        do { \
            flashinfer::paged_kv_mla_t<DType, int32_t> paged_kv_mla( \
                page_size, \
                head_dim_ckv, \
                head_dim_kpe, \
                batch_size, \
                static_cast<DType*>(ckv_cache), \
                static_cast<DType*>(kpe_cache), \
                const_cast<int32_t*>(kv_indices), \
                const_cast<int32_t*>(kv_indptr), \
                const_cast<int32_t*>(kv_last_page_len), \
                nullptr /* rope_pos_offset */ \
            ); \
            err = flashinfer::AppendPagedKVMlaCache( \
                paged_kv_mla, \
                const_cast<DType*>(static_cast<const DType*>(append_ckv)), \
                const_cast<DType*>(static_cast<const DType*>(append_kpe)), \
                const_cast<int32_t*>(batch_indices), \
                const_cast<int32_t*>(positions), \
                nnz, \
                append_ckv_stride_n, \
                append_kpe_stride_n, \
                cuda_stream \
            ); \
        } while (0)

    if (dtype == FLASHINFER_DTYPE_FLOAT16) {
        DISPATCH_APPEND_MLA(__half);
    } else if (dtype == FLASHINFER_DTYPE_BFLOAT16) {
        DISPATCH_APPEND_MLA(__nv_bfloat16);
    } else if (dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
        // FP8 append is a byte-level copy into the paged cache. Caller is
        // responsible for pre-quantizing the input tensors to FP8.
        DISPATCH_APPEND_MLA(__nv_fp8_e4m3);
    } else if (dtype == FLASHINFER_DTYPE_FLOAT8_E5M2) {
        DISPATCH_APPEND_MLA(__nv_fp8_e5m2);
    } else {
        set_error("append_paged_mla_kv_cache only supports float16, bfloat16, fp8_e4m3, fp8_e5m2");
        return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_APPEND_MLA

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_concat_mla_k(
    void* k,
    const void* k_nope,
    const void* k_rope,
    int32_t num_tokens,
    int64_t k_stride_n,
    int32_t k_stride_h,
    int64_t k_nope_stride_n,
    int32_t k_nope_stride_h,
    int64_t k_rope_stride_n,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!k || !k_nope || !k_rope) {
        set_error("Null pointer passed to concat_mla_k");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (num_tokens == 0) {
        return FLASHINFER_SUCCESS;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // ConcatMLAK has compile-time dispatch over FP16 / BF16 / FP8 (E4M3, E5M2)
    // since FlashInfer #3129 — see include/flashinfer/concat_mla.cuh ConcatMLAVecTraits.
    #define DISPATCH_CONCAT_MLA(DType)                                         \
        err = flashinfer::ConcatMLAK<DType>(                                   \
            static_cast<DType*>(k),                                            \
            static_cast<const DType*>(k_nope),                                 \
            static_cast<const DType*>(k_rope),                                 \
            num_tokens,                                                        \
            k_stride_n,                                                        \
            k_stride_h,                                                        \
            k_nope_stride_n,                                                   \
            k_nope_stride_h,                                                   \
            k_rope_stride_n,                                                   \
            cuda_stream                                                        \
        )

    if (dtype == FLASHINFER_DTYPE_FLOAT16) {
        DISPATCH_CONCAT_MLA(__half);
    } else if (dtype == FLASHINFER_DTYPE_BFLOAT16) {
        DISPATCH_CONCAT_MLA(__nv_bfloat16);
    } else if (dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
        DISPATCH_CONCAT_MLA(__nv_fp8_e4m3);
    } else if (dtype == FLASHINFER_DTYPE_FLOAT8_E5M2) {
        DISPATCH_CONCAT_MLA(__nv_fp8_e5m2);
    } else {
        set_error("concat_mla_k only supports float16, bfloat16, fp8_e4m3, fp8_e5m2");
        return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_CONCAT_MLA

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_mla_workspace_size(
    size_t* float_size,
    size_t* int_size,
    int32_t batch_size,
    int32_t max_seq_len,
    int32_t num_heads,
    int32_t head_dim_ckv,
    int32_t head_dim_kpe,
    int32_t page_size
) {
    clear_error();

    if (!float_size || !int_size) {
        set_error("Null pointer passed to mla_workspace_size");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Validate MLA dimensions
    if (head_dim_ckv != MLA_HEAD_DIM_CKV || head_dim_kpe != MLA_HEAD_DIM_KPE) {
        set_error("MLA requires head_dim_ckv=512 and head_dim_kpe=64");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Conservative workspace estimation for MLA attention
    // Based on FlashInfer's MLAPlanInfo requirements
    uint32_t max_num_partitions = (max_seq_len + 127) / 128;

    // Float workspace: partial outputs and LSE
    // partial_o: [batch_size * num_heads * max_partitions * head_dim_ckv]
    // partial_lse: [batch_size * num_heads * max_partitions]
    size_t partial_o_size = static_cast<size_t>(batch_size) * num_heads *
                            max_num_partitions * head_dim_ckv * sizeof(float);
    size_t partial_lse_size = static_cast<size_t>(batch_size) * num_heads *
                              max_num_partitions * sizeof(float);

    *float_size = partial_o_size + partial_lse_size + 1024; // Extra padding

    // Int workspace: indptr arrays, lengths, offsets
    // Conservative: 20 int32 arrays of size batch_size + 1
    *int_size = 20 * (batch_size + 1) * sizeof(int32_t) + 4096; // Extra padding

    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_mla_plan(
    FlashInferMLAPlanHandle* plan_handle,
    void* float_workspace,
    size_t float_workspace_size,
    void* int_workspace,
    size_t int_workspace_size,
    void* page_locked_workspace,
    size_t page_locked_size,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_len,
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim_ckv,
    int32_t head_dim_kpe,
    int32_t page_size,
    int causal,
    void* stream
) {
    clear_error();

    if (!plan_handle || !float_workspace || !int_workspace || !qo_indptr || !kv_indptr || !kv_len) {
        set_error("Null pointer passed to mla_plan");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Validate MLA dimensions
    if (head_dim_ckv != MLA_HEAD_DIM_CKV || head_dim_kpe != MLA_HEAD_DIM_KPE) {
        set_error("MLA requires head_dim_ckv=512 and head_dim_kpe=64");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (num_heads != MLA_NUM_HEADS) {
        set_error("MLA requires num_heads=128 (DeepSeek configuration)");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Allocate plan structure
    MLAPlan* plan = new (std::nothrow) MLAPlan();
    if (!plan) {
        set_error("Failed to allocate MLA plan");
        return FLASHINFER_OUT_OF_MEMORY;
    }

    plan->batch_size = batch_size;
    plan->num_heads = num_heads;
    plan->head_dim_ckv = head_dim_ckv;
    plan->head_dim_kpe = head_dim_kpe;
    plan->page_size = page_size;
    plan->dtype = FLASHINFER_DTYPE_FLOAT16; // Will be set at run time
    plan->causal = causal != 0;
    plan->float_workspace = float_workspace;
    plan->int_workspace = int_workspace;
    plan->float_workspace_size = float_workspace_size;
    plan->int_workspace_size = int_workspace_size;
    // MLA softmax scale: 1/sqrt(nope_dim + rope_dim) = 1/sqrt(128 + 64) = 1/sqrt(192)
    plan->sm_scale = 1.0f / std::sqrt(192.0f);

    // Call FlashInfer's MLAPlan
    cudaError_t err = flashinfer::MLAPlan(
        float_workspace,
        float_workspace_size,
        int_workspace,
        page_locked_workspace,
        int_workspace_size,
        plan->plan_info,
        const_cast<int32_t*>(qo_indptr),
        const_cast<int32_t*>(kv_indptr),
        const_cast<int32_t*>(kv_len),
        batch_size,
        num_heads,
        head_dim_ckv,
        causal != 0,
        cuda_stream
    );

    if (err != cudaSuccess) {
        delete plan;
        set_error(std::string("MLAPlan failed: ") + cudaGetErrorString(err));
        return FLASHINFER_CUDA_ERROR;
    }

    *plan_handle = reinterpret_cast<FlashInferMLAPlanHandle>(plan);
    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_mla_run(
    FlashInferMLAPlanHandle plan_handle,
    const void* q_nope,
    const void* q_pe,
    const void* ckv_cache,
    const void* kpe_cache,
    const int32_t* kv_indices,
    void* output,
    float* lse,
    FlashInferMaskMode mask_mode,
    float sm_scale,
    FlashInferDType dtype_q,
    FlashInferDType dtype_kv,
    void* stream
) {
    clear_error();

    if (!plan_handle || !q_nope || !q_pe || !ckv_cache || !kpe_cache || !kv_indices || !output) {
        set_error("Null pointer passed to mla_run");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    MLAPlan* plan = reinterpret_cast<MLAPlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Use plan's sm_scale if caller passes 0
    float actual_sm_scale = (sm_scale == 0.0f) ? plan->sm_scale : sm_scale;

    // Convert mask mode
    flashinfer::MaskMode fi_mask_mode;
    switch (mask_mode) {
        case FLASHINFER_MASK_CAUSAL:
            fi_mask_mode = flashinfer::MaskMode::kCausal;
            break;
        case FLASHINFER_MASK_NONE:
        default:
            fi_mask_mode = flashinfer::MaskMode::kNone;
            break;
    }

    // MLA attention dispatch
    // Note: FlashInfer's BatchMLAPagedAttention has fixed dimensions
    #define DISPATCH_MLA_RUN(DTypeQ, DTypeKV) \
        do { \
            flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeQ, int32_t> params; \
            \
            params.q_nope = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_nope)); \
            params.q_pe = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_pe)); \
            params.ckv = const_cast<DTypeKV*>(static_cast<const DTypeKV*>(ckv_cache)); \
            params.kpe = const_cast<DTypeKV*>(static_cast<const DTypeKV*>(kpe_cache)); \
            \
            params.q_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.q_indptr_offset); \
            params.kv_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.kv_indptr_offset); \
            params.partial_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.partial_indptr_offset); \
            params.kv_indices = const_cast<int32_t*>(kv_indices); \
            params.q_len = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.q_len_offset); \
            params.kv_len = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.kv_len_offset); \
            params.q_start = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.q_start_offset); \
            params.kv_start = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.kv_start_offset); \
            params.kv_end = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.kv_end_offset); \
            params.work_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.work_indptr_offset); \
            params.merge_packed_offset_start = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.merge_packed_offset_start_offset); \
            params.merge_packed_offset_end = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.merge_packed_offset_end_offset); \
            params.merge_partial_packed_offset_start = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.merge_partial_packed_offset_start_offset); \
            params.merge_partial_packed_offset_end = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.merge_partial_packed_offset_end_offset); \
            params.merge_partial_stride = flashinfer::GetPtrFromBaseOffset<int32_t>( \
                plan->int_workspace, plan->plan_info.merge_partial_stride_offset); \
            \
            params.final_o = static_cast<DTypeQ*>(output); \
            params.final_lse = lse; \
            params.partial_o = flashinfer::GetPtrFromBaseOffset<DTypeQ>( \
                plan->float_workspace, plan->plan_info.partial_o_offset); \
            params.partial_lse = flashinfer::GetPtrFromBaseOffset<float>( \
                plan->float_workspace, plan->plan_info.partial_lse_offset); \
            \
            params.num_heads = flashinfer::uint_fastdiv(plan->num_heads); \
            params.block_size = flashinfer::uint_fastdiv(plan->page_size); \
            \
            /* Strides for contiguous layout */ \
            params.q_nope_stride_n = static_cast<uint32_t>(plan->num_heads) * plan->head_dim_ckv; \
            params.q_nope_stride_h = plan->head_dim_ckv; \
            params.q_pe_stride_n = static_cast<uint32_t>(plan->num_heads) * plan->head_dim_kpe; \
            params.q_pe_stride_h = plan->head_dim_kpe; \
            params.ckv_stride_page = static_cast<uint32_t>(plan->page_size) * plan->head_dim_ckv; \
            params.ckv_stride_n = plan->head_dim_ckv; \
            params.kpe_stride_page = static_cast<uint32_t>(plan->page_size) * plan->head_dim_kpe; \
            params.kpe_stride_n = plan->head_dim_kpe; \
            params.o_stride_n = params.q_nope_stride_n; \
            params.o_stride_h = params.q_nope_stride_h; \
            \
            params.sm_scale = actual_sm_scale; \
            params.return_lse_base_on_e = false; \
            \
            cudaError_t err; \
            if (fi_mask_mode == flashinfer::MaskMode::kCausal) { \
                err = flashinfer::mla::BatchMLAPagedAttention< \
                    flashinfer::MaskMode::kCausal, MLA_HEAD_DIM_CKV, MLA_HEAD_DIM_KPE>( \
                    params, plan->plan_info.num_blks_x, plan->plan_info.num_blks_y, cuda_stream); \
            } else { \
                err = flashinfer::mla::BatchMLAPagedAttention< \
                    flashinfer::MaskMode::kNone, MLA_HEAD_DIM_CKV, MLA_HEAD_DIM_KPE>( \
                    params, plan->plan_info.num_blks_x, plan->plan_info.num_blks_y, cuda_stream); \
            } \
            return from_cuda_error(err); \
        } while (0)

    // Dispatch based on dtype
    if (dtype_q == FLASHINFER_DTYPE_FLOAT16 && dtype_kv == FLASHINFER_DTYPE_FLOAT16) {
        DISPATCH_MLA_RUN(__half, __half);
    } else if (dtype_q == FLASHINFER_DTYPE_BFLOAT16 && dtype_kv == FLASHINFER_DTYPE_BFLOAT16) {
        DISPATCH_MLA_RUN(__nv_bfloat16, __nv_bfloat16);
    } else if (dtype_q == FLASHINFER_DTYPE_FLOAT16 && dtype_kv == FLASHINFER_DTYPE_BFLOAT16) {
        DISPATCH_MLA_RUN(__half, __nv_bfloat16);
    } else if (dtype_q == FLASHINFER_DTYPE_BFLOAT16 && dtype_kv == FLASHINFER_DTYPE_FLOAT16) {
        DISPATCH_MLA_RUN(__nv_bfloat16, __half);
    } else {
        set_error("mla_run only supports float16 and bfloat16");
        return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_MLA_RUN
}

FlashInferStatus flashinfer_mla_plan_destroy(FlashInferMLAPlanHandle plan_handle) {
    if (plan_handle) {
        MLAPlan* plan = reinterpret_cast<MLAPlan*>(plan_handle);
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

} // extern "C"

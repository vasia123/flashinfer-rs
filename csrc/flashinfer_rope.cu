/*
 * FlashInfer RoPE (Rotary Position Embedding) Operations
 *
 * Contains apply_rope, apply_rope_inplace, and cosine/sine cache operations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/pos_enc.cuh>

using namespace flashinfer_rs;

extern "C" {

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
        set_error("Null pointer passed to apply_rope");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // Extract config
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;
    float rope_scale = config->scale;
    float rope_theta = config->theta;

    // Contiguous layout strides: [total_tokens, num_heads, head_dim]
    size_t q_stride_n = static_cast<size_t>(num_qo_heads) * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t k_stride_h = head_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::BatchQKApplyRotary<__half, int32_t>(
                const_cast<__half*>(static_cast<const __half*>(q)),
                const_cast<__half*>(static_cast<const __half*>(k)),
                static_cast<__half*>(q_out),
                static_cast<__half*>(k_out),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,   // q_rope_stride_n (same as input for contiguous)
                q_stride_h,   // q_rope_stride_h
                k_stride_n,   // k_rope_stride_n
                k_stride_h,   // k_rope_stride_h
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::BatchQKApplyRotary<__nv_bfloat16, int32_t>(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k)),
                static_cast<__nv_bfloat16*>(q_out),
                static_cast<__nv_bfloat16*>(k_out),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for apply_rope");
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
    clear_error();

    if (!q || !k || !indptr || !offsets || !config) {
        set_error("Null pointer passed to apply_rope_inplace");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // Extract config
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;
    float rope_scale = config->scale;
    float rope_theta = config->theta;

    // Contiguous layout strides
    size_t q_stride_n = static_cast<size_t>(num_qo_heads) * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t k_stride_h = head_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::BatchQKApplyRotaryInPlace<__half, int32_t>(
                static_cast<__half*>(q),
                static_cast<__half*>(k),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::BatchQKApplyRotaryInPlace<__nv_bfloat16, int32_t>(
                static_cast<__nv_bfloat16*>(q),
                static_cast<__nv_bfloat16*>(k),
                const_cast<int32_t*>(indptr),
                const_cast<int32_t*>(offsets),
                batch_size,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for apply_rope_inplace");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
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
        set_error("Null pointer passed to apply_rope_pos_ids");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // Extract config
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;
    float rope_scale = config->scale;
    float rope_theta = config->theta;

    // Contiguous layout strides
    size_t q_stride_n = static_cast<size_t>(num_qo_heads) * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t k_stride_h = head_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::BatchQKApplyRotaryPosIds<__half, int32_t>(
                const_cast<__half*>(static_cast<const __half*>(q)),
                const_cast<__half*>(static_cast<const __half*>(k)),
                static_cast<__half*>(q_out),
                static_cast<__half*>(k_out),
                const_cast<int32_t*>(pos_ids),
                total_tokens,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,   // q_rope_stride_n
                q_stride_h,   // q_rope_stride_h
                k_stride_n,   // k_rope_stride_n
                k_stride_h,   // k_rope_stride_h
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::BatchQKApplyRotaryPosIds<__nv_bfloat16, int32_t>(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k)),
                static_cast<__nv_bfloat16*>(q_out),
                static_cast<__nv_bfloat16*>(k_out),
                const_cast<int32_t*>(pos_ids),
                total_tokens,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                interleave,
                rope_scale,
                rope_theta,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for apply_rope_pos_ids");
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
        set_error("Null pointer passed to apply_rope_with_cos_sin_cache");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // Extract config
    uint32_t rotary_dim = config->rotary_dim;
    bool interleave = config->interleave != 0;

    // Contiguous layout strides
    size_t q_stride_n = static_cast<size_t>(num_qo_heads) * head_dim;
    size_t q_stride_h = head_dim;
    size_t k_stride_n = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t k_stride_h = head_dim;

    // FlashInfer's BatchQKApplyRotaryPosIdsCosSinCache expects a combined cos_sin_cache
    // where cos and sin are interleaved: [max_seq_len, rotary_dim] with first half cos, second half sin
    // We need to merge the separate cos_cache and sin_cache into a single buffer
    //
    // NOTE: The FlashInfer API expects cos_sin_cache layout as [max_seq_len, rotary_dim]
    // where the first rotary_dim/2 elements are cos values and the next rotary_dim/2 are sin values.
    // Since our C API provides separate cos_cache and sin_cache, we cast cos_cache as the combined buffer
    // assuming the caller has prepared the data in the expected interleaved format.
    //
    // If caller provides separate buffers, they need to be merged before calling this function.
    // For now, we assume cos_cache points to a properly formatted cos_sin_cache buffer.

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::BatchQKApplyRotaryPosIdsCosSinCache<__half, int32_t>(
                const_cast<__half*>(static_cast<const __half*>(q)),
                const_cast<__half*>(static_cast<const __half*>(k)),
                static_cast<__half*>(q_out),
                static_cast<__half*>(k_out),
                const_cast<float*>(static_cast<const float*>(cos_cache)),  // cos_sin_cache (float)
                const_cast<int32_t*>(pos_ids),
                total_tokens,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,   // q_rope_stride_n
                q_stride_h,   // q_rope_stride_h
                k_stride_n,   // k_rope_stride_n
                k_stride_h,   // k_rope_stride_h
                interleave,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::BatchQKApplyRotaryPosIdsCosSinCache<__nv_bfloat16, int32_t>(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k)),
                static_cast<__nv_bfloat16*>(q_out),
                static_cast<__nv_bfloat16*>(k_out),
                const_cast<float*>(static_cast<const float*>(cos_cache)),  // cos_sin_cache (float)
                const_cast<int32_t*>(pos_ids),
                total_tokens,
                num_qo_heads,
                num_kv_heads,
                rotary_dim,
                head_dim,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                q_stride_n,
                q_stride_h,
                k_stride_n,
                k_stride_h,
                interleave,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for apply_rope_with_cos_sin_cache");
            return FLASHINFER_UNSUPPORTED;
    }

    return from_cuda_error(err);
}

} // extern "C"

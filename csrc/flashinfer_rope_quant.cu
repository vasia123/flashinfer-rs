/*
 * FlashInfer Fused RoPE + Quantize + Append KV Cache Operations
 *
 * Requires SM89+ (Ada Lovelace, Hopper) for FP8 support.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

// FP8 types for quantization (requires CUDA 11.8+)
#include <cuda_fp8.h>

#include <flashinfer/pos_enc.cuh>
#include <flashinfer/page.cuh>

using namespace flashinfer_rs;

extern "C" {

FlashInferStatus flashinfer_rope_quant_append_paged_kv_cache(
    const void* q_rope_in,
    const void* k_rope_in,
    const void* q_nope_in,
    const void* k_nope_in,
    const void* v_in,
    void* q_rope_out,
    void* q_nope_out,
    void* k_cache,
    void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* batch_indices,
    const int32_t* positions,
    const float* cos_sin_cache,
    const int32_t* pos_ids,
    uint32_t nnz,
    const FlashInferRopeQuantAppendConfig* config,
    void* stream
) {
    clear_error();

    // Validate inputs
    if (!q_rope_in || !k_rope_in || !v_in || !q_rope_out || !k_cache || !v_cache ||
        !kv_indptr || !kv_indices || !kv_last_page_len || !batch_indices || !positions ||
        !cos_sin_cache || !pos_ids || !config) {
        set_error("Null pointer passed to rope_quant_append_paged_kv_cache");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // For MLA with no_rope_dim > 0, nope tensors are required
    if (config->no_rope_dim > 0 && (!q_nope_in || !q_nope_out || !k_nope_in)) {
        set_error("MLA mode (no_rope_dim > 0) requires q_nope_in, q_nope_out, and k_nope_in");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Runtime SM capability check - FP8 requires SM89+
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    int major, minor;
    err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }
    err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    int sm_version = major * 10 + minor;
    if (sm_version < 89) {
        set_error("RoPE + Quantize + Append requires SM89+ (Ada Lovelace or newer). "
                  "Current device: SM" + std::to_string(sm_version));
        return FLASHINFER_UNSUPPORTED;
    }

    // Validate output dtype is FP8
    if (config->output_dtype != FLASHINFER_DTYPE_FLOAT8_E4M3 &&
        config->output_dtype != FLASHINFER_DTYPE_FLOAT8_E5M2) {
        set_error("rope_quant_append output must be FP8 (E4M3 or E5M2)");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Validate input dtype
    if (config->input_dtype != FLASHINFER_DTYPE_FLOAT16 &&
        config->input_dtype != FLASHINFER_DTYPE_BFLOAT16) {
        set_error("rope_quant_append input must be FP16 or BF16");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Extract config
    uint32_t num_qo_heads = config->num_qo_heads;
    uint32_t num_kv_heads = config->num_kv_heads;
    uint32_t rope_dim = config->rope_dim;
    uint32_t no_rope_dim = config->no_rope_dim;
    uint32_t page_size = config->page_size;
    float quant_scale_q = config->quant_scale_q;
    float quant_scale_kv = config->quant_scale_kv;
    bool interleave = config->interleave != 0;
    bool enable_pdl = config->enable_pdl != 0;
    flashinfer::QKVLayout kv_layout = to_qkv_layout(config->kv_layout);

    // Compute strides for contiguous layouts
    // Input Q_rope: [nnz, num_qo_heads, rope_dim]
    size_t q_rope_in_stride_n = static_cast<size_t>(num_qo_heads) * rope_dim;
    size_t q_rope_in_stride_h = rope_dim;

    // Input Q_nope: [nnz, num_qo_heads, no_rope_dim]
    size_t q_nope_in_stride_n = static_cast<size_t>(num_qo_heads) * no_rope_dim;
    size_t q_nope_in_stride_h = no_rope_dim;

    // Output Q_rope: [nnz, num_qo_heads, rope_dim]
    size_t q_rope_out_stride_n = q_rope_in_stride_n;
    size_t q_rope_out_stride_h = q_rope_in_stride_h;

    // Output Q_nope: [nnz, num_qo_heads, no_rope_dim]
    size_t q_nope_out_stride_n = q_nope_in_stride_n;
    size_t q_nope_out_stride_h = q_nope_in_stride_h;

    // Input K_rope: [nnz, num_kv_heads, rope_dim]
    size_t k_rope_in_stride_n = static_cast<size_t>(num_kv_heads) * rope_dim;
    size_t k_rope_in_stride_h = rope_dim;

    // Input K_nope: [nnz, num_kv_heads, no_rope_dim]
    size_t k_nope_in_stride_n = static_cast<size_t>(num_kv_heads) * no_rope_dim;
    size_t k_nope_in_stride_h = no_rope_dim;

    // Input V: [nnz, num_kv_heads, head_dim] where head_dim = rope_dim + no_rope_dim for MLA
    // or head_dim = rope_dim for GQA/MHA
    uint32_t head_dim = rope_dim + no_rope_dim;
    size_t v_in_stride_n = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t v_in_stride_h = head_dim;

    // NOTE: FlashInfer's RopeQuantizeAppendPagedKVCache requires FP8 compilation.
    // This code will compile but only run on SM89+ hardware.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) || !defined(__CUDA_ARCH__)

    // Macro to dispatch based on input/output dtype combinations
    #define DISPATCH_ROPE_QUANT_APPEND(InputType, QuantType) \
        do { \
            /* Construct paged_kv_t for K and V caches */ \
            /* For FP8 KV cache, head_dim includes both rope and nope parts */ \
            flashinfer::paged_kv_t<QuantType, int32_t> paged_kv( \
                num_kv_heads, \
                page_size, \
                head_dim, \
                0, /* batch_size - not used directly in kernel */ \
                kv_layout, \
                static_cast<QuantType*>(k_cache), \
                static_cast<QuantType*>(v_cache), \
                const_cast<int32_t*>(kv_indices), \
                const_cast<int32_t*>(kv_indptr), \
                const_cast<int32_t*>(kv_last_page_len), \
                nullptr /* rope_pos_offset */ \
            ); \
            \
            err = flashinfer::RopeQuantizeAppendPagedKVCache<InputType, int32_t, int32_t, QuantType>( \
                const_cast<InputType*>(static_cast<const InputType*>(q_rope_in)), \
                const_cast<InputType*>(static_cast<const InputType*>(k_rope_in)), \
                const_cast<InputType*>(static_cast<const InputType*>(q_nope_in)), \
                const_cast<InputType*>(static_cast<const InputType*>(k_nope_in)), \
                const_cast<InputType*>(static_cast<const InputType*>(v_in)), \
                static_cast<QuantType*>(q_rope_out), \
                static_cast<QuantType*>(q_nope_out), \
                paged_kv, \
                const_cast<int32_t*>(batch_indices), \
                const_cast<int32_t*>(positions), \
                const_cast<float*>(cos_sin_cache), \
                const_cast<int32_t*>(pos_ids), \
                nnz, \
                num_qo_heads, \
                num_kv_heads, \
                rope_dim, \
                no_rope_dim, \
                q_rope_in_stride_n, \
                q_rope_in_stride_h, \
                q_nope_in_stride_n, \
                q_nope_in_stride_h, \
                q_rope_out_stride_n, \
                q_rope_out_stride_h, \
                q_nope_out_stride_n, \
                q_nope_out_stride_h, \
                k_rope_in_stride_n, \
                k_rope_in_stride_h, \
                k_nope_in_stride_n, \
                k_nope_in_stride_h, \
                v_in_stride_n, \
                v_in_stride_h, \
                quant_scale_q, \
                quant_scale_kv, \
                interleave, \
                enable_pdl, \
                cuda_stream \
            ); \
        } while (0)

    if (config->input_dtype == FLASHINFER_DTYPE_FLOAT16) {
        if (config->output_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_ROPE_QUANT_APPEND(__half, __nv_fp8_e4m3);
        } else {
            DISPATCH_ROPE_QUANT_APPEND(__half, __nv_fp8_e5m2);
        }
    } else if (config->input_dtype == FLASHINFER_DTYPE_BFLOAT16) {
        if (config->output_dtype == FLASHINFER_DTYPE_FLOAT8_E4M3) {
            DISPATCH_ROPE_QUANT_APPEND(__nv_bfloat16, __nv_fp8_e4m3);
        } else {
            DISPATCH_ROPE_QUANT_APPEND(__nv_bfloat16, __nv_fp8_e5m2);
        }
    } else {
        set_error("Unsupported input dtype for rope_quant_append (must be FP16 or BF16)");
        return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_ROPE_QUANT_APPEND

#else
    set_error("RoPE + Quantize + Append compiled without SM89+ support");
    return FLASHINFER_UNSUPPORTED;
#endif

    return from_cuda_error(err);
}

} // extern "C"

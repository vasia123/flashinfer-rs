/*
 * FlashInfer Normalization Operations
 *
 * Contains RMSNorm, LayerNorm, QK RMSNorm, and Gemma RMSNorm implementations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/norm.cuh>

using namespace flashinfer_rs;

extern "C" {

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
        set_error("Null pointer passed to rmsnorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // FlashInfer RMSNorm requires stride parameters
    // For contiguous [batch_size, hidden_dim] layout: stride = hidden_dim
    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::RMSNorm(
                const_cast<__half*>(static_cast<const __half*>(input)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                static_cast<__half*>(output),
                batch_size,
                hidden_dim,
                stride,  // stride_input
                stride,  // stride_output
                eps,
                false,  // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::RMSNorm(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(input)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
                static_cast<__nv_bfloat16*>(output),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for rmsnorm");
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
    set_error("FP8 quantized RMSNorm requires SM89+ (Ada Lovelace or newer)");
    return FLASHINFER_UNSUPPORTED;
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
        set_error("Null pointer passed to fused_add_rmsnorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // FlashInfer's FusedAddRMSNorm signature:
    // FusedAddRMSNorm(input, residual, weight, batch_size, d, stride_input, stride_residual, eps, enable_pdl, stream)
    // It does: residual += input, then output = RMSNorm(residual)
    // The result is written back to input (which is the normalized output)
    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::FusedAddRMSNorm(
                static_cast<__half*>(input),
                const_cast<__half*>(static_cast<const __half*>(residual)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                batch_size,
                hidden_dim,
                stride,  // stride_input
                stride,  // stride_residual
                eps,
                false,   // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::FusedAddRMSNorm(
                static_cast<__nv_bfloat16*>(input),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(residual)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for fused_add_rmsnorm");
            return FLASHINFER_UNSUPPORTED;
    }

    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    // Copy to output if different from input
    if (output != input) {
        size_t size = static_cast<size_t>(batch_size) * hidden_dim * dtype_size(dtype);
        err = cudaMemcpyAsync(output, input, size, cudaMemcpyDeviceToDevice, cuda_stream);
    }

    return from_cuda_error(err);
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
        set_error("Null pointer passed to layernorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // FlashInfer LayerNorm signature:
    // LayerNorm(input, gemma, beta, out, tokens, hidden_dim, eps, stream)
    // gemma = weight, beta = bias
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::LayerNorm(
                const_cast<__half*>(static_cast<const __half*>(input)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                const_cast<__half*>(static_cast<const __half*>(bias)),
                static_cast<__half*>(output),
                batch_size,
                hidden_dim,
                eps,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            // FlashInfer LayerNorm only supports float16 based on the implementation
            set_error("BF16 LayerNorm not supported by FlashInfer");
            return FLASHINFER_UNSUPPORTED;
        default:
            set_error("Unsupported dtype for layernorm");
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
        set_error("Null pointer passed to qk_rmsnorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    // FlashInfer QKRMSNorm signature:
    // QKRMSNorm(input, weight, output, batch_size, num_heads, d,
    //           stride_input_n, stride_input_h, stride_output_n, stride_output_h, eps, enable_pdl, stream)
    // For layout [batch_size, num_heads, head_dim]:
    uint32_t stride_n = num_heads * head_dim;  // stride between batches
    uint32_t stride_h = head_dim;               // stride between heads

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::QKRMSNorm(
                const_cast<__half*>(static_cast<const __half*>(input)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                static_cast<__half*>(output),
                batch_size,
                num_heads,
                head_dim,
                stride_n,  // stride_input_n
                stride_h,  // stride_input_h
                stride_n,  // stride_output_n
                stride_h,  // stride_output_h
                eps,
                false,     // enable_pdl
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::QKRMSNorm(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(input)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
                static_cast<__nv_bfloat16*>(output),
                batch_size,
                num_heads,
                head_dim,
                stride_n,
                stride_h,
                stride_n,
                stride_h,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for qk_rmsnorm");
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
        set_error("Null pointer passed to gemma_rmsnorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::GemmaRMSNorm(
                const_cast<__half*>(static_cast<const __half*>(input)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                static_cast<__half*>(output),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::GemmaRMSNorm(
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(input)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
                static_cast<__nv_bfloat16*>(output),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for gemma_rmsnorm");
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
        set_error("Null pointer passed to gemma_fused_add_rmsnorm");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t err;

    uint32_t stride = hidden_dim;

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            err = flashinfer::norm::GemmaFusedAddRMSNorm(
                static_cast<__half*>(input),
                const_cast<__half*>(static_cast<const __half*>(residual)),
                const_cast<__half*>(static_cast<const __half*>(weight)),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            err = flashinfer::norm::GemmaFusedAddRMSNorm(
                static_cast<__nv_bfloat16*>(input),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(residual)),
                const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
                batch_size,
                hidden_dim,
                stride,
                stride,
                eps,
                false,
                cuda_stream
            );
            break;
        default:
            set_error("Unsupported dtype for gemma_fused_add_rmsnorm");
            return FLASHINFER_UNSUPPORTED;
    }

    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    // Copy to output if different from input
    if (output != input) {
        size_t size = static_cast<size_t>(batch_size) * hidden_dim * dtype_size(dtype);
        err = cudaMemcpyAsync(output, input, size, cudaMemcpyDeviceToDevice, cuda_stream);
    }

    return from_cuda_error(err);
}

} // extern "C"

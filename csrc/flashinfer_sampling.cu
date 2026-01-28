/*
 * FlashInfer Sampling Operations
 *
 * Contains top_k, top_p, min_p, softmax, and renormalization implementations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/sampling.cuh>

using namespace flashinfer_rs;

extern "C" {

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
        set_error("Null pointer passed to top_k_sampling");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // FlashInfer sampling only supports float32 probabilities
    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("top_k_sampling only supports float32 probabilities");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // FlashInfer TopKSamplingFromProb signature:
    // TopKSamplingFromProb(probs, output, indices, top_k_arr, batch_size, top_k_val, d,
    //                      deterministic, philox_seed, philox_offset, stream)
    // NOTE: indices can be nullptr if we don't need the sorted indices
    cudaError_t err = flashinfer::sampling::TopKSamplingFromProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        output,                    // output
        static_cast<int32_t*>(nullptr),  // indices (not needed)
        const_cast<float*>(reinterpret_cast<const float*>(top_k_arr)),  // top_k_arr (per-batch)
        batch_size,
        top_k_val,
        vocab_size,
        deterministic != 0,
        philox_seed,
        philox_offset,
        cuda_stream
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
        set_error("Null pointer passed to top_p_sampling");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("top_p_sampling only supports float32 probabilities");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err = flashinfer::sampling::TopPSamplingFromProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        output,
        static_cast<int32_t*>(nullptr),  // indices
        const_cast<float*>(top_p_arr),   // top_p_arr
        batch_size,
        top_p_val,
        vocab_size,
        deterministic != 0,
        philox_seed,
        philox_offset,
        cuda_stream
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
        set_error("Null pointer passed to min_p_sampling");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("min_p_sampling only supports float32 probabilities");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err = flashinfer::sampling::MinPSamplingFromProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        const_cast<float*>(min_p_arr),  // min_p_arr
        output,
        static_cast<int32_t*>(nullptr),  // indices
        batch_size,
        min_p_val,
        vocab_size,
        deterministic != 0,
        philox_seed,
        philox_offset,
        cuda_stream
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
        set_error("Null pointer passed to top_k_top_p_sampling");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("top_k_top_p_sampling only supports float32 probabilities");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err = flashinfer::sampling::TopKTopPSamplingFromProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        const_cast<int32_t*>(top_k_arr),  // top_k_arr
        const_cast<float*>(top_p_arr),    // top_p_arr
        output,
        static_cast<int32_t*>(nullptr),   // indices
        batch_size,
        static_cast<int32_t>(top_k_val),
        top_p_val,
        vocab_size,
        deterministic != 0,
        philox_seed,
        philox_offset,
        cuda_stream
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
        set_error("Null pointer passed to softmax");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // FlashInfer OnlineSoftmax only supports float32
    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("softmax only supports float32");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // NOTE: FlashInfer's OnlineSoftmax has an optimized multi-pass path for
    // large vocabularies (vocab_size >= 24576) with small batches (batch_size <= 128)
    // that requires a workspace buffer. Since our C API doesn't expose workspace
    // for softmax, we pass nullptr which forces the single-block path.
    // This is still efficient for most use cases, but the caller should allocate
    // workspace for optimal performance with large vocabularies.
    //
    // Required workspace size for large vocab path:
    //   batch_size * ceil_div(vocab_size, 8192) * sizeof(PartialSoftmaxResult)
    //   where PartialSoftmaxResult is 8 bytes (2 floats: max_val, denominator)
    cudaError_t err = flashinfer::sampling::OnlineSoftmax<float>(
        const_cast<float*>(static_cast<const float*>(logits)),
        static_cast<float*>(probs),
        batch_size,
        vocab_size,
        const_cast<float*>(temperature_arr),
        temp_val,
        nullptr,  // workspace_buffer - nullptr triggers single-block path
        0,        // workspace_buffer_size_in_bytes
        false,    // enable_pdl - disabled since we don't have workspace for PDL either
        cuda_stream
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
    set_error("top_k_mask_logits not provided by FlashInfer");
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
        set_error("Null pointer passed to top_p_renorm_probs");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("top_p_renorm_probs only supports float32");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err = flashinfer::sampling::TopPRenormProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        static_cast<float*>(renormed_probs),
        const_cast<float*>(top_p_arr),
        batch_size,
        top_p_val,
        vocab_size,
        cuda_stream
    );

    return from_cuda_error(err);
}

} // extern "C"

/*
 * FlashInfer Sampling Operations
 *
 * Contains top_k, top_p, min_p, softmax, and renormalization implementations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/air_top_p.cuh>
#include <flashinfer/sampling.cuh>

using namespace flashinfer_rs;

namespace {

// Shared workspace-layout math for AirTopP renorm (kept in sync with
// AirTopPRenormProb in include/flashinfer/air_top_p.cuh).
inline size_t air_top_p_align256(size_t x) {
    return (x + 255) / 256 * 256;
}

// Workspace for IsDeterministic = true, DType = float (our only supported config).
// Layout: counters | histograms (uint64) | count_histograms (int) | buf1 (float) | buf2 (float)
inline size_t air_top_p_workspace_size_f32(uint32_t batch_size, uint32_t vocab_size) {
    using namespace flashinfer::sampling::air_top_p;
    constexpr size_t kCounterSize = 128;  // alignas(128) Counter<float>
    const size_t buf_len = static_cast<size_t>(calcBufLen<float>(static_cast<IdxT>(vocab_size)));

    const size_t counters_sz = air_top_p_align256(kCounterSize * batch_size);
    const size_t hist_sz     = air_top_p_align256(sizeof(uint64_t) * NUM_BUCKETS * batch_size);
    const size_t counthist_sz = air_top_p_align256(sizeof(IdxT) * NUM_BUCKETS * batch_size);
    const size_t buf_sz      = air_top_p_align256(sizeof(float) * buf_len * batch_size);
    return counters_sz + hist_sz + counthist_sz + 2 * buf_sz;
}

}  // namespace

extern "C" {

FlashInferStatus flashinfer_top_k_sampling(
    const void* probs,
    int32_t* output,
    bool* valid_out,
    const int32_t* top_k_arr,
    uint32_t top_k_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    const uint64_t* philox_seed_arr,
    uint64_t philox_seed,
    const uint64_t* philox_offset_arr,
    uint64_t philox_offset,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !output) {
        set_error("Null pointer passed to top_k_sampling");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("top_k_sampling only supports float32 probabilities");
        return FLASHINFER_UNSUPPORTED;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err = flashinfer::sampling::TopKSamplingFromProb(
        const_cast<float*>(static_cast<const float*>(probs)),
        output,
        valid_out,
        static_cast<int32_t*>(nullptr),  // sorted-indices output (not exposed)
        const_cast<float*>(reinterpret_cast<const float*>(top_k_arr)),
        batch_size,
        top_k_val,
        vocab_size,
        deterministic != 0,
        const_cast<uint64_t*>(philox_seed_arr),   philox_seed,
        const_cast<uint64_t*>(philox_offset_arr), philox_offset,
        cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_top_p_sampling(
    const void* probs,
    int32_t* output,
    bool* valid_out,
    const float* top_p_arr,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    const uint64_t* philox_seed_arr,
    uint64_t philox_seed,
    const uint64_t* philox_offset_arr,
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
        valid_out,
        static_cast<int32_t*>(nullptr),
        const_cast<float*>(top_p_arr),
        batch_size,
        top_p_val,
        vocab_size,
        deterministic != 0,
        const_cast<uint64_t*>(philox_seed_arr),   philox_seed,
        const_cast<uint64_t*>(philox_offset_arr), philox_offset,
        cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_min_p_sampling(
    const void* probs,
    int32_t* output,
    bool* valid_out,
    const float* min_p_arr,
    float min_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    const uint64_t* philox_seed_arr,
    uint64_t philox_seed,
    const uint64_t* philox_offset_arr,
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
        const_cast<float*>(min_p_arr),
        output,
        valid_out,
        static_cast<int32_t*>(nullptr),
        batch_size,
        min_p_val,
        vocab_size,
        deterministic != 0,
        const_cast<uint64_t*>(philox_seed_arr),   philox_seed,
        const_cast<uint64_t*>(philox_offset_arr), philox_offset,
        cuda_stream
    );

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_top_k_top_p_sampling(
    const void* probs,
    int32_t* output,
    bool* valid_out,
    const int32_t* top_k_arr,
    const float* top_p_arr,
    uint32_t top_k_val,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    const uint64_t* philox_seed_arr,
    uint64_t philox_seed,
    const uint64_t* philox_offset_arr,
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
        const_cast<int32_t*>(top_k_arr),
        const_cast<float*>(top_p_arr),
        output,
        valid_out,
        static_cast<int32_t*>(nullptr),
        batch_size,
        static_cast<int32_t>(top_k_val),
        top_p_val,
        vocab_size,
        deterministic != 0,
        const_cast<uint64_t*>(philox_seed_arr),   philox_seed,
        const_cast<uint64_t*>(philox_offset_arr), philox_offset,
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

FlashInferStatus flashinfer_air_top_p_renorm_probs_workspace_size(
    uint32_t batch_size,
    uint32_t vocab_size,
    size_t* workspace_size_out
) {
    clear_error();
    if (!workspace_size_out) {
        set_error("Null pointer passed to air_top_p_renorm_probs_workspace_size");
        return FLASHINFER_INVALID_ARGUMENT;
    }
    *workspace_size_out = air_top_p_workspace_size_f32(batch_size, vocab_size);
    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_air_top_p_renorm_probs(
    const void* probs,
    void* renormed_probs,
    const float* top_p_arr,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    int deterministic,
    void* workspace,
    size_t workspace_size,
    FlashInferDType dtype,
    void* stream
) {
    clear_error();

    if (!probs || !renormed_probs || !workspace) {
        set_error("Null pointer passed to air_top_p_renorm_probs");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    if (dtype != FLASHINFER_DTYPE_FLOAT32) {
        set_error("air_top_p_renorm_probs only supports float32");
        return FLASHINFER_UNSUPPORTED;
    }

    const size_t required = air_top_p_workspace_size_f32(batch_size, vocab_size);
    if (workspace_size < required) {
        set_error("air_top_p_renorm_probs workspace too small: need " +
                  std::to_string(required) + " bytes, got " + std::to_string(workspace_size));
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    cudaError_t err;
    if (deterministic) {
        err = flashinfer::sampling::air_top_p::AirTopPRenormProb<true, float>(
            const_cast<float*>(static_cast<const float*>(probs)),
            static_cast<float*>(renormed_probs),
            const_cast<float*>(top_p_arr),
            batch_size,
            top_p_val,
            vocab_size,
            workspace,
            cuda_stream
        );
    } else {
        err = flashinfer::sampling::air_top_p::AirTopPRenormProb<false, float>(
            const_cast<float*>(static_cast<const float*>(probs)),
            static_cast<float*>(renormed_probs),
            const_cast<float*>(top_p_arr),
            batch_size,
            top_p_val,
            vocab_size,
            workspace,
            cuda_stream
        );
    }

    return from_cuda_error(err);
}

} // extern "C"

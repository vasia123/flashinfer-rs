/*
 * FlashInfer Utility Functions
 *
 * Contains GPU info, error handling, workspace queries, etc.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

using namespace flashinfer_rs;

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
    return "0.1.0-flashinfer-rs";
}

FlashInferStatus flashinfer_check_gpu_support(int* supported, int* sm_version) {
    clear_error();

    if (!supported || !sm_version) {
        set_error("supported and sm_version must not be null");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    *sm_version = prop.major * 10 + prop.minor;
    // FlashInfer requires SM75+ (Turing) for basic support, SM80+ for full features
    *supported = (*sm_version >= 75) ? 1 : 0;

    return FLASHINFER_SUCCESS;
}

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

    if (!float_workspace_size || !int_workspace_size) {
        set_error("float_workspace_size and int_workspace_size must not be null");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // General workspace estimation for both decode and prefill
    // Float workspace: for split KV accumulation
    size_t split_kv_workspace = static_cast<size_t>(batch_size) * num_heads * head_dim * sizeof(float) * 16;

    // Int workspace: for indices and offsets
    size_t split_idx_workspace = static_cast<size_t>(batch_size) * sizeof(int32_t) * 8;

    // Align to 256 bytes
    *float_workspace_size = ((split_kv_workspace + 255) / 256) * 256;
    *int_workspace_size = ((split_idx_workspace + 255) / 256) * 256;

    return FLASHINFER_SUCCESS;
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

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return from_cuda_error(err);
    }

    if (compute_major) {
        *compute_major = prop.major;
    }
    if (compute_minor) {
        *compute_minor = prop.minor;
    }
    if (sm_count) {
        *sm_count = prop.multiProcessorCount;
    }

    int sm_version = prop.major * 10 + prop.minor;

    if (supports_pdl) {
        // PDL (Persistent Data Loader) requires SM90+ (Hopper)
        *supports_pdl = (sm_version >= 90) ? 1 : 0;
    }
    if (supports_fp8) {
        // FP8 requires SM89+ (Ada Lovelace)
        *supports_fp8 = (sm_version >= 89) ? 1 : 0;
    }

    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_set_stream(void* stream) {
    // This is a placeholder - individual operations take stream as parameter
    // Some implementations might want a global default stream
    clear_error();
    return FLASHINFER_SUCCESS;
}

} // extern "C"

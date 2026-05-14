/*
 * FlashInfer Common Definitions
 *
 * Shared structures, helpers, and utilities for modular CUDA compilation.
 * This header is included by all flashinfer_*.cu modules.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#ifndef FLASHINFER_COMMON_H_
#define FLASHINFER_COMMON_H_

#include "flashinfer_c_api.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstring>
#include <string>
#include <cmath>
#include <vector>

// FlashInfer headers
#include <flashinfer/attention/scheduler.cuh>

// Enable FP16 and BF16 support
#define FLASHINFER_ENABLE_F16
#define FLASHINFER_ENABLE_BF16

namespace flashinfer_rs {

// Thread-local error message storage
inline thread_local std::string g_last_error;

inline void set_error(const std::string& msg) {
    g_last_error = msg;
}

inline void clear_error() {
    g_last_error.clear();
}

// Convert CUDA error to FlashInfer status
inline FlashInferStatus from_cuda_error(cudaError_t err) {
    if (err == cudaSuccess) {
        return FLASHINFER_SUCCESS;
    }
    set_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    return FLASHINFER_CUDA_ERROR;
}

// Data type size in bytes
inline size_t dtype_size(FlashInferDType dtype) {
    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
        case FLASHINFER_DTYPE_BFLOAT16:
            return 2;
        case FLASHINFER_DTYPE_FLOAT32:
            return 4;
        case FLASHINFER_DTYPE_FLOAT8_E4M3:
        case FLASHINFER_DTYPE_FLOAT8_E5M2:
            return 1;
        default:
            return 0;
    }
}

// Plan info storage for batch decode
struct BatchDecodePlan {
    flashinfer::DecodePlanInfo plan_info;
    int32_t batch_size;
    int32_t num_qo_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t page_size;
    FlashInferDType dtype;
    FlashInferPosEncoding pos_encoding;
    float logits_soft_cap;
    void* float_workspace;
    void* int_workspace;
    size_t float_workspace_size;
    size_t int_workspace_size;
    bool enable_cuda_graph;
    float sm_scale;
    int32_t window_left;
    float* alibi_slopes;      // GPU-allocated ALiBi slopes [num_qo_heads], nullptr if not using ALiBi
    bool owns_alibi_slopes;   // Whether we should free alibi_slopes on destroy
};

// Plan info storage for batch prefill
struct BatchPrefillPlan {
    flashinfer::PrefillPlanInfo plan_info;
    int32_t batch_size;
    int32_t num_qo_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t page_size;
    FlashInferDType dtype;
    FlashInferPosEncoding pos_encoding;
    float logits_soft_cap;
    int causal;
    void* float_workspace;
    void* int_workspace;
    size_t float_workspace_size;
    size_t int_workspace_size;
    bool enable_cuda_graph;
    float sm_scale;
    int32_t window_left;
    uint32_t total_num_rows;  // Sum of all query lengths
    float* alibi_slopes;      // GPU-allocated ALiBi slopes [num_qo_heads], nullptr if not using ALiBi
    bool owns_alibi_slopes;   // Whether we should free alibi_slopes on destroy
};

// Compute ALiBi slopes for a given number of heads
// Formula: slope[i] = 2^(-8 * (i + 1) / n) where n = num_heads
// This follows the original ALiBi paper: https://arxiv.org/abs/2108.12409
inline cudaError_t compute_alibi_slopes(float** d_slopes, uint32_t num_heads, cudaStream_t stream) {
    // Allocate host memory for slopes
    std::vector<float> h_slopes(num_heads);

    // Compute slopes: ratio = 2^(-8/n), slope[i] = ratio^(i+1)
    float ratio = std::pow(2.0f, -8.0f / static_cast<float>(num_heads));
    for (uint32_t i = 0; i < num_heads; ++i) {
        h_slopes[i] = std::pow(ratio, static_cast<float>(i + 1));
    }

    // Allocate device memory
    float* slopes = nullptr;
    cudaError_t err = cudaMalloc(&slopes, num_heads * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }

    // Copy to device
    err = cudaMemcpyAsync(slopes, h_slopes.data(), num_heads * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(slopes);
        return err;
    }

    *d_slopes = slopes;
    return cudaSuccess;
}

// Convert our enums to FlashInfer enums
inline flashinfer::QKVLayout to_qkv_layout(FlashInferKVLayout layout) {
    return layout == FLASHINFER_KV_LAYOUT_HND
        ? flashinfer::QKVLayout::kHND
        : flashinfer::QKVLayout::kNHD;
}

inline flashinfer::PosEncodingMode to_pos_encoding(FlashInferPosEncoding enc) {
    switch (enc) {
        case FLASHINFER_POS_ENCODING_ROPE_LLAMA:
            return flashinfer::PosEncodingMode::kRoPELlama;
        case FLASHINFER_POS_ENCODING_ALIBI:
            return flashinfer::PosEncodingMode::kALiBi;
        default:
            return flashinfer::PosEncodingMode::kNone;
    }
}

// Dispatch macro for dtype and head_dim combinations
#define DISPATCH_DTYPE_HEAD_DIM(dtype, head_dim, DTYPE_T, HEAD_DIM_V, ...)   \
    do {                                                                      \
        if (dtype == FLASHINFER_DTYPE_FLOAT16) {                             \
            using DTYPE_T = half;                                            \
            if (head_dim == 64) {                                            \
                constexpr int HEAD_DIM_V = 64;                               \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 128) {                                    \
                constexpr int HEAD_DIM_V = 128;                              \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 256) {                                    \
                constexpr int HEAD_DIM_V = 256;                              \
                { __VA_ARGS__ }                                              \
            } else {                                                         \
                set_error("Unsupported head_dim for float16");               \
                return FLASHINFER_UNSUPPORTED;                               \
            }                                                                \
        } else if (dtype == FLASHINFER_DTYPE_BFLOAT16) {                     \
            using DTYPE_T = nv_bfloat16;                                     \
            if (head_dim == 64) {                                            \
                constexpr int HEAD_DIM_V = 64;                               \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 128) {                                    \
                constexpr int HEAD_DIM_V = 128;                              \
                { __VA_ARGS__ }                                              \
            } else if (head_dim == 256) {                                    \
                constexpr int HEAD_DIM_V = 256;                              \
                { __VA_ARGS__ }                                              \
            } else {                                                         \
                set_error("Unsupported head_dim for bfloat16");              \
                return FLASHINFER_UNSUPPORTED;                               \
            }                                                                \
        } else {                                                             \
            set_error("Unsupported dtype");                                  \
            return FLASHINFER_UNSUPPORTED;                                   \
        }                                                                    \
    } while (0)

}  // namespace flashinfer_rs

#endif  // FLASHINFER_COMMON_H_

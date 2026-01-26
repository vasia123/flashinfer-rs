/*
 * FlashInfer C API Implementation
 *
 * This file provides C API wrappers over FlashInfer's C++ template kernels.
 * It includes explicit template instantiations for supported dtype/head_dim combinations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_c_api.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstring>
#include <string>
#include <mutex>

// FlashInfer headers
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

namespace {

// Thread-local error message storage
thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_error() {
    g_last_error.clear();
}

// Convert FlashInfer status to our status codes
FlashInferStatus from_cuda_error(cudaError_t err) {
    if (err == cudaSuccess) {
        return FLASHINFER_SUCCESS;
    }
    set_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    return FLASHINFER_CUDA_ERROR;
}

// Data type size in bytes
size_t dtype_size(FlashInferDType dtype) {
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

// Plan info storage
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
};

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
};

// Convert our enums to FlashInfer enums
flashinfer::QKVLayout to_qkv_layout(FlashInferKVLayout layout) {
    return layout == FLASHINFER_KV_LAYOUT_HND
        ? flashinfer::QKVLayout::kHND
        : flashinfer::QKVLayout::kNHD;
}

flashinfer::PosEncodingMode to_pos_encoding(FlashInferPosEncoding enc) {
    switch (enc) {
        case FLASHINFER_POS_ENCODING_ROPE_LLAMA:
            return flashinfer::PosEncodingMode::kRoPELlama;
        case FLASHINFER_POS_ENCODING_ALIBI:
            return flashinfer::PosEncodingMode::kALiBi;
        default:
            return flashinfer::PosEncodingMode::kNone;
    }
}

// Template dispatch helpers
template <typename DType, int HEAD_DIM>
struct KernelDispatcher {
    static cudaError_t batch_decode(
        const BatchDecodePlan* plan,
        const void* q,
        const void* k_cache,
        const void* v_cache,
        const int32_t* kv_indptr,
        const int32_t* kv_indices,
        const int32_t* kv_last_page_len,
        void* output,
        float* lse,
        FlashInferKVLayout kv_layout,
        cudaStream_t stream
    );

    static cudaError_t batch_prefill(
        const BatchPrefillPlan* plan,
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
        cudaStream_t stream
    );
};

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

}  // namespace

/* ============================================================================
 * C API Implementation
 * ============================================================================ */

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
    return "0.1.0-rs";  // Version of our Rust bindings
}

FlashInferStatus flashinfer_check_gpu_support(int* supported, int* sm_version) {
    clear_error();

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        *supported = 0;
        *sm_version = 0;
        return from_cuda_error(err);
    }

    *sm_version = props.major * 10 + props.minor;

    // FlashInfer requires SM75+ (Turing or newer)
    *supported = (props.major >= 7 && props.minor >= 5) || props.major >= 8;

    return FLASHINFER_SUCCESS;
}

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

    // Estimate workspace sizes based on FlashInfer requirements
    // These are conservative estimates

    int32_t max_num_pages = (max_seq_len + page_size - 1) / page_size;
    int32_t gqa_ratio = num_qo_heads / num_kv_heads;

    // Float workspace: tmp_v for split-k, tmp_s for softmax
    size_t tmp_v_size = batch_size * num_qo_heads * head_dim * sizeof(float);
    size_t tmp_s_size = batch_size * num_qo_heads * sizeof(float);
    *float_workspace_size = tmp_v_size + tmp_s_size + 4096;  // padding

    // Int workspace: request_indices, kv_tile_indices, o_indptr, etc.
    size_t partition_info_size = batch_size * num_kv_heads * 2 * sizeof(int32_t);
    size_t page_indices_size = batch_size * max_num_pages * sizeof(int32_t);
    *int_workspace_size = partition_info_size + page_indices_size + 4096;  // padding

    return FLASHINFER_SUCCESS;
}

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

    if (batch_size <= 0 || total_tokens <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0) {
        set_error("Invalid argument: all sizes must be positive");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // Prefill needs larger workspace due to larger attention matrices
    size_t tmp_size = total_tokens * num_qo_heads * head_dim * sizeof(float);
    *float_workspace_size = tmp_size + 8192;  // padding

    size_t request_info_size = batch_size * 4 * sizeof(int32_t);
    size_t tile_info_size = total_tokens * sizeof(int32_t);
    *int_workspace_size = request_info_size + tile_info_size + 4096;  // padding

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
    int enable_cuda_graph,
    void* stream
) {
    clear_error();

    if (!plan_handle || !float_workspace || !int_workspace || !kv_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

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

    // TODO: Call FlashInfer's DecodePlan function
    // This requires explicit template instantiation which will be added
    // when we verify the build system works

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

    // TODO: Dispatch to correct template instantiation based on dtype and head_dim
    // For now, return unsupported until we verify the build works
    set_error("Batch decode kernel not yet instantiated - build system in progress");
    return FLASHINFER_UNSUPPORTED;
}

FlashInferStatus flashinfer_batch_decode_plan_destroy(
    FlashInferBatchDecodePlanHandle plan_handle
) {
    if (plan_handle) {
        BatchDecodePlan* plan = reinterpret_cast<BatchDecodePlan*>(plan_handle);
        delete plan;
    }
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
    int causal,
    int enable_cuda_graph,
    void* stream
) {
    clear_error();

    if (!plan_handle || !float_workspace || !int_workspace ||
        !qo_indptr || !kv_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    BatchPrefillPlan* plan = new BatchPrefillPlan();
    plan->batch_size = batch_size;
    plan->num_qo_heads = num_qo_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype = dtype;
    plan->pos_encoding = pos_encoding;
    plan->logits_soft_cap = logits_soft_cap;
    plan->causal = causal;
    plan->float_workspace = float_workspace;
    plan->int_workspace = int_workspace;
    plan->float_workspace_size = float_workspace_size;
    plan->int_workspace_size = int_workspace_size;

    // TODO: Call FlashInfer's PrefillPlan function

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

    if (!plan_handle || !q || !k_cache || !v_cache || !output ||
        !kv_indptr || !kv_indices || !kv_last_page_len || !qo_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    BatchPrefillPlan* plan = reinterpret_cast<BatchPrefillPlan*>(plan_handle);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // TODO: Dispatch to correct template instantiation
    set_error("Batch prefill kernel not yet instantiated - build system in progress");
    return FLASHINFER_UNSUPPORTED;
}

FlashInferStatus flashinfer_batch_prefill_plan_destroy(
    FlashInferBatchPrefillPlanHandle plan_handle
) {
    if (plan_handle) {
        BatchPrefillPlan* plan = reinterpret_cast<BatchPrefillPlan*>(plan_handle);
        delete plan;
    }
    return FLASHINFER_SUCCESS;
}

FlashInferStatus flashinfer_append_paged_kv_cache(
    const void* k,
    const void* v,
    void* k_cache,
    void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    const int32_t* append_indptr,
    int32_t batch_size,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    FlashInferDType dtype,
    FlashInferKVLayout kv_layout,
    void* stream
) {
    clear_error();

    if (!k || !v || !k_cache || !v_cache ||
        !kv_indptr || !kv_indices || !kv_last_page_len || !append_indptr) {
        set_error("Invalid argument: null pointer");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    // TODO: Implement append kernel dispatch
    set_error("Append paged KV cache not yet implemented");
    return FLASHINFER_UNSUPPORTED;
}

}  // extern "C"

/* ============================================================================
 * Template instantiations
 * ============================================================================
 *
 * These explicit instantiations ensure the kernels are compiled for
 * the supported dtype/head_dim combinations:
 *
 * - half (fp16) × {64, 128, 256}
 * - nv_bfloat16 (bf16) × {64, 128, 256}
 *
 * Adding more combinations increases compile time and binary size.
 * ============================================================================ */

// TODO: Add explicit template instantiations once build system is verified
//
// The instantiations will look like:
//
// template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
//     64, flashinfer::PosEncodingMode::kNone,
//     flashinfer::DefaultAttentionVariant,
//     flashinfer::BatchDecodeParams<half, half, half, int32_t, 64, 64>
// >(flashinfer::BatchDecodeParams<...>, half*, float*, bool, cudaStream_t);
//
// For now, we provide the framework and validate the build works.

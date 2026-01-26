/*
 * FlashInfer C API for Rust FFI bindings
 *
 * This header provides a thin C interface over FlashInfer's C++ template API.
 * The API follows the Plan-Run pattern used by FlashInfer:
 * 1. Plan: Compute workspace requirements and prepare metadata
 * 2. Run: Execute the actual kernel with prepared plan
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#ifndef FLASHINFER_C_API_H_
#define FLASHINFER_C_API_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Status codes
 * ============================================================================ */

typedef enum {
    FLASHINFER_SUCCESS = 0,
    FLASHINFER_CUDA_ERROR = 1,
    FLASHINFER_INVALID_ARGUMENT = 2,
    FLASHINFER_OUT_OF_MEMORY = 3,
    FLASHINFER_UNSUPPORTED = 4,
    FLASHINFER_INTERNAL_ERROR = 5,
} FlashInferStatus;

/* ============================================================================
 * Data types
 * ============================================================================ */

typedef enum {
    FLASHINFER_DTYPE_FLOAT16 = 0,
    FLASHINFER_DTYPE_BFLOAT16 = 1,
    FLASHINFER_DTYPE_FLOAT32 = 2,
    FLASHINFER_DTYPE_FLOAT8_E4M3 = 3,
    FLASHINFER_DTYPE_FLOAT8_E5M2 = 4,
} FlashInferDType;

/* KV cache layout: HND = [num_pages, num_heads, page_size, head_dim]
 *                  NHD = [num_pages, page_size, num_heads, head_dim] */
typedef enum {
    FLASHINFER_KV_LAYOUT_HND = 0,
    FLASHINFER_KV_LAYOUT_NHD = 1,
} FlashInferKVLayout;

/* Position encoding mode */
typedef enum {
    FLASHINFER_POS_ENCODING_NONE = 0,
    FLASHINFER_POS_ENCODING_ROPE_LLAMA = 1,
    FLASHINFER_POS_ENCODING_ALIBI = 2,
} FlashInferPosEncoding;

/* ============================================================================
 * Opaque handles
 * ============================================================================ */

/* Opaque handle for batch decode plan */
typedef struct FlashInferBatchDecodePlan* FlashInferBatchDecodePlanHandle;

/* Opaque handle for batch prefill plan */
typedef struct FlashInferBatchPrefillPlan* FlashInferBatchPrefillPlanHandle;

/* ============================================================================
 * Workspace size queries
 * ============================================================================ */

/**
 * Get required workspace size for batch decode operation.
 *
 * @param[out] float_workspace_size  Size in bytes for float workspace buffer
 * @param[out] int_workspace_size    Size in bytes for int workspace buffer
 * @param batch_size       Number of sequences in the batch
 * @param num_qo_heads     Number of query/output heads
 * @param num_kv_heads     Number of key/value heads (can be < num_qo_heads for GQA)
 * @param head_dim         Dimension of each head (64, 128, or 256)
 * @param page_size        Number of tokens per page
 * @param max_seq_len      Maximum sequence length
 * @return Status code
 */
FlashInferStatus flashinfer_batch_decode_workspace_size(
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    int32_t max_seq_len
);

/**
 * Get required workspace size for batch prefill operation.
 */
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
);

/* ============================================================================
 * Batch Decode API
 * ============================================================================ */

/**
 * Create a batch decode plan.
 *
 * This function prepares metadata for the decode operation based on
 * the batch configuration. Call this once per batch shape change.
 *
 * @param[out] plan_handle         Handle to the created plan
 * @param float_workspace          GPU buffer for float workspace (pre-allocated)
 * @param float_workspace_size     Size of float workspace in bytes
 * @param int_workspace            GPU buffer for int workspace (pre-allocated)
 * @param int_workspace_size       Size of int workspace in bytes
 * @param page_locked_int_workspace  Page-locked host memory for async transfers
 * @param page_locked_int_size     Size of page-locked buffer in bytes
 * @param kv_indptr                Cumulative page counts [batch_size + 1], device memory
 * @param batch_size               Number of sequences
 * @param num_qo_heads             Number of query/output heads
 * @param num_kv_heads             Number of key/value heads
 * @param head_dim                 Head dimension (64, 128, or 256)
 * @param page_size                Tokens per page
 * @param dtype                    Data type for Q/K/V/O
 * @param pos_encoding             Position encoding mode
 * @param logits_soft_cap          Soft cap for attention logits (0 to disable)
 * @param enable_cuda_graph        Whether this will run in a CUDA graph
 * @param stream                   CUDA stream for async operations
 * @return Status code
 */
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
);

/**
 * Execute batch decode with a prepared plan.
 *
 * Computes: output[i] = softmax(Q[i] @ K_cache[i].T / sqrt(head_dim)) @ V_cache[i]
 *
 * @param plan_handle              Plan from flashinfer_batch_decode_plan
 * @param q                        Query tensor [batch_size, num_qo_heads, head_dim], device
 * @param k_cache                  Key cache [num_pages, layout...], device
 * @param v_cache                  Value cache [num_pages, layout...], device
 * @param kv_indptr                Cumulative page counts [batch_size + 1], device
 * @param kv_indices               Page indices [total_pages], device
 * @param kv_last_page_len         Tokens in last page per sequence [batch_size], device
 * @param output                   Output tensor [batch_size, num_qo_heads, head_dim], device
 * @param lse                      Optional log-sum-exp [batch_size, num_qo_heads], device (NULL to skip)
 * @param kv_layout                KV cache memory layout
 * @param stream                   CUDA stream
 * @return Status code
 */
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
);

/**
 * Destroy a batch decode plan and free associated resources.
 */
FlashInferStatus flashinfer_batch_decode_plan_destroy(
    FlashInferBatchDecodePlanHandle plan_handle
);

/* ============================================================================
 * Batch Prefill API
 * ============================================================================ */

/**
 * Create a batch prefill plan.
 *
 * @param[out] plan_handle         Handle to the created plan
 * @param float_workspace          GPU buffer for float workspace
 * @param float_workspace_size     Size in bytes
 * @param int_workspace            GPU buffer for int workspace
 * @param int_workspace_size       Size in bytes
 * @param page_locked_int_workspace  Page-locked host memory
 * @param page_locked_int_size     Size in bytes
 * @param qo_indptr                Query token offsets [batch_size + 1], device
 * @param kv_indptr                KV page offsets [batch_size + 1], device
 * @param batch_size               Number of sequences
 * @param num_qo_heads             Number of query/output heads
 * @param num_kv_heads             Number of key/value heads
 * @param head_dim                 Head dimension
 * @param page_size                Tokens per page
 * @param dtype                    Data type
 * @param pos_encoding             Position encoding mode
 * @param logits_soft_cap          Soft cap (0 to disable)
 * @param causal                   Whether to apply causal masking
 * @param enable_cuda_graph        CUDA graph mode
 * @param stream                   CUDA stream
 * @return Status code
 */
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
);

/**
 * Execute batch prefill with a prepared plan.
 *
 * @param plan_handle              Plan from flashinfer_batch_prefill_plan
 * @param q                        Query tensor [total_tokens, num_qo_heads, head_dim]
 * @param k_cache                  Key cache [num_pages, layout...]
 * @param v_cache                  Value cache [num_pages, layout...]
 * @param kv_indptr                KV page offsets [batch_size + 1]
 * @param kv_indices               Page indices [total_pages]
 * @param kv_last_page_len         Tokens in last page [batch_size]
 * @param qo_indptr                Query offsets [batch_size + 1]
 * @param output                   Output tensor [total_tokens, num_qo_heads, head_dim]
 * @param lse                      Optional log-sum-exp [total_tokens, num_qo_heads] (NULL to skip)
 * @param kv_layout                KV cache memory layout
 * @param stream                   CUDA stream
 * @return Status code
 */
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
);

/**
 * Destroy a batch prefill plan.
 */
FlashInferStatus flashinfer_batch_prefill_plan_destroy(
    FlashInferBatchPrefillPlanHandle plan_handle
);

/* ============================================================================
 * Append KV Cache API
 * ============================================================================ */

/**
 * Append new key-value pairs to paged KV cache.
 *
 * This is typically called during prefill to populate the KV cache.
 *
 * @param k                New keys [total_tokens, num_kv_heads, head_dim]
 * @param v                New values [total_tokens, num_kv_heads, head_dim]
 * @param k_cache          Key cache [num_pages, layout...]
 * @param v_cache          Value cache [num_pages, layout...]
 * @param kv_indptr        Page offsets [batch_size + 1]
 * @param kv_indices       Page indices [total_pages]
 * @param kv_last_page_len Tokens in last page before append [batch_size]
 * @param append_indptr    Token offsets for append [batch_size + 1]
 * @param batch_size       Number of sequences
 * @param num_kv_heads     Number of KV heads
 * @param head_dim         Head dimension
 * @param page_size        Tokens per page
 * @param dtype            Data type
 * @param kv_layout        KV cache layout
 * @param stream           CUDA stream
 * @return Status code
 */
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
);

/* ============================================================================
 * Utility functions
 * ============================================================================ */

/**
 * Get the last error message (thread-local).
 * Returns NULL if no error occurred.
 */
const char* flashinfer_get_last_error(void);

/**
 * Clear the last error message.
 */
void flashinfer_clear_last_error(void);

/**
 * Get FlashInfer version string.
 */
const char* flashinfer_version(void);

/**
 * Check if the current GPU supports FlashInfer operations.
 *
 * @param[out] supported  Set to 1 if supported, 0 otherwise
 * @param[out] sm_version Set to the SM version (e.g., 80 for SM80)
 * @return Status code
 */
FlashInferStatus flashinfer_check_gpu_support(
    int* supported,
    int* sm_version
);

#ifdef __cplusplus
}
#endif

#endif /* FLASHINFER_C_API_H_ */

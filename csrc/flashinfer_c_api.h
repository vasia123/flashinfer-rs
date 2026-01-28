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
    FLASHINFER_NOT_INITIALIZED = 6,
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
    FLASHINFER_POS_ENCODING_ROPE_LLAMA_FREQ_SCALE = 3,
} FlashInferPosEncoding;

/* Attention mask mode */
typedef enum {
    FLASHINFER_MASK_NONE = 0,
    FLASHINFER_MASK_CAUSAL = 1,
    FLASHINFER_MASK_CUSTOM = 2,
} FlashInferMaskMode;

/* Attention backend selection */
typedef enum {
    FLASHINFER_BACKEND_AUTO = 0,
    FLASHINFER_BACKEND_FA2 = 1,      /* FlashAttention-2 */
    FLASHINFER_BACKEND_FA3 = 2,      /* FlashAttention-3 (SM90+) */
    FLASHINFER_BACKEND_CUDNN = 3,    /* cuDNN */
} FlashInferBackend;

/* ============================================================================
 * Opaque handles
 * ============================================================================ */

/* Opaque handle for batch decode plan */
typedef struct FlashInferBatchDecodePlan* FlashInferBatchDecodePlanHandle;

/* Opaque handle for batch prefill plan */
typedef struct FlashInferBatchPrefillPlan* FlashInferBatchPrefillPlanHandle;

/* ============================================================================
 * Configuration structures
 * ============================================================================ */

/**
 * Attention configuration for batch operations.
 */
typedef struct {
    uint32_t num_qo_heads;           /* Number of query/output heads */
    uint32_t num_kv_heads;           /* Number of key/value heads (< num_qo_heads for GQA) */
    uint32_t head_dim_qk;            /* Head dimension for Q and K */
    uint32_t head_dim_vo;            /* Head dimension for V and O (can differ for MLA) */

    FlashInferPosEncoding pos_encoding_mode;
    FlashInferMaskMode mask_mode;
    FlashInferBackend backend;

    float sm_scale;                  /* 1/sqrt(head_dim) by default, 0 for auto */
    float rope_scale;                /* RoPE scale factor */
    float rope_theta;                /* RoPE theta parameter */
    float logits_soft_cap;           /* Soft cap for attention logits (0 to disable) */
    int32_t window_left;             /* Sliding window size (-1 for full attention) */

    int use_fp16_qk_reduction;       /* Use FP16 for QK reduction */
    int return_lse;                  /* Return log-sum-exp values */
} FlashInferAttentionConfig;

/**
 * RoPE configuration.
 */
typedef struct {
    uint32_t rotary_dim;             /* Dimension to apply RoPE to */
    int interleave;                  /* Use interleaved RoPE format */
    float scale;                     /* RoPE scale factor */
    float theta;                     /* RoPE theta parameter */
} FlashInferRoPEConfig;

/**
 * Sampling configuration.
 */
typedef struct {
    int deterministic;               /* Use deterministic sampling */
} FlashInferSamplingConfig;

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

/**
 * Get required workspace size for general operations.
 */
FlashInferStatus flashinfer_get_workspace_size(
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    uint32_t batch_size,
    uint32_t max_seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size
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
    int32_t window_left,  /* Sliding window size (-1 for full attention) */
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
 * Execute batch decode with FP8 scaling.
 */
FlashInferStatus flashinfer_batch_decode_run_fp8(
    FlashInferBatchDecodePlanHandle plan_handle,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_last_page_len,
    void* output,
    float* lse,
    float q_scale,
    float k_scale,
    float v_scale,
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
    int32_t window_left,  /* Sliding window size (-1 for full attention) */
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
 * Single Request Attention API
 * ============================================================================ */

/**
 * Single decode attention (no batching overhead).
 *
 * @param q           Query tensor [num_qo_heads, head_dim]
 * @param k           Key tensor [seq_len, num_kv_heads, head_dim]
 * @param v           Value tensor [seq_len, num_kv_heads, head_dim]
 * @param output      Output tensor [num_qo_heads, head_dim]
 * @param lse         Optional log-sum-exp [num_qo_heads] (NULL to skip)
 * @param num_qo_heads Number of query/output heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim    Head dimension
 * @param seq_len     Sequence length
 * @param dtype       Data type
 * @param pos_encoding Position encoding mode
 * @param sm_scale    Softmax scale (0 for auto)
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_single_decode(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    float* lse,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t seq_len,
    FlashInferDType dtype,
    FlashInferPosEncoding pos_encoding,
    float sm_scale,
    void* stream
);

/**
 * Single prefill attention (no batching overhead).
 *
 * @param q           Query tensor [qo_len, num_qo_heads, head_dim]
 * @param k           Key tensor [kv_len, num_kv_heads, head_dim]
 * @param v           Value tensor [kv_len, num_kv_heads, head_dim]
 * @param output      Output tensor [qo_len, num_qo_heads, head_dim]
 * @param lse         Optional log-sum-exp [qo_len, num_qo_heads] (NULL to skip)
 * @param num_qo_heads Number of query/output heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim    Head dimension
 * @param qo_len      Query sequence length
 * @param kv_len      Key/Value sequence length
 * @param dtype       Data type
 * @param pos_encoding Position encoding mode
 * @param causal      Apply causal masking
 * @param sm_scale    Softmax scale (0 for auto)
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_single_prefill(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    float* lse,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t qo_len,
    int32_t kv_len,
    FlashInferDType dtype,
    FlashInferPosEncoding pos_encoding,
    int causal,
    float sm_scale,
    void* stream
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

/**
 * Compute batch indices and positions from indptr for appending.
 *
 * @param append_indptr    Token offsets [batch_size + 1]
 * @param seq_lens         Sequence lengths before append [batch_size]
 * @param batch_indices    Output batch indices [total_tokens]
 * @param positions        Output position indices [total_tokens]
 * @param batch_size       Number of sequences
 * @param total_tokens     Total number of tokens
 * @param stream           CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_get_batch_indices_positions(
    const int32_t* append_indptr,
    const int32_t* seq_lens,
    int32_t* batch_indices,
    int32_t* positions,
    uint32_t batch_size,
    uint32_t total_tokens,
    void* stream
);

/* ============================================================================
 * Normalization API
 * ============================================================================ */

/**
 * RMSNorm operation.
 *
 * output = input / sqrt(mean(input^2) + eps) * weight
 *
 * @param input       Input tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param output      Output tensor [batch_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon for numerical stability
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
);

/**
 * RMSNorm with quantized FP8 output.
 *
 * @param input       Input tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param output      Output tensor [batch_size, hidden_dim] in FP8
 * @param scale       Output scale tensor [1] or [batch_size]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon
 * @param input_dtype Input data type
 * @param output_dtype Output data type (FP8)
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Fused residual add + RMSNorm.
 *
 * input += residual
 * output = rmsnorm(input)
 *
 * @param input       Input tensor [batch_size, hidden_dim] (modified in-place)
 * @param residual    Residual tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param output      Output tensor [batch_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * LayerNorm operation.
 *
 * @param input       Input tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param bias        Bias tensor [hidden_dim] (can be NULL)
 * @param output      Output tensor [batch_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * QK RMSNorm for attention heads.
 *
 * Applies RMSNorm independently to each attention head. Used in some
 * architectures that normalize Q and K before attention computation.
 *
 * @param input       Input tensor [batch_size, num_heads, head_dim]
 * @param weight      Weight tensor [head_dim]
 * @param output      Output tensor [batch_size, num_heads, head_dim]
 * @param batch_size  Batch size (number of tokens)
 * @param num_heads   Number of attention heads
 * @param head_dim    Head dimension
 * @param eps         Epsilon for numerical stability
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Gemma-style RMSNorm.
 *
 * Similar to RMSNorm but adds 1.0 to the weight before applying.
 * This matches the Gemma model's normalization implementation:
 *   output = x * rsqrt(mean(x^2) + eps) * (weight + 1)
 *
 * @param input       Input tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param output      Output tensor [batch_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon for numerical stability
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_gemma_rmsnorm(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    FlashInferDType dtype,
    void* stream
);

/**
 * Gemma-style fused residual add + RMSNorm.
 *
 * Combines residual addition with Gemma RMSNorm:
 *   input += residual
 *   output = GemmaRMSNorm(input)
 *
 * @param input       Input tensor [batch_size, hidden_dim] (modified in-place)
 * @param residual    Residual tensor [batch_size, hidden_dim]
 * @param weight      Weight tensor [hidden_dim]
 * @param output      Output tensor [batch_size, hidden_dim]
 * @param batch_size  Batch size
 * @param hidden_dim  Hidden dimension
 * @param eps         Epsilon
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/* ============================================================================
 * RoPE API
 * ============================================================================ */

/**
 * Configuration for fused RoPE + Quantize + Append operation.
 */
typedef struct {
    uint32_t num_qo_heads;       /* Number of query/output heads */
    uint32_t num_kv_heads;       /* Number of key/value heads */
    uint32_t rope_dim;           /* Dimension for RoPE (rotary part) */
    uint32_t no_rope_dim;        /* Dimension for non-RoPE part (for MLA) */
    uint32_t page_size;          /* Tokens per page */
    float quant_scale_q;         /* Quantization scale for Q */
    float quant_scale_kv;        /* Quantization scale for K/V */
    int interleave;              /* Use interleaved RoPE format */
    int enable_pdl;              /* Enable PDL (SM90+) */
    FlashInferDType input_dtype; /* Input data type (FLOAT16 or BFLOAT16) */
    FlashInferDType output_dtype;/* Output data type (FLOAT8_E4M3 or FLOAT8_E5M2) */
    FlashInferKVLayout kv_layout;/* KV cache memory layout */
} FlashInferRopeQuantAppendConfig;

/**
 * Fused RoPE + Quantize + Append to paged KV cache (requires SM89+).
 *
 * This operation combines:
 * 1. Apply RoPE to Q_rope and K_rope tensors
 * 2. Quantize all outputs to FP8
 * 3. Append K/V to paged cache
 *
 * Used for FP8 inference in models with MLA or standard attention.
 *
 * @param q_rope_in       Query RoPE tensor [nnz, num_qo_heads, rope_dim]
 * @param k_rope_in       Key RoPE tensor [nnz, num_kv_heads, rope_dim]
 * @param q_nope_in       Query non-RoPE tensor [nnz, num_qo_heads, no_rope_dim]
 * @param k_nope_in       Key non-RoPE tensor [nnz, num_kv_heads, no_rope_dim]
 * @param v_in            Value tensor [nnz, num_kv_heads, head_dim]
 * @param q_rope_out      Output query RoPE tensor (FP8) [nnz, num_qo_heads, rope_dim]
 * @param q_nope_out      Output query non-RoPE tensor (FP8) [nnz, num_qo_heads, no_rope_dim]
 * @param k_cache         Paged K cache (FP8)
 * @param v_cache         Paged V cache (FP8)
 * @param kv_indptr       Page offsets [batch_size + 1]
 * @param kv_indices      Page indices [total_pages]
 * @param kv_last_page_len Tokens in last page before append [batch_size]
 * @param batch_indices   Batch index for each token [nnz]
 * @param positions       Position for each token within its sequence [nnz]
 * @param cos_sin_cache   Precomputed cos/sin cache [max_pos, rope_dim]
 * @param pos_ids         Position IDs for RoPE [nnz]
 * @param nnz             Total number of tokens
 * @param config          Configuration for the operation
 * @param stream          CUDA stream
 * @return Status code
 */
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
);

/**
 * Apply RoPE to Q and K tensors using batch indptr/offsets.
 *
 * @param q           Query tensor [total_tokens, num_qo_heads, head_dim]
 * @param k           Key tensor [total_tokens, num_kv_heads, head_dim]
 * @param q_out       Output query tensor (can be same as q for in-place)
 * @param k_out       Output key tensor (can be same as k for in-place)
 * @param indptr      Token offsets [batch_size + 1]
 * @param offsets     Position offsets [batch_size]
 * @param batch_size  Number of sequences in batch
 * @param num_qo_heads Number of query heads
 * @param num_kv_heads Number of key heads
 * @param head_dim    Head dimension
 * @param config      RoPE configuration
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Apply RoPE in-place using batch indptr/offsets.
 *
 * @param q           Query tensor [total_tokens, num_qo_heads, head_dim] (modified in-place)
 * @param k           Key tensor [total_tokens, num_kv_heads, head_dim] (modified in-place)
 * @param indptr      Token offsets [batch_size + 1]
 * @param offsets     Position offsets [batch_size]
 * @param batch_size  Number of sequences in batch
 * @param num_qo_heads Number of query heads
 * @param num_kv_heads Number of key heads
 * @param head_dim    Head dimension
 * @param config      RoPE configuration
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Apply RoPE with explicit position IDs.
 *
 * @param q           Query tensor [total_tokens, num_qo_heads, head_dim]
 * @param k           Key tensor [total_tokens, num_kv_heads, head_dim]
 * @param q_out       Output query tensor
 * @param k_out       Output key tensor
 * @param pos_ids     Position IDs [total_tokens]
 * @param total_tokens Total number of tokens
 * @param num_qo_heads Number of query heads
 * @param num_kv_heads Number of key heads
 * @param head_dim    Head dimension
 * @param config      RoPE configuration
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Apply RoPE with precomputed cos/sin cache.
 *
 * @param q           Query tensor
 * @param k           Key tensor
 * @param q_out       Output query tensor
 * @param k_out       Output key tensor
 * @param cos_cache   Precomputed cos values [max_seq_len, rotary_dim/2]
 * @param sin_cache   Precomputed sin values [max_seq_len, rotary_dim/2]
 * @param pos_ids     Position IDs [total_tokens]
 * @param total_tokens Total number of tokens
 * @param num_qo_heads Number of query heads
 * @param num_kv_heads Number of key heads
 * @param head_dim    Head dimension
 * @param config      RoPE configuration (rotary_dim used)
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/* ============================================================================
 * Sampling API
 * ============================================================================ */

/**
 * Top-K sampling from probability distribution.
 *
 * Uses Philox RNG for random number generation. The combination of
 * philox_seed and philox_offset determines the random state.
 *
 * @param probs       Probability tensor [batch_size, vocab_size]
 * @param output      Sampled indices [batch_size]
 * @param top_k_arr   Per-batch Top-K values [batch_size] (NULL for uniform top_k_val)
 * @param top_k_val   Default Top-K value (used when top_k_arr is NULL or per-batch)
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param deterministic Use deterministic sampling
 * @param philox_seed Random seed for Philox RNG
 * @param philox_offset Offset for Philox RNG
 * @param dtype       Data type (float16 or bfloat16)
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Top-P (nucleus) sampling from probability distribution.
 *
 * @param probs       Probability tensor [batch_size, vocab_size]
 * @param output      Sampled indices [batch_size]
 * @param top_p_arr   Per-batch Top-P values [batch_size] (NULL for uniform top_p_val)
 * @param top_p_val   Default Top-P value (used when top_p_arr is NULL)
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param deterministic Use deterministic sampling
 * @param philox_seed Random seed for Philox RNG
 * @param philox_offset Offset for Philox RNG
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Min-P sampling from probability distribution.
 *
 * @param probs       Probability tensor [batch_size, vocab_size]
 * @param output      Sampled indices [batch_size]
 * @param min_p_arr   Per-batch Min-P values [batch_size] (NULL for uniform min_p_val)
 * @param min_p_val   Default Min-P value (used when min_p_arr is NULL)
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param deterministic Use deterministic sampling
 * @param philox_seed Random seed for Philox RNG
 * @param philox_offset Offset for Philox RNG
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Combined Top-K + Top-P sampling.
 *
 * @param probs       Probability tensor [batch_size, vocab_size]
 * @param output      Sampled indices [batch_size]
 * @param top_k_arr   Per-batch Top-K values [batch_size] (NULL for uniform top_k_val)
 * @param top_p_arr   Per-batch Top-P values [batch_size] (NULL for uniform top_p_val)
 * @param top_k_val   Default Top-K value
 * @param top_p_val   Default Top-P value
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param deterministic Use deterministic sampling
 * @param philox_seed Random seed for Philox RNG
 * @param philox_offset Offset for Philox RNG
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
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
);

/**
 * Softmax with optional temperature scaling.
 *
 * For large vocabularies (>24K) with small batches, this uses a multi-pass
 * algorithm with workspace. Otherwise, uses a single-pass fused kernel.
 *
 * @param logits      Logits tensor [batch_size, vocab_size]
 * @param probs       Output probability tensor [batch_size, vocab_size]
 * @param temperature_arr Per-batch temperature values [batch_size] (NULL for temp_val)
 * @param temp_val    Default temperature value (used when temperature_arr is NULL)
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_softmax(
    const void* logits,
    void* probs,
    const float* temperature_arr,
    float temp_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
);

/**
 * Apply Top-K mask to logits (set non-top-K to -inf).
 *
 * @param logits      Logits tensor [batch_size, vocab_size] (modified in-place)
 * @param top_k_arr   Top-K values [batch_size]
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_top_k_mask_logits(
    void* logits,
    const int32_t* top_k_arr,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
);

/**
 * Renormalize probabilities after top-p filtering.
 *
 * @param probs       Probabilities [batch_size, vocab_size] (modified in-place)
 * @param renormed_probs Output renormalized probs [batch_size, vocab_size] (can be same as probs)
 * @param top_p_arr   Per-batch Top-P values [batch_size] (NULL for top_p_val)
 * @param top_p_val   Default Top-P value (1.0 = no filtering, just normalize)
 * @param batch_size  Batch size
 * @param vocab_size  Vocabulary size
 * @param dtype       Data type
 * @param stream      CUDA stream
 * @return Status code
 */
FlashInferStatus flashinfer_top_p_renorm_probs(
    const void* probs,
    void* renormed_probs,
    const float* top_p_arr,
    float top_p_val,
    uint32_t batch_size,
    uint32_t vocab_size,
    FlashInferDType dtype,
    void* stream
);

/* ============================================================================
 * MLA (Multi-head Latent Attention) API - DeepSeek Support
 * ============================================================================ */

/**
 * Opaque handle for MLA attention plan.
 */
typedef struct FlashInferMLAPlan* FlashInferMLAPlanHandle;

/**
 * Append compressed KV and k_pe to MLA paged cache.
 *
 * MLA uses two separate caches:
 * - ckv_cache: compressed KV (dimension 512 for DeepSeek)
 * - kpe_cache: K position embedding (dimension 64 for DeepSeek)
 *
 * @param append_ckv       Compressed KV to append [nnz, head_dim_ckv=512]
 * @param append_kpe       K position embeddings [nnz, head_dim_kpe=64]
 * @param ckv_cache        Paged compressed KV cache [num_pages, page_size, 512]
 * @param kpe_cache        Paged K-PE cache [num_pages, page_size, 64]
 * @param kv_indptr        Page offsets [batch_size + 1]
 * @param kv_indices       Page indices [total_pages]
 * @param kv_last_page_len Tokens in last page [batch_size]
 * @param batch_indices    Batch index for each token [nnz]
 * @param positions        Position for each token [nnz]
 * @param nnz              Total number of tokens
 * @param batch_size       Number of sequences
 * @param page_size        Tokens per page
 * @param dtype            Data type (fp16/bf16)
 * @param stream           CUDA stream
 * @return Status code
 */
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
);

/**
 * Concatenate k_nope and k_rope for MLA.
 *
 * k_nope: [num_tokens, num_heads=128, nope_dim=128]
 * k_rope: [num_tokens, 1, rope_dim=64] (broadcast to all heads)
 * k:      [num_tokens, num_heads=128, k_head_dim=192]
 *
 * @param k                Output tensor [num_tokens, 128, 192]
 * @param k_nope           Input k_nope tensor [num_tokens, 128, 128]
 * @param k_rope           Input k_rope tensor [num_tokens, 1, 64] (shared)
 * @param num_tokens       Number of tokens
 * @param k_stride_n       Token stride for k
 * @param k_stride_h       Head stride for k
 * @param k_nope_stride_n  Token stride for k_nope
 * @param k_nope_stride_h  Head stride for k_nope
 * @param k_rope_stride_n  Token stride for k_rope
 * @param dtype            Data type
 * @param stream           CUDA stream
 * @return Status code
 */
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
);

/**
 * Create MLA attention plan.
 *
 * @param[out] plan_handle       Handle to created plan
 * @param float_workspace        GPU float workspace buffer
 * @param float_workspace_size   Size of float workspace
 * @param int_workspace          GPU int workspace buffer
 * @param int_workspace_size     Size of int workspace
 * @param page_locked_workspace  Page-locked host memory (can be NULL)
 * @param page_locked_size       Size of page-locked buffer
 * @param qo_indptr              Query token offsets [batch_size + 1]
 * @param kv_indptr              KV page offsets [batch_size + 1]
 * @param kv_len                 KV lengths [batch_size]
 * @param batch_size             Number of sequences
 * @param num_heads              Number of attention heads (128 for DeepSeek)
 * @param head_dim_ckv           Compressed KV head dim (512)
 * @param head_dim_kpe           K position embedding dim (64)
 * @param page_size              Tokens per page
 * @param causal                 Whether to apply causal masking
 * @param stream                 CUDA stream
 * @return Status code
 */
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
);

/**
 * Execute MLA attention with paged KV cache.
 *
 * @param plan_handle    Plan from flashinfer_mla_plan
 * @param q_nope         Query nope [nnz, num_heads, head_dim_ckv=512]
 * @param q_pe           Query PE [nnz, num_heads, head_dim_kpe=64]
 * @param ckv_cache      Compressed KV cache [num_pages, page_size, 512]
 * @param kpe_cache      K position embedding cache [num_pages, page_size, 64]
 * @param kv_indices     Page indices [total_pages]
 * @param output         Output tensor [nnz, num_heads, head_dim_ckv=512]
 * @param lse            Optional LSE output [nnz, num_heads] (NULL to skip)
 * @param mask_mode      Attention mask mode
 * @param sm_scale       Softmax scale (1/sqrt(192) for DeepSeek)
 * @param dtype_q        Query dtype
 * @param dtype_kv       KV dtype
 * @param stream         CUDA stream
 * @return Status code
 */
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
);

/**
 * Destroy MLA plan and free resources.
 *
 * @param plan_handle Handle to destroy
 * @return Status code
 */
FlashInferStatus flashinfer_mla_plan_destroy(FlashInferMLAPlanHandle plan_handle);

/**
 * Get required workspace sizes for MLA attention.
 *
 * @param[out] float_size    Required float workspace size
 * @param[out] int_size      Required int workspace size
 * @param batch_size         Number of sequences
 * @param max_seq_len        Maximum sequence length
 * @param num_heads          Number of heads (128)
 * @param head_dim_ckv       Compressed KV dim (512)
 * @param head_dim_kpe       K PE dim (64)
 * @param page_size          Tokens per page
 * @return Status code
 */
FlashInferStatus flashinfer_mla_workspace_size(
    size_t* float_size,
    size_t* int_size,
    int32_t batch_size,
    int32_t max_seq_len,
    int32_t num_heads,
    int32_t head_dim_ckv,
    int32_t head_dim_kpe,
    int32_t page_size
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

/**
 * Query device capabilities.
 *
 * @param device_id       CUDA device ID
 * @param[out] compute_major Major compute capability
 * @param[out] compute_minor Minor compute capability
 * @param[out] sm_count   Number of SMs
 * @param[out] supports_pdl Whether device supports PDL (SM90+)
 * @param[out] supports_fp8 Whether device supports FP8 (SM89+)
 * @return Status code
 */
FlashInferStatus flashinfer_get_device_info(
    int device_id,
    int* compute_major,
    int* compute_minor,
    int* sm_count,
    int* supports_pdl,
    int* supports_fp8
);

/**
 * Set default CUDA stream for subsequent operations.
 */
FlashInferStatus flashinfer_set_stream(void* stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASHINFER_C_API_H_ */

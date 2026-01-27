/*
 * FlashInfer RoPE (Rotary Position Embedding) Operations
 *
 * Contains apply_rope, apply_rope_inplace, and cosine/sine cache operations.
 *
 * TODO: Implement after studying FlashInfer's pos_enc.cuh API more thoroughly.
 * The API has changed significantly and requires careful study of the new parameters.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

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
    set_error("apply_rope not yet implemented - requires study of new FlashInfer pos_enc.cuh API");
    return FLASHINFER_UNSUPPORTED;
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
    set_error("apply_rope_inplace not yet implemented - requires study of new FlashInfer pos_enc.cuh API");
    return FLASHINFER_UNSUPPORTED;
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
    set_error("apply_rope_pos_ids not yet implemented - requires study of new FlashInfer pos_enc.cuh API");
    return FLASHINFER_UNSUPPORTED;
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
    set_error("apply_rope_with_cos_sin_cache not yet implemented - requires study of new FlashInfer pos_enc.cuh API");
    return FLASHINFER_UNSUPPORTED;
}

} // extern "C"

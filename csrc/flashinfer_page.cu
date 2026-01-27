/*
 * FlashInfer Paged KV Cache Operations
 *
 * Contains append_paged_kv_cache for decode and prefill operations.
 *
 * Copyright (c) 2024 flashinfer-rs contributors
 * Licensed under Apache-2.0
 */

#include "flashinfer_common.h"

#include <flashinfer/page.cuh>

using namespace flashinfer_rs;

extern "C" {

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

    if (!k || !v || !k_cache || !v_cache || !kv_indptr || !kv_indices || !kv_last_page_len || !append_indptr) {
        set_error("Null pointer passed to append_paged_kv_cache");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    flashinfer::QKVLayout layout = (kv_layout == FLASHINFER_KV_LAYOUT_NHD) ?
        flashinfer::QKVLayout::kNHD : flashinfer::QKVLayout::kHND;

    // We need to compute batch_indices and positions from append_indptr
    // First, read append_indptr to get total tokens
    std::vector<int32_t> h_append_indptr(batch_size + 1);
    cudaError_t err = cudaMemcpyAsync(h_append_indptr.data(), append_indptr,
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    int32_t nnz = h_append_indptr[batch_size];

    // Compute batch_indices and positions on CPU
    std::vector<int32_t> h_batch_indices(nnz);
    std::vector<int32_t> h_positions(nnz);

    // We also need to read kv_last_page_len to compute correct positions
    std::vector<int32_t> h_kv_last_page_len(batch_size);
    err = cudaMemcpyAsync(h_kv_last_page_len.data(), kv_last_page_len,
        batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    // Read kv_indptr to compute seq_lens
    std::vector<int32_t> h_kv_indptr(batch_size + 1);
    err = cudaMemcpyAsync(h_kv_indptr.data(), kv_indptr,
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    for (int32_t b = 0; b < batch_size; ++b) {
        int32_t start = h_append_indptr[b];
        int32_t end = h_append_indptr[b + 1];
        // Compute current sequence length from pages
        int32_t num_pages = h_kv_indptr[b + 1] - h_kv_indptr[b];
        int32_t seq_len = (num_pages > 0) ? (num_pages - 1) * page_size + h_kv_last_page_len[b] : 0;
        for (int32_t i = start; i < end; ++i) {
            h_batch_indices[i] = b;
            h_positions[i] = seq_len + (i - start);
        }
    }

    // Allocate device memory for batch_indices and positions
    int32_t* d_batch_indices = nullptr;
    int32_t* d_positions = nullptr;
    err = cudaMalloc(&d_batch_indices, nnz * sizeof(int32_t));
    if (err != cudaSuccess) return from_cuda_error(err);
    err = cudaMalloc(&d_positions, nnz * sizeof(int32_t));
    if (err != cudaSuccess) {
        cudaFree(d_batch_indices);
        return from_cuda_error(err);
    }

    // Copy to device
    err = cudaMemcpyAsync(d_batch_indices, h_batch_indices.data(),
        nnz * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_batch_indices);
        cudaFree(d_positions);
        return from_cuda_error(err);
    }
    err = cudaMemcpyAsync(d_positions, h_positions.data(),
        nnz * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) {
        cudaFree(d_batch_indices);
        cudaFree(d_positions);
        return from_cuda_error(err);
    }

    // Dispatch based on dtype and head_dim
    #define DISPATCH_APPEND(DType, HEAD_DIM_V) do { \
        flashinfer::paged_kv_t<DType, int32_t> paged_kv( \
            static_cast<uint32_t>(num_kv_heads), \
            static_cast<uint32_t>(page_size), \
            static_cast<uint32_t>(HEAD_DIM_V), \
            static_cast<uint32_t>(batch_size), \
            layout, \
            static_cast<DType*>(k_cache), \
            static_cast<DType*>(v_cache), \
            const_cast<int32_t*>(kv_indices), \
            const_cast<int32_t*>(kv_indptr), \
            const_cast<int32_t*>(kv_last_page_len), \
            static_cast<int32_t*>(nullptr) \
        ); \
        size_t k_stride_n = static_cast<size_t>(num_kv_heads) * HEAD_DIM_V; \
        size_t k_stride_h = HEAD_DIM_V; \
        size_t v_stride_n = static_cast<size_t>(num_kv_heads) * HEAD_DIM_V; \
        size_t v_stride_h = HEAD_DIM_V; \
        err = flashinfer::AppendPagedKVCache<DType, int32_t>( \
            paged_kv, \
            const_cast<DType*>(static_cast<const DType*>(k)), \
            const_cast<DType*>(static_cast<const DType*>(v)), \
            d_batch_indices, \
            d_positions, \
            static_cast<uint32_t>(nnz), \
            k_stride_n, k_stride_h, v_stride_n, v_stride_h, \
            cuda_stream \
        ); \
    } while(0)

    switch (dtype) {
        case FLASHINFER_DTYPE_FLOAT16:
            switch (head_dim) {
                case 64: DISPATCH_APPEND(__half, 64); break;
                case 128: DISPATCH_APPEND(__half, 128); break;
                case 256: DISPATCH_APPEND(__half, 256); break;
                default:
                    cudaFree(d_batch_indices);
                    cudaFree(d_positions);
                    set_error("Unsupported head_dim: " + std::to_string(head_dim));
                    return FLASHINFER_INVALID_ARGUMENT;
            }
            break;
        case FLASHINFER_DTYPE_BFLOAT16:
            switch (head_dim) {
                case 64: DISPATCH_APPEND(__nv_bfloat16, 64); break;
                case 128: DISPATCH_APPEND(__nv_bfloat16, 128); break;
                case 256: DISPATCH_APPEND(__nv_bfloat16, 256); break;
                default:
                    cudaFree(d_batch_indices);
                    cudaFree(d_positions);
                    set_error("Unsupported head_dim: " + std::to_string(head_dim));
                    return FLASHINFER_INVALID_ARGUMENT;
            }
            break;
        default:
            cudaFree(d_batch_indices);
            cudaFree(d_positions);
            set_error("Unsupported dtype for append_paged_kv_cache");
            return FLASHINFER_UNSUPPORTED;
    }

    #undef DISPATCH_APPEND

    // Free temporary allocations
    cudaFree(d_batch_indices);
    cudaFree(d_positions);

    return from_cuda_error(err);
}

FlashInferStatus flashinfer_get_batch_indices_positions(
    const int32_t* append_indptr,
    const int32_t* seq_lens,
    int32_t* batch_indices,
    int32_t* positions,
    uint32_t batch_size,
    uint32_t total_tokens,
    void* stream
) {
    clear_error();

    if (!append_indptr || !seq_lens || !batch_indices || !positions) {
        set_error("Null pointer passed to get_batch_indices_positions");
        return FLASHINFER_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Read append_indptr and seq_lens from GPU
    std::vector<int32_t> h_indptr(batch_size + 1);
    std::vector<int32_t> h_seq_lens(batch_size);

    cudaError_t err = cudaMemcpyAsync(h_indptr.data(), append_indptr,
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    err = cudaMemcpyAsync(h_seq_lens.data(), seq_lens,
        batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    err = cudaStreamSynchronize(cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    // Compute batch_indices and positions
    std::vector<int32_t> h_batch_indices(total_tokens);
    std::vector<int32_t> h_positions(total_tokens);

    for (uint32_t b = 0; b < batch_size; ++b) {
        int32_t start = h_indptr[b];
        int32_t end = h_indptr[b + 1];
        int32_t seq_len = h_seq_lens[b];
        for (int32_t i = start; i < end; ++i) {
            h_batch_indices[i] = static_cast<int32_t>(b);
            h_positions[i] = seq_len + (i - start);
        }
    }

    // Copy to GPU
    err = cudaMemcpyAsync(batch_indices, h_batch_indices.data(),
        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    err = cudaMemcpyAsync(positions, h_positions.data(),
        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) return from_cuda_error(err);

    return FLASHINFER_SUCCESS;
}

} // extern "C"

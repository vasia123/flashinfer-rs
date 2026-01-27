//! Integration tests for BatchDecodeHandler and BatchPrefillHandler.
//!
//! These tests require a CUDA-capable GPU and are ignored by default.
//! Run with: `cargo test --features cuda -- --ignored`

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaDevice, CudaSlice};
use flashinfer_rs::{AttentionConfig, BatchDecodeHandler, BatchPrefillHandler, PageTable};
use std::sync::Arc;

type Device = Arc<CudaDevice>;

/// Helper to check if CUDA is available
fn cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Create test query tensor for decode (batch_size, num_qo_heads, head_dim)
fn create_decode_query(
    device: &Device,
    batch_size: usize,
    num_qo_heads: usize,
    head_dim: usize,
) -> CudaSlice<half::f16> {
    let size = batch_size * num_qo_heads * head_dim;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32 % 10.0) * 0.1))
        .collect();
    device.htod_sync_copy(&data).unwrap()
}

/// Create test query tensor for prefill (total_tokens, num_qo_heads, head_dim)
fn create_prefill_query(
    device: &Device,
    total_tokens: usize,
    num_qo_heads: usize,
    head_dim: usize,
) -> CudaSlice<half::f16> {
    let size = total_tokens * num_qo_heads * head_dim;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32 % 10.0) * 0.1))
        .collect();
    device.htod_sync_copy(&data).unwrap()
}

/// Create test KV cache (num_pages, num_kv_heads, page_size, head_dim) for HND layout
fn create_kv_cache(
    device: &Device,
    num_pages: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> CudaSlice<half::f16> {
    let size = num_pages * num_kv_heads * page_size * head_dim;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i as f32 % 10.0) * 0.05))
        .collect();
    device.htod_sync_copy(&data).unwrap()
}

/// Create output buffer
fn create_output(
    device: &Device,
    total_tokens: usize,
    num_qo_heads: usize,
    head_dim: usize,
) -> CudaSlice<half::f16> {
    let size = total_tokens * num_qo_heads * head_dim;
    device.alloc_zeros(size).unwrap()
}

/// Create a simple page table for testing
fn create_test_page_table(batch_size: usize, pages_per_seq: usize) -> PageTable {
    let mut table = PageTable::new(batch_size, pages_per_seq);
    let mut block_id = 0i32;
    for seq in 0..batch_size {
        for page in 0..pages_per_seq {
            table.set(seq, page, block_id).unwrap();
            block_id += 1;
        }
    }
    table
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_handler_basic() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config: 8 QO heads, 2 KV heads (GQA ratio 4), head_dim 64
    let config = AttentionConfig::new(8, 2, 64).with_page_size(16);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    // Test params
    let batch_size = 4;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size; // 32 tokens per sequence

    // Create page table
    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    // Create tensors
    let query = create_decode_query(&device, batch_size, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, batch_size, 8, 64);

    // Plan
    handler
        .plan::<half::f16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());
    assert_eq!(handler.batch_size(), Some(batch_size));

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    // Sync and verify output is not NaN
    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Output contains NaN values");
    assert!(!has_inf, "Output contains Inf values");

    // Verify output is not all zeros (attention should produce some values)
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Output is all zeros, attention may not have run");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_prefill_handler_basic() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config: 8 QO heads, 2 KV heads, head_dim 64, causal mask
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_causal_mask();

    let mut handler = BatchPrefillHandler::new(device.clone(), config).unwrap();

    // Test params
    let batch_size = 2;
    let seq_lens = [10, 20]; // Variable length sequences
    let total_tokens: usize = seq_lens.iter().sum();
    let page_size = 16;

    // Build qo_indptr
    let mut qo_indptr = vec![0i32];
    let mut cumsum = 0i32;
    for &len in &seq_lens {
        cumsum += len as i32;
        qo_indptr.push(cumsum);
    }

    // Create page table (enough pages for each sequence)
    let pages_per_seq = 2; // At least enough for max seq len
    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = seq_lens.iter().map(|&x| x as usize).collect();

    // Create tensors
    let query = create_prefill_query(&device, total_tokens, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, total_tokens, 8, 64);

    // Plan
    handler
        .plan::<half::f16>(&qo_indptr, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());
    assert_eq!(handler.batch_size(), Some(batch_size));
    assert_eq!(handler.total_tokens(), Some(total_tokens));

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    // Sync and verify
    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Output contains NaN values");
    assert!(!has_inf, "Output contains Inf values");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_handler_bf16() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let config = AttentionConfig::new(8, 2, 128).with_page_size(16);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let batch_size = 2;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size;

    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    // Create bf16 tensors
    let query_size = batch_size * 8 * 128;
    let query_data: Vec<half::bf16> = (0..query_size)
        .map(|i| half::bf16::from_f32((i as f32 % 10.0) * 0.1))
        .collect();
    let query = device.htod_sync_copy(&query_data).unwrap();

    let total_pages = batch_size * pages_per_seq;
    let cache_size = total_pages * 2 * page_size * 128;
    let cache_data: Vec<half::bf16> = (0..cache_size)
        .map(|i| half::bf16::from_f32((i as f32 % 10.0) * 0.05))
        .collect();
    let kv_cache_k = device.htod_sync_copy(&cache_data).unwrap();
    let kv_cache_v = device.htod_sync_copy(&cache_data).unwrap();

    let mut output: CudaSlice<half::bf16> = device.alloc_zeros(query_size).unwrap();

    // Plan with bf16
    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    assert!(!has_nan, "BF16 output contains NaN values");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_prefill_non_causal() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Non-causal (bidirectional) attention
    let config = AttentionConfig::new(8, 2, 64).with_page_size(16);

    let mut handler = BatchPrefillHandler::new(device.clone(), config).unwrap();

    let batch_size = 1;
    let seq_len = 16;
    let qo_indptr = vec![0i32, seq_len as i32];

    let page_table = create_test_page_table(batch_size, 1);
    let kv_lengths = vec![seq_len];

    let query = create_prefill_query(&device, seq_len, 8, 64);
    let kv_cache_k = create_kv_cache(&device, 1, 2, 16, 64);
    let kv_cache_v = create_kv_cache(&device, 1, 2, 16, 64);
    let mut output = create_output(&device, seq_len, 8, 64);

    handler
        .plan::<half::f16>(&qo_indptr, &kv_lengths, &page_table, &stream)
        .unwrap();

    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    assert!(!has_nan, "Non-causal output contains NaN values");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_handler_replan() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let config = AttentionConfig::new(4, 4, 64).with_page_size(16);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    // First plan with batch_size=2
    {
        let page_table = create_test_page_table(2, 1);
        let kv_lengths = vec![16, 16];

        handler
            .plan::<half::f16>(2, &kv_lengths, &page_table, &stream)
            .unwrap();

        assert_eq!(handler.batch_size(), Some(2));
    }

    // Replan with batch_size=4
    {
        let page_table = create_test_page_table(4, 1);
        let kv_lengths = vec![16, 16, 16, 16];

        handler
            .plan::<half::f16>(4, &kv_lengths, &page_table, &stream)
            .unwrap();

        assert_eq!(handler.batch_size(), Some(4));
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_sliding_window() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with sliding window of 256 tokens (typical for Mistral)
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_sliding_window(256);

    assert_eq!(config.window_left, Some(256));

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let batch_size = 4;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size;

    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    let query = create_decode_query(&device, batch_size, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, batch_size, 8, 64);

    // Plan with sliding window config
    handler
        .plan::<half::f16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Sliding window decode output contains NaN values");
    assert!(!has_inf, "Sliding window decode output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Sliding window decode output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_prefill_sliding_window() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with sliding window and causal mask
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_causal_mask()
        .with_sliding_window(128); // Smaller window for prefill test

    assert_eq!(config.window_left, Some(128));

    let mut handler = BatchPrefillHandler::new(device.clone(), config).unwrap();

    let batch_size = 2;
    let seq_lens = [32, 48]; // Sequences longer than window to test masking
    let total_tokens: usize = seq_lens.iter().sum();
    let page_size = 16;

    let mut qo_indptr = vec![0i32];
    let mut cumsum = 0i32;
    for &len in &seq_lens {
        cumsum += len as i32;
        qo_indptr.push(cumsum);
    }

    let pages_per_seq = 4; // Enough pages
    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = seq_lens.iter().map(|&x| x as usize).collect();

    let query = create_prefill_query(&device, total_tokens, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, total_tokens, 8, 64);

    // Plan with sliding window config
    handler
        .plan::<half::f16>(&qo_indptr, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());
    assert_eq!(handler.total_tokens(), Some(total_tokens));

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Sliding window prefill output contains NaN values");
    assert!(!has_inf, "Sliding window prefill output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Sliding window prefill output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_logits_soft_cap() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with logits soft cap (typical value for Gemma 2 is around 30.0)
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_logits_soft_cap(30.0);

    assert_eq!(config.logits_soft_cap, Some(30.0));

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let batch_size = 4;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size;

    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    let query = create_decode_query(&device, batch_size, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, batch_size, 8, 64);

    // Plan with soft cap config
    handler
        .plan::<half::f16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Soft cap decode output contains NaN values");
    assert!(!has_inf, "Soft cap decode output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Soft cap decode output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_prefill_logits_soft_cap() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with logits soft cap and causal mask
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_causal_mask()
        .with_logits_soft_cap(50.0); // Higher cap for prefill

    assert_eq!(config.logits_soft_cap, Some(50.0));

    let mut handler = BatchPrefillHandler::new(device.clone(), config).unwrap();

    let batch_size = 2;
    let seq_lens = [16, 24];
    let total_tokens: usize = seq_lens.iter().sum();
    let page_size = 16;

    let mut qo_indptr = vec![0i32];
    let mut cumsum = 0i32;
    for &len in &seq_lens {
        cumsum += len as i32;
        qo_indptr.push(cumsum);
    }

    let pages_per_seq = 2;
    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = seq_lens.iter().map(|&x| x as usize).collect();

    let query = create_prefill_query(&device, total_tokens, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, total_tokens, 8, 64);

    // Plan with soft cap config
    handler
        .plan::<half::f16>(&qo_indptr, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());
    assert_eq!(handler.total_tokens(), Some(total_tokens));

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Soft cap prefill output contains NaN values");
    assert!(!has_inf, "Soft cap prefill output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Soft cap prefill output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_combined_features() {
    // Test combining sliding window + soft cap (for future models that might use both)
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with both sliding window AND logits soft cap
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_sliding_window(128)
        .with_logits_soft_cap(30.0);

    assert_eq!(config.window_left, Some(128));
    assert_eq!(config.logits_soft_cap, Some(30.0));

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let batch_size = 4;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size;

    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    let query = create_decode_query(&device, batch_size, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, batch_size, 8, 64);

    // Plan with combined config
    handler
        .plan::<half::f16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "Combined features decode output contains NaN values");
    assert!(!has_inf, "Combined features decode output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "Combined features decode output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_decode_alibi() {
    // Test ALiBi positional encoding (used by BLOOM, MPT)
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with ALiBi position encoding
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_alibi();

    assert_eq!(
        config.pos_encoding,
        flashinfer_rs::types::PosEncodingMode::ALiBi
    );

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let batch_size = 4;
    let pages_per_seq = 2;
    let page_size = 16;
    let kv_len_per_seq = pages_per_seq * page_size;

    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = vec![kv_len_per_seq; batch_size];

    let query = create_decode_query(&device, batch_size, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, batch_size, 8, 64);

    // Plan with ALiBi config
    handler
        .plan::<half::f16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "ALiBi decode output contains NaN values");
    assert!(!has_inf, "ALiBi decode output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "ALiBi decode output is all zeros");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_batch_prefill_alibi() {
    // Test ALiBi positional encoding for prefill
    if !cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    // Config with ALiBi and causal mask
    let config = AttentionConfig::new(8, 2, 64)
        .with_page_size(16)
        .with_causal_mask()
        .with_alibi();

    assert_eq!(
        config.pos_encoding,
        flashinfer_rs::types::PosEncodingMode::ALiBi
    );

    let mut handler = BatchPrefillHandler::new(device.clone(), config).unwrap();

    let batch_size = 2;
    let seq_lens = [16, 24];
    let total_tokens: usize = seq_lens.iter().sum();
    let page_size = 16;

    let mut qo_indptr = vec![0i32];
    let mut cumsum = 0i32;
    for &len in &seq_lens {
        cumsum += len as i32;
        qo_indptr.push(cumsum);
    }

    let pages_per_seq = 2;
    let page_table = create_test_page_table(batch_size, pages_per_seq);
    let kv_lengths: Vec<usize> = seq_lens.iter().map(|&x| x as usize).collect();

    let query = create_prefill_query(&device, total_tokens, 8, 64);
    let total_pages = batch_size * pages_per_seq;
    let kv_cache_k = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let kv_cache_v = create_kv_cache(&device, total_pages, 2, page_size, 64);
    let mut output = create_output(&device, total_tokens, 8, 64);

    // Plan with ALiBi config
    handler
        .plan::<half::f16>(&qo_indptr, &kv_lengths, &page_table, &stream)
        .unwrap();

    assert!(handler.is_planned());
    assert_eq!(handler.total_tokens(), Some(total_tokens));

    // Forward
    handler
        .forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)
        .unwrap();

    device.synchronize().unwrap();

    let output_host = device.dtoh_sync_copy(&output).unwrap();
    let has_nan = output_host.iter().any(|x| x.is_nan());
    let has_inf = output_host.iter().any(|x| x.is_infinite());

    assert!(!has_nan, "ALiBi prefill output contains NaN values");
    assert!(!has_inf, "ALiBi prefill output contains Inf values");

    // Verify output is not all zeros
    let all_zero = output_host.iter().all(|x| x.to_f32() == 0.0);
    assert!(!all_zero, "ALiBi prefill output is all zeros");
}

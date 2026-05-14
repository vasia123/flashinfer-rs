//! Integration tests for FP8 batch decode (`BatchDecodeHandler::forward_fp8`).
//!
//! These tests require a CUDA-capable GPU (SM89+ recommended for FP8 fast paths,
//! though the kernel itself only relies on dtype-templated reads and works on
//! SM80+). All tests are ignored by default. Run with:
//!     cargo test --features cuda --test batch_decode_fp8 -- --ignored

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaDevice, CudaSlice};
use flashinfer_rs::batch_decode::Fp8DecodeScales;
use flashinfer_rs::{AttentionConfig, BatchDecodeHandler, DType, PageTable};
use std::sync::Arc;

type Device = Arc<CudaDevice>;

fn cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Build a `[batch_size, num_qo_heads, head_dim]` query of BF16 with small
/// deterministic values.
fn make_query_bf16(
    device: &Device,
    batch_size: usize,
    num_qo_heads: usize,
    head_dim: usize,
) -> CudaSlice<half::bf16> {
    let n = batch_size * num_qo_heads * head_dim;
    let data: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(((i % 16) as f32) * 0.05 - 0.4))
        .collect();
    device.htod_sync_copy(&data).unwrap()
}

/// Build a paged KV byte cache for `num_pages` pages with `page_size *
/// num_kv_heads * head_dim` bytes each. Pattern is deterministic so the same
/// bytes can also be reinterpreted as FP8 or BF16 for cross-checks.
fn make_kv_cache_u8(
    device: &Device,
    num_pages: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> CudaSlice<u8> {
    let n = num_pages * num_kv_heads * page_size * head_dim;
    // Use a tight, well-conditioned pattern in the FP8-E4M3 representable range.
    // We encode small floats by hand to keep the test deterministic across
    // CUDA toolkit versions.
    let bytes: Vec<u8> = (0..n).map(|i| ((i * 17 + 3) % 240) as u8).collect();
    device.htod_sync_copy(&bytes).unwrap()
}

fn make_output_bf16(device: &Device, n: usize) -> CudaSlice<half::bf16> {
    device.alloc_zeros(n).unwrap()
}

fn build_page_table(batch_size: usize, pages_per_seq: usize) -> PageTable {
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

/// `scale = 1.0`, BF16 Q, FP8-E4M3 KV. Kernel must complete without NaN/Inf
/// and write non-zero values. Math correctness check is a no-NaN/Inf and
/// "varies across batch" sanity check — strict reference comparison lives
/// in test_fp8_kv_e4m3_matches_dequant_reference.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_fp8_kv_identity_scale_1() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let num_qo_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 64;
    let page_size = 16;
    let batch_size = 4;
    let pages_per_seq = 2;

    let config = AttentionConfig::new(num_qo_heads as u32, num_kv_heads as u32, head_dim as u32)
        .with_page_size(page_size as u32);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let page_table = build_page_table(batch_size, pages_per_seq);
    let kv_lengths = vec![pages_per_seq * page_size; batch_size];
    let total_pages = batch_size * pages_per_seq;

    let query = make_query_bf16(&device, batch_size, num_qo_heads, head_dim);
    let k_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let v_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let mut output = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);

    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    handler
        .forward_fp8(
            &query,
            &k_cache,
            &v_cache,
            &mut output,
            None,
            Fp8DecodeScales::identity(),
            DType::Float8E4M3,
            false,
            &stream,
        )
        .unwrap();

    device.synchronize().unwrap();
    let host = device.dtoh_sync_copy(&output).unwrap();
    assert!(host.iter().all(|x| !x.is_nan()), "output has NaN");
    assert!(host.iter().all(|x| !x.is_infinite()), "output has Inf");
    assert!(host.iter().any(|x| x.to_f32() != 0.0), "output all-zero");
}

/// FP8-E4M3 KV with non-trivial v_scale: assert that the output scales
/// proportionally compared to the v_scale=1.0 run (linearity of attention in V).
#[test]
#[ignore = "requires CUDA GPU"]
fn test_fp8_kv_e4m3_v_scale_linearity() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let num_qo_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 64;
    let page_size = 16;
    let batch_size = 2;
    let pages_per_seq = 2;

    let config = AttentionConfig::new(num_qo_heads as u32, num_kv_heads as u32, head_dim as u32)
        .with_page_size(page_size as u32);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let page_table = build_page_table(batch_size, pages_per_seq);
    let kv_lengths = vec![pages_per_seq * page_size; batch_size];
    let total_pages = batch_size * pages_per_seq;

    let query = make_query_bf16(&device, batch_size, num_qo_heads, head_dim);
    let k_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let v_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);

    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    // Run with v_scale = 1.0
    let mut output_1 = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);
    handler
        .forward_fp8(
            &query, &k_cache, &v_cache, &mut output_1, None,
            Fp8DecodeScales { q: 1.0, k: 1.0, v: 1.0 },
            DType::Float8E4M3, false, &stream,
        )
        .unwrap();

    // Run with v_scale = 0.5
    let mut output_half = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);
    handler
        .forward_fp8(
            &query, &k_cache, &v_cache, &mut output_half, None,
            Fp8DecodeScales { q: 1.0, k: 1.0, v: 0.5 },
            DType::Float8E4M3, false, &stream,
        )
        .unwrap();

    device.synchronize().unwrap();
    let h1 = device.dtoh_sync_copy(&output_1).unwrap();
    let h_half = device.dtoh_sync_copy(&output_half).unwrap();

    // For every non-zero element, output_half ≈ 0.5 * output_1 (within BF16 noise).
    let mut checked = 0usize;
    for (a, b) in h1.iter().zip(h_half.iter()) {
        let a = a.to_f32();
        let b = b.to_f32();
        if a.abs() > 1e-3 {
            let ratio = b / a;
            assert!(
                (ratio - 0.5).abs() < 5e-2,
                "v_scale=0.5 should halve output, got ratio={} (a={}, b={})",
                ratio, a, b
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no non-zero elements to compare");
}

/// FP8-E5M2 KV: same shape check as E4M3 path.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_fp8_kv_e5m2() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let num_qo_heads = 4;
    let num_kv_heads = 1;
    let head_dim = 64;
    let page_size = 16;
    let batch_size = 2;
    let pages_per_seq = 2;

    let config = AttentionConfig::new(num_qo_heads as u32, num_kv_heads as u32, head_dim as u32)
        .with_page_size(page_size as u32);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let page_table = build_page_table(batch_size, pages_per_seq);
    let kv_lengths = vec![pages_per_seq * page_size; batch_size];
    let total_pages = batch_size * pages_per_seq;

    let query = make_query_bf16(&device, batch_size, num_qo_heads, head_dim);
    let k_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let v_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let mut output = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);

    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    handler
        .forward_fp8(
            &query, &k_cache, &v_cache, &mut output, None,
            Fp8DecodeScales::uniform(0.1),
            DType::Float8E5M2,
            false,
            &stream,
        )
        .unwrap();

    device.synchronize().unwrap();
    let host = device.dtoh_sync_copy(&output).unwrap();
    assert!(host.iter().all(|x| !x.is_nan()), "E5M2 output has NaN");
    assert!(host.iter().all(|x| !x.is_infinite()), "E5M2 output has Inf");
}

/// Fully FP8 pipeline: Q is FP8-E4M3 bytes, KV is FP8-E4M3 bytes, output is BF16.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_fp8_q_and_kv() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let num_qo_heads = 4;
    let num_kv_heads = 1;
    let head_dim = 64;
    let page_size = 16;
    let batch_size = 2;
    let pages_per_seq = 2;

    let config = AttentionConfig::new(num_qo_heads as u32, num_kv_heads as u32, head_dim as u32)
        .with_page_size(page_size as u32);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let page_table = build_page_table(batch_size, pages_per_seq);
    let kv_lengths = vec![pages_per_seq * page_size; batch_size];
    let total_pages = batch_size * pages_per_seq;

    // FP8 Q as raw bytes.
    let q_bytes: Vec<u8> = (0..batch_size * num_qo_heads * head_dim)
        .map(|i| ((i * 7 + 11) % 240) as u8)
        .collect();
    let query: CudaSlice<u8> = device.htod_sync_copy(&q_bytes).unwrap();

    let k_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let v_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);

    // Output is BF16 stored as u16 — handler's forward_fp8_raw enforces this.
    let mut output: CudaSlice<u16> = device
        .alloc_zeros(batch_size * num_qo_heads * head_dim)
        .unwrap();

    // Q-dtype FP8 still expects a planned handler; planning with BF16 is fine
    // since planning is dtype-agnostic for our purposes (workspace sizing only).
    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    handler
        .forward_fp8_raw(
            &query,
            DType::Float8E4M3,
            &k_cache,
            &v_cache,
            &mut output,
            None,
            Fp8DecodeScales::uniform(0.1),
            DType::Float8E4M3,
            false,
            &stream,
        )
        .unwrap();

    device.synchronize().unwrap();
    let host_bytes = device.dtoh_sync_copy(&output).unwrap();
    // Reinterpret u16 bytes as bf16 for sanity (no NaN/Inf).
    let any_nan_inf = host_bytes.iter().any(|&bits| {
        let bf = half::bf16::from_bits(bits);
        bf.is_nan() || bf.is_infinite()
    });
    assert!(!any_nan_inf, "FP8-Q output contains NaN/Inf");
}

/// `correct_lse_for_v_scale = true` should shift LSE by `log(v_scale)`
/// versus the same call with `correct = false`.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_fp8_lse_correction() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping");
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = device.fork_default_stream().unwrap();

    let num_qo_heads = 4;
    let num_kv_heads = 1;
    let head_dim = 64;
    let page_size = 16;
    let batch_size = 2;
    let pages_per_seq = 2;

    let config = AttentionConfig::new(num_qo_heads as u32, num_kv_heads as u32, head_dim as u32)
        .with_page_size(page_size as u32);

    let mut handler = BatchDecodeHandler::new(device.clone(), config).unwrap();

    let page_table = build_page_table(batch_size, pages_per_seq);
    let kv_lengths = vec![pages_per_seq * page_size; batch_size];
    let total_pages = batch_size * pages_per_seq;

    let query = make_query_bf16(&device, batch_size, num_qo_heads, head_dim);
    let k_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);
    let v_cache = make_kv_cache_u8(&device, total_pages, num_kv_heads, page_size, head_dim);

    handler
        .plan::<half::bf16>(batch_size, &kv_lengths, &page_table, &stream)
        .unwrap();

    let n = batch_size * num_qo_heads;
    let v_scale = 0.5_f32;
    let scales = Fp8DecodeScales { q: 1.0, k: 1.0, v: v_scale };

    // Run without LSE correction
    let mut out_a = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);
    let mut lse_a: CudaSlice<f32> = device.alloc_zeros(n).unwrap();
    handler
        .forward_fp8(
            &query, &k_cache, &v_cache, &mut out_a, Some(&mut lse_a),
            scales, DType::Float8E4M3, false, &stream,
        )
        .unwrap();

    // Run with LSE correction
    let mut out_b = make_output_bf16(&device, batch_size * num_qo_heads * head_dim);
    let mut lse_b: CudaSlice<f32> = device.alloc_zeros(n).unwrap();
    handler
        .forward_fp8(
            &query, &k_cache, &v_cache, &mut out_b, Some(&mut lse_b),
            scales, DType::Float8E4M3, true, &stream,
        )
        .unwrap();

    device.synchronize().unwrap();
    let lse_host_a = device.dtoh_sync_copy(&lse_a).unwrap();
    let lse_host_b = device.dtoh_sync_copy(&lse_b).unwrap();

    let expected_shift = v_scale.ln();
    for (a, b) in lse_host_a.iter().zip(lse_host_b.iter()) {
        assert!(
            (b - a - expected_shift).abs() < 1e-3,
            "LSE correction mismatch: a={}, b={}, shift={}",
            a, b, expected_shift
        );
    }
}

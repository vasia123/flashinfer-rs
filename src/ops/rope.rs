//! Rotary Position Embedding (RoPE) operations.
//!
//! This module provides RoPE operations for LLM inference.
//! RoPE is a position encoding method that rotates query and key vectors
//! to encode positional information.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::config::RoPEConfig;
#[cfg(feature = "cuda")]
use crate::ffi::FlashInferRoPEConfig;
use crate::types::GpuFloat;
use crate::Result;

/// Convert RoPEConfig to FFI config.
#[cfg(feature = "cuda")]
fn to_ffi_config(config: &RoPEConfig) -> FlashInferRoPEConfig {
    FlashInferRoPEConfig {
        rotary_dim: config.rotary_dim,
        interleave: if config.interleave { 1 } else { 0 },
        scale: config.scale,
        theta: config.theta,
    }
}

/// Apply RoPE to query and key tensors.
///
/// # Arguments
///
/// * `q` - Query tensor of shape `[total_tokens, num_qo_heads, head_dim]`
/// * `k` - Key tensor of shape `[total_tokens, num_kv_heads, head_dim]`
/// * `q_out` - Output query tensor
/// * `k_out` - Output key tensor
/// * `indptr` - Cumulative token counts `[batch_size + 1]`
/// * `offsets` - Position offsets for each sequence `[batch_size]`
/// * `batch_size` - Number of sequences
/// * `num_qo_heads` - Number of query heads
/// * `num_kv_heads` - Number of key heads
/// * `head_dim` - Head dimension
/// * `config` - RoPE configuration
/// * `stream` - CUDA stream
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::ops::rope::apply_rope;
/// use flashinfer_rs::config::RoPEConfig;
///
/// let config = RoPEConfig::llama(128);
/// apply_rope::<half::f16>(
///     &q, &k, &mut q_out, &mut k_out,
///     &indptr, &offsets, batch_size,
///     num_qo_heads, num_kv_heads, head_dim,
///     &config, &stream
/// )?;
/// ```
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn apply_rope<T: GpuFloat>(
    q: &CudaSlice<T>,
    k: &CudaSlice<T>,
    q_out: &mut CudaSlice<T>,
    k_out: &mut CudaSlice<T>,
    indptr: &CudaSlice<i32>,
    offsets: &CudaSlice<i32>,
    batch_size: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    config: &RoPEConfig,
    stream: &CudaStream,
) -> Result<()> {
    let ffi_config = to_ffi_config(config);

    unsafe {
        crate::ffi::apply_rope(
            *q.device_ptr() as *const std::ffi::c_void,
            *k.device_ptr() as *const std::ffi::c_void,
            *q_out.device_ptr() as *mut std::ffi::c_void,
            *k_out.device_ptr() as *mut std::ffi::c_void,
            *indptr.device_ptr() as *const i32,
            *offsets.device_ptr() as *const i32,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            &ffi_config,
            T::DTYPE.into(),
            stream.stream as *mut std::ffi::c_void,
        )
    }
}

/// Apply RoPE in-place to query and key tensors.
///
/// More memory efficient version that modifies tensors in place.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_inplace<T: GpuFloat>(
    q: &mut CudaSlice<T>,
    k: &mut CudaSlice<T>,
    indptr: &CudaSlice<i32>,
    offsets: &CudaSlice<i32>,
    batch_size: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    config: &RoPEConfig,
    stream: &CudaStream,
) -> Result<()> {
    let ffi_config = to_ffi_config(config);

    unsafe {
        crate::ffi::apply_rope_inplace(
            *q.device_ptr() as *mut std::ffi::c_void,
            *k.device_ptr() as *mut std::ffi::c_void,
            *indptr.device_ptr() as *const i32,
            *offsets.device_ptr() as *const i32,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            &ffi_config,
            T::DTYPE.into(),
            stream.stream as *mut std::ffi::c_void,
        )
    }
}

/// Apply RoPE with explicit position IDs.
///
/// This variant accepts explicit position IDs for each token
/// rather than computing them from indptr and offsets.
///
/// # Arguments
///
/// * `q` - Query tensor of shape `[total_tokens, num_qo_heads, head_dim]`
/// * `k` - Key tensor of shape `[total_tokens, num_kv_heads, head_dim]`
/// * `q_out` - Output query tensor
/// * `k_out` - Output key tensor
/// * `pos_ids` - Position IDs for each token `[total_tokens]`
/// * `total_tokens` - Number of tokens
/// * `num_qo_heads` - Number of query heads
/// * `num_kv_heads` - Number of key heads
/// * `head_dim` - Head dimension
/// * `config` - RoPE configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_pos_ids<T: GpuFloat>(
    q: &CudaSlice<T>,
    k: &CudaSlice<T>,
    q_out: &mut CudaSlice<T>,
    k_out: &mut CudaSlice<T>,
    pos_ids: &CudaSlice<i32>,
    total_tokens: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    config: &RoPEConfig,
    stream: &CudaStream,
) -> Result<()> {
    let ffi_config = to_ffi_config(config);

    unsafe {
        crate::ffi::apply_rope_pos_ids(
            *q.device_ptr() as *const std::ffi::c_void,
            *k.device_ptr() as *const std::ffi::c_void,
            *q_out.device_ptr() as *mut std::ffi::c_void,
            *k_out.device_ptr() as *mut std::ffi::c_void,
            *pos_ids.device_ptr() as *const i32,
            total_tokens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            &ffi_config,
            T::DTYPE.into(),
            stream.stream as *mut std::ffi::c_void,
        )
    }
}

/// Apply RoPE with precomputed cos/sin cache.
///
/// This variant uses precomputed cosine and sine values
/// for better performance when the same positions are used repeatedly.
///
/// # Arguments
///
/// * `q` - Query tensor
/// * `k` - Key tensor
/// * `q_out` - Output query tensor
/// * `k_out` - Output key tensor
/// * `cos_sin_cache` - Precomputed cos/sin values `[max_seq_len, rotary_dim]` (interleaved)
/// * `pos_ids` - Position IDs for each token
/// * `total_tokens` - Number of tokens
/// * `num_qo_heads` - Number of query heads
/// * `num_kv_heads` - Number of key heads
/// * `head_dim` - Head dimension
/// * `config` - RoPE configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_cached<T: GpuFloat>(
    q: &CudaSlice<T>,
    k: &CudaSlice<T>,
    q_out: &mut CudaSlice<T>,
    k_out: &mut CudaSlice<T>,
    cos_sin_cache: &CudaSlice<f32>,
    pos_ids: &CudaSlice<i32>,
    total_tokens: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    config: &RoPEConfig,
    stream: &CudaStream,
) -> Result<()> {
    let ffi_config = to_ffi_config(config);

    unsafe {
        crate::ffi::apply_rope_with_cos_sin_cache(
            *q.device_ptr() as *const std::ffi::c_void,
            *k.device_ptr() as *const std::ffi::c_void,
            *q_out.device_ptr() as *mut std::ffi::c_void,
            *k_out.device_ptr() as *mut std::ffi::c_void,
            *cos_sin_cache.device_ptr() as *const std::ffi::c_void,
            *pos_ids.device_ptr() as *const i32,
            total_tokens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            &ffi_config,
            T::DTYPE.into(),
            stream.stream as *mut std::ffi::c_void,
        )
    }
}

/// Precompute RoPE cos/sin cache.
///
/// Generates the cosine and sine tables for RoPE that can be
/// reused across multiple forward passes.
///
/// # Arguments
///
/// * `device` - CUDA device
/// * `max_seq_len` - Maximum sequence length to support
/// * `config` - RoPE configuration
///
/// # Returns
///
/// Combined cos/sin cache tensor `[max_seq_len, rotary_dim]` (interleaved).
#[cfg(feature = "cuda")]
pub fn precompute_rope_cache(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    max_seq_len: u32,
    config: &RoPEConfig,
) -> Result<CudaSlice<f32>> {
    // NOTE: This is a CPU-side computation that could be GPU-accelerated.
    // For now, we compute on CPU and upload to GPU.
    let rotary_dim = config.rotary_dim as usize;
    let cache_size = max_seq_len as usize * rotary_dim;

    let mut cache = vec![0.0f32; cache_size];

    for pos in 0..max_seq_len as usize {
        for i in 0..rotary_dim / 2 {
            let freq = 1.0 / config.theta.powf(2.0 * i as f32 / rotary_dim as f32);
            let angle = (pos as f32) * freq / config.scale;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            // Interleaved layout: [cos_0, sin_0, cos_1, sin_1, ...]
            cache[pos * rotary_dim + 2 * i] = cos_val;
            cache[pos * rotary_dim + 2 * i + 1] = sin_val;
        }
    }

    device
        .htod_sync_copy(&cache)
        .map_err(|e| crate::FlashInferError::cuda(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RoPEConfig;

    #[test]
    fn test_rope_config_llama() {
        let config = RoPEConfig::llama(128);
        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.theta, 10000.0);
        assert!(!config.interleave);
    }

    #[test]
    fn test_rope_config_llama3_1() {
        let config = RoPEConfig::llama3_1(128, 500000.0);
        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.theta, 500000.0);
    }

    #[test]
    fn test_rope_config_validation() {
        let valid = RoPEConfig::new(128);
        assert!(valid.validate().is_ok());

        let invalid = RoPEConfig::new(127); // Must be even
        assert!(invalid.validate().is_err());
    }
}

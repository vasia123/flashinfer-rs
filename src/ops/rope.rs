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

// =============================================================================
// Fused RoPE + Quantize + Append API
// =============================================================================

/// Configuration for fused RoPE + Quantize + Append operation.
///
/// This struct configures the fused operation that:
/// 1. Applies RoPE to Q_rope and K_rope tensors
/// 2. Quantizes all outputs to FP8
/// 3. Appends K/V to paged cache
///
/// **Requires SM89+ (Ada Lovelace, Hopper)**
///
/// # Examples
///
/// ```ignore
/// // Standard GQA/MHA config (no separate nope dimension)
/// let config = RopeQuantAppendConfig::gqa(
///     32,   // num_qo_heads
///     8,    // num_kv_heads
///     128,  // head_dim (= rope_dim)
///     16,   // page_size
/// );
///
/// // MLA config (separate rope and nope dimensions)
/// let config = RopeQuantAppendConfig::mla(
///     32,   // num_qo_heads
///     1,    // num_kv_heads (MLA typically uses 1)
///     64,   // rope_dim
///     512,  // kv_lora_rank (= no_rope_dim)
///     16,   // page_size
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RopeQuantAppendConfig {
    /// Number of query/output heads
    pub num_qo_heads: u32,
    /// Number of key/value heads
    pub num_kv_heads: u32,
    /// Dimension for RoPE (rotary part)
    pub rope_dim: u32,
    /// Dimension for non-RoPE part (for MLA). Set to 0 for GQA/MHA.
    pub no_rope_dim: u32,
    /// Tokens per page in paged KV cache
    pub page_size: u32,
    /// Quantization scale for Q outputs
    pub quant_scale_q: f32,
    /// Quantization scale for K/V outputs
    pub quant_scale_kv: f32,
    /// Use interleaved RoPE format
    pub interleave: bool,
    /// Enable PDL (SM90+ only)
    pub enable_pdl: bool,
    /// KV cache memory layout
    pub kv_layout: crate::types::KVLayout,
    /// Output FP8 format (E4M3 or E5M2)
    pub output_dtype: crate::types::DType,
}

impl RopeQuantAppendConfig {
    /// Create configuration for GQA/MHA (no separate rope/nope dimensions).
    ///
    /// In this mode, head_dim equals rope_dim and no_rope_dim is 0.
    pub fn gqa(num_qo_heads: u32, num_kv_heads: u32, head_dim: u32, page_size: u32) -> Self {
        Self {
            num_qo_heads,
            num_kv_heads,
            rope_dim: head_dim,
            no_rope_dim: 0,
            page_size,
            quant_scale_q: 1.0,
            quant_scale_kv: 1.0,
            interleave: false,
            enable_pdl: false,
            kv_layout: crate::types::KVLayout::NHD,
            output_dtype: crate::types::DType::Float8E4M3,
        }
    }

    /// Create configuration for MLA (Multi-head Latent Attention).
    ///
    /// MLA uses separate rope and nope dimensions:
    /// - rope_dim: dimension for rotary position encoding
    /// - kv_lora_rank: dimension for the latent space (no_rope_dim)
    pub fn mla(
        num_qo_heads: u32,
        num_kv_heads: u32,
        rope_dim: u32,
        kv_lora_rank: u32,
        page_size: u32,
    ) -> Self {
        Self {
            num_qo_heads,
            num_kv_heads,
            rope_dim,
            no_rope_dim: kv_lora_rank,
            page_size,
            quant_scale_q: 1.0,
            quant_scale_kv: 1.0,
            interleave: false,
            enable_pdl: false,
            kv_layout: crate::types::KVLayout::NHD,
            output_dtype: crate::types::DType::Float8E4M3,
        }
    }

    /// Set quantization scale for Q outputs.
    pub fn with_q_scale(mut self, scale: f32) -> Self {
        self.quant_scale_q = scale;
        self
    }

    /// Set quantization scale for K/V outputs.
    pub fn with_kv_scale(mut self, scale: f32) -> Self {
        self.quant_scale_kv = scale;
        self
    }

    /// Enable interleaved RoPE format.
    pub fn with_interleave(mut self, interleave: bool) -> Self {
        self.interleave = interleave;
        self
    }

    /// Enable PDL (SM90+ only).
    pub fn with_pdl(mut self, enable: bool) -> Self {
        self.enable_pdl = enable;
        self
    }

    /// Set KV cache layout.
    pub fn with_kv_layout(mut self, layout: crate::types::KVLayout) -> Self {
        self.kv_layout = layout;
        self
    }

    /// Set output FP8 format.
    pub fn with_output_dtype(mut self, dtype: crate::types::DType) -> Self {
        self.output_dtype = dtype;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.rope_dim == 0 {
            return Err(crate::FlashInferError::invalid_config(
                "rope_dim must be > 0",
            ));
        }
        if self.rope_dim % 2 != 0 {
            return Err(crate::FlashInferError::invalid_config(
                "rope_dim must be even",
            ));
        }
        if self.page_size == 0 {
            return Err(crate::FlashInferError::invalid_config(
                "page_size must be > 0",
            ));
        }
        if self.num_qo_heads == 0 || self.num_kv_heads == 0 {
            return Err(crate::FlashInferError::invalid_config(
                "num_qo_heads and num_kv_heads must be > 0",
            ));
        }
        if self.num_qo_heads % self.num_kv_heads != 0 {
            return Err(crate::FlashInferError::invalid_config(
                "num_qo_heads must be divisible by num_kv_heads",
            ));
        }
        if self.output_dtype != crate::types::DType::Float8E4M3
            && self.output_dtype != crate::types::DType::Float8E5M2
        {
            return Err(crate::FlashInferError::invalid_config(
                "output_dtype must be Float8E4M3 or Float8E5M2",
            ));
        }
        Ok(())
    }
}

/// Convert RopeQuantAppendConfig to FFI config.
#[cfg(feature = "cuda")]
fn to_ffi_rope_quant_config<T: GpuFloat>(
    config: &RopeQuantAppendConfig,
) -> crate::ffi::FlashInferRopeQuantAppendConfig {
    use crate::ffi::{DType, KVLayout};

    crate::ffi::FlashInferRopeQuantAppendConfig {
        num_qo_heads: config.num_qo_heads,
        num_kv_heads: config.num_kv_heads,
        rope_dim: config.rope_dim,
        no_rope_dim: config.no_rope_dim,
        page_size: config.page_size,
        quant_scale_q: config.quant_scale_q,
        quant_scale_kv: config.quant_scale_kv,
        interleave: if config.interleave { 1 } else { 0 },
        enable_pdl: if config.enable_pdl { 1 } else { 0 },
        input_dtype: DType::from(T::DTYPE).into(),
        output_dtype: DType::from(config.output_dtype).into(),
        kv_layout: KVLayout::from(config.kv_layout).into(),
    }
}

/// Fused RoPE + Quantize + Append to paged KV cache.
///
/// This operation combines:
/// 1. Apply RoPE to Q_rope and K_rope tensors
/// 2. Quantize all outputs to FP8
/// 3. Append K/V to paged cache
///
/// **Requires SM89+ (Ada Lovelace, Hopper)**
///
/// # Arguments
///
/// * `q_rope_in` - Query RoPE tensor `[nnz, num_qo_heads, rope_dim]`
/// * `k_rope_in` - Key RoPE tensor `[nnz, num_kv_heads, rope_dim]`
/// * `q_nope_in` - Query non-RoPE tensor `[nnz, num_qo_heads, no_rope_dim]` (empty for GQA)
/// * `k_nope_in` - Key non-RoPE tensor `[nnz, num_kv_heads, no_rope_dim]` (empty for GQA)
/// * `v_in` - Value tensor `[nnz, num_kv_heads, head_dim]`
/// * `q_rope_out` - Output query RoPE tensor (FP8) `[nnz, num_qo_heads, rope_dim]`
/// * `q_nope_out` - Output query non-RoPE tensor (FP8) (empty for GQA)
/// * `k_cache` - Paged K cache (FP8)
/// * `v_cache` - Paged V cache (FP8)
/// * `kv_indptr` - Page offsets `[batch_size + 1]`
/// * `kv_indices` - Page indices `[total_pages]`
/// * `kv_last_page_len` - Tokens in last page before append `[batch_size]`
/// * `batch_indices` - Batch index for each token `[nnz]`
/// * `positions` - Position for each token within its sequence `[nnz]`
/// * `cos_sin_cache` - Precomputed cos/sin cache `[max_pos, rope_dim]`
/// * `pos_ids` - Position IDs for RoPE `[nnz]`
/// * `nnz` - Total number of tokens
/// * `config` - Configuration for the operation
/// * `stream` - CUDA stream
///
/// # Errors
///
/// Returns `FLASHINFER_UNSUPPORTED` on GPUs with SM < 89 (pre-Ada Lovelace).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn rope_quant_append_paged_kv_cache<T: GpuFloat>(
    q_rope_in: &CudaSlice<T>,
    k_rope_in: &CudaSlice<T>,
    q_nope_in: Option<&CudaSlice<T>>,
    k_nope_in: Option<&CudaSlice<T>>,
    v_in: &CudaSlice<T>,
    q_rope_out: &mut CudaSlice<u8>,
    q_nope_out: Option<&mut CudaSlice<u8>>,
    k_cache: &mut CudaSlice<u8>,
    v_cache: &mut CudaSlice<u8>,
    kv_indptr: &CudaSlice<i32>,
    kv_indices: &CudaSlice<i32>,
    kv_last_page_len: &CudaSlice<i32>,
    batch_indices: &CudaSlice<i32>,
    positions: &CudaSlice<i32>,
    cos_sin_cache: &CudaSlice<f32>,
    pos_ids: &CudaSlice<i32>,
    nnz: u32,
    config: &RopeQuantAppendConfig,
    stream: &CudaStream,
) -> Result<()> {
    config.validate()?;

    let ffi_config = to_ffi_rope_quant_config::<T>(config);

    // Handle optional nope tensors
    let q_nope_ptr = q_nope_in
        .map(|s| *s.device_ptr() as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let k_nope_ptr = k_nope_in
        .map(|s| *s.device_ptr() as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let q_nope_out_ptr = q_nope_out
        .map(|s| *s.device_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        crate::ffi::rope_quant_append_paged_kv_cache(
            *q_rope_in.device_ptr() as *const std::ffi::c_void,
            *k_rope_in.device_ptr() as *const std::ffi::c_void,
            q_nope_ptr,
            k_nope_ptr,
            *v_in.device_ptr() as *const std::ffi::c_void,
            *q_rope_out.device_ptr() as *mut std::ffi::c_void,
            q_nope_out_ptr,
            *k_cache.device_ptr() as *mut std::ffi::c_void,
            *v_cache.device_ptr() as *mut std::ffi::c_void,
            *kv_indptr.device_ptr() as *const i32,
            *kv_indices.device_ptr() as *const i32,
            *kv_last_page_len.device_ptr() as *const i32,
            *batch_indices.device_ptr() as *const i32,
            *positions.device_ptr() as *const i32,
            *cos_sin_cache.device_ptr() as *const f32,
            *pos_ids.device_ptr() as *const i32,
            nnz,
            &ffi_config,
            stream.stream as *mut std::ffi::c_void,
        )
    }
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

    #[test]
    fn test_rope_quant_append_config_gqa() {
        let config = RopeQuantAppendConfig::gqa(32, 8, 128, 16);
        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.rope_dim, 128);
        assert_eq!(config.no_rope_dim, 0);
        assert_eq!(config.page_size, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rope_quant_append_config_mla() {
        let config = RopeQuantAppendConfig::mla(32, 1, 64, 512, 16);
        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 1);
        assert_eq!(config.rope_dim, 64);
        assert_eq!(config.no_rope_dim, 512);
        assert_eq!(config.page_size, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rope_quant_append_config_builder() {
        let config = RopeQuantAppendConfig::gqa(32, 8, 128, 16)
            .with_q_scale(0.5)
            .with_kv_scale(0.25)
            .with_interleave(true)
            .with_output_dtype(crate::types::DType::Float8E5M2);

        assert_eq!(config.quant_scale_q, 0.5);
        assert_eq!(config.quant_scale_kv, 0.25);
        assert!(config.interleave);
        assert_eq!(config.output_dtype, crate::types::DType::Float8E5M2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rope_quant_append_config_validation_errors() {
        // Invalid: rope_dim = 0
        let mut config = RopeQuantAppendConfig::gqa(32, 8, 0, 16);
        assert!(config.validate().is_err());

        // Invalid: rope_dim not even
        config = RopeQuantAppendConfig::gqa(32, 8, 127, 16);
        assert!(config.validate().is_err());

        // Invalid: page_size = 0
        config = RopeQuantAppendConfig::gqa(32, 8, 128, 0);
        assert!(config.validate().is_err());

        // Invalid: num_qo_heads not divisible by num_kv_heads
        config = RopeQuantAppendConfig::gqa(32, 7, 128, 16);
        assert!(config.validate().is_err());

        // Invalid: output_dtype not FP8
        config = RopeQuantAppendConfig::gqa(32, 8, 128, 16)
            .with_output_dtype(crate::types::DType::Float16);
        assert!(config.validate().is_err());
    }
}

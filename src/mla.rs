//! MLA (Multi-head Latent Attention) support for DeepSeek models.
//!
//! DeepSeek v2/v3 use MLA (Multi-head Latent Attention), which has a different
//! architecture from standard GQA/MHA:
//!
//! - **Compressed KV cache**: Instead of storing K and V separately per head,
//!   MLA uses a compressed KV representation (dimension 512).
//! - **K position embedding**: RoPE is applied to a separate K-PE tensor
//!   (dimension 64), not to the full K.
//! - **Matrix absorption**: W_UQ is absorbed into W_UK, W_UV into W_O.
//! - **Softmax scale**: 1/sqrt(nope_dim + rope_dim) = 1/sqrt(192), not 1/sqrt(512).
//!
//! # Fixed Dimensions (DeepSeek v2/v3)
//!
//! | Parameter | Value | Description |
//! |-----------|-------|-------------|
//! | qk_nope_head_dim | 128 | Non-RoPE part of Q/K per head |
//! | kv_lora_rank | 512 | Compressed KV dimension |
//! | qk_rope_head_dim | 64 | RoPE part of Q/K |
//! | num_heads | 128 | Number of attention heads |
//!
//! # Example
//!
//! ```ignore
//! use flashinfer_rs::mla::{MLAConfig, MLAHandler};
//!
//! // Create DeepSeek MLA configuration
//! let config = MLAConfig::deepseek();
//! assert_eq!(config.num_heads, 128);
//! assert_eq!(config.sm_scale(), 1.0 / 192.0_f32.sqrt());
//!
//! // Create handler for MLA attention
//! let handler = MLAHandler::new(config, batch_size, max_seq_len, page_size)?;
//! ```

use crate::{FlashInferError, Result};

/// DeepSeek MLA fixed dimensions.
pub mod dims {
    /// Number of attention heads for DeepSeek MLA.
    pub const NUM_HEADS: u32 = 128;

    /// Compressed KV dimension (kv_lora_rank).
    pub const HEAD_DIM_CKV: u32 = 512;

    /// K position embedding dimension.
    pub const HEAD_DIM_KPE: u32 = 64;

    /// Q/K non-RoPE dimension per head.
    pub const QK_NOPE_HEAD_DIM: u32 = 128;

    /// Q/K RoPE dimension.
    pub const QK_ROPE_HEAD_DIM: u32 = 64;

    /// Total K head dimension (nope + rope).
    pub const K_HEAD_DIM: u32 = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
}

/// Configuration for MLA attention.
#[derive(Debug, Clone)]
pub struct MLAConfig {
    /// Number of attention heads (128 for DeepSeek).
    pub num_heads: u32,

    /// Compressed KV dimension (512 for DeepSeek).
    pub head_dim_ckv: u32,

    /// K position embedding dimension (64 for DeepSeek).
    pub head_dim_kpe: u32,

    /// Page size for paged KV cache.
    pub page_size: u32,
}

impl Default for MLAConfig {
    fn default() -> Self {
        Self::deepseek()
    }
}

impl MLAConfig {
    /// Create DeepSeek v2/v3 default configuration.
    pub fn deepseek() -> Self {
        Self {
            num_heads: dims::NUM_HEADS,
            head_dim_ckv: dims::HEAD_DIM_CKV,
            head_dim_kpe: dims::HEAD_DIM_KPE,
            page_size: 16,
        }
    }

    /// Create configuration with custom page size.
    pub fn deepseek_with_page_size(page_size: u32) -> Self {
        Self {
            page_size,
            ..Self::deepseek()
        }
    }

    /// Softmax scale for MLA attention.
    ///
    /// For DeepSeek MLA, this is 1/sqrt(nope_dim + rope_dim) = 1/sqrt(128 + 64) = 1/sqrt(192).
    pub fn sm_scale(&self) -> f32 {
        // nope_dim = 128, rope_dim = 64
        1.0 / (192.0_f32).sqrt()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.head_dim_ckv != dims::HEAD_DIM_CKV {
            return Err(FlashInferError::invalid_config(format!(
                "MLA requires head_dim_ckv={}, got {}",
                dims::HEAD_DIM_CKV,
                self.head_dim_ckv
            )));
        }

        if self.head_dim_kpe != dims::HEAD_DIM_KPE {
            return Err(FlashInferError::invalid_config(format!(
                "MLA requires head_dim_kpe={}, got {}",
                dims::HEAD_DIM_KPE,
                self.head_dim_kpe
            )));
        }

        if self.num_heads != dims::NUM_HEADS {
            return Err(FlashInferError::invalid_config(format!(
                "MLA requires num_heads={}, got {}",
                dims::NUM_HEADS,
                self.num_heads
            )));
        }

        if self.page_size == 0 || !self.page_size.is_power_of_two() {
            return Err(FlashInferError::invalid_config(format!(
                "page_size must be a power of 2, got {}",
                self.page_size
            )));
        }

        Ok(())
    }

    /// Calculate required memory for paged caches.
    ///
    /// Returns (ckv_cache_bytes, kpe_cache_bytes) for the given number of pages.
    pub fn cache_memory_bytes(&self, num_pages: u32, dtype_size: usize) -> (usize, usize) {
        // ckv_cache: [num_pages, page_size, head_dim_ckv]
        let ckv_bytes = (num_pages as usize)
            * (self.page_size as usize)
            * (self.head_dim_ckv as usize)
            * dtype_size;

        // kpe_cache: [num_pages, page_size, head_dim_kpe]
        let kpe_bytes = (num_pages as usize)
            * (self.page_size as usize)
            * (self.head_dim_kpe as usize)
            * dtype_size;

        (ckv_bytes, kpe_bytes)
    }
}

/// Shape information for MLA Q tensors.
#[derive(Debug, Clone)]
pub struct MLAQueryShape {
    /// Number of tokens (nnz).
    pub num_tokens: u32,

    /// Number of heads.
    pub num_heads: u32,

    /// Q-nope dimension (512 for absorbed matrix).
    pub dim_nope: u32,

    /// Q-PE dimension (64 for RoPE).
    pub dim_pe: u32,
}

impl MLAQueryShape {
    /// Create shape for DeepSeek MLA queries.
    pub fn deepseek(num_tokens: u32) -> Self {
        Self {
            num_tokens,
            num_heads: dims::NUM_HEADS,
            dim_nope: dims::HEAD_DIM_CKV, // Absorbed Q-nope is 512
            dim_pe: dims::HEAD_DIM_KPE,
        }
    }

    /// Total elements in q_nope tensor: [num_tokens, num_heads, dim_nope]
    pub fn q_nope_elements(&self) -> usize {
        (self.num_tokens as usize) * (self.num_heads as usize) * (self.dim_nope as usize)
    }

    /// Total elements in q_pe tensor: [num_tokens, num_heads, dim_pe]
    pub fn q_pe_elements(&self) -> usize {
        (self.num_tokens as usize) * (self.num_heads as usize) * (self.dim_pe as usize)
    }

    /// Total elements in output tensor: [num_tokens, num_heads, dim_nope]
    pub fn output_elements(&self) -> usize {
        self.q_nope_elements() // Same as q_nope
    }
}

/// Shape information for MLA K tensors (before concatenation).
#[derive(Debug, Clone)]
pub struct MLAKeyShape {
    /// Number of tokens.
    pub num_tokens: u32,

    /// Number of heads.
    pub num_heads: u32,

    /// K-nope dimension per head (128).
    pub dim_nope: u32,

    /// K-rope dimension (64, shared across heads).
    pub dim_rope: u32,
}

impl MLAKeyShape {
    /// Create shape for DeepSeek MLA keys.
    pub fn deepseek(num_tokens: u32) -> Self {
        Self {
            num_tokens,
            num_heads: dims::NUM_HEADS,
            dim_nope: dims::QK_NOPE_HEAD_DIM,
            dim_rope: dims::QK_ROPE_HEAD_DIM,
        }
    }

    /// Total elements in k_nope tensor: [num_tokens, num_heads, dim_nope]
    pub fn k_nope_elements(&self) -> usize {
        (self.num_tokens as usize) * (self.num_heads as usize) * (self.dim_nope as usize)
    }

    /// Total elements in k_rope tensor: [num_tokens, 1, dim_rope] (shared)
    pub fn k_rope_elements(&self) -> usize {
        (self.num_tokens as usize) * (self.dim_rope as usize)
    }

    /// Total elements in concatenated K tensor: [num_tokens, num_heads, dim_nope + dim_rope]
    pub fn k_concat_elements(&self) -> usize {
        (self.num_tokens as usize)
            * (self.num_heads as usize)
            * ((self.dim_nope + self.dim_rope) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_config_deepseek() {
        let config = MLAConfig::deepseek();
        assert_eq!(config.num_heads, 128);
        assert_eq!(config.head_dim_ckv, 512);
        assert_eq!(config.head_dim_kpe, 64);
        assert_eq!(config.page_size, 16);
    }

    #[test]
    fn test_mla_config_sm_scale() {
        let config = MLAConfig::deepseek();
        let expected = 1.0 / (192.0_f32).sqrt();
        assert!((config.sm_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mla_config_validate() {
        // Valid config
        let config = MLAConfig::deepseek();
        assert!(config.validate().is_ok());

        // Invalid head_dim_ckv
        let mut bad_config = MLAConfig::deepseek();
        bad_config.head_dim_ckv = 256;
        assert!(bad_config.validate().is_err());

        // Invalid page_size (not power of 2)
        let mut bad_config = MLAConfig::deepseek();
        bad_config.page_size = 17;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_mla_config_cache_memory() {
        let config = MLAConfig::deepseek();
        let (ckv_bytes, kpe_bytes) = config.cache_memory_bytes(100, 2); // 2 bytes per element (fp16)

        // ckv: 100 pages * 16 page_size * 512 head_dim * 2 bytes
        assert_eq!(ckv_bytes, 100 * 16 * 512 * 2);

        // kpe: 100 pages * 16 page_size * 64 head_dim * 2 bytes
        assert_eq!(kpe_bytes, 100 * 16 * 64 * 2);
    }

    #[test]
    fn test_mla_query_shape() {
        let shape = MLAQueryShape::deepseek(32);
        assert_eq!(shape.num_tokens, 32);
        assert_eq!(shape.num_heads, 128);
        assert_eq!(shape.dim_nope, 512);
        assert_eq!(shape.dim_pe, 64);

        // q_nope: [32, 128, 512]
        assert_eq!(shape.q_nope_elements(), 32 * 128 * 512);

        // q_pe: [32, 128, 64]
        assert_eq!(shape.q_pe_elements(), 32 * 128 * 64);
    }

    #[test]
    fn test_mla_key_shape() {
        let shape = MLAKeyShape::deepseek(32);
        assert_eq!(shape.num_tokens, 32);
        assert_eq!(shape.num_heads, 128);
        assert_eq!(shape.dim_nope, 128);
        assert_eq!(shape.dim_rope, 64);

        // k_nope: [32, 128, 128]
        assert_eq!(shape.k_nope_elements(), 32 * 128 * 128);

        // k_rope: [32, 64] (shared across heads)
        assert_eq!(shape.k_rope_elements(), 32 * 64);

        // k_concat: [32, 128, 192]
        assert_eq!(shape.k_concat_elements(), 32 * 128 * 192);
    }

    #[test]
    fn test_dims_constants() {
        assert_eq!(dims::NUM_HEADS, 128);
        assert_eq!(dims::HEAD_DIM_CKV, 512);
        assert_eq!(dims::HEAD_DIM_KPE, 64);
        assert_eq!(dims::QK_NOPE_HEAD_DIM, 128);
        assert_eq!(dims::QK_ROPE_HEAD_DIM, 64);
        assert_eq!(dims::K_HEAD_DIM, 192);
    }
}

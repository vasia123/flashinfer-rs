//! FlashInfer Rust Bindings
//!
//! High-performance attention kernels for LLM inference serving.
//!
//! FlashInfer provides optimized CUDA kernels for:
//! - Paged KV cache attention (decode and prefill)
//! - Variable-length batch processing
//! - Multi-head and Grouped-Query Attention (GQA)
//! - RMSNorm, LayerNorm, and other normalization operations
//! - RoPE (Rotary Position Embedding)
//! - Top-K/P sampling
//!
//! # Features
//!
//! - `cuda` - Enable CUDA support via cudarc
//! - `cuda-11` / `cuda-12` - CUDA version selection
//! - `sm80` / `sm90` - Target GPU architecture
//!
//! # Example
//!
//! ```ignore
//! use flashinfer_rs::{AttentionConfig, BatchDecodeHandler, PagedKVCache};
//!
//! // Configure attention
//! let config = AttentionConfig::new(32, 8, 128)
//!     .with_causal_mask()
//!     .with_rope(1.0, 10000.0);
//!
//! // Create handler
//! let handler = BatchDecodeHandler::new(&device, config)?;
//!
//! // Run attention
//! handler.run(&query, &kv_cache, &mut output)?;
//! ```

pub mod config;
mod error;
pub mod page_table;
pub mod types;
pub mod workspace;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod ffi;

#[cfg(feature = "cuda")]
pub mod batch_decode;

#[cfg(feature = "cuda")]
pub mod batch_prefill;

#[cfg(feature = "cuda")]
pub mod ops;

pub mod mla;
pub mod page;

// Re-export core types
pub use config::{AttentionConfig, NormConfig, RoPEConfig, SamplingConfig};
pub use error::{FlashInferError, Result};
pub use page_table::{PageTable, PageTableBuilder};
pub use types::{Backend, DType, GpuFloat, HeadDim, KVLayout, MaskMode, PosEncodingMode};
pub use workspace::WorkspaceSizes;

#[cfg(feature = "cuda")]
pub use workspace::Workspace;

#[cfg(feature = "cuda")]
pub use batch_decode::BatchDecodeHandler;

#[cfg(feature = "cuda")]
pub use batch_prefill::BatchPrefillHandler;

pub use page::{PagedKVCacheBuilder, PagedKVMetadata};

#[cfg(feature = "cuda")]
pub use page::PagedKVCache;

// MLA (DeepSeek) support
pub use mla::{MLAConfig, MLAKeyShape, MLAQueryShape};

// Backwards compatibility re-exports
pub use types::DType as DataType;

/// Attention computation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    /// Decode: single query token per sequence, attending to cached KV.
    Decode,
    /// Prefill: multiple query tokens, building KV cache.
    Prefill,
    /// Append: adding new tokens to existing KV cache.
    Append,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_compat() {
        let config = AttentionConfig::new(32, 8, 128)
            .with_page_size(16)
            .with_dtype(DType::BFloat16);

        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim_qk, 128);
        assert_eq!(config.page_size, 16);
        assert_eq!(config.gqa_ratio(), 4);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
        assert_eq!(DType::Float32.size_bytes(), 4);
    }

    #[test]
    fn test_types_reexports() {
        // Verify types are accessible
        let _ = KVLayout::NHD;
        let _ = PosEncodingMode::RoPELlama;
        let _ = MaskMode::Causal;
        let _ = Backend::Auto;
        let _ = HeadDim::D128;
    }
}

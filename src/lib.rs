//! FlashInfer Rust Bindings
//!
//! High-performance attention kernels for LLM inference serving.
//!
//! FlashInfer provides optimized CUDA kernels for:
//! - Paged KV cache attention (decode and prefill)
//! - Variable-length batch processing
//! - Multi-head and Grouped-Query Attention (GQA)
//!
//! # Features
//!
//! - `cuda` - Enable CUDA support via cudarc
//!
//! # Example
//!
//! ```ignore
//! use flashinfer_rs::{BatchDecodeHandler, PagedKVCache};
//!
//! let handler = BatchDecodeHandler::new(num_heads, head_dim)?;
//! let output = handler.forward(&query, &kv_cache, &page_table)?;
//! ```

mod error;
pub mod page_table;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
pub mod batch_decode;

#[cfg(feature = "cuda")]
pub mod batch_prefill;

pub use error::{FlashInferError, Result};
pub use page_table::{PageTable, PageTableBuilder};

#[cfg(feature = "cuda")]
pub use batch_decode::BatchDecodeHandler;

#[cfg(feature = "cuda")]
pub use batch_prefill::BatchPrefillHandler;

/// Configuration for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_qo_heads: usize,
    /// Number of key-value heads (for GQA, can be < num_qo_heads).
    pub num_kv_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Page/block size for paged KV cache.
    pub page_size: usize,
    /// Data type for computation.
    pub dtype: DataType,
}

impl AttentionConfig {
    pub fn new(num_qo_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size: 16,
            dtype: DataType::Float16,
        }
    }

    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    pub fn with_dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Returns the number of query heads per KV head (GQA ratio).
    pub fn num_qo_heads_per_kv_head(&self) -> usize {
        self.num_qo_heads / self.num_kv_heads
    }
}

/// Supported data types for attention computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float16,
    BFloat16,
    Float32,
}

impl DataType {
    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Float32 => 4,
        }
    }
}

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
    fn test_attention_config() {
        let config = AttentionConfig::new(32, 8, 128)
            .with_page_size(16)
            .with_dtype(DataType::BFloat16);

        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.page_size, 16);
        assert_eq!(config.num_qo_heads_per_kv_head(), 4);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DataType::Float16.size_bytes(), 2);
        assert_eq!(DataType::BFloat16.size_bytes(), 2);
        assert_eq!(DataType::Float32.size_bytes(), 4);
    }
}

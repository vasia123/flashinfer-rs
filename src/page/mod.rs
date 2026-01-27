//! Paged KV cache management for FlashInfer.
//!
//! This module provides GPU-side paged KV cache structures for efficient
//! memory management in LLM inference. The paged approach allows:
//!
//! - Dynamic memory allocation without fragmentation
//! - Efficient memory sharing across sequences
//! - Support for variable-length sequences
//!
//! # Architecture
//!
//! The paged KV cache uses a block-based memory layout:
//!
//! - **Page Pool**: Contiguous GPU memory divided into fixed-size pages
//! - **Page Table**: Maps logical sequence positions to physical pages
//! - **Metadata**: Tracks page assignments and sequence lengths
//!
//! # Example
//!
//! ```ignore
//! use flashinfer_rs::page::PagedKVCache;
//!
//! // Create a paged KV cache with 1024 pages
//! let kv_cache = PagedKVCache::<half::f16>::new(
//!     &device,
//!     1024,       // max_num_pages
//!     16,         // page_size
//!     8,          // num_kv_heads
//!     128,        // head_dim
//!     KVLayout::NHD,
//! )?;
//! ```

pub mod paged_kv;

pub use paged_kv::{PagedKVCacheBuilder, PagedKVMetadata};

#[cfg(feature = "cuda")]
pub use paged_kv::PagedKVCache;

//! GPU operations for FlashInfer.
//!
//! This module provides high-level Rust APIs for various GPU operations:
//! - `norm`: Normalization operations (RMSNorm, LayerNorm)
//! - `rope`: Rotary Position Embedding operations
//! - `sampling`: Sampling operations (Top-K, Top-P, softmax)

pub mod norm;
pub mod rope;
pub mod sampling;

pub use norm::{layernorm, rmsnorm, rmsnorm_inplace};
pub use rope::{apply_rope, apply_rope_inplace};
pub use sampling::{softmax, top_k_sampling, top_p_sampling};

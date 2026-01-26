//! Error types for FlashInfer operations.

use thiserror::Error;

/// Result type for FlashInfer operations.
pub type Result<T> = std::result::Result<T, FlashInferError>;

/// Errors that can occur during FlashInfer operations.
#[derive(Debug, Error)]
pub enum FlashInferError {
    /// CUDA error.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shape mismatch.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Page table error.
    #[error("Page table error: {0}")]
    PageTable(String),

    /// Unsupported operation.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Out of memory.
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl FlashInferError {
    pub fn cuda(msg: impl Into<String>) -> Self {
        Self::Cuda(msg.into())
    }

    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    pub fn shape_mismatch(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            got: got.into(),
        }
    }

    pub fn page_table(msg: impl Into<String>) -> Self {
        Self::PageTable(msg.into())
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for FlashInferError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        FlashInferError::Cuda(e.to_string())
    }
}

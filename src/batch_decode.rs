//! Batch decode attention with paged KV cache.
//!
//! This module provides the `BatchDecodeHandler` which implements
//! efficient attention computation for the decode phase where each
//! sequence has exactly one new query token.

use crate::cuda::Workspace;
use crate::page_table::PageTable;
use crate::{AttentionConfig, DataType, FlashInferError, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Handler for batched decode attention with paged KV cache.
///
/// This is the main interface for decode-phase attention in FlashInfer.
/// It manages workspace memory and provides optimized attention computation
/// for single-token queries against cached KV pairs.
///
/// # Example
///
/// ```ignore
/// let handler = BatchDecodeHandler::new(device, config)?;
///
/// // Plan the computation (call once per batch shape)
/// handler.plan(batch_size, kv_lengths, page_table)?;
///
/// // Run attention
/// let output = handler.forward(&query, &kv_cache)?;
/// ```
pub struct BatchDecodeHandler {
    device: Arc<CudaDevice>,
    config: AttentionConfig,
    workspace: Workspace,
    // Cached plan data
    planned_batch_size: Option<usize>,
}

impl BatchDecodeHandler {
    /// Create a new batch decode handler.
    pub fn new(device: Arc<CudaDevice>, config: AttentionConfig) -> Result<Self> {
        let workspace = Workspace::new(device.clone());

        Ok(Self {
            device,
            config,
            workspace,
            planned_batch_size: None,
        })
    }

    /// Get the attention configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Plan the decode operation for a specific batch.
    ///
    /// This precomputes metadata and allocates workspace for the given
    /// batch configuration. Call this when batch size or sequence lengths change.
    ///
    /// # Arguments
    /// * `batch_size` - Number of sequences in the batch
    /// * `kv_lengths` - KV cache length for each sequence
    /// * `page_table` - Page table mapping sequences to cache blocks
    pub fn plan(
        &mut self,
        batch_size: usize,
        kv_lengths: &[usize],
        page_table: &PageTable,
    ) -> Result<()> {
        if kv_lengths.len() != batch_size {
            return Err(FlashInferError::invalid_config(format!(
                "kv_lengths.len()={} != batch_size={}",
                kv_lengths.len(),
                batch_size
            )));
        }

        if page_table.batch_size() != batch_size {
            return Err(FlashInferError::invalid_config(format!(
                "page_table.batch_size()={} != batch_size={}",
                page_table.batch_size(),
                batch_size
            )));
        }

        // Calculate workspace size
        let workspace_size = crate::cuda::batch_decode_workspace_size(
            batch_size,
            self.config.num_qo_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.page_size,
            page_table.max_num_pages(),
        );

        self.workspace.ensure_size(workspace_size)?;
        self.planned_batch_size = Some(batch_size);

        Ok(())
    }

    /// Check if plan has been called.
    pub fn is_planned(&self) -> bool {
        self.planned_batch_size.is_some()
    }

    /// Run batch decode attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor, shape `[batch_size, num_qo_heads, head_dim]`
    /// * `kv_cache_k` - Key cache, shape `[num_blocks, page_size, num_kv_heads, head_dim]`
    /// * `kv_cache_v` - Value cache, shape `[num_blocks, page_size, num_kv_heads, head_dim]`
    /// * `page_table` - Page table for block mapping
    /// * `kv_lengths` - KV length for each sequence
    /// * `output` - Output tensor, shape `[batch_size, num_qo_heads, head_dim]`
    ///
    /// # Returns
    /// Result indicating success or error.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_inplace<T: DevicePtrType>(
        &self,
        query: &CudaSlice<T>,
        kv_cache_k: &CudaSlice<T>,
        kv_cache_v: &CudaSlice<T>,
        page_table: &PageTable,
        kv_lengths: &[usize],
        output: &mut CudaSlice<T>,
    ) -> Result<()> {
        let batch_size = self.planned_batch_size.ok_or_else(|| {
            FlashInferError::invalid_config("must call plan() before forward()")
        })?;

        if kv_lengths.len() != batch_size {
            return Err(FlashInferError::shape_mismatch(
                format!("batch_size={}", batch_size),
                format!("kv_lengths.len()={}", kv_lengths.len()),
            ));
        }

        // TODO: Launch actual FlashInfer CUDA kernel
        // For now, this is a placeholder that documents the expected interface
        //
        // The actual implementation would:
        // 1. Upload page_table and kv_lengths to GPU
        // 2. Launch flashinfer::BatchDecodeWithPagedKVCacheWrapper kernel
        // 3. Synchronize and return

        Err(FlashInferError::unsupported(
            "CUDA kernel not yet implemented - requires FlashInfer C++ bindings",
        ))
    }
}

/// Marker trait for supported device pointer types.
pub trait DevicePtrType: cudarc::driver::DeviceRepr + Clone {}

impl DevicePtrType for half::f16 {}
impl DevicePtrType for half::bf16 {}
impl DevicePtrType for f32 {}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require CUDA device, so they're marked ignore by default
    #[test]
    #[ignore]
    fn test_batch_decode_handler_creation() {
        let device = CudaDevice::new(0).unwrap();
        let config = AttentionConfig::new(32, 8, 128);
        let handler = BatchDecodeHandler::new(device, config).unwrap();

        assert_eq!(handler.config().num_qo_heads, 32);
        assert_eq!(handler.config().num_kv_heads, 8);
        assert!(!handler.is_planned());
    }
}

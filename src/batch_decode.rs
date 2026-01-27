//! Batch decode attention with paged KV cache.
//!
//! This module provides the `BatchDecodeHandler` which implements
//! efficient attention computation for the decode phase where each
//! sequence has exactly one new query token.

use crate::cuda::Workspace;
use crate::page_table::PageTable;
use crate::{AttentionConfig, FlashInferError, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
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
            self.config.num_qo_heads as usize,
            self.config.num_kv_heads as usize,
            self.config.head_dim_qk as usize,
            self.config.page_size as usize,
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

        // TODO(Phase 2): Implement using ffi::BatchDecodePlan
        //
        // This high-level handler is not yet functional. The implementation requires:
        // 1. Store ffi::BatchDecodePlan handle in struct (created in plan())
        // 2. Convert PageTable to kv_indptr/kv_indices/kv_last_page_len arrays
        // 3. Upload these arrays to GPU
        // 4. Call plan.run() with the GPU pointers
        //
        // WORKAROUND: Use ffi::BatchDecodePlan directly for now. Example:
        //   let plan = unsafe { ffi::BatchDecodePlan::new(...) }?;
        //   unsafe { plan.run(q, k_cache, v_cache, ...) }?;
        //
        // See ffi.rs for the working low-level API.

        let _ = (query, kv_cache_k, kv_cache_v, page_table, kv_lengths, output, batch_size);
        Err(FlashInferError::unsupported(
            "BatchDecodeHandler::forward_inplace not implemented. Use ffi::BatchDecodePlan directly.",
        ))
    }
}

/// Marker trait for supported device pointer types.
///
/// NOTE: This trait is a placeholder - actual CUDA operations require
/// types that implement cudarc's internal traits.
pub trait DevicePtrType: Copy + Send + Sync + 'static {}

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

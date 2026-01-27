//! Batch prefill attention with paged KV cache.
//!
//! This module provides the `BatchPrefillHandler` which implements
//! efficient attention computation for the prefill phase where each
//! sequence may have multiple query tokens.

use crate::cuda::Workspace;
use crate::page_table::PageTable;
use crate::{AttentionConfig, FlashInferError, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use crate::batch_decode::DevicePtrType;

/// Handler for batched prefill attention with paged KV cache.
///
/// This is the main interface for prefill-phase attention in FlashInfer.
/// It handles variable-length sequences efficiently using ragged tensors.
///
/// # Example
///
/// ```ignore
/// let handler = BatchPrefillHandler::new(device, config)?;
///
/// // Plan the computation
/// handler.plan(qo_indptr, kv_indptr, kv_lengths, page_table)?;
///
/// // Run attention
/// let output = handler.forward(&query, &key, &value, &kv_cache)?;
/// ```
pub struct BatchPrefillHandler {
    device: Arc<CudaDevice>,
    config: AttentionConfig,
    workspace: Workspace,
    // Cached plan data
    planned: bool,
    total_tokens: usize,
}

impl BatchPrefillHandler {
    /// Create a new batch prefill handler.
    pub fn new(device: Arc<CudaDevice>, config: AttentionConfig) -> Result<Self> {
        let workspace = Workspace::new(device.clone());

        Ok(Self {
            device,
            config,
            workspace,
            planned: false,
            total_tokens: 0,
        })
    }

    /// Get the attention configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Plan the prefill operation for a specific batch.
    ///
    /// # Arguments
    /// * `qo_indptr` - Cumulative query lengths, shape `[batch_size + 1]`
    /// * `kv_indptr` - Cumulative KV lengths, shape `[batch_size + 1]`
    /// * `kv_lengths` - KV length for each sequence
    /// * `page_table` - Page table mapping sequences to cache blocks
    pub fn plan(
        &mut self,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        kv_lengths: &[usize],
        _page_table: &PageTable,
    ) -> Result<()> {
        if qo_indptr.len() != kv_indptr.len() {
            return Err(FlashInferError::invalid_config(format!(
                "qo_indptr.len()={} != kv_indptr.len()={}",
                qo_indptr.len(),
                kv_indptr.len()
            )));
        }

        let batch_size = qo_indptr.len().saturating_sub(1);
        if kv_lengths.len() != batch_size {
            return Err(FlashInferError::invalid_config(format!(
                "kv_lengths.len()={} != batch_size={}",
                kv_lengths.len(),
                batch_size
            )));
        }

        let total_tokens = *qo_indptr.last().unwrap_or(&0) as usize;

        // Calculate workspace size
        let workspace_size = crate::cuda::batch_prefill_workspace_size(
            batch_size,
            total_tokens,
            self.config.num_qo_heads as usize,
            self.config.num_kv_heads as usize,
            self.config.head_dim_qk as usize,
            self.config.page_size as usize,
        );

        self.workspace.ensure_size(workspace_size)?;
        self.planned = true;
        self.total_tokens = total_tokens;

        Ok(())
    }

    /// Check if plan has been called.
    pub fn is_planned(&self) -> bool {
        self.planned
    }

    /// Run batch prefill attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor, shape `[total_tokens, num_qo_heads, head_dim]`
    /// * `key` - Key tensor for new tokens, shape `[total_tokens, num_kv_heads, head_dim]`
    /// * `value` - Value tensor for new tokens, shape `[total_tokens, num_kv_heads, head_dim]`
    /// * `kv_cache_k` - Key cache, shape `[num_blocks, page_size, num_kv_heads, head_dim]`
    /// * `kv_cache_v` - Value cache, shape `[num_blocks, page_size, num_kv_heads, head_dim]`
    /// * `page_table` - Page table for block mapping
    /// * `qo_indptr` - Cumulative query lengths
    /// * `kv_indptr` - Cumulative KV lengths
    /// * `output` - Output tensor, shape `[total_tokens, num_qo_heads, head_dim]`
    #[allow(clippy::too_many_arguments)]
    pub fn forward_inplace<T: DevicePtrType>(
        &self,
        query: &CudaSlice<T>,
        key: &CudaSlice<T>,
        value: &CudaSlice<T>,
        kv_cache_k: &CudaSlice<T>,
        kv_cache_v: &CudaSlice<T>,
        page_table: &PageTable,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        output: &mut CudaSlice<T>,
    ) -> Result<()> {
        if !self.planned {
            return Err(FlashInferError::invalid_config(
                "must call plan() before forward()",
            ));
        }

        // TODO(Phase 2): Implement using ffi::BatchPrefillPlan
        //
        // This high-level handler is not yet functional. The implementation requires:
        // 1. Store ffi::BatchPrefillPlan handle in struct (created in plan())
        // 2. Convert PageTable to kv_indptr/kv_indices/kv_last_page_len arrays
        // 3. Upload these arrays plus qo_indptr to GPU
        // 4. Call plan.run() with the GPU pointers
        //
        // WORKAROUND: Use ffi::BatchPrefillPlan directly for now. Example:
        //   let plan = unsafe { ffi::BatchPrefillPlan::new(...) }?;
        //   unsafe { plan.run(q, k_cache, v_cache, ...) }?;
        //
        // See ffi.rs for the working low-level API.

        let _ = (query, key, value, kv_cache_k, kv_cache_v, page_table, qo_indptr, kv_indptr, output);
        Err(FlashInferError::unsupported(
            "BatchPrefillHandler::forward_inplace not implemented. Use ffi::BatchPrefillPlan directly.",
        ))
    }

    /// Run prefill attention and write new KV to cache.
    ///
    /// This is a fused operation that both computes attention and
    /// appends the new key-value pairs to the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_and_append<T: DevicePtrType>(
        &self,
        query: &CudaSlice<T>,
        key: &CudaSlice<T>,
        value: &CudaSlice<T>,
        kv_cache_k: &mut CudaSlice<T>,
        kv_cache_v: &mut CudaSlice<T>,
        page_table: &PageTable,
        qo_indptr: &[i32],
        kv_indptr: &[i32],
        append_indptr: &[i32],
        output: &mut CudaSlice<T>,
    ) -> Result<()> {
        if !self.planned {
            return Err(FlashInferError::invalid_config(
                "must call plan() before forward()",
            ));
        }

        // TODO(Phase 2): Implement fused prefill + append kernel
        //
        // This would combine prefill attention with KV cache append in one kernel
        // to reduce memory bandwidth. FlashInfer may not have a direct fused API,
        // so this might require:
        // 1. Run prefill attention via ffi::BatchPrefillPlan
        // 2. Append KV to cache via ffi::append_paged_kv_cache
        //
        // WORKAROUND: Call forward_inplace() then append_paged_kv_cache() separately.

        let _ = (query, key, value, kv_cache_k, kv_cache_v, page_table, qo_indptr, kv_indptr, append_indptr, output);
        Err(FlashInferError::unsupported(
            "Fused prefill+append not implemented. Call forward_inplace() then append_paged_kv_cache() separately.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_batch_prefill_handler_creation() {
        let device = CudaDevice::new(0).unwrap();
        let config = AttentionConfig::new(32, 8, 128);
        let handler = BatchPrefillHandler::new(device, config).unwrap();

        assert_eq!(handler.config().num_qo_heads, 32);
        assert!(!handler.is_planned());
    }
}

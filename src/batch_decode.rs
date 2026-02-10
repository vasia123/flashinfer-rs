//! Batch decode attention with paged KV cache.
//!
//! This module provides the `BatchDecodeHandler` which implements
//! efficient attention computation for the decode phase where each
//! sequence has exactly one new query token.

use crate::page_table::PageTable;
use crate::types::GpuFloat;
use crate::workspace::Workspace;
use crate::{ffi, AttentionConfig, FlashInferError, KVLayout, Result};
use cudarc::driver::{CudaStream, CudaSlice, DevicePtr, DevicePtrMut};
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
/// let handler = BatchDecodeHandler::new(stream, config)?;
///
/// // Plan the computation (call once per batch shape)
/// handler.plan(batch_size, kv_lengths, page_table, &stream)?;
///
/// // Run attention
/// handler.forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)?;
/// ```
pub struct BatchDecodeHandler {
    stream: Arc<CudaStream>,
    config: AttentionConfig,
    workspace: Workspace,
    // Cached plan data (None until plan() is called)
    plan_data: Option<DecodePlanData>,
}

/// Internal state for a planned decode operation.
struct DecodePlanData {
    /// FFI plan handle.
    plan: ffi::BatchDecodePlan,
    /// GPU buffer for kv_indptr.
    kv_indptr_d: CudaSlice<i32>,
    /// GPU buffer for kv_indices.
    kv_indices_d: CudaSlice<i32>,
    /// GPU buffer for kv_last_page_len.
    kv_last_page_len_d: CudaSlice<i32>,
    /// Batch size for validation.
    batch_size: usize,
    /// KV layout.
    kv_layout: KVLayout,
}

impl BatchDecodeHandler {
    /// Create a new batch decode handler.
    pub fn new(stream: Arc<CudaStream>, config: AttentionConfig) -> Result<Self> {
        let workspace = Workspace::new(&stream)?;

        Ok(Self {
            stream,
            config,
            workspace,
            plan_data: None,
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
    /// * `stream` - CUDA stream for async operations
    ///
    /// # Type Parameters
    /// * `T` - Data type (half::f16, half::bf16, or f32)
    pub fn plan<T: GpuFloat>(
        &mut self,
        batch_size: usize,
        kv_lengths: &[usize],
        page_table: &PageTable,
        stream: &CudaStream,
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

        // Convert PageTable to FFI format
        let kv_indptr = page_table.to_kv_indptr();
        let kv_indices = page_table.to_kv_indices();
        let kv_last_page_len =
            page_table.compute_kv_last_page_len(kv_lengths, self.config.page_size as usize);

        // Upload arrays to GPU
        let kv_indptr_d = self.stream.memcpy_stod(&kv_indptr)?;
        let kv_indices_d = self.stream.memcpy_stod(&kv_indices)?;
        let kv_last_page_len_d = self.stream.memcpy_stod(&kv_last_page_len)?;

        // Ensure workspace is large enough
        let max_seq_len = kv_lengths.iter().copied().max().unwrap_or(0) as u32;
        let (float_size, int_size) = Workspace::batch_decode_sizes(
            batch_size as u32,
            max_seq_len,
            self.config.num_qo_heads,
            self.config.num_kv_heads,
            self.config.head_dim_qk,
            self.config.page_size,
        );
        self.workspace.ensure_sizes(float_size, int_size)?;

        // Create FFI plan — scope the guard so it drops before we move kv_indptr_d
        let plan = {
            let (indptr_ptr, _indptr_guard) = kv_indptr_d.device_ptr(stream);
            unsafe {
                ffi::BatchDecodePlan::new(
                    self.workspace.float_ptr() as *mut std::ffi::c_void,
                    self.workspace.float_size(),
                    self.workspace.int_ptr() as *mut std::ffi::c_void,
                    self.workspace.int_size(),
                    std::ptr::null_mut(), // page_locked_workspace (optional)
                    0,                    // page_locked_size
                    indptr_ptr as *const i32,
                    batch_size as i32,
                    self.config.num_qo_heads as i32,
                    self.config.num_kv_heads as i32,
                    self.config.head_dim_qk as i32,
                    self.config.page_size as i32,
                    T::DTYPE.into(),
                    self.config.pos_encoding.into(),
                    self.config.logits_soft_cap.unwrap_or(0.0),
                    self.config.window_left.unwrap_or(-1), // -1 = full attention
                    false,                                 // enable_cuda_graph
                    stream.cu_stream() as *mut std::ffi::c_void,
                )?
            }
        };

        self.plan_data = Some(DecodePlanData {
            plan,
            kv_indptr_d,
            kv_indices_d,
            kv_last_page_len_d,
            batch_size,
            kv_layout: self.config.kv_layout,
        });

        Ok(())
    }

    /// Check if plan has been called.
    pub fn is_planned(&self) -> bool {
        self.plan_data.is_some()
    }

    /// Run batch decode attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor, shape `[batch_size, num_qo_heads, head_dim]`
    /// * `kv_cache_k` - Key cache, shape depends on kv_layout
    /// * `kv_cache_v` - Value cache, shape depends on kv_layout
    /// * `output` - Output tensor, shape `[batch_size, num_qo_heads, head_dim]`
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// Result indicating success or error.
    pub fn forward<T: GpuFloat>(
        &self,
        query: &CudaSlice<T>,
        kv_cache_k: &CudaSlice<T>,
        kv_cache_v: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan_data = self
            .plan_data
            .as_ref()
            .ok_or_else(|| FlashInferError::invalid_config("must call plan() before forward()"))?;

        // Extract all pointers before the unsafe block so guards live long enough
        let (q_ptr, _q_guard) = query.device_ptr(stream);
        let (kk_ptr, _kk_guard) = kv_cache_k.device_ptr(stream);
        let (kv_ptr, _kv_guard) = kv_cache_v.device_ptr(stream);
        let (indptr_ptr, _indptr_guard) = plan_data.kv_indptr_d.device_ptr(stream);
        let (indices_ptr, _indices_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = plan_data.kv_last_page_len_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                q_ptr as *const std::ffi::c_void,
                kk_ptr as *const std::ffi::c_void,
                kv_ptr as *const std::ffi::c_void,
                indptr_ptr as *const i32,
                indices_ptr as *const i32,
                last_page_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                std::ptr::null_mut(), // lse (optional)
                plan_data.kv_layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Run batch decode attention with log-sum-exp output.
    ///
    /// Same as `forward()` but also returns the log-sum-exp values
    /// which can be used for computing attention statistics.
    pub fn forward_with_lse<T: GpuFloat>(
        &self,
        query: &CudaSlice<T>,
        kv_cache_k: &CudaSlice<T>,
        kv_cache_v: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        lse: &mut CudaSlice<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan_data = self
            .plan_data
            .as_ref()
            .ok_or_else(|| FlashInferError::invalid_config("must call plan() before forward()"))?;

        let (q_ptr, _q_guard) = query.device_ptr(stream);
        let (kk_ptr, _kk_guard) = kv_cache_k.device_ptr(stream);
        let (kv_ptr, _kv_guard) = kv_cache_v.device_ptr(stream);
        let (indptr_ptr, _indptr_guard) = plan_data.kv_indptr_d.device_ptr(stream);
        let (indices_ptr, _indices_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = plan_data.kv_last_page_len_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);
        let (lse_ptr, _lse_guard) = lse.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                q_ptr as *const std::ffi::c_void,
                kk_ptr as *const std::ffi::c_void,
                kv_ptr as *const std::ffi::c_void,
                indptr_ptr as *const i32,
                indices_ptr as *const i32,
                last_page_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                lse_ptr as *mut f32,
                plan_data.kv_layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Get the planned batch size, if planned.
    pub fn batch_size(&self) -> Option<usize> {
        self.plan_data.as_ref().map(|d| d.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    // Tests require CUDA device, so they're marked ignore by default
    #[test]
    #[ignore]
    fn test_batch_decode_handler_creation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let config = AttentionConfig::new(32, 8, 128);
        let handler = BatchDecodeHandler::new(stream, config).unwrap();

        assert_eq!(handler.config().num_qo_heads, 32);
        assert_eq!(handler.config().num_kv_heads, 8);
        assert!(!handler.is_planned());
    }
}

//! Batch prefill attention with paged KV cache.
//!
//! This module provides the `BatchPrefillHandler` which implements
//! efficient attention computation for the prefill phase where each
//! sequence may have multiple query tokens.

use crate::page_table::PageTable;
use crate::types::GpuFloat;
use crate::workspace::Workspace;
use crate::{ffi, AttentionConfig, FlashInferError, KVLayout, MaskMode, Result};
use cudarc::driver::{CudaStream, CudaSlice, DevicePtr, DevicePtrMut};
use std::sync::Arc;

/// Handler for batched prefill attention with paged KV cache.
///
/// This is the main interface for prefill-phase attention in FlashInfer.
/// It handles variable-length sequences efficiently using ragged tensors.
///
/// # Example
///
/// ```ignore
/// let handler = BatchPrefillHandler::new(stream, config)?;
///
/// // Plan the computation
/// handler.plan::<half::f16>(qo_indptr, kv_lengths, page_table, &stream)?;
///
/// // Run attention
/// handler.forward(&query, &kv_cache_k, &kv_cache_v, &mut output, &stream)?;
/// ```
pub struct BatchPrefillHandler {
    stream: Arc<CudaStream>,
    config: AttentionConfig,
    workspace: Workspace,
    // Cached plan data (None until plan() is called)
    plan_data: Option<PrefillPlanData>,
}

/// Internal state for a planned prefill operation.
struct PrefillPlanData {
    /// FFI plan handle.
    plan: ffi::BatchPrefillPlan,
    /// GPU buffer for qo_indptr.
    qo_indptr_d: CudaSlice<i32>,
    /// GPU buffer for kv_indptr.
    kv_indptr_d: CudaSlice<i32>,
    /// GPU buffer for kv_indices.
    kv_indices_d: CudaSlice<i32>,
    /// GPU buffer for kv_last_page_len.
    kv_last_page_len_d: CudaSlice<i32>,
    /// Batch size for validation.
    batch_size: usize,
    /// Total query tokens.
    total_tokens: usize,
    /// KV layout.
    kv_layout: KVLayout,
}

impl BatchPrefillHandler {
    /// Create a new batch prefill handler.
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

    /// Plan the prefill operation for a specific batch.
    ///
    /// # Arguments
    /// * `qo_indptr` - Cumulative query lengths, shape `[batch_size + 1]`
    /// * `kv_lengths` - KV length for each sequence
    /// * `page_table` - Page table mapping sequences to cache blocks
    /// * `stream` - CUDA stream for async operations
    ///
    /// # Type Parameters
    /// * `T` - Data type (half::f16, half::bf16, or f32)
    pub fn plan<T: GpuFloat>(
        &mut self,
        qo_indptr: &[i32],
        kv_lengths: &[usize],
        page_table: &PageTable,
        stream: &CudaStream,
    ) -> Result<()> {
        let batch_size = qo_indptr.len().saturating_sub(1);

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

        let total_tokens = *qo_indptr.last().unwrap_or(&0) as usize;

        // Convert PageTable to FFI format
        let kv_indptr = page_table.to_kv_indptr();
        let kv_indices = page_table.to_kv_indices();
        let kv_last_page_len =
            page_table.compute_kv_last_page_len(kv_lengths, self.config.page_size as usize);

        // Upload arrays to GPU
        let qo_indptr_d = self.stream.memcpy_stod(qo_indptr)?;
        let kv_indptr_d = self.stream.memcpy_stod(&kv_indptr)?;
        let kv_indices_d = self.stream.memcpy_stod(&kv_indices)?;
        let kv_last_page_len_d = self.stream.memcpy_stod(&kv_last_page_len)?;

        // Ensure workspace is large enough
        let (float_size, int_size) = Workspace::batch_prefill_sizes(
            batch_size as u32,
            total_tokens as u32,
            self.config.num_qo_heads,
            self.config.num_kv_heads,
            self.config.head_dim_qk,
            self.config.page_size,
        );
        self.workspace.ensure_sizes(float_size, int_size)?;

        // Determine causal mode
        let causal = matches!(self.config.mask_mode, MaskMode::Causal);

        // Create FFI plan — scope guards so they drop before we move the slices
        let plan = {
            let (qo_indptr_ptr, _qo_guard) = qo_indptr_d.device_ptr(stream);
            let (kv_indptr_ptr, _kv_guard) = kv_indptr_d.device_ptr(stream);
            unsafe {
                ffi::BatchPrefillPlan::new(
                    self.workspace.float_ptr() as *mut std::ffi::c_void,
                    self.workspace.float_size(),
                    self.workspace.int_ptr() as *mut std::ffi::c_void,
                    self.workspace.int_size(),
                    std::ptr::null_mut(), // page_locked_workspace (optional)
                    0,                    // page_locked_size
                    qo_indptr_ptr as *const i32,
                    kv_indptr_ptr as *const i32,
                    batch_size as i32,
                    self.config.num_qo_heads as i32,
                    self.config.num_kv_heads as i32,
                    self.config.head_dim_qk as i32,
                    self.config.page_size as i32,
                    T::DTYPE.into(),
                    self.config.pos_encoding.into(),
                    self.config.logits_soft_cap.unwrap_or(0.0),
                    self.config.window_left.unwrap_or(-1), // -1 = full attention
                    causal,
                    false, // enable_cuda_graph
                    stream.cu_stream() as *mut std::ffi::c_void,
                )?
            }
        };

        self.plan_data = Some(PrefillPlanData {
            plan,
            qo_indptr_d,
            kv_indptr_d,
            kv_indices_d,
            kv_last_page_len_d,
            batch_size,
            total_tokens,
            kv_layout: self.config.kv_layout,
        });

        Ok(())
    }

    /// Check if plan has been called.
    pub fn is_planned(&self) -> bool {
        self.plan_data.is_some()
    }

    /// Run batch prefill attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor, shape `[total_tokens, num_qo_heads, head_dim]`
    /// * `kv_cache_k` - Key cache, shape depends on kv_layout
    /// * `kv_cache_v` - Value cache, shape depends on kv_layout
    /// * `output` - Output tensor, shape `[total_tokens, num_qo_heads, head_dim]`
    /// * `stream` - CUDA stream
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

        let (q_ptr, _q_guard) = query.device_ptr(stream);
        let (kk_ptr, _kk_guard) = kv_cache_k.device_ptr(stream);
        let (kv_ptr, _kv_guard) = kv_cache_v.device_ptr(stream);
        let (kv_indptr_ptr, _ki_guard) = plan_data.kv_indptr_d.device_ptr(stream);
        let (kv_indices_ptr, _kix_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = plan_data.kv_last_page_len_d.device_ptr(stream);
        let (qo_indptr_ptr, _qo_guard) = plan_data.qo_indptr_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                q_ptr as *const std::ffi::c_void,
                kk_ptr as *const std::ffi::c_void,
                kv_ptr as *const std::ffi::c_void,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_page_ptr as *const i32,
                qo_indptr_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                std::ptr::null_mut(), // lse (optional)
                plan_data.kv_layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Run batch prefill attention with log-sum-exp output.
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
        let (kv_indptr_ptr, _ki_guard) = plan_data.kv_indptr_d.device_ptr(stream);
        let (kv_indices_ptr, _kix_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = plan_data.kv_last_page_len_d.device_ptr(stream);
        let (qo_indptr_ptr, _qo_guard) = plan_data.qo_indptr_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);
        let (lse_ptr, _lse_guard) = lse.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                q_ptr as *const std::ffi::c_void,
                kk_ptr as *const std::ffi::c_void,
                kv_ptr as *const std::ffi::c_void,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_page_ptr as *const i32,
                qo_indptr_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                lse_ptr as *mut f32,
                plan_data.kv_layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Run prefill attention and append new KV to cache.
    ///
    /// This is a convenience method that runs prefill attention and then
    /// appends new key-value pairs to the paged cache. The operations are
    /// serialized on the provided stream.
    ///
    /// # Arguments
    /// * `query` - Query tensor, shape `[total_tokens, num_qo_heads, head_dim]`
    /// * `key` - New key tensor, shape `[total_tokens, num_kv_heads, head_dim]`
    /// * `value` - New value tensor, shape `[total_tokens, num_kv_heads, head_dim]`
    /// * `kv_cache_k` - Key cache (will be modified)
    /// * `kv_cache_v` - Value cache (will be modified)
    /// * `append_indptr` - Cumulative append lengths per sequence
    /// * `output` - Output tensor
    /// * `stream` - CUDA stream
    #[allow(clippy::too_many_arguments)]
    pub fn forward_and_append<T: GpuFloat>(
        &self,
        query: &CudaSlice<T>,
        key: &CudaSlice<T>,
        value: &CudaSlice<T>,
        kv_cache_k: &mut CudaSlice<T>,
        kv_cache_v: &mut CudaSlice<T>,
        append_indptr: &[i32],
        output: &mut CudaSlice<T>,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan_data = self.plan_data.as_ref().ok_or_else(|| {
            FlashInferError::invalid_config("must call plan() before forward_and_append()")
        })?;

        // Step 1: Run prefill attention
        self.forward(query, kv_cache_k, kv_cache_v, output, stream)?;

        // Step 2: Append new KV to cache
        let append_indptr_d = self.stream.memcpy_stod(append_indptr)?;

        let (k_ptr, _k_guard) = key.device_ptr(stream);
        let (v_ptr, _v_guard) = value.device_ptr(stream);
        let (kck_ptr, _kck_guard) = kv_cache_k.device_ptr_mut(stream);
        let (kcv_ptr, _kcv_guard) = kv_cache_v.device_ptr_mut(stream);
        let (kv_indptr_ptr, _ki_guard) = plan_data.kv_indptr_d.device_ptr(stream);
        let (kv_indices_ptr, _kix_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = plan_data.kv_last_page_len_d.device_ptr(stream);
        let (append_ptr, _ap_guard) = append_indptr_d.device_ptr(stream);

        unsafe {
            ffi::append_paged_kv_cache(
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                kck_ptr as *mut std::ffi::c_void,
                kcv_ptr as *mut std::ffi::c_void,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_page_ptr as *const i32,
                append_ptr as *const i32,
                plan_data.batch_size as i32,
                self.config.num_kv_heads as i32,
                self.config.head_dim_qk as i32,
                self.config.page_size as i32,
                T::DTYPE.into(),
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

    /// Get the total tokens count, if planned.
    pub fn total_tokens(&self) -> Option<usize> {
        self.plan_data.as_ref().map(|d| d.total_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    #[ignore]
    fn test_batch_prefill_handler_creation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let config = AttentionConfig::new(32, 8, 128);
        let handler = BatchPrefillHandler::new(stream, config).unwrap();

        assert_eq!(handler.config().num_qo_heads, 32);
        assert!(!handler.is_planned());
    }
}

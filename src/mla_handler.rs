//! MLA (Multi-head Latent Attention) handler for paged KV cache.
//!
//! Provides [`MLAHandler`] for DeepSeek v2/v3 MLA attention computation.
//! Unlike standard attention which uses separate K and V caches, MLA uses:
//! - `ckv_cache`: compressed KV `[num_pages, page_size, head_dim_ckv=512]`
//! - `kpe_cache`: K position embeddings `[num_pages, page_size, head_dim_kpe=64]`
//!
//! Queries are split into:
//! - `q_nope`: absorbed non-RoPE component `[nnz, num_heads, head_dim_ckv=512]`
//! - `q_pe`: RoPE component `[nnz, num_heads, head_dim_kpe=64]`

use crate::mla::MLAConfig;
use crate::page_table::PageTable;
use crate::types::GpuFloat;
use crate::workspace::Workspace;
use crate::{ffi, FlashInferError, MaskMode, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;

/// Handler for MLA (Multi-head Latent Attention) with paged KV cache.
///
/// Supports both prefill and decode phases. The `qo_indptr` parameter
/// in [`plan()`](Self::plan) determines the mode:
/// - Decode: `[0, 1, 2, ..., batch_size]` (1 token per sequence)
/// - Prefill: `[0, n1, n1+n2, ...]` (variable tokens per sequence)
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::{MLAConfig, MLAHandler};
///
/// let config = MLAConfig::deepseek_with_page_size(16);
/// let mut handler = MLAHandler::new(stream, config)?;
///
/// // Plan (call when batch shape changes)
/// handler.plan::<half::bf16>(&qo_indptr, &kv_lengths, &page_table, true, &stream)?;
///
/// // Run attention
/// handler.forward::<half::bf16>(&q_nope, &q_pe, &ckv_cache, &kpe_cache, &mut output, &stream)?;
/// ```
pub struct MLAHandler {
    stream: Arc<CudaStream>,
    config: MLAConfig,
    workspace: Workspace,
    plan_data: Option<MLAPlanData>,
}

/// Internal state for a planned MLA operation.
struct MLAPlanData {
    /// FFI plan handle.
    plan: ffi::MLAPlan,
    /// GPU buffer for kv_indices (page indices, used in run()).
    kv_indices_d: CudaSlice<i32>,
    /// Batch size.
    batch_size: usize,
    /// Total query tokens.
    total_tokens: usize,
    /// Whether causal masking is enabled.
    causal: bool,
}

impl MLAHandler {
    /// Create a new MLA handler.
    ///
    /// Validates the MLA configuration and allocates default workspace buffers.
    pub fn new(stream: Arc<CudaStream>, config: MLAConfig) -> Result<Self> {
        config.validate()?;
        let workspace = Workspace::new(&stream)?;
        Ok(Self {
            stream,
            config,
            workspace,
            plan_data: None,
        })
    }

    /// Get the MLA configuration.
    pub fn config(&self) -> &MLAConfig {
        &self.config
    }

    /// Plan the MLA attention operation for a specific batch.
    ///
    /// Precomputes metadata and allocates workspace. Call when batch shape
    /// or sequence lengths change.
    ///
    /// # Arguments
    /// * `qo_indptr` - Cumulative query token counts `[batch_size + 1]`.
    ///   For decode: `[0, 1, 2, ..., batch_size]`.
    ///   For prefill: `[0, n1, n1+n2, ...]`.
    /// * `kv_lengths` - KV cache length per sequence `[batch_size]`
    /// * `page_table` - Page table mapping sequences to cache blocks
    /// * `causal` - Whether to apply causal masking
    /// * `stream` - CUDA stream
    pub fn plan<T: GpuFloat>(
        &mut self,
        qo_indptr: &[i32],
        kv_lengths: &[usize],
        page_table: &PageTable,
        causal: bool,
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
        let kv_len_i32: Vec<i32> = kv_lengths.iter().map(|&l| l as i32).collect();

        // Upload arrays to GPU
        let qo_indptr_d = self.stream.memcpy_stod(qo_indptr)?;
        let kv_indptr_d = self.stream.memcpy_stod(&kv_indptr)?;
        let kv_indices_d = self.stream.memcpy_stod(&kv_indices)?;
        let kv_len_d = self.stream.memcpy_stod(&kv_len_i32)?;

        // Query workspace sizes from FFI
        let max_seq_len = kv_lengths.iter().copied().max().unwrap_or(0) as i32;
        let (float_size, int_size) = ffi::mla_workspace_size(
            batch_size as i32,
            max_seq_len,
            self.config.num_heads as i32,
            self.config.head_dim_ckv as i32,
            self.config.head_dim_kpe as i32,
            self.config.page_size as i32,
        )?;
        self.workspace.ensure_sizes(float_size, int_size)?;

        // Create FFI plan — scope guards so they drop before we move the slices
        let plan = {
            let (qo_ptr, _qo_guard) = qo_indptr_d.device_ptr(stream);
            let (kv_ind_ptr, _kvi_guard) = kv_indptr_d.device_ptr(stream);
            let (kv_len_ptr, _kvl_guard) = kv_len_d.device_ptr(stream);

            unsafe {
                ffi::MLAPlan::new(
                    self.workspace.float_ptr() as *mut std::ffi::c_void,
                    self.workspace.float_size(),
                    self.workspace.int_ptr() as *mut std::ffi::c_void,
                    self.workspace.int_size(),
                    std::ptr::null_mut(), // page_locked_workspace
                    0,                    // page_locked_size
                    qo_ptr as *const i32,
                    kv_ind_ptr as *const i32,
                    kv_len_ptr as *const i32,
                    batch_size as i32,
                    self.config.num_heads as i32,
                    self.config.head_dim_ckv as i32,
                    self.config.head_dim_kpe as i32,
                    self.config.page_size as i32,
                    causal,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )?
            }
        };

        // qo_indptr_d, kv_indptr_d, kv_len_d are consumed by plan and can be freed.
        // kv_indices_d must be kept alive for run().
        self.plan_data = Some(MLAPlanData {
            plan,
            kv_indices_d,
            batch_size,
            total_tokens,
            causal,
        });

        Ok(())
    }

    /// Check if plan has been called.
    pub fn is_planned(&self) -> bool {
        self.plan_data.is_some()
    }

    /// Run MLA attention.
    ///
    /// # Arguments
    /// * `q_nope` - Query nope `[nnz, num_heads, head_dim_ckv]`
    /// * `q_pe` - Query PE `[nnz, num_heads, head_dim_kpe]`
    /// * `ckv_cache` - Compressed KV cache `[num_pages, page_size, head_dim_ckv]`
    /// * `kpe_cache` - K position embedding cache `[num_pages, page_size, head_dim_kpe]`
    /// * `output` - Output tensor `[nnz, num_heads, head_dim_ckv]`
    /// * `stream` - CUDA stream
    pub fn forward<T: GpuFloat>(
        &self,
        q_nope: &CudaSlice<T>,
        q_pe: &CudaSlice<T>,
        ckv_cache: &CudaSlice<T>,
        kpe_cache: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan_data = self
            .plan_data
            .as_ref()
            .ok_or_else(|| FlashInferError::invalid_config("must call plan() before forward()"))?;

        let mask_mode = if plan_data.causal {
            MaskMode::Causal
        } else {
            MaskMode::None
        };

        let (qn_ptr, _qn_guard) = q_nope.device_ptr(stream);
        let (qp_ptr, _qp_guard) = q_pe.device_ptr(stream);
        let (ckv_ptr, _ckv_guard) = ckv_cache.device_ptr(stream);
        let (kpe_ptr, _kpe_guard) = kpe_cache.device_ptr(stream);
        let (indices_ptr, _idx_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                qn_ptr as *const std::ffi::c_void,
                qp_ptr as *const std::ffi::c_void,
                ckv_ptr as *const std::ffi::c_void,
                kpe_ptr as *const std::ffi::c_void,
                indices_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                std::ptr::null_mut(), // no LSE
                mask_mode.into(),
                self.config.sm_scale(),
                T::DTYPE.into(),
                T::DTYPE.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Run MLA attention with log-sum-exp output.
    ///
    /// Same as [`forward()`](Self::forward) but also returns log-sum-exp values
    /// for attention statistics (useful for speculative decoding verification).
    pub fn forward_with_lse<T: GpuFloat>(
        &self,
        q_nope: &CudaSlice<T>,
        q_pe: &CudaSlice<T>,
        ckv_cache: &CudaSlice<T>,
        kpe_cache: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        lse: &mut CudaSlice<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan_data = self
            .plan_data
            .as_ref()
            .ok_or_else(|| FlashInferError::invalid_config("must call plan() before forward()"))?;

        let mask_mode = if plan_data.causal {
            MaskMode::Causal
        } else {
            MaskMode::None
        };

        let (qn_ptr, _qn_guard) = q_nope.device_ptr(stream);
        let (qp_ptr, _qp_guard) = q_pe.device_ptr(stream);
        let (ckv_ptr, _ckv_guard) = ckv_cache.device_ptr(stream);
        let (kpe_ptr, _kpe_guard) = kpe_cache.device_ptr(stream);
        let (indices_ptr, _idx_guard) = plan_data.kv_indices_d.device_ptr(stream);
        let (out_ptr, _out_guard) = output.device_ptr_mut(stream);
        let (lse_ptr, _lse_guard) = lse.device_ptr_mut(stream);

        unsafe {
            plan_data.plan.run(
                qn_ptr as *const std::ffi::c_void,
                qp_ptr as *const std::ffi::c_void,
                ckv_ptr as *const std::ffi::c_void,
                kpe_ptr as *const std::ffi::c_void,
                indices_ptr as *const i32,
                out_ptr as *mut std::ffi::c_void,
                lse_ptr as *mut f32,
                mask_mode.into(),
                self.config.sm_scale(),
                T::DTYPE.into(),
                T::DTYPE.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )?;
        }

        Ok(())
    }

    /// Get the planned batch size, if planned.
    pub fn batch_size(&self) -> Option<usize> {
        self.plan_data.as_ref().map(|d| d.batch_size)
    }

    /// Get the total query tokens count, if planned.
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
    fn test_mla_handler_creation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let config = MLAConfig::deepseek_with_page_size(16);
        let handler = MLAHandler::new(stream, config).unwrap();

        assert_eq!(handler.config().num_heads, 128);
        assert_eq!(handler.config().head_dim_ckv, 512);
        assert_eq!(handler.config().head_dim_kpe, 64);
        assert!(!handler.is_planned());
        assert_eq!(handler.batch_size(), None);
        assert_eq!(handler.total_tokens(), None);
    }

    #[test]
    #[ignore]
    fn test_mla_handler_invalid_config() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut bad_config = MLAConfig::deepseek();
        bad_config.head_dim_ckv = 256;
        assert!(MLAHandler::new(stream, bad_config).is_err());
    }

    #[test]
    #[ignore]
    fn test_mla_handler_plan_validation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let config = MLAConfig::deepseek_with_page_size(16);
        let mut handler = MLAHandler::new(stream.clone(), config).unwrap();

        // Mismatched kv_lengths
        let qo_indptr = vec![0, 1, 2];
        let kv_lengths = vec![10]; // only 1, but batch_size=2
        let page_table = PageTable::new(2, 4);

        let result = handler.plan::<half::bf16>(&qo_indptr, &kv_lengths, &page_table, true, &stream);
        assert!(result.is_err());

        // Mismatched page_table batch_size
        let kv_lengths = vec![10, 20];
        let bad_page_table = PageTable::new(3, 4); // batch_size=3, but qo says 2

        let result =
            handler.plan::<half::bf16>(&qo_indptr, &kv_lengths, &bad_page_table, true, &stream);
        assert!(result.is_err());
    }
}

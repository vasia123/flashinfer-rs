//! CUDA utilities and workspace management.
//!
//! This module provides CUDA memory management for FlashInfer operations.

use crate::{FlashInferError, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// CUDA workspace buffer for FlashInfer operations.
///
/// FlashInfer kernels require workspace memory for intermediate results.
/// This struct manages that memory efficiently.
pub struct Workspace {
    device: Arc<CudaDevice>,
    buffer: Option<CudaSlice<u8>>,
    size: usize,
}

impl Workspace {
    /// Create a new workspace on the given device.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            buffer: None,
            size: 0,
        }
    }

    /// Ensure the workspace has at least `required_size` bytes.
    pub fn ensure_size(&mut self, required_size: usize) -> Result<()> {
        if self.size >= required_size {
            return Ok(());
        }

        // Allocate with some extra room to avoid frequent reallocations
        let new_size = required_size.next_power_of_two();
        self.buffer = Some(self.device.alloc_zeros::<u8>(new_size)?);
        self.size = new_size;

        Ok(())
    }

    /// Get the workspace buffer pointer.
    pub fn ptr(&self) -> Option<DevicePtr<u8>> {
        self.buffer.as_ref().map(|b| *b.device_ptr())
    }

    /// Get the current workspace size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Calculate required workspace size for batch decode.
pub fn batch_decode_workspace_size(
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    max_num_pages: usize,
) -> usize {
    // Based on FlashInfer's workspace requirements
    // This is a conservative estimate
    let qo_heads_per_kv = num_qo_heads / num_kv_heads;

    // Partition info
    let partition_size = batch_size * num_kv_heads * 2 * std::mem::size_of::<i32>();

    // Temporary output for split-k
    let tmp_output_size = batch_size * num_qo_heads * head_dim * std::mem::size_of::<f32>();

    // LSE (log-sum-exp) for split-k
    let lse_size = batch_size * num_qo_heads * std::mem::size_of::<f32>();

    // Page indices buffer
    let page_indices_size = batch_size * max_num_pages * std::mem::size_of::<i32>();

    partition_size + tmp_output_size + lse_size + page_indices_size + 1024 // padding
}

/// Calculate required workspace size for batch prefill.
pub fn batch_prefill_workspace_size(
    batch_size: usize,
    total_tokens: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
) -> usize {
    // Prefill typically needs more workspace due to larger attention matrices
    let qo_heads_per_kv = num_qo_heads / num_kv_heads;

    // Request indices and positions
    let request_info_size = batch_size * 3 * std::mem::size_of::<i32>();

    // Temporary storage for partial results
    let tmp_size = total_tokens * num_qo_heads * head_dim * std::mem::size_of::<f32>();

    request_info_size + tmp_size + 4096 // padding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_size_calculation() {
        let size = batch_decode_workspace_size(
            32,  // batch_size
            32,  // num_qo_heads
            8,   // num_kv_heads
            128, // head_dim
            16,  // page_size
            64,  // max_num_pages
        );

        // Should be non-zero and reasonable
        assert!(size > 0);
        assert!(size < 100 * 1024 * 1024); // Less than 100MB
    }
}

//! GPU workspace management for FlashInfer operations.
//!
//! FlashInfer kernels require temporary workspace buffers for intermediate
//! computations. This module provides utilities for allocating and managing
//! these workspaces.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::Result;

/// Default float workspace size (16 MB).
const DEFAULT_FLOAT_WORKSPACE_SIZE: usize = 16 * 1024 * 1024;

/// Default int workspace size (8 MB).
const DEFAULT_INT_WORKSPACE_SIZE: usize = 8 * 1024 * 1024;

/// GPU workspace for FlashInfer operations.
///
/// The workspace provides temporary memory buffers that kernels use for
/// intermediate computations like split-K reduction and auxiliary data
/// structures.
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::workspace::Workspace;
///
/// let device = CudaDevice::new(0)?;
/// let workspace = Workspace::new(&device)?;
///
/// // Or with custom sizes:
/// let workspace = Workspace::with_sizes(&device, 32 * 1024 * 1024, 16 * 1024 * 1024)?;
/// ```
#[cfg(feature = "cuda")]
pub struct Workspace {
    /// Float workspace buffer for temporary float computations.
    float_buffer: CudaSlice<u8>,
    /// Int workspace buffer for auxiliary data structures.
    int_buffer: CudaSlice<u8>,
    /// Float buffer size (tracked separately since CudaSlice doesn't have len()).
    float_buffer_size: usize,
    /// Int buffer size.
    int_buffer_size: usize,
    /// Device reference.
    device: std::sync::Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl Workspace {
    /// Creates a new workspace with default sizes.
    pub fn new(device: &std::sync::Arc<CudaDevice>) -> Result<Self> {
        Self::with_sizes(
            device,
            DEFAULT_FLOAT_WORKSPACE_SIZE,
            DEFAULT_INT_WORKSPACE_SIZE,
        )
    }

    /// Creates a new workspace with specified sizes.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device
    /// * `float_size` - Size of float workspace in bytes
    /// * `int_size` - Size of int workspace in bytes
    pub fn with_sizes(
        device: &std::sync::Arc<CudaDevice>,
        float_size: usize,
        int_size: usize,
    ) -> Result<Self> {
        let float_buffer = device.alloc_zeros::<u8>(float_size)?;
        let int_buffer = device.alloc_zeros::<u8>(int_size)?;

        Ok(Self {
            float_buffer,
            int_buffer,
            float_buffer_size: float_size,
            int_buffer_size: int_size,
            device: device.clone(),
        })
    }

    /// Ensures the workspace has at least the specified sizes.
    ///
    /// If the current buffers are too small, they will be reallocated.
    pub fn ensure_sizes(&mut self, float_size: usize, int_size: usize) -> Result<()> {
        if self.float_buffer_size < float_size {
            self.float_buffer = self.device.alloc_zeros::<u8>(float_size)?;
            self.float_buffer_size = float_size;
        }
        if self.int_buffer_size < int_size {
            self.int_buffer = self.device.alloc_zeros::<u8>(int_size)?;
            self.int_buffer_size = int_size;
        }
        Ok(())
    }

    /// Returns the float workspace buffer pointer.
    #[inline]
    pub fn float_ptr(&self) -> *mut u8 {
        *self.float_buffer.device_ptr() as *mut u8
    }

    /// Returns the int workspace buffer pointer.
    #[inline]
    pub fn int_ptr(&self) -> *mut u8 {
        *self.int_buffer.device_ptr() as *mut u8
    }

    /// Returns the float workspace size in bytes.
    #[inline]
    pub fn float_size(&self) -> usize {
        self.float_buffer_size
    }

    /// Returns the int workspace size in bytes.
    #[inline]
    pub fn int_size(&self) -> usize {
        self.int_buffer_size
    }

    /// Returns a reference to the device.
    #[inline]
    pub fn device(&self) -> &std::sync::Arc<CudaDevice> {
        &self.device
    }

    /// Computes required workspace sizes for batch decode operations.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of sequences
    /// * `max_seq_len` - Maximum sequence length
    /// * `num_qo_heads` - Number of query/output heads
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Head dimension
    /// * `page_size` - Page size for paged KV cache
    ///
    /// # Returns
    ///
    /// Tuple of (float_workspace_size, int_workspace_size)
    pub fn batch_decode_sizes(
        batch_size: u32,
        max_seq_len: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
    ) -> (usize, usize) {
        // Float workspace: tmp_v for split-k, tmp_s for softmax
        let tmp_v_size = batch_size as usize * num_qo_heads as usize * head_dim as usize * 4; // f32
        let tmp_s_size = batch_size as usize * num_qo_heads as usize * 4; // f32
        let float_size = tmp_v_size + tmp_s_size + 4096; // padding

        // Int workspace: partition info, page indices
        let max_num_pages = (max_seq_len + page_size - 1) / page_size;
        let partition_info_size = batch_size as usize * num_kv_heads as usize * 2 * 4; // int32
        let page_indices_size = batch_size as usize * max_num_pages as usize * 4; // int32
        let int_size = partition_info_size + page_indices_size + 4096; // padding

        (float_size, int_size)
    }

    /// Computes required workspace sizes for batch prefill operations.
    pub fn batch_prefill_sizes(
        batch_size: u32,
        total_tokens: u32,
        num_qo_heads: u32,
        _num_kv_heads: u32,
        head_dim: u32,
        _page_size: u32,
    ) -> (usize, usize) {
        // Float workspace: temporary output
        let tmp_size = total_tokens as usize * num_qo_heads as usize * head_dim as usize * 4;
        let float_size = tmp_size + 8192;

        // Int workspace: request info, tile info
        let request_info_size = batch_size as usize * 4 * 4; // int32
        let tile_info_size = total_tokens as usize * 4; // int32
        let int_size = request_info_size + tile_info_size + 4096;

        (float_size, int_size)
    }

    /// Computes general workspace sizes for given configuration.
    pub fn required_sizes(
        batch_size: u32,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        page_size: u32,
    ) -> (usize, usize) {
        Self::batch_decode_sizes(
            batch_size,
            max_seq_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
    }
}

/// Workspace size query without CUDA (for planning).
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceSizes {
    /// Float workspace size in bytes.
    pub float_size: usize,
    /// Int workspace size in bytes.
    pub int_size: usize,
}

impl WorkspaceSizes {
    /// Creates workspace sizes for batch decode.
    pub fn for_batch_decode(
        batch_size: u32,
        max_seq_len: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
    ) -> Self {
        let tmp_v_size = batch_size as usize * num_qo_heads as usize * head_dim as usize * 4;
        let tmp_s_size = batch_size as usize * num_qo_heads as usize * 4;
        let float_size = tmp_v_size + tmp_s_size + 4096;

        let max_num_pages = (max_seq_len + page_size - 1) / page_size;
        let partition_info_size = batch_size as usize * num_kv_heads as usize * 2 * 4;
        let page_indices_size = batch_size as usize * max_num_pages as usize * 4;
        let int_size = partition_info_size + page_indices_size + 4096;

        Self {
            float_size,
            int_size,
        }
    }

    /// Creates workspace sizes for batch prefill.
    pub fn for_batch_prefill(
        batch_size: u32,
        total_tokens: u32,
        num_qo_heads: u32,
        head_dim: u32,
    ) -> Self {
        let tmp_size = total_tokens as usize * num_qo_heads as usize * head_dim as usize * 4;
        let float_size = tmp_size + 8192;

        let request_info_size = batch_size as usize * 4 * 4;
        let tile_info_size = total_tokens as usize * 4;
        let int_size = request_info_size + tile_info_size + 4096;

        Self {
            float_size,
            int_size,
        }
    }

    /// Returns total size (float + int).
    #[inline]
    pub fn total(&self) -> usize {
        self.float_size + self.int_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_sizes_batch_decode() {
        let sizes = WorkspaceSizes::for_batch_decode(
            4,    // batch_size
            2048, // max_seq_len
            32,   // num_qo_heads
            8,    // num_kv_heads
            128,  // head_dim
            16,   // page_size
        );

        // tmp_v = 4 * 32 * 128 * 4 = 65536
        // tmp_s = 4 * 32 * 4 = 512
        // float = 65536 + 512 + 4096 = 70144
        assert!(sizes.float_size >= 65536 + 512);

        // max_pages = 2048 / 16 = 128
        // partition = 4 * 8 * 2 * 4 = 256
        // page_indices = 4 * 128 * 4 = 2048
        assert!(sizes.int_size >= 256 + 2048);
    }

    #[test]
    fn test_workspace_sizes_batch_prefill() {
        let sizes = WorkspaceSizes::for_batch_prefill(
            4,    // batch_size
            1024, // total_tokens
            32,   // num_qo_heads
            128,  // head_dim
        );

        // tmp = 1024 * 32 * 128 * 4 = 16777216
        assert!(sizes.float_size >= 1024 * 32 * 128 * 4);
    }

    #[test]
    fn test_workspace_sizes_total() {
        let sizes = WorkspaceSizes {
            float_size: 1000,
            int_size: 500,
        };
        assert_eq!(sizes.total(), 1500);
    }
}

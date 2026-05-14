//! Paged KV cache implementation.
//!
//! This module provides the GPU-side PagedKVCache struct for managing
//! key-value caches in transformer inference.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaStream, CudaSlice, DevicePtr, DevicePtrMut};

#[cfg(feature = "cuda")]
use std::marker::PhantomData;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::types::DType;
use crate::types::{GpuFloat, KVLayout};
use crate::Result;

/// Metadata for a batch of sequences in paged KV cache.
///
/// This structure holds the auxiliary data needed to index into
/// the paged KV cache for a batch of sequences.
#[derive(Debug, Clone)]
pub struct PagedKVMetadata {
    /// Cumulative page counts: `[batch_size + 1]`.
    /// `indptr[i+1] - indptr[i]` gives the number of pages for sequence i.
    pub indptr: Vec<i32>,

    /// Page indices for all sequences: `[total_pages]`.
    /// Maps logical page positions to physical page IDs.
    pub indices: Vec<i32>,

    /// Number of valid tokens in the last page: `[batch_size]`.
    pub last_page_len: Vec<i32>,

    /// Optional position offsets for RoPE: `[batch_size]`.
    pub rope_pos_offset: Option<Vec<i32>>,
}

impl PagedKVMetadata {
    /// Create new metadata with given capacity.
    pub fn new(batch_size: usize, max_total_pages: usize) -> Self {
        Self {
            indptr: vec![0; batch_size + 1],
            indices: Vec::with_capacity(max_total_pages),
            last_page_len: vec![0; batch_size],
            rope_pos_offset: None,
        }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Get the total number of pages across all sequences.
    pub fn total_pages(&self) -> usize {
        self.indices.len()
    }

    /// Get the number of pages for a specific sequence.
    pub fn num_pages(&self, seq_idx: usize) -> usize {
        if seq_idx >= self.batch_size() {
            return 0;
        }
        (self.indptr[seq_idx + 1] - self.indptr[seq_idx]) as usize
    }

    /// Set metadata for a single sequence.
    ///
    /// # Arguments
    ///
    /// * `seq_idx` - Sequence index in the batch
    /// * `page_indices` - Physical page indices for this sequence
    /// * `last_page_tokens` - Number of valid tokens in the last page
    pub fn set_sequence(
        &mut self,
        seq_idx: usize,
        page_indices: &[i32],
        last_page_tokens: i32,
    ) -> Result<()> {
        if seq_idx >= self.batch_size() {
            return Err(crate::FlashInferError::invalid_config(format!(
                "sequence index {} out of bounds (batch_size={})",
                seq_idx,
                self.batch_size()
            )));
        }

        // Update indptr for this sequence
        let start = self.indptr[seq_idx] as usize;
        let new_end = start + page_indices.len();

        // Shift subsequent sequences if necessary
        let old_end = self.indptr[seq_idx + 1] as usize;
        let diff = new_end as i32 - old_end as i32;

        if diff != 0 {
            for i in (seq_idx + 1)..self.indptr.len() {
                self.indptr[i] += diff;
            }
        }

        // Resize indices vector if needed
        let total_pages = *self.indptr.last().unwrap_or(&0) as usize;
        self.indices.resize(total_pages, 0);

        // Copy page indices
        self.indices[start..new_end].copy_from_slice(page_indices);

        // Set last page length
        self.last_page_len[seq_idx] = last_page_tokens;

        Ok(())
    }

    /// Compute sequence lengths from page metadata.
    ///
    /// Returns the total number of tokens for each sequence.
    pub fn get_seq_lens(&self, page_size: u32) -> Vec<i32> {
        let batch_size = self.batch_size();
        let mut seq_lens = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let num_pages = self.num_pages(i);
            if num_pages == 0 {
                seq_lens.push(0);
            } else {
                let full_pages = (num_pages - 1) as i32;
                let seq_len = full_pages * page_size as i32 + self.last_page_len[i];
                seq_lens.push(seq_len);
            }
        }

        seq_lens
    }

    /// Clear all metadata.
    pub fn clear(&mut self) {
        self.indptr.fill(0);
        self.indices.clear();
        self.last_page_len.fill(0);
        if let Some(ref mut offsets) = self.rope_pos_offset {
            offsets.fill(0);
        }
    }

    /// Validate metadata consistency.
    ///
    /// Note: FlashInfer's MLA kernel widens page indices to `int64_t` before
    /// the `* stride_page` multiply (upstream PR #3136), so the total cache
    /// footprint may exceed 2^31 bytes. The page *count* itself is still
    /// limited to `i32::MAX` because `indices`/`indptr` are still `int32_t`
    /// arrays at the FFI boundary.
    pub fn validate(&self) -> Result<()> {
        // Check indptr is monotonically increasing
        for i in 0..self.indptr.len() - 1 {
            if self.indptr[i] > self.indptr[i + 1] {
                return Err(crate::FlashInferError::invalid_config(
                    "indptr must be monotonically increasing",
                ));
            }
        }

        // Check indices length matches indptr
        let expected_pages = *self.indptr.last().unwrap_or(&0) as usize;
        if self.indices.len() != expected_pages {
            return Err(crate::FlashInferError::invalid_config(format!(
                "indices length {} doesn't match expected {}",
                self.indices.len(),
                expected_pages
            )));
        }

        // FFI passes indices/indptr as int32_t arrays — explicit check so the
        // caller gets a clear error rather than a silent overflow at the C++ boundary.
        if self.indices.len() > i32::MAX as usize {
            return Err(crate::FlashInferError::invalid_config(format!(
                "total page count {} exceeds i32::MAX; FlashInfer indices are int32_t",
                self.indices.len()
            )));
        }

        // Check last_page_len length matches batch size
        if self.last_page_len.len() != self.batch_size() {
            return Err(crate::FlashInferError::invalid_config(
                "last_page_len length doesn't match batch size",
            ));
        }

        Ok(())
    }
}

/// GPU-side paged KV cache with block-based memory management.
///
/// This struct manages the GPU memory for key and value tensors
/// using a paging approach that enables efficient memory utilization.
#[cfg(feature = "cuda")]
pub struct PagedKVCache<T: GpuFloat> {
    /// Key storage: `[num_pages, page_layout...]`
    k_data: CudaSlice<T>,

    /// Value storage: `[num_pages, page_layout...]`
    v_data: CudaSlice<T>,

    /// Page indices on GPU: `[total_pages]`
    indices_gpu: Option<CudaSlice<i32>>,

    /// Cumulative page counts on GPU: `[batch_size + 1]`
    indptr_gpu: Option<CudaSlice<i32>>,

    /// Last page lengths on GPU: `[batch_size]`
    last_page_len_gpu: Option<CudaSlice<i32>>,

    /// Current batch size (set by set_batch_metadata)
    batch_size: usize,

    /// Stream reference
    stream: Arc<CudaStream>,

    /// Maximum number of pages in the pool
    max_num_pages: usize,

    /// Tokens per page
    page_size: u32,

    /// Number of KV heads
    num_kv_heads: u32,

    /// Head dimension
    head_dim: u32,

    /// Memory layout
    layout: KVLayout,

    /// Data type
    dtype: DType,

    /// Type marker
    _marker: PhantomData<T>,
}

#[cfg(feature = "cuda")]
impl<T: GpuFloat> PagedKVCache<T> {
    /// Create a new paged KV cache.
    ///
    /// # Arguments
    ///
    /// * `stream` - CUDA stream
    /// * `max_num_pages` - Maximum number of pages to allocate
    /// * `page_size` - Number of tokens per page
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension of each head
    /// * `layout` - Memory layout (NHD or HND)
    pub fn new(
        stream: Arc<CudaStream>,
        max_num_pages: usize,
        page_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        layout: KVLayout,
    ) -> Result<Self> {
        // Calculate page size in elements based on layout
        let elements_per_page = match layout {
            KVLayout::NHD => page_size as usize * num_kv_heads as usize * head_dim as usize,
            KVLayout::HND => num_kv_heads as usize * page_size as usize * head_dim as usize,
        };

        let total_elements = max_num_pages * elements_per_page;

        // Allocate K and V buffers
        let k_data = stream
            .alloc_zeros::<T>(total_elements)
            .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;

        let v_data = stream
            .alloc_zeros::<T>(total_elements)
            .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;

        Ok(Self {
            k_data,
            v_data,
            indices_gpu: None,
            indptr_gpu: None,
            last_page_len_gpu: None,
            batch_size: 0,
            stream,
            max_num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            layout,
            dtype: T::DTYPE,
            _marker: PhantomData,
        })
    }

    /// Get the key storage buffer.
    pub fn k_data(&self) -> &CudaSlice<T> {
        &self.k_data
    }

    /// Get the value storage buffer.
    pub fn v_data(&self) -> &CudaSlice<T> {
        &self.v_data
    }

    /// Get mutable key storage buffer.
    pub fn k_data_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.k_data
    }

    /// Get mutable value storage buffer.
    pub fn v_data_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.v_data
    }

    /// Get the page indices on GPU.
    pub fn indices_gpu(&self) -> Option<&CudaSlice<i32>> {
        self.indices_gpu.as_ref()
    }

    /// Get the indptr on GPU.
    pub fn indptr_gpu(&self) -> Option<&CudaSlice<i32>> {
        self.indptr_gpu.as_ref()
    }

    /// Get the last page lengths on GPU.
    pub fn last_page_len_gpu(&self) -> Option<&CudaSlice<i32>> {
        self.last_page_len_gpu.as_ref()
    }

    /// Get maximum number of pages.
    pub fn max_num_pages(&self) -> usize {
        self.max_num_pages
    }

    /// Get page size.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Get number of KV heads.
    pub fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }

    /// Get head dimension.
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Get layout.
    pub fn layout(&self) -> KVLayout {
        self.layout
    }

    /// Get data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get stream reference.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Upload batch metadata to GPU.
    ///
    /// This transfers the indptr, indices, and last_page_len arrays
    /// from host to device memory.
    pub fn set_batch_metadata(&mut self, metadata: &PagedKVMetadata) -> Result<()> {
        metadata.validate()?;

        // Store batch size
        self.batch_size = metadata.batch_size();

        // Upload indptr
        let indptr_gpu = self
            .stream
            .memcpy_stod(&metadata.indptr)
            .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;
        self.indptr_gpu = Some(indptr_gpu);

        // Upload indices
        if !metadata.indices.is_empty() {
            let indices_gpu = self
                .stream
                .memcpy_stod(&metadata.indices)
                .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;
            self.indices_gpu = Some(indices_gpu);
        } else {
            self.indices_gpu = None;
        }

        // Upload last_page_len
        let last_page_len_gpu = self
            .stream
            .memcpy_stod(&metadata.last_page_len)
            .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;
        self.last_page_len_gpu = Some(last_page_len_gpu);

        Ok(())
    }

    /// Append new KV pairs during decode phase.
    ///
    /// Appends a single token's KV for each sequence in the batch.
    ///
    /// # Arguments
    ///
    /// * `key` - Key tensor `[batch_size, num_kv_heads, head_dim]`
    /// * `value` - Value tensor `[batch_size, num_kv_heads, head_dim]`
    /// * `stream` - CUDA stream
    ///
    /// # Note
    ///
    /// You must call `set_batch_metadata` before calling this function to upload
    /// the page table metadata to the GPU.
    pub fn append_decode(
        &mut self,
        key: &CudaSlice<T>,
        value: &CudaSlice<T>,
        stream: &CudaStream,
    ) -> Result<()> {
        // Check that metadata is uploaded
        let indptr_gpu = self.indptr_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;
        let indices_gpu = self.indices_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;
        let last_page_len_gpu = self.last_page_len_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;

        // Use stored batch size
        let batch_size = self.batch_size as i32;
        if batch_size <= 0 {
            return Ok(());
        }

        // For decode, each sequence appends exactly 1 token
        // Create append_indptr: [0, 1, 2, ..., batch_size]
        let append_indptr: Vec<i32> = (0..=batch_size).collect();
        let append_indptr_gpu = self
            .stream
            .memcpy_stod(&append_indptr)
            .map_err(|e| crate::FlashInferError::cuda(e.to_string()))?;

        // Extract device pointers
        let (key_ptr, _key_guard) = key.device_ptr(stream);
        let (value_ptr, _value_guard) = value.device_ptr(stream);
        let (k_data_ptr, _k_data_guard) = self.k_data.device_ptr_mut(stream);
        let (v_data_ptr, _v_data_guard) = self.v_data.device_ptr_mut(stream);
        let (indptr_ptr, _indptr_guard) = indptr_gpu.device_ptr(stream);
        let (indices_ptr, _indices_guard) = indices_gpu.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = last_page_len_gpu.device_ptr(stream);
        let (append_indptr_ptr, _ai_guard) = append_indptr_gpu.device_ptr(stream);

        // Call FFI
        unsafe {
            crate::ffi::append_paged_kv_cache(
                key_ptr as *const std::ffi::c_void,
                value_ptr as *const std::ffi::c_void,
                k_data_ptr as *mut std::ffi::c_void,
                v_data_ptr as *mut std::ffi::c_void,
                indptr_ptr as *const i32,
                indices_ptr as *const i32,
                last_page_ptr as *const i32,
                append_indptr_ptr as *const i32,
                batch_size,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                self.page_size as i32,
                self.dtype.into(),
                self.layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        }
    }

    /// Append new KV pairs during prefill phase.
    ///
    /// Appends variable-length KV for each sequence in the batch.
    ///
    /// # Arguments
    ///
    /// * `keys` - Key tensor `[total_tokens, num_kv_heads, head_dim]`
    /// * `values` - Value tensor `[total_tokens, num_kv_heads, head_dim]`
    /// * `append_indptr` - Token offsets for append `[batch_size + 1]`
    ///   (e.g., `[0, 5, 12, 20]` means seq 0 appends tokens 0..5, seq 1 appends 5..12, etc.)
    /// * `stream` - CUDA stream
    ///
    /// # Note
    ///
    /// You must call `set_batch_metadata` before calling this function to upload
    /// the page table metadata to the GPU.
    pub fn append_prefill(
        &mut self,
        keys: &CudaSlice<T>,
        values: &CudaSlice<T>,
        append_indptr: &CudaSlice<i32>,
        stream: &CudaStream,
    ) -> Result<()> {
        // Check that metadata is uploaded
        let indptr_gpu = self.indptr_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;
        let indices_gpu = self.indices_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;
        let last_page_len_gpu = self.last_page_len_gpu.as_ref().ok_or_else(|| {
            crate::FlashInferError::invalid_config(
                "batch metadata not uploaded - call set_batch_metadata first",
            )
        })?;

        // Use stored batch size
        let batch_size = self.batch_size as i32;
        if batch_size <= 0 {
            return Ok(());
        }

        // Extract device pointers
        let (keys_ptr, _keys_guard) = keys.device_ptr(stream);
        let (values_ptr, _values_guard) = values.device_ptr(stream);
        let (k_data_ptr, _k_data_guard) = self.k_data.device_ptr_mut(stream);
        let (v_data_ptr, _v_data_guard) = self.v_data.device_ptr_mut(stream);
        let (indptr_ptr, _indptr_guard) = indptr_gpu.device_ptr(stream);
        let (indices_ptr, _indices_guard) = indices_gpu.device_ptr(stream);
        let (last_page_ptr, _lp_guard) = last_page_len_gpu.device_ptr(stream);
        let (append_indptr_ptr, _ai_guard) = append_indptr.device_ptr(stream);

        // Call FFI
        unsafe {
            crate::ffi::append_paged_kv_cache(
                keys_ptr as *const std::ffi::c_void,
                values_ptr as *const std::ffi::c_void,
                k_data_ptr as *mut std::ffi::c_void,
                v_data_ptr as *mut std::ffi::c_void,
                indptr_ptr as *const i32,
                indices_ptr as *const i32,
                last_page_ptr as *const i32,
                append_indptr_ptr as *const i32,
                batch_size,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                self.page_size as i32,
                self.dtype.into(),
                self.layout.into(),
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        }
    }

    /// Get raw device pointers for FFI.
    ///
    /// Returns (k_ptr, v_ptr) as raw device pointers.
    /// Requires a stream reference for pointer extraction in cudarc 0.16.
    pub fn device_ptrs(&self, stream: &CudaStream) -> (*const T, *const T) {
        let (k_ptr, _k_guard) = self.k_data.device_ptr(stream);
        let (v_ptr, _v_guard) = self.v_data.device_ptr(stream);
        (
            k_ptr as *const T,
            v_ptr as *const T,
        )
    }
}

/// Builder for PagedKVCache with configuration options.
#[derive(Debug, Clone)]
pub struct PagedKVCacheBuilder {
    max_num_pages: usize,
    page_size: u32,
    num_kv_heads: u32,
    head_dim: u32,
    layout: KVLayout,
}

impl PagedKVCacheBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            max_num_pages: 1024,
            page_size: 16,
            num_kv_heads: 8,
            head_dim: 128,
            layout: KVLayout::NHD,
        }
    }

    /// Set maximum number of pages.
    pub fn max_num_pages(mut self, max_num_pages: usize) -> Self {
        self.max_num_pages = max_num_pages;
        self
    }

    /// Set page size (tokens per page).
    pub fn page_size(mut self, page_size: u32) -> Self {
        self.page_size = page_size;
        self
    }

    /// Set number of KV heads.
    pub fn num_kv_heads(mut self, num_kv_heads: u32) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    /// Set head dimension.
    pub fn head_dim(mut self, head_dim: u32) -> Self {
        self.head_dim = head_dim;
        self
    }

    /// Set memory layout.
    pub fn layout(mut self, layout: KVLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Calculate required GPU memory in bytes.
    pub fn required_memory<T: GpuFloat>(&self) -> usize {
        let elements_per_page = match self.layout {
            KVLayout::NHD => {
                self.page_size as usize * self.num_kv_heads as usize * self.head_dim as usize
            }
            KVLayout::HND => {
                self.num_kv_heads as usize * self.page_size as usize * self.head_dim as usize
            }
        };

        // K + V buffers
        2 * self.max_num_pages * elements_per_page * std::mem::size_of::<T>()
    }

    /// Build the PagedKVCache.
    #[cfg(feature = "cuda")]
    pub fn build<T: GpuFloat>(self, stream: Arc<CudaStream>) -> Result<PagedKVCache<T>> {
        PagedKVCache::new(
            stream,
            self.max_num_pages,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            self.layout,
        )
    }
}

impl Default for PagedKVCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_metadata_new() {
        let metadata = PagedKVMetadata::new(4, 64);
        assert_eq!(metadata.batch_size(), 4);
        assert_eq!(metadata.total_pages(), 0);
        assert_eq!(metadata.indptr.len(), 5);
        assert_eq!(metadata.last_page_len.len(), 4);
    }

    #[test]
    fn test_paged_kv_metadata_set_sequence() {
        let mut metadata = PagedKVMetadata::new(2, 10);

        // Set first sequence with 3 pages
        metadata.set_sequence(0, &[0, 1, 2], 8).unwrap();
        assert_eq!(metadata.num_pages(0), 3);
        assert_eq!(metadata.indptr, vec![0, 3, 3]);
        assert_eq!(metadata.indices, vec![0, 1, 2]);
        assert_eq!(metadata.last_page_len[0], 8);

        // Set second sequence with 2 pages
        metadata.set_sequence(1, &[5, 6], 12).unwrap();
        assert_eq!(metadata.num_pages(1), 2);
        assert_eq!(metadata.indptr, vec![0, 3, 5]);
        assert_eq!(metadata.indices, vec![0, 1, 2, 5, 6]);
        assert_eq!(metadata.last_page_len[1], 12);
    }

    #[test]
    fn test_paged_kv_metadata_get_seq_lens() {
        let mut metadata = PagedKVMetadata::new(2, 10);
        let page_size = 16;

        // Seq 0: 3 pages, last page has 8 tokens = 2*16 + 8 = 40 tokens
        metadata.set_sequence(0, &[0, 1, 2], 8).unwrap();

        // Seq 1: 1 page, last page has 5 tokens = 5 tokens
        metadata.set_sequence(1, &[3], 5).unwrap();

        let seq_lens = metadata.get_seq_lens(page_size);
        assert_eq!(seq_lens, vec![40, 5]);
    }

    #[test]
    fn test_paged_kv_metadata_validation() {
        let mut metadata = PagedKVMetadata::new(2, 10);
        metadata.set_sequence(0, &[0, 1], 8).unwrap();
        assert!(metadata.validate().is_ok());

        // Create invalid metadata with wrong indices length
        let mut invalid = PagedKVMetadata::new(2, 10);
        invalid.indptr = vec![0, 2, 4];
        invalid.indices = vec![0, 1]; // Should be 4 elements
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_paged_kv_metadata_clear() {
        let mut metadata = PagedKVMetadata::new(2, 10);
        metadata.set_sequence(0, &[0, 1, 2], 8).unwrap();
        metadata.set_sequence(1, &[3, 4], 5).unwrap();

        metadata.clear();

        assert_eq!(metadata.total_pages(), 0);
        assert!(metadata.indptr.iter().all(|&x| x == 0));
        assert!(metadata.last_page_len.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_paged_kv_cache_builder() {
        let builder = PagedKVCacheBuilder::new()
            .max_num_pages(2048)
            .page_size(32)
            .num_kv_heads(16)
            .head_dim(128)
            .layout(KVLayout::HND);

        assert_eq!(builder.max_num_pages, 2048);
        assert_eq!(builder.page_size, 32);
        assert_eq!(builder.num_kv_heads, 16);
        assert_eq!(builder.head_dim, 128);
        assert_eq!(builder.layout, KVLayout::HND);
    }

    #[test]
    fn test_paged_kv_cache_builder_memory_calculation() {
        let builder = PagedKVCacheBuilder::new()
            .max_num_pages(1024)
            .page_size(16)
            .num_kv_heads(8)
            .head_dim(128);

        // Elements per page: 16 * 8 * 128 = 16384
        // Total elements: 1024 * 16384 = 16,777,216
        // Memory for fp16: 16,777,216 * 2 bytes = 33,554,432 bytes per buffer
        // K + V: 67,108,864 bytes = 64 MB
        let memory = builder.required_memory::<half::f16>();
        assert_eq!(memory, 64 * 1024 * 1024);
    }
}

//! FFI bindings to FlashInfer C++/CUDA kernels.
//!
//! This module will contain the raw FFI declarations for FlashInfer kernels.
//! The bindings are generated using bindgen from FlashInfer's C++ headers.
//!
//! # Building
//!
//! To build the FFI bindings, you need:
//! 1. FlashInfer source code (submodule or downloaded)
//! 2. CUDA toolkit installed
//! 3. C++ compiler with C++17 support
//!
//! The build.rs script handles compilation of the CUDA kernels and
//! generation of Rust bindings.

// TODO: Add bindgen-generated FFI declarations
//
// Expected functions to wrap:
//
// BatchDecodeWithPagedKVCache:
// - BatchDecodeWithPagedKVCacheWrapperBeginForward
// - BatchDecodeWithPagedKVCacheWrapperForward
// - BatchDecodeWithPagedKVCacheWrapperEndForward
//
// BatchPrefillWithPagedKVCache:
// - BatchPrefillWithPagedKVCacheWrapperBeginForward
// - BatchPrefillWithPagedKVCacheWrapperForward
// - BatchPrefillWithPagedKVCacheWrapperEndForward
//
// Append operations:
// - AppendPagedKVCache

/// Placeholder for FFI error codes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashInferStatus {
    Success = 0,
    CudaError = 1,
    InvalidArgument = 2,
    OutOfMemory = 3,
    InternalError = 4,
}

impl FlashInferStatus {
    pub fn is_ok(&self) -> bool {
        *self == FlashInferStatus::Success
    }
}

// When we have actual bindings, they will look something like:
//
// extern "C" {
//     pub fn flashinfer_batch_decode_f16(
//         output: *mut c_void,
//         query: *const c_void,
//         kv_cache: *const c_void,
//         page_table: *const i32,
//         kv_lengths: *const i32,
//         batch_size: i32,
//         num_qo_heads: i32,
//         num_kv_heads: i32,
//         head_dim: i32,
//         page_size: i32,
//         workspace: *mut c_void,
//         workspace_size: usize,
//         stream: cudaStream_t,
//     ) -> FlashInferStatus;
// }

//! Core types for FlashInfer Rust bindings.
//!
//! This module defines the fundamental types used throughout the library,
//! including data types, layouts, encoding modes, and the `GpuFloat` trait
//! for type-safe GPU operations.

use std::fmt;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum DType {
    /// 16-bit floating point (IEEE 754 half precision)
    #[default]
    Float16 = 0,
    /// 16-bit brain floating point
    BFloat16 = 1,
    /// 32-bit floating point (IEEE 754 single precision)
    Float32 = 2,
    /// 8-bit floating point (E4M3 format)
    Float8E4M3 = 3,
    /// 8-bit floating point (E5M2 format)
    Float8E5M2 = 4,
}

impl DType {
    /// Returns the size of this data type in bytes.
    #[inline]
    pub const fn size_bytes(&self) -> usize {
        match self {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float8E4M3 | DType::Float8E5M2 => 1,
        }
    }

    /// Returns whether this is a floating point 8 type.
    #[inline]
    pub const fn is_fp8(&self) -> bool {
        matches!(self, DType::Float8E4M3 | DType::Float8E5M2)
    }

    /// Returns whether this is a 16-bit type.
    #[inline]
    pub const fn is_16bit(&self) -> bool {
        matches!(self, DType::Float16 | DType::BFloat16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float16 => write!(f, "float16"),
            DType::BFloat16 => write!(f, "bfloat16"),
            DType::Float32 => write!(f, "float32"),
            DType::Float8E4M3 => write!(f, "float8_e4m3"),
            DType::Float8E5M2 => write!(f, "float8_e5m2"),
        }
    }
}

/// KV cache memory layout.
///
/// Determines how the key-value cache tensors are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum KVLayout {
    /// Head-first layout: `[num_pages, num_heads, page_size, head_dim]`
    HND = 0,
    /// Sequence-first layout: `[num_pages, page_size, num_heads, head_dim]`
    #[default]
    NHD = 1,
}

impl fmt::Display for KVLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVLayout::HND => write!(f, "HND"),
            KVLayout::NHD => write!(f, "NHD"),
        }
    }
}

/// Position encoding mode for attention computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum PosEncodingMode {
    /// No positional encoding applied
    #[default]
    None = 0,
    /// Llama-style rotary position embedding (RoPE)
    RoPELlama = 1,
    /// Attention with Linear Biases (ALiBi)
    ALiBi = 2,
    /// RoPE with frequency scaling (Llama 3.1 style)
    RoPELlamaFreqScale = 3,
}

impl fmt::Display for PosEncodingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PosEncodingMode::None => write!(f, "none"),
            PosEncodingMode::RoPELlama => write!(f, "rope_llama"),
            PosEncodingMode::ALiBi => write!(f, "alibi"),
            PosEncodingMode::RoPELlamaFreqScale => write!(f, "rope_llama_freq_scale"),
        }
    }
}

/// Attention mask mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum MaskMode {
    /// No masking (full attention)
    #[default]
    None = 0,
    /// Causal masking (autoregressive)
    Causal = 1,
    /// Custom mask provided by user
    Custom = 2,
}

impl fmt::Display for MaskMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskMode::None => write!(f, "none"),
            MaskMode::Causal => write!(f, "causal"),
            MaskMode::Custom => write!(f, "custom"),
        }
    }
}

/// Attention computation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum Backend {
    /// Automatically select the best backend
    #[default]
    Auto = 0,
    /// FlashAttention-2 (SM80+)
    FA2 = 1,
    /// FlashAttention-3 (SM90+ only)
    FA3 = 2,
    /// cuDNN backend
    CuDNN = 3,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Auto => write!(f, "auto"),
            Backend::FA2 => write!(f, "fa2"),
            Backend::FA3 => write!(f, "fa3"),
            Backend::CuDNN => write!(f, "cudnn"),
        }
    }
}

/// Trait for types that can be used in GPU floating-point operations.
///
/// This trait provides a type-safe way to work with different floating-point
/// types on the GPU. It maps Rust types to their FFI counterparts and provides
/// metadata about the types.
///
/// # Safety
///
/// This trait is unsafe to implement because incorrect implementations could
/// lead to memory corruption when interfacing with CUDA kernels.
///
/// When the `cuda` feature is enabled, this trait also requires cudarc's
/// `DeviceRepr` and `ValidAsZeroBits` traits for safe GPU memory operations.
#[cfg(feature = "cuda")]
pub unsafe trait GpuFloat:
    Copy + Send + Sync + 'static + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits
{
    /// The corresponding FFI data type enum value.
    const DTYPE: DType;

    /// Size in bytes of a single element.
    const SIZE_BYTES: usize;

    /// Returns the zero value for this type.
    fn zero() -> Self;

    /// Returns the one value for this type.
    fn one() -> Self;

    /// Converts to f32 for debugging/testing purposes.
    fn to_f32(self) -> f32;

    /// Creates from f32 for testing purposes.
    fn from_f32(val: f32) -> Self;
}

/// Trait for types that can be used in GPU floating-point operations.
///
/// This trait provides a type-safe way to work with different floating-point
/// types on the GPU. It maps Rust types to their FFI counterparts and provides
/// metadata about the types.
///
/// # Safety
///
/// This trait is unsafe to implement because incorrect implementations could
/// lead to memory corruption when interfacing with CUDA kernels.
#[cfg(not(feature = "cuda"))]
pub unsafe trait GpuFloat: Copy + Send + Sync + 'static {
    /// The corresponding FFI data type enum value.
    const DTYPE: DType;

    /// Size in bytes of a single element.
    const SIZE_BYTES: usize;

    /// Returns the zero value for this type.
    fn zero() -> Self;

    /// Returns the one value for this type.
    fn one() -> Self;

    /// Converts to f32 for debugging/testing purposes.
    fn to_f32(self) -> f32;

    /// Creates from f32 for testing purposes.
    fn from_f32(val: f32) -> Self;
}

// Implement GpuFloat for half::f16
unsafe impl GpuFloat for half::f16 {
    const DTYPE: DType = DType::Float16;
    const SIZE_BYTES: usize = 2;

    #[inline]
    fn zero() -> Self {
        half::f16::from_f32(0.0)
    }

    #[inline]
    fn one() -> Self {
        half::f16::from_f32(1.0)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }

    #[inline]
    fn from_f32(val: f32) -> Self {
        half::f16::from_f32(val)
    }
}

// Implement GpuFloat for half::bf16
unsafe impl GpuFloat for half::bf16 {
    const DTYPE: DType = DType::BFloat16;
    const SIZE_BYTES: usize = 2;

    #[inline]
    fn zero() -> Self {
        half::bf16::from_f32(0.0)
    }

    #[inline]
    fn one() -> Self {
        half::bf16::from_f32(1.0)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }

    #[inline]
    fn from_f32(val: f32) -> Self {
        half::bf16::from_f32(val)
    }
}

// Implement GpuFloat for f32
unsafe impl GpuFloat for f32 {
    const DTYPE: DType = DType::Float32;
    const SIZE_BYTES: usize = 4;

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(val: f32) -> Self {
        val
    }
}

/// Head dimension configuration.
///
/// FlashInfer supports specific head dimensions for optimal performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadDim {
    /// 64-dimensional heads
    D64 = 64,
    /// 128-dimensional heads (most common)
    D128 = 128,
    /// 256-dimensional heads
    D256 = 256,
}

impl HeadDim {
    /// Creates a HeadDim from a raw value, if supported.
    pub fn from_value(dim: usize) -> Option<Self> {
        match dim {
            64 => Some(HeadDim::D64),
            128 => Some(HeadDim::D128),
            256 => Some(HeadDim::D256),
            _ => None,
        }
    }

    /// Returns the dimension value.
    #[inline]
    pub const fn value(&self) -> usize {
        *self as usize
    }
}

impl From<HeadDim> for usize {
    fn from(dim: HeadDim) -> Self {
        dim.value()
    }
}

impl fmt::Display for HeadDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float8E4M3.size_bytes(), 1);
        assert_eq!(DType::Float8E5M2.size_bytes(), 1);
    }

    #[test]
    fn test_dtype_is_fp8() {
        assert!(!DType::Float16.is_fp8());
        assert!(!DType::BFloat16.is_fp8());
        assert!(!DType::Float32.is_fp8());
        assert!(DType::Float8E4M3.is_fp8());
        assert!(DType::Float8E5M2.is_fp8());
    }

    #[test]
    fn test_dtype_default() {
        assert_eq!(DType::default(), DType::Float16);
    }

    #[test]
    fn test_kv_layout_default() {
        assert_eq!(KVLayout::default(), KVLayout::NHD);
    }

    #[test]
    fn test_pos_encoding_default() {
        assert_eq!(PosEncodingMode::default(), PosEncodingMode::None);
    }

    #[test]
    fn test_mask_mode_default() {
        assert_eq!(MaskMode::default(), MaskMode::None);
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(Backend::default(), Backend::Auto);
    }

    #[test]
    fn test_head_dim_from_value() {
        assert_eq!(HeadDim::from_value(64), Some(HeadDim::D64));
        assert_eq!(HeadDim::from_value(128), Some(HeadDim::D128));
        assert_eq!(HeadDim::from_value(256), Some(HeadDim::D256));
        assert_eq!(HeadDim::from_value(100), None);
    }

    #[test]
    fn test_gpu_float_f16() {
        let zero = half::f16::zero();
        let one = half::f16::one();
        assert_eq!(zero.to_f32(), 0.0);
        assert_eq!(one.to_f32(), 1.0);
        assert_eq!(half::f16::DTYPE, DType::Float16);
        assert_eq!(half::f16::SIZE_BYTES, 2);
    }

    #[test]
    fn test_gpu_float_bf16() {
        let zero = half::bf16::zero();
        let one = half::bf16::one();
        assert_eq!(zero.to_f32(), 0.0);
        assert_eq!(one.to_f32(), 1.0);
        assert_eq!(half::bf16::DTYPE, DType::BFloat16);
        assert_eq!(half::bf16::SIZE_BYTES, 2);
    }

    #[test]
    fn test_gpu_float_f32() {
        let zero = f32::zero();
        let one = f32::one();
        assert_eq!(zero, 0.0);
        assert_eq!(one, 1.0);
        assert_eq!(f32::DTYPE, DType::Float32);
        assert_eq!(f32::SIZE_BYTES, 4);
    }

    #[test]
    fn test_display_traits() {
        assert_eq!(format!("{}", DType::Float16), "float16");
        assert_eq!(format!("{}", KVLayout::NHD), "NHD");
        assert_eq!(format!("{}", PosEncodingMode::RoPELlama), "rope_llama");
        assert_eq!(format!("{}", MaskMode::Causal), "causal");
        assert_eq!(format!("{}", Backend::FA3), "fa3");
        assert_eq!(format!("{}", HeadDim::D128), "128");
    }
}

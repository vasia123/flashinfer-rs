//! Normalization operations.
//!
//! This module provides RMSNorm and LayerNorm operations for LLM inference.
//!
//! # Example
//!
//! ```ignore
//! use flashinfer_rs::ops::norm::rmsnorm;
//! use flashinfer_rs::config::NormConfig;
//!
//! let config = NormConfig::new().with_eps(1e-6);
//! rmsnorm::<half::f16>(&input, &weight, &mut output, batch_size, hidden_dim, &config, &stream)?;
//! ```

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::config::NormConfig;
use crate::types::{DType, GpuFloat};
use crate::Result;

#[cfg(feature = "cuda")]
use crate::ffi;

/// Convert FFI status to Result (local helper to avoid importing from ffi module)
#[cfg(feature = "cuda")]
fn check_status(status: ffi::FlashInferStatus) -> Result<()> {
    use crate::FlashInferError;
    match status {
        ffi::FlashInferStatus::FLASHINFER_SUCCESS => Ok(()),
        ffi::FlashInferStatus::FLASHINFER_CUDA_ERROR => {
            let msg = ffi::get_last_error().unwrap_or_else(|| "Unknown CUDA error".to_string());
            Err(FlashInferError::cuda(msg))
        }
        ffi::FlashInferStatus::FLASHINFER_INVALID_ARGUMENT => {
            let msg = ffi::get_last_error().unwrap_or_else(|| "Invalid argument".to_string());
            Err(FlashInferError::invalid_config(msg))
        }
        ffi::FlashInferStatus::FLASHINFER_OUT_OF_MEMORY => {
            let msg = ffi::get_last_error().unwrap_or_else(|| "Out of memory".to_string());
            Err(FlashInferError::OutOfMemory(msg))
        }
        ffi::FlashInferStatus::FLASHINFER_UNSUPPORTED => {
            let msg = ffi::get_last_error().unwrap_or_else(|| "Unsupported operation".to_string());
            Err(FlashInferError::unsupported(msg))
        }
        _ => {
            let msg = ffi::get_last_error().unwrap_or_else(|| "Internal error".to_string());
            Err(FlashInferError::Internal(msg))
        }
    }
}

/// Convert types::DType to FlashInferDType
#[cfg(feature = "cuda")]
fn dtype_to_ffi(dtype: DType) -> ffi::FlashInferDType {
    match dtype {
        DType::Float16 => ffi::FlashInferDType::FLASHINFER_DTYPE_FLOAT16,
        DType::BFloat16 => ffi::FlashInferDType::FLASHINFER_DTYPE_BFLOAT16,
        DType::Float32 => ffi::FlashInferDType::FLASHINFER_DTYPE_FLOAT32,
        DType::Float8E4M3 => ffi::FlashInferDType::FLASHINFER_DTYPE_FLOAT8_E4M3,
        DType::Float8E5M2 => ffi::FlashInferDType::FLASHINFER_DTYPE_FLOAT8_E5M2,
    }
}

/// RMSNorm operation configuration.
#[derive(Debug, Clone)]
pub struct RMSNormParams {
    /// Hidden dimension (last dimension of input).
    pub hidden_dim: u32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Data type.
    pub dtype: DType,
}

impl RMSNormParams {
    /// Creates new RMSNorm parameters.
    pub fn new(hidden_dim: u32) -> Self {
        Self {
            hidden_dim,
            eps: 1e-5,
            dtype: DType::Float16,
        }
    }

    /// Sets epsilon value.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Sets data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
}

/// LayerNorm operation configuration.
#[derive(Debug, Clone)]
pub struct LayerNormParams {
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Whether to use bias.
    pub use_bias: bool,
    /// Data type.
    pub dtype: DType,
}

impl LayerNormParams {
    /// Creates new LayerNorm parameters.
    pub fn new(hidden_dim: u32) -> Self {
        Self {
            hidden_dim,
            eps: 1e-5,
            use_bias: true,
            dtype: DType::Float16,
        }
    }

    /// Sets epsilon value.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Disables bias.
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Sets data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
}

/// Apply RMSNorm to input tensor.
///
/// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`
/// * `weight` - Weight tensor of shape `[hidden_dim]`
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]`
/// * `batch_size` - Number of rows in input/output
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::ops::norm::rmsnorm;
/// use flashinfer_rs::config::NormConfig;
///
/// let config = NormConfig::new().with_eps(1e-6);
/// rmsnorm::<half::f16>(&input, &weight, &mut output, 32, 4096, &config, &stream)?;
/// ```
#[cfg(feature = "cuda")]
pub fn rmsnorm<T: GpuFloat>(
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    let status = unsafe {
        ffi::flashinfer_rmsnorm(
            *input.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Apply RMSNorm in-place.
///
/// This is more memory efficient than the out-of-place version
/// as it reuses the input buffer for output.
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`, modified in-place
/// * `weight` - Weight tensor of shape `[hidden_dim]`
/// * `batch_size` - Number of rows in input
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn rmsnorm_inplace<T: GpuFloat>(
    input: &mut CudaSlice<T>,
    weight: &CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    // In-place: input and output are the same buffer
    let ptr = *input.device_ptr();
    let status = unsafe {
        ffi::flashinfer_rmsnorm(
            ptr as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            ptr as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// RMSNorm with FP8 quantized output.
///
/// Applies RMSNorm and quantizes the output to FP8 format in a single fused operation.
/// This is useful for inference with quantized models.
///
/// **Requires SM89+ GPU (Ada Lovelace, Hopper, or newer).** Will return
/// `Unsupported` error on older GPUs.
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]` (FP16 or BF16)
/// * `weight` - Weight tensor of shape `[hidden_dim]` (same type as input)
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]` (FP8)
/// * `scale` - Quantization scale (output = normalized_value / scale, clamped to [-448, 448])
/// * `batch_size` - Number of rows
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `output_dtype` - Output FP8 type (Float8E4M3 or Float8E5M2)
/// * `stream` - CUDA stream
///
/// # Notes
///
/// - The scale parameter is a divisor: quantized = normalized / scale
/// - Values are clamped to [-448, 448] before conversion to FP8
/// - FP8 E4M3 format provides better precision for values near zero
/// - FP8 E5M2 format provides larger dynamic range
#[cfg(feature = "cuda")]
pub fn rmsnorm_quant<T: GpuFloat>(
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<u8>, // FP8 is stored as u8
    scale: f32,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    output_dtype: DType,
    stream: &CudaStream,
) -> Result<()> {
    // Validate output dtype is FP8
    if output_dtype != DType::Float8E4M3 && output_dtype != DType::Float8E5M2 {
        return Err(crate::FlashInferError::invalid_config(
            "rmsnorm_quant output must be FP8 (Float8E4M3 or Float8E5M2)",
        ));
    }

    // Validate input dtype is FP16 or BF16
    if T::DTYPE != DType::Float16 && T::DTYPE != DType::BFloat16 {
        return Err(crate::FlashInferError::invalid_config(
            "rmsnorm_quant input must be FP16 or BF16",
        ));
    }

    let mut scale_val = scale;
    let status = unsafe {
        ffi::flashinfer_rmsnorm_quant(
            *input.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            &mut scale_val,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            dtype_to_ffi(output_dtype),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Fused residual add and RMSNorm.
///
/// Computes: input += residual; output = RMSNorm(input) * weight
///
/// This operation fuses the residual connection with normalization
/// for better memory efficiency. Note that `input` is modified in-place
/// (residual is added to it) before normalization.
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`, modified in-place
/// * `residual` - Residual tensor of shape `[batch_size, hidden_dim]`
/// * `weight` - Weight tensor of shape `[hidden_dim]`
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]`
/// * `batch_size` - Number of rows
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn fused_add_rmsnorm<T: GpuFloat>(
    input: &mut CudaSlice<T>,
    residual: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    let status = unsafe {
        ffi::flashinfer_fused_add_rmsnorm(
            *input.device_ptr() as *mut std::ffi::c_void,
            *residual.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Apply LayerNorm to input tensor.
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`
/// * `weight` - Weight tensor of shape `[hidden_dim]`
/// * `bias` - Optional bias tensor of shape `[hidden_dim]`
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]`
/// * `batch_size` - Number of rows
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn layernorm<T: GpuFloat>(
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    bias: Option<&CudaSlice<T>>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    // NOTE: LayerNorm only supports float16 due to type compatibility issues in FlashInfer.
    // Use RMSNorm for bfloat16 workloads.
    if T::DTYPE != DType::Float16 {
        return Err(crate::FlashInferError::unsupported(
            "LayerNorm only supports float16. Use RMSNorm for bfloat16.",
        ));
    }

    let bias_ptr = bias
        .map(|b| *b.device_ptr() as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());

    let status = unsafe {
        ffi::flashinfer_layernorm(
            *input.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            bias_ptr,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Query-Key RMSNorm for attention heads.
///
/// Applies RMSNorm independently to each attention head.
/// Used in architectures that normalize Q and K before attention computation.
///
/// # Arguments
///
/// * `input` - Input tensor `[batch_size, num_heads, head_dim]`
/// * `weight` - Weight tensor `[head_dim]`
/// * `output` - Output tensor `[batch_size, num_heads, head_dim]`
/// * `batch_size` - Number of tokens
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Head dimension
/// * `eps` - Epsilon for numerical stability
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn qk_rmsnorm<T: GpuFloat>(
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let status = unsafe {
        ffi::flashinfer_qk_rmsnorm(
            *input.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            num_heads,
            head_dim,
            eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Gemma-style RMSNorm with weight bias of 1.0.
///
/// GemmaRMSNorm(x) = x * rsqrt(mean(x^2) + eps) * (weight + 1.0)
///
/// The "+1.0" to weight is a Gemma-specific modification that allows
/// the weight tensor to be initialized to zeros while still having
/// non-zero effective weights.
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`
/// * `weight` - Weight tensor of shape `[hidden_dim]` (will have 1.0 added)
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]`
/// * `batch_size` - Number of rows
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn gemma_rmsnorm<T: GpuFloat>(
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    let status = unsafe {
        ffi::flashinfer_gemma_rmsnorm(
            *input.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

/// Fused residual add and Gemma-style RMSNorm.
///
/// Computes: input += residual; output = GemmaRMSNorm(input) * (weight + 1.0)
///
/// This operation fuses the residual connection with Gemma normalization
/// for better memory efficiency. Note that `input` is modified in-place
/// (residual is added to it) before normalization.
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch_size, hidden_dim]`, modified in-place
/// * `residual` - Residual tensor of shape `[batch_size, hidden_dim]`
/// * `weight` - Weight tensor of shape `[hidden_dim]` (will have 1.0 added)
/// * `output` - Output tensor of shape `[batch_size, hidden_dim]`
/// * `batch_size` - Number of rows
/// * `hidden_dim` - Hidden dimension (number of columns)
/// * `config` - Normalization configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn gemma_fused_add_rmsnorm<T: GpuFloat>(
    input: &mut CudaSlice<T>,
    residual: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    batch_size: u32,
    hidden_dim: u32,
    config: &NormConfig,
    stream: &CudaStream,
) -> Result<()> {
    let status = unsafe {
        ffi::flashinfer_gemma_fused_add_rmsnorm(
            *input.device_ptr() as *mut std::ffi::c_void,
            *residual.device_ptr() as *const std::ffi::c_void,
            *weight.device_ptr() as *const std::ffi::c_void,
            *output.device_ptr() as *mut std::ffi::c_void,
            batch_size,
            hidden_dim,
            config.eps,
            dtype_to_ffi(T::DTYPE),
            stream.stream as *mut std::ffi::c_void,
        )
    };

    check_status(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_params() {
        let params = RMSNormParams::new(4096)
            .with_eps(1e-6)
            .with_dtype(DType::BFloat16);

        assert_eq!(params.hidden_dim, 4096);
        assert_eq!(params.eps, 1e-6);
        assert_eq!(params.dtype, DType::BFloat16);
    }

    #[test]
    fn test_layernorm_params() {
        let params = LayerNormParams::new(4096).with_eps(1e-5).without_bias();

        assert_eq!(params.hidden_dim, 4096);
        assert!(!params.use_bias);
    }
}

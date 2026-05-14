//! Sampling operations for LLM inference.
//!
//! This module provides sampling operations including:
//! - Top-K sampling
//! - Top-P (nucleus) sampling
//! - Min-P sampling
//! - Combined Top-K + Top-P sampling
//! - Softmax with temperature
//!
//! NOTE: All sampling operations only support float32 probabilities due to
//! type compatibility issues in FlashInfer's internal arithmetic with half/bf16.
//! Convert your probabilities to f32 before calling these functions.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::config::SamplingConfig;
use crate::Result;

/// Top-K sampling from probability distribution.
///
/// Samples a token index from each row by keeping only the top-K
/// highest probability tokens and sampling from that subset.
/// Uses Philox RNG for deterministic random number generation.
///
/// # Arguments
///
/// * `probs` - Probability distribution `[batch_size, vocab_size]` (float32 only)
/// * `output` - Output token indices `[batch_size]`
/// * `top_k` - Optional per-batch Top-K values `[batch_size]`, None uses `config.top_k`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (includes RNG seed, top_k default)
/// * `stream` - CUDA stream
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::ops::sampling::top_k_sampling;
/// use flashinfer_rs::config::SamplingConfig;
///
/// let config = SamplingConfig::new()
///     .with_top_k(50)
///     .with_seed(42);
/// top_k_sampling(
///     &probs, &mut output, None, batch_size, vocab_size, &config, &stream
/// )?;
/// ```
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_sampling(
    probs: &CudaSlice<f32>,
    output: &mut CudaSlice<i32>,
    top_k: Option<&CudaSlice<i32>>,
    valid_out: Option<&mut CudaSlice<bool>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let top_k_arr = top_k
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const i32
        })
        .unwrap_or(std::ptr::null());

    let valid_ptr = valid_out
        .map(|v| {
            let (ptr, _guard) = v.device_ptr_mut(stream);
            ptr as *mut bool
        })
        .unwrap_or(std::ptr::null_mut());

    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::top_k_sampling(
            probs_ptr as *const f32,
            output_ptr as *mut i32,
            valid_ptr,
            top_k_arr,
            config.top_k,
            batch_size,
            vocab_size,
            config.deterministic,
            std::ptr::null(),
            config.seed,
            std::ptr::null(),
            config.offset,
            cu_stream,
        )
    }
}

/// Top-P (nucleus) sampling from probability distribution.
///
/// Samples a token by keeping only the smallest set of tokens
/// whose cumulative probability exceeds P.
/// Uses Philox RNG for deterministic random number generation.
///
/// # Arguments
///
/// * `probs` - Probability distribution `[batch_size, vocab_size]` (float32 only)
/// * `output` - Output token indices `[batch_size]`
/// * `top_p` - Optional per-batch Top-P values `[batch_size]`, None uses `config.top_p`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (includes RNG seed, top_p default)
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn top_p_sampling(
    probs: &CudaSlice<f32>,
    output: &mut CudaSlice<i32>,
    top_p: Option<&CudaSlice<f32>>,
    valid_out: Option<&mut CudaSlice<bool>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let top_p_arr = top_p
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let valid_ptr = valid_out
        .map(|v| {
            let (ptr, _guard) = v.device_ptr_mut(stream);
            ptr as *mut bool
        })
        .unwrap_or(std::ptr::null_mut());

    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::top_p_sampling(
            probs_ptr as *const f32,
            output_ptr as *mut i32,
            valid_ptr,
            top_p_arr,
            config.top_p,
            batch_size,
            vocab_size,
            config.deterministic,
            std::ptr::null(),
            config.seed,
            std::ptr::null(),
            config.offset,
            cu_stream,
        )
    }
}

/// Min-P sampling from probability distribution.
///
/// Keeps tokens with probability at least min_p * max_prob.
/// Uses Philox RNG for deterministic random number generation.
///
/// # Arguments
///
/// * `probs` - Probability distribution `[batch_size, vocab_size]` (float32 only)
/// * `output` - Output token indices `[batch_size]`
/// * `min_p` - Optional per-batch Min-P thresholds `[batch_size]`, None uses `config.min_p`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (includes RNG seed, min_p default)
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn min_p_sampling(
    probs: &CudaSlice<f32>,
    output: &mut CudaSlice<i32>,
    min_p: Option<&CudaSlice<f32>>,
    valid_out: Option<&mut CudaSlice<bool>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let min_p_arr = min_p
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let valid_ptr = valid_out
        .map(|v| {
            let (ptr, _guard) = v.device_ptr_mut(stream);
            ptr as *mut bool
        })
        .unwrap_or(std::ptr::null_mut());

    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::min_p_sampling(
            probs_ptr as *const f32,
            output_ptr as *mut i32,
            valid_ptr,
            min_p_arr,
            config.min_p,
            batch_size,
            vocab_size,
            config.deterministic,
            std::ptr::null(),
            config.seed,
            std::ptr::null(),
            config.offset,
            cu_stream,
        )
    }
}

/// Combined Top-K and Top-P sampling.
///
/// First applies Top-K filtering, then Top-P filtering on the result.
/// This provides both absolute (K) and relative (P) truncation.
/// Uses Philox RNG for deterministic random number generation.
///
/// # Arguments
///
/// * `probs` - Probability distribution `[batch_size, vocab_size]` (float32 only)
/// * `output` - Output token indices `[batch_size]`
/// * `top_k` - Optional per-batch Top-K values `[batch_size]`, None uses `config.top_k`
/// * `top_p` - Optional per-batch Top-P values `[batch_size]`, None uses `config.top_p`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_top_p_sampling(
    probs: &CudaSlice<f32>,
    output: &mut CudaSlice<i32>,
    top_k: Option<&CudaSlice<i32>>,
    top_p: Option<&CudaSlice<f32>>,
    valid_out: Option<&mut CudaSlice<bool>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let top_k_arr = top_k
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const i32
        })
        .unwrap_or(std::ptr::null());
    let top_p_arr = top_p
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let valid_ptr = valid_out
        .map(|v| {
            let (ptr, _guard) = v.device_ptr_mut(stream);
            ptr as *mut bool
        })
        .unwrap_or(std::ptr::null_mut());

    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::top_k_top_p_sampling(
            probs_ptr as *const f32,
            output_ptr as *mut i32,
            valid_ptr,
            top_k_arr,
            top_p_arr,
            config.top_k,
            config.top_p,
            batch_size,
            vocab_size,
            config.deterministic,
            std::ptr::null(),
            config.seed,
            std::ptr::null(),
            config.offset,
            cu_stream,
        )
    }
}

/// Softmax with optional temperature scaling.
///
/// Computes softmax(logits / temperature) for each row.
///
/// # Arguments
///
/// * `logits` - Input logits `[batch_size, vocab_size]` (float32 only)
/// * `probs` - Output probabilities `[batch_size, vocab_size]`
/// * `temperature` - Optional per-batch temperatures `[batch_size]`, None uses `config.temperature`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (includes default temperature)
/// * `stream` - CUDA stream
///
/// # Example
///
/// ```ignore
/// use flashinfer_rs::ops::sampling::softmax;
/// use flashinfer_rs::config::SamplingConfig;
///
/// let config = SamplingConfig::new().with_temperature(0.8);
/// softmax(&logits, &mut probs, None, batch_size, vocab_size, &config, &stream)?;
/// ```
#[cfg(feature = "cuda")]
pub fn softmax(
    logits: &CudaSlice<f32>,
    probs: &mut CudaSlice<f32>,
    temperature: Option<&CudaSlice<f32>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let temp_arr = temperature
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let (logits_ptr, _logits_guard) = logits.device_ptr(stream);
    let (probs_ptr, _probs_guard) = probs.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::softmax(
            logits_ptr as *const f32,
            probs_ptr as *mut f32,
            temp_arr,
            config.temperature,
            batch_size,
            vocab_size,
            cu_stream,
        )
    }
}

/// Renormalize probabilities after top-p filtering.
///
/// After applying top-P masking, this function
/// renormalizes the remaining probabilities to sum to 1.
///
/// # Arguments
///
/// * `probs` - Input probabilities `[batch_size, vocab_size]` (float32 only)
/// * `renormed_probs` - Output renormalized probabilities `[batch_size, vocab_size]`
/// * `top_p` - Optional per-batch Top-P values `[batch_size]`, None uses `config.top_p`
/// * `batch_size` - Number of samples
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (includes default top_p)
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
pub fn top_p_renorm_probs(
    probs: &CudaSlice<f32>,
    renormed_probs: &mut CudaSlice<f32>,
    top_p: Option<&CudaSlice<f32>>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let top_p_arr = top_p
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (renormed_ptr, _renormed_guard) = renormed_probs.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::top_p_renorm_probs(
            probs_ptr as *const f32,
            renormed_ptr as *mut f32,
            top_p_arr,
            config.top_p,
            batch_size,
            vocab_size,
            cu_stream,
        )
    }
}

/// AIR Top-P renormalization (radix-based, no full sort).
///
/// Faster alternative to [`top_p_renorm_probs`] for large vocabularies.
/// Uses TensorRT-LLM-style multi-pass radix bucketing.
///
/// Required workspace size can be obtained via
/// [`crate::ffi::air_top_p_renorm_probs_workspace_size`]. The workspace must
/// be allocated on device and live at least as long as this call.
///
/// # Arguments
///
/// * `probs` - Input probabilities `[batch_size, vocab_size]` (float32 only)
/// * `renormed_probs` - Output `[batch_size, vocab_size]`
/// * `top_p` - Optional per-batch Top-P values, None uses `config.top_p`
/// * `workspace` - Device scratch buffer (see workspace_size helper)
/// * `batch_size` - Batch size
/// * `vocab_size` - Vocabulary size
/// * `config` - Sampling configuration (uses `top_p`, `deterministic`)
/// * `stream` - CUDA stream
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn air_top_p_renorm_probs(
    probs: &CudaSlice<f32>,
    renormed_probs: &mut CudaSlice<f32>,
    top_p: Option<&CudaSlice<f32>>,
    workspace: &mut CudaSlice<u8>,
    batch_size: u32,
    vocab_size: u32,
    config: &SamplingConfig,
    stream: &CudaStream,
) -> Result<()> {
    let required = crate::ffi::air_top_p_renorm_probs_workspace_size(batch_size, vocab_size)?;
    if workspace.len() < required {
        return Err(crate::FlashInferError::invalid_config(format!(
            "air_top_p_renorm_probs workspace too small: need {} bytes, got {}",
            required,
            workspace.len()
        )));
    }

    let top_p_arr = top_p
        .map(|t| {
            let (ptr, _guard) = t.device_ptr(stream);
            ptr as *const f32
        })
        .unwrap_or(std::ptr::null());

    let ws_len = workspace.len();
    let (probs_ptr, _probs_guard) = probs.device_ptr(stream);
    let (renormed_ptr, _renormed_guard) = renormed_probs.device_ptr_mut(stream);
    let (ws_ptr, _ws_guard) = workspace.device_ptr_mut(stream);
    let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

    unsafe {
        crate::ffi::air_top_p_renorm_probs(
            probs_ptr as *const f32,
            renormed_ptr as *mut f32,
            top_p_arr,
            config.top_p,
            batch_size,
            vocab_size,
            config.deterministic,
            ws_ptr as *mut std::ffi::c_void,
            ws_len,
            cu_stream,
        )
    }
}

/// Parameters for sampling operations.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Top-K value (0 to disable)
    pub top_k: u32,
    /// Top-P value (1.0 to disable)
    pub top_p: f32,
    /// Min-P value (0.0 to disable)
    pub min_p: f32,
    /// Temperature for softmax
    pub temperature: f32,
    /// Whether to use deterministic sampling
    pub deterministic: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 1.0,
            deterministic: false,
        }
    }
}

impl SamplingParams {
    /// Create new sampling parameters with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set top-K value.
    pub fn with_top_k(mut self, k: u32) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-P value.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set min-P value.
    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Enable deterministic sampling.
    pub fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<()> {
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(crate::FlashInferError::invalid_config(
                "top_p must be in (0, 1]",
            ));
        }
        if self.min_p < 0.0 || self.min_p >= 1.0 {
            return Err(crate::FlashInferError::invalid_config(
                "min_p must be in [0, 1)",
            ));
        }
        if self.temperature <= 0.0 {
            return Err(crate::FlashInferError::invalid_config(
                "temperature must be positive",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.top_k, 0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.temperature, 1.0);
        assert!(!params.deterministic);
    }

    #[test]
    fn test_sampling_params_builder() {
        let params = SamplingParams::new()
            .with_top_k(50)
            .with_top_p(0.9)
            .with_temperature(0.8)
            .deterministic();

        assert_eq!(params.top_k, 50);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.temperature, 0.8);
        assert!(params.deterministic);
    }

    #[test]
    fn test_sampling_params_validation() {
        let valid = SamplingParams::new().with_top_p(0.9);
        assert!(valid.validate().is_ok());

        let invalid_top_p = SamplingParams::new().with_top_p(0.0);
        assert!(invalid_top_p.validate().is_err());

        let invalid_temp = SamplingParams::new().with_temperature(0.0);
        assert!(invalid_temp.validate().is_err());
    }
}

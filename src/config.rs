//! Configuration types for FlashInfer operations.
//!
//! This module provides configuration structs with builder patterns for
//! attention, RoPE, and sampling operations.

use crate::types::{Backend, DType, KVLayout, MaskMode, PosEncodingMode};

/// Configuration for attention computation.
///
/// # Example
///
/// ```
/// use flashinfer_rs::config::AttentionConfig;
/// use flashinfer_rs::types::{MaskMode, PosEncodingMode};
///
/// let config = AttentionConfig::new(32, 8, 128)
///     .with_causal_mask()
///     .with_rope(1.0, 10000.0)
///     .with_page_size(16);
/// ```
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query/output heads.
    pub num_qo_heads: u32,
    /// Number of key/value heads (can be < num_qo_heads for GQA).
    pub num_kv_heads: u32,
    /// Head dimension for Q and K.
    pub head_dim_qk: u32,
    /// Head dimension for V and O (can differ for MLA).
    pub head_dim_vo: u32,
    /// Page size for paged KV cache.
    pub page_size: u32,
    /// Data type for computation.
    pub dtype: DType,
    /// KV cache layout.
    pub kv_layout: KVLayout,
    /// Position encoding mode.
    pub pos_encoding: PosEncodingMode,
    /// Attention mask mode.
    pub mask_mode: MaskMode,
    /// Attention backend.
    pub backend: Backend,
    /// Softmax scale (1/sqrt(head_dim) by default).
    pub sm_scale: Option<f32>,
    /// RoPE scale factor.
    pub rope_scale: f32,
    /// RoPE theta parameter.
    pub rope_theta: f32,
    /// Soft cap for attention logits (0 or None to disable).
    pub logits_soft_cap: Option<f32>,
    /// Sliding window size (None for full attention).
    pub window_left: Option<i32>,
    /// Use FP16 for QK reduction.
    pub use_fp16_qk_reduction: bool,
    /// Return log-sum-exp values.
    pub return_lse: bool,
}

impl AttentionConfig {
    /// Creates a new attention configuration with default settings.
    ///
    /// # Arguments
    ///
    /// * `num_qo_heads` - Number of query/output heads
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Head dimension (used for both QK and VO)
    pub fn new(num_qo_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            num_qo_heads,
            num_kv_heads,
            head_dim_qk: head_dim,
            head_dim_vo: head_dim,
            page_size: 16,
            dtype: DType::Float16,
            kv_layout: KVLayout::NHD,
            pos_encoding: PosEncodingMode::None,
            mask_mode: MaskMode::None,
            backend: Backend::Auto,
            sm_scale: None,
            rope_scale: 1.0,
            rope_theta: 10000.0,
            logits_soft_cap: None,
            window_left: None,
            use_fp16_qk_reduction: false,
            return_lse: false,
        }
    }

    /// Sets the page size for paged KV cache.
    pub fn with_page_size(mut self, page_size: u32) -> Self {
        self.page_size = page_size;
        self
    }

    /// Sets the data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Sets the KV cache layout.
    pub fn with_kv_layout(mut self, layout: KVLayout) -> Self {
        self.kv_layout = layout;
        self
    }

    /// Enables causal masking.
    pub fn with_causal_mask(mut self) -> Self {
        self.mask_mode = MaskMode::Causal;
        self
    }

    /// Sets custom mask mode.
    pub fn with_mask_mode(mut self, mode: MaskMode) -> Self {
        self.mask_mode = mode;
        self
    }

    /// Configures RoPE position encoding.
    pub fn with_rope(mut self, scale: f32, theta: f32) -> Self {
        self.pos_encoding = PosEncodingMode::RoPELlama;
        self.rope_scale = scale;
        self.rope_theta = theta;
        self
    }

    /// Configures RoPE with frequency scaling (Llama 3.1 style).
    pub fn with_rope_freq_scale(mut self, scale: f32, theta: f32) -> Self {
        self.pos_encoding = PosEncodingMode::RoPELlamaFreqScale;
        self.rope_scale = scale;
        self.rope_theta = theta;
        self
    }

    /// Configures ALiBi position encoding.
    pub fn with_alibi(mut self) -> Self {
        self.pos_encoding = PosEncodingMode::ALiBi;
        self
    }

    /// Sets the position encoding mode directly.
    pub fn with_pos_encoding(mut self, mode: PosEncodingMode) -> Self {
        self.pos_encoding = mode;
        self
    }

    /// Sets the attention backend.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets a custom softmax scale (overrides default 1/sqrt(head_dim)).
    pub fn with_sm_scale(mut self, scale: f32) -> Self {
        self.sm_scale = Some(scale);
        self
    }

    /// Enables logits soft cap.
    pub fn with_logits_soft_cap(mut self, cap: f32) -> Self {
        self.logits_soft_cap = Some(cap);
        self
    }

    /// Enables sliding window attention.
    pub fn with_sliding_window(mut self, window: i32) -> Self {
        self.window_left = Some(window);
        self
    }

    /// Enables FP16 QK reduction.
    pub fn with_fp16_qk_reduction(mut self, enable: bool) -> Self {
        self.use_fp16_qk_reduction = enable;
        self
    }

    /// Enables returning log-sum-exp values.
    pub fn with_return_lse(mut self, enable: bool) -> Self {
        self.return_lse = enable;
        self
    }

    /// Sets different head dimensions for QK and VO (for MLA).
    pub fn with_head_dims(mut self, head_dim_qk: u32, head_dim_vo: u32) -> Self {
        self.head_dim_qk = head_dim_qk;
        self.head_dim_vo = head_dim_vo;
        self
    }

    /// Returns the GQA ratio (num_qo_heads / num_kv_heads).
    #[inline]
    pub fn gqa_ratio(&self) -> u32 {
        self.num_qo_heads / self.num_kv_heads
    }

    /// Returns the effective softmax scale.
    #[inline]
    pub fn effective_sm_scale(&self) -> f32 {
        self.sm_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim_qk as f32).sqrt())
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_qo_heads == 0 {
            return Err("num_qo_heads must be positive".to_string());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be positive".to_string());
        }
        if self.num_qo_heads % self.num_kv_heads != 0 {
            return Err("num_qo_heads must be divisible by num_kv_heads".to_string());
        }
        if self.head_dim_qk == 0 || self.head_dim_vo == 0 {
            return Err("head dimensions must be positive".to_string());
        }
        if self.page_size == 0 {
            return Err("page_size must be positive".to_string());
        }
        if let Some(cap) = self.logits_soft_cap {
            if cap <= 0.0 {
                return Err("logits_soft_cap must be positive if set".to_string());
            }
        }
        if let Some(window) = self.window_left {
            if window <= 0 {
                return Err("window_left must be positive if set".to_string());
            }
        }
        Ok(())
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self::new(32, 8, 128)
    }
}

/// Configuration for RoPE (Rotary Position Embedding).
///
/// # Example
///
/// ```
/// use flashinfer_rs::config::RoPEConfig;
///
/// let config = RoPEConfig::new(128)
///     .with_theta(10000.0)
///     .with_scale(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Dimension to apply RoPE to (usually equal to head_dim).
    pub rotary_dim: u32,
    /// Use interleaved RoPE format.
    pub interleave: bool,
    /// RoPE scale factor.
    pub scale: f32,
    /// RoPE theta parameter (base frequency).
    pub theta: f32,
}

impl RoPEConfig {
    /// Creates a new RoPE configuration.
    pub fn new(rotary_dim: u32) -> Self {
        Self {
            rotary_dim,
            interleave: false,
            scale: 1.0,
            theta: 10000.0,
        }
    }

    /// Sets the theta parameter.
    pub fn with_theta(mut self, theta: f32) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enables interleaved format.
    pub fn with_interleave(mut self, interleave: bool) -> Self {
        self.interleave = interleave;
        self
    }

    /// Creates a Llama-style RoPE configuration.
    pub fn llama(head_dim: u32) -> Self {
        Self::new(head_dim).with_theta(10000.0)
    }

    /// Creates a Llama 3.1 style RoPE configuration with frequency scaling.
    pub fn llama3_1(head_dim: u32, theta: f32) -> Self {
        Self::new(head_dim).with_theta(theta).with_scale(1.0)
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.rotary_dim == 0 {
            return Err("rotary_dim must be positive".to_string());
        }
        if self.rotary_dim % 2 != 0 {
            return Err("rotary_dim must be even".to_string());
        }
        if self.theta <= 0.0 {
            return Err("theta must be positive".to_string());
        }
        if self.scale <= 0.0 {
            return Err("scale must be positive".to_string());
        }
        Ok(())
    }
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self::new(128)
    }
}

/// Configuration for sampling operations.
///
/// NOTE: All sampling operations only support float32 probabilities due to
/// type compatibility issues in FlashInfer's internal arithmetic with half/bf16.
///
/// # Example
///
/// ```
/// use flashinfer_rs::config::SamplingConfig;
///
/// let config = SamplingConfig::new()
///     .with_top_k(50)
///     .with_top_p(0.9)
///     .with_temperature(0.8)
///     .with_seed(42)
///     .with_deterministic(true);
/// ```
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Top-K value (0 to disable).
    pub top_k: u32,
    /// Top-P (nucleus) value (1.0 to disable).
    pub top_p: f32,
    /// Min-P value (0.0 to disable).
    pub min_p: f32,
    /// Temperature for softmax scaling.
    pub temperature: f32,
    /// Use deterministic sampling (reproducible results).
    pub deterministic: bool,
    /// Philox RNG seed for random number generation.
    pub seed: u64,
    /// Philox RNG offset (advanced: used for parallel streams).
    pub offset: u64,
}

impl SamplingConfig {
    /// Creates a new sampling configuration with defaults.
    pub fn new() -> Self {
        Self {
            top_k: 0,       // disabled
            top_p: 1.0,     // disabled
            min_p: 0.0,     // disabled
            temperature: 1.0,
            deterministic: false,
            seed: 0,
            offset: 0,
        }
    }

    /// Sets the Top-K value.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the Top-P (nucleus) value.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets the Min-P value.
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Sets the temperature for softmax scaling.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Enables deterministic sampling.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Sets the Philox RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the Philox RNG offset (advanced).
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> crate::Result<()> {
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

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for normalization operations (RMSNorm, LayerNorm).
#[derive(Debug, Clone)]
pub struct NormConfig {
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Whether to add residual before normalization.
    pub add_residual: bool,
    /// Whether to output quantized (FP8) result.
    pub quantize_output: bool,
}

impl NormConfig {
    /// Creates a new normalization configuration.
    pub fn new() -> Self {
        Self {
            eps: 1e-5,
            add_residual: false,
            quantize_output: false,
        }
    }

    /// Sets epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Enables residual addition.
    pub fn with_residual(mut self) -> Self {
        self.add_residual = true;
        self
    }

    /// Enables quantized output.
    pub fn with_quantize(mut self) -> Self {
        self.quantize_output = true;
        self
    }
}

impl Default for NormConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_builder() {
        let config = AttentionConfig::new(32, 8, 128)
            .with_page_size(16)
            .with_dtype(DType::BFloat16)
            .with_causal_mask()
            .with_rope(1.0, 10000.0);

        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim_qk, 128);
        assert_eq!(config.page_size, 16);
        assert_eq!(config.dtype, DType::BFloat16);
        assert_eq!(config.mask_mode, MaskMode::Causal);
        assert_eq!(config.pos_encoding, PosEncodingMode::RoPELlama);
        assert_eq!(config.gqa_ratio(), 4);
    }

    #[test]
    fn test_attention_config_validation() {
        let valid = AttentionConfig::new(32, 8, 128);
        assert!(valid.validate().is_ok());

        let invalid_heads = AttentionConfig::new(32, 0, 128);
        assert!(invalid_heads.validate().is_err());

        let invalid_gqa = AttentionConfig::new(32, 7, 128);
        assert!(invalid_gqa.validate().is_err());
    }

    #[test]
    fn test_attention_config_effective_sm_scale() {
        let config = AttentionConfig::new(32, 8, 64);
        let expected = 1.0 / 8.0; // 1/sqrt(64)
        assert!((config.effective_sm_scale() - expected).abs() < 1e-6);

        let config_with_scale = config.with_sm_scale(0.5);
        assert_eq!(config_with_scale.effective_sm_scale(), 0.5);
    }

    #[test]
    fn test_rope_config_builder() {
        let config = RoPEConfig::new(128)
            .with_theta(500000.0)
            .with_scale(0.5)
            .with_interleave(true);

        assert_eq!(config.rotary_dim, 128);
        assert_eq!(config.theta, 500000.0);
        assert_eq!(config.scale, 0.5);
        assert!(config.interleave);
    }

    #[test]
    fn test_rope_config_validation() {
        let valid = RoPEConfig::new(128);
        assert!(valid.validate().is_ok());

        let invalid_dim = RoPEConfig::new(0);
        assert!(invalid_dim.validate().is_err());

        let invalid_odd = RoPEConfig::new(127);
        assert!(invalid_odd.validate().is_err());
    }

    #[test]
    fn test_rope_config_presets() {
        let llama = RoPEConfig::llama(128);
        assert_eq!(llama.theta, 10000.0);

        let llama3_1 = RoPEConfig::llama3_1(128, 500000.0);
        assert_eq!(llama3_1.theta, 500000.0);
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::new().with_deterministic(true);
        assert!(config.deterministic);

        let default = SamplingConfig::default();
        assert!(!default.deterministic);
    }

    #[test]
    fn test_norm_config() {
        let config = NormConfig::new()
            .with_eps(1e-6)
            .with_residual()
            .with_quantize();

        assert_eq!(config.eps, 1e-6);
        assert!(config.add_residual);
        assert!(config.quantize_output);
    }
}

# flashinfer-rs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

High-performance Rust bindings for [FlashInfer](https://github.com/flashinfer-ai/flashinfer) attention kernels.

## Features

### Attention
- BatchDecode with Plan-Run pattern (8 attention variants)
- BatchPrefill (causal/non-causal)
- DeepSeek MLA (Multi-head Latent Attention)
- Sliding window, logits soft cap, ALiBi

### Operations
- RMSNorm, FusedAddRMSNorm, LayerNorm, Gemma variants
- RoPE (inplace, pos_ids, cos_sin_cache)
- Sampling: top_k, top_p, min_p, softmax
- Fused RoPE + Quantize + Append (SM89+)

### KV Cache
- Paged KV cache management
- append_paged_kv_cache for decode/prefill
- MLA dual cache (ckv + kpe)

## Requirements

- CUDA Toolkit 12.0+
- C++17 compiler (gcc 11+, clang 14+)
- Rust 1.70+
- FlashInfer source (auto-cloned during build)

## Quick Start

### Installation

```toml
[dependencies]
flashinfer-rs = { git = "https://github.com/vasia123/flashinfer-rs", features = ["cuda"] }
```

### Minimal Example

```rust
use flashinfer_rs::{AttentionConfig, BatchDecodeHandler};

// Configure attention: 32 Q heads, 8 KV heads, 128 head dim
let config = AttentionConfig::new(32, 8, 128)
    .with_page_size(16);

// Create handler with Plan-Run pattern
let handler = BatchDecodeHandler::new(&device, config)?;

// Plan for batch
handler.plan(batch_size, &kv_lengths, &page_indices)?;

// Run attention
handler.run(&query, &kv_cache, &mut output)?;
```

## Building from Source

### Prerequisites

```bash
# Clone FlashInfer source (required for headers)
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer references/flashinfer
```

### Build

```bash
CUDA_HOME=/usr/local/cuda cargo build --features cuda
```

### Feature Flags

| Flag | Description |
|------|-------------|
| `cuda` | Enable CUDA support |
| `cuda-11` | Target CUDA 11.x |
| `cuda-12` | Target CUDA 12.x (default with cuda) |
| `sm80` | Target Ampere (A100, RTX 30xx) |
| `sm90` | Target Hopper (H100, H200) |

## API Reference

### Attention

```rust
// Decode: single token per sequence
let handler = BatchDecodeHandler::new(&device, config)?;
handler.plan(batch_size, &kv_lengths, &page_indices)?;
handler.run(&query, &kv_cache, &mut output)?;

// Prefill: variable-length sequences
let handler = BatchPrefillHandler::new(&device, config)?;
handler.plan(&qo_indptr, &kv_indptr, &kv_lengths)?;
handler.run(&query, &kv_cache, &mut output)?;

// Attention variants via config
let config = AttentionConfig::new(32, 8, 128)
    .with_sliding_window(256)     // Mistral-style local attention
    .with_logits_soft_cap(50.0)   // Gemma 2 soft-capping
    .with_alibi_slopes(&slopes);  // ALiBi positional encoding
```

### MLA (DeepSeek)

```rust
use flashinfer_rs::{MLAConfig, MLAQueryShape, MLAKeyShape};

// DeepSeek v2/v3 configuration (fixed dimensions)
let config = MLAConfig::deepseek();
// num_heads: 128, head_dim_ckv: 512, head_dim_kpe: 64

// MLA uses separate caches for compressed KV and key position embedding
// ckv_cache: [num_pages, page_size, 512]
// kpe_cache: [num_pages, page_size, 64]
```

### Normalization

```rust
use flashinfer_rs::ops::{rmsnorm, fused_add_rmsnorm, layernorm};

// RMSNorm
rmsnorm(&input, &weight, &mut output, eps, stream)?;

// Fused residual + RMSNorm
fused_add_rmsnorm(&input, &residual, &weight, &mut output, eps, stream)?;

// LayerNorm (float16 only)
layernorm(&input, &weight, &bias, &mut output, eps, stream)?;

// Gemma-style RMSNorm (weight + 1.0)
gemma_rmsnorm(&input, &weight, &mut output, eps, stream)?;
```

### RoPE

```rust
use flashinfer_rs::ops::{apply_rope, apply_rope_inplace, apply_rope_with_cos_sin_cache};

// Apply RoPE with position IDs
apply_rope(&q, &k, &pos_ids, &mut q_out, &mut k_out, config, stream)?;

// Inplace RoPE
apply_rope_inplace(&mut q, &mut k, &pos_ids, config, stream)?;

// With precomputed cos/sin cache
apply_rope_with_cos_sin_cache(&q, &k, &cos_sin_cache, &pos_ids, &mut q_out, &mut k_out, stream)?;
```

### Sampling

```rust
use flashinfer_rs::ops::{top_k_sampling, top_p_sampling, softmax};

// Top-K sampling
top_k_sampling(&logits, &uniform_samples, &mut output, k, stream)?;

// Top-P (nucleus) sampling
top_p_sampling(&probs, &uniform_samples, &mut output, p, stream)?;

// Softmax
softmax(&logits, &mut probs, stream)?;
```

## Project Structure

```
flashinfer-rs/
├── src/
│   ├── lib.rs              # Public API
│   ├── ffi.rs              # FFI bindings
│   ├── batch_decode.rs     # Decode handler
│   ├── batch_prefill.rs    # Prefill handler
│   ├── mla.rs              # DeepSeek MLA
│   ├── config.rs           # Configuration types
│   ├── workspace.rs        # GPU workspace management
│   └── ops/                # norm, rope, sampling
│       ├── norm.rs
│       ├── rope.rs
│       └── sampling.rs
├── csrc/
│   ├── flashinfer_c_api.h      # C API declarations
│   ├── flashinfer_c_api.cu     # Main dispatch
│   ├── flashinfer_decode.cu    # Decode kernels
│   ├── flashinfer_prefill.cu   # Prefill kernels
│   ├── flashinfer_mla.cu       # MLA kernels
│   ├── flashinfer_norm.cu      # Normalization
│   ├── flashinfer_rope.cu      # RoPE
│   ├── flashinfer_sampling.cu  # Sampling
│   └── flashinfer_rope_quant.cu # Fused RoPE+Quant (SM89+)
└── build.rs                # CUDA compilation
```

## Roadmap

### Completed
- Attention (8 variants: standard, sliding window, soft cap, ALiBi combinations)
- DeepSeek MLA (Multi-head Latent Attention)
- Normalization (RMSNorm, LayerNorm, Gemma variants)
- RoPE (all modes)
- Sampling (top_k, top_p, min_p)
- FP8 quantization (SM89+)

### Not Yet Implemented
- Custom attention masks
- TensorRT-LLM MLA backend
- XQA MLA backend (SM120)

## Contributing

```bash
cargo fmt && cargo clippy
cargo test --features cuda
```

## License

Apache-2.0

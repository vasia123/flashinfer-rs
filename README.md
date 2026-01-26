# flashinfer-rs

Rust bindings for [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - high-performance attention kernels for LLM inference serving.

## Status

🚧 **Work in Progress** 🚧

This crate provides the Rust interface for FlashInfer. CUDA kernel compilation is not yet implemented.

## Features

- **Paged KV Cache**: Efficient memory management with block-based caching
- **Batch Decode**: Optimized single-token attention for decode phase
- **Batch Prefill**: Variable-length batch processing for prefill phase
- **GQA Support**: Grouped-Query Attention with flexible head configurations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
flashinfer-rs = { git = "https://github.com/vasia123/flashinfer-rs" }
```

For CUDA support:

```toml
[dependencies]
flashinfer-rs = { git = "https://github.com/vasia123/flashinfer-rs", features = ["cuda"] }
```

## Usage

```rust
use flashinfer_rs::{AttentionConfig, BatchDecodeHandler, PageTable};

// Configure attention
let config = AttentionConfig::new(32, 8, 128)  // 32 Q heads, 8 KV heads, 128 dim
    .with_page_size(16);

// Create handler (requires CUDA device)
let handler = BatchDecodeHandler::new(device, config)?;

// Plan for batch
let page_table = PageTable::new(batch_size, max_pages);
handler.plan(batch_size, &kv_lengths, &page_table)?;

// Run attention
handler.forward_inplace(&query, &kv_k, &kv_v, &page_table, &kv_lengths, &mut output)?;
```

## Building with CUDA

### Prerequisites

1. CUDA Toolkit (11.8+ recommended)
2. C++ compiler with C++17 support
3. FlashInfer source code

### Setup

```bash
# Clone FlashInfer source
git clone https://github.com/flashinfer-ai/flashinfer.git

# Set environment variable
export FLASHINFER_PATH=/path/to/flashinfer

# Build with CUDA
cargo build --features cuda
```

## Architecture

```
flashinfer-rs/
├── src/
│   ├── lib.rs           # Public API
│   ├── error.rs         # Error types
│   ├── page_table.rs    # Page table management
│   ├── cuda/            # CUDA utilities
│   ├── batch_decode.rs  # Decode attention
│   ├── batch_prefill.rs # Prefill attention
│   └── ffi.rs           # C++ FFI bindings
├── build.rs             # CUDA compilation
└── csrc/                # C++ glue code (TODO)
```

## Roadmap

- [x] Core data structures (PageTable, AttentionConfig)
- [x] Rust API design
- [ ] C++ glue code for FlashInfer kernels
- [ ] CUDA kernel compilation in build.rs
- [ ] bindgen FFI generation
- [ ] Integration tests with candle
- [ ] Benchmarks

## Related Projects

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Original CUDA kernels
- [candle](https://github.com/huggingface/candle) - Rust ML framework
- [vllm-rust](https://github.com/vasia123/vllm-rust) - Rust LLM inference engine

## License

Apache-2.0

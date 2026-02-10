# CLAUDE.md

## Project

Rust bindings for FlashInfer — high-performance attention kernels for LLM inference.

## Stack

- Language: Rust (edition 2021)
- Build: Cargo + build.rs for CUDA compilation
- GPU: CUDA via cudarc, custom kernels via cc/nvcc
- FFI: bindgen for C++ header bindings

## Reference

FlashInfer source: https://github.com/flashinfer-ai/flashinfer
Pinned commit: `bd0b27b4cc68b2e5ba30178b4b3b781c5ed1ece6`

### FlashInfer C++ source

The build system auto-downloads FlashInfer headers to `OUT_DIR/flashinfer-source`
when building as a Cargo dependency. For local development, you can also pre-clone:

```bash
cd ~/projects_hobby/flashinfer-rs
mkdir -p references
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git references/flashinfer
cd references/flashinfer && git checkout bd0b27b4cc68b2e5ba30178b4b3b781c5ed1ece6
```

Override with `FLASHINFER_PATH` env var if needed.

### Key directories to study (in references/flashinfer/)
- `include/flashinfer/` — C++ headers with kernel interfaces
- `csrc/` — CUDA kernel implementations
- `flashinfer/` — Python API (shows intended usage patterns)
- `python/csrc/` — PyTorch C++ bindings (reference for FFI design)

## Principles

- TDD: test first, implement second, refactor third
- Performance over convenience
- Zero-copy where possible
- Unsafe only when measured and justified
- No premature abstraction — earn generality through repetition
- Errors: thiserror for libraries
- No unwrap() in production paths

## Code Style

- `cargo fmt` — non-negotiable
- `cargo clippy` — zero warnings
- Names: descriptive, no abbreviations except domain-standard (KV, QKV, GQA, PagedKV)
- Comments: only "why", never "what"
- Tests: unit tests in-module, integration tests in /tests

## Architecture Decisions

Document in `/docs/adr/NNNN-title.md` when:
- Choosing between competing approaches
- Introducing unsafe
- Adding dependencies

## Quality Bar

Production-grade. Every line, every commit, every decision — as if it ships to thousands of GPUs tomorrow. No prototyping mindset, no "fix later", no shortcuts. Code reviews would pass at a top-tier infra team.

## Agent Expectations

- Read before write. Always.
- Verify assumptions with code, not guesses
- Study FlashInfer reference before implementing any kernel wrapper
- If unsure — ask, don't assume
- Run `cargo check` after every edit
- Run `cargo test` before declaring done
- No stubs. Either complete the implementation or document the gap with a detailed TODO.
- Workarounds require TODO with explanation of the proper fix.
- Use NOTE for non-obvious decisions or important context that isn't actionable.
- Think before generating. Less code that works > more code that might.

## Strategy: Hybrid Approach

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for full details.

### Phase 1: FFI Bindings (Current)
- C++ wrapper library over FlashInfer templates
- Rust FFI layer with safe abstractions
- Target: production-ready in 6-8 weeks

### Phase 2: Utility Kernels in Rust (Future)
- Port RoPE, normalization, masking to Rust via rust-cuda
- Validate rust-cuda stability
- Maintain C++ fallback

### Phase 3: Native Attention (Long-term)
- Port batch_decode, batch_prefill to Rust
- Only if Phase 2 proves rust-cuda viable
- Goal: full independence from C++ FlashInfer

## Build Requirements

- CUDA Toolkit 12.0+ (nvcc, cudart)
- C++17 compiler (for FlashInfer headers)
- Rust nightly (for future rust-cuda integration)
- Environment: `CUDA_HOME` must be set

### Feature Flags
- `cuda` — enable CUDA support (default off)
- `cuda-11` / `cuda-12` — CUDA version selection
- `sm80` / `sm90` — target architecture

## Implementation Priority

### Phase 1 (FFI Bindings)
1. **C++ Wrapper API** — thin C layer over FlashInfer templates
2. **build.rs** — nvcc compilation, bindgen integration
3. **BatchDecodeWithPagedKVCache** — FFI wrapper
4. **BatchPrefillWithPagedKVCache** — FFI wrapper
5. **Workspace allocation** — cudarc integration

### Phase 2 (Rust Kernels)
1. **RoPE** — rotary position embedding
2. **RMSNorm** — normalization
3. **Masking** — attention masks

### Phase 3 (Native Attention)
1. **Batch decode attention** — Rust implementation
2. **Batch prefill attention** — Rust implementation

## Key Components (Original FlashInfer)

1. **BatchDecodeWithPagedKVCache** — decode attention with paged KV
2. **BatchPrefillWithPagedKVCache** — prefill attention with paged KV
3. **AppendPagedKVCache** — write new KV to paged cache
4. **Page table management** — block mapping for sequences
5. **Workspace allocation** — temporary GPU memory for kernels

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

Key directories to study:
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
- Mark all workarounds with TODO explaining the proper solution.
- Think before generating. Less code that works > more code that might.

## Key Components to Port

1. **BatchDecodeWithPagedKVCache** — decode attention with paged KV
2. **BatchPrefillWithPagedKVCache** — prefill attention with paged KV
3. **AppendPagedKVCache** — write new KV to paged cache
4. **Page table management** — block mapping for sequences
5. **Workspace allocation** — temporary GPU memory for kernels

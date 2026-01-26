# FlashInfer-RS Implementation Plan

## Executive Summary

This document outlines the implementation strategy for `flashinfer-rs`, Rust bindings for FlashInfer high-performance attention kernels.

**Chosen Strategy:** Hybrid approach - FFI bindings first, gradual migration to native Rust CUDA kernels.

## FlashInfer Complexity Analysis

| Metric | Value |
|--------|-------|
| CUDA code | ~39,000 lines |
| C++ headers | ~72,000 lines |
| Python API + JIT | ~44,000 lines |
| **Total** | **~155,000 lines** |
| .cu/.cuh files | 191 |
| Header files | 84 |
| Kernel variants | 600+ (dtype x head_dim x arch x variant) |
| Template parameters | 60+ |

This is one of the most complex GPU projects. Flash Attention requires:
- Warp-level synchronization
- Tensor cores via CUTLASS
- Paged KV cache with indirection
- Online softmax with numerical stability
- Architecture-specific optimizations (SM80, SM90, SM100+)

---

## Strategy Comparison

### Path 1: FFI Bindings

**Requirements:**
1. C++ wrapper library (no ready C API exists)
   - Explicit template instantiations (20-30 combinations)
   - `extern "C"` functions for each kernel
   - ~2000-3000 lines of C++ code

2. Rust FFI layer:
   - bindgen for binding generation
   - Safe wrappers over unsafe
   - cudarc integration

**Pros:**
- Fast path to working solution
- Proven FlashInfer kernels
- Low performance regression risk
- Usable within 2 months

**Cons:**
- Mixed C++/Rust codebase
- Upstream dependency
- ABI instability between versions
- FFI boundary debugging complexity

### Path 2: Native Rust Implementation

**Rust CUDA Ecosystem Status (2026):**
- rust-cuda: Experimental but improving (rustc_codegen_nvvm backend)
- cudarc: Production-ready for host-side GPU management

**Limitations:**
- no_std environment in device code
- Limited std library support
- Fewer debugging tools than CUDA C++
- Template specialization harder than C++

**Pros:**
- Single language - easier maintenance
- Type safety at host/device boundary
- No C++ ABI issues
- Rust-specific optimization potential
- Independence from upstream
- Long-term maintainability

**Cons:**
- 10-20x more time than bindings
- rust-cuda ecosystem not mature
- Performance regression risk
- Need to reimplement 155K lines of proven code

---

## Selected Strategy: Hybrid Approach

### Phase 1: FFI Bindings (6-8 weeks)

Fast path to production-ready solution.

### Phase 2: Utility Kernels in Rust (3-6 months)

Validate rust-cuda stability with simple kernels.

### Phase 3: Native Attention (6-12+ months)

Full migration if Phase 2 proves viable.

---

## Phase 1: FFI Bindings Implementation

### 1.1 C++ Wrapper Preparation (Weeks 1-2)

**Goal:** Create thin C API over FlashInfer C++ templates

**Tasks:**
- [ ] Create `csrc/flashinfer_c_api.h` with `extern "C"` declarations
- [ ] Implement `csrc/flashinfer_c_api.cu` with explicit instantiations:
  - `batch_decode_f16_128d`, `batch_decode_bf16_128d`
  - `batch_decode_f16_256d`, `batch_decode_bf16_256d`
  - `batch_prefill_*` similarly
  - `append_paged_kv_cache_*`
- [ ] Minimal set: fp16 + bf16, head_dim 64/128/256

**File Structure:**
```
csrc/
├── flashinfer_c_api.h      # C interface
├── flashinfer_c_api.cu     # Implementation (instantiations)
└── CMakeLists.txt          # For standalone builds
```

### 1.2 Build System (Weeks 2-3)

**Goal:** Integrate nvcc compilation into cargo build

**Tasks:**
- [ ] Implement `build.rs`:
  - CUDA_HOME verification
  - .cu compilation via cc crate with `cuda(true)`
  - CUDA runtime linking
  - bindgen for FFI generation
- [ ] Feature flag support:
  - `cuda-11` / `cuda-12`
  - `sm80` / `sm90` architectures
- [ ] Compilation caching

**Key build.rs code:**
```rust
cc::Build::new()
    .cuda(true)
    .flag("-std=c++17")
    .flag(&format!("-gencode=arch=compute_{},code=sm_{}", arch, arch))
    .include("references/flashinfer/include")
    .file("csrc/flashinfer_c_api.cu")
    .compile("flashinfer_kernels");
```

### 1.3 Rust FFI Layer (Weeks 3-5)

**Goal:** Safe wrappers over C API

**Tasks:**
- [ ] Generate `src/ffi/generated.rs` via bindgen
- [ ] Create `src/ffi.rs` with safe Rust API:
  - `BatchDecodeHandler` - RAII wrapper
  - `BatchPrefillHandler` - RAII wrapper
  - `Workspace` - GPU memory for intermediate computations
- [ ] Integration with existing `PageTable`
- [ ] Integration with cudarc for stream/memory management

**API Example:**
```rust
pub struct BatchDecodeHandler {
    workspace: Workspace,
    config: AttentionConfig,
}

impl BatchDecodeHandler {
    pub fn new(config: AttentionConfig, stream: &CudaStream) -> Result<Self>;

    pub fn forward(
        &mut self,
        query: &DeviceSlice<f16>,
        paged_kv: &PagedKvCache,
        output: &mut DeviceSlice<f16>,
    ) -> Result<()>;
}
```

### 1.4 Testing (Weeks 5-7)

**Tasks:**
- [ ] Unit tests for each component
- [ ] Integration tests with real GPU operations
- [ ] Numerical verification against PyTorch FlashInfer
- [ ] Performance benchmarks

### 1.5 Documentation (Week 8)

**Tasks:**
- [ ] API documentation (rustdoc)
- [ ] Usage examples
- [ ] Build guide

---

## Phase 2: Simple Rust Kernels (3-6 months)

### 2.1 rust-cuda Setup (2-3 weeks)

**Goal:** Prepare environment for writing CUDA in Rust

**Tasks:**
- [ ] Configure rustc_codegen_nvvm
- [ ] Create template project for kernels
- [ ] Validate toolchain on simple examples

### 2.2 Utility Kernel Porting (2-3 months)

**Priority:**
1. **RoPE (Rotary Position Embedding)** - isolated, well-defined
2. **RMSNorm / LayerNorm** - simple math
3. **Masking kernels** - element-wise operations

**For each:**
- Rust implementation
- Numerical accuracy tests
- Benchmarks vs C++ version
- Fallback to C++ on regression

### 2.3 Validation (1 month)

**Tasks:**
- [ ] A/B testing Rust vs C++ kernels
- [ ] Profiling (NCU, Nsight)
- [ ] Decision on migration continuation

---

## Phase 3: Attention Kernels in Rust (6-12+ months)

**Condition:** Only if Phase 2 successful and rust-cuda sufficiently stable

### 3.1 Batch Decode (3-4 months)

**Approach:**
- Start with one variant: fp16, head_dim=128, SM90
- Gradually add dtype/head_dim combinations
- Paged KV cache support

### 3.2 Batch Prefill (3-4 months)

**Complexity factors:**
- Q×K^T matrix multiplication
- Causal masking
- Higher memory bandwidth requirements

### 3.3 Full Migration (2-4 months)

- Remove C++ dependency
- Unified Rust codebase
- Final optimizations

---

## Project Structure (After Phase 1)

```
flashinfer-rs/
├── Cargo.toml
├── build.rs                    # nvcc + bindgen
├── csrc/
│   ├── flashinfer_c_api.h      # C interface
│   └── flashinfer_c_api.cu     # Template instantiations
├── references/
│   └── flashinfer/             # Git submodule
├── src/
│   ├── lib.rs                  # Public API
│   ├── error.rs                # Error types
│   ├── page_table.rs           # Paged KV cache (ready)
│   ├── cuda/
│   │   ├── mod.rs              # CUDA utilities
│   │   └── workspace.rs        # Workspace allocation
│   ├── ffi/
│   │   ├── mod.rs              # FFI module
│   │   └── generated.rs        # bindgen output
│   ├── batch_decode.rs         # Decode API
│   └── batch_prefill.rs        # Prefill API
└── tests/
    ├── decode_test.rs
    ├── prefill_test.rs
    └── numerical_validation.rs
```

---

## Success Criteria

### Phase 1
- [ ] `cargo build` compiles on machine with CUDA 12+
- [ ] Tests pass on NVIDIA GPU (SM80+)
- [ ] Performance within 5% of Python FlashInfer
- [ ] Documentation and examples

### Phase 2
- [ ] 3+ utility kernels rewritten in Rust
- [ ] No performance regression
- [ ] rust-cuda stable enough for continuation

### Phase 3
- [ ] Attention kernels in Rust
- [ ] Full independence from C++ FlashInfer
- [ ] Competitive performance

---

## Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| rust-cuda instability | Medium | Fallback to C++ bindings |
| Performance regression | Medium | Benchmarks at each stage |
| FlashInfer API changes | Low | Version pinning |
| CUDA version incompatibility | Low | Feature flags for versions |

---

## Verification

1. **Unit tests:** `cargo test`
2. **GPU tests:** `cargo test --features cuda` (requires NVIDIA GPU)
3. **Numerical validation:** Comparison with PyTorch FlashInfer
4. **Benchmarks:** `cargo bench` with criterion

---

## Next Steps

1. Create `csrc/` directory structure
2. Start with `flashinfer_c_api.h` - define minimal C interface
3. Implement `build.rs` for nvcc compilation
4. Generate FFI bindings with bindgen
5. Wrap in safe Rust API

//! Build script for flashinfer-rs.
//!
//! This script handles:
//! 1. Compiling FlashInfer CUDA kernels
//! 2. Generating Rust FFI bindings via bindgen
//!
//! # Prerequisites
//!
//! - CUDA toolkit (nvcc)
//! - C++ compiler with C++17 support
//! - FlashInfer source (place in `flashinfer/` or set FLASHINFER_PATH)

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=flashinfer/");

    // Check if we have CUDA
    let cuda_available = std::env::var("CUDA_HOME").is_ok()
        || std::env::var("CUDA_PATH").is_ok()
        || std::path::Path::new("/usr/local/cuda").exists();

    if !cuda_available {
        println!("cargo:warning=CUDA not found, building without CUDA support");
        return;
    }

    // Check if FlashInfer source is available
    let flashinfer_path = std::env::var("FLASHINFER_PATH")
        .unwrap_or_else(|_| "flashinfer".to_string());

    if !std::path::Path::new(&flashinfer_path).exists() {
        println!(
            "cargo:warning=FlashInfer source not found at '{}'. \
             Download from https://github.com/flashinfer-ai/flashinfer \
             or set FLASHINFER_PATH environment variable.",
            flashinfer_path
        );
        return;
    }

    // TODO: Implement actual kernel compilation
    //
    // Steps:
    // 1. Use cc crate to compile C++ glue code
    // 2. Use nvcc to compile CUDA kernels
    // 3. Link everything together
    // 4. Generate bindings with bindgen
    //
    // Example (pseudo-code):
    //
    // cc::Build::new()
    //     .cuda(true)
    //     .flag("-std=c++17")
    //     .include(&flashinfer_path)
    //     .include(format!("{}/include", flashinfer_path))
    //     .file("csrc/flashinfer_ops.cu")
    //     .compile("flashinfer_kernels");
    //
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=flashinfer_kernels");

    println!("cargo:warning=FlashInfer CUDA kernels not yet compiled (TODO)");
}

//! Build script for flashinfer-rs.
//!
//! This script handles:
//! 1. Detecting CUDA toolkit and GPU architecture
//! 2. Compiling FlashInfer CUDA kernels via cc/nvcc
//! 3. Generating Rust FFI bindings via bindgen
//!
//! # Environment Variables
//!
//! - `CUDA_HOME` or `CUDA_PATH`: Path to CUDA toolkit
//! - `FLASHINFER_PATH`: Path to FlashInfer source (default: `references/flashinfer`)
//! - `FLASHINFER_CUDA_ARCH`: Target SM architecture (e.g., "80", "90")
//!
//! # Feature Flags
//!
//! - `cuda`: Enable CUDA support (required for GPU operations)
//! - `cuda-11`: Target CUDA 11.x
//! - `cuda-12`: Target CUDA 12.x (default when cuda feature enabled)
//! - `sm80`: Target SM80 (Ampere)
//! - `sm90`: Target SM90 (Hopper)

use std::env;
use std::path::{Path, PathBuf};

/// CUDA source modules for incremental compilation.
/// Split from monolithic flashinfer_c_api.cu for faster rebuilds.
const CUDA_MODULES: &[&str] = &[
    "flashinfer_decode.cu",     // Batch decode attention (8 variants) - slowest
    "flashinfer_prefill.cu",    // Batch prefill attention (8 variants) - slowest
    "flashinfer_norm.cu",       // RMSNorm, LayerNorm, etc.
    "flashinfer_sampling.cu",   // top_k, top_p, etc.
    "flashinfer_rope.cu",       // Rotary position embedding
    "flashinfer_rope_quant.cu", // RoPE + Quantize + Append (SM89+)
    "flashinfer_page.cu",       // KV cache append
    "flashinfer_utils.cu",      // Utilities (GPU info, workspace)
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/flashinfer_common.h");
    println!("cargo:rerun-if-changed=csrc/flashinfer_c_api.h");
    for module in CUDA_MODULES {
        println!("cargo:rerun-if-changed=csrc/{}", module);
    }

    // Check if CUDA feature is enabled
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        println!("cargo:warning=Building without CUDA support. Enable 'cuda' feature for GPU operations.");
        return;
    }

    // Find CUDA toolkit
    let cuda_path = find_cuda_path();
    let cuda_path = match cuda_path {
        Some(path) => path,
        None => {
            println!("cargo:warning=CUDA toolkit not found. Set CUDA_HOME or CUDA_PATH environment variable.");
            println!("cargo:warning=Building without CUDA kernel support.");
            return;
        }
    };

    println!("cargo:warning=Found CUDA at: {}", cuda_path.display());

    // Find FlashInfer source
    let flashinfer_path = find_flashinfer_path();
    let flashinfer_path = match flashinfer_path {
        Some(path) => path,
        None => {
            println!(
                "cargo:warning=FlashInfer source not found. Run: git clone --depth 1 \
                 https://github.com/flashinfer-ai/flashinfer.git references/flashinfer"
            );
            println!("cargo:warning=Building without FlashInfer kernel support.");
            // Still generate stub bindings for the C API
            generate_stub_bindings();
            return;
        }
    };

    println!(
        "cargo:warning=Found FlashInfer at: {}",
        flashinfer_path.display()
    );

    // Determine target architecture
    let cuda_arch = determine_cuda_arch();
    println!("cargo:warning=Target CUDA architecture: SM{}", cuda_arch);

    // Compile CUDA kernels
    if let Err(e) = compile_cuda_kernels(&cuda_path, &flashinfer_path, cuda_arch) {
        println!("cargo:warning=Failed to compile CUDA kernels: {}", e);
        println!("cargo:warning=FFI functions will return UNSUPPORTED at runtime.");
        generate_stub_bindings();
        return;
    }

    // Generate Rust FFI bindings
    generate_bindings(&cuda_path, &flashinfer_path);

    // Link CUDA runtime
    link_cuda(&cuda_path);
}

/// Find CUDA toolkit installation path
fn find_cuda_path() -> Option<PathBuf> {
    // Check environment variables
    if let Ok(path) = env::var("CUDA_HOME") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "/opt/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
    ];

    for path in &common_paths {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

/// Find FlashInfer source directory
fn find_flashinfer_path() -> Option<PathBuf> {
    // Check environment variable
    if let Ok(path) = env::var("FLASHINFER_PATH") {
        let path = PathBuf::from(path);
        if path.join("include/flashinfer").exists() {
            return Some(path);
        }
    }

    // Check default location
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let default_path = PathBuf::from(&manifest_dir).join("references/flashinfer");
    if default_path.join("include/flashinfer").exists() {
        return Some(default_path);
    }

    // Check alternative location
    let alt_path = PathBuf::from(&manifest_dir).join("flashinfer");
    if alt_path.join("include/flashinfer").exists() {
        return Some(alt_path);
    }

    None
}

/// Determine target CUDA architecture
fn determine_cuda_arch() -> u32 {
    // Check environment variable
    if let Ok(arch) = env::var("FLASHINFER_CUDA_ARCH") {
        if let Ok(arch) = arch.parse::<u32>() {
            return arch;
        }
    }

    // Check feature flags
    if env::var("CARGO_FEATURE_SM90").is_ok() {
        return 90;
    }
    if env::var("CARGO_FEATURE_SM80").is_ok() {
        return 80;
    }

    // Default to SM80 (Ampere) for broad compatibility
    80
}

/// Compile CUDA kernels using cc crate.
///
/// Each module is compiled as a separate object file, enabling:
/// - Incremental builds: only changed modules recompile
/// - Parallel compilation: independent modules build concurrently
fn compile_cuda_kernels(
    cuda_path: &Path,
    flashinfer_path: &Path,
    cuda_arch: u32,
) -> Result<(), String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let csrc_path = PathBuf::from(&manifest_dir).join("csrc");

    // Verify common header exists
    let common_header = csrc_path.join("flashinfer_common.h");
    if !common_header.exists() {
        return Err(format!(
            "Common header not found: {}",
            common_header.display()
        ));
    }

    // Collect source files
    let mut source_files = Vec::new();
    for module in CUDA_MODULES {
        let source_path = csrc_path.join(module);
        if !source_path.exists() {
            return Err(format!("CUDA module not found: {}", source_path.display()));
        }
        source_files.push(source_path);
    }

    // Build with cc crate
    let mut build = cc::Build::new();

    build
        .cuda(true)
        .cudart("shared")
        .cpp(true)
        .std("c++17")
        // Include paths
        .include(&csrc_path)
        .include(flashinfer_path.join("include"))
        .include(cuda_path.join("include"))
        // Use gcc-12 as host compiler for compatibility with nvcc 12.0
        // NOTE: gcc-13 + glibc 2.39 is incompatible with nvcc 12.0
        .flag("-ccbin=g++-12")
        // CUDA architecture
        .flag(format!(
            "-gencode=arch=compute_{},code=sm_{}",
            cuda_arch, cuda_arch
        ))
        // Optimization flags
        .flag("-O3")
        .flag("--expt-relaxed-constexpr")
        .flag("--expt-extended-lambda")
        // Suppress some warnings
        .flag("-Wno-deprecated-gpu-targets");

    // Add all source files
    for source in &source_files {
        build.file(source);
    }

    // Add debug flags in debug mode
    if env::var("PROFILE").unwrap_or_default() == "debug" {
        build.flag("-G").flag("-lineinfo");
    }

    build.compile("flashinfer_kernels");

    Ok(())
}

/// Generate Rust FFI bindings using bindgen
fn generate_bindings(cuda_path: &Path, _flashinfer_path: &Path) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&manifest_dir).join("csrc/flashinfer_c_api.h");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .clang_arg(format!("-I{}", cuda_path.join("include").display()))
        // Parse settings
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Generate settings
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_hash(true)
        // Allowlist our API
        .allowlist_function("flashinfer_.*")
        .allowlist_type("FlashInfer.*")
        .allowlist_var("FLASHINFER_.*")
        // Block system types we don't need
        .blocklist_type("__.*")
        .blocklist_type("cuda.*")
        // Generate Rust-friendly enums
        .rustified_enum("FlashInferStatus")
        .rustified_enum("FlashInferDType")
        .rustified_enum("FlashInferKVLayout")
        .rustified_enum("FlashInferPosEncoding")
        .generate()
        .expect("Failed to generate FFI bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ffi_generated.rs"))
        .expect("Failed to write FFI bindings");
}

/// Generate stub bindings when FlashInfer is not available
fn generate_stub_bindings() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&manifest_dir).join("csrc/flashinfer_c_api.h");

    // Only generate if header exists
    if !header_path.exists() {
        println!("cargo:warning=C API header not found, skipping bindgen");
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .derive_eq(true)
        .allowlist_function("flashinfer_.*")
        .allowlist_type("FlashInfer.*")
        .rustified_enum("FlashInferStatus")
        .rustified_enum("FlashInferDType")
        .rustified_enum("FlashInferKVLayout")
        .rustified_enum("FlashInferPosEncoding")
        .generate()
        .expect("Failed to generate stub bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ffi_generated.rs"))
        .expect("Failed to write stub bindings");
}

/// Link CUDA runtime libraries
fn link_cuda(cuda_path: &Path) {
    // Add library search path
    let lib_path = if cfg!(target_os = "windows") {
        cuda_path.join("lib/x64")
    } else {
        cuda_path.join("lib64")
    };

    if lib_path.exists() {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }

    // Link our compiled kernels first (static library)
    // NOTE: Order matters! Static libraries must come before the libraries they depend on.
    println!("cargo:rustc-link-lib=static=flashinfer_kernels");

    // Link CUDA runtime (provides __cudaLaunchKernel, __cudaPopCallConfiguration, etc.)
    println!("cargo:rustc-link-lib=cudart");

    // Link C++ standard library (provides __cxa_guard_acquire, etc.)
    println!("cargo:rustc-link-lib=stdc++");
}

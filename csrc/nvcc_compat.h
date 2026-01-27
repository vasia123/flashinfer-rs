// Compatibility header for nvcc with newer glibc headers.
//
// NVCC (CUDA 12.0) doesn't understand _Float32, _Float64, etc. types that
// appear in glibc 2.38+ headers. This header provides forward declarations
// to suppress errors.
//
// NOTE: This must be included BEFORE any system headers that use these types.

#ifndef FLASHINFER_NVCC_COMPAT_H
#define FLASHINFER_NVCC_COMPAT_H

#ifdef __CUDACC__
// Define the missing floating point types as aliases to avoid nvcc errors.
// These are only used in function prototypes we don't actually call.
typedef float _Float32;
typedef double _Float64;
typedef long double _Float128;
typedef float _Float32x;
typedef double _Float64x;
#endif

#endif // FLASHINFER_NVCC_COMPAT_H

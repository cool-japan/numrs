# NumRS2 v0.1.1 Release Notes

**First Stable Release** - Production-Ready NumPy + SciPy Implementation in Rust

*Release Date: December 30, 2025*

NumRS2 v0.1.1 is the **first stable release** of NumRS2, a comprehensive numerical computing library for Rust. This release delivers production-ready NumPy and SciPy compatibility with SIMD-optimized operations, expression templates for lazy evaluation, and seamless integration with the SciRS2 ecosystem.

## 🎯 Overview

NumRS2 provides a complete numerical computing stack in pure Rust:
- **NumPy-compatible array operations** with broadcasting and advanced indexing
- **SciPy-equivalent modules** for optimization, interpolation, signal processing, and more
- **SIMD optimization** with AVX2/AVX512 and ARM NEON support
- **Expression templates** for lazy evaluation and automatic optimization
- **Pure Rust dependencies** with OxiBLAS (no C/C++ dependencies)

## ✨ Key Features

### Core Array Operations
- N-dimensional arrays with efficient memory layout
- NumPy-compatible broadcasting
- Advanced indexing (fancy indexing, boolean masking)
- Zero-copy views and slicing
- Expression templates for lazy evaluation
- Common Subexpression Elimination (CSE)

### Linear Algebra
- Matrix operations (multiplication, transpose, inverse, determinant)
- Decompositions (SVD, QR, LU, Cholesky, Eigenvalue)
- Iterative solvers (CG, GMRES, BiCGSTAB)
- Randomized algorithms for large-scale computations
- Sparse matrix support (COO, CSR, CSC, DIA)

### SIMD Optimization
- **86 AVX2-optimized functions** with automatic threshold-based dispatch
- **42 ARM NEON operations** for f64 vectorization
- 4-way loop unrolling and FMA (fused multiply-add) instructions
- Support for both f32 and f64 numeric types
- Automatic fallback to scalar implementations

### Mathematical & Statistical Functions
- Comprehensive mathematical operations (trigonometric, exponential, logarithmic)
- Special functions (gamma, beta, error functions, Bessel functions)
- Polynomial operations (evaluation, fitting, root finding)
- Cubic spline interpolation with multiple boundary conditions
- Statistical analysis and distribution functions

### Numerical Optimization
- BFGS & L-BFGS quasi-Newton methods
- Trust Region optimization
- Nelder-Mead simplex method
- Levenberg-Marquardt for nonlinear least squares
- Constrained optimization algorithms

### Root-Finding Algorithms
- Bracketing methods (Bisection, Brent, Ridder)
- Open methods (Newton-Raphson, Secant, Halley)
- Fixed-point iteration

### Signal Processing
- Fast Fourier Transform (FFT/IFFT)
- Convolution and correlation
- Digital filtering operations

### Interoperability
- NumPy format (.npy, .npz) support
- Apache Arrow integration for zero-copy data exchange
- CSV and binary serialization
- Memory-mapped file I/O
- Optional Python bindings via PyO3

### SciRS2 Ecosystem Integration

NumRS2 uses the SciRS2 ecosystem (v0.1.1):
```toml
scirs2-core = "0.1.1"
scirs2-stats = "0.1.1"
scirs2-linalg = "0.1.1"
scirs2-ndimage = "0.1.1"
scirs2-spatial = "0.1.1"
scirs2-special = "0.1.1"
scirs2-fft = "0.1.1"
scirs2-signal = "0.1.1"
```

All dependencies use **stable releases** with:
- OxiBLAS v0.1.2 (pure Rust BLAS/LAPACK)
- Oxicode v0.1.1 (pure Rust serialization)
- No C/C++ dependencies

## 📦 Installation

Add to your `Cargo.toml`:

```toml
numrs2 = "0.1.1"
```

With optional features:
```toml
numrs2 = { version = "0.1.1", features = ["arrow"] }
numrs2 = { version = "0.1.1", features = ["python"] }
numrs2 = { version = "0.1.1", features = ["lapack"] }
numrs2 = { version = "0.1.1", features = ["gpu"] }
```

## 📊 Technical Metrics

- **Total Rust Code**: ~155,000 lines of production code
- **Test Coverage**: 1,111+ unit tests passing
- **Quality Metrics**: Zero compilation warnings, zero clippy errors
- **SIMD Operations**: 128 vectorized functions (86 AVX2 + 42 NEON)
- **Documentation**: Comprehensive docs with examples and migration guides

## 🚀 Performance

- **SIMD-optimized** operations with automatic threshold-based dispatch
- **Cache-aware** memory access patterns
- **Expression templates** eliminate temporary allocations
- **Parallel operations** with work-stealing scheduler
- **Pure Rust** implementation with no C/C++ overhead

## 🔧 Optional Features

- `matrix_decomp` (default): Matrix decomposition functions
- `lapack`: LAPACK-dependent operations (via OxiBLAS)
- `validation`: Additional runtime validation
- `arrow`: Apache Arrow integration
- `python`: Python bindings via PyO3
- `gpu`: GPU acceleration via WGPU

## 📚 Documentation

- [Getting Started Guide](GETTING_STARTED.md)
- [API Documentation](https://docs.rs/numrs2)
- [Examples Directory](examples/)
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [SciRS2 Integration Guide](SCIRS2_INTEGRATION_POLICY.md)

## 🎉 What's New in 0.1.1

This is the **first stable release** of NumRS2. Key highlights:

- Production-ready quality with comprehensive test coverage
- Pure Rust dependencies (SciRS2 v0.1.1, OxiBLAS v0.1.2)
- Complete NumPy and SciPy compatibility
- SIMD optimization for maximum performance
- Expression templates for automatic optimization
- Zero compilation warnings and clippy errors

## 🔗 Links

- **Repository**: https://github.com/cool-japan/numrs
- **Crates.io**: https://crates.io/crates/numrs2
- **Documentation**: https://docs.rs/numrs2
- **License**: MIT OR Apache-2.0

## 🙏 Acknowledgments

NumRS2 builds on the excellent work of:
- The SciRS2 ecosystem for scientific computing
- OxiBLAS for pure Rust BLAS/LAPACK
- The Rust community for foundational libraries

---

**NumRS2 v0.1.1** - Production-ready numerical computing for Rust 🚀

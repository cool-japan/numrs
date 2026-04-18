# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2026-04-18

### Added
- **New Benchmarks**: Added comprehensive benchmarks for I/O operations (`bench/io_benchmarks.rs`), complex number operations (`benches/complex_benchmark.rs`), and sparse matrix operations (`benches/sparse_benchmark.rs`)
- **Distributed Optimization**: Enhanced distributed optimization module in `src/distributed/optimization.rs`

### Changed
- **Dependency Upgrades**: Updated all dependencies to latest versions in Cargo.toml

### Fixed
- **Linter Compliance**: Resolved clippy warnings in benchmark files and distributed optimization module

## [0.3.2] - 2026-03-27

### Changed
- **Version Bump**: Updated to v0.3.2 patch release
- **PyPI Compatibility**: Improved PyPI publishing configuration

### Fixed
- **MOS (Minimum Output Size)**: Resolved minimum output size constraints

## [0.3.1] - 2026-03-21

### Fixed
- **Clippy Warnings**: Resolved all 24 clippy warnings for MSRV compatibility
  - Fixed `Color` trait ambiguity in viz modules (`matrix.rs`, `perf.rs`, `plot2d.rs`, `plot3d.rs`, `stats.rs`) by adding explicit `use plotters::style::Color` imports
  - Replaced explicit counter loops with idiomatic `enumerate`/`zip` pattern in `src/cluster.rs`
  - Replaced manual checked division patterns with `.checked_div()` in `performance_tuning.rs`, `access_patterns.rs`, and `scheduler.rs`

## [0.3.0] - 2026-03-06

### Changed
- **SciRS2 Ecosystem Update**: Updated all scirs2-* dependencies to v0.3.0
  - scirs2-core v0.3.0: Latest core with enhanced SIMD, parallel, random operations
  - scirs2-linalg v0.3.0: Linear algebra improvements with OxiBLAS
  - scirs2-stats v0.3.0: Statistical functions enhancements
  - scirs2-fft v0.3.0: FFT operations improvements
  - scirs2-ndimage v0.3.0: N-dimensional image processing updates
  - scirs2-spatial v0.3.0: Spatial algorithms with improved KD-trees
  - scirs2-special v0.3.0: Special functions updates
  - scirs2-numpy v0.3.0: Python bindings compatibility updates
- **NPZ Compression**: Enabled DEFLATE compression for .npz files (OxiARC v0.3.0+ multi-file bug fixed)
- **Cyclic Spline Solver**: Replaced O(n²) Gaussian elimination with Sherman-Morrison O(n) cyclic Thomas algorithm

### Fixed
- Fixed WASM version assertion tests to use "0.3" instead of "0.2"

## [0.2.0] - 2026-01-30

### Changed
- **COOLJAPAN Ecosystem Compliance**: Full compliance with COOLJAPAN pure Rust policies
  - Replaced `numpy` dependency with `scirs2-numpy` (v0.3.0) for Python bindings
  - Removed OpenBLAS linker flags from `.cargo/config.toml` (now using OxiBLAS pure Rust backend)
  - Removed `cdylib` crate-type (Python extension builds handled by maturin)

### Fixed
- Fixed linking errors when building with `--all-features` due to openblas flags
- Fixed Python symbol resolution issues in test builds

### Dependencies
- **scirs2-numpy**: v0.3.0 (replaces direct numpy dependency)
- **SciRS2 Ecosystem**: scirs2-* v0.3.0 (latest stable releases)
- All Python bindings now go through SciRS2 ecosystem

## [0.1.1] - 2025-12-30

### Added
- **Initial Release**: First stable release of NumRS2, a NumPy-inspired numerical computing library for Rust
- **Core Array Operations**: Comprehensive ndarray-like API with multi-dimensional arrays
  - Array creation, manipulation, and reshaping operations
  - Broadcasting support for element-wise operations
  - Advanced indexing (fancy indexing, boolean masking, multi-dimensional slicing)
  - Zero-copy views and efficient memory management

- **Expression Templates**: Lifetime-free expression template system for lazy evaluation
  - SharedArray<T> with reference-counted storage for O(1) cloning
  - Operator overloading for natural syntax (+, -, *, /, scalar operations)
  - Common Subexpression Elimination (CSE) for automatic optimization
  - Cache-aware memory access patterns for improved performance

- **SIMD Optimization**: Comprehensive vectorization support
  - AVX2/AVX512 support for x86_64 architectures
  - ARM NEON support for ARM architectures
  - Automatic threshold-based dispatch between SIMD and scalar implementations
  - 86+ optimized functions with 4-way loop unrolling and FMA instructions

- **Linear Algebra**: Complete linear algebra stack
  - Matrix operations (multiplication, transpose, inverse, determinant)
  - Decompositions (SVD, QR, LU, Cholesky, Eigenvalue)
  - Iterative solvers (Conjugate Gradient, GMRES, BiCGSTAB)
  - Randomized algorithms for large-scale computations
  - Sparse matrix support (COO, CSR, CSC, DIA formats)

- **Mathematical Functions**: Extensive mathematical operations
  - Trigonometric, hyperbolic, exponential, logarithmic functions
  - Special functions (gamma, beta, error functions, Bessel functions)
  - Polynomial operations (evaluation, fitting, root finding)
  - Cubic spline interpolation with multiple boundary conditions

- **Statistical Functions**: Comprehensive statistical toolkit
  - Descriptive statistics (mean, median, variance, standard deviation)
  - Distribution functions and random number generation
  - Hypothesis testing and correlation analysis
  - Integration with SciRS2 statistical modules

- **Signal Processing**: FFT and filtering operations
  - Fast Fourier Transform (FFT/IFFT)
  - Convolution and correlation
  - Digital filtering operations

- **Interoperability**: Multiple data format support
  - NumPy format (.npy, .npz) for Python compatibility
  - Apache Arrow integration for zero-copy data exchange
  - CSV and binary serialization support
  - Memory-mapped file I/O

- **Financial Computing**: Financial analysis tools
  - Options pricing models
  - Bond valuation
  - Time value of money calculations
  - Financial metrics and indicators

- **Automatic Differentiation**: Forward and reverse mode AD
  - Dual numbers for forward mode
  - Tape-based backpropagation for reverse mode
  - Higher-order derivatives (Hessian, Taylor series)

- **SciRS2 Ecosystem Integration**: Built on the SciRS2 scientific computing foundation
  - scirs2-core v0.3.0: SIMD, parallel, random, array operations
  - scirs2-linalg v0.3.0: Linear algebra with OxiBLAS
  - scirs2-stats v0.3.0: Statistical functions
  - scirs2-fft v0.3.0: FFT operations
  - scirs2-signal v0.3.0: Signal processing
  - scirs2-special v0.3.0: Special functions
  - scirs2-ndimage v0.3.0: N-dimensional image processing
  - scirs2-spatial v0.3.0: Spatial algorithms

### Technical Details
- **Total Rust Code**: ~155,000 lines of production-ready code
- **Test Coverage**: 1,111+ unit tests passing, comprehensive test suite
- **Quality Metrics**: Zero compilation warnings, zero clippy errors
- **Performance**: SIMD-optimized operations with automatic fallback
- **Pure Rust**: No C/C++ dependencies, built on OxiBLAS (pure Rust BLAS/LAPACK)

### Dependencies
- **SciRS2 Ecosystem**: scirs2-* v0.3.0 (stable releases)
- **OxiBLAS**: v0.3.0 (pure Rust BLAS/LAPACK implementation)
- **Oxicode**: v0.3.0 (pure Rust serialization)
- All dependencies use stable, production-ready versions

This initial release provides a comprehensive NumPy-like experience in Rust with production-ready quality, extensive test coverage, and pure Rust dependencies for maximum portability and safety.

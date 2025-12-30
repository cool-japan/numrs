# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - scirs2-core v0.1.1: SIMD, parallel, random, array operations
  - scirs2-linalg v0.1.1: Linear algebra with OxiBLAS
  - scirs2-stats v0.1.1: Statistical functions
  - scirs2-fft v0.1.1: FFT operations
  - scirs2-signal v0.1.1: Signal processing
  - scirs2-special v0.1.1: Special functions
  - scirs2-ndimage v0.1.1: N-dimensional image processing
  - scirs2-spatial v0.1.1: Spatial algorithms

### Technical Details
- **Total Rust Code**: ~155,000 lines of production-ready code
- **Test Coverage**: 1,111+ unit tests passing, comprehensive test suite
- **Quality Metrics**: Zero compilation warnings, zero clippy errors
- **Performance**: SIMD-optimized operations with automatic fallback
- **Pure Rust**: No C/C++ dependencies, built on OxiBLAS (pure Rust BLAS/LAPACK)

### Dependencies
- **SciRS2 Ecosystem**: scirs2-* v0.1.1 (stable releases)
- **OxiBLAS**: v0.1.2 (pure Rust BLAS/LAPACK implementation)
- **Oxicode**: v0.1.1 (pure Rust serialization)
- All dependencies use stable, production-ready versions

This initial release provides a comprehensive NumPy-like experience in Rust with production-ready quality, extensive test coverage, and pure Rust dependencies for maximum portability and safety.

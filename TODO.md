# NumRS2 Development Roadmap

This document outlines the plan for porting NumPy's functionality to NumRS2, a high-performance numerical computing library for Rust. The goal is to provide a comprehensive, NumPy-like experience with native Rust performance.

## 🔍 Current Status

NumRS2 currently provides:
- ✅ Basic n-dimensional array implementation with essential operations
- ✅ Integration with BLAS/LAPACK for linear algebra
- ✅ SIMD support for vectorized operations
- ✅ Foundational mathematical and statistical functions
- ✅ Parallel computation with Rayon
- ✅ Matrix decompositions (SVD, QR, LU, Cholesky, Schur, etc.)
- ✅ Eigenvalue calculations
- ✅ Polynomial functions and interpolation
- ✅ Fast Fourier Transform (FFT) implementation
- ✅ Sparse matrix support
- ✅ Broadcasting and array manipulation operations
- ✅ Special functions (erf, gamma, Bessel functions)
- ✅ Array comparison operations (allclose, isclose, array_equal)
- ✅ Shape manipulation operations (ravel, flatten, swapaxes, moveaxis)
- ✅ Array serialization/deserialization (JSON, CSV, Binary, NPY/NPZ)
- ✅ Masked arrays for handling missing or invalid data
- ✅ Matrix Library with matrix-specific behavior and special matrices
- ✅ Memory-mapped arrays for handling large datasets efficiently

## 🎯 NumPy Feature Parity Plan

### 1. Array Creation and Manipulation (Priority: 🔴)

#### Array Creation Functions
- [x] Full implementation of `zeros_like`, `ones_like`, `empty_like`
- [x] Implement `identity`, `eye`, `diag`, `diagflat`
- [x] Add `tri`, `tril`, `triu` for triangular matrix creation
- [x] Support for `meshgrid` with different indexing modes
- [x] Add `mgrid`, `ogrid` for dense and open mesh grid creation
- [x] Implement `r_` and `c_` for row and column stack
- [ ] Support for advanced random distributions comparable to `numpy.random`

#### Array Manipulation Functions
- [x] Complete implementation of `reshape` with copy/no-copy options
- [x] Add `ravel`, `flatten` with memory layout options
- [x] Implement `roll`, `rollaxis` for rolling elements
- [x] Add `rot90` for rotating arrays in planes
- [x] Implement `flip`, `fliplr`, `flipud` for reversing arrays
- [x] Enhance `concatenate` with ability to specify multiple axes
- [x] Add `block` for constructing arrays from blocks
- [x] Implement `hsplit`, `vsplit`, `dsplit` for array splitting

### 2. Indexing and Slicing (Priority: 🔴)

- [x] Complete implementation of advanced indexing with boolean and integer arrays
- [x] Add support for mixing slice objects and advanced indices
- [x] Support for ellipsis (`...`) in indexing expressions
- [x] Add `take`, `choose`, `compress`, `diagonal` indexing functions
- [x] Implement `put` and `putmask` for setting array values
- [x] Add `ix_` for generating index arrays
- [x] Support for `mask_indices` for n-dimensional indexing

### 3. Mathematical Functions (Priority: 🔴)

#### Element-wise Mathematical Functions
- [x] Complete suite of trigonometric functions (`sin`, `cos`, etc.)
- [x] Add hyperbolic functions (`sinh`, `cosh`, etc.)
- [x] Implement rounding functions (`round`, `floor`, `ceil`, etc.)
- [x] Add `clip` for limiting values within an interval
- [x] Support for `conj`, `real`, `imag`, `angle` for complex numbers
- [x] Implement `unwrap` function for phase unwrapping

#### Arithmetic Operations
- [x] Enhance broadcasting with full NumPy semantics
- [x] Add `power` with broadcasting
- [x] Implement `outer` for outer product
- [x] Add `kron` for Kronecker product
- [x] Support for `tensordot` and `einsum` for tensor calculations
- [x] Implement complex number arithmetic operations

#### Exponential and Logarithmic Functions
- [x] Add `expm1`, `log1p` for improved accuracy
- [x] Implement `logaddexp`, `logaddexp2` for log-domain calculations
- [x] Add `log10`, `log2` for different log bases

### 4. Linear Algebra (Priority: 🔴)

- [x] Enhance matrix multiplication with `matmul`, `dot` with full broadcasting
- [x] Implement `vdot` for complex conjugating dot product
- [x] Add `inner`, `outer` for inner and outer products
- [x] Support for `linalg.matrix_power` for matrix exponentiation
- [x] Implement tensor contraction operations
- [x] Add `pinv` for Moore-Penrose pseudoinverse
- [x] Implement `matrix_rank` for rank determination
- [x] Add `trace` for sum of diagonal elements
- [x] Support for solving linear matrix equation with `solve`
- [x] Enhance eigenvalue calculations with sorting options

### 5. Statistics (Priority: 🟠)

- [x] Implement percentile and quantile calculations
- [x] Add `histogram2d`, `histogramdd` for multi-dimensional histograms
- [x] Support for `bincount` for counting occurrences
- [x] Implement `digitize` for binning elements
- [x] Add `corrcoef` for correlation coefficient matrix
- [x] Support for `cov` for covariance matrix
- [x] Implement various statistical functions (`ptp`, `average`, etc.)
- [x] Add weighted statistics calculations

### 6. Random Number Generation (Priority: 🟠)

- [x] Complete redesign based on NumPy's `numpy.random` API
- [x] Implement Generator class with various distributions
- [x] Add thread-safe RNG infrastructure
- [x] Support for proper seeding
- [x] Implement `RandomState` for backward compatibility
- [x] Add ability to create custom distributions
- [x] Support for `permutation`, `shuffle` operations

### 7. FFT Module (Priority: 🟠)

- [x] Enhance FFT with various types (`rfft`, `irfft`, `hfft`, `ihfft`)
- [x] Add `fftfreq`, `rfftfreq` for frequency calculations
- [x] Support for `fftshift`, `ifftshift` for shifting frequencies
- [x] Implement multi-dimensional FFT operations
- [x] Add windowing functions (`hanning`, `hamming`, etc.)
- [x] Support for real FFT with improved performance

### 8. Polynomial Module (Priority: 🟠)

- [x] Complete implementation of polynomial operations
- [x] Add `polyfit` for polynomial fitting
- [x] Support for `polyval` for polynomial evaluation
- [x] Implement `roots` for finding polynomial roots
- [x] Add Chebyshev polynomials support
- [x] Support for Hermite, Laguerre, and Legendre polynomials
- [x] Implement polynomial integration and differentiation

### 9. Special Functions (Priority: 🟢)

- [x] Add special mathematical functions from `scipy.special`
- [x] Implement error functions (`erf`, `erfc`, etc.)
- [x] Add gamma functions (`gamma`, `gammaln`, etc.)
- [x] Support for Bessel functions
- [x] Implement elliptic functions
- [x] Add orthogonal polynomials

### 10. Array Manipulation and Utility Functions (Priority: 🟠)

- [x] Complete implementation of shape manipulation operations (`ravel`, `flatten`, `swapaxes`, `moveaxis`, `atleast_1d/2d/3d`)
- [x] Add `broadcast_to`, `broadcast_arrays` functions
- [x] Implement `apply_along_axis`, `apply_over_axes` for function application
- [x] Support for `vectorize` to create universal functions
- [x] Add `lib.stride_tricks` module for advanced array manipulation (as_strided, sliding_window_view)
- [x] Implement array comparison operations with tolerance

### 11. Masked Arrays (Priority: 🔴)

- [x] Implement masked array data structure
- [x] Add masked array operations and methods
- [x] Support for mask creation and manipulation functions
- [x] Implement filling and transformation of masked values
- [x] Add masked array-specific functions equivalent to NumPy's maskedarray

### 12. Matrix Library (Priority: 🔴)

- [x] Add matrix class with matrix-specific behavior
- [x] Implement matrix-specific methods
- [x] Support for banded matrices
- [x] Add special matrix functions

### 13. Interoperability (Priority: 🔴)

- [x] Add serialization/deserialization for arrays (serde implementation)
- [x] Support for common file formats (CSV, JSON, Binary)
- [x] Add conversion functions for standard Rust data structures (Vec, Vec<Vec>)
- [x] Support for NPY/NPZ file formats
- [x] Implement conversion to/from other array libraries (ndarray, nalgebra)

### 14. Advanced Features (Priority: 🔴)

- [x] Implement memory-mapped arrays
- [x] Add support for datetime64 and timedelta64 data types
- [x] Implement structured arrays for heterogeneous data
- [x] Add record arrays for named fields
- [x] Support for custom dtypes

## 🚀 Project Development Tasks

### Documentation (Priority: 🔴)

- [ ] Comprehensive API documentation for all functions
- [ ] Detailed guides and tutorials mirroring NumPy's documentation
- [ ] Example notebook collection showing NumRS2 vs NumPy usage
- [ ] Cheat sheet for NumPy users switching to NumRS2
- [ ] Performance benchmarks against NumPy

### Testing (Priority: 🔴)

- [ ] Comprehensive test coverage for all functions
- [ ] Property-based testing for mathematical operations
- [ ] Reference tests against NumPy for correctness verification
- [ ] Benchmark suite for performance monitoring
- [ ] Test fixtures for common use cases

### Performance Optimization (Priority: 🔴)

- [x] CPU feature detection for optimal SIMD usage
- [x] Memory layout optimization for cache efficiency
- [x] Optimized data placement strategies
- [x] Enhanced parallel processing with optimized thresholds
- [x] Fine-grained parallelization strategies
- [x] Custom memory allocators for numerical workloads
- [ ] GPU acceleration for applicable operations

### Development Practices (Priority: 🔴)

- [ ] Zero warnings approach with clippy checks for all code
- [ ] Maintain zero warnings for tests and examples
- [ ] Add clippy checks to CI workflow
- [ ] Test-driven development (TDD) approach
- [ ] Efficient code review process
- [ ] Clear ticket/issue management

### Advanced Research Areas (Priority: 🟢)

- [ ] Explore differentiable programming capabilities
- [ ] Research into domain-specific language extensions
- [ ] Investigate distributed array operations
- [ ] Explore hardware acceleration beyond CPU/GPU
- [ ] Research into alternative memory layouts for specific workloads

## 📅 Development Phases

### Phase 1: Critical Features Implementation

Focus on implementing remaining major features:
- Advanced Features (memory-mapped arrays, datetime/timedelta, structured arrays)

### Phase 2: Performance Optimization

Enhance performance through:
- ✅ Advanced SIMD optimization with simba library
- ✅ CPU feature detection implementation
- ✅ SIMD utilization in more operations
- ✅ Cache efficiency improvements
- ✅ Memory layout optimization
- ✅ Data placement strategy improvements
- ✅ Parallel processing enhancements
- ✅ Parallel processing threshold optimization
- ✅ Fine-grained parallelization strategies
- ✅ Custom memory allocators for numerical workloads

### Phase 3: Documentation, Testing and Publishing

Complete the project with:
- Comprehensive unit tests for all functionality
- Reference testing against NumPy
- Property-based testing implementation
- Complete API documentation
- Tutorials and guides
- NumPy migration guide
- crates.io publishing preparation
- Repository preparation for public release
- Version planning

## 📋 Priority Legend

- 🔴 High priority - critical for NumPy parity
- 🟠 Medium priority - important for comprehensive functionality
- 🟢 Nice to have - enhances library but not essential for parity

## 🤝 Contributing

We welcome contributions to NumRS2! If you're interested in helping achieve NumPy parity in Rust:

1. Pick a feature from this roadmap that interests you
2. Create an issue describing what you'd like to implement
3. Discuss implementation approach with maintainers
4. Submit a PR with your implementation
5. Ensure proper documentation and tests

Let's work together to create a comprehensive numerical computing ecosystem for Rust!
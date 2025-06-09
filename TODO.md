# NumRS2 Development Roadmap (Updated)

This roadmap outlines the development plan for NumRS2, considering the integration strategy with SciRS2. The goal is to focus development efforts efficiently by leveraging existing functionality in SciRS2 while enhancing NumRS2's core capabilities.

## Development Strategy

NumRS2 development will follow these principles:

1. **Core Array Operations and Basic Functionality**: Implemented natively in NumRS2
2. **Advanced Scientific Computing Features**: Provided through SciRS2 integration
3. **Performance Optimization**: Important for both, but with special focus on NumRS2 core components

## Priority Implementation Items (NumRS2 Native)

### 1. Core Array Operation Optimization (Priority: High)

- ✓ Memory layout optimization for n-dimensional arrays
  - ✓ Real CPU feature detection with CPUID
  - ✓ Cache-friendly layout algorithms (Morton, Hilbert, blocked)
  - ✓ Cache oblivious algorithms for recursive optimization
- ✓ Data placement strategies for improved cache efficiency
  - ✓ Runtime CPU feature detection for optimal alignment
  - ✓ SIMD-friendly data alignment and placement
  - ✓ Cache-aware data organization
- ✓ Further SIMD optimization for basic operations
  - ✓ AVX2-optimized arithmetic operations (add, mul, div, sqrt)
  - ✓ SIMD-accelerated reductions and dot products
  - ✓ Fused multiply-add (FMA) operations
  - ✓ Runtime CPU feature detection and dispatch
- ✓ Advanced parameter support for array operation functions (axis parameter for unique())

### 2. Memory Management Improvements (Priority: High)

- ✓ I/O optimization for memory-mapped arrays
  - ✓ Access pattern detection and adaptive prefetching
  - ✓ Cache-optimized memory layout for mmap files
  - ✓ Sequential and strided access optimization
- ✓ Memory management strategies for large-scale data
  - ✓ Large-scale memory manager with spilling and cleanup
  - ✓ Out-of-core arrays for datasets larger than memory
  - ✓ Memory usage tracking and monitoring
  - ✓ Chunked processing iterators for memory efficiency
  - ✓ Automatic data spilling with configurable thresholds
  - ✓ Background cleanup of temporary data

### 3. Numerical Stability Improvements (Priority: Medium)

- ✓ Improved Cholesky decomposition numerical stability
- ✓ Enhanced QR decomposition orthogonality preservation
- ✓ Bessel K function implementation with better numerical properties

## Testing and Benchmarking (Both)

### 1. Benchmark Suite (Priority: High)

- [ ] Performance benchmarks against NumPy
- [ ] Core array operations benchmarks
- [ ] Large-scale data processing performance tests
- ✓ Numerical stability implementation benchmarks
- ✓ Matrix decomposition benchmarks for stability vs. performance

### 2. Test Enhancement (Priority: High)

- [ ] Comprehensive test coverage for core array operations
- [ ] Additional reference tests against NumPy
- ✓ Enhanced numerical stability tests with edge cases
- ✓ Mathematical property verification tests
- ✓ Advanced numerical properties testing with explicit ignorable cases

## Development Process Improvements

- [x] Zero warnings approach with clippy checks
  - ✓ FFT module optimization with improved transpose patterns and iterator usage
- [ ] CI/CD pipeline enhancement
- [ ] Introduction of Test-Driven Development (TDD)

## Documentation Improvements

- ✓ Detailed numerical stability enhancement documentation
- ✓ Technical implementation details documentation
- ✓ Code review checklists for numerical code
- ✓ Project development summary documentation

## SciRS2 Integration Features

The following features will be provided through SciRS2 integration:

1. **Advanced Mathematical and Special Functions**
2. **Matrix Decomposition Algorithms**
3. **FFT Implementation**
4. **Optimization Algorithms**
5. **Advanced Probability Distributions**

## Future Research Areas

- [ ] Differentiable programming capabilities
- [ ] Domain-specific language extensions
- [ ] Distributed array operations
- [ ] New hardware acceleration approaches

## Completed Items

- ✓ Basic n-dimensional array implementation with essential operations
- ✓ Integration with BLAS/LAPACK for linear algebra
- ✓ SIMD support for vectorized operations with auto CPU feature detection
- ✓ Parallel computation with workload-adaptive scheduling
- ✓ Comprehensive mathematical and statistical functions
- ✓ Modern random number generation API with thread safety
- ✓ Matrix decompositions (SVD, QR, LU, Cholesky, Schur, etc.)
  - ✓ With enhanced numerical stability for Cholesky and QR
- ✓ Eigenvalue calculations
- ✓ Polynomial functions and interpolation
- ✓ Fast Fourier Transform (FFT) implementation
  - ✓ 1D and 2D FFT/IFFT with optimized transpose operations
  - ✓ Real FFT optimization for improved memory usage
  - ✓ Frequency shifting operations (fftshift/ifftshift)
  - ✓ Window functions for spectral analysis
- ✓ Sparse matrix support
- ✓ Broadcasting and array manipulation operations
- ✓ Special functions (erf, gamma, Bessel functions)
  - ✓ With enhanced numerical stability for Bessel K
- ✓ Array comparison operations (allclose, isclose, array_equal)
- ✓ Shape manipulation operations (ravel, flatten, swapaxes, moveaxis)
- ✓ Array serialization/deserialization (JSON, CSV, Binary, NPY/NPZ)
- ✓ Masked arrays for handling missing or invalid data
- ✓ Matrix Library with matrix-specific behavior and special matrices
- ✓ Memory-mapped arrays for handling large datasets efficiently
- ✓ Advanced random distributions (noncentral chi-square, noncentral F, etc.)
- ✓ Compiler warning-free codebase with improved maintenance quality

## Recently Completed Enhancements

### Numerical Stability Improvements

- ✓ Improved Cholesky decomposition numerical stability with:
  - Enhanced diagonal perturbation strategies
  - Dynamic scaling to improve condition number
  - Gershgorin circle-based eigenvalue estimation
  - Better positive-definiteness detection
  - Symmetrization for handling minor numerical asymmetry
  - Special case detection for improved robustness

- ✓ Enhanced QR decomposition orthogonality with:
  - Comprehensive orthogonality assessment with both maximum and average deviation metrics
  - Modified Gram-Schmidt reorthogonalization for severe orthogonality issues
  - Improved tolerance calculation for ill-conditioned matrices
  - Update of R after improving Q to maintain A = QR identity
  - Matrix scaling to avoid overflow in large-magnitude entries

- ✓ Bessel K function improvements for numerically stable evaluation:
  - Specialized algorithm selection based on argument range (small, medium, large)
  - Enhanced small argument handling with series expansion
  - Monotonicity preservation for all argument values, especially K1 function
  - Recurrence relation accuracy for higher-order terms with overflow prevention
  - Accurate asymptotic expansions for large arguments with correction terms
  - Specialized implementation for K2 using exact recurrence

### Code Quality Improvements

- ✓ FFT module enhancements:
  - Improved matrix transpose operations using iterators and enumeration
  - Better memory management with pre-allocation of result vectors
  - More efficient and cleaner transpose operations in 2D FFT/IFFT implementations
  - Enhanced documentation and comments for complex operations
  - Clearer handling of odd-sized array operations in frequency shifting functions

### Performance Optimization Enhancements

- ✓ Advanced memory layout optimization:
  - CPU cache information detection using CPUID instructions
  - Space-filling curve algorithms (Morton Z-order, Hilbert curves)
  - Cache-oblivious recursive algorithms for multi-level cache optimization
  - Blocked matrix layouts for improved cache utilization

- ✓ Enhanced SIMD operations:
  - AVX2-optimized vectorized arithmetic operations for f32/f64
  - Runtime CPU feature detection and automatic dispatch
  - Fused multiply-add (FMA) instructions for improved numerical performance
  - SIMD-accelerated dot products and reductions with horizontal summation
  - Memory alignment optimization for SIMD register sizes

- ✓ Memory-mapped array optimizations:
  - Access pattern detection and classification (sequential, strided, random)
  - Adaptive prefetching based on detected access patterns
  - Cache-aligned data layout and page-boundary optimization
  - Global access pattern tracking for cross-session optimization

- ✓ Large-scale data management:
  - Memory usage tracking and monitoring with configurable limits
  - Automatic data spilling to disk when memory thresholds are exceeded
  - Out-of-core array implementation for datasets larger than available memory
  - Chunked processing iterators for memory-efficient operations
  - Background cleanup of temporary files and spilled data
  - LRU, LFU, and FIFO cache replacement strategies for chunk management
  - Configurable memory management policies and thresholds

## Priority Legend

- High - Critical for NumPy parity and core functionality
- Medium - Important for comprehensive functionality
- Low - Enhances library but not essential for parity

## Contributing

We welcome contributions to NumRS2! If you're interested in helping achieve NumPy parity in Rust:

1. Pick a feature from this roadmap that interests you
2. Create an issue describing what you'd like to implement
3. Discuss implementation approach with maintainers
4. Submit a PR with your implementation
5. Ensure proper documentation and tests

Let's work together to create a comprehensive numerical computing ecosystem for Rust!
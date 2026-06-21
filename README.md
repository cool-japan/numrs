# NumRS2 - High-Performance Numerical Computing for Rust

[![Build Status](https://github.com/cool-japan/numrs/workflows/CI/badge.svg)](https://github.com/cool-japan/numrs/actions)
[![Crates.io](https://img.shields.io/crates/v/numrs2.svg)](https://crates.io/crates/numrs2)
[![Documentation](https://docs.rs/numrs2/badge.svg)](https://docs.rs/numrs2)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NumRS2 is a high-performance numerical computing library for Rust, designed as a Rust-native alternative to NumPy. It provides N-dimensional arrays, linear algebra operations, and comprehensive mathematical functions with a focus on performance, safety, and ease of use.

> **Version 0.4.1** - Patch release (2026-06-21): zero-copy `Array::as_slice()` / `as_cow_1d()` API; erfinv/Bessel Y_n/K_n precision fixes (Winitzki 2008 + DLMF recurrences); irfft Hermitian reconstruction corrected; generic controlled-U quantum gate + multi-qubit gate fusion; `ArrayView::iter()` restored; `split` axis=1 enabled; visualization example activated; benchmark suite completed; pure-Rust `alloca` patch unblocks criterion on Apple Silicon. Features 128+ SIMD-vectorized functions (AVX2, AVX512, ARM NEON), 4,223 tests passing, 297,657+ lines of production Rust code, built on pure Rust SciRS2 v0.5.0 ecosystem. The core array, linear algebra, SIMD, and autodiff paths are production-ready; some advanced modules (e.g. quantum and parts of the reinforcement-learning suite) are experimental.

## ✨ Architecture Highlights

### 🏗️ Enhanced Design
- **Trait-based architecture** for extensibility and generic programming
- **Hierarchical error system** with rich context and recovery suggestions  
- **Memory management** with pluggable allocators (Arena, Pool, NUMA-aware)
- **Comprehensive documentation** with migration guides and best practices

### 🔧 Core Features
- **N-dimensional arrays** with efficient memory layout and broadcasting
- **Advanced linear algebra** with BLAS/LAPACK integration and matrix decompositions
- **SIMD optimization** with automatic vectorization and CPU feature detection
- **Thread safety** with parallel processing support via Rayon
- **Python interoperability** for easy migration from NumPy

## Main Features

- **N-dimensional Array**: Core `Array` type with efficient memory layout and NumPy-compatible broadcasting
- **Advanced Linear Algebra**:
  - Matrix operations, decompositions, solvers through BLAS/LAPACK integration
  - Sparse matrices (COO, CSR, CSC, DIA formats) with format conversions
  - Iterative solvers (CG, GMRES, BiCGSTAB) for large systems
  - Randomized algorithms (randomized SVD, random projections, range finders)
- **Numerical Optimization**: BFGS, L-BFGS, Trust Region, Nelder-Mead, Levenberg-Marquardt, constrained optimization
- **Root-Finding**: Bisection, Brent, Newton-Raphson, Secant, Halley, fixed-point iteration
- **Numerical Differentiation**: Gradient, Jacobian, Hessian with Richardson extrapolation
- **Automatic Differentiation**: Forward and reverse mode AD with higher-order derivatives
- **Data Interoperability**:
  - Apache Arrow integration for zero-copy data exchange
  - Feather format support for fast columnar storage
  - IPC streaming for inter-process communication
  - Python bindings via PyO3 for NumPy compatibility
- **Expression Templates**: Lazy evaluation and operation fusion for performance
- **Advanced Indexing**: Fancy indexing, boolean masking, and conditional selection
- **Polynomial Functions**: Interpolation, evaluation, and arithmetic operations
- **Fast Fourier Transform**: Optimized FFT implementation with 1D/2D transforms, real FFT specialization, frequency shifting, and various windowing functions
- **SIMD Acceleration**: Enhanced vectorized operations via SciRS2-Core with AVX2/AVX512/NEON support
- **Parallel Computing**: Advanced multi-threaded execution with adaptive chunking and work-stealing
- **GPU Acceleration**: Optional GPU-accelerated array operations via the WGPU backend (cross-platform: runs on Vulkan, Metal, DirectX 12, and WebGPU)
- **Mathematical Functions**: Comprehensive set of element-wise mathematical operations
- **Statistical Analysis**: Descriptive statistics, probability distributions, and more
- **Random Number Generation**: Modern interface for various distributions with fast generation and NumPy-compatible API
- **SciRS2 Integration**: Integration with SciRS2 for advanced statistical distributions and scientific computing functionality
- **Fully Type-Safe**: Leverage Rust's type system for compile-time guarantees

## Optional Features

NumRS2 includes several optional features that can be enabled in your `Cargo.toml`:

- **matrix_decomp** (enabled by default): Matrix decomposition functions (SVD, QR, LU, etc.)
- **lapack**: Enable LAPACK-dependent linear algebra operations (eigenvalues, matrix decompositions)
- **validation**: Additional runtime validation checks for array operations
- **arrow**: Apache Arrow integration for zero-copy data exchange with Python/Polars/DataFusion
- **python**: Python bindings via PyO3 for NumPy interoperability
- **gpu**: GPU acceleration for array operations using WGPU

To enable a feature:

```toml
[dependencies]
numrs2 = { version = "0.4.1", features = ["arrow"] }
```

Or, when building:

```bash
cargo build --features scirs
```

### 🚀 Performance Optimizations

NumRS2 leverages SciRS2-Core (v0.5.0) for cutting-edge performance optimizations:

- **Unified SIMD Operations**: All SIMD code goes through SciRS2-Core's SimdUnifiedOps trait
- **Adaptive Algorithm Selection**: AutoOptimizer automatically chooses between scalar, SIMD, or GPU implementations
- **Platform Detection**: Automatic detection of AVX2, AVX512, NEON, and GPU capabilities
- **Parallel Operations**: Optimized parallel processing with intelligent work distribution
- **Memory-Efficient Chunking**: Process large datasets without memory bottlenecks

See the [optimization example](examples/scirs2_optimization.rs) for usage details.

### SciRS2 Integration

The SciRS2 integration provides additional advanced statistical distributions:

- **Noncentral Chi-square**: Extends the standard chi-square with a noncentrality parameter
- **Noncentral F**: Extends the standard F distribution with a noncentrality parameter
- **Von Mises**: Circular normal distribution for directional statistics
- **Maxwell-Boltzmann**: Used for modeling particle velocities in physics
- **Truncated Normal**: Normal distribution with bounded support
- **Multivariate Normal with Rotation**: Allows rotation of the coordinate system

For examples, see [scirs_integration_example.rs](examples/scirs_integration_example.rs)

### GPU Acceleration

The GPU acceleration feature provides:

- GPU-accelerated array operations for significant performance improvements
- Seamless CPU/GPU interoperability with the same API
- Support for various operations: arithmetic, matrix multiplication, element-wise functions, etc.
- GPU acceleration via the WGPU backend (cross-platform: runs on Vulkan, Metal, DirectX 12, and WebGPU). NumRS2 does not ship native CUDA/ROCm/Metal/OpenCL backends; all GPU execution goes through WGPU.

For examples, see [gpu_example.rs](examples/gpu_example.rs)

### 🎯 Key Features

**Numerical Optimization (scipy.optimize equivalent)**
- BFGS & L-BFGS: Quasi-Newton methods for large-scale optimization
- Trust Region: Robust optimization with dogleg path
- Nelder-Mead: Derivative-free simplex method
- Levenberg-Marquardt: Nonlinear least squares
- Constrained optimization: Projected gradient, penalty methods

**Root-Finding Algorithms (scipy.optimize.root_scalar)**
- Bracketing methods: Bisection, Brent, Ridder, Illinois
- Open methods: Newton-Raphson, Secant, Halley
- Fixed-point iteration for implicit equations

**Numerical Differentiation**
- Gradient, Jacobian, and Hessian computation
- Forward, backward, central differences
- Richardson extrapolation for high accuracy

**SIMD Optimization Infrastructure**
- 86 AVX2-optimized functions with automatic threshold-based dispatch
- 4-way loop unrolling and FMA (fused multiply-add) instructions
- ARM NEON support with 42 vectorized f64 operations
- Support for both f32 and f64 numeric types

**Production-Ready Features**
- Complete multi-array NPZ support for NumPy compatibility
- Zero clippy warnings and zero critical errors
- 4,223 tests passing across all features (0 failures)
- Enhanced scheduler with critical deadlock fix (1,143x speedup)
- 297,657+ lines of production Rust code (676 Rust files)
- 5,813+ public API items; core array, linear algebra, SIMD, and autodiff paths are production-ready, with some advanced modules (e.g. quantum and parts of the reinforcement-learning suite) still experimental

**Enhanced Modules**
- Linear algebra: Extended iterative solvers (CG, GMRES, BiCGSTAB, FGMRES, MINRES)
- Mathematical functions: 1,187 lines of enhanced operations
- Statistics: 1,397 lines of enhanced distributions and testing
- Polynomial operations: Complete NumPy polynomial compatibility
- Special functions: Spherical harmonics, Jacobi elliptic, Lambert W, and more

## Example

```rust
use numrs2::prelude::*;

fn main() -> Result<()> {
    // Create arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
    
    // Basic operations with broadcasting
    let c = a.add(&b);
    let d = a.multiply_broadcast(&b)?;
    
    // Matrix multiplication
    let e = a.matmul(&b)?;
    println!("a @ b = {}", e);
    
    // Linear algebra operations
    let (u, s, vt) = a.svd_compute()?;
    println!("SVD components: U = {}, S = {}, Vt = {}", u, s, vt);
    
    // Eigenvalues and eigenvectors
    let symmetric = Array::from_vec(vec![2.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
    let (eigenvalues, eigenvectors) = symmetric.eigh("lower")?;
    println!("Eigenvalues: {}", eigenvalues);
    
    // Polynomial interpolation
    let x = Array::linspace(0.0, 1.0, 5)?;
    let y = Array::from_vec(vec![0.0, 0.1, 0.4, 0.9, 1.6]);
    let poly = PolynomialInterpolation::lagrange(&x, &y)?;
    println!("Interpolated value at 0.5: {}", poly.evaluate(0.5));
    
    // FFT operations
    let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // Window the signal before transforming
    let windowed_signal = signal.apply_window("hann")?;
    // Compute FFT
    let spectrum = windowed_signal.fft()?;
    // Shift frequencies to center the spectrum
    let centered = spectrum.fftshift_complex()?;
    println!("FFT magnitude: {}", spectrum.power_spectrum()?);
    
    // Statistical operations
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("mean = {}", data.mean()?);
    println!("std = {}", data.std()?);
    
    // Sparse array operations
    let mut sparse = SparseArray::new(&[10, 10]);
    sparse.set(&[0, 0], 1.0)?;
    sparse.set(&[5, 5], 2.0)?;
    println!("Density: {}", sparse.density());
    
    // SIMD-accelerated operations
    let result = simd_ops::apply_simd(&data, |x| x * x + 2.0 * x + 1.0)?;
    println!("SIMD result: {}", result);

    // Random number generation
    let rng = random::default_rng();
    let uniform = rng.random::<f64>(&[3])?;
    let normal = rng.normal(0.0, 1.0, &[3])?;
    println!("Random uniform [0,1): {}", uniform);
    println!("Random normal: {}", normal);

    Ok(())
}
```

## Performance

NumRS is designed with performance as a primary goal:

- **Rust's Zero-Cost Abstractions**: Compile-time optimization without runtime overhead
- **BLAS/LAPACK Integration**: Industry-standard libraries for linear algebra operations
- **SIMD Vectorization**: Parallel processing at the CPU instruction level with automatic CPU feature detection
- **Memory Layout Optimization**: Cache-friendly data structures and memory alignment
- **Data Placement Strategies**: Optimized memory placement for better cache utilization
- **Adaptive Parallelization**: Smart thresholds to determine when parallel execution is beneficial
- **Scheduling Optimization**: Intelligent selection of work scheduling strategies based on workload
- **Fine-grained Parallelism**: Advanced workload partitioning for better load balancing
- **Modern Random Generation**: Advanced thread-safe RNG with PCG64 algorithm for high-quality randomness

## Expression Templates

NumRS2 provides a powerful expression templates system for lazy evaluation and performance optimization:

### SharedArray - Reference-Counted Arrays

```rust
use numrs2::prelude::*;

// Create shared arrays with natural operator syntax
let a: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b: SharedArray<f64> = SharedArray::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

// Cheap cloning (O(1) - just increments reference count)
let a_clone = a.clone();

// Natural operator overloading
let sum = a.clone() + b.clone();         // [11.0, 22.0, 33.0, 44.0]
let product = a.clone() * b.clone();     // [10.0, 40.0, 90.0, 160.0]
let scaled = a.clone() * 2.0;            // [2.0, 4.0, 6.0, 8.0]
let result = (a.clone() + b.clone()) * 2.0 - 5.0;  // Chained operations
```

### SharedExpr - Lifetime-Free Lazy Evaluation

```rust
use numrs2::expr::{SharedExpr, SharedExprBuilder};

// Build expressions lazily - no computation until eval()
let c: SharedArray<f64> = SharedArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let expr = SharedExprBuilder::from_shared_array(c);
let squared = expr.map(|x| x * x);   // Expression built, not evaluated
let result = squared.eval();         // [1.0, 4.0, 9.0, 16.0] - evaluated here
```

### Common Subexpression Elimination (CSE)

```rust
use numrs2::expr::{CachedExpr, ExprCache};

// Automatic caching of repeated computations
let cache: ExprCache<f64> = ExprCache::new();
let cached_expr = CachedExpr::new(sum_expr.into_expr(), cache.clone());

let result1 = cached_expr.eval();  // Computes and caches
let result2 = cached_expr.eval();  // Uses cached result
```

### Memory Access Pattern Optimization

```rust
use numrs2::memory_optimize::access_patterns::*;

// Detect memory layout for optimization
let layout = detect_layout(&[100, 100], &[100, 1]);  // CContiguous

// Get optimization hints for array shapes
let hints = OptimizationHints::default_for::<f64>(10000);
println!("Block size: {}", hints.block_size);
println!("Use parallel: {}", hints.use_parallel);

// Cache-aware iteration for large arrays
let block_iter = BlockedIterator::new(10000, 64);
for block in block_iter {
    // Process block.start..block.end with cache efficiency
}

// Cache-aware operations
cache_aware_transform(&src, &mut dst, |x| x * 2.0);
cache_aware_binary_op(&a, &b, &mut result, |x, y| x + y);
```

See the [expression templates example](examples/expression_templates_example.rs) for a comprehensive demonstration.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.4.1"
```

For BLAS/LAPACK support, ensure you have the necessary system libraries:

**Note:** NumRS2 uses OxiBLAS, a pure Rust BLAS/LAPACK implementation with no C dependencies. You do NOT need to install system BLAS/LAPACK libraries.

To use LAPACK functionality (pure Rust via OxiBLAS):
```bash
cargo build --features lapack
cargo test --features lapack
```

OxiBLAS provides:
- Pure Rust implementation with SIMD optimizations (AVX2/NEON)
- No external C dependencies required
- 80-172% of OpenBLAS performance (competitive or faster on Apple M3)
- Complete BLAS Level 1/2/3 and LAPACK operations

## Implementation Details

NumRS2 is built on top of the SciRS2 ecosystem and pure Rust libraries:

- **SciRS2 ecosystem** (scirs2-core, scirs2-linalg, scirs2-stats, etc. v0.5.0): Provides the foundation for n-dimensional arrays, linear algebra, statistics, survival analysis, causal inference, bioinformatics, and combinatorics
- **OxiBLAS** (pure Rust BLAS/LAPACK): Powers high-performance linear algebra routines with no C dependencies
- **Oxicode**: Pure Rust serialization for data persistence
- **Rayon**: Enables parallel computation capabilities
- **num-traits / num-complex**: Provides generic numeric traits and complex number support for numerical operations

## Features

NumRS2 provides a comprehensive suite of numerical computing capabilities:

### Core Functionality
- **N-dimensional arrays** with efficient memory layout and broadcasting
- **Linear algebra operations** with BLAS/LAPACK integration
- **Matrix decompositions** (SVD, QR, Cholesky, LU, Schur, COD)
- **Eigenvalue and eigenvector computation**
- **Mathematical functions** with numerical stability optimizations

### Performance Optimizations
- **SIMD acceleration** with automatic CPU feature detection
- **Parallel processing** with adaptive scheduling and load balancing  
- **Memory optimization** with cache-friendly data structures
- **Vectorized operations** for improved computational efficiency

### Advanced Features
- **Fast Fourier Transform** with 1D/2D transforms and windowing functions
- **Polynomial operations** and interpolation methods
- **Sparse matrix support** for memory-efficient computations
- **Random number generation** with multiple distribution support
- **Statistical analysis** functions and descriptive statistics

### Integration & Interoperability
- **GPU acceleration** support via WGPU (optional)
- **SciRS2 integration** for advanced statistical distributions (optional)
- **Memory-mapped arrays** for large dataset handling
- **Serialization support** for data persistence

## 📖 Documentation

### 📚 Comprehensive Guides
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and core concepts
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrading from previous versions
- **[Trait System Guide](docs/TRAIT_GUIDE.md)** - Generic programming with NumRS2
- **[Error Handling Guide](docs/ERROR_HANDLING.md)** - Robust error management
- **[Memory Management Guide](docs/MEMORY_MANAGEMENT.md)** - Optimizing memory usage

### 🔗 Additional Resources
- [Official API Documentation](https://docs.rs/numrs2) - Complete API reference
- [Getting Started Guide](GETTING_STARTED.md) - Essential information for beginners
- [Installation Guide](INSTALL.md) - Detailed installation instructions
- [User Guide](GUIDE.md) - Comprehensive guide to all NumRS features
- [NumPy Migration Guide](NUMPY_MIGRATION.md) - Guide for NumPy users transitioning to NumRS2
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Current status and next steps
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to NumRS2

Module-specific documentation:
  - [Random Module Guide](examples/README_RANDOM.md) - Random number generation
  - [Statistics Module Guide](examples/README_STATISTICS.md) - Statistical functions
  - [Linear Algebra Guide](examples/README_LINALG.md) - Linear algebra operations
  - [Polynomial Guide](examples/README_POLYNOMIAL.md) - Polynomial operations
  - [FFT Guide](examples/README_FFT.md) - Fast Fourier Transform

Testing Documentation:
  - [Testing Guide](tests/README.md) - Guide for NumRS testing approach
  - Property-based testing for mathematical operations
    - Property tests for linear algebra operations
    - Property tests for special functions
    - Statistical validation of random distributions
  - Reference testing
    - Reference tests for random distributions
    - Reference tests for linear algebra operations
    - Reference tests for special functions
  - Benchmarking
    - Linear algebra benchmarks
    - Special functions benchmarks

## Examples

Check out the `examples/` directory for more usage examples:

- `basic_usage.rs`: Core array operations and manipulations
- `linalg_example.rs`: Linear algebra operations and solvers
- `simd_example.rs`: SIMD-accelerated computations
- `memory_optimize_example.rs`: Memory layout optimization for cache efficiency
- `parallel_optimize_example.rs`: Parallelization optimization techniques
- `random_distributions_example.rs`: Comprehensive examples of random number generation
- See the [examples README](examples/README.md) for more details

## Development

NumRS is in active development. See [TODO.md](TODO.md) for upcoming features and development roadmap.

## Testing

NumRS requires the `approx` crate for testing. Tests can be run after installation with:

```bash
cargo test
```

For running property-based and statistical tests for the random module:

```bash
cargo test --test test_random_statistical
cargo test --test test_random_properties
cargo test --test test_random_advanced
```

## Contributing

NumRS2 is a community-driven project, and we welcome contributions from everyone. There are many ways to contribute:

- **Code**: Implement new features or fix bugs
- **Documentation**: Improve guides, docstrings, or examples
- **Testing**: Write tests or improve existing ones
- **Reviewing**: Review pull requests from other contributors
- **Performance**: Identify bottlenecks or implement optimizations
- **Examples**: Create example code showing library usage

If you're interested in contributing, please read our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on how to get started.

For significant changes, please open an issue to discuss your ideas first.

## Sponsorship

NumRS2 is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find NumRS2 useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
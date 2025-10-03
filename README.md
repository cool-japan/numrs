# NumRS2 - High-Performance Numerical Computing for Rust

[![Build Status](https://github.com/cool-japan/numrs/workflows/CI/badge.svg)](https://github.com/cool-japan/numrs/actions)
[![Crates.io](https://img.shields.io/crates/v/numrs2.svg)](https://crates.io/crates/numrs2)
[![Documentation](https://docs.rs/numrs2/badge.svg)](https://docs.rs/numrs2)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NumRS2 is a high-performance numerical computing library for Rust, designed as a Rust-native alternative to NumPy. It provides N-dimensional arrays, linear algebra operations, and comprehensive mathematical functions with a focus on performance, safety, and ease of use.

> **🚀 Version 0.1.0-beta.3** - This beta release completes Phase 4 with advanced features including Apache Arrow integration, randomized linear algebra algorithms, comprehensive sparse matrix support, and automatic differentiation. Production-ready with 659 tests passing and zero warnings.

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
- **GPU Acceleration**: Optional GPU-accelerated array operations using WGPU
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
numrs2 = { version = "0.1.0-beta.3", features = ["arrow"] }
```

Or, when building:

```bash
cargo build --features scirs
```

### 🚀 Performance Optimizations (New in 0.1.0-beta.2)

NumRS2 now leverages SciRS2-Core for cutting-edge performance optimizations:

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
- WGPU backend for cross-platform GPU support (Vulkan, Metal, DX12, WebGPU)

For examples, see [gpu_example.rs](examples/gpu_example.rs)

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

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.1.0-beta.3"
```

For BLAS/LAPACK support, ensure you have the necessary system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# macOS
brew install openblas lapack
```

### macOS Apple Silicon Configuration

For Apple Silicon Macs (M1/M2/M3), additional configuration is required to properly link LAPACK libraries. Create a `.cargo/config.toml` file in your project root:

```toml
[build]
rustflags = ["-L", "/opt/homebrew/opt/openblas/lib", "-l", "openblas"]
```

This configuration ensures that the OpenBLAS library installed via Homebrew is properly linked when using LAPACK features. Without this configuration, you may encounter linking errors when building with the `lapack` feature enabled.

To use LAPACK functionality:
```bash
cargo build --features lapack
cargo test --features lapack
```

## Implementation Details

NumRS is built on top of several battle-tested libraries:

- **ndarray**: Provides the foundation for n-dimensional arrays
- **ndarray-linalg**: Provides BLAS/LAPACK bindings for linear algebra
- **num-complex**: Complex number support for advanced operations
- **BLAS/LAPACK**: Powers high-performance linear algebra routines
- **Rayon**: Enables parallel computation capabilities
- **num-traits**: Provides generic numeric traits for numerical operations

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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
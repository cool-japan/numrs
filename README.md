<h1 align="center">
NumRS2: High-Performance Numerical Computing in Rust
</h1><br>

[![Rust CI](https://github.com/cool-japan/numrs/actions/workflows/rust.yml/badge.svg)](https://github.com/cool-japan/numrs/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crate](https://img.shields.io/crates/v/numrs2.svg)](https://crates.io/crates/numrs2)

NumRS2 is the fundamental package for scientific computing with Rust. It provides:

- A powerful N-dimensional array object with broadcasting capabilities
- Sophisticated mathematical functions that operate on arrays
- Advanced linear algebra, statistical, and random number functionality
- High-performance SIMD and parallel computation abstractions
- Memory-optimized data structures for scientific computing

## Main Features

- **N-dimensional Array**: Core `Array` type with efficient memory layout and broadcasting
- **Linear Algebra**: Matrix operations, decompositions, solvers through BLAS/LAPACK integration
- **Polynomial Functions**: Interpolation, evaluation, and arithmetic operations
- **Fast Fourier Transform**: Optimized FFT implementation with 1D/2D transforms, real FFT specialization, frequency shifting, and various windowing functions
- **Sparse Arrays**: Memory-efficient representation for sparse data
- **SIMD Acceleration**: Vectorized math operations using SIMD instructions
- **Parallel Computing**: Multi-threaded execution with Rayon
- **GPU Acceleration**: Optional GPU-accelerated array operations using WGPU
- **Mathematical Functions**: Comprehensive set of element-wise mathematical operations
- **Statistical Analysis**: Descriptive statistics, probability distributions, and more
- **Random Number Generation**: Modern interface for various distributions with fast generation and NumPy-compatible API
- **SciRS2 Integration**: Optional integration with SciRS2 for advanced statistical distributions and scientific computing functionality
- **Fully Type-Safe**: Leverage Rust's type system for compile-time guarantees

## Optional Features

NumRS2 includes several optional features that can be enabled in your `Cargo.toml`:

- **matrix_decomp** (enabled by default): Matrix decomposition functions (SVD, QR, LU, etc.)
- **validation**: Additional runtime validation checks for array operations
- **scirs**: Integration with SciRS2 for advanced statistical distributions and scientific computing
- **gpu**: GPU acceleration for array operations using WGPU

To enable a feature:

```toml
[dependencies]
numrs2 = { version = "0.1.0-alpha.4", features = ["scirs"] }
```

Or, when building:

```bash
cargo build --features scirs
```

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
numrs2 = "0.1.0-alpha.4"
```

For BLAS/LAPACK support, ensure you have the necessary system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# macOS
brew install openblas lapack
```

## Implementation Details

NumRS is built on top of several battle-tested libraries:

- **ndarray**: Provides the foundation for n-dimensional arrays
- **ndarray-linalg**: Provides BLAS/LAPACK bindings for linear algebra
- **num-complex**: Complex number support for advanced operations
- **BLAS/LAPACK**: Powers high-performance linear algebra routines
- **Rayon**: Enables parallel computation capabilities
- **num-traits**: Provides generic numeric traits for numerical operations

## Current Status

NumRS is currently under active development. The current implementation includes:

✅ Complete:
- Basic array operations with broadcasting
- Integration with BLAS for fundamental operations
- SIMD optimization with CPU feature detection
- Memory layout optimization for cache efficiency
- Optimized data placement strategies
- Enhanced parallel processing with optimized thresholds
- Fine-grained parallelization strategies
- Adaptive scheduling for parallel computations
- Foundational mathematical functions
- Numerically stable matrix decompositions (SVD, QR, Cholesky, LU, Schur, COD)
- Condition number calculation and numerical stability assessment
- Eigenvalue and eigenvector computation
- Fast Fourier Transform (FFT) implementation:
  - Optimized 1D and 2D transforms with efficient transpose operations
  - Real FFT specialization for better memory usage
  - Frequency domain manipulation (fftshift/ifftshift)
  - Window functions for spectral analysis
  - Numerically stable implementation with code quality improvements
- Polynomial operations and interpolation
- Sparse matrix representations
- Memory-mapped arrays for large datasets
- Support for datetime64 and timedelta64 data types
- Structured arrays and record arrays with custom dtypes

## Numerical Stability and Code Quality Enhancements

NumRS includes advanced numerical stability features and code quality improvements to handle challenging computational scenarios:

### Numerical Stability

- **Enhanced Cholesky Decomposition**: Improved stability for ill-conditioned matrices using:
  - Dynamic scaling to improve conditioning
  - Gershgorin circle-based eigenvalue estimation
  - Intelligent diagonal perturbation
  - Special case detection for 2×2 matrices

- **Robust QR Decomposition**: Enhanced orthogonality preservation with:
  - Comprehensive orthogonality verification with multiple metrics
  - Modified Gram-Schmidt reorthogonalization for severe cases
  - Matrix consistency enforcement through QR = A verification
  - Adaptive tolerance thresholds based on matrix condition

- **Stable Special Functions**: Numerically stable implementations including:
  - Bessel K function with guaranteed monotonicity
  - Specialized algorithms for different argument ranges
  - Recurrence relation accuracy for higher orders
  - Asymptotic expansions with correction terms

### Code Quality Improvements

- **Optimized FFT Implementation**: Enhanced FFT module with:
  - Improved matrix transpose operations using iterators and enumeration
  - Better memory management with pre-allocation of result vectors
  - More efficient and cleaner transpose operations in 2D transforms
  - Enhanced documentation and comments for complex operations
  - Clearer handling of odd-sized array operations in frequency shifting

- **Zero-Warnings Approach**: Systematic elimination of compiler and clippy warnings:
  - Improved code patterns and idioms throughout the codebase
  - Better type safety with appropriate trait constraints
  - More consistent naming and code organization

For more details, see [Numerical Stability Enhancements](docs/NUMERICAL_STABILITY_ENHANCEMENTS.md)

🚧 In Progress:
- Custom memory allocators for numerical workloads
- GPU acceleration for applicable operations
- PyO3 bindings for Python interoperability
- Comprehensive test suite expansion
  - Property testing for numerical operations
  - Reference testing against NumPy
- Performance benchmarking and optimization
- Advanced NumPy/SciPy compatibility features

## Documentation

For detailed documentation, examples, and API reference:

- [Official API Documentation](https://docs.rs/numrs2)
- [Getting Started Guide](GETTING_STARTED.md) - Essential information for beginners
- [Installation Guide](INSTALL.md) - Detailed installation instructions
- [User Guide](GUIDE.md) - Comprehensive guide to all NumRS features
- [NumPy Migration Guide](NUMPY_MIGRATION.md) - Guide for NumPy users transitioning to NumRS2
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Current status and next steps
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to NumRS2
- [Benchmarking Guide](BENCHMARKING.md) - How to run and write benchmarks
- [Release Process](RELEASE.md) - Instructions for creating releases
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community standards and expectations
- [Governance](GOVERNANCE.md) - Project governance model

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
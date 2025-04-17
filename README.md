# NumRS2: High-Performance Numerical Computing in Rust

[![Rust CI](https://github.com/cool-japan/numrs/actions/workflows/rust.yml/badge.svg)](https://github.com/cool-japan/numrs/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crate](https://img.shields.io/crates/v/numrs2.svg)](https://crates.io/crates/numrs2)

NumRS2 is a comprehensive numerical computing library for Rust, designed to provide NumPy-like functionality with native performance. Built on top of highly optimized libraries and leveraging Rust's safety and performance features, NumRS2 aims to be the go-to solution for scientific computing, data analysis, and machine learning in Rust.

## 🚀 Features

- **N-dimensional Array**: Core `Array` type with efficient memory layout and broadcasting
- **Linear Algebra**: Matrix operations, decompositions, solvers through BLAS/LAPACK integration
- **Polynomial Functions**: Interpolation, evaluation, and arithmetic operations
- **Fast Fourier Transform**: FFT implementation with various windowing functions
- **Sparse Arrays**: Memory-efficient representation for sparse data
- **SIMD Acceleration**: Vectorized math operations using SIMD instructions
- **Parallel Computing**: Multi-threaded execution with Rayon
- **Mathematical Functions**: Comprehensive set of element-wise mathematical operations
- **Statistical Analysis**: Descriptive statistics, probability distributions, and more
- **Random Number Generation**: Various distributions with fast generation
- **Fully Type-Safe**: Leverage Rust's type system for compile-time guarantees

## 🧮 Example

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
    let spectrum = signal.fft()?;
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
    
    Ok(())
}
```

## 📊 Performance

NumRS is designed with performance as a primary goal:

- **Rust's Zero-Cost Abstractions**: Compile-time optimization without runtime overhead
- **BLAS/LAPACK Integration**: Industry-standard libraries for linear algebra operations
- **SIMD Vectorization**: Parallel processing at the CPU instruction level with automatic CPU feature detection
- **Memory Layout Optimization**: Cache-friendly data structures and memory alignment
- **Data Placement Strategies**: Optimized memory placement for better cache utilization
- **Adaptive Parallelization**: Smart thresholds to determine when parallel execution is beneficial
- **Scheduling Optimization**: Intelligent selection of work scheduling strategies based on workload
- **Fine-grained Parallelism**: Advanced workload partitioning for better load balancing

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.1.0-alpha.1"
```

For BLAS/LAPACK support, ensure you have the necessary system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# macOS
brew install openblas lapack
```

## 🔍 Implementation Details

NumRS is built on top of several battle-tested libraries:

- **ndarray**: Provides the foundation for n-dimensional arrays
- **ndarray-linalg**: Provides BLAS/LAPACK bindings for linear algebra
- **num-complex**: Complex number support for advanced operations
- **BLAS/LAPACK**: Powers high-performance linear algebra routines
- **Rayon**: Enables parallel computation capabilities
- **num-traits**: Provides generic numeric traits for numerical operations

## 📋 Current Status

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
- Fast Fourier Transform (FFT) implementation
- Polynomial operations and interpolation
- Sparse matrix representations
- Memory-mapped arrays for large datasets
- Support for datetime64 and timedelta64 data types
- Structured arrays and record arrays with custom dtypes

🚧 In Progress:
- Custom memory allocators for numerical workloads
- GPU acceleration for supported operations
- Adding more comprehensive documentation
- Building a more comprehensive test suite

## 📚 Documentation

For detailed documentation, examples, and API reference, visit [docs.rs/numrs2](https://docs.rs/numrs2).

## 🧪 Examples

Check out the `examples/` directory for more usage examples:

- `basic_usage.rs`: Core array operations and manipulations
- `linalg_example.rs`: Linear algebra operations and solvers
- `simd_example.rs`: SIMD-accelerated computations
- `memory_optimize_example.rs`: Memory layout optimization for cache efficiency
- `parallel_optimize_example.rs`: Parallelization optimization techniques
- See the [examples README](examples/README.md) for more details

## 🛠️ Development

NumRS is in active development. See [TODO.md](TODO.md) for upcoming features and development roadmap.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
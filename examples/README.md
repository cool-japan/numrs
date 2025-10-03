# NumRS Examples

This directory contains example code demonstrating various features of the NumRS library. These examples showcase how to use NumRS for different numerical computing tasks.

## Running Examples

You can run any example using Cargo:

```bash
cargo run --example basic_usage
cargo run --example linalg_example
cargo run --example simd_example
cargo run --example memory_optimize_example
cargo run --example parallel_optimize_example
cargo run --example autodiff_example
cargo run --example randomized_linalg_example

# For Arrow example, enable the arrow feature:
cargo run --example arrow_example --features arrow
```

## Example Descriptions

### Basic Usage (`basic_usage.rs`)

Demonstrates fundamental array operations:
- Array creation and reshaping
- Element-wise operations (addition, subtraction, multiplication, division)
- Matrix multiplication
- Array slicing and indexing
- Mapping operations
- Parallel operations with Rayon

### Random Number Generation (`random_distributions_example.rs`)

Demonstrates the modern random number generation API:
- Creating random number generators with different bit generators
- Thread-safe random generation with proper seeding
- Generating arrays from various probability distributions
- Working with the Generator API
- Statistical properties of distributions
- Advanced distributions for scientific computing

### Linear Algebra (`linalg_example.rs`)

Illustrates linear algebra operations:
- Matrix determinant and inverse
- Matrix-matrix and matrix-vector multiplication
- Solving linear systems (Ax = b)
- Vector norms
- Matrix decompositions (SVD, QR, LU, etc.)
- Eigenvalue and eigenvector computation

### SIMD Operations (`simd_example.rs`)

Shows SIMD-accelerated vectorized operations:
- CPU feature detection for optimal SIMD instruction selection
- Element-wise addition, multiplication, and division with SIMD
- Vectorized mathematical functions (exp, log, sqrt)
- SIMD reductions (sum, product)
- Performance comparison between scalar and SIMD implementations

### Memory Optimization (`memory_optimize_example.rs`)

Demonstrates memory layout optimization techniques:
- Cache-friendly memory layout strategies
- Data placement optimization for better cache utilization
- Memory alignment for SIMD operations
- Performance comparison for different memory layouts

### Parallel Optimization (`parallel_optimize_example.rs`)

Demonstrates parallel processing optimization techniques:
- Adaptive parallelization thresholds based on workload characteristics
- Different scheduling strategies for optimal load balancing
- Workload partitioning strategies for improved cache efficiency
- Fine-grained parallelization with performance benchmarks
- Dynamic thread count selection based on workload size and complexity

### Array Views (`views_example.rs`)

Demonstrates array views for zero-copy operations:
- Creating read-only and mutable views
- Slicing operations with views
- Strided views for non-contiguous access
- Broadcasting views
- View transformations (transpose, etc.)
- Operations on views (arithmetic, mapping)
- Lifetime semantics and memory efficiency

### Type Conversions (`type_conversion_example.rs`)

Shows type conversion capabilities:
- Converting between numeric types (astype)
- Upcasting and downcasting
- Complex number conversions
- Mixed-type operations
- Type promotion rules
- Safe conversion with error handling

### Broadcasting (`broadcasting_example.rs`)

Illustrates NumRS's broadcasting system:
- Broadcasting rules for differently shaped arrays
- Broadcasting with scalar values
- Broadcasting with mixed dimensionality
- Broadcasting with type conversion

### Array Creation (`array_creation_example.rs`)

Shows different ways to create arrays:
- From vectors and slices
- Using factory functions (zeros, ones, full)
- Using generators (arange, linspace, logspace)
- Random array creation

### Indexing (`indexing_example.rs`)

Demonstrates array indexing operations:
- Basic indexing with indices
- Slicing operations
- Boolean masking
- Fancy indexing with index arrays
- Setting values through indices

### Axis Operations (`axis_ops_example.rs`)

Shows operations along specific axes:
- Reduction operations (sum, mean, min/max)
- Cumulative operations (cumsum, cumprod)
- Statistical operations by axis (var, std)
- Concatenation and stacking along axes

### Universal Functions (`ufuncs_example.rs`)

Demonstrates universal functions (ufuncs):
- Element-wise mathematical operations
- Trigonometric functions
- Exponential and logarithmic functions
- Statistical functions

### Matrix Decompositions (`matrix_decomp_example.rs`)

Illustrates matrix decomposition techniques:
- Singular Value Decomposition (SVD)
- QR Decomposition
- LU Decomposition
- Cholesky Decomposition
- Eigendecomposition

### Polynomial Operations (`polynomial_example.rs`)

Shows polynomial functionality:
- Creating polynomials
- Polynomial arithmetic
- Evaluating polynomials
- Polynomial interpolation
- Root finding

### FFT Operations (`fft_example.rs`)

Demonstrates Fast Fourier Transform operations:
- Forward and inverse FFT
- FFT with real data
- 2D FFT
- Frequency domain operations

### Sparse Arrays (`sparse_example.rs`)

Shows sparse array functionality:
- Creating sparse arrays
- Sparse array operations
- Conversion between dense and sparse formats
- Sparse matrix multiplication

### Automatic Differentiation (`autodiff_example.rs`)

Demonstrates automatic differentiation capabilities:
- Forward mode AD with dual numbers
- Reverse mode AD with tape-based backpropagation
- Higher-order derivatives (Hessian, Taylor series)
- Gradient descent optimization
- Jacobian matrix computation
- Neural network activation functions (sigmoid, ReLU)
- Directional derivatives
- Application to numerical optimization

### Apache Arrow Integration (`arrow_example.rs`)

Shows Apache Arrow interoperability (requires `--features arrow`):
- Zero-copy conversions between NumRS and Arrow arrays
- IPC streaming for inter-process communication
- Feather format for fast columnar storage
- Reading and writing Arrow data files
- Support for multiple data types (f32, f64, i8-i64, u8-u64, bool)
- Large array performance benchmarking
- Integration with Python ecosystem (PyArrow, Pandas, Polars)

### Randomized Linear Algebra (`randomized_linalg_example.rs`)

Illustrates randomized linear algebra algorithms:
- Random projections for dimensionality reduction (Gaussian, Sparse, Rademacher)
- Randomized range finder for column space approximation
- Low-rank approximation algorithms
- Johnson-Lindenstrauss lemma application
- Performance comparisons with full algorithms
- Memory-efficient processing of large matrices

## Contributing Examples

We welcome contributions of new examples! If you have created an example that demonstrates NumRS capabilities, please consider submitting a pull request.

Guidelines for contributed examples:
- Include comprehensive comments explaining the code
- Begin with a concise description of the example's purpose
- Ensure the example runs with the current version of NumRS
- Keep the example focused on demonstrating specific features
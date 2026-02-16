# NumRS2 Examples

This directory contains comprehensive example code demonstrating the features of NumRS2. These examples showcase how to use NumRS2 for various numerical computing, machine learning, and scientific computing tasks.

## Quick Start

You can run any example using Cargo:

```bash
# Basic examples
cargo run --example basic_usage
cargo run --example array_creation_example
cargo run --example matrix_example

# Advanced examples
cargo run --example advanced_optimization
cargo run --example statistical_analysis
cargo run --example signal_processing
cargo run --example time_series_basics
cargo run --example ml_pipeline

# Feature-specific examples (requires features)
cargo run --example distributed_computing --features distributed
cargo run --example gpu_acceleration --features gpu
cargo run --example arrow_example --features arrow
```

## Example Categories

### 🎯 New Comprehensive Examples (Recommended Starting Point)

These examples provide in-depth coverage of NumRS2's advanced features:

#### Distributed Computing (`distributed_computing.rs`)
**Run:** `cargo run --example distributed_computing --features distributed`

Comprehensive distributed computing patterns:
- Distributed array operations (scatter, gather, allreduce)
- Collective communications
- Distributed linear algebra
- Error handling and fault tolerance
- Process synchronization and coordination

**Key Topics:** MPI-like operations, data distribution strategies, parallel matrix operations

#### Advanced Optimization (`advanced_optimization.rs`)
**Run:** `cargo run --example advanced_optimization`

Complete guide to all optimization algorithms:
- **Gradient-based:** BFGS, L-BFGS, Conjugate Gradient (FR, PR, HS)
- **Derivative-free:** Nelder-Mead
- **Global optimization:** PSO, Simulated Annealing, Differential Evolution, Genetic Algorithm
- **Constrained:** Projected Gradient, Penalty Method, Interior Point, SQP
- **Least-squares:** Levenberg-Marquardt, Trust Region
- Real-world applications: portfolio optimization, parameter tuning
- Performance comparisons

**Key Topics:** Optimization methods, constraint handling, algorithm selection

#### Statistical Analysis (`statistical_analysis.rs`)
**Run:** `cargo run --example statistical_analysis`

Comprehensive statistical analysis toolkit:
- Distribution fitting and parameter estimation
- Hypothesis testing (t-tests, chi-square, ANOVA)
- Confidence intervals
- Bootstrapping and resampling
- Correlation and covariance analysis
- Regression analysis and residuals
- Statistical inference

**Key Topics:** Inferential statistics, distribution theory, hypothesis testing

#### Time Series Basics (`time_series_basics.rs`)
**Run:** `cargo run --example time_series_basics`

Complete time series analysis workflow:
- Moving averages (SMA, WMA)
- Exponential smoothing (SES, Holt's method)
- Autocorrelation and partial autocorrelation
- Trend analysis and decomposition
- Seasonal decomposition
- Differencing for stationarity
- Basic forecasting methods

**Key Topics:** Time series decomposition, forecasting, stationarity

#### Signal Processing (`signal_processing.rs`)
**Run:** `cargo run --example signal_processing`

Advanced signal processing techniques:
- FFT and spectral analysis
- Window functions (Hann, Hamming, Blackman, Bartlett)
- Filtering (lowpass, highpass, bandpass)
- Convolution and correlation
- Signal generation (sine, square, sawtooth, chirp)
- Frequency domain operations
- SNR improvement

**Key Topics:** Fourier analysis, digital filtering, spectral estimation

#### Machine Learning Pipeline (`ml_pipeline.rs`)
**Run:** `cargo run --example ml_pipeline`

Complete ML workflow from data to deployment:
- Data preprocessing (normalization, standardization)
- Train/test split and stratification
- K-fold cross-validation
- Feature engineering (polynomial, interactions, binning)
- Model training and optimization
- Evaluation metrics (accuracy, precision, recall, F1)
- Classification and regression pipelines

**Key Topics:** ML workflow, preprocessing, model evaluation

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

## All Examples Summary

### Complete Example List

| Category | Example | Description | Features Required |
|----------|---------|-------------|-------------------|
| **Getting Started** | `basic_usage.rs` | Fundamental array operations | None |
| **Distributed Computing** | `distributed_computing.rs` | Comprehensive distributed computing patterns | `distributed` |
| | `distributed_basics.rs` | Basic distributed array operations | `distributed` |
| **Optimization** | `advanced_optimization.rs` | All 15+ optimization algorithms | None |
| | `scirs2_optimization.rs` | SciRS2 SIMD/parallel optimizations | `scirs` |
| **Statistics** | `statistical_analysis.rs` | Complete statistical analysis toolkit | None |
| | `statistics_example.rs` | Basic statistical functions | None |
| **Time Series** | `time_series_basics.rs` | Time series analysis and forecasting | None |
| **Signal Processing** | `signal_processing.rs` | Advanced signal processing | None |
| | `fft_example.rs` | Basic FFT operations | None |
| **Machine Learning** | `ml_pipeline.rs` | Complete ML pipeline | None |
| | `machine_learning_example.rs` | ML algorithms from scratch | None |
| | `neural_network_basics.rs` | Neural network operations | None |
| **GPU Computing** | `gpu_acceleration.rs` | GPU-accelerated operations | `gpu` |
| | `gpu_example.rs` | Basic GPU operations | `gpu` |
| | `gpu_benchmark.rs` | GPU performance benchmarks | `gpu` |
| **Visualization** | `visualization.rs` | Plotting and visualization | `visualization` |
| **Symbolic Math** | `symbolic_math.rs` | Symbolic computation | None |
| **Linear Algebra** | `matrix_example.rs` | Matrix operations | None |
| | `matrix_decomp_example.rs` | Matrix decompositions | None |
| | `randomized_linalg_example.rs` | Randomized linear algebra | None |
| **Array Operations** | `array_creation_example.rs` | Array creation methods | None |
| | `array_manipulation_example.rs` | Array manipulation | None |
| | `broadcasting_example.rs` | Broadcasting rules | None |
| | `indexing_example.rs` | Array indexing | None |
| | `views_example.rs` | Array views and slicing | None |
| **Performance** | `simd_example.rs` | SIMD operations | None |
| | `parallel_optimize_example.rs` | Parallel processing | None |
| | `memory_optimize_example.rs` | Memory optimization | None |
| | `performance_benchmark.rs` | Performance benchmarks | None |
| **I/O** | `arrow_example.rs` | Apache Arrow integration | `arrow` |
| | `npy_npz_example.rs` | NumPy file format | None |
| | `mmap_example.rs` | Memory-mapped I/O | None |
| **Advanced** | `autodiff_example.rs` | Automatic differentiation | None |
| | `sparse_example.rs` | Sparse arrays | None |
| | `masked_array_example.rs` | Masked arrays | None |
| | `polynomial_example.rs` | Polynomial operations | None |
| | `special_functions_example.rs` | Special functions | None |

## Building and Running

### Build All Examples

```bash
# Build all examples (default features)
cargo build --examples

# Build with all features
cargo build --examples --all-features

# Build specific feature sets
cargo build --examples --features distributed
cargo build --examples --features gpu
cargo build --examples --features visualization
```

### Run Examples

```bash
# Run without features
cargo run --example <example_name>

# Run with features
cargo run --example distributed_computing --features distributed
cargo run --example gpu_acceleration --features gpu
cargo run --example visualization --features visualization
```

### Verify All Examples Compile

```bash
# Check that all examples compile
cargo check --examples

# Check with all features
cargo check --examples --all-features
```

## Learning Paths

### For Beginners
1. `basic_usage.rs` - Start here
2. `array_creation_example.rs` - Learn array creation
3. `array_manipulation_example.rs` - Array operations
4. `statistics_example.rs` - Basic statistics

### For Data Scientists
1. `statistical_analysis.rs` - Statistical methods
2. `time_series_basics.rs` - Time series analysis
3. `ml_pipeline.rs` - Machine learning workflow
4. `visualization.rs` - Data visualization

### For ML Engineers
1. `ml_pipeline.rs` - Complete ML pipeline
2. `neural_network_basics.rs` - Neural networks
3. `advanced_optimization.rs` - Optimization algorithms
4. `gpu_acceleration.rs` - GPU acceleration

### For Signal Processing
1. `signal_processing.rs` - Comprehensive signal processing
2. `fft_example.rs` - Fourier transforms
3. `time_series_basics.rs` - Time series methods

### For High-Performance Computing
1. `distributed_computing.rs` - Distributed computing
2. `parallel_optimize_example.rs` - Parallel processing
3. `simd_example.rs` - SIMD optimization
4. `gpu_acceleration.rs` - GPU computing

## Contributing Examples

We welcome contributions of new examples! If you have created an example that demonstrates NumRS2 capabilities, please consider submitting a pull request.

Guidelines for contributed examples:
- Include comprehensive comments explaining the code
- Begin with a concise description of the example's purpose
- Ensure the example runs with the current version of NumRS2
- Keep examples under 500 lines when possible
- Follow SciRS2 ecosystem policies (use scirs2-core abstractions)
- Add proper error handling (no unwrap in production code)
- Include expected output in comments
- Add feature gates where needed (e.g., `#[cfg(feature = "distributed")]`)
# NumRS User Guide

This guide provides a detailed overview of NumRS functionalities, with examples and best practices.

## Table of Contents

1. [Basic Array Operations](#basic-array-operations)
2. [Mathematical Functions](#mathematical-functions)
3. [Linear Algebra](#linear-algebra)
4. [Random Number Generation](#random-number-generation)
5. [Statistics](#statistics)
6. [Matrix Operations](#matrix-operations)
7. [Fast Fourier Transform](#fast-fourier-transform)
8. [Polynomials](#polynomials)
9. [Special Functions](#special-functions)
10. [Performance Optimization](#performance-optimization)

## Basic Array Operations

NumRS provides a central `Array` class for n-dimensional array operations, similar to NumPy's ndarray.

```rust
use numrs2::prelude::*;

// Create arrays
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::zeros(&[2, 2]);
let c = Array::ones(&[2, 2]);
let d = Array::eye(3);  // 3x3 identity matrix

// Access elements
let value = a.get(&[0, 1])?;  // Get element at row 0, column 1

// Shape manipulation
let flattened = a.flatten(None);
let transposed = a.transpose();
let stacked = Array::stack(&[&a, &c], 0)?;  // Stack along axis 0

// Element-wise operations
let sum = a.add(&c);
let product = a.multiply(&c);

// Broadcasting
let broadcasted = a.add_broadcast(&Array::from_vec(vec![10.0, 20.0]))?;
```

For more detailed examples, see `examples/basic_usage.rs` and `examples/array_manipulation_example.rs`.

## Mathematical Functions

NumRS includes a wide range of mathematical functions that operate element-wise on arrays.

```rust
use numrs2::prelude::*;

let a = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

// Trigonometric functions
let sin_values = a.sin();
let cos_values = a.cos();

// Exponential and logarithmic functions
let exp_values = a.exp();
let log_values = a.log10();

// Power functions
let squared = a.powf(2.0);
let sqrt_values = a.sqrt();

// Rounding functions
let rounded = a.round();
let ceiled = a.ceil();
let floored = a.floor();
```

## Linear Algebra

NumRS offers powerful linear algebra capabilities through integration with BLAS and LAPACK.

```rust
use numrs2::prelude::*;

// Create matrices
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

// Matrix multiplication
let product = a.matmul(&b)?;  // Matrix product

// Decompositions
let (u, s, vt) = a.svd()?;  // Singular Value Decomposition
let (q, r) = a.qr()?;       // QR Decomposition
let (l, u, p) = a.lu()?;    // LU Decomposition

// Solve linear systems
let x = Array::from_vec(vec![1.0, 2.0]).reshape(&[2, 1]);
let solution = a.solve(&x)?;  // Solve Ax = b

// Eigenvalues and eigenvectors
let (eigenvalues, eigenvectors) = a.eig()?;

// Matrix inverse
let inv_a = a.inv()?;

// Matrix determinant
let det_a = a.det()?;
```

For detailed examples, see `examples/linalg_example.rs`, `examples/matrix_decomp_example.rs`, and the [Linear Algebra Module Documentation](examples/README_LINALG.md).

## Random Number Generation

NumRS provides a modern interface for random number generation with various probability distributions.

```rust
use numrs2::prelude::*;
use numrs2::random::{default_rng, pcg64_rng};

// Create generators
let rng = default_rng();  // Standard generator
let pcg = pcg64_rng();    // PCG64 generator (high quality)

// Generate uniform random numbers
let uniform = rng.random::<f64>(&[3, 3])?;

// Generate from distributions
let normal = rng.normal(0.0, 1.0, &[5])?;
let beta = rng.beta(2.0, 5.0, &[5])?;
let gamma = rng.gamma(1.0, 2.0, &[5])?;
let poisson = rng.poisson::<u32>(5.0, &[5])?;

// Seeded for reproducibility
let seeded_rng = numrs2::random::seed_rng(42);
let samples = seeded_rng.normal(0.0, 1.0, &[3])?;
```

For comprehensive examples, see `examples/random_distributions_example.rs` and the [Random Module Documentation](examples/README_RANDOM.md).

## Statistics

NumRS offers various statistical functions for data analysis.

```rust
use numrs2::prelude::*;
use numrs2::stats::{cov, corrcoef, histogram, percentile, bincount};

// Create sample data
let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Basic statistics
let mean = data.mean();       // 3.0
let variance = data.var();    // 2.0
let std_dev = data.std();     // 1.41...

// Percentiles
let p_points = Array::from_vec(vec![0.0, 25.0, 50.0, 75.0, 100.0]);
let p_values = percentile(&data, &p_points, Some("linear"))?;

// Correlation and covariance
let data2 = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
let correlation = corrcoef(&data, &data2)?;  // -1.0 (perfect negative correlation)

// Histograms
let (hist, bin_edges) = histogram(&data, 5, None, None)?;
```

For detailed examples, see `examples/statistics_example.rs` and the [Statistics Module Documentation](examples/README_STATISTICS.md).

## Matrix Operations

NumRS provides specialized matrix operations through the matrix module.

```rust
use numrs2::prelude::*;
use numrs2::matrix::{create_matrix, solve, condition_number};

// Create matrices
let a_mat = create_matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2)?;
let b_vec = create_matrix(vec![5.0, 6.0], 2, 1)?;

// Solve a linear system
let x = solve(&a_mat, &b_vec)?;

// Check condition number (numerical stability)
let cond = condition_number(&a_mat)?;

// Create special matrices
let hilbert = create_hilbert_matrix(5)?;
let vandermonde = create_vandermonde_matrix(&[1.0, 2.0, 3.0, 4.0], 3)?;
```

For more examples, see `examples/matrix_example.rs`.

## Fast Fourier Transform

NumRS includes FFT functionality for signal processing.

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, ifft, fft2, ifft2, fftshift, ifftshift};

// Create a signal
let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

// Compute FFT
let spectrum = fft(&signal)?;
let magnitude = spectrum.abs();
let phase = spectrum.angle();

// Shift zero-frequency component to center
let shifted = fftshift(&spectrum)?;

// Compute inverse FFT to recover signal
let recovered = ifft(&spectrum)?;
```

For detailed examples, see `examples/fft_example.rs` and the [FFT Module Documentation](examples/README_FFT.md).

## Polynomials

NumRS provides polynomial functionality for fitting, evaluation, and root finding.

```rust
use numrs2::prelude::*;
use numrs2::polynomial::{polyfit, polyval, roots};

// Create data for fitting
let x = Array::linspace(0.0, 10.0, 20)?;
let y = x.power(2.0).multiply_scalar(2.0).add_scalar(3.0);  // 2x² + 3

// Fit polynomial to data
let coeffs = polyfit(&x, &y, 2)?;  // Expect [3, 0, 2] approximately

// Evaluate polynomial
let x_new = Array::linspace(0.0, 5.0, 10)?;
let y_pred = polyval(&coeffs, &x_new)?;

// Find roots of polynomial
let poly = Array::from_vec(vec![1.0, -3.0, 2.0]);  // x² - 3x + 2 = 0
let r = roots(&poly)?;  // Expect [1, 2]
```

For more examples, see `examples/polynomial_example.rs` and the [Polynomial Module Documentation](examples/README_POLYNOMIAL.md).

## Special Functions

NumRS implements various special mathematical functions.

```rust
use numrs2::prelude::*;
use numrs2::special::{erf, gamma, bessel_j0, bessel_j1};

// Create an array
let x = Array::linspace(-3.0, 3.0, 7)?;

// Apply special functions
let erf_vals = erf(&x);                 // Error function
let gamma_vals = gamma(&x.add_scalar(5.0));  // Gamma function (avoiding negative values)
let bessel_j0_vals = bessel_j0(&x);     // Bessel function of first kind, order 0
let bessel_j1_vals = bessel_j1(&x);     // Bessel function of first kind, order 1
```

For detailed examples, see `examples/special_functions_example.rs`.

## Performance Optimization

NumRS includes various performance optimization techniques.

### SIMD Acceleration

```rust
use numrs2::prelude::*;
use numrs2::simd_optimize::apply_simd;

// Create a large array
let data = Array::linspace(0.0, 100.0, 10000)?;

// Apply SIMD-accelerated function
let result = apply_simd(&data, |x| x * x + 2.0 * x + 1.0)?;
```

### Memory Optimization

```rust
use numrs2::prelude::*;
use numrs2::memory_optimize::{optimize_layout, align_data};

// Create a large array
let data = Array::zeros(&[1000, 1000]);

// Optimize memory layout for cache efficiency
let optimized = optimize_layout(&data, CacheLevel::L2)?;

// Ensure data is aligned for SIMD operations
let aligned = align_data(&data, 32)?;  // 32-byte alignment
```

### Parallel Processing

```rust
use numrs2::prelude::*;
use numrs2::parallel_optimize::{parallel_apply, adaptive_parallel};

// Create a large array
let data = Array::zeros(&[10000, 100]);

// Apply function in parallel
let result = parallel_apply(&data, |x| x.powf(2.0), None)?;

// Use adaptive parallelization based on workload
let result = adaptive_parallel(&data, |x| {
    let mut v = x.clone();
    for _ in 0..1000 {
        v = v.sin().cos();  // Expensive operation
    }
    v
}, None)?;
```

For more examples, see `examples/simd_example.rs`, `examples/memory_optimize_example.rs`, and `examples/parallel_optimize_example.rs`.

## Additional Resources

- [API Documentation](https://docs.rs/numrs2)
- [GitHub Repository](https://github.com/cool-japan/numrs)
- [Example Code Directory](examples/)
- [Random Module Guide](examples/README_RANDOM.md)
- [Statistics Module Guide](examples/README_STATISTICS.md)

## License

NumRS is licensed under the Apache License 2.0.
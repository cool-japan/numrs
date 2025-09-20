# Getting Started with NumRS2

This guide offers a quick introduction to NumRS2 for beginners. If you're familiar with NumPy, you'll find the API and concepts very similar, but with Rust's type safety and performance advantages.

## Installation

Add NumRS2 to your Rust project by adding this to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.1.0-beta.1"
```

For BLAS/LAPACK support, ensure you have the necessary system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# macOS
brew install openblas lapack
```

## Importing NumRS2

The easiest way to import NumRS2 is to use the prelude module which includes all the essential components:

```rust
use numrs2::prelude::*;
```

Or you can import specific modules:

```rust
use numrs2::array::Array;
use numrs2::random;
use numrs2::linalg;
```

## Core Concepts

### The Array Object

The central object of NumRS2 is the `Array` class, a powerful N-dimensional array with broadcasting capabilities. It supports a wide range of operations and data types.

```rust
// Creating arrays
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::zeros(&[2, 2]);  // Array of zeros
let c = Array::ones(&[2, 2]);   // Array of ones
let d = Array::eye(3);          // 3x3 identity matrix

// Array info
println!("Shape: {:?}", a.shape());
println!("Size: {}", a.size());
println!("Dimension: {}", a.ndim());

// Array operations
let result = a.add(&b);         // Element-wise addition
let result = a.matmul(&b)?;     // Matrix multiplication
```

### Data Types

NumRS2 supports various data types including:
- Floating point: `f32`, `f64`
- Integers: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Complex numbers
- Boolean

### Basic Operations

NumRS2 supports a wide range of operations:

```rust
// Element-wise operations
let sum = a.add(&b);
let diff = a.subtract(&b);
let product = a.multiply(&b);
let quotient = a.divide(&b);

// Mathematical functions
let sin_values = a.sin();
let log_values = a.log10();
let sqrt_values = a.sqrt();
let abs_values = a.abs();

// Statistics
let mean = a.mean()?;
let std_dev = a.std()?;
let max = a.max()?;
let min = a.min()?;
```

### Broadcasting

NumRS2 implements NumPy-style broadcasting for operating on arrays of different shapes:

```rust
// Broadcasting example: Adding a scalar to each element
let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
let b = Array::from_vec(vec![10.0]);

let result = a.add_broadcast(&b)?;  // Results in [11.0, 12.0, 13.0]
```

### Random Number Generation

NumRS2 provides a modern random number generation API:

```rust
use numrs2::random::{default_rng, PCG64BitGenerator};

// Create a generator with default bit generator
let rng = default_rng();

// Generate arrays from various distributions
let uniform = rng.random::<f64>(&[5])?;
let normal = rng.normal(0.0, 1.0, &[5])?;
let poisson = rng.poisson::<u32>(5.0, &[5])?;

// Specify a particular bit generator
let pcg = numrs2::random::Generator::new(PCG64BitGenerator::new(Some(42)));
let samples = pcg.random::<f64>(&[5])?;
```

### Linear Algebra

NumRS2 provides comprehensive linear algebra operations:

```rust
// Matrix operations
let inverse = a.inv()?;
let determinant = a.det()?;

// Decompositions
let (u, s, vt) = a.svd()?;
let (q, r) = a.qr()?;

// Solving linear equations
let solution = a.solve(&b)?;
```

## Next Steps

For more detailed information, refer to:
- [User Guide](GUIDE.md) - Comprehensive guide to all NumRS features
- [Examples](examples/) - Learn through example code
- [API Reference](https://docs.rs/numrs2) - Complete API documentation
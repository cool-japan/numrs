# NumRS Random Module

The NumRS random module provides a NumPy-like interface for generating random values from various probability distributions. The module has been designed to be both easy to use for simple cases while also supporting advanced use cases like custom random number generators.

## Key Features

- Modern Generator class interface similar to NumPy
- Multiple bit generator backends (Standard and PCG64)
- Support for common probability distributions
- Advanced distributions not available in standard Rust crates
- Thread-safe design with Arc<Mutex<>> wrapped BitGenerators
- Reproducible sequences with seed control

## Basic Usage

### Modern Interface (Recommended)

```rust
use numrs2::prelude::*;
use numrs2::random::default_rng;

// Create a generator with the default bit generator
let rng = default_rng();

// Generate random values from various distributions
let uniform_arr = rng.random::<f64>(&[3, 3])?;
let normal_arr = rng.normal(0.0, 1.0, &[5])?;
let beta_arr = rng.beta(2.0, 5.0, &[5])?;
```

### Create Seeded Generator

```rust
use numrs2::random::seed_rng;
// Or for modern interface
use numrs2::random::pcg64_seed_rng;

// Create generators with specific seeds for reproducibility
let seeded_rng = seed_rng(12345);
let seeded_pcg_rng = pcg64_seed_rng(67890);
```

### Legacy Interface

```rust
use numrs2::random::{normal, beta, gamma, set_seed};

// Set the global seed for reproducibility (affects all global functions)
set_seed(12345);

// Use global functions to generate random arrays
let normal_samples = normal(0.0, 1.0, &[5])?;
let beta_samples = beta(2.0, 5.0, &[5])?;
```

## Available Distributions

### Common Distributions

- Uniform: `uniform()`, or `random()` for [0,1)
- Normal: `normal()`, `standard_normal()`
- Beta: `beta()`
- Gamma: `gamma()`
- Binomial: `binomial()`
- Poisson: `poisson()`
- Exponential: `exponential()`
- Chi-square: `chisquare()`
- Lognormal: `lognormal()`
- Student's t: `student_t()`
- Cauchy: `cauchy()`
- Weibull: `weibull()`

### Advanced Distributions

- Noncentral Chi-square: `noncentral_chisquare()`
- Noncentral F: `noncentral_f()`
- Von Mises: `vonmises()`
- Maxwell: `maxwell()`
- Wald (Inverse Gaussian): `wald()`
- Triangular: `triangular()`
- PERT: `pert()`
- Dirichlet: `dirichlet()`
- Multinomial: `multinomial()`
- Laplace: `laplace()`
- Gumbel: `gumbel()`
- Logistic: `logistic()`
- Pareto: `pareto()`
- Rayleigh: `rayleigh()`
- Hypergeometric: `hypergeometric()`
- Zipf: `zipf()`
- Logseries: `logseries()`

## Custom Bit Generators

You can directly use BitGenerator implementations for low-level access:

```rust
use numrs2::random::{PCG64BitGenerator, StdBitGenerator};

// Create bit generators with a specific seed
let mut std_gen = StdBitGenerator::new(42);
let mut pcg_gen = PCG64BitGenerator::new(42);

// Generate raw random values
let u64_val = pcg_gen.next_u64();
let u32_val = pcg_gen.next_u32();
let f64_val = pcg_gen.next_f64();
```

## Implementing Your Own Bit Generator

You can implement the BitGenerator trait to create your own random number generator:

```rust
use numrs2::random::{BitGenerator, Generator};

struct MyBitGenerator {
    // Your state here
}

impl BitGenerator for MyBitGenerator {
    fn next_u64(&mut self) -> u64 {
        // Your implementation here
    }
    
    fn next_u32(&mut self) -> u32 {
        // Your implementation here
    }
    
    fn next_f64(&mut self) -> f64 {
        // Your implementation here
    }
    
    fn seed(&mut self, seed: u64) {
        // Your implementation here
    }
}

// Use your custom bit generator
let my_gen = MyBitGenerator { /* ... */ };
let rng = Generator::new(my_gen);
```

## Examples

For detailed examples, see:
- `random_distributions_example.rs`: Comprehensive demonstration of all distributions
- `random_simple_test.rs`: Simple test for quick verification

## Performance Notes

- The PCG64BitGenerator generally provides better statistical quality and is recommended for scientific applications
- For cryptography, use a proper cryptographic RNG instead
- BitGenerators are wrapped in Arc<Mutex<>> to allow thread-safe usage, but this adds some overhead
- For high-performance needs with large arrays, consider using parallelization with multiple generators
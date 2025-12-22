# NumRS2 v0.1.0-rc.3 Release Notes

**Release Date:** December 2025
**Release Candidate 3** - Production-Ready NumPy + SciPy Implementation in Rust with Expression Templates

## Overview

NumRS2 v0.1.0-rc.3 marks a major milestone: **production-ready status** with comprehensive NumPy and SciPy compatibility. This release delivers a complete numerical computing library in Rust with 11 SciPy-equivalent modules, SIMD-optimized operations, expression templates for lazy evaluation, and seamless integration with the SciRS2 ecosystem.

## Release Highlights

### Production-Ready Status

- **119,358 lines** of production Rust code (298 files)
- **647 tests** passing (unit + doc tests)
- **Zero warnings** in compilation
- **$4.2M COCOMO** estimated development value

### Complete NumPy Compatibility (100%)

All core NumPy functionality is implemented:

- **Array Operations**: Creation, manipulation, indexing, broadcasting, views
- **Mathematical Functions**: 200+ ufuncs with SIMD optimization
- **Linear Algebra**: Complete `np.linalg` + BLAS/LAPACK integration
- **Statistical Functions**: Full scipy.stats equivalents via scirs2-stats
- **Random Generation**: Advanced distributions via scirs2-core
- **I/O Operations**: NPY/NPZ, CSV, text formats, memory mapping

### 11 SciPy-Equivalent Modules

| Module | Description | Functions |
|--------|-------------|-----------|
| `scipy.optimize` | Numerical optimization | BFGS, L-BFGS, Nelder-Mead, Trust Region, Levenberg-Marquardt |
| `scipy.optimize.root` | Root-finding | Bisection, Brent, Ridder, Newton-Raphson, Secant, Halley |
| `scipy.misc.derivative` | Numerical differentiation | Gradient, Jacobian, Hessian with Richardson extrapolation |
| `scipy.interpolate` | Interpolation | Linear, Cubic, Spline variants (Natural, Clamped, Not-a-Knot, Periodic) |
| `scipy.spatial.distance` | Distance metrics | Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Correlation, Hamming |
| `scipy.cluster` | Clustering | K-means++, Hierarchical (single/complete/average/ward linkage) |
| `scipy.ndimage` | Image processing | Filters, morphology, measurements, segmentation |
| `scipy.spatial` | Computational geometry | KD-tree, Convex hull, Voronoi, Delaunay triangulation |
| `scipy.special` | Special functions | 50+ functions (Gamma, Bessel, Error functions, Elliptic integrals) |
| `scipy.fft` | Fast Fourier Transform | FFT, RFFT, DCT, DST, STFT, GPU acceleration, plan caching |
| `scipy.signal` | Signal processing | Digital filters, wavelets, convolution, spectral analysis |

### SIMD Performance Optimization

- **86 AVX2-optimized functions** with automatic threshold-based dispatch
- **42 ARM NEON f64 vectorized operations** for Apple Silicon
- **4-way loop unrolling** and FMA (fused multiply-add) instructions
- **Runtime CPU feature detection** for optimal dispatch
- **2.4-3.6x performance improvement** on vectorized operations

### Advanced Linear Algebra

- **Iterative Solvers**: CG, PCG, GMRES, FGMRES, BiCGSTAB, MINRES
- **Preconditioners**: Jacobi, SSOR, Incomplete Cholesky, ILU
- **Sparse Matrices**: COO, CSR, CSC, DIA formats with optimized operations
- **Randomized Algorithms**: Randomized SVD, random projections, range finders
- **Tensor Decompositions**: Tucker (HOSVD), CP/PARAFAC with ALS

### Automatic Differentiation

- **Forward Mode**: Dual numbers for Jacobian-vector products
- **Reverse Mode**: Tape-based backpropagation for gradients
- **Higher-Order**: Hessian, Taylor series, nth-order derivatives
- **1,178 lines** of production AD code

### Data Interoperability

- **Apache Arrow**: Zero-copy exchange with Python ecosystem
- **Feather Format**: Fast columnar storage
- **IPC Streaming**: Inter-process communication
- **Python Bindings**: PyO3 integration ready for maturin build

## What's New in RC.3

### New Features

1. **Expression Templates System** - Complete lazy evaluation infrastructure
   - **SharedArray<T>**: Reference-counted arrays with ArcArray storage for O(1) cloning
   - **Operator Overloading**: Natural syntax for array operations (+, -, *, /, scalar ops)
   - **SharedExpr**: Lifetime-free expression templates using Arc for zero-lifetime DAG construction
   - **CachedExpr**: Common Subexpression Elimination (CSE) with automatic result caching
   - **CSEOptimizer**: Automatic detection and elimination of repeated computations

2. **Memory Access Pattern Optimization** - Cache-aware iteration strategies
   - **BlockedIterator**: Cache-efficient blocked iteration for 1D arrays
   - **TiledIterator2D**: 2D cache-blocking for matrix operations
   - **StrideOptimizer**: Analyzes memory layout and suggests optimal iteration order
   - Cache-aware operations for improved data locality

3. **Enhanced FFT Module** (scirs2-fft integration)
   - DCT/DST Types I-IV for compression standards
   - Advanced transforms: FrFT, NUFFT, FHT
   - STFT and spectrograms for time-frequency analysis
   - FFT plan caching for repeated transforms

2. **Enhanced Signal Processing** (scirs2-signal integration)
   - Digital filters: Butterworth, Chebyshev, Elliptic, Bessel, FIR
   - Wavelet transforms: DWT, CWT, 2D wavelets
   - LTI systems: Transfer functions, step/impulse response
   - SIMD-accelerated convolution

3. **Algorithm Improvements**
   - Fixed Ridder's Method formula (sign calculation)
   - Fixed Richardson Extrapolation step size shadowing
   - Fixed MINRES Givens rotation tracking
   - Fixed parallel scheduler contention (1,143x speedup)

4. **ARM NEON Support** (2,119 lines)
   - Complete f64 vectorized operations
   - NEON SIMD dispatch in ufuncs
   - Apple Silicon optimization

### Performance Improvements

- Parallel scheduler deadlock fix with 1,143x speedup on multi-core systems
- FFT plan caching eliminates repeated planning overhead
- SIMD threshold tuning for optimal scalar/vector dispatch

## Breaking Changes

None. This release maintains full API compatibility with beta.3.

## Dependencies

NumRS2 uses the SciRS2 ecosystem (v0.1.0-rc.4):

```toml
[dependencies]
scirs2-core = "0.1.0-rc.4"
scirs2-stats = "0.1.0-rc.4"
scirs2-linalg = "0.1.0-rc.4"
scirs2-ndimage = "0.1.0-rc.4"
scirs2-spatial = "0.1.0-rc.4"
scirs2-special = "0.1.0-rc.4"
scirs2-fft = "0.1.0-rc.4"
scirs2-signal = "0.1.0-rc.4"
```

## Installation

```toml
[dependencies]
numrs2 = "0.1.0-rc.3"
```

With optional features:

```toml
# Apache Arrow integration
numrs2 = { version = "0.1.0-rc.3", features = ["arrow"] }

# Python bindings
numrs2 = { version = "0.1.0-rc.3", features = ["python"] }

# LAPACK support
numrs2 = { version = "0.1.0-rc.3", features = ["lapack"] }

# GPU acceleration
numrs2 = { version = "0.1.0-rc.3", features = ["gpu"] }
```

## Quick Start

```rust
use numrs2::prelude::*;

fn main() -> Result<()> {
    // Create arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    // Matrix operations
    let c = a.matmul(&b)?;
    println!("Matrix multiplication: {}", c);

    // Linear algebra
    let (u, s, vt) = a.svd_compute()?;
    println!("SVD: U={}, S={}, Vt={}", u, s, vt);

    // Statistical operations
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Mean: {}, Std: {}", data.mean()?, data.std()?);

    // FFT
    let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let spectrum = signal.fft()?;
    println!("FFT: {}", spectrum);

    Ok(())
}
```

## Resources

- **Documentation**: https://docs.rs/numrs2
- **Repository**: https://github.com/cool-japan/numrs
- **Examples**: See `examples/` directory
- **SciRS2 Ecosystem**: https://github.com/cool-japan/scirs

## Acknowledgments

NumRS2 is part of the SciRS2 ecosystem, bringing production-ready scientific computing to Rust with the safety and performance guarantees that Rust provides.

---

*For the complete changelog, see the git history.*

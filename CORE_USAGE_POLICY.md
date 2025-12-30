# Core Usage Policy for NumRS2

This document outlines the mandatory policies for developing and maintaining the NumRS2 codebase. All contributors MUST adhere to these guidelines to ensure consistency, maintainability, and optimal performance.

## Table of Contents

1. [Overview](#overview)
2. [SIMD Operations Policy](#simd-operations-policy)
3. [Parallel Processing Policy](#parallel-processing-policy)
4. [SciRS2 Integration Policy](#scirs2-integration-policy)
5. [BLAS/LAPACK Operations Policy](#blaslapack-operations-policy)
6. [Platform Detection Policy](#platform-detection-policy)
7. [Performance Optimization Policy](#performance-optimization-policy)
8. [Error Handling Policy](#error-handling-policy)
9. [Memory Management Policy](#memory-management-policy)
10. [Testing Policy](#testing-policy)
11. [Examples](#examples)

## Overview

NumRS2 is a Rust implementation inspired by NumPy for numerical computing. The project aims to provide:

- **Performance**: Leveraging Rust's zero-cost abstractions and SIMD optimizations
- **Safety**: Memory safety and thread safety guaranteed by Rust
- **Compatibility**: Interoperability with NumPy data formats and optional SciRS2 integration
- **Flexibility**: Modular design with optional features for different use cases

## SIMD Operations Policy

### Current Implementation

NumRS2 maintains its own SIMD abstractions in the `simd` module using the `simba` crate. This provides portability across different architectures.

### Mandatory Rules

1. **Use the `SimdOps` trait** for all SIMD-accelerated operations
2. **Always provide scalar fallbacks** for all SIMD operations
3. **Use `simd_optimize` module** for automatic SIMD implementation selection
4. **Benchmark SIMD implementations** to ensure they provide actual speedup

### SIMD Implementation Pattern

```rust
use crate::simd::SimdOps;
use crate::simd_optimize::{detect_cpu_features, select_simd_implementation};

// CORRECT - Uses NumRS2's SIMD abstractions
impl SimdOps<f32> for Array<f32> {
    fn simd_add(&self, other: &Array<f32>) -> Result<Array<f32>> {
        // Implementation using simba abstractions
    }
}

// INCORRECT - Direct SIMD implementation
// use std::arch::x86_64::*;  // Avoid direct arch-specific code
```

### Available SIMD Operations

The `SimdOps` trait provides:
- `simd_map` - Unary operations with SIMD
- `simd_zip_with` - Binary operations with SIMD
- `simd_reduce` - Reduction operations
- `simd_add`, `simd_mul` - Element-wise operations
- `simd_dot` - Dot product
- `simd_sum` - Sum reduction
- `simd_mean` - Mean calculation
- `simd_var`, `simd_std` - Statistical operations

## Parallel Processing Policy

### Current Implementation

NumRS2 uses Rayon directly for parallel operations, managed through the `parallel` and `parallel_optimize` modules.

### Mandatory Rules

1. **Use adaptive thresholds** from `parallel_optimize::threshold`
2. **Respect the `NUMRS2_NUM_THREADS` environment variable**
3. **Profile parallel operations** to ensure they provide speedup
4. **Provide sequential fallbacks** for small problem sizes

### Parallel Processing Pattern

```rust
use crate::parallel_optimize::threshold::DynamicThreshold;
use rayon::prelude::*;

// CORRECT - Uses adaptive thresholds
pub fn parallel_operation<T>(data: &[T]) -> Vec<T> 
where T: Send + Sync 
{
    let threshold = DynamicThreshold::new();
    
    if data.len() < threshold.get_threshold() {
        // Sequential processing for small data
        sequential_operation(data)
    } else {
        // Parallel processing for large data
        data.par_iter()
            .map(|x| process(x))
            .collect()
    }
}
```

### Workload Scheduling

Use the `parallel_optimize::scheduling` module for optimal work distribution:

```rust
use crate::parallel_optimize::scheduling::{AdaptiveScheduler, ChunkStrategy};

let scheduler = AdaptiveScheduler::new();
let chunks = scheduler.partition_workload(data.len(), ChunkStrategy::Dynamic);
```

## SciRS2 Integration Policy

### Integration Approach

SciRS2 is an optional dependency that provides additional scientific computing capabilities. When available, NumRS2 should leverage SciRS2's optimized implementations while maintaining its own functionality.

### Mandatory Rules

1. **SciRS2 features are always optional** - NumRS2 must work without SciRS2
2. **Use feature gates** for all SciRS2-dependent code
3. **Provide fallback implementations** when SciRS2 is not available
4. **Follow SciRS2's Core Usage Policy** when using SciRS2 features

### Integration Pattern

```rust
#[cfg(feature = "scirs")]
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn compute_statistic<T>(data: &Array<T>) -> T {
    #[cfg(feature = "scirs")]
    {
        // Use SciRS2's optimized implementation when available
        T::simd_mean(&data.view())
    }
    
    #[cfg(not(feature = "scirs"))]
    {
        // Use NumRS2's own implementation
        data.mean()
    }
}
```

### When to Use SciRS2

Prefer SciRS2 implementations for:
- Advanced statistical distributions
- Specialized scientific functions
- GPU-accelerated operations (when SciRS2 GPU support is available)

## BLAS/LAPACK Operations Policy

### Current Implementation

NumRS2 uses `ndarray-linalg` for BLAS/LAPACK operations, which is optional via the `lapack` feature.

### Mandatory Rules

1. **LAPACK features must be optional** via feature flags
2. **Provide pure-Rust fallbacks** for essential operations
3. **Document LAPACK requirements** clearly in function documentation
4. **Handle missing LAPACK gracefully** with informative error messages

### BLAS/LAPACK Pattern

```rust
#[cfg(feature = "lapack")]
pub fn solve_linear_system<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float + ndarray_linalg::Lapack
{
    // LAPACK-based implementation
    use ndarray_linalg::Solve;
    // ...
}

#[cfg(not(feature = "lapack"))]
pub fn solve_linear_system<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float
{
    // Pure-Rust fallback or error
    Err(NumRs2Error::FeatureNotEnabled("lapack".to_string()))
}
```

## Platform Detection Policy

### Mandatory Rules

1. **Use `simd_optimize::detect_cpu_features()`** for CPU feature detection
2. **Cache platform capabilities** to avoid repeated detection
3. **Provide runtime feature detection** not just compile-time
4. **Support feature override** via environment variables

### Platform Detection Pattern

```rust
use crate::simd_optimize::{detect_cpu_features, CpuFeatures};

lazy_static! {
    static ref CPU_FEATURES: CpuFeatures = detect_cpu_features();
}

pub fn select_implementation() {
    if CPU_FEATURES.has_avx2 {
        // Use AVX2 implementation
    } else if CPU_FEATURES.has_sse {
        // Use SSE implementation
    } else {
        // Use scalar implementation
    }
}
```

## Performance Optimization Policy

### Optimization Strategy

1. **Profile first** - Use benchmarks to identify bottlenecks
2. **Optimize algorithms before low-level code** 
3. **Use adaptive algorithms** that select implementations based on input size
4. **Cache-friendly algorithms** - Consider memory layout and access patterns

### Memory Layout Optimization

```rust
use crate::memory_optimize::{optimize_layout, CacheOptimizer};

// CORRECT - Uses cache-friendly layout
let optimizer = CacheOptimizer::new();
let optimized_array = optimizer.optimize_for_operation(array, Operation::MatMul);
```

## Error Handling Policy

### Mandatory Rules

1. **Use `NumRs2Error` for all errors** in the public API
2. **Provide context** in error messages
3. **Use `Result<T>` for fallible operations**
4. **Never panic in library code** (except for programmer errors)

### Error Pattern

```rust
use crate::error::{NumRs2Error, Result};

pub fn operation(input: &Array<f64>) -> Result<Array<f64>> {
    // Validate input
    if input.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "Cannot operate on empty array".to_string()
        ));
    }
    
    // Perform operation with proper error propagation
    let result = risky_operation(input)?;
    
    Ok(result)
}
```

## Memory Management Policy

### Memory Efficiency Rules

1. **Use views instead of copies** when possible
2. **Implement chunked operations** for very large arrays
3. **Use custom allocators** for specific workloads (via `memory_optimize` module)
4. **Profile memory usage** for operations on large datasets

### Memory-Efficient Pattern

```rust
use crate::memory_optimize::MemoryPool;

// CORRECT - Uses memory pool for temporary allocations
pub fn large_operation(data: &Array<f64>) -> Result<Array<f64>> {
    let pool = MemoryPool::new();
    
    // Process in chunks to avoid excessive memory usage
    let chunk_size = pool.optimal_chunk_size(data.len());
    
    data.chunks(chunk_size)
        .map(|chunk| process_chunk(chunk, &pool))
        .collect()
}
```

## Testing Policy

### Test Requirements

1. **Unit tests for all public APIs**
2. **Property-based tests** for mathematical properties
3. **Benchmark tests** for performance-critical operations
4. **Integration tests** with and without optional features

### Test Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_operation() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = operation(&input).unwrap();
        
        // Use approx for floating-point comparisons
        assert_relative_eq!(result[0], expected_value, epsilon = 1e-10);
    }
    
    #[test]
    #[cfg(feature = "scirs")]
    fn test_with_scirs() {
        // Test SciRS2 integration
    }
}
```

## Examples

### Example 1: SIMD-Accelerated Operation

```rust
use crate::array::Array;
use crate::simd::SimdOps;

pub fn fast_euclidean_norm(v: &Array<f64>) -> f64 {
    // Automatically uses SIMD when beneficial
    v.simd_dot(v).unwrap().sqrt()
}
```

### Example 2: Adaptive Parallel Processing

```rust
use crate::parallel_optimize::threshold::DynamicThreshold;
use rayon::prelude::*;

pub fn parallel_map<T, F>(array: &Array<T>, f: F) -> Array<T>
where
    T: Send + Sync + Clone,
    F: Fn(&T) -> T + Send + Sync,
{
    let threshold = DynamicThreshold::new();
    
    if array.len() < threshold.get_threshold() {
        // Sequential for small arrays
        array.map(f)
    } else {
        // Parallel for large arrays
        let data: Vec<T> = array.to_vec()
            .par_iter()
            .map(f)
            .collect();
        Array::from_vec(data)
    }
}
```

### Example 3: SciRS2 Integration with Fallback

```rust
pub fn advanced_distribution(params: &DistParams) -> Result<Array<f64>> {
    #[cfg(feature = "scirs")]
    {
        // Use SciRS2's advanced distribution
        use crate::interop::scirs_compat;
        scirs_compat::noncentral_chisquare(params.df, params.nonc, &params.shape)
    }
    
    #[cfg(not(feature = "scirs"))]
    {
        // Use NumRS2's implementation
        crate::random::distributions::noncentral_chisquare(
            params.df, params.nonc, &params.shape
        )
    }
}
```

## Build Configuration

### Feature Flags

- `default = ["matrix_decomp"]` - Basic features
- `matrix_decomp` - Matrix decomposition algorithms
- `validation` - Runtime validation checks
- `fast` - Minimal feature set for fast compilation
- `scirs` - SciRS2 integration
- `gpu` - GPU acceleration support
- `lapack` - LAPACK-based linear algebra
- `ci-safe` - Features safe for CI without external dependencies

### Development Workflow

1. **Development builds**: Use `--features fast` for quick iteration
2. **Testing**: Test with various feature combinations
3. **Benchmarking**: Use `--features lapack,scirs` for best performance
4. **CI builds**: Use `--features ci-safe` to avoid external dependencies

## Enforcement

- Code reviews MUST check for policy compliance
- CI pipeline includes tests with different feature combinations
- Benchmarks track performance regressions
- Documentation must be kept up-to-date with API changes

## Benefits

By following these policies, NumRS2 achieves:

1. **High Performance**: Adaptive algorithms that scale from small to large data
2. **Portability**: Works across different platforms and configurations
3. **Maintainability**: Clear structure and consistent patterns
4. **Flexibility**: Optional features for different use cases
5. **Reliability**: Comprehensive testing and error handling

## Questions or Clarifications

If you have questions about these policies or need clarification:

1. Check the NumRS2 documentation
2. Review existing implementations in the codebase
3. Open an issue on GitHub for discussion
4. Consult with maintainers

Remember: Performance with safety and clarity!
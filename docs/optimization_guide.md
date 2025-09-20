# NumRS2 Optimization Guide

This guide explains how to leverage the various optimization features in NumRS2 for maximum performance.

## Table of Contents
- [Overview](#overview)
- [SIMD Operations](#simd-operations)
- [GPU Acceleration](#gpu-acceleration)
- [Parallel Processing](#parallel-processing)
- [AVX-512 Features](#avx-512-features)
- [Performance Tips](#performance-tips)

## Overview

NumRS2 provides multiple optimization layers through integration with scirs2-core:

1. **SIMD Operations**: Vectorized operations for CPU performance
2. **GPU Acceleration**: WGPU-based GPU compute
3. **Parallel Processing**: Multi-threaded operations using Rayon
4. **Platform-Specific**: AVX-512, ARM NEON optimizations

## SIMD Operations

### Basic Usage

```rust
use numrs2::optimized_ops::{simd_elementwise_ops, simd_vector_ops};
use ndarray::Array1;

// Element-wise operations
let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

let result = simd_elementwise_ops(&a.view(), &b.view()).unwrap();
let sum = result.add;  // SIMD-optimized addition
let product = result.mul;  // SIMD-optimized multiplication

// Vector operations
let stats = simd_vector_ops(&a.view());
println!("Sum: {}, Mean: {}, Norm: {}", stats.sum, stats.mean, stats.norm);
```

### Complex Number Operations

```rust
use numrs2::optimized_ops::simd_complex::{SimdComplexOps, SimdFft};
use num_complex::Complex;
use ndarray::Array1;

// Complex multiplication
let a = Array1::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
let b = Array1::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
let result = SimdComplexOps::complex_multiply(&a.view(), &b.view()).unwrap();

// FFT
let signal = Array1::from_vec(vec![
    Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
    Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
]);
let fft_result = SimdFft::fft(&signal.view()).unwrap();
```

## GPU Acceleration

### Setup

Enable GPU support in your `Cargo.toml`:

```toml
[dependencies]
numrs2 = { version = "0.1.0-beta.1", features = ["gpu"] }
```

### Basic GPU Operations

```rust
use numrs2::gpu::{GpuArray, add, multiply, matmul};
use numrs2::array::Array;

// Create GPU arrays
let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

let gpu_a = GpuArray::from_array(&a)?;
let gpu_b = GpuArray::from_array(&b)?;

// GPU operations
let gpu_sum = add(&gpu_a, &gpu_b)?;
let gpu_product = multiply(&gpu_a, &gpu_b)?;
let gpu_matmul = matmul(&gpu_a, &gpu_b)?;

// Transfer back to CPU
let result = gpu_matmul.to_array()?;
```

### GPU Performance Considerations

- GPU is most effective for large arrays (> 10,000 elements)
- Minimize CPU ↔ GPU transfers
- Batch operations when possible

## Parallel Processing

### Automatic Parallelization

```rust
use numrs2::optimized_ops::enhanced_math::*;
use ndarray::Array1;

let data = Array1::from_vec((0..1_000_000).map(|x| x as f64 * 0.01).collect());

// These functions automatically use parallel processing for large arrays
let sin_result = parallel_sin(&data.view());
let exp_result = parallel_exp(&data.view());
let sqrt_result = simd_sqrt(&data.view());
```

### Chunked Processing

```rust
use numrs2::optimized_ops::process_large_array;

let large_data = Array1::from_vec((0..10_000_000).map(|x| x as f64).collect());

// Process in chunks to optimize memory usage
let result = process_large_array(
    &large_data.view(),
    100_000,  // chunk size
    |chunk| chunk.map(|&x| x.sin() + x.cos())
)?;
```

## AVX-512 Features

### Masked Operations

```rust
#[cfg(target_arch = "x86_64")]
use numrs2::optimized_ops::avx512::Avx512Ops;
use ndarray::Array1;

if Avx512Ops::is_available() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let mask = Array1::from_vec(vec![true, false, true, false, true]);
    
    // Only add where mask is true
    let result = Avx512Ops::masked_add(&a.view(), &b.view(), &mask.view())?;
}
```

### Gather/Scatter Operations

```rust
#[cfg(target_arch = "x86_64")]
use numrs2::optimized_ops::avx512::Avx512Ops;

let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
let indices = Array1::from_vec(vec![4, 2, 0, 3, 1]);

// Gather values from specified indices
let gathered = Avx512Ops::gather(&data.view(), &indices.view())?;

// Scatter values to specified indices
let scattered = Avx512Ops::scatter(&gathered.view(), &indices.view(), 5)?;
```

## Performance Tips

### 1. Choose the Right Backend

```rust
use numrs2::optimized_ops::{get_optimization_info, should_use_parallel};

// Check available optimizations
println!("{}", get_optimization_info());

// Automatic selection based on data size
if should_use_parallel(data.len()) {
    // Use parallel operations
} else {
    // Use scalar operations
}
```

### 2. Adaptive Algorithm Selection

```rust
use numrs2::optimized_ops::SimdMathOps;

// Automatically chooses between scalar and SIMD based on array size
let result = SimdMathOps::adaptive_math_function(
    &data.view(),
    |data| enhanced_exp::simd_sqrt(data),  // SIMD path
    |x| x.sqrt()                           // Scalar path
);
```

### 3. Performance Benchmarking

Run the included benchmarks to understand performance characteristics:

```bash
# Basic performance benchmark
cargo run --example performance_benchmark --features scirs

# GPU performance benchmark
cargo run --example gpu_benchmark --features "gpu scirs"
```

### 4. Memory Considerations

- Use chunked processing for arrays larger than L3 cache
- Align data for SIMD operations when possible
- Consider memory bandwidth limitations

## Platform-Specific Notes

### x86_64 (Intel/AMD)
- AVX2 is widely supported and provides good performance
- AVX-512 provides additional features but check availability
- Use platform detection to enable optimal code paths

### ARM (Apple Silicon, etc.)
- NEON optimizations are automatically used when available
- Performance comparable to AVX2 on modern ARM chips

### GPU Considerations
- NVIDIA GPUs: Best performance with CUDA backend
- AMD GPUs: Good performance with ROCm backend
- Intel GPUs: Supported through WebGPU
- Apple GPUs: Metal backend provides native performance

## Example: Complete Optimization Pipeline

```rust
use numrs2::prelude::*;
use numrs2::optimized_ops::*;

fn optimized_computation(data: &Array<f64>) -> Result<Array<f64>> {
    // 1. Check optimization capabilities
    let caps = get_optimization_info();
    println!("Available optimizations: {}", caps);
    
    // 2. Convert to appropriate format
    let ndarray_data = data.to_ndarray_1d()?;
    
    // 3. Apply optimized operations
    let result = if data.len() > 1_000_000 {
        // Large data: use chunked processing
        process_large_array(&ndarray_data.view(), 100_000, |chunk| {
            enhanced_math::parallel_sin(&chunk)
        })?
    } else if data.len() > 1000 {
        // Medium data: use parallel processing
        enhanced_math::parallel_sin(&ndarray_data.view())
    } else {
        // Small data: use SIMD
        ndarray_data.map(|&x| x.sin())
    };
    
    // 4. Convert back to Array
    Ok(Array::from_ndarray(result.into_dyn()))
}
```

## Troubleshooting

### GPU Not Detected
- Ensure GPU drivers are installed
- Check if WGPU supports your GPU
- Try setting `WGPU_BACKEND` environment variable

### Performance Not as Expected
- Profile your code to identify bottlenecks
- Check data alignment for SIMD operations
- Ensure you're using appropriate chunk sizes
- Verify platform-specific optimizations are enabled

### Build Issues
- AVX-512 requires recent compiler versions
- GPU features require additional system dependencies
- Check feature flag combinations for conflicts
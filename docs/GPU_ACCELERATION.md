# GPU Acceleration in NumRS2

This document describes the GPU acceleration capabilities in NumRS2, focusing on the API, performance considerations, and usage patterns.

## Overview

NumRS2 provides GPU acceleration for array operations through the WGPU backend, which supports various GPU APIs including Vulkan, Metal, DirectX 12, and WebGPU. This allows NumRS2 to provide cross-platform GPU acceleration with minimal dependencies.

The GPU acceleration is designed to be seamlessly integrated with the existing NumRS2 API, making it easy to switch between CPU and GPU operations as needed.

## Enabling GPU Acceleration

GPU acceleration is an optional feature that needs to be explicitly enabled in your `Cargo.toml`:

```toml
[dependencies]
numrs2 = { version = "0.1.0-alpha.4", features = ["gpu"] }
```

Or when building NumRS2 directly:

```bash
cargo build --features gpu
```

## GPU Array API

The primary types for GPU acceleration are:

1. `GpuContext`: Manages the GPU device, queue, and resources
2. `GpuArray<T>`: Represents an array stored on the GPU

### Creating GPU Arrays

You can create GPU arrays from existing CPU arrays:

```rust
use numrs2::array::Array;
use numrs2::gpu;

// Create a CPU array
let cpu_array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Transfer to GPU
let gpu_array = gpu::GpuArray::from_array(&cpu_array)?;
```

### GPU Operations

The GPU module provides various operations that can be performed on GPU arrays:

```rust
use numrs2::gpu;

// Element-wise operations
let result = gpu::add(&a, &b)?;
let result = gpu::subtract(&a, &b)?;
let result = gpu::multiply(&a, &b)?;
let result = gpu::divide(&a, &b)?;

// Mathematical functions
let result = gpu::exp(&a)?;
let result = gpu::log(&a)?;
let result = gpu::sin(&a)?;
let result = gpu::cos(&a)?;

// Matrix operations
let result = gpu::matmul(&a, &b)?;
let result = gpu::transpose(&a)?;
```

### Transferring Results Back to CPU

Once you've performed operations on the GPU, you can transfer the results back to the CPU:

```rust
// Transfer GPU array back to CPU
let cpu_result = gpu_result.to_array()?;
```

## Performance Considerations

The following factors affect the performance of GPU operations:

1. **Data Transfer Overhead**: Transferring data between CPU and GPU involves overhead. For optimal performance, minimize the number of transfers.

2. **Array Size**: GPU acceleration provides the most benefit for large arrays. For small arrays, the overhead of GPU operations may outweigh the benefits.

3. **Operation Complexity**: Complex operations like matrix multiplication benefit more from GPU acceleration than simple operations like addition.

4. **Batched Operations**: Performing multiple operations in sequence on the GPU can significantly improve performance by avoiding intermediate transfers.

## Example: Matrix Multiplication

Here's a complete example of using GPU acceleration for matrix multiplication:

```rust
use numrs2::array::Array;
use numrs2::error::Result;
use numrs2::gpu;
use std::time::Instant;

fn main() -> Result<()> {
    // Create two random matrices on the CPU
    let a = create_random_matrix(1000, 1000)?;
    let b = create_random_matrix(1000, 1000)?;
    
    // CPU matrix multiplication
    let cpu_start = Instant::now();
    let cpu_result = a.dot(&b)?;
    let cpu_duration = cpu_start.elapsed();
    println!("CPU time: {:.2?}", cpu_duration);
    
    // GPU matrix multiplication
    let gpu_start = Instant::now();
    
    // Transfer matrices to GPU
    let gpu_a = gpu::GpuArray::from_array(&a)?;
    let gpu_b = gpu::GpuArray::from_array(&b)?;
    
    // Perform matrix multiplication on GPU
    let gpu_result = gpu::matmul(&gpu_a, &gpu_b)?;
    
    // Transfer result back to CPU
    let result = gpu_result.to_array()?;
    
    let gpu_duration = gpu_start.elapsed();
    println!("GPU time: {:.2?}", gpu_duration);
    
    // Calculate speedup
    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
    
    // Verify results match
    let max_diff = cpu_result
        .substract(&result)?
        .abs()?
        .max()?;
    
    println!("Maximum difference: {}", max_diff);
    
    Ok(())
}

fn create_random_matrix(rows: usize, cols: usize) -> Result<Array<f32>> {
    use numrs2::random::distributions::uniform;
    uniform(0.0, 1.0, &[rows, cols])
}
```

## Advanced Usage: Custom GPU Contexts

By default, NumRS2 uses a global GPU context. However, you can create and manage your own contexts:

```rust
use numrs2::gpu::GpuContext;

// Create a new GPU context
let context = gpu::context::new_context()?;

// Create an array with this context
let gpu_array = gpu::GpuArray::from_array_with_context(&cpu_array, context.clone())?;
```

This is useful for advanced scenarios where you need to manage multiple GPU devices or separate contexts for different parts of your application.

## Limitations

The current GPU implementation has some limitations:

1. Supported data types are limited to `f32` and `f64`
2. Not all NumRS2 operations are accelerated
3. Only dense arrays are supported (no sparse arrays)
4. Higher-dimensional transpose (>2D) is not implemented yet

## Future Directions

Future enhancements to the GPU acceleration module may include:

1. More operations and functions
2. Support for sparse arrays
3. Custom kernels and user-defined operations
4. Multi-GPU support
5. Integration with specialized deep learning operations

## Troubleshooting

### No GPU Detected

If you get a message saying that no compatible GPU was detected, you may need to install the appropriate GPU drivers.

### Type Conversion Errors

Ensure that your CPU and GPU arrays have the same data type. Mixing `f32` and `f64` is not supported.

### Performance Issues

If you're not seeing significant performance improvements, consider:

1. Increasing the size of your arrays
2. Batching multiple operations together
3. Keeping data on the GPU as long as possible to avoid transfer overhead

### Out of Memory Errors

GPU memory is limited. If you encounter out-of-memory errors, try:

1. Using smaller arrays
2. Processing data in batches
3. Using a more memory-efficient operation if available
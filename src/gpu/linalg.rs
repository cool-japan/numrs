//! GPU Linear Algebra Operations
//!
//! This module provides GPU-accelerated linear algebra operations including
//! matrix multiplication, vector operations, dot products, and norms.
//!
//! ## Features
//!
//! - **Matrix Multiplication**: GEMM operations with GPU acceleration
//! - **Vector Operations**: Dot products, norms, and vector math
//! - **High Performance**: Optimized compute shaders with workgroup tiling
//! - **Type Support**: f32 and f64 precision
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::gpu::linalg;
//! use numrs2::array::Array;
//!
//! # #[cfg(feature = "gpu")]
//! # fn example() -> numrs2::error::Result<()> {
//! // Create two matrices on CPU
//! let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
//! let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
//!
//! // Convert to GPU arrays
//! let gpu_a = numrs2::gpu::GpuArray::from_array(&a)?;
//! let gpu_b = numrs2::gpu::GpuArray::from_array(&b)?;
//!
//! // Perform GPU matrix multiplication
//! let gpu_c = linalg::matmul(&gpu_a, &gpu_b)?;
//!
//! // Convert back to CPU
//! let c = gpu_c.to_array()?;
//! # Ok(())
//! # }
//! ```

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::context::GpuContextRef;
use bytemuck::{Pod, Zeroable};

/// Parameters for matrix multiplication
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct MatMulParams {
    a_rows: u32,
    a_cols: u32,
    b_cols: u32,
    _padding: u32,
}

/// Parameters for reduction operations
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ReductionParams {
    op_type: u32,
    array_size: u32,
    workgroup_size: u32,
    _padding: u32,
}

/// Performs GPU-accelerated matrix multiplication (C = A * B)
///
/// # Arguments
///
/// * `a` - First matrix (M x K)
/// * `b` - Second matrix (K x N)
///
/// # Returns
///
/// Result matrix C (M x N)
///
/// # Errors
///
/// Returns an error if:
/// - Matrix dimensions are incompatible
/// - Either matrix is not 2D
/// - GPU computation fails
pub fn matmul<T: Pod + Zeroable + 'static>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    // Validate dimensions
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix multiplication requires 2D arrays".to_string(),
        ));
    }

    let a_rows = a.shape()[0];
    let a_cols = a.shape()[1];
    let b_rows = b.shape()[0];
    let b_cols = b.shape()[1];

    if a_cols != b_rows {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Incompatible matrix dimensions for multiplication: ({}, {}) * ({}, {})",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    // Get the GPU context
    let context = a.context();

    // Create output buffer
    let output_shape = vec![a_rows, b_cols];
    let output = GpuArray::<T>::new_with_shape(&output_shape, context.clone())?;

    // Create parameter buffer
    let params = MatMulParams {
        a_rows: a_rows as u32,
        a_cols: a_cols as u32,
        b_cols: b_cols as u32,
        _padding: 0,
    };

    let params_buffer = context.create_buffer(
        &[params],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    // Create bind group layout
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Matrix Multiplication Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matrix Multiplication Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Get the appropriate shader module
    let shader_module = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        context.matmul_f32_shader()
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        context.matmul_f64_shader()
    } else {
        return Err(NumRs2Error::TypeCastError(
            "Matrix multiplication only supports f32 and f64 types".to_string(),
        ));
    };

    // Create compute pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Matrix Multiplication Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

    let compute_pipeline =
        context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matrix Multiplication Pipeline"),
                layout: Some(&pipeline_layout),
                module: shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

    // Calculate workgroup counts (16x16 workgroup size from shader)
    let workgroup_count_x = (b_cols as u32).div_ceil(16);
    let workgroup_count_y = (a_rows as u32).div_ceil(16);

    // Execute the compute shader
    context.run_compute(
        &compute_pipeline,
        &[&bind_group],
        (workgroup_count_x, workgroup_count_y, 1),
    );

    Ok(output)
}

/// Computes the dot product of two vectors on GPU
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The dot product as a scalar
///
/// # Errors
///
/// Returns an error if vectors have different lengths or are not 1D
pub fn dot<T: Pod + Zeroable + num_traits::Zero + Clone + 'static>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<T> {
    // Validate dimensions
    if a.shape().len() != 1 || b.shape().len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Dot product requires 1D arrays".to_string(),
        ));
    }

    if a.size() != b.size() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Vectors must have same length: {} != {}",
            a.size(),
            b.size()
        )));
    }

    // For dot product, we can reshape to (1, n) and (n, 1) and use matmul
    // Then extract the single result
    let a_2d = a.reshape(&[1, a.size()])?;
    let b_2d = b.reshape(&[b.size(), 1])?;

    let result_2d = matmul(&a_2d, &b_2d)?;
    let result_array = result_2d.to_array()?;

    // Extract the single value
    result_array.get(&[0, 0]).map_err(|e| {
        NumRs2Error::IndexError(format!("Failed to extract dot product result: {}", e))
    })
}

/// Computes the L2 norm (Euclidean norm) of a vector on GPU
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// The L2 norm as a scalar
///
/// # Errors
///
/// Returns an error if the array is not 1D
pub fn norm_l2<T: Pod + Zeroable + num_traits::Float + 'static>(a: &GpuArray<T>) -> Result<T> {
    // Validate dimensions
    if a.shape().len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Norm requires a 1D array".to_string(),
        ));
    }

    // Compute dot product with itself
    let squared_norm = dot(a, a)?;

    // Return square root
    Ok(squared_norm.sqrt())
}

/// Computes the L1 norm (Manhattan norm) of a vector on GPU
///
/// For GPU implementation, this requires an absolute value operation followed by sum.
/// Currently returns a placeholder error as it requires additional shader support.
pub fn norm_l1<T: Pod + Zeroable + 'static>(_a: &GpuArray<T>) -> Result<T> {
    Err(NumRs2Error::NotImplemented(
        "L1 norm on GPU requires additional shader support".to_string(),
    ))
}

/// Matrix-vector multiplication (y = A * x)
///
/// # Arguments
///
/// * `a` - Matrix (M x N)
/// * `x` - Vector (N)
///
/// # Returns
///
/// Result vector y (M)
///
/// # Errors
///
/// Returns an error if dimensions are incompatible
pub fn matvec<T: Pod + Zeroable + 'static>(
    a: &GpuArray<T>,
    x: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    // Validate dimensions
    if a.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix-vector multiplication requires a 2D matrix".to_string(),
        ));
    }

    if x.shape().len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix-vector multiplication requires a 1D vector".to_string(),
        ));
    }

    let n = a.shape()[1];
    if x.size() != n {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Incompatible dimensions: matrix has {} columns but vector has {} elements",
            n,
            x.size()
        )));
    }

    // Reshape vector to column matrix (N x 1)
    let x_col = x.reshape(&[n, 1])?;

    // Perform matrix multiplication
    let result = matmul(a, &x_col)?;

    // Reshape back to 1D
    result.reshape(&[a.shape()[0]])
}

/// Vector-matrix multiplication (y = x^T * A)
///
/// # Arguments
///
/// * `x` - Vector (M)
/// * `a` - Matrix (M x N)
///
/// # Returns
///
/// Result vector y (N)
///
/// # Errors
///
/// Returns an error if dimensions are incompatible
pub fn vecmat<T: Pod + Zeroable + 'static>(
    x: &GpuArray<T>,
    a: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    // Validate dimensions
    if x.shape().len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Vector-matrix multiplication requires a 1D vector".to_string(),
        ));
    }

    if a.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Vector-matrix multiplication requires a 2D matrix".to_string(),
        ));
    }

    let m = a.shape()[0];
    if x.size() != m {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Incompatible dimensions: vector has {} elements but matrix has {} rows",
            x.size(),
            m
        )));
    }

    // Reshape vector to row matrix (1 x M)
    let x_row = x.reshape(&[1, m])?;

    // Perform matrix multiplication
    let result = matmul(&x_row, a)?;

    // Reshape back to 1D
    result.reshape(&[a.shape()[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_params_size() {
        // Verify that MatMulParams has correct size and alignment
        assert_eq!(std::mem::size_of::<MatMulParams>(), 16);
        assert_eq!(std::mem::align_of::<MatMulParams>(), 4);
    }

    #[test]
    fn test_reduction_params_size() {
        // Verify that ReductionParams has correct size and alignment
        assert_eq!(std::mem::size_of::<ReductionParams>(), 16);
        assert_eq!(std::mem::align_of::<ReductionParams>(), 4);
    }
}

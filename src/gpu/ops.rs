//! GPU Array Operations
//!
//! This module provides GPU-accelerated operations for NumRS2 arrays.
//! These operations leverage the GPU for significant performance improvements
//! on large data sets.

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::nd::{self, SliceRange};
use crate::gpu::reduce::{reduce_to_scalar, ReductionOp};
use wgpu::util::DeviceExt;

// Constants for compute shader configuration
const WORKGROUP_SIZE: u32 = 256;

/// Enumerates the types of element-wise operations
///
/// The discriminants are part of the shader ABI: they are written into the
/// operation-type field of the element-wise and broadcast kernels and switched
/// on inside WGSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ElementWiseOp {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
    Exp = 4,
    Log = 5,
    Sin = 6,
    Cos = 7,
    Tan = 8,
    Sqrt = 9,
    Abs = 10,
    Neg = 11,
    Pow = 12,
}

/// Adds two GPU arrays element-wise
pub fn add<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    element_wise_op(a, b, ElementWiseOp::Add)
}

/// Subtracts two GPU arrays element-wise
pub fn subtract<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    element_wise_op(a, b, ElementWiseOp::Subtract)
}

/// Multiplies two GPU arrays element-wise
pub fn multiply<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    element_wise_op(a, b, ElementWiseOp::Multiply)
}

/// Divides two GPU arrays element-wise
pub fn divide<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    element_wise_op(a, b, ElementWiseOp::Divide)
}

/// Performs element-wise exponentiation (e^x)
pub fn exp<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Exp)
}

/// Performs element-wise natural logarithm (ln)
pub fn log<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Log)
}

/// Performs element-wise sine
pub fn sin<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Sin)
}

/// Performs element-wise cosine
pub fn cos<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Cos)
}

/// Performs element-wise tangent
pub fn tan<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Tan)
}

/// Performs element-wise square root
pub fn sqrt<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Sqrt)
}

/// Performs element-wise absolute value
pub fn abs<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Abs)
}

/// Performs element-wise negation
pub fn neg<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    unary_element_wise_op(a, ElementWiseOp::Neg)
}

/// Performs element-wise power (a^b)
pub fn pow<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    element_wise_op(a, b, ElementWiseOp::Pow)
}

/// Performs matrix multiplication of two GPU arrays
pub fn matmul<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    // Validate shapes for matrix multiplication
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![2],
            actual: vec![a.shape().len(), b.shape().len()],
        });
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Cannot multiply matrices with shapes {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    // Output shape is [a_rows, b_cols]
    let out_shape = vec![a_shape[0], b_shape[1]];
    let context = a.context().clone();

    // Create output array
    let result = GpuArray::<T>::new_with_shape(&out_shape, context.clone())?;

    // Create bind group layout and pipeline
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NumRS2 MatMul Bind Group Layout"),
                entries: &[
                    // Input matrix A
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
                    // Input matrix B
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
                    // Output matrix C
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
                    // Dimensions
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

    // Select the appropriate shader based on the type
    let shader = if std::mem::size_of::<T>() == 4 {
        context.matmul_f32_shader()
    } else {
        context.matmul_f64_shader()?
    };

    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 MatMul Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

    // Create dimensions buffer
    let dims = [
        a_shape[0] as u32, // a_rows
        a_shape[1] as u32, // a_cols (same as b_rows)
        b_shape[1] as u32, // b_cols
        0,                 // padding
    ];

    let dimensions_buffer =
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Dimensions"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NumRS2 MatMul Bind Group"),
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
                    resource: result.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
        });

    // Calculate workgroup dimensions (one workgroup per output element with tiling)
    let workgroup_count_x = (out_shape[1] as f32 / 16.0).ceil() as u32;
    let workgroup_count_y = (out_shape[0] as f32 / 16.0).ceil() as u32;

    // Run the compute pass
    context.run_compute(
        &pipeline,
        &[&bind_group],
        (workgroup_count_x, workgroup_count_y, 1),
    );

    Ok(result)
}

/// Transposes a GPU array
pub fn transpose<T: bytemuck::Pod + bytemuck::Zeroable>(a: &GpuArray<T>) -> Result<GpuArray<T>> {
    // Validate that the array has at least 2 dimensions
    if a.shape().len() < 2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Cannot transpose array with less than 2 dimensions, got shape {:?}",
            a.shape()
        )));
    }

    // For 2D arrays, simply swap the dimensions
    if a.shape().len() == 2 {
        let mut out_shape = a.shape().to_vec();
        out_shape.swap(0, 1);

        let context = a.context().clone();
        let result = GpuArray::<T>::new_with_shape(&out_shape, context.clone())?;

        // Create bind group layout and pipeline
        let bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NumRS2 Transpose Bind Group Layout"),
                    entries: &[
                        // Input array
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
                        // Output array
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Dimensions
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        // Select the appropriate shader based on the type
        let shader = if std::mem::size_of::<T>() == 4 {
            context.element_wise_f32_shader()
        } else {
            context.element_wise_f64_shader()?
        };

        let pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NumRS2 Transpose Pipeline Layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NumRS2 Transpose Pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some("transpose"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create dimensions buffer. The shader indexes the input as
        // `y * width + x`, so `width` is the length of a row - the number of
        // input columns - and `height` is the number of input rows.
        let rows = a.shape()[0] as u32;
        let cols = a.shape()[1] as u32;
        let dims = [cols, rows, 0, 0];

        let dimensions_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transpose Dimensions"),
                    contents: bytemuck::cast_slice(&dims),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Create bind group
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("NumRS2 Transpose Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dimensions_buffer.as_entire_binding(),
                    },
                ],
            });

        // One thread per input element, tiled 16x16 over (column, row).
        let workgroup_count_x = cols.div_ceil(16);
        let workgroup_count_y = rows.div_ceil(16);

        // Run the compute pass
        context.run_compute(
            &pipeline,
            &[&bind_group],
            (workgroup_count_x, workgroup_count_y, 1),
        );

        return Ok(result);
    }

    // Higher-rank arrays go through the general permutation kernel, which
    // reverses the axis order exactly like NumPy's `a.T`.
    nd::reverse_axes(a)
}

/// Permutes the axes of a GPU array (NumPy's `transpose(a, axes)`)
///
/// `axes` must be a permutation of `0..ndim`; output axis `d` is taken from
/// input axis `axes[d]`.
///
/// # Errors
///
/// Returns an error if `axes` is not a permutation of the array's axes.
pub fn permute_axes<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    axes: &[usize],
) -> Result<GpuArray<T>> {
    nd::permute_axes(a, axes)
}

/// Helper function for element-wise binary operations
fn element_wise_op<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
    op: ElementWiseOp,
) -> Result<GpuArray<T>> {
    // Validate shapes
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    // Create output array with same shape
    let context = a.context().clone();
    let result = GpuArray::<T>::new_with_shape(a.shape(), context.clone())?;

    // Create bind group layout
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NumRS2 Element-wise Bind Group Layout"),
                entries: &[
                    // Input array A
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
                    // Input array B
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
                    // Output array
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
                    // Operation type and array size
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

    // Select the appropriate shader based on the type
    let shader = if std::mem::size_of::<T>() == 4 {
        context.element_wise_f32_shader()
    } else {
        context.element_wise_f64_shader()?
    };

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Element-wise Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("binary_op"),
            cache: None,
            compilation_options: Default::default(),
        });

    // Create uniform buffer with operation type and size
    let params = [op as u32, a.size() as u32, 0, 0];

    let params_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Element-wise Op Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NumRS2 Element-wise Bind Group"),
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
                    resource: result.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Calculate workgroup count
    let total_threads = a.size() as u32;
    let workgroup_count = total_threads.div_ceil(WORKGROUP_SIZE);

    // Run the compute pass
    context.run_compute(&pipeline, &[&bind_group], (workgroup_count, 1, 1));

    Ok(result)
}

/// Computes the sum of all elements in a GPU array (f32 version)
pub fn sum_f32(a: &GpuArray<f32>) -> Result<f32> {
    reduction_op_f32(a, ReductionOp::Sum)
}

/// Computes the sum of all elements in a GPU array (f64 version)
pub fn sum_f64(a: &GpuArray<f64>) -> Result<f64> {
    reduction_op_f64(a, ReductionOp::Sum)
}

/// Computes the mean of all elements in a GPU array (f32 version)
pub fn mean_f32(a: &GpuArray<f32>) -> Result<f32> {
    reduction_op_f32(a, ReductionOp::Mean)
}

/// Computes the mean of all elements in a GPU array (f64 version)
pub fn mean_f64(a: &GpuArray<f64>) -> Result<f64> {
    reduction_op_f64(a, ReductionOp::Mean)
}

/// Computes the maximum value in a GPU array (f32 version)
pub fn max_f32(a: &GpuArray<f32>) -> Result<f32> {
    reduction_op_f32(a, ReductionOp::Max)
}

/// Computes the maximum value in a GPU array (f64 version)
pub fn max_f64(a: &GpuArray<f64>) -> Result<f64> {
    reduction_op_f64(a, ReductionOp::Max)
}

/// Computes the minimum value in a GPU array (f32 version)
pub fn min_f32(a: &GpuArray<f32>) -> Result<f32> {
    reduction_op_f32(a, ReductionOp::Min)
}

/// Computes the minimum value in a GPU array (f64 version)
pub fn min_f64(a: &GpuArray<f64>) -> Result<f64> {
    reduction_op_f64(a, ReductionOp::Min)
}
/// Helper function for f32 reduction operations
///
/// The whole reduction runs on the GPU: repeated passes of the workgroup
/// tree-reduction kernel shrink the array by a factor of 256 per pass until a
/// single value remains, which is the only value copied back to the host.
fn reduction_op_f32(a: &GpuArray<f32>, op: ReductionOp) -> Result<f32> {
    let total_elements = a.size();
    let value = reduce_to_scalar(a, op, false)?;

    // The kernel sums for `Mean`; scaling by the element count on the host
    // keeps the per-workgroup element bookkeeping out of the shader.
    match op {
        ReductionOp::Mean => Ok(value / total_elements as f32),
        _ => Ok(value),
    }
}

/// Helper function for f64 reduction operations
///
/// See [`reduction_op_f32`]; this variant requires a device with the
/// `SHADER_F64` feature.
fn reduction_op_f64(a: &GpuArray<f64>, op: ReductionOp) -> Result<f64> {
    let total_elements = a.size();
    let value = reduce_to_scalar(a, op, false)?;

    match op {
        ReductionOp::Mean => Ok(value / total_elements as f64),
        _ => Ok(value),
    }
}

/// Helper function for element-wise unary operations
fn unary_element_wise_op<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    op: ElementWiseOp,
) -> Result<GpuArray<T>> {
    // Create output array with same shape
    let context = a.context().clone();
    let result = GpuArray::<T>::new_with_shape(a.shape(), context.clone())?;

    // Create bind group layout
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NumRS2 Unary Element-wise Bind Group Layout"),
                entries: &[
                    // Input array
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
                    // Output array
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Operation type and array size
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

    // Select the appropriate shader based on the type
    let shader = if std::mem::size_of::<T>() == 4 {
        context.element_wise_f32_shader()
    } else {
        context.element_wise_f64_shader()?
    };

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Unary Element-wise Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Unary Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("unary_op"),
            cache: None,
            compilation_options: Default::default(),
        });

    // Create uniform buffer with operation type and size
    let params = [op as u32, a.size() as u32, 0, 0];

    let params_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Unary Element-wise Op Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NumRS2 Unary Element-wise Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Calculate workgroup count
    let total_threads = a.size() as u32;
    let workgroup_count = total_threads.div_ceil(WORKGROUP_SIZE);

    // Run the compute pass
    context.run_compute(&pipeline, &[&bind_group], (workgroup_count, 1, 1));

    Ok(result)
}

/// Performs broadcasting-aware element-wise addition
///
/// Supports NumPy-style broadcasting where arrays with different shapes
/// can be combined if they are compatible.
pub fn broadcast_add<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let output_shape = broadcast_shapes(a.shape(), b.shape())?;
    broadcast_binary_op(a, b, &output_shape, ElementWiseOp::Add)
}

/// Performs broadcasting-aware element-wise subtraction
pub fn broadcast_subtract<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let output_shape = broadcast_shapes(a.shape(), b.shape())?;
    broadcast_binary_op(a, b, &output_shape, ElementWiseOp::Subtract)
}

/// Performs broadcasting-aware element-wise multiplication
pub fn broadcast_multiply<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let output_shape = broadcast_shapes(a.shape(), b.shape())?;
    broadcast_binary_op(a, b, &output_shape, ElementWiseOp::Multiply)
}

/// Performs broadcasting-aware element-wise division
pub fn broadcast_divide<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let output_shape = broadcast_shapes(a.shape(), b.shape())?;
    broadcast_binary_op(a, b, &output_shape, ElementWiseOp::Divide)
}

/// Performs broadcasting-aware element-wise power (`a ** b`)
pub fn broadcast_pow<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let output_shape = broadcast_shapes(a.shape(), b.shape())?;
    broadcast_binary_op(a, b, &output_shape, ElementWiseOp::Pow)
}

/// Determines the output shape for broadcasting two arrays
///
/// Implements NumPy's rule: the shapes are right-aligned, and each axis pair
/// must either match or contain a one, which is stretched.
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let max_dims = shape_a.len().max(shape_b.len());
    let mut result = vec![1; max_dims];

    for i in 0..max_dims {
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            result[max_dims - 1 - i] = dim_a;
        } else if dim_a == 1 {
            result[max_dims - 1 - i] = dim_b;
        } else if dim_b == 1 {
            result[max_dims - 1 - i] = dim_a;
        } else {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape_a.to_vec(),
                actual: shape_b.to_vec(),
            });
        }
    }

    Ok(result)
}

/// Helper function for broadcasting binary operations
///
/// Equal shapes take the dense element-wise kernel, which needs no index
/// arithmetic at all; every other compatible shape pair goes to the
/// stride-driven broadcast kernel, where a stretched axis is expressed as a
/// stride of zero.
fn broadcast_binary_op<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
    output_shape: &[usize],
    op: ElementWiseOp,
) -> Result<GpuArray<T>> {
    if a.shape() == b.shape() {
        return element_wise_op(a, b, op);
    }

    nd::broadcast_binary(a, b, output_shape, op)
}

/// Copies a GPU array with optional format conversion
pub fn copy_with_format<T: bytemuck::Pod + bytemuck::Zeroable>(
    src: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let context = src.context().clone();
    let result = GpuArray::<T>::new_with_shape(src.shape(), context.clone())?;

    // Create command encoder for the copy
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NumRS2 Copy Encoder"),
        });

    encoder.copy_buffer_to_buffer(
        src.buffer(),
        0,
        result.buffer(),
        0,
        (src.size() * src.element_size()) as u64,
    );

    context.queue().submit(std::iter::once(encoder.finish()));

    Ok(result)
}

/// Fills a GPU array with a scalar value
pub fn fill<T: bytemuck::Pod + bytemuck::Zeroable + Clone>(
    array: &mut GpuArray<T>,
    value: T,
) -> Result<()> {
    let data = vec![value; array.size()];
    array
        .context()
        .queue()
        .write_buffer(array.buffer(), 0, bytemuck::cast_slice(&data));
    Ok(())
}

/// Extracts a contiguous slice of a GPU array
///
/// One `(start, end)` range per axis is required. The data never leaves the
/// GPU: the strided gather kernel writes the selected elements directly into
/// a new, dense array.
///
/// # Errors
///
/// Returns an error if the number of ranges does not match the rank, or if a
/// range is empty or out of bounds.
pub fn slice<T: bytemuck::Pod + bytemuck::Zeroable>(
    array: &GpuArray<T>,
    ranges: &[(usize, usize)],
) -> Result<GpuArray<T>> {
    let ranges: Vec<SliceRange> = ranges.iter().copied().map(SliceRange::from).collect();
    nd::slice_strided(array, &ranges)
}

/// Extracts a strided slice of a GPU array
///
/// Like [`slice`], but every axis also carries a step, so
/// `SliceRange::with_step(0, 8, 2)` keeps indices `0, 2, 4, 6` of that axis -
/// NumPy's `a[0:8:2]`.
pub fn slice_with_steps<T: bytemuck::Pod + bytemuck::Zeroable>(
    array: &GpuArray<T>,
    ranges: &[SliceRange],
) -> Result<GpuArray<T>> {
    nd::slice_strided(array, ranges)
}

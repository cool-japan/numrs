//! GPU Array Operations
//!
//! This module provides GPU-accelerated operations for NumRS2 arrays.
//! These operations leverage the GPU for significant performance improvements
//! on large data sets.

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use wgpu::util::DeviceExt;

// Constants for compute shader configuration
const WORKGROUP_SIZE: u32 = 256;

/// Enumerates the types of element-wise operations
enum ElementWiseOp {
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
        context.matmul_f64_shader()
    };

    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 MatMul Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
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
            context.element_wise_f64_shader()
        };

        let pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NumRS2 Transpose Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NumRS2 Transpose Pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "transpose",
            });

        // Create dimensions buffer
        let dims = [a.shape()[0] as u32, a.shape()[1] as u32, 0, 0];

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

        // Calculate workgroup dimensions
        let workgroup_count_x = (out_shape[1] as f32 / 16.0).ceil() as u32;
        let workgroup_count_y = (out_shape[0] as f32 / 16.0).ceil() as u32;

        // Run the compute pass
        context.run_compute(
            &pipeline,
            &[&bind_group],
            (workgroup_count_x, workgroup_count_y, 1),
        );

        return Ok(result);
    }

    // For higher dimensions, we need a more complex implementation
    // (not covered in this initial version)
    Err(NumRs2Error::NotImplemented(
        "Transpose for arrays with more than 2 dimensions is not implemented yet".to_string(),
    ))
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
        context.element_wise_f64_shader()
    };

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Element-wise Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "binary_op",
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

/// Enumerates the types of reduction operations
#[derive(Clone, Copy)]
enum ReductionOp {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Min = 3,
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
fn reduction_op_f32(a: &GpuArray<f32>, op: ReductionOp) -> Result<f32> {
    let context = a.context().clone();
    let total_elements = a.size() as u32;

    // Calculate number of workgroups needed
    let workgroup_count = (total_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // Create output buffer for partial results (one per workgroup)
    let partial_results_size = workgroup_count as usize * std::mem::size_of::<f32>();
    let partial_results_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Partial Results"),
        size: partial_results_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create bind group layout
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NumRS2 Reduction Bind Group Layout"),
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
                    // Output partial results
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
                    // Parameters (operation type, array size)
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

    let shader = context.reduction_f32_shader();

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Reduction Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "reduction",
        });

    // Create uniform buffer with operation type and size
    let params = [op as u32, total_elements, 0, 0];

    let params_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reduction Op Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NumRS2 Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Run the compute pass
    context.run_compute(&pipeline, &[&bind_group], (workgroup_count, 1, 1));

    // Read back partial results
    let staging_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Staging Buffer"),
        size: partial_results_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy from GPU to staging buffer
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduction Copy Encoder"),
        });

    encoder.copy_buffer_to_buffer(
        &partial_results_buffer,
        0,
        &staging_buffer,
        0,
        partial_results_size as u64,
    );

    context.queue().submit(Some(encoder.finish()));

    // Map the staging buffer and read results
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    context.device().poll(wgpu::Maintain::Wait);
    rx.recv()
        .unwrap()
        .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to map buffer: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let partial_results: &[f32] = bytemuck::cast_slice(&data);

    // Perform final reduction on CPU
    let final_result = match op {
        ReductionOp::Sum => partial_results.iter().sum(),
        ReductionOp::Mean => partial_results.iter().sum::<f32>() / total_elements as f32,
        ReductionOp::Max => partial_results
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        ReductionOp::Min => partial_results
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
    };

    drop(data);
    staging_buffer.unmap();

    Ok(final_result)
}

/// Helper function for f64 reduction operations
fn reduction_op_f64(a: &GpuArray<f64>, op: ReductionOp) -> Result<f64> {
    let context = a.context().clone();
    let total_elements = a.size() as u32;

    // Calculate number of workgroups needed
    let workgroup_count = (total_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // Create output buffer for partial results (one per workgroup)
    let partial_results_size = workgroup_count as usize * std::mem::size_of::<f64>();
    let partial_results_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Partial Results"),
        size: partial_results_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create bind group layout
    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NumRS2 Reduction Bind Group Layout"),
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
                    // Output partial results
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
                    // Parameters (operation type, array size)
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

    let shader = context.reduction_f64_shader();

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Reduction Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "reduction",
        });

    // Create uniform buffer with operation type and size
    let params = [op as u32, total_elements, 0, 0];

    let params_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reduction Op Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NumRS2 Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

    // Run the compute pass
    context.run_compute(&pipeline, &[&bind_group], (workgroup_count, 1, 1));

    // Read back partial results
    let staging_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Staging Buffer"),
        size: partial_results_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy from GPU to staging buffer
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduction Copy Encoder"),
        });

    encoder.copy_buffer_to_buffer(
        &partial_results_buffer,
        0,
        &staging_buffer,
        0,
        partial_results_size as u64,
    );

    context.queue().submit(Some(encoder.finish()));

    // Map the staging buffer and read results
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    context.device().poll(wgpu::Maintain::Wait);
    rx.recv()
        .unwrap()
        .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to map buffer: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let partial_results: &[f64] = bytemuck::cast_slice(&data);

    // Perform final reduction on CPU
    let final_result = match op {
        ReductionOp::Sum => partial_results.iter().sum(),
        ReductionOp::Mean => partial_results.iter().sum::<f64>() / total_elements as f64,
        ReductionOp::Max => partial_results
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        ReductionOp::Min => partial_results
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
    };

    drop(data);
    staging_buffer.unmap();

    Ok(final_result)
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
        context.element_wise_f64_shader()
    };

    // Create pipeline
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NumRS2 Unary Element-wise Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NumRS2 Unary Element-wise Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "unary_op",
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

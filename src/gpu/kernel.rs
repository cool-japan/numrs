//! Compute-kernel plumbing shared by the GPU operation modules.
//!
//! Every GPU kernel in this crate needs the same four things: a bind group
//! layout matching its bindings, a pipeline, a bind group and a dispatch. This
//! module builds all four from a compact description so that the operation
//! modules only have to express what is actually specific to them - the
//! buffers and the metadata.
//!
//! It also centralises the two rules that are easy to get wrong when a kernel
//! runs one thread per element:
//!
//! * the workgroup grid must not exceed `max_compute_workgroups_per_dimension`
//!   along any axis, which for large arrays means spilling into a second
//!   dimension ([`linear_dispatch`]), and
//! * the shader has to be told how wide the grid is so it can flatten the
//!   two-dimensional workgroup id back into a linear element index.

use crate::error::{NumRs2Error, Result};
use crate::gpu::context::GpuContext;
use wgpu::util::DeviceExt;

/// Threads per workgroup used by every one-thread-per-element kernel here.
///
/// This must stay in sync with the `@workgroup_size(256)` attributes in the
/// WGSL sources under `shaders/`.
pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// A single resource binding of a compute kernel.
///
/// The variants map onto the WGSL address spaces used by the shaders:
/// `var<storage, read>`, `var<storage, read_write>` and `var<uniform>`.
pub(crate) enum Binding<'a> {
    /// Read-only storage buffer.
    Storage(&'a wgpu::Buffer),
    /// Writable storage buffer.
    StorageMut(&'a wgpu::Buffer),
    /// Uniform buffer.
    Uniform(&'a wgpu::Buffer),
}

impl Binding<'_> {
    fn binding_type(&self) -> wgpu::BindingType {
        match self {
            Binding::Storage(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Binding::StorageMut(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            Binding::Uniform(_) => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        }
    }

    fn buffer(&self) -> &wgpu::Buffer {
        match self {
            Binding::Storage(buffer) | Binding::StorageMut(buffer) | Binding::Uniform(buffer) => {
                buffer
            }
        }
    }
}

/// Builds the pipeline for `entry_point` and dispatches `groups` workgroups.
///
/// The bind group layout is derived from `bindings`, whose order must match
/// the `@binding(n)` indices of the shader.
pub(crate) fn dispatch(
    context: &GpuContext,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    label: &str,
    bindings: &[Binding<'_>],
    groups: (u32, u32, u32),
) {
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: binding.binding_type(),
            count: None,
        })
        .collect();

    let bind_group_layout =
        context
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &layout_entries,
            });

    let group_entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: binding.buffer().as_entire_binding(),
        })
        .collect();

    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bind_group_layout,
            entries: &group_entries,
        });

    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry_point),
            cache: None,
            compilation_options: Default::default(),
        });

    context.run_compute(&pipeline, &[&bind_group], groups);
}

/// Splits a one-thread-per-element dispatch into a legal workgroup grid.
///
/// Returns `(groups_x, groups_y)`. The shaders recover the linear element
/// index as `(workgroup_id.y * groups_x + workgroup_id.x) * 256 + local_id.x`,
/// so `groups_x` has to be handed to the kernel through its metadata.
pub(crate) fn linear_dispatch(context: &GpuContext, count: usize) -> Result<(u32, u32)> {
    let total_groups = count.div_ceil(WORKGROUP_SIZE as usize).max(1);
    let max_per_dim = context
        .device()
        .limits()
        .max_compute_workgroups_per_dimension as usize;

    if total_groups <= max_per_dim {
        return Ok((total_groups as u32, 1));
    }

    let groups_y = total_groups.div_ceil(max_per_dim);
    if groups_y > max_per_dim {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Operation over {} elements exceeds the device dispatch limit of {} workgroups per dimension",
            count, max_per_dim
        )));
    }

    Ok((max_per_dim as u32, groups_y as u32))
}

/// Uploads a kernel metadata table as a read-only storage buffer.
pub(crate) fn meta_buffer(context: &GpuContext, label: &str, meta: &[u32]) -> wgpu::Buffer {
    context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(meta),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

/// Uploads a small uniform block.
pub(crate) fn uniform_buffer<T: bytemuck::Pod>(
    context: &GpuContext,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Number of 32-bit words occupied by one element of type `T`.
///
/// The type-agnostic data-movement kernels address memory as `array<u32>`, so
/// they only support element types whose size is a non-zero multiple of four
/// bytes. That covers every type this crate puts on the GPU (f32, f64, i32,
/// u32, ...) and rejects anything else with a clear error instead of
/// corrupting data.
pub(crate) fn words_per_element<T>() -> Result<u32> {
    let size = std::mem::size_of::<T>();
    if size == 0 || !size.is_multiple_of(4) {
        return Err(NumRs2Error::InvalidOperation(format!(
            "GPU data movement requires an element size that is a non-zero multiple of 4 bytes, got {} bytes",
            size
        )));
    }
    Ok((size / 4) as u32)
}

/// Row-major (C-contiguous) strides, in elements, for `shape`.
///
/// GPU buffers created by this crate always hold data in logical row-major
/// order - [`crate::gpu::GpuArray::from_array`] materialises even a
/// non-contiguous CPU array in logical order before uploading it - so the
/// strides of a GPU buffer are always derivable from its shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// Converts a `usize` metadata entry into the `u32` the shaders expect.
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        NumRs2Error::InvalidOperation(format!(
            "{} of {} exceeds the 32-bit range addressable by GPU kernels",
            what, value
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert!(contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn test_words_per_element() {
        assert_eq!(words_per_element::<f32>().ok(), Some(1));
        assert_eq!(words_per_element::<f64>().ok(), Some(2));
        assert!(words_per_element::<u8>().is_err());
    }

    #[test]
    fn test_to_u32() {
        assert_eq!(to_u32(7, "value").ok(), Some(7));
        assert!(to_u32(usize::MAX, "value").is_err());
    }
}

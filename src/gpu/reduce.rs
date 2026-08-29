//! Whole-array reductions executed entirely on the GPU.
//!
//! A reduction is run as a sequence of passes of the same kernel
//! (`shaders/reduction_template.wgsl`): every pass folds each workgroup's 256
//! values into one partial result, so the working set shrinks by a factor of
//! 256 per pass until a single value is left. Only that one value is copied
//! back to the host, instead of the array of per-workgroup partials the first
//! implementation used to reduce on the CPU.
//!
//! The first pass can optionally take the absolute value of every input,
//! which is what turns the plain sum into an L1 norm without a second kernel.

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::context::GpuContextRef;
use crate::gpu::kernel::{dispatch, linear_dispatch, to_u32, uniform_buffer, Binding};
use bytemuck::{Pod, Zeroable};

/// Reduction kinds understood by the reduction kernel.
///
/// The discriminants are part of the kernel ABI: they are written into the
/// `op_type` field of `ReduceParams` and switched on inside the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReductionOp {
    /// Sum of all elements.
    Sum = 0,
    /// Arithmetic mean; the kernel sums and the host divides.
    Mean = 1,
    /// Largest element, propagating NaN like NumPy.
    Max = 2,
    /// Smallest element, propagating NaN like NumPy.
    Min = 3,
}

/// Uniform block handed to the reduction kernel, one per pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ReduceParams {
    op_type: u32,
    n_elements: u32,
    groups_x: u32,
    apply_abs: u32,
}

/// Reduces `array` to a single value without leaving the GPU.
///
/// `apply_abs` folds an absolute value into the first pass. `Mean` is summed
/// here and scaled by the caller, which keeps the kernel free of the
/// per-workgroup element counts an in-shader mean would need.
pub(crate) fn reduce_to_scalar<T: Pod + Zeroable>(
    array: &GpuArray<T>,
    op: ReductionOp,
    apply_abs: bool,
) -> Result<T> {
    let context = array.context().clone();
    let shader = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        context.reduction_f32_shader()
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        context.reduction_f64_shader()?
    } else {
        return Err(NumRs2Error::TypeCastError(
            "GPU reductions only support f32 and f64 element types".to_string(),
        ));
    };

    if array.size() == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot reduce an empty GPU array".to_string(),
        ));
    }

    let element_size = std::mem::size_of::<T>();
    let mut current_len = array.size();
    let mut current: Option<wgpu::Buffer> = None;
    let mut abs_pass = apply_abs;

    while current_len > 1 || abs_pass {
        let input = match current.as_ref() {
            Some(buffer) => buffer,
            None => array.buffer(),
        };

        let (groups_x, groups_y) = linear_dispatch(&context, current_len)?;
        // Every dispatched workgroup writes exactly one partial, including the
        // tail groups that only saw identity elements, so the output length is
        // the full grid size rather than ceil(len / 256).
        let out_len = (groups_x as usize) * (groups_y as usize);
        if current_len > 1 && out_len >= current_len {
            return Err(NumRs2Error::RuntimeError(format!(
                "GPU reduction failed to make progress: {} elements would reduce to {}",
                current_len, out_len
            )));
        }

        let output = context.create_empty_buffer(
            (out_len * element_size) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = ReduceParams {
            op_type: op as u32,
            n_elements: to_u32(current_len, "element count")?,
            groups_x,
            apply_abs: u32::from(abs_pass),
        };
        let params_buffer = uniform_buffer(&context, "Reduction Params", &params);

        dispatch(
            &context,
            shader,
            "reduce",
            "NumRS2 Reduction",
            &[
                Binding::Storage(input),
                Binding::StorageMut(&output),
                Binding::Uniform(&params_buffer),
            ],
            (groups_x, groups_y, 1),
        );

        current = Some(output);
        current_len = out_len;
        abs_pass = false;
    }

    let final_buffer = match current.as_ref() {
        Some(buffer) => buffer,
        // A single-element array with no absolute value to apply never enters
        // the loop; read the input straight back.
        None => array.buffer(),
    };

    read_first_element::<T>(&context, final_buffer)
}

/// Copies the first element of `buffer` back to the host.
fn read_first_element<T: Pod + Zeroable>(
    context: &GpuContextRef,
    buffer: &wgpu::Buffer,
) -> Result<T> {
    let element_size = std::mem::size_of::<T>() as u64;

    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Result Staging Buffer"),
        size: element_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduction Readback Encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, element_size);
    context.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        // Nothing to report to if the receiver has already been dropped.
        let _ = sender.send(result);
    });

    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| {
            NumRs2Error::RuntimeError(format!(
                "GPU device poll failed while reading a reduction result: {}",
                e
            ))
        })?;

    receiver
        .recv()
        .map_err(|e| {
            NumRs2Error::RuntimeError(format!(
                "Failed to receive the reduction buffer mapping result: {}",
                e
            ))
        })?
        .map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to map the reduction result buffer: {}", e))
        })?;

    let value = {
        let data = slice.get_mapped_range().map_err(|e| {
            NumRs2Error::RuntimeError(format!(
                "Failed to access the mapped reduction result: {}",
                e
            ))
        })?;
        bytemuck::try_pod_read_unaligned::<T>(&data).map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to decode the reduction result: {:?}", e))
        })?
    };

    staging.unmap();
    Ok(value)
}

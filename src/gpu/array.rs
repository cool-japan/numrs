//! GPU Array Implementation
//!
//! This module provides the GpuArray struct which represents an N-dimensional array
//! stored on the GPU. GpuArray provides methods for transferring data between CPU and GPU,
//! as well as accessing array metadata.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::gpu::context::GpuContextRef;
use std::fmt;
use std::marker::PhantomData;

/// A multi-dimensional array stored on the GPU
pub struct GpuArray<T> {
    /// Reference to the GPU context
    context: GpuContextRef,
    /// Buffer storing the array data on the GPU
    buffer: wgpu::Buffer,
    /// Shape of the array
    shape: Vec<usize>,
    /// Strides of the array in elements
    strides: Vec<usize>,
    /// Total number of elements in the array
    size: usize,
    /// Size of a single element in bytes
    element_size: usize,
    /// Phantom data to track the type parameter
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> GpuArray<T> {
    /// Creates a new GPU array from a CPU array
    pub fn from_array(array: &Array<T>) -> Result<Self> {
        // Create a default context if needed
        let context = crate::gpu::util::get_default_context()?;
        Self::from_array_with_context(array, context)
    }

    /// Creates a new GPU array from a CPU array with the specified context
    pub fn from_array_with_context(array: &Array<T>, context: GpuContextRef) -> Result<Self> {
        // `to_vec` walks the CPU array in logical order, so the GPU buffer is
        // always C-contiguous regardless of the source layout. The strides
        // recorded here therefore describe the *uploaded* buffer, in elements,
        // matching `new_with_shape` and `reshape` (and this type's `strides`
        // documentation) rather than the source array's memory layout.
        let data = array.to_vec();
        let shape = array.shape().to_vec();
        let strides = crate::gpu::kernel::contiguous_strides(&shape);
        let size = array.size();
        let element_size = std::mem::size_of::<T>();

        // Create the GPU buffer
        let buffer = context.create_buffer(
            &data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Ok(Self {
            context,
            buffer,
            shape,
            strides,
            size,
            element_size,
            _phantom: PhantomData,
        })
    }

    /// Creates a new GPU array with the specified shape
    pub fn new_with_shape(shape: &[usize], context: GpuContextRef) -> Result<Self> {
        let size = shape.iter().product();
        let element_size = std::mem::size_of::<T>();
        let buffer_size = (size * element_size) as u64;

        // Create strides (row-major layout)
        let strides = crate::gpu::kernel::contiguous_strides(shape);

        // Create an empty buffer
        let buffer = context.create_empty_buffer(
            buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Ok(Self {
            context,
            buffer,
            shape: shape.to_vec(),
            strides,
            size,
            element_size,
            _phantom: PhantomData,
        })
    }

    /// Converts the GPU array back to a CPU array
    pub fn to_array(&self) -> Result<Array<T>> {
        // Create a staging buffer to read data from the GPU
        let staging_buffer = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("NumRS2 GPU Staging Buffer"),
                size: (self.size * self.element_size) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Copy data from the GPU buffer to the staging buffer
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("NumRS2 Copy Encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (self.size * self.element_size) as u64,
        );

        self.context
            .queue()
            .submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read the data
        let buffer_slice = staging_buffer.slice(..);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                NumRs2Error::RuntimeError(format!("Failed to create async runtime: {}", e))
            })?;

        // Create a temporary buffer to store the data from the staging buffer
        let mut data = vec![0; self.size * self.element_size];

        rt.block_on(async {
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                // The callback has no channel of its own to report a send
                // failure through; if the receiver was already dropped there
                // is nothing left to notify.
                let _ = tx.send(result);
            });

            self.context
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| {
                    NumRs2Error::RuntimeError(format!(
                        "GPU device poll failed during buffer mapping: {}",
                        e
                    ))
                })?;

            rx.receive()
                .await
                .ok_or_else(|| {
                    NumRs2Error::RuntimeError(
                        "Failed to receive buffer mapping result - channel closed".to_string(),
                    )
                })?
                .map_err(|e| {
                    NumRs2Error::RuntimeError(format!("Buffer mapping operation failed: {}", e))
                })?;

            // Copy the data from the staging buffer
            let mapped_data = buffer_slice.get_mapped_range().map_err(|e| {
                NumRs2Error::RuntimeError(format!("Failed to get mapped buffer range: {}", e))
            })?;
            data.copy_from_slice(&mapped_data);

            Ok::<(), NumRs2Error>(())
        })?;

        // Unmap the buffer
        staging_buffer.unmap();

        // Convert the raw bytes to the actual type and create a CPU array
        let typed_data: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        let array = Array::from_vec_shape(typed_data, &self.shape)?;

        Ok(array)
    }

    /// Returns the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of the array in elements
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the total number of elements in the array
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the element size in bytes
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Returns a reference to the GPU buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns a reference to the GPU context
    pub fn context(&self) -> &GpuContextRef {
        &self.context
    }

    /// Reshapes the GPU array to a new shape without copying data
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The new shape for the array
    ///
    /// # Returns
    ///
    /// A new GpuArray with the same data but different shape
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is incompatible with the current size
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Cannot reshape array of size {} to shape {:?} (size {})",
                self.size, new_shape, new_size
            )));
        }

        // Calculate new strides (row-major layout)
        let strides = crate::gpu::kernel::contiguous_strides(new_shape);

        Ok(Self {
            context: self.context.clone(),
            buffer: self.buffer.clone(),
            shape: new_shape.to_vec(),
            strides,
            size: self.size,
            element_size: self.element_size,
            _phantom: PhantomData,
        })
    }
}

impl<T> Clone for GpuArray<T> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
            element_size: self.element_size,
            _phantom: PhantomData,
        }
    }
}

impl<T> fmt::Debug for GpuArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuArray {{ shape: {:?}, size: {} }}",
            self.shape, self.size
        )
    }
}

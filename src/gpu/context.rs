//! GPU Context Management
//!
//! This module provides the GpuContext struct which manages the WGPU device and queue.
//! The context is required for creating and operating on GPU arrays.

use crate::error::{NumRs2Error, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Thread-safe reference to the GPU context
pub type GpuContextRef = Arc<GpuContext>;

/// Manages GPU device, queue, and other resources
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_modules: ShaderModules,
}

/// Stores compiled shader modules for reuse
struct ShaderModules {
    element_wise_f32: wgpu::ShaderModule,
    element_wise_f64: wgpu::ShaderModule,
    reduction_f32: wgpu::ShaderModule,
    reduction_f64: wgpu::ShaderModule,
    matmul_f32: wgpu::ShaderModule,
    matmul_f64: wgpu::ShaderModule,
}

impl GpuContext {
    /// Creates a new GPU context using the default adapter
    pub async fn new() -> Result<Self> {
        // Get an adapter that supports compute operations
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| {
                NumRs2Error::RuntimeError("Failed to find an appropriate GPU adapter".to_string())
            })?;

        // Get information about the adapter
        let info = adapter.get_info();
        println!("Selected GPU: {} ({:?})", info.name, info.backend);

        // Create the device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("NumRS2 GPU device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                NumRs2Error::RuntimeError(format!("Failed to create GPU device: {}", e))
            })?;

        // Load all the shader modules
        let shader_modules = Self::create_shader_modules(&device)?;

        Ok(Self {
            device,
            queue,
            shader_modules,
        })
    }

    /// Creates all the shader modules needed for GPU operations
    fn create_shader_modules(device: &wgpu::Device) -> Result<ShaderModules> {
        // Load element-wise operation shaders
        let element_wise_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/element_wise_f32.wgsl").into()),
        });

        let element_wise_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise F64 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/element_wise_f64.wgsl").into()),
        });

        // Load reduction operation shaders
        let reduction_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduction F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduction_f32.wgsl").into()),
        });

        let reduction_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduction F64 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduction_f64.wgsl").into()),
        });

        // Load matrix multiplication shaders
        let matmul_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_f32.wgsl").into()),
        });

        let matmul_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication F64 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_f64.wgsl").into()),
        });

        Ok(ShaderModules {
            element_wise_f32,
            element_wise_f64,
            reduction_f32,
            reduction_f64,
            matmul_f32,
            matmul_f64,
        })
    }

    /// Get a reference to the device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a reference to the element-wise shader for f32
    pub fn element_wise_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.element_wise_f32
    }

    /// Get a reference to the element-wise shader for f64
    pub fn element_wise_f64_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.element_wise_f64
    }

    /// Get a reference to the reduction shader for f32
    pub fn reduction_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.reduction_f32
    }

    /// Get a reference to the reduction shader for f64
    pub fn reduction_f64_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.reduction_f64
    }

    /// Get a reference to the matrix multiplication shader for f32
    pub fn matmul_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.matmul_f32
    }

    /// Get a reference to the matrix multiplication shader for f64
    pub fn matmul_f64_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.matmul_f64
    }

    /// Creates a GPU buffer with the given data
    pub fn create_buffer<T: bytemuck::Pod + bytemuck::Zeroable>(
        &self,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NumRS2 GPU Buffer"),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Creates an empty GPU buffer with the given size
    pub fn create_empty_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NumRS2 GPU Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Runs a GPU computation using the given compute pipeline and bind groups
    pub fn run_compute(
        &self,
        compute_pipeline: &wgpu::ComputePipeline,
        bind_groups: &[&wgpu::BindGroup],
        workgroup_count: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NumRS2 Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NumRS2 Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(compute_pipeline);

            for (i, bind_group) in bind_groups.iter().enumerate() {
                compute_pass.set_bind_group(i as u32, bind_group, &[]);
            }

            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Creates a new context with an async runtime
pub fn new_context() -> Result<GpuContextRef> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| NumRs2Error::RuntimeError(format!("Failed to create async runtime: {}", e)))?;

    let context = rt.block_on(GpuContext::new())?;
    Ok(Arc::new(context))
}

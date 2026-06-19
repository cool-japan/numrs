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
///
/// CACHE ALIGNMENT: Aligned to 64-byte cache lines for optimal GPU command submission.
/// The device and queue are accessed on every GPU operation, and cache alignment
/// ensures these hot fields are efficiently cached, reducing latency for GPU kernel
/// launches and data transfers. This is especially important for high-frequency
/// GPU operations where submission overhead can become a bottleneck.
#[repr(align(64))]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_modules: ShaderModules,
    /// Whether the underlying device exposes the `SHADER_F64` feature.
    ///
    /// WGSL `f64`/`array<f64>` shaders require the `wgpu::Features::SHADER_F64`
    /// device feature, which many GPUs do not support. When this is `false`,
    /// the f64 shader modules are not compiled and any attempt to run an f64
    /// GPU operation returns an error instead of silently producing wrong
    /// results from an f32 kernel.
    f64_supported: bool,
}

/// Stores compiled shader modules for reuse
///
/// The f64 modules are `None` when the device does not support the
/// `SHADER_F64` feature; in that case f64 GPU operations are rejected at
/// dispatch time rather than falling back to an f32 kernel.
struct ShaderModules {
    element_wise_f32: wgpu::ShaderModule,
    element_wise_f64: Option<wgpu::ShaderModule>,
    reduction_f32: wgpu::ShaderModule,
    reduction_f64: Option<wgpu::ShaderModule>,
    matmul_f32: wgpu::ShaderModule,
    matmul_f64: Option<wgpu::ShaderModule>,
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
            .map_err(|e| {
                NumRs2Error::RuntimeError(format!(
                    "Failed to find an appropriate GPU adapter: {}",
                    e
                ))
            })?;

        // Get information about the adapter
        let info = adapter.get_info();
        println!("Selected GPU: {} ({:?})", info.name, info.backend);

        // Determine whether the adapter advertises 64-bit floating point shader
        // support. WGSL `f64`/`array<f64>` kernels require the `SHADER_F64`
        // device feature, which is unavailable on many GPUs. We request only the
        // subset of `SHADER_F64` that the adapter actually exposes so that
        // device creation never fails because of an unsupported feature.
        let f64_supported = adapter.features().contains(wgpu::Features::SHADER_F64);
        let required_features = adapter.features() & wgpu::Features::SHADER_F64;

        // Create the device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("NumRS2 GPU device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| {
                NumRs2Error::RuntimeError(format!("Failed to create GPU device: {}", e))
            })?;

        // Load all the shader modules. The real f64 shaders are only compiled
        // when the device supports `SHADER_F64`.
        let shader_modules = Self::create_shader_modules(&device, f64_supported)?;

        Ok(Self {
            device,
            queue,
            shader_modules,
            f64_supported,
        })
    }

    /// Creates all the shader modules needed for GPU operations
    ///
    /// When `f64_supported` is `true`, the genuine f64 WGSL shaders are
    /// compiled. When it is `false`, the f64 modules are left uncompiled
    /// (`None`) so that f64 GPU operations fail with a clear error instead of
    /// silently running an f32 kernel and returning wrong results.
    fn create_shader_modules(device: &wgpu::Device, f64_supported: bool) -> Result<ShaderModules> {
        // Load element-wise operation shaders
        let element_wise_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/element_wise_f32.wgsl").into()),
        });

        // Compile the genuine f64 element-wise shader only when the device
        // supports 64-bit floating point shaders. Otherwise leave it
        // uncompiled so f64 operations are rejected rather than silently
        // running the f32 kernel.
        let element_wise_f64 = if f64_supported {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Element-wise F64 Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/element_wise_f64.wgsl").into(),
                ),
            }))
        } else {
            None
        };

        // Load reduction operation shaders
        let reduction_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduction F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduction_f32.wgsl").into()),
        });

        let reduction_f64 = if f64_supported {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Reduction F64 Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduction_f64.wgsl").into()),
            }))
        } else {
            None
        };

        // Load matrix multiplication shaders
        let matmul_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_f32.wgsl").into()),
        });

        let matmul_f64 = if f64_supported {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Matrix Multiplication F64 Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_f64.wgsl").into()),
            }))
        } else {
            None
        };

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

    /// Returns whether this context's device supports 64-bit floating point
    /// (`SHADER_F64`) GPU shaders.
    pub fn f64_supported(&self) -> bool {
        self.f64_supported
    }

    /// Get a reference to the element-wise shader for f64
    ///
    /// Returns an error when the device does not support the `SHADER_F64`
    /// feature, ensuring f64 operations never silently fall back to an f32
    /// kernel.
    pub fn element_wise_f64_shader(&self) -> Result<&wgpu::ShaderModule> {
        self.shader_modules
            .element_wise_f64
            .as_ref()
            .ok_or_else(Self::f64_unsupported_error)
    }

    /// Get a reference to the reduction shader for f32
    pub fn reduction_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.reduction_f32
    }

    /// Get a reference to the reduction shader for f64
    ///
    /// Returns an error when the device does not support the `SHADER_F64`
    /// feature, ensuring f64 operations never silently fall back to an f32
    /// kernel.
    pub fn reduction_f64_shader(&self) -> Result<&wgpu::ShaderModule> {
        self.shader_modules
            .reduction_f64
            .as_ref()
            .ok_or_else(Self::f64_unsupported_error)
    }

    /// Get a reference to the matrix multiplication shader for f32
    pub fn matmul_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.matmul_f32
    }

    /// Get a reference to the matrix multiplication shader for f64
    ///
    /// Returns an error when the device does not support the `SHADER_F64`
    /// feature, ensuring f64 operations never silently fall back to an f32
    /// kernel.
    pub fn matmul_f64_shader(&self) -> Result<&wgpu::ShaderModule> {
        self.shader_modules
            .matmul_f64
            .as_ref()
            .ok_or_else(Self::f64_unsupported_error)
    }

    /// Builds the error returned when an f64 GPU operation is requested on a
    /// device that does not support the `SHADER_F64` feature.
    fn f64_unsupported_error() -> NumRs2Error {
        NumRs2Error::FeatureNotEnabled(
            "f64 GPU operations require the wgpu `SHADER_F64` device feature, \
             which is not supported by the selected GPU adapter. Use f32 GPU \
             arrays or run the computation on the CPU."
                .to_string(),
        )
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
                compute_pass.set_bind_group(i as u32, *bind_group, &[]);
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

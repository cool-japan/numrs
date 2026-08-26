//! GPU Context Management
//!
//! This module provides the GpuContext struct which manages the WGPU device and queue.
//! The context is required for creating and operating on GPU arrays.

use crate::error::{NumRs2Error, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Thread-safe reference to the GPU context
pub type GpuContextRef = Arc<GpuContext>;

/// Environment variable that opts into a software (fallback) adapter.
///
/// Setting `NUMRS2_GPU_FALLBACK=1` makes [`GpuContext::new`] request an
/// adapter with `force_fallback_adapter: true`, which selects the platform's
/// software rasteriser (for example lavapipe on Linux or WARP on Windows)
/// instead of a physical GPU. That allows the GPU test suite to run on
/// machines that have no usable GPU at all, at the cost of speed.
///
/// Note that a fallback adapter only exists when the platform provides one;
/// macOS/Metal has no software adapter, so on macOS the variable makes
/// context creation fail rather than silently using the real GPU.
pub const FALLBACK_ENV_VAR: &str = "NUMRS2_GPU_FALLBACK";

/// Returns whether a software (fallback) adapter has been requested through
/// the [`FALLBACK_ENV_VAR`] environment variable.
pub fn fallback_adapter_requested() -> bool {
    match std::env::var(FALLBACK_ENV_VAR) {
        Ok(value) => {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true")
        }
        Err(_) => false,
    }
}

/// Builds the adapter options used by every entry point in this crate.
pub(crate) fn adapter_options(
    force_fallback: bool,
) -> wgpu::RequestAdapterOptions<'static, 'static> {
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: force_fallback,
        compatible_surface: None,
        // Limit bucketing exists to reduce GPU fingerprinting when
        // `wgpu` is exposed to untrusted web content; this is a
        // native compute library with no such surface, so it is
        // unneeded here (matches `wgpu`'s own `Default` behavior).
        apply_limit_buckets: false,
    }
}

/// Replaces the scalar placeholders of a WGSL template with a concrete type.
///
/// The shader templates in `shaders/` are written once and instantiated for
/// every supported scalar type, so that the f32 and f64 kernels can never
/// drift apart.
///
/// The placeholders are:
///
/// * `SCALAR` - the WGSL type name,
/// * `NEG_LIMIT` / `POS_LIMIT` - the finite extrema of that type, used as the
///   identity elements of `max` and `min`,
/// * `IS_NAN_EXPR` - a NaN test on the parameter `x`. For f32 it inspects the
///   bit pattern rather than writing `x != x`, because backends that compile
///   with fast-math relaxations (Metal among them) fold the comparison away
///   and would stop propagating NaN the way NumPy does.
fn instantiate_template(template: &str, substitutions: &[(&str, &str)]) -> String {
    let mut source = template.to_string();
    for (placeholder, value) in substitutions {
        source = source.replace(placeholder, value);
    }
    source
}

/// Template substitutions producing the f32 variant of a shader.
const F32_SUBSTITUTIONS: &[(&str, &str)] = &[
    ("SCALAR", "f32"),
    ("NEG_LIMIT", "-3.4028234663852886e38"),
    ("POS_LIMIT", "3.4028234663852886e38"),
    (
        "IS_NAN_EXPR",
        "(bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u",
    ),
];

/// Template substitutions producing the f64 variant of a shader.
///
/// f64 has no `bitcast` to a single integer in WGSL, so the NaN test stays a
/// comparison; devices that expose `SHADER_F64` are Vulkan-class and do not
/// apply the fast-math folding that makes this unreliable on Metal.
const F64_SUBSTITUTIONS: &[(&str, &str)] = &[
    ("SCALAR", "f64"),
    ("NEG_LIMIT", "-1.7976931348623157e308"),
    ("POS_LIMIT", "1.7976931348623157e308"),
    ("IS_NAN_EXPR", "x != x"),
];

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
    broadcast_f32: wgpu::ShaderModule,
    broadcast_f64: Option<wgpu::ShaderModule>,
    /// Type-agnostic strided gather (transpose / permutation / slicing).
    gather: wgpu::ShaderModule,
    /// Type-agnostic im2col patch-matrix materialisation.
    im2col: wgpu::ShaderModule,
}

impl GpuContext {
    /// Creates a new GPU context using the default adapter
    ///
    /// When the [`FALLBACK_ENV_VAR`] environment variable is set, a software
    /// (fallback) adapter is requested instead of a physical GPU.
    pub async fn new() -> Result<Self> {
        Self::with_fallback_adapter(fallback_adapter_requested()).await
    }

    /// Creates a new GPU context, optionally forcing a software adapter
    ///
    /// `force_fallback` maps directly onto wgpu's `force_fallback_adapter`
    /// option: when it is `true` only a software rasteriser is considered,
    /// which makes GPU code runnable (slowly) on machines without a usable
    /// GPU. Platforms that do not ship a software adapter - macOS/Metal in
    /// particular - fail to find one instead of falling back to the real GPU.
    pub async fn with_fallback_adapter(force_fallback: bool) -> Result<Self> {
        // Get an adapter that supports compute operations
        let adapter = wgpu::Instance::default()
            .request_adapter(&adapter_options(force_fallback))
            .await
            .map_err(|e| {
                if force_fallback {
                    NumRs2Error::RuntimeError(format!(
                        "Failed to find a software (fallback) GPU adapter requested through {}=1: {}",
                        FALLBACK_ENV_VAR, e
                    ))
                } else {
                    NumRs2Error::RuntimeError(format!(
                        "Failed to find an appropriate GPU adapter: {}",
                        e
                    ))
                }
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

        // Load reduction operation shaders. Both precisions are instantiated
        // from one template so the f32 and f64 kernels cannot drift apart.
        let reduction_template = include_str!("shaders/reduction_template.wgsl");
        let reduction_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduction F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(
                instantiate_template(reduction_template, F32_SUBSTITUTIONS).into(),
            ),
        });

        let reduction_f64 = if f64_supported {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Reduction F64 Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    instantiate_template(reduction_template, F64_SUBSTITUTIONS).into(),
                ),
            }))
        } else {
            None
        };

        // Broadcasting element-wise binary operations, same template scheme.
        let broadcast_template = include_str!("shaders/broadcast_template.wgsl");
        let broadcast_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Broadcast F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(
                instantiate_template(broadcast_template, F32_SUBSTITUTIONS).into(),
            ),
        });

        let broadcast_f64 = if f64_supported {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Broadcast F64 Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    instantiate_template(broadcast_template, F64_SUBSTITUTIONS).into(),
                ),
            }))
        } else {
            None
        };

        // Data-movement kernels. These copy raw 32-bit words and perform no
        // arithmetic on the payload, so a single module serves every element
        // type whose size is a multiple of four bytes.
        let gather = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strided Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gather_words.wgsl").into()),
        });

        let im2col = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("im2col Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/im2col_words.wgsl").into()),
        });

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
            broadcast_f32,
            broadcast_f64,
            gather,
            im2col,
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

    /// Get a reference to the broadcasting binary-operation shader for f32
    pub fn broadcast_f32_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.broadcast_f32
    }

    /// Get a reference to the broadcasting binary-operation shader for f64
    ///
    /// Returns an error when the device does not support the `SHADER_F64`
    /// feature, ensuring f64 operations never silently fall back to an f32
    /// kernel.
    pub fn broadcast_f64_shader(&self) -> Result<&wgpu::ShaderModule> {
        self.shader_modules
            .broadcast_f64
            .as_ref()
            .ok_or_else(Self::f64_unsupported_error)
    }

    /// Get a reference to the type-agnostic strided gather shader
    ///
    /// The gather kernel moves raw 32-bit words and therefore serves every
    /// element type whose size is a multiple of four bytes, including f64 on
    /// devices without the `SHADER_F64` feature.
    pub fn gather_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.gather
    }

    /// Get a reference to the type-agnostic im2col shader
    pub fn im2col_shader(&self) -> &wgpu::ShaderModule {
        &self.shader_modules.im2col
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

/// How the current thread relates to a Tokio runtime, as seen by
/// [`new_context`] and other synchronous entry points that need to drive an
/// `async fn` to completion without risking a nested-runtime panic.
///
/// Tokio forbids starting a runtime (or calling
/// [`Runtime::block_on`](tokio::runtime::Runtime::block_on)) on a thread that
/// is already being driven by one - doing so panics with "Cannot start a
/// runtime from within a runtime" instead of deadlocking, which is exactly
/// the failure this type exists to route around: every caller of
/// [`runtime_access`] matches on the three cases below instead of
/// unconditionally building a private runtime.
pub(crate) enum RuntimeAccess {
    /// No Tokio runtime is driving this thread. It is safe to build a
    /// private one and block on it directly.
    None,
    /// A multi-thread Tokio runtime is driving this thread. Blocking here
    /// directly would still stall that runtime's worker, but
    /// [`tokio::task::block_in_place`] hands this worker's other tasks off
    /// to the remaining workers first, so blocking through the returned
    /// [`Handle`](tokio::runtime::Handle) is safe.
    MultiThread(tokio::runtime::Handle),
    /// A current-thread Tokio runtime is driving this thread (the default
    /// flavor for `#[tokio::test]`). There is no other worker to hand
    /// blocking work off to - `block_in_place` itself panics on this
    /// flavor - so there is no safe way to block here at all. Callers
    /// should refuse (or, for `Option`-returning APIs, report "unavailable")
    /// rather than nest a runtime; async code should drive the future
    /// directly with `.await` instead of going through a sync entry point.
    CurrentThread,
}

/// Classifies the current thread's relationship to a Tokio runtime; see
/// [`RuntimeAccess`] for what each case means and how to act on it.
pub(crate) fn runtime_access() -> RuntimeAccess {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            RuntimeAccess::MultiThread(handle)
        }
        Ok(_) => RuntimeAccess::CurrentThread,
        Err(_) => RuntimeAccess::None,
    }
}

/// Creates a new context, driving the adapter/device request with a plain
/// `.await` on the caller's own async executor.
///
/// This is the constructor to use from `async` code, including
/// `#[tokio::test]` bodies (which default to a single-threaded runtime):
/// unlike [`new_context`], it never builds a runtime of its own, so it
/// cannot nest one inside the caller's and panic.
pub async fn new_context_async() -> Result<GpuContextRef> {
    GpuContext::new().await.map(Arc::new)
}

/// Creates a new context, blocking the calling thread until it is ready.
///
/// This is the constructor for **synchronous, non-async** callers - a plain
/// `fn main`, a `#[test]`, a benchmark. It has no async executor of its own
/// to run on, so by default it spins up a small private Tokio runtime and
/// blocks on it to perform the adapter/device request.
///
/// Calling this from code that is already running on a Tokio runtime is
/// handled rather than left to panic (see `RuntimeAccess`, this module's
/// internal classification of the three cases):
///
/// * on a **multi-thread** runtime the request is round-tripped through
///   [`tokio::task::block_in_place`], which is safe because that runtime has
///   other workers to pick up the slack;
/// * on a **current-thread** runtime - the default `#[tokio::test]` flavor -
///   blocking is never safe (there are no other workers), so this returns a
///   clear [`NumRs2Error::RuntimeError`] instructing the caller to use
///   [`new_context_async`] instead, rather than nesting a runtime and
///   panicking or silently deadlocking.
pub fn new_context() -> Result<GpuContextRef> {
    match runtime_access() {
        RuntimeAccess::MultiThread(handle) => {
            tokio::task::block_in_place(|| handle.block_on(GpuContext::new())).map(Arc::new)
        }
        RuntimeAccess::CurrentThread => Err(NumRs2Error::RuntimeError(
            "new_context() was called from within a single-threaded Tokio runtime \
             (the default #[tokio::test] flavor); it cannot safely block this thread \
             to wait for the adapter/device request without nesting a second runtime \
             inside the caller's, which Tokio forbids. Call \
             `new_context_async().await` instead from async code, or annotate the \
             test with #[tokio::test(flavor = \"multi_thread\")]."
                .to_string(),
        )),
        RuntimeAccess::None => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    NumRs2Error::RuntimeError(format!("Failed to create async runtime: {}", e))
                })?;
            rt.block_on(GpuContext::new()).map(Arc::new)
        }
    }
}

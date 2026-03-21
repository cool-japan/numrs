//! Advanced Compute Shader Management
//!
//! This module provides advanced shader compilation, caching, and composition
//! capabilities for GPU compute operations. It enables building complex
//! operations from simpler kernels and optimizes shader reuse.
//!
//! ## Features
//!
//! - **Shader Compilation**: Compile WGSL shaders with validation
//! - **Pipeline Caching**: Cache compiled pipelines for reuse
//! - **Kernel Composition**: Chain multiple operations efficiently
//! - **Specialization**: Generate specialized shaders for specific types
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::gpu::compute::{ShaderCache, KernelBuilder};
//!
//! # #[cfg(feature = "gpu")]
//! # fn example() -> numrs2::error::Result<()> {
//! let context = numrs2::gpu::new_context()?;
//! let cache = ShaderCache::new(context.clone());
//!
//! // Build a composite kernel
//! let kernel = KernelBuilder::new()
//!     .add_operation("add")
//!     .add_operation("sqrt")
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use crate::error::{NumRs2Error, Result};
use crate::gpu::context::GpuContextRef;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache for compiled shader modules and compute pipelines
pub struct ShaderCache {
    context: GpuContextRef,
    shader_modules: Arc<Mutex<HashMap<String, wgpu::ShaderModule>>>,
    pipelines: Arc<Mutex<HashMap<String, wgpu::ComputePipeline>>>,
}

impl ShaderCache {
    /// Creates a new shader cache
    pub fn new(context: GpuContextRef) -> Self {
        Self {
            context,
            shader_modules: Arc::new(Mutex::new(HashMap::new())),
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Compiles and caches a shader module from WGSL source
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this shader
    /// * `source` - WGSL shader source code
    ///
    /// # Returns
    ///
    /// Reference to the compiled shader module
    pub fn compile_shader(&self, name: &str, source: &str) -> Result<()> {
        let mut modules = self.shader_modules.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock shader cache: {}", e))
        })?;

        if !modules.contains_key(name) {
            let module = self
                .context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            modules.insert(name.to_string(), module);
        }

        Ok(())
    }

    /// Gets a cached shader module
    pub fn get_shader(&self, name: &str) -> Result<Option<wgpu::ShaderModule>> {
        let modules = self.shader_modules.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock shader cache: {}", e))
        })?;

        Ok(modules.get(name).cloned())
    }

    /// Caches a compute pipeline
    pub fn cache_pipeline(&self, name: &str, pipeline: wgpu::ComputePipeline) -> Result<()> {
        let mut pipelines = self.pipelines.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock pipeline cache: {}", e))
        })?;

        pipelines.insert(name.to_string(), pipeline);
        Ok(())
    }

    /// Gets a cached compute pipeline
    pub fn get_pipeline(&self, name: &str) -> Result<Option<wgpu::ComputePipeline>> {
        let pipelines = self.pipelines.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock pipeline cache: {}", e))
        })?;

        Ok(pipelines.get(name).cloned())
    }

    /// Clears all cached shaders and pipelines
    pub fn clear(&self) -> Result<()> {
        let mut modules = self.shader_modules.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock shader cache: {}", e))
        })?;

        let mut pipelines = self.pipelines.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock pipeline cache: {}", e))
        })?;

        modules.clear();
        pipelines.clear();

        Ok(())
    }

    /// Returns the number of cached shader modules
    pub fn shader_count(&self) -> Result<usize> {
        let modules = self.shader_modules.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock shader cache: {}", e))
        })?;

        Ok(modules.len())
    }

    /// Returns the number of cached pipelines
    pub fn pipeline_count(&self) -> Result<usize> {
        let pipelines = self.pipelines.lock().map_err(|e| {
            NumRs2Error::RuntimeError(format!("Failed to lock pipeline cache: {}", e))
        })?;

        Ok(pipelines.len())
    }
}

/// Kernel operation types for composition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOp {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Subtract,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise division
    Divide,
    /// Element-wise exponential
    Exp,
    /// Element-wise logarithm
    Log,
    /// Element-wise square root
    Sqrt,
    /// Element-wise sine
    Sin,
    /// Element-wise cosine
    Cos,
    /// Element-wise absolute value
    Abs,
    /// Element-wise negation
    Neg,
}

impl KernelOp {
    /// Returns the WGSL code for this operation
    pub fn to_wgsl(&self, input_var: &str, output_var: &str) -> String {
        match self {
            KernelOp::Add => format!("{} = {} + input_b[idx]", output_var, input_var),
            KernelOp::Subtract => format!("{} = {} - input_b[idx]", output_var, input_var),
            KernelOp::Multiply => format!("{} = {} * input_b[idx]", output_var, input_var),
            KernelOp::Divide => format!("{} = {} / input_b[idx]", output_var, input_var),
            KernelOp::Exp => format!("{} = exp({})", output_var, input_var),
            KernelOp::Log => format!("{} = log({})", output_var, input_var),
            KernelOp::Sqrt => format!("{} = sqrt({})", output_var, input_var),
            KernelOp::Sin => format!("{} = sin({})", output_var, input_var),
            KernelOp::Cos => format!("{} = cos({})", output_var, input_var),
            KernelOp::Abs => format!("{} = abs({})", output_var, input_var),
            KernelOp::Neg => format!("{} = -({})", output_var, input_var),
        }
    }

    /// Returns whether this operation requires two inputs
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            KernelOp::Add | KernelOp::Subtract | KernelOp::Multiply | KernelOp::Divide
        )
    }
}

/// Builder for composing multiple kernel operations into a single shader
pub struct KernelBuilder {
    operations: Vec<KernelOp>,
    data_type: DataType,
}

/// Supported data types for kernel operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

impl DataType {
    /// Returns the WGSL type name
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        }
    }
}

impl KernelBuilder {
    /// Creates a new kernel builder with default f32 data type
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            data_type: DataType::F32,
        }
    }

    /// Sets the data type for this kernel
    pub fn with_data_type(mut self, data_type: DataType) -> Self {
        self.data_type = data_type;
        self
    }

    /// Adds an operation to the kernel composition
    pub fn add_operation(mut self, op: KernelOp) -> Self {
        self.operations.push(op);
        self
    }

    /// Builds the composed kernel shader source
    pub fn build(&self) -> Result<String> {
        if self.operations.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot build kernel with no operations".to_string(),
            ));
        }

        let dtype = self.data_type.to_wgsl();

        // Count binary operations to determine buffer requirements
        let binary_count = self.operations.iter().filter(|op| op.is_binary()).count();

        // Build shader header
        let mut shader = format!(
            r#"
// Composite kernel with {} operations
struct Params {{
    array_size: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}}

@group(0) @binding(0) var<storage, read> input_a: array<{}>;
"#,
            self.operations.len(),
            dtype
        );

        // Add additional input buffers for binary operations
        if binary_count > 0 {
            shader.push_str(&format!(
                "@group(0) @binding(1) var<storage, read> input_b: array<{}>;\n",
                dtype
            ));
            shader.push_str(&format!(
                "@group(0) @binding(2) var<storage, read_write> output: array<{}>;\n",
                dtype
            ));
            shader.push_str("@group(0) @binding(3) var<uniform> params: Params;\n");
        } else {
            shader.push_str(&format!(
                "@group(0) @binding(1) var<storage, read_write> output: array<{}>;\n",
                dtype
            ));
            shader.push_str("@group(0) @binding(2) var<uniform> params: Params;\n");
        }

        // Build compute function
        shader.push_str(
            r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.array_size) {
        return;
    }

    var temp = input_a[idx];
"#,
        );

        // Add operations in sequence
        for op in self.operations.iter() {
            let input_var = "temp";
            let output_var = "temp";
            shader.push_str("    ");
            shader.push_str(&op.to_wgsl(input_var, output_var));
            shader.push_str(";\n");
        }

        // Write final result
        shader.push_str(
            r#"
    output[idx] = temp;
}
"#,
        );

        Ok(shader)
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline builder for creating compute pipelines with custom configurations
pub struct PipelineBuilder {
    context: GpuContextRef,
    shader_module: Option<wgpu::ShaderModule>,
    entry_point: String,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl PipelineBuilder {
    /// Creates a new pipeline builder
    pub fn new(context: GpuContextRef) -> Self {
        Self {
            context,
            shader_module: None,
            entry_point: "main".to_string(),
            bind_group_layout: None,
        }
    }

    /// Sets the shader module
    pub fn with_shader(mut self, shader: wgpu::ShaderModule) -> Self {
        self.shader_module = Some(shader);
        self
    }

    /// Sets the entry point function name
    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    /// Sets the bind group layout
    pub fn with_bind_group_layout(mut self, layout: wgpu::BindGroupLayout) -> Self {
        self.bind_group_layout = Some(layout);
        self
    }

    /// Builds the compute pipeline
    pub fn build(self) -> Result<wgpu::ComputePipeline> {
        let shader_module = self.shader_module.ok_or_else(|| {
            NumRs2Error::InvalidOperation("No shader module provided".to_string())
        })?;

        let bind_group_layout = self.bind_group_layout.ok_or_else(|| {
            NumRs2Error::InvalidOperation("No bind group layout provided".to_string())
        })?;

        let pipeline_layout =
            self.context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Composite Pipeline Layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            self.context
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Composite Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(&self.entry_point),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_op_to_wgsl() {
        assert_eq!(KernelOp::Add.to_wgsl("a", "b"), "b = a + input_b[idx]");
        assert_eq!(KernelOp::Exp.to_wgsl("a", "b"), "b = exp(a)");
        assert_eq!(KernelOp::Sin.to_wgsl("a", "b"), "b = sin(a)");
    }

    #[test]
    fn test_kernel_op_is_binary() {
        assert!(KernelOp::Add.is_binary());
        assert!(KernelOp::Multiply.is_binary());
        assert!(!KernelOp::Exp.is_binary());
        assert!(!KernelOp::Sin.is_binary());
    }

    #[test]
    fn test_data_type_to_wgsl() {
        assert_eq!(DataType::F32.to_wgsl(), "f32");
        assert_eq!(DataType::F64.to_wgsl(), "f64");
    }

    #[test]
    fn test_kernel_builder_empty() {
        let builder = KernelBuilder::new();
        assert!(builder.build().is_err());
    }

    #[test]
    fn test_kernel_builder_single_op() {
        let builder = KernelBuilder::new().add_operation(KernelOp::Exp);
        let shader = builder.build();
        assert!(shader.is_ok());
        let shader_src = shader.expect("Shader build failed");
        assert!(shader_src.contains("exp("));
    }

    #[test]
    fn test_kernel_builder_multiple_ops() {
        let builder = KernelBuilder::new()
            .add_operation(KernelOp::Add)
            .add_operation(KernelOp::Sqrt)
            .add_operation(KernelOp::Exp);
        let shader = builder.build();
        assert!(shader.is_ok());
        let shader_src = shader.expect("Shader build failed");
        assert!(shader_src.contains("sqrt("));
        assert!(shader_src.contains("exp("));
    }

    #[test]
    fn test_kernel_builder_with_data_type() {
        let builder = KernelBuilder::new()
            .with_data_type(DataType::F64)
            .add_operation(KernelOp::Sin);
        let shader = builder.build();
        assert!(shader.is_ok());
        let shader_src = shader.expect("Shader build failed");
        assert!(shader_src.contains("f64"));
    }
}

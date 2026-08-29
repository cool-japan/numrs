//! Tests for GPU compute shader management

use numrs2::gpu::compute::{DataType, KernelBuilder, KernelOp, ShaderCache};
use numrs2::gpu::new_context_async;

#[tokio::test]
async fn test_shader_cache_creation() {
    let context = new_context_async()
        .await
        .expect("Failed to create GPU context");
    let cache = ShaderCache::new(context);

    assert_eq!(cache.shader_count().ok(), Some(0));
    assert_eq!(cache.pipeline_count().ok(), Some(0));
}

#[tokio::test]
async fn test_shader_compilation() {
    let context = new_context_async()
        .await
        .expect("Failed to create GPU context");
    let cache = ShaderCache::new(context);

    let shader_source = r#"
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            // Simple test shader
        }
    "#;

    cache
        .compile_shader("test_shader", shader_source)
        .expect("Failed to compile shader");

    assert_eq!(cache.shader_count().ok(), Some(1));

    let shader = cache
        .get_shader("test_shader")
        .expect("Failed to get shader");
    assert!(shader.is_some());
}

#[tokio::test]
async fn test_shader_cache_get_nonexistent() {
    let context = new_context_async()
        .await
        .expect("Failed to create GPU context");
    let cache = ShaderCache::new(context);

    let shader = cache
        .get_shader("nonexistent")
        .expect("Failed to query shader");
    assert!(shader.is_none());
}

#[tokio::test]
async fn test_shader_cache_clear() {
    let context = new_context_async()
        .await
        .expect("Failed to create GPU context");
    let cache = ShaderCache::new(context.clone());

    // Add a shader
    let shader_source = r#"
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {}
    "#;

    cache
        .compile_shader("test", shader_source)
        .expect("Failed to compile shader");
    assert_eq!(cache.shader_count().ok(), Some(1));

    // Clear cache
    cache.clear().expect("Failed to clear cache");
    assert_eq!(cache.shader_count().ok(), Some(0));
}

#[tokio::test]
async fn test_kernel_builder_single_unary_op() {
    let builder = KernelBuilder::new().add_operation(KernelOp::Exp);
    let shader = builder.build().expect("Failed to build kernel");

    assert!(shader.contains("exp("));
    assert!(shader.contains("f32")); // Default data type
    assert!(shader.contains("@compute"));
    assert!(shader.contains("@workgroup_size"));
}

#[tokio::test]
async fn test_kernel_builder_single_binary_op() {
    let builder = KernelBuilder::new().add_operation(KernelOp::Add);
    let shader = builder.build().expect("Failed to build kernel");

    assert!(shader.contains("input_a"));
    assert!(shader.contains("input_b"));
    assert!(shader.contains("output"));
}

#[tokio::test]
async fn test_kernel_builder_multiple_ops() {
    let builder = KernelBuilder::new()
        .add_operation(KernelOp::Add)
        .add_operation(KernelOp::Sqrt)
        .add_operation(KernelOp::Exp);

    let shader = builder.build().expect("Failed to build kernel");

    assert!(shader.contains("sqrt("));
    assert!(shader.contains("exp("));
    assert!(shader.contains("// Composite kernel with 3 operations"));
}

#[tokio::test]
async fn test_kernel_builder_with_f64() {
    let builder = KernelBuilder::new()
        .with_data_type(DataType::F64)
        .add_operation(KernelOp::Sin);

    let shader = builder.build().expect("Failed to build kernel");
    assert!(shader.contains("f64"));
    assert!(!shader.contains("f32"));
}

#[tokio::test]
async fn test_kernel_builder_empty_fails() {
    let builder = KernelBuilder::new();
    let result = builder.build();
    assert!(result.is_err());
}

#[tokio::test]
async fn test_kernel_builder_all_unary_ops() {
    let unary_ops = vec![
        KernelOp::Exp,
        KernelOp::Log,
        KernelOp::Sqrt,
        KernelOp::Sin,
        KernelOp::Cos,
        KernelOp::Abs,
        KernelOp::Neg,
    ];

    for op in unary_ops {
        let builder = KernelBuilder::new().add_operation(op);
        let shader = builder.build();
        assert!(
            shader.is_ok(),
            "Failed to build shader for operation {:?}",
            op
        );
    }
}

#[tokio::test]
async fn test_kernel_builder_all_binary_ops() {
    let binary_ops = vec![
        KernelOp::Add,
        KernelOp::Subtract,
        KernelOp::Multiply,
        KernelOp::Divide,
    ];

    for op in binary_ops {
        let builder = KernelBuilder::new().add_operation(op);
        let shader = builder.build();
        assert!(
            shader.is_ok(),
            "Failed to build shader for operation {:?}",
            op
        );
    }
}

#[tokio::test]
async fn test_kernel_composition_complex() {
    let builder = KernelBuilder::new()
        .add_operation(KernelOp::Add) // a + b
        .add_operation(KernelOp::Sqrt) // sqrt(a + b)
        .add_operation(KernelOp::Exp) // exp(sqrt(a + b))
        .add_operation(KernelOp::Log); // log(exp(sqrt(a + b)))

    let shader = builder.build().expect("Failed to build complex kernel");

    // Verify all operations are present
    assert!(shader.contains("sqrt("));
    assert!(shader.contains("exp("));
    assert!(shader.contains("log("));
    assert!(shader.contains("// Composite kernel with 4 operations"));
}

#[tokio::test]
async fn test_data_type_wgsl_conversion() {
    assert_eq!(DataType::F32.to_wgsl(), "f32");
    assert_eq!(DataType::F64.to_wgsl(), "f64");
}

#[tokio::test]
async fn test_kernel_op_is_binary() {
    assert!(KernelOp::Add.is_binary());
    assert!(KernelOp::Subtract.is_binary());
    assert!(KernelOp::Multiply.is_binary());
    assert!(KernelOp::Divide.is_binary());

    assert!(!KernelOp::Exp.is_binary());
    assert!(!KernelOp::Log.is_binary());
    assert!(!KernelOp::Sin.is_binary());
    assert!(!KernelOp::Cos.is_binary());
}

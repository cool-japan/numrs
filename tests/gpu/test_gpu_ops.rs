//! Tests for GPU operations

use numrs2::array::Array;
// `ops` is a private module inside `numrs2::gpu` -- its functions (`add`,
// `subtract`, `exp`, `transpose`, `sum_f32`, ...) are re-exported at the
// `gpu` root via `pub use ops::*`, not reachable as `gpu::ops::*`. Glob-import
// the root instead of naming the (inaccessible) module.
use numrs2::gpu::*;

#[tokio::test]
async fn test_add() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = add(&a_gpu, &b_gpu).expect("Failed to add");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[4]);
    assert!((c.get(&[0]).expect("Invalid index") - 6.0).abs() < 1e-5);
    assert!((c.get(&[1]).expect("Invalid index") - 8.0).abs() < 1e-5);
    assert!((c.get(&[2]).expect("Invalid index") - 10.0).abs() < 1e-5);
    assert!((c.get(&[3]).expect("Invalid index") - 12.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_subtract() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![10.0f32, 8.0, 6.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = subtract(&a_gpu, &b_gpu).expect("Failed to subtract");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[4]);
    assert!((c.get(&[0]).expect("Invalid index") - 9.0).abs() < 1e-5);
    assert!((c.get(&[1]).expect("Invalid index") - 6.0).abs() < 1e-5);
    assert!((c.get(&[2]).expect("Invalid index") - 3.0).abs() < 1e-5);
    assert!((c.get(&[3]).expect("Invalid index") - 0.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_multiply() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![2.0f32, 3.0, 4.0, 5.0]).reshape(&[4]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = multiply(&a_gpu, &b_gpu).expect("Failed to multiply");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[4]);
    assert!((c.get(&[0]).expect("Invalid index") - 2.0).abs() < 1e-5);
    assert!((c.get(&[1]).expect("Invalid index") - 6.0).abs() < 1e-5);
    assert!((c.get(&[2]).expect("Invalid index") - 12.0).abs() < 1e-5);
    assert!((c.get(&[3]).expect("Invalid index") - 20.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_divide() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![10.0f32, 20.0, 30.0, 40.0]).reshape(&[4]);
    let b = Array::from_vec(vec![2.0f32, 4.0, 5.0, 8.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = divide(&a_gpu, &b_gpu).expect("Failed to divide");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[4]);
    assert!((c.get(&[0]).expect("Invalid index") - 5.0).abs() < 1e-5);
    assert!((c.get(&[1]).expect("Invalid index") - 5.0).abs() < 1e-5);
    assert!((c.get(&[2]).expect("Invalid index") - 6.0).abs() < 1e-5);
    assert!((c.get(&[3]).expect("Invalid index") - 5.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_gpu_array_creation() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    assert_eq!(a_gpu.shape(), &[2, 3]);
    assert_eq!(a_gpu.size(), 6);
}

#[tokio::test]
async fn test_gpu_array_round_trip() {
    let context = new_context().expect("Failed to create GPU context");

    let original = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let gpu_array = GpuArray::from_array_with_context(&original, context.clone())
        .expect("Failed to create GPU array");

    let result = gpu_array.to_array().expect("Failed to convert back to CPU");

    assert_eq!(result.shape(), original.shape());
    for i in 0..2 {
        for j in 0..3 {
            let orig_val = original.get(&[i, j]).expect("Invalid index");
            let result_val = result.get(&[i, j]).expect("Invalid index");
            assert!((orig_val - result_val).abs() < 1e-5);
        }
    }
}

#[tokio::test]
async fn test_gpu_array_new_with_shape() {
    let context = new_context().expect("Failed to create GPU context");

    let gpu_array = GpuArray::<f32>::new_with_shape(&[3, 4], context)
        .expect("Failed to create GPU array with shape");

    assert_eq!(gpu_array.shape(), &[3, 4]);
    assert_eq!(gpu_array.size(), 12);
}

#[tokio::test]
async fn test_incompatible_shapes() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0]).reshape(&[3]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    // Operations on arrays with different sizes should fail
    let result = add(&a_gpu, &b_gpu);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multidimensional_arrays() {
    let context = new_context().expect("Failed to create GPU context");

    // Create 2D arrays
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = add(&a_gpu, &b_gpu).expect("Failed to add");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[2, 2]);
    assert!((c.get(&[0, 0]).expect("Invalid index") - 6.0).abs() < 1e-5);
    assert!((c.get(&[0, 1]).expect("Invalid index") - 8.0).abs() < 1e-5);
    assert!((c.get(&[1, 0]).expect("Invalid index") - 10.0).abs() < 1e-5);
    assert!((c.get(&[1, 1]).expect("Invalid index") - 12.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_large_array() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a larger array to test GPU performance benefits
    let size = 1000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    let a = Array::from_vec(data_a.clone()).reshape(&[size]);
    let b = Array::from_vec(data_b.clone()).reshape(&[size]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = add(&a_gpu, &b_gpu).expect("Failed to add");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    assert_eq!(c.shape(), &[size]);

    // Verify a few values
    for i in [0, size / 2, size - 1] {
        let expected = data_a[i] + data_b[i];
        let actual = c.get(&[i]).expect("Invalid index");
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[tokio::test]
async fn test_unary_operations() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    // Test exp
    let exp_result = exp(&a_gpu).expect("Failed to compute exp");
    let exp_cpu = exp_result.to_array().expect("Failed to convert to CPU");
    assert!((exp_cpu.get(&[0]).expect("Invalid index") - 1.0f32.exp()).abs() < 1e-5);

    // Test sqrt
    let sqrt_result = sqrt(&a_gpu).expect("Failed to compute sqrt");
    let sqrt_cpu = sqrt_result.to_array().expect("Failed to convert to CPU");
    assert!((sqrt_cpu.get(&[0]).expect("Invalid index") - 1.0f32.sqrt()).abs() < 1e-5);

    // Test sin
    let sin_result = sin(&a_gpu).expect("Failed to compute sin");
    let sin_cpu = sin_result.to_array().expect("Failed to convert to CPU");
    assert!((sin_cpu.get(&[0]).expect("Invalid index") - 1.0f32.sin()).abs() < 1e-5);

    // Test cos
    let cos_result = cos(&a_gpu).expect("Failed to compute cos");
    let cos_cpu = cos_result.to_array().expect("Failed to convert to CPU");
    assert!((cos_cpu.get(&[0]).expect("Invalid index") - 1.0f32.cos()).abs() < 1e-5);
}

#[tokio::test]
async fn test_reduction_operations() {
    let context = new_context().expect("Failed to create GPU context");

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a = Array::from_vec(data.clone()).reshape(&[5]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    // Test sum
    let sum = sum_f32(&a_gpu).expect("Failed to compute sum");
    let expected_sum: f32 = data.iter().sum();
    assert!((sum - expected_sum).abs() < 1e-5);

    // Test mean
    let mean = mean_f32(&a_gpu).expect("Failed to compute mean");
    let expected_mean = expected_sum / data.len() as f32;
    assert!((mean - expected_mean).abs() < 1e-5);

    // Test max
    let max_val = max_f32(&a_gpu).expect("Failed to compute max");
    let expected_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((max_val - expected_max).abs() < 1e-5);

    // Test min
    let min_val = min_f32(&a_gpu).expect("Failed to compute min");
    let expected_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!((min_val - expected_min).abs() < 1e-5);
}

#[tokio::test]
async fn test_copy_operation() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    let b_gpu = copy_with_format(&a_gpu).expect("Failed to copy array");
    let b = b_gpu.to_array().expect("Failed to convert to CPU");

    assert_eq!(a.shape(), b.shape());
    for i in 0..4 {
        let a_val = a.get(&[i]).expect("Invalid index");
        let b_val = b.get(&[i]).expect("Invalid index");
        assert!((a_val - b_val).abs() < 1e-10);
    }
}

#[tokio::test]
async fn test_transpose_operation() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    let b_gpu = transpose(&a_gpu).expect("Failed to transpose array");
    assert_eq!(b_gpu.shape(), &[3, 2]);

    let b = b_gpu.to_array().expect("Failed to convert to CPU");

    // Check a few values
    assert!(
        (b.get(&[0, 0]).expect("Invalid index") - a.get(&[0, 0]).expect("Invalid index")).abs()
            < 1e-5
    );
    assert!(
        (b.get(&[0, 1]).expect("Invalid index") - a.get(&[1, 0]).expect("Invalid index")).abs()
            < 1e-5
    );
    assert!(
        (b.get(&[1, 0]).expect("Invalid index") - a.get(&[0, 1]).expect("Invalid index")).abs()
            < 1e-5
    );
}

#[tokio::test]
async fn test_broadcast_shapes_compatible() {
    // Test broadcasting shape calculation
    // This is implicitly tested by broadcast_add, but we test the logic here

    let context = new_context().expect("Failed to create GPU context");

    // Same shapes - should work
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    // For now, broadcast operations with same shape should fall back to regular ops
    let c_gpu = broadcast_add(&a_gpu, &b_gpu).expect("Failed to broadcast add");
    let c = c_gpu.to_array().expect("Failed to convert to CPU");

    assert_eq!(c.shape(), &[4]);
}

#[tokio::test]
async fn test_pow_operation() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![2.0f32, 3.0, 4.0, 5.0]).reshape(&[4]);
    let b = Array::from_vec(vec![2.0f32, 2.0, 2.0, 2.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = pow(&a_gpu, &b_gpu).expect("Failed to compute pow");
    let c = c_gpu.to_array().expect("Failed to convert to CPU");

    assert_eq!(c.shape(), &[4]);
    assert!((c.get(&[0]).expect("Invalid index") - 4.0).abs() < 1e-5);
    assert!((c.get(&[1]).expect("Invalid index") - 9.0).abs() < 1e-5);
    assert!((c.get(&[2]).expect("Invalid index") - 16.0).abs() < 1e-5);
    assert!((c.get(&[3]).expect("Invalid index") - 25.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_abs_and_neg_operations() {
    let context = new_context().expect("Failed to create GPU context");

    let a = Array::from_vec(vec![-1.0f32, 2.0, -3.0, 4.0]).reshape(&[4]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    // Test abs
    let abs_result = abs(&a_gpu).expect("Failed to compute abs");
    let abs_cpu = abs_result.to_array().expect("Failed to convert to CPU");
    assert!((abs_cpu.get(&[0]).expect("Invalid index") - 1.0).abs() < 1e-5);
    assert!((abs_cpu.get(&[2]).expect("Invalid index") - 3.0).abs() < 1e-5);

    // Test neg
    let neg_result = neg(&a_gpu).expect("Failed to compute neg");
    let neg_cpu = neg_result.to_array().expect("Failed to convert to CPU");
    assert!((neg_cpu.get(&[0]).expect("Invalid index") - 1.0).abs() < 1e-5);
    assert!((neg_cpu.get(&[1]).expect("Invalid index") + 2.0).abs() < 1e-5);
}

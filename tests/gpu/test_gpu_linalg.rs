//! Tests for GPU linear algebra operations

use numrs2::array::Array;
use numrs2::gpu::linalg;
use numrs2::gpu::{new_context, GpuArray};

#[tokio::test]
async fn test_matmul_basic() {
    let context = new_context().expect("Failed to create GPU context");

    // Create 2x2 matrices
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = linalg::matmul(&a_gpu, &b_gpu).expect("Failed to perform matmul");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    // Expected result: [[19, 22], [43, 50]]
    assert_eq!(c.shape(), &[2, 2]);
    assert!((c.get(&[0, 0]).expect("Invalid index") - 19.0).abs() < 1e-5);
    assert!((c.get(&[0, 1]).expect("Invalid index") - 22.0).abs() < 1e-5);
    assert!((c.get(&[1, 0]).expect("Invalid index") - 43.0).abs() < 1e-5);
    assert!((c.get(&[1, 1]).expect("Invalid index") - 50.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_matmul_rectangular() {
    let context = new_context().expect("Failed to create GPU context");

    // Create 3x2 and 2x4 matrices
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(&[2, 4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let c_gpu = linalg::matmul(&a_gpu, &b_gpu).expect("Failed to perform matmul");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    // Result should be 3x4
    assert_eq!(c.shape(), &[3, 4]);
}

#[tokio::test]
async fn test_matmul_incompatible_shapes() {
    let context = new_context().expect("Failed to create GPU context");

    // Create incompatible matrices (2x3 and 2x3)
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    // This should fail
    let result = linalg::matmul(&a_gpu, &b_gpu);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_dot_product() {
    let context = new_context().expect("Failed to create GPU context");

    // Create two vectors
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);
    let b = Array::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    let result = linalg::dot(&a_gpu, &b_gpu).expect("Failed to compute dot product");

    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert!((result - 70.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_dot_incompatible_shapes() {
    let context = new_context().expect("Failed to create GPU context");

    // Create vectors of different lengths
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0]).reshape(&[3]);
    let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[4]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let b_gpu =
        GpuArray::from_array_with_context(&b, context.clone()).expect("Failed to create GPU array");

    // This should fail
    let result = linalg::dot(&a_gpu, &b_gpu);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_norm_l2() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a vector [3.0, 4.0]
    let a = Array::from_vec(vec![3.0f32, 4.0]).reshape(&[2]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    let norm = linalg::norm_l2(&a_gpu).expect("Failed to compute L2 norm");

    // Expected: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
    assert!((norm - 5.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_matvec() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a 2x3 matrix and a 3-element vector
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let x = Array::from_vec(vec![1.0f32, 2.0, 3.0]).reshape(&[3]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let x_gpu =
        GpuArray::from_array_with_context(&x, context.clone()).expect("Failed to create GPU array");

    let y_gpu = linalg::matvec(&a_gpu, &x_gpu).expect("Failed to compute matvec");
    let y = y_gpu.to_array().expect("Failed to convert to CPU array");

    // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    assert_eq!(y.shape(), &[2]);
    assert!((y.get(&[0]).expect("Invalid index") - 14.0).abs() < 1e-5);
    assert!((y.get(&[1]).expect("Invalid index") - 32.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_vecmat() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a 2-element vector and a 2x3 matrix
    let x = Array::from_vec(vec![1.0f32, 2.0]).reshape(&[2]);
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let x_gpu =
        GpuArray::from_array_with_context(&x, context.clone()).expect("Failed to create GPU array");
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    let y_gpu = linalg::vecmat(&x_gpu, &a_gpu).expect("Failed to compute vecmat");
    let y = y_gpu.to_array().expect("Failed to convert to CPU array");

    // Expected: [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    assert_eq!(y.shape(), &[3]);
    assert!((y.get(&[0]).expect("Invalid index") - 9.0).abs() < 1e-5);
    assert!((y.get(&[1]).expect("Invalid index") - 12.0).abs() < 1e-5);
    assert!((y.get(&[2]).expect("Invalid index") - 15.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_matmul_identity() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a matrix and identity matrix
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let i = Array::from_vec(vec![1.0f32, 0.0, 0.0, 1.0]).reshape(&[2, 2]);

    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");
    let i_gpu =
        GpuArray::from_array_with_context(&i, context.clone()).expect("Failed to create GPU array");

    let c_gpu = linalg::matmul(&a_gpu, &i_gpu).expect("Failed to perform matmul");
    let c = c_gpu.to_array().expect("Failed to convert to CPU array");

    // A * I should equal A
    assert_eq!(c.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            let expected = a.get(&[i, j]).expect("Invalid index");
            let actual = c.get(&[i, j]).expect("Invalid index");
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

#[tokio::test]
async fn test_reshape() {
    let context = new_context().expect("Failed to create GPU context");

    // Create a 1D array
    let a = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[6]);
    let a_gpu =
        GpuArray::from_array_with_context(&a, context.clone()).expect("Failed to create GPU array");

    // Reshape to 2x3
    let a_2d = a_gpu.reshape(&[2, 3]).expect("Failed to reshape");
    assert_eq!(a_2d.shape(), &[2, 3]);

    // Reshape to 3x2
    let a_3x2 = a_gpu.reshape(&[3, 2]).expect("Failed to reshape");
    assert_eq!(a_3x2.shape(), &[3, 2]);

    // Invalid reshape should fail
    let result = a_gpu.reshape(&[2, 2]);
    assert!(result.is_err());
}

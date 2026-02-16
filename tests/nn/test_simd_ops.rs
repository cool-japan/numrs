/// Tests for SIMD-optimized operations
use numrs2::nn::simd_ops::*;
use scirs2_core::ndarray::array;

const EPSILON_F32: f32 = 1e-5;
const EPSILON_F64: f64 = 1e-6;

#[test]
fn test_simd_capabilities() {
    let caps = detect_simd_capabilities();
    println!("SIMD Available: {}", caps.simd_available);
    println!("AVX2: {}", caps.avx2_available);
    println!("AVX512: {}", caps.avx512_available);
    println!("NEON: {}", caps.neon_available);

    // Verify the function runs without panic
    assert!(is_simd_available() || !is_simd_available());
}

#[test]
fn test_simd_info() {
    let info = get_simd_info();
    println!("{}", info);

    // Verify it returns a non-empty string
    assert!(!info.is_empty());
}

#[test]
fn test_recommended_batch_size() {
    let batch_size = recommended_batch_size();
    println!("Recommended batch size: {}", batch_size);

    // Should return a reasonable value
    assert!(batch_size >= 64);
    assert!(batch_size <= 512);
}

// ================================
// SIMD Activation Function Tests
// ================================

#[test]
fn test_simd_relu_f32() {
    let x = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let y = simd_relu_f32(&x.view());
    let expected = array![0.0f32, 0.0, 0.0, 1.0, 2.0];

    for (actual, expected) in y.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < EPSILON_F32);
    }
}

#[test]
fn test_simd_relu_f64() {
    let x = array![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let y = simd_relu_f64(&x.view());
    let expected = array![0.0f64, 0.0, 0.0, 1.0, 2.0];

    for (actual, expected) in y.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < EPSILON_F64);
    }
}

#[test]
fn test_simd_relu_2d_f32() {
    let x = array![[-1.0f32, 0.0, 1.0], [2.0, -2.0, 3.0]];
    let y = simd_relu_2d_f32(&x.view());
    let expected = array![[0.0f32, 0.0, 1.0], [2.0, 0.0, 3.0]];

    assert_eq!(y.shape(), expected.shape());
    for (actual, expected) in y.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < EPSILON_F32);
    }
}

#[test]
fn test_simd_leaky_relu_f32() {
    let x = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01f32;
    let y = simd_leaky_relu_f32(&x.view(), alpha);

    assert!((y[0] - (-0.02)).abs() < EPSILON_F32);
    assert!((y[1] - (-0.01)).abs() < EPSILON_F32);
    assert!(y[2].abs() < EPSILON_F32);
    assert!((y[3] - 1.0).abs() < EPSILON_F32);
    assert!((y[4] - 2.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_sigmoid_f32() {
    let x = array![0.0f32, 1.0, -1.0];
    let y = simd_sigmoid_f32(&x.view());

    // sigmoid(0) = 0.5
    assert!((y[0] - 0.5).abs() < EPSILON_F32);

    // sigmoid(x) should be in (0, 1)
    for &val in y.iter() {
        assert!(val > 0.0 && val < 1.0);
    }

    // sigmoid(1) ≈ 0.731
    assert!((y[1] - 0.7310586).abs() < 1e-4);
}

#[test]
fn test_simd_sigmoid_f64() {
    let x = array![0.0f64, 1.0, -1.0];
    let y = simd_sigmoid_f64(&x.view());

    assert!((y[0] - 0.5).abs() < EPSILON_F64);

    for &val in y.iter() {
        assert!(val > 0.0 && val < 1.0);
    }
}

#[test]
fn test_simd_tanh_f32() {
    let x = array![0.0f32, 1.0, -1.0];
    let y = simd_tanh_f32(&x.view());

    // tanh(0) = 0
    assert!(y[0].abs() < EPSILON_F32);

    // tanh is odd: tanh(-x) = -tanh(x)
    assert!((y[1] + y[2]).abs() < 1e-4);

    // tanh(x) should be in (-1, 1)
    for &val in y.iter() {
        assert!(val > -1.0 && val < 1.0);
    }
}

#[test]
fn test_simd_tanh_f64() {
    let x = array![0.0f64, 1.0, -1.0];
    let y = simd_tanh_f64(&x.view());

    assert!(y[0].abs() < EPSILON_F64);
    assert!((y[1] + y[2]).abs() < EPSILON_F64);
}

#[test]
fn test_simd_gelu_f32() {
    let x = array![0.0f32, 1.0, -1.0];
    let y = simd_gelu_f32(&x.view());

    // GELU(0) ≈ 0
    assert!(y[0].abs() < 0.01);

    // GELU should be smooth and monotonic
    assert!(y[1] > y[0]);
    assert!(y[2] < y[0]);
}

#[test]
fn test_simd_gelu_f64() {
    let x = array![0.0f64, 1.0, -1.0];
    let y = simd_gelu_f64(&x.view());

    assert!(y[0].abs() < 0.01);
    assert!(y[1] > y[0]);
    assert!(y[2] < y[0]);
}

#[test]
fn test_simd_swish_f32() {
    let x = array![0.0f32, 1.0, -1.0, 2.0];
    let y = simd_swish_f32(&x.view());

    // Swish(0) ≈ 0
    assert!(y[0].abs() < 0.01);

    // For large positive x, swish(x) ≈ x (≈1.76 at x=2.0, so tolerance needs to be ~0.3)
    assert!((y[3] - 2.0).abs() < 0.3);
}

#[test]
fn test_simd_swish_f64() {
    let x = array![0.0f64, 1.0, 2.0];
    let y = simd_swish_f64(&x.view());

    assert!(y[0].abs() < 0.01);
}

#[test]
fn test_simd_mish_f32() {
    let x = array![0.0f32, 1.0, -1.0];
    let y = simd_mish_f32(&x.view());

    // Mish(0) ≈ 0
    assert!(y[0].abs() < 0.01);

    // Mish should be smooth
    assert!(y[1] > y[0]);
    assert!(y[2] < y[0]);
}

#[test]
fn test_simd_elu_f32() {
    let x = array![-1.0f32, 0.0, 1.0];
    let alpha = 1.0f32;
    let y = simd_elu_f32(&x.view(), alpha);

    // ELU(0) = 0
    assert!(y[1].abs() < EPSILON_F32);

    // For x > 0, ELU(x) = x
    assert!((y[2] - 1.0).abs() < EPSILON_F32);

    // For x < 0, ELU(x) < 0
    assert!(y[0] < 0.0);
}

#[test]
fn test_simd_selu_f32() {
    let x = array![-1.0f32, 0.0, 1.0];
    let y = simd_selu_f32(&x.view());

    // SELU(0) ≈ 0
    assert!(y[1].abs() < EPSILON_F32);

    // For x > 0, SELU(x) = λ * x
    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    assert!((y[2] - LAMBDA).abs() < EPSILON_F32);
}

// ================================
// SIMD Matrix Operation Tests
// ================================

#[test]
fn test_simd_matmul_f32() {
    let a = array![[1.0f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0f32, 6.0], [7.0, 8.0]];
    let c = simd_matmul_f32(&a.view(), &b.view()).expect("matmul failed");

    // Expected: [[19, 22], [43, 50]]
    assert!((c[[0, 0]] - 19.0).abs() < EPSILON_F32);
    assert!((c[[0, 1]] - 22.0).abs() < EPSILON_F32);
    assert!((c[[1, 0]] - 43.0).abs() < EPSILON_F32);
    assert!((c[[1, 1]] - 50.0).abs() < EPSILON_F32);
}

// Temporarily ignored due to upstream issue in scirs2-core-0.1.5/src/simd/dot.rs:1167
// TODO: Re-enable when scirs2-core SIMD f64 matmul issue is resolved
#[test]
#[ignore]
fn test_simd_matmul_f64() {
    let a = array![[1.0f64, 2.0], [3.0, 4.0]];
    let b = array![[5.0f64, 6.0], [7.0, 8.0]];
    let c = simd_matmul_f64(&a.view(), &b.view()).expect("matmul failed");

    assert!((c[[0, 0]] - 19.0).abs() < EPSILON_F64);
    assert!((c[[0, 1]] - 22.0).abs() < EPSILON_F64);
    assert!((c[[1, 0]] - 43.0).abs() < EPSILON_F64);
    assert!((c[[1, 1]] - 50.0).abs() < EPSILON_F64);
}

#[test]
fn test_simd_matmul_dimension_mismatch() {
    let a = array![[1.0f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0f32, 6.0, 7.0]]; // Wrong dimensions

    let result = simd_matmul_f32(&a.view(), &b.view());
    assert!(result.is_err());
}

// ================================
// SIMD Element-wise Operation Tests
// ================================

#[test]
fn test_simd_add_f32() {
    let a = array![1.0f32, 2.0, 3.0];
    let b = array![4.0f32, 5.0, 6.0];
    let c = simd_add_f32(&a.view(), &b.view()).expect("add failed");

    assert!((c[0] - 5.0).abs() < EPSILON_F32);
    assert!((c[1] - 7.0).abs() < EPSILON_F32);
    assert!((c[2] - 9.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_mul_f32() {
    let a = array![1.0f32, 2.0, 3.0];
    let b = array![4.0f32, 5.0, 6.0];
    let c = simd_mul_f32(&a.view(), &b.view()).expect("mul failed");

    assert!((c[0] - 4.0).abs() < EPSILON_F32);
    assert!((c[1] - 10.0).abs() < EPSILON_F32);
    assert!((c[2] - 18.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_sub_f32() {
    let a = array![5.0f32, 7.0, 9.0];
    let b = array![1.0f32, 2.0, 3.0];
    let c = simd_sub_f32(&a.view(), &b.view()).expect("sub failed");

    assert!((c[0] - 4.0).abs() < EPSILON_F32);
    assert!((c[1] - 5.0).abs() < EPSILON_F32);
    assert!((c[2] - 6.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_div_f32() {
    let a = array![4.0f32, 10.0, 18.0];
    let b = array![2.0f32, 5.0, 6.0];
    let c = simd_div_f32(&a.view(), &b.view()).expect("div failed");

    assert!((c[0] - 2.0).abs() < EPSILON_F32);
    assert!((c[1] - 2.0).abs() < EPSILON_F32);
    assert!((c[2] - 3.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_elementwise_dimension_mismatch() {
    let a = array![1.0f32, 2.0, 3.0];
    let b = array![4.0f32, 5.0];

    let result = simd_add_f32(&a.view(), &b.view());
    assert!(result.is_err());
}

// ================================
// SIMD Reduction Operation Tests
// ================================

#[test]
fn test_simd_dot_f32() {
    let a = array![1.0f32, 2.0, 3.0];
    let b = array![4.0f32, 5.0, 6.0];
    let dot = simd_dot_f32(&a.view(), &b.view()).expect("dot failed");

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((dot - 32.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_sum_f32() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let sum = simd_sum_f32(&x.view());

    // 1 + 2 + 3 + 4 + 5 = 15
    assert!((sum - 15.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_mean_f32() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mean = simd_mean_f32(&x.view());

    // (1 + 2 + 3 + 4 + 5) / 5 = 3.0
    assert!((mean - 3.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_norm_f32() {
    let x = array![3.0f32, 4.0];
    let norm = simd_norm_f32(&x.view());

    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5.0
    assert!((norm - 5.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_min_f32() {
    let x = array![5.0f32, 2.0, 8.0, 1.0, 9.0];
    let min = simd_min_f32(&x.view());

    assert!((min - 1.0).abs() < EPSILON_F32);
}

#[test]
fn test_simd_max_f32() {
    let x = array![5.0f32, 2.0, 8.0, 1.0, 9.0];
    let max = simd_max_f32(&x.view());

    assert!((max - 9.0).abs() < EPSILON_F32);
}

// ================================
// Large Array Tests
// ================================

#[test]
fn test_simd_relu_large_array() {
    use scirs2_core::ndarray::Array1;

    let x = Array1::linspace(-1000.0f32, 1000.0f32, 10000);
    let y = simd_relu_f32(&x.view());

    // All negative values should be zero
    for i in 0..5000 {
        assert!(y[i].abs() < EPSILON_F32);
    }

    // All positive values should be preserved
    for i in 5000..10000 {
        assert!(y[i] >= 0.0);
    }
}

#[test]
fn test_simd_operations_large_array() {
    use scirs2_core::ndarray::Array1;

    let a = Array1::from_vec((0..10000).map(|i| i as f32).collect());
    let b = Array1::from_vec((0..10000).map(|i| (i + 1) as f32).collect());

    let sum = simd_add_f32(&a.view(), &b.view()).expect("add failed");
    let dot = simd_dot_f32(&a.view(), &b.view()).expect("dot failed");

    // Verify results are finite
    assert!(sum.iter().all(|&x| x.is_finite()));
    assert!(dot.is_finite());
}

#[test]
fn test_simd_activations_edge_cases() {
    // Test with extreme values
    let x = array![-1000.0f32, 0.0, 1000.0];

    let relu = simd_relu_f32(&x.view());
    assert!(relu.iter().all(|&v| v.is_finite()));

    let sigmoid = simd_sigmoid_f32(&x.view());
    assert!(sigmoid.iter().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));

    let tanh = simd_tanh_f32(&x.view());
    assert!(tanh.iter().all(|&v| v.is_finite() && (-1.0..=1.0).contains(&v)));
}

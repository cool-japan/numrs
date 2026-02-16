/// Tests for activation functions
use numrs2::nn::activation::*;
use scirs2_core::ndarray::{array, Array1, Array2};

const EPSILON: f64 = 1e-6;

#[test]
fn test_relu_basic() {
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = relu(&x.view()).expect("relu failed");
    let expected = array![0.0, 0.0, 0.0, 1.0, 2.0];

    for (&actual, &expected) in y.iter().zip(expected.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_relu_2d() {
    let x = array![[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]];
    let y = relu_2d(&x.view()).expect("relu_2d failed");
    let expected = array![[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]];

    assert_eq!(y.shape(), expected.shape());
    for (&actual, &expected) in y.iter().zip(expected.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_relu_all_positive() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = relu(&x.view()).expect("relu failed");

    // All values should remain unchanged
    for (&actual, &expected) in y.iter().zip(x.iter()) {
        let actual: f64 = actual;
        let expected: f64 = expected;
        assert!((actual - expected).abs() < EPSILON);
    }
}

#[test]
fn test_relu_all_negative() {
    let x = array![-5.0, -4.0, -3.0, -2.0, -1.0];
    let y = relu(&x.view()).expect("relu failed");

    // All values should be zero
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.abs() < EPSILON);
    }
}

#[test]
fn test_leaky_relu_basic() {
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01_f64;
    let y = leaky_relu(&x.view(), alpha).expect("leaky_relu failed");

    assert!((y[0] - (-0.02_f64)).abs() < EPSILON); // -2 * 0.01 = -0.02
    assert!((y[1] - (-0.01_f64)).abs() < EPSILON); // -1 * 0.01 = -0.01
    assert!(y[2].abs() < EPSILON); // 0
    assert!((y[3] - 1.0_f64).abs() < EPSILON); // 1
    assert!((y[4] - 2.0_f64).abs() < EPSILON); // 2
}

#[test]
fn test_leaky_relu_negative_alpha_error() {
    let x = array![1.0, 2.0, 3.0];
    let result = leaky_relu(&x.view(), -0.1_f64);

    assert!(result.is_err());
}

#[test]
fn test_sigmoid_basic() {
    let x = array![0.0, 1.0, -1.0, 2.0, -2.0];
    let y = sigmoid(&x.view()).expect("sigmoid failed");

    // sigmoid(0) = 0.5
    assert!((y[0] - 0.5_f64).abs() < EPSILON);

    // sigmoid(x) should be in (0, 1)
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val > 0.0_f64 && val < 1.0_f64);
    }

    // sigmoid(-x) = 1 - sigmoid(x)
    let x_neg = array![-1.0];
    let x_pos = array![1.0];
    let y_neg = sigmoid(&x_neg.view()).expect("sigmoid failed");
    let y_pos = sigmoid(&x_pos.view()).expect("sigmoid failed");

    assert!((y_neg[0] + y_pos[0] - 1.0_f64).abs() < EPSILON);
}

#[test]
fn test_tanh_basic() {
    let x = array![0.0, 1.0, -1.0, 2.0, -2.0];
    let y = tanh(&x.view()).expect("tanh failed");

    // tanh(0) = 0
    let y0: f64 = y[0];
    assert!(y0.abs() < EPSILON);

    // tanh(x) should be in (-1, 1)
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val > -1.0_f64 && val < 1.0_f64);
    }

    // tanh is odd: tanh(-x) = -tanh(x)
    assert!((y[1] + y[2]).abs() < EPSILON);
    assert!((y[3] + y[4]).abs() < EPSILON);
}

#[test]
fn test_gelu_basic() {
    let x = array![0.0, 1.0, -1.0];
    let y = gelu(&x.view()).expect("gelu failed");

    // GELU(0) ≈ 0
    let y0: f64 = y[0];
    assert!(y0.abs() < 0.01_f64);

    // GELU should be smooth and monotonic
    assert!(y[1] > y[0]); // GELU(1) > GELU(0)
    assert!(y[2] < y[0]); // GELU(-1) < GELU(0)
}

#[test]
fn test_swish_basic() {
    let x = array![0.0, 1.0, -1.0, 2.0];
    let y = swish(&x.view()).expect("swish failed");

    // Swish(0) ≈ 0
    let y0: f64 = y[0];
    assert!(y0.abs() < 0.01_f64);

    // For positive x, swish(x) should be close to x for large x
    // swish(2) should be close to 2 (≈1.76, so tolerance needs to be ~0.3)
    assert!((y[3] - 2.0_f64).abs() < 0.3_f64);
}

#[test]
fn test_mish_basic() {
    let x = array![0.0, 1.0, -1.0, 2.0];
    let y = mish(&x.view()).expect("mish failed");

    // Mish(0) ≈ 0
    let y0: f64 = y[0];
    assert!(y0.abs() < 0.01_f64);

    // Mish should be smooth
    assert!(y[1] > y[0]);
    assert!(y[2] < y[0]);
}

#[test]
fn test_elu_basic() {
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0_f64;
    let y = elu(&x.view(), alpha).expect("elu failed");

    // For x >= 0, ELU(x) = x
    let y2: f64 = y[2];
    assert!(y2.abs() < EPSILON); // ELU(0) = 0
    assert!((y[3] - 1.0_f64).abs() < EPSILON); // ELU(1) = 1
    assert!((y[4] - 2.0_f64).abs() < EPSILON); // ELU(2) = 2

    // For x < 0, ELU(x) = alpha * (exp(x) - 1)
    assert!(y[0] < 0.0_f64); // ELU(-2) < 0
    assert!(y[1] < 0.0_f64); // ELU(-1) < 0
    assert!(y[0] < y[1]); // ELU is monotonic
}

#[test]
fn test_selu_basic() {
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = selu(&x.view()).expect("selu failed");

    // SELU(0) ≈ 0
    let y2_selu: f64 = y[2];
    assert!(y2_selu.abs() < EPSILON);

    // For x > 0, SELU(x) = lambda * x
    const LAMBDA: f64 = 1.0507009873554804934193349852946;
    assert!((y[3] - LAMBDA * 1.0_f64).abs() < EPSILON);
    assert!((y[4] - LAMBDA * 2.0_f64).abs() < EPSILON);
}

#[test]
fn test_softmax_basic() {
    let x = array![1.0, 2.0, 3.0];
    let y = softmax(&x.view()).expect("softmax failed");

    // Sum should be 1
    let sum: f64 = y.iter().sum();
    assert!((sum - 1.0_f64).abs() < EPSILON);

    // All values should be positive
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val > 0.0_f64);
    }

    // Larger inputs should have larger outputs
    assert!(y[0] < y[1]);
    assert!(y[1] < y[2]);
}

#[test]
fn test_softmax_2d() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = softmax_2d(&x.view(), 1).expect("softmax_2d failed");

    // Each row should sum to 1
    for i in 0..y.nrows() {
        let row_sum: f64 = y.row(i).iter().sum();
        assert!((row_sum - 1.0_f64).abs() < EPSILON);
    }
}

#[test]
fn test_softmax_numerical_stability() {
    // Test with large values (should not overflow)
    let x = array![1000.0, 1001.0, 1002.0];
    let y = softmax(&x.view()).expect("softmax failed");

    // Should not contain NaN or infinity
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val.is_finite());
    }

    // Sum should still be 1
    let sum: f64 = y.iter().sum();
    assert!((sum - 1.0_f64).abs() < EPSILON);
}

#[test]
fn test_log_softmax_basic() {
    let x = array![1.0, 2.0, 3.0];
    let y = log_softmax(&x.view()).expect("log_softmax failed");

    // All values should be negative (since log(p) where 0 < p < 1)
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val < 0.0_f64);
    }

    // exp(log_softmax) should give softmax
    let softmax_from_log: Array1<f64> = y.mapv(|v| v.exp());
    let softmax_direct = softmax(&x.view()).expect("softmax failed");

    for (&a, &b) in softmax_from_log.iter().zip(softmax_direct.iter()) {
        let a: f64 = a;
        let b: f64 = b;
        assert!((a - b).abs() < EPSILON);
    }
}

#[test]
fn test_softplus_basic() {
    let x = array![0.0, 1.0, -1.0, 10.0];
    let y = softplus(&x.view()).expect("softplus failed");

    // softplus(0) = ln(2)
    assert!((y[0] - std::f64::consts::LN_2).abs() < 1e-5_f64);

    // softplus should always be positive
    for &val in y.iter() {
        let val: f64 = val;
        assert!(val > 0.0_f64);
    }

    // For large positive x, softplus(x) ≈ x
    assert!((y[3] - 10.0_f64).abs() < 0.01_f64);
}

#[test]
fn test_activation_empty_array() {
    let x: Array1<f64> = Array1::zeros(0);

    // Should handle empty arrays gracefully
    let y = relu(&x.view()).expect("relu failed");
    assert_eq!(y.len(), 0);
}

#[test]
fn test_activation_large_array() {
    let x = Array1::linspace(-100.0, 100.0, 10000);
    let y = relu(&x.view()).expect("relu failed");

    // Should handle large arrays without issues
    assert_eq!(y.len(), 10000);
}

#[test]
fn test_activation_f32() {
    // Test f32 versions
    let x = array![-1.0f32, 0.0, 1.0];
    let y = relu(&x.view()).expect("relu failed");

    assert!((y[0] - 0.0_f32).abs() < 1e-6_f32);
    assert!((y[1] - 0.0_f32).abs() < 1e-6_f32);
    assert!((y[2] - 1.0_f32).abs() < 1e-6_f32);
}

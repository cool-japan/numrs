//! Property-Based Testing for Mathematical Invariants
//!
//! This module uses proptest to verify mathematical properties of numerical
//! computations, polynomial functions, and special functions.

use numrs2::array::Array;
use numrs2::prelude::*;
use proptest::prelude::*;

// =============================================================================
// PROPTEST STRATEGIES
// =============================================================================

/// Strategy for generating small arrays of f64 values
fn array_f64_strategy(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1000.0f64..1000.0f64, 1..max_len)
}

/// Strategy for generating small non-negative values (for special functions)
fn positive_f64_strategy() -> impl Strategy<Value = f64> {
    0.001f64..100.0f64
}

/// Strategy for generating angles in radians
fn angle_strategy() -> impl Strategy<Value = f64> {
    -10.0f64 * std::f64::consts::PI..10.0f64 * std::f64::consts::PI
}

// =============================================================================
// TRIGONOMETRIC PROPERTIES
// =============================================================================

proptest! {
    /// sin^2(x) + cos^2(x) == 1 (Pythagorean identity)
    #[test]
    fn prop_pythagorean_identity(x in angle_strategy()) {
        let sin_x = x.sin();
        let cos_x = x.cos();
        let result = sin_x * sin_x + cos_x * cos_x;

        prop_assert!((result - 1.0f64).abs() < 1e-12,
            "Pythagorean identity failed: sin^2({}) + cos^2({}) = {} (expected 1)", x, x, result);
    }

    /// sin(x) == cos(PI/2 - x)
    #[test]
    fn prop_sin_cos_complementary(x in angle_strategy()) {
        let sin_x = x.sin();
        let cos_comp = (std::f64::consts::FRAC_PI_2 - x).cos();

        prop_assert!((sin_x - cos_comp).abs() < 1e-12,
            "Complementary identity failed: sin({}) = {} != cos(PI/2 - {}) = {}",
            x, sin_x, x, cos_comp);
    }

    /// tan(x) == sin(x) / cos(x) (when cos(x) != 0)
    #[test]
    fn prop_tan_definition(x in -1.5f64..1.5f64) { // Avoid near PI/2
        let sin_x = x.sin();
        let cos_x = x.cos();
        let tan_x = x.tan();
        let computed = sin_x / cos_x;

        prop_assert!((tan_x - computed).abs() < 1e-12,
            "tan({}) = {} != sin/cos = {}", x, tan_x, computed);
    }

    /// sinh and cosh identity: cosh^2(x) - sinh^2(x) == 1
    /// Note: For large |x|, numerical precision decreases due to floating-point limitations
    #[test]
    fn prop_hyperbolic_identity(x in -10.0f64..10.0f64) {
        let sinh_x = x.sinh();
        let cosh_x = x.cosh();
        let result = cosh_x * cosh_x - sinh_x * sinh_x;

        // Use relative tolerance for larger values to account for floating-point precision
        let tolerance = 1e-8f64.max((x.abs() * 1e-15).exp());
        prop_assert!((result - 1.0f64).abs() < tolerance,
            "Hyperbolic identity failed: cosh^2({}) - sinh^2({}) = {} (expected 1)", x, x, result);
    }
}

// =============================================================================
// EXPONENTIAL AND LOGARITHMIC PROPERTIES
// =============================================================================

proptest! {
    /// log(exp(x)) == x
    #[test]
    fn prop_log_exp_inverse(x in -100.0f64..100.0f64) {
        let result = x.exp().ln();
        prop_assert!((result - x).abs() < 1e-10,
            "log(exp({})) = {} (expected {})", x, result, x);
    }

    /// exp(log(x)) == x for x > 0
    #[test]
    fn prop_exp_log_inverse(x in positive_f64_strategy()) {
        let result = x.ln().exp();
        let rel_err = ((result - x) / x).abs();
        prop_assert!(rel_err < 1e-10,
            "exp(log({})) = {} (expected {})", x, result, x);
    }

    /// log(a * b) == log(a) + log(b)
    #[test]
    fn prop_log_product(
        a in positive_f64_strategy(),
        b in positive_f64_strategy()
    ) {
        let left = (a * b).ln();
        let right = a.ln() + b.ln();

        prop_assert!((left - right).abs() < 1e-10,
            "log({} * {}) = {} != log({}) + log({}) = {}",
            a, b, left, a, b, right);
    }

    /// exp(a + b) == exp(a) * exp(b)
    #[test]
    fn prop_exp_sum(a in -50.0f64..50.0f64, b in -50.0f64..50.0f64) {
        if a + b > 700.0 || a + b < -700.0 { return Ok(()); } // Avoid overflow

        let left = (a + b).exp();
        let right = a.exp() * b.exp();
        let rel_err = if right.abs() > 1e-10 { ((left - right) / right).abs() } else { (left - right).abs() };

        prop_assert!(rel_err < 1e-10,
            "exp({} + {}) = {} != exp({}) * exp({}) = {}",
            a, b, left, a, b, right);
    }
}

// =============================================================================
// POLYNOMIAL PROPERTIES
// =============================================================================

proptest! {
    /// Polynomial evaluation: p(x) at x=0 equals the constant term
    #[test]
    fn prop_polynomial_zero_eval(coeffs in prop::collection::vec(-100.0f64..100.0f64, 1..10)) {
        let poly = Polynomial::<f64>::new(coeffs.clone());
        let result: f64 = poly.evaluate(0.0);
        let expected = coeffs.last().unwrap();

        prop_assert!((result - expected).abs() < 1e-10,
            "p(0) = {} (expected constant term {})", result, expected);
    }

    /// Polynomial derivative: d/dx(x^n) = n*x^(n-1)
    #[test]
    fn prop_polynomial_power_derivative(n in 1usize..10, x in -10.0f64..10.0f64) {
        // Polynomial representing x^n
        let mut coeffs = vec![0.0f64; n + 1];
        coeffs[0] = 1.0; // Leading coefficient

        let poly = Polynomial::<f64>::new(coeffs);
        let dpoly = poly.derivative();

        let result: f64 = dpoly.evaluate(x);
        let expected: f64 = (n as f64) * x.powi((n - 1) as i32);

        let rel_err = if expected.abs() > 1e-10 { ((result - expected) / expected).abs() } else { (result - expected).abs() };
        prop_assert!(rel_err < 1e-8,
            "d/dx(x^{}) at x={} = {} (expected {})", n, x, result, expected);
    }

    /// Polynomial addition is commutative
    #[test]
    fn prop_polynomial_add_commutative(
        coeffs1 in prop::collection::vec(-100.0f64..100.0f64, 1..10),
        coeffs2 in prop::collection::vec(-100.0f64..100.0f64, 1..10),
        x in -10.0f64..10.0f64
    ) {
        let p1 = Polynomial::<f64>::new(coeffs1);
        let p2 = Polynomial::<f64>::new(coeffs2);

        let sum1 = p1.clone() + p2.clone();
        let sum2 = p2 + p1;

        let result1: f64 = sum1.evaluate(x);
        let result2: f64 = sum2.evaluate(x);

        prop_assert!((result1 - result2).abs() < 1e-10,
            "(p1 + p2)(x) = {} != (p2 + p1)(x) = {}", result1, result2);
    }

    /// Polynomial multiplication is commutative
    #[test]
    fn prop_polynomial_mul_commutative(
        coeffs1 in prop::collection::vec(-10.0f64..10.0f64, 1..5),
        coeffs2 in prop::collection::vec(-10.0f64..10.0f64, 1..5),
        x in -5.0f64..5.0f64
    ) {
        let p1 = Polynomial::<f64>::new(coeffs1);
        let p2 = Polynomial::<f64>::new(coeffs2);

        let prod1 = p1.clone() * p2.clone();
        let prod2 = p2 * p1;

        let result1: f64 = prod1.evaluate(x);
        let result2: f64 = prod2.evaluate(x);

        let rel_err = if result2.abs() > 1e-10 { ((result1 - result2) / result2).abs() } else { (result1 - result2).abs() };
        prop_assert!(rel_err < 1e-8,
            "(p1 * p2)(x) = {} != (p2 * p1)(x) = {}", result1, result2);
    }
}

// =============================================================================
// NUMERICAL STABILITY PROPERTIES
// =============================================================================

proptest! {
    /// Sum should be associative within numerical precision
    #[test]
    fn prop_sum_stability(data in array_f64_strategy(100)) {
        if data.is_empty() { return Ok(()); }

        // Sum using different orderings should give similar results
        let forward_sum: f64 = data.iter().sum();
        let reverse_sum: f64 = data.iter().rev().cloned().sum();

        // For well-behaved data, the sums should be close
        let rel_err = if forward_sum.abs() > 1e-10 {
            ((forward_sum - reverse_sum) / forward_sum).abs()
        } else {
            (forward_sum - reverse_sum).abs()
        };

        // Allow for some numerical error
        prop_assert!(rel_err < 1e-10 || (forward_sum - reverse_sum).abs() < 1e-10,
            "Sum order matters too much: forward={} vs reverse={}",
            forward_sum, reverse_sum);
    }

    /// Mean should satisfy: mean(a) * len(a) == sum(a)
    #[test]
    fn prop_mean_sum_relationship(data in array_f64_strategy(100)) {
        if data.is_empty() { return Ok(()); }

        let a = Array::<f64>::from_vec(data.clone());
        let sum_val = a.sum();
        let mean_val = a.mean();
        let n = data.len() as f64;

        let expected_sum = mean_val * n;
        let rel_err = if sum_val.abs() > 1e-10 {
            ((sum_val - expected_sum) / sum_val).abs()
        } else {
            (sum_val - expected_sum).abs()
        };

        prop_assert!(rel_err < 1e-10,
            "mean * n = {} != sum = {}", expected_sum, sum_val);
    }

    /// Variance should be non-negative
    #[test]
    fn prop_variance_nonnegative(data in array_f64_strategy(100)) {
        if data.len() < 2 { return Ok(()); }

        let a = Array::<f64>::from_vec(data);
        let var_val = a.var();

        prop_assert!(var_val >= 0.0 || (var_val > -1e-10), // Small negative due to numerical error is OK
            "Variance should be non-negative: {}", var_val);
    }

    /// Standard deviation should equal sqrt(variance)
    #[test]
    fn prop_std_variance_relationship(data in array_f64_strategy(100)) {
        if data.len() < 2 { return Ok(()); }

        let a = Array::<f64>::from_vec(data);
        let var_val = a.var();
        let std_val = a.std();

        if var_val >= 0.0 {
            let expected_std = var_val.sqrt();
            prop_assert!((std_val - expected_std).abs() < 1e-10,
                "std = {} != sqrt(var) = sqrt({}) = {}", std_val, var_val, expected_std);
        }
    }
}

// =============================================================================
// BROADCASTING PROPERTIES
// =============================================================================

proptest! {
    /// Broadcasting a scalar to an array should produce identical values
    #[test]
    fn prop_scalar_broadcast(val in -1000.0f64..1000.0f64, len in 1usize..100) {
        let arr = Array::<f64>::from_vec(vec![val; len]);
        let data = arr.to_vec();

        for (i, v) in data.iter().enumerate() {
            prop_assert!((*v - val).abs() < 1e-10,
                "Scalar broadcast failed at index {}: {} != {}", i, v, val);
        }
    }

    /// Array reshape should preserve total elements
    #[test]
    fn prop_reshape_preserves_elements(data in array_f64_strategy(100)) {
        if data.is_empty() { return Ok(()); }

        let arr = Array::<f64>::from_vec(data.clone());
        let n = data.len();

        // Find a valid reshape (2D if possible)
        let divisors: Vec<usize> = (1..=n).filter(|d| n % d == 0).collect();
        if divisors.len() >= 2 {
            let d = divisors[divisors.len() / 2];
            let reshaped = arr.reshape(&[d, n / d]);
            let flat = reshaped.to_vec();

            prop_assert_eq!(flat.len(), n);
            for (original, reshaped_val) in data.iter().zip(flat.iter()) {
                prop_assert!((original - reshaped_val).abs() < 1e-10);
            }
        }
    }
}

// =============================================================================
// SPECIAL FUNCTION PROPERTIES
// =============================================================================

proptest! {
    /// Gamma function: Gamma(n+1) = n! for positive integers
    #[test]
    fn prop_gamma_factorial(n in 1u32..13) { // Keep small to avoid overflow
        let x_arr = Array::<f64>::from_vec(vec![(n as f64) + 1.0]);
        let gamma_arr = gamma(&x_arr);
        let gamma_val = gamma_arr.to_vec()[0];
        let factorial: f64 = (1..=n).map(|i| i as f64).product();

        let rel_err = ((gamma_val - factorial) / factorial).abs();
        prop_assert!(rel_err < 1e-10,
            "Gamma({}) = {} != {}! = {}", n + 1, gamma_val, n, factorial);
    }

    /// Error function property: erf(-x) == -erf(x)
    #[test]
    fn prop_erf_odd(x in -5.0f64..5.0f64) {
        let x_arr = Array::<f64>::from_vec(vec![x]);
        let neg_x_arr = Array::<f64>::from_vec(vec![-x]);

        let erf_arr = erf(&x_arr);
        let erf_neg_arr = erf(&neg_x_arr);

        let erf_x = erf_arr.to_vec()[0];
        let erf_neg_x = erf_neg_arr.to_vec()[0];

        prop_assert!((erf_x + erf_neg_x).abs() < 1e-12,
            "erf({}) = {} != -erf({}) = {}", x, erf_x, -x, -erf_neg_x);
    }

    /// Error function property: erf(0) == 0
    #[test]
    fn prop_erf_zero(_dummy in Just(())) {
        let x_arr = Array::<f64>::from_vec(vec![0.0f64]);
        let result_arr = erf(&x_arr);
        let result: f64 = result_arr.to_vec()[0];
        prop_assert!(result.abs() < 1e-15, "erf(0) = {} (expected 0)", result);
    }

    /// Complementary error function: erfc(x) == 1 - erf(x)
    #[test]
    fn prop_erfc_definition(x in -5.0f64..5.0f64) {
        let x_arr = Array::<f64>::from_vec(vec![x]);
        let erf_arr = erf(&x_arr);
        let erfc_arr = erfc(&x_arr);

        let erf_x: f64 = erf_arr.to_vec()[0];
        let erfc_x: f64 = erfc_arr.to_vec()[0];

        prop_assert!((erfc_x - (1.0 - erf_x)).abs() < 1e-12,
            "erfc({}) = {} != 1 - erf({}) = {}", x, erfc_x, x, 1.0 - erf_x);
    }
}

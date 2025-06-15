#![allow(deprecated)] // Allow deprecated warnings during API transition
#![allow(unused_imports)] // Allow unused imports during API transition

use approx::{assert_abs_diff_eq, assert_relative_eq};
/// Property-based tests for special functions
///
/// This file tests the mathematical properties and relationships
/// that should be satisfied by the special functions in NumRS2.
use numrs2::prelude::*;

// Constants for testing
const SAMPLE_SIZE: usize = 100;
const TOLERANCE: f64 = 1e-8; // Tolerance for floating point comparisons

// Commented out since it's not being used
// /// Helper function to create an array of random positive values
// fn random_positive_array(size: usize, max_value: f64) -> Array<f64> {
//     let rng = random::default_rng();
//     let values = rng.random::<f64>(&[size]).unwrap();
//     values.multiply_scalar(max_value)
// }

/// Helper function to create an array of random values in a specific range
fn random_array_in_range(size: usize, min_value: f64, max_value: f64) -> Array<f64> {
    let rng = random::default_rng();
    let values = rng.random::<f64>(&[size]).unwrap();
    values
        .multiply_scalar(max_value - min_value)
        .add_scalar(min_value)
}

/// Helper function to check if arrays are approximately equal
fn arrays_approx_equal(a: &Array<f64>, b: &Array<f64>, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    for (a_val, b_val) in a_vec.iter().zip(b_vec.iter()) {
        if (a_val - b_val).abs() > tol {
            return false;
        }
    }

    true
}

#[test]
fn test_erf_properties() {
    // Property 1: erf(0) = 0
    let zero = Array::from_vec(vec![0.0]);
    let erf_zero = erf(&zero);
    assert_abs_diff_eq!(erf_zero.get(&[0]).unwrap(), 0.0, epsilon = TOLERANCE);

    // Property 2: erf(-x) = -erf(x) (odd function)
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 5.0);
    let minus_x = x.map(|v| -v);

    let erf_x = erf(&x);
    let erf_minus_x = erf(&minus_x);
    let expected_erf_minus_x = erf_x.map(|v| -v);

    assert!(
        arrays_approx_equal(&erf_minus_x, &expected_erf_minus_x, TOLERANCE),
        "erf(-x) should equal -erf(x)"
    );

    // Property 3: |erf(x)| < 1 for all finite x
    let large_x = random_array_in_range(SAMPLE_SIZE, -10.0, 10.0);
    let erf_large_x = erf(&large_x);

    for i in 0..SAMPLE_SIZE {
        assert!(
            erf_large_x.get(&[i]).unwrap().abs() <= 1.0,
            "erf(x) should have magnitude <= 1"
        );
    }

    // Property 4: erf(x) approaches 1 as x approaches infinity
    let very_large_x = Array::from_vec(vec![10.0, 15.0, 20.0]);
    let erf_very_large_x = erf(&very_large_x);

    for i in 0..3 {
        assert_abs_diff_eq!(erf_very_large_x.get(&[i]).unwrap(), 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_erfc_properties() {
    // Property 1: erfc(0) = 1
    let zero = Array::from_vec(vec![0.0]);
    let erfc_zero = erfc(&zero);
    assert_abs_diff_eq!(erfc_zero.get(&[0]).unwrap(), 1.0, epsilon = TOLERANCE);

    // Property 2: erfc(x) = 1 - erf(x)
    let x = random_array_in_range(SAMPLE_SIZE, -5.0, 5.0);

    let erfc_x = erfc(&x);
    let erf_x = erf(&x);
    let one_minus_erf_x = erf_x.map(|v| 1.0 - v);

    assert!(
        arrays_approx_equal(&erfc_x, &one_minus_erf_x, TOLERANCE),
        "erfc(x) should equal 1 - erf(x)"
    );

    // Property 3: erfc(x) approaches 0 as x approaches infinity
    let very_large_x = Array::from_vec(vec![10.0, 15.0, 20.0]);
    let erfc_very_large_x = erfc(&very_large_x);

    for i in 0..3 {
        assert_abs_diff_eq!(erfc_very_large_x.get(&[i]).unwrap(), 0.0, epsilon = 1e-6);
    }

    // Property 4: erfc(x) approaches 2 as x approaches negative infinity
    let very_negative_x = Array::from_vec(vec![-10.0, -15.0, -20.0]);
    let erfc_very_negative_x = erfc(&very_negative_x);

    for i in 0..3 {
        assert_abs_diff_eq!(erfc_very_negative_x.get(&[i]).unwrap(), 2.0, epsilon = 1e-6);
    }
}

#[test]
fn test_erfinv_properties() {
    // Property 1: erfinv(0) = 0
    let zero = Array::from_vec(vec![0.0]);
    let erfinv_zero = erfinv(&zero);
    assert_abs_diff_eq!(erfinv_zero.get(&[0]).unwrap(), 0.0, epsilon = TOLERANCE);

    // Property 2: erfinv(-x) = -erfinv(x) (odd function)
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 0.9);
    let minus_x = x.map(|v| -v);

    let erfinv_x = erfinv(&x);
    let erfinv_minus_x = erfinv(&minus_x);
    let expected_erfinv_minus_x = erfinv_x.map(|v| -v);

    assert!(
        arrays_approx_equal(&erfinv_minus_x, &expected_erfinv_minus_x, TOLERANCE),
        "erfinv(-x) should equal -erfinv(x)"
    );

    // Property 3: erf(erfinv(x)) = x (inverse property)
    let y = random_array_in_range(SAMPLE_SIZE, -0.9, 0.9);
    let erfinv_y = erfinv(&y);
    let erf_erfinv_y = erf(&erfinv_y);

    assert!(
        arrays_approx_equal(&erf_erfinv_y, &y, 0.005),
        "erf(erfinv(x)) should equal x"
    );
}

#[test]
// Previously ignored, now passes
fn test_erfcinv_properties() {
    // Property 1: erfcinv(1) = 0
    let one = Array::from_vec(vec![1.0]);
    let erfcinv_one = erfcinv(&one);
    assert_abs_diff_eq!(erfcinv_one.get(&[0]).unwrap(), 0.0, epsilon = TOLERANCE);

    // Property 2: erfcinv(x) = erfinv(1 - x)
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 1.9);

    let erfcinv_x = erfcinv(&x);
    let one_minus_x = x.map(|v| 1.0 - v);
    let erfinv_one_minus_x = erfinv(&one_minus_x);

    assert!(
        arrays_approx_equal(&erfcinv_x, &erfinv_one_minus_x, TOLERANCE),
        "erfcinv(x) should equal erfinv(1 - x)"
    );

    // Property 3: erfc(erfcinv(x)) = x (inverse property)
    let y = random_array_in_range(SAMPLE_SIZE, 0.1, 1.9);
    let erfcinv_y = erfcinv(&y);
    let erfc_erfcinv_y = erfc(&erfcinv_y);

    assert!(
        arrays_approx_equal(&erfc_erfcinv_y, &y, 0.01),
        "erfc(erfcinv(x)) should equal x"
    );
}

#[test]
// Previously ignored, now passes
fn test_gamma_properties() {
    // Property 1: gamma(n) = (n-1)! for positive integers
    let integers = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let gamma_integers = gamma(&integers);

    // Expected factorials: 0!, 1!, 2!, 3!, 4!
    let expected = vec![1.0, 1.0, 2.0, 6.0, 24.0];

    for (i, &expected_val) in expected.iter().enumerate() {
        assert_abs_diff_eq!(
            gamma_integers.get(&[i]).unwrap(),
            expected_val,
            epsilon = TOLERANCE
        );
    }

    // Property 2: gamma(x+1) = x * gamma(x) (recurrence relation)
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);
    let x_plus_1 = x.add_scalar(1.0);

    let gamma_x = gamma(&x);
    let gamma_x_plus_1 = gamma(&x_plus_1);
    let x_times_gamma_x = Array::from_vec(
        x.to_vec()
            .iter()
            .zip(gamma_x.to_vec().iter())
            .map(|(&x_val, &gamma_val)| x_val * gamma_val)
            .collect(),
    );

    assert!(
        arrays_approx_equal(&gamma_x_plus_1, &x_times_gamma_x, TOLERANCE * 10.0),
        "gamma(x+1) should equal x * gamma(x)"
    );

    // Property 3: gamma(0.5) = sqrt(π)
    let half = Array::from_vec(vec![0.5]);
    let gamma_half = gamma(&half);
    let sqrt_pi = std::f64::consts::PI.sqrt();

    assert_abs_diff_eq!(gamma_half.get(&[0]).unwrap(), sqrt_pi, epsilon = TOLERANCE);
}

#[test]
// Previously ignored, now passes
fn test_gammaln_properties() {
    // Property 1: gammaln(x) = ln(gamma(x))
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);

    let gammaln_x = gammaln(&x);
    let gamma_x = gamma(&x);
    let ln_gamma_x = gamma_x.map(|v| v.ln());

    assert!(
        arrays_approx_equal(&gammaln_x, &ln_gamma_x, TOLERANCE),
        "gammaln(x) should equal ln(gamma(x))"
    );

    // Property 2: gammaln(x+1) = ln(x) + gammaln(x) (logarithmic recurrence relation)
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);
    let x_plus_1 = x.add_scalar(1.0);

    let gammaln_x = gammaln(&x);
    let gammaln_x_plus_1 = gammaln(&x_plus_1);
    let ln_x_vec = x.map(|v| v.ln());
    let ln_x_plus_gammaln_x = Array::from_vec(
        ln_x_vec
            .to_vec()
            .iter()
            .zip(gammaln_x.to_vec().iter())
            .map(|(&ln_val, &gammaln_val)| ln_val + gammaln_val)
            .collect(),
    );

    assert!(
        arrays_approx_equal(&gammaln_x_plus_1, &ln_x_plus_gammaln_x, 0.1),
        "gammaln(x+1) should equal ln(x) + gammaln(x)"
    );
}

#[test]
fn test_digamma_properties() {
    // Property 1: digamma(x+1) = digamma(x) + 1/x (recurrence relation)
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);
    let x_plus_1 = x.add_scalar(1.0);

    let digamma_x = digamma(&x);
    let digamma_x_plus_1 = digamma(&x_plus_1);
    let one_over_x = x.map(|v| 1.0 / v);
    let expected = Array::from_vec(
        digamma_x
            .to_vec()
            .iter()
            .zip(one_over_x.to_vec().iter())
            .map(|(&digamma_val, &one_over_val)| digamma_val + one_over_val)
            .collect(),
    );

    assert!(
        arrays_approx_equal(&digamma_x_plus_1, &expected, TOLERANCE * 10.0),
        "digamma(x+1) should equal digamma(x) + 1/x"
    );

    // Property 2: digamma(1) = -γ (negative Euler-Mascheroni constant)
    let one = Array::from_vec(vec![1.0]);
    let digamma_one = digamma(&one);
    let euler_mascheroni = 0.57721566490153286060;

    assert_abs_diff_eq!(
        digamma_one.get(&[0]).unwrap(),
        -euler_mascheroni,
        epsilon = TOLERANCE
    );
}

#[test]
fn test_bessel_properties() {
    // Test various properties of Bessel functions

    // Property 1: J_0(0) = 1, J_n(0) = 0 for n > 0
    let zero = Array::from_vec(vec![0.0]);
    let j0_at_zero = bessel_j(0, &zero);
    assert_abs_diff_eq!(j0_at_zero.get(&[0]).unwrap(), 1.0, epsilon = TOLERANCE);

    for n in 1..5 {
        let jn_at_zero = bessel_j(n, &zero);
        assert_abs_diff_eq!(jn_at_zero.get(&[0]).unwrap(), 0.0, epsilon = TOLERANCE);
    }

    // Property 2: J_{-n}(x) = (-1)^n * J_n(x) for integer n
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 10.0);

    for n in 1..5 {
        let j_n = bessel_j(n, &x);
        let j_minus_n = bessel_j(-n, &x);

        let factor = if n % 2 == 0 { 1.0 } else { -1.0 };
        let expected = j_n.multiply_scalar(factor);

        assert!(
            arrays_approx_equal(&j_minus_n, &expected, TOLERANCE * 10.0),
            "J_{{-n}}(x) should equal (-1)^n * J_n(x)"
        );
    }

    // Property 3: I_0(0) = 1, I_n(0) = 0 for n > 0
    let zero = Array::from_vec(vec![0.0]);
    let i0_at_zero = bessel_i(0, &zero);
    assert_abs_diff_eq!(i0_at_zero.get(&[0]).unwrap(), 1.0, epsilon = TOLERANCE);

    for n in 1..5 {
        let in_at_zero = bessel_i(n, &zero);
        assert_abs_diff_eq!(in_at_zero.get(&[0]).unwrap(), 0.0, epsilon = TOLERANCE);
    }

    // Property 4: I_{-n}(x) = I_n(x) for integer n
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 10.0);

    for n in 0..5 {
        let i_n = bessel_i(n, &x);
        let i_minus_n = bessel_i(-n, &x);

        assert!(
            arrays_approx_equal(&i_minus_n, &i_n, TOLERANCE * 10.0),
            "I_{{-n}}(x) should equal I_n(x)"
        );
    }

    // Property 5: K_{-n}(x) = K_n(x) for all n
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 10.0);

    for n in 0..5 {
        let k_n = bessel_k(n, &x);
        let k_minus_n = bessel_k(-n, &x);

        assert!(
            arrays_approx_equal(&k_minus_n, &k_n, TOLERANCE * 10.0),
            "K_{{-n}}(x) should equal K_n(x)"
        );
    }
}

#[test]
fn test_elliptic_integrals_properties() {
    // Test properties of elliptic integrals

    // Property 1: K(0) = π/2, E(0) = π/2
    let zero = Array::from_vec(vec![0.0]);
    let k_at_zero = ellipk(&zero);
    let e_at_zero = ellipe(&zero);
    let pi_half = std::f64::consts::PI / 2.0;

    assert_abs_diff_eq!(k_at_zero.get(&[0]).unwrap(), pi_half, epsilon = TOLERANCE);
    assert_abs_diff_eq!(e_at_zero.get(&[0]).unwrap(), pi_half, epsilon = TOLERANCE);

    // Property 2: E(1) = 1
    let one = Array::from_vec(vec![1.0]);
    let e_at_one = ellipe(&one);

    assert_abs_diff_eq!(e_at_one.get(&[0]).unwrap(), 1.0, epsilon = TOLERANCE);

    // Property 3: K(m) is always ≥ π/2 for 0 ≤ m < 1
    let m_values = random_array_in_range(SAMPLE_SIZE, 0.0, 0.99);
    let k_values = ellipk(&m_values);

    for i in 0..SAMPLE_SIZE {
        assert!(
            k_values.get(&[i]).unwrap() >= pi_half,
            "K(m) should be ≥ π/2 for 0 ≤ m < 1"
        );
    }

    // Property 4: E(m) is always ≤ π/2 for 0 ≤ m ≤ 1
    let m_values = random_array_in_range(SAMPLE_SIZE, 0.0, 0.99);
    let e_values = ellipe(&m_values);

    for i in 0..SAMPLE_SIZE {
        assert!(
            e_values.get(&[i]).unwrap() <= pi_half,
            "E(m) should be ≤ π/2 for 0 ≤ m ≤ 1"
        );
    }
}

#[test]
// Previously ignored, now passes
fn test_compound_special_functions() {
    // Test interactions between different special functions

    // Property 1: erfinv(erf(x)) = x
    let x = random_array_in_range(SAMPLE_SIZE, -2.0, 2.0);
    let erf_x = erf(&x);
    let erfinv_erf_x = erfinv(&erf_x);

    assert!(
        arrays_approx_equal(&erfinv_erf_x, &x, 0.2),
        "erfinv(erf(x)) should equal x"
    );

    // Property 2: exp(gammaln(x)) = gamma(x)
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);
    let gammaln_x = gammaln(&x);
    let exp_gammaln_x = gammaln_x.map(|v| v.exp());
    let gamma_x = gamma(&x);

    assert!(
        arrays_approx_equal(&exp_gammaln_x, &gamma_x, TOLERANCE * 10.0),
        "exp(gammaln(x)) should equal gamma(x)"
    );

    // Property 3: digamma(x) = d/dx ln(gamma(x))
    // We can approximate this derivative using central difference
    let x = random_array_in_range(SAMPLE_SIZE, 1.0, 10.0);
    let h = 1e-6;

    let x_plus_h = x.add_scalar(h);
    let x_minus_h = x.add_scalar(-h);

    let gammaln_x_plus_h = gammaln(&x_plus_h);
    let gammaln_x_minus_h = gammaln(&x_minus_h);

    // Central difference approximation of the derivative
    let difference = Array::from_vec(
        gammaln_x_plus_h
            .to_vec()
            .iter()
            .zip(gammaln_x_minus_h.to_vec().iter())
            .map(|(&plus_h, &minus_h)| plus_h - minus_h)
            .collect(),
    );
    let derivative_approx = difference.multiply_scalar(1.0 / (2.0 * h));

    let digamma_x = digamma(&x);

    // Use a larger tolerance for the derivative approximation
    assert!(
        arrays_approx_equal(&derivative_approx, &digamma_x, 1e-4),
        "digamma(x) should approximately equal the derivative of ln(gamma(x))"
    );
}

#[test]
fn test_bessel_recurrence_relations() {
    // Test recurrence relations for Bessel functions

    // Recurrence relation: J_{n-1}(x) + J_{n+1}(x) = (2n/x) * J_n(x)
    let x = random_array_in_range(SAMPLE_SIZE, 1.0, 10.0); // Avoid x near 0 to prevent division issues

    for n in 1..4 {
        let j_n_minus_1 = bessel_j(n - 1, &x);
        let j_n = bessel_j(n, &x);
        let j_n_plus_1 = bessel_j(n + 1, &x);

        let left_side = Array::from_vec(
            j_n_minus_1
                .to_vec()
                .iter()
                .zip(j_n_plus_1.to_vec().iter())
                .map(|(&minus_1, &plus_1)| minus_1 + plus_1)
                .collect(),
        );
        let factor_vec = x.map(|v| (2.0 * n as f64) / v);
        let right_side = Array::from_vec(
            factor_vec
                .to_vec()
                .iter()
                .zip(j_n.to_vec().iter())
                .map(|(&factor, &j_val)| factor * j_val)
                .collect(),
        );

        // Use a larger tolerance for this relation as it's more sensitive to numerical precision
        assert!(
            arrays_approx_equal(&left_side, &right_side, 1e-4),
            "Bessel function recurrence relation should hold"
        );
    }

    // Similar recurrence relation for modified Bessel functions
    // I_{n-1}(x) - I_{n+1}(x) = (2n/x) * I_n(x)
    let x = random_array_in_range(SAMPLE_SIZE, 1.0, 10.0);

    for n in 1..4 {
        let i_n_minus_1 = bessel_i(n - 1, &x);
        let i_n = bessel_i(n, &x);
        let i_n_plus_1 = bessel_i(n + 1, &x);

        let left_side = Array::from_vec(
            i_n_minus_1
                .to_vec()
                .iter()
                .zip(i_n_plus_1.to_vec().iter())
                .map(|(&minus_1, &plus_1)| minus_1 - plus_1)
                .collect(),
        );
        let factor_vec = x.map(|v| (2.0 * n as f64) / v);
        let right_side = Array::from_vec(
            factor_vec
                .to_vec()
                .iter()
                .zip(i_n.to_vec().iter())
                .map(|(&factor, &i_val)| factor * i_val)
                .collect(),
        );

        // Use a larger tolerance for this relation
        assert!(
            arrays_approx_equal(&left_side, &right_side, 1e-4),
            "Modified Bessel function recurrence relation should hold"
        );
    }
}

#[test]
// Previously ignored, now passes
fn test_special_function_limits() {
    // Test limiting behavior of special functions

    // Test erf(x) approaches 1 as x becomes large
    let large_x = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    let erf_large_x = erf(&large_x);

    for i in 0..4 {
        assert!(
            erf_large_x.get(&[i]).unwrap() > 0.99999,
            "erf(x) should approach 1 for large x"
        );
    }

    // Test erfc(x) approaches 0 as x becomes large
    let erfc_large_x = erfc(&large_x);

    for i in 0..4 {
        assert!(
            erfc_large_x.get(&[i]).unwrap() < 0.00001,
            "erfc(x) should approach 0 for large x"
        );
    }

    // Test gamma(x) grows rapidly for large x
    let x_values = Array::from_vec(vec![10.0, 11.0, 12.0, 13.0]);
    let gamma_x = gamma(&x_values);
    let factorial_values = vec![
        362880.0,    // gamma(10) = 9!
        3628800.0,   // gamma(11) = 10!
        39916800.0,  // gamma(12) = 11!
        479001600.0, // gamma(13) = 12!
    ];

    for (i, &expected) in factorial_values.iter().enumerate() {
        assert_relative_eq!(
            gamma_x.get(&[i]).unwrap(),
            expected,
            epsilon = expected * 1e-8
        );
    }

    // Test K_n(x) approaches 0 as x becomes large
    let large_x = Array::from_vec(vec![15.0, 20.0, 25.0]);

    for n in 0..3 {
        let k_n_large_x = bessel_k(n, &large_x);

        for i in 0..3 {
            assert!(
                k_n_large_x.get(&[i]).unwrap() < 0.001,
                "K_n(x) should approach 0 for large x"
            );
        }
    }
}

#[test]
fn test_special_function_symmetry() {
    // Test symmetry properties of special functions

    // Test erf is an odd function: erf(-x) = -erf(x)
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 5.0);
    let neg_x = x.map(|v| -v);

    let erf_x = erf(&x);
    let erf_neg_x = erf(&neg_x);
    let neg_erf_x = erf_x.map(|v| -v);

    assert!(
        arrays_approx_equal(&erf_neg_x, &neg_erf_x, TOLERANCE),
        "erf(-x) should equal -erf(x)"
    );

    // Test gamma function recurrence: gamma(x+1) = x * gamma(x)
    let x = random_array_in_range(SAMPLE_SIZE, 0.5, 10.0);
    let x_plus_1 = x.add_scalar(1.0);

    let gamma_x = gamma(&x);
    let gamma_x_plus_1 = gamma(&x_plus_1);
    let x_gamma_x = Array::from_vec(
        x.to_vec()
            .iter()
            .zip(gamma_x.to_vec().iter())
            .map(|(&x_val, &gamma_val)| x_val * gamma_val)
            .collect(),
    );

    assert!(
        arrays_approx_equal(&gamma_x_plus_1, &x_gamma_x, TOLERANCE * 100.0),
        "gamma(x+1) should equal x * gamma(x)"
    );

    // Test J_(-n)(x) = (-1)^n * J_n(x) for integer n
    let x = random_array_in_range(10, 0.5, 5.0);

    for n in 0..5 {
        let j_n = bessel_j(n, &x);
        let j_neg_n = bessel_j(-n, &x);

        if n % 2 == 0 {
            // For even n, J_(-n)(x) = J_n(x)
            assert!(
                arrays_approx_equal(&j_neg_n, &j_n, TOLERANCE * 10.0),
                "J_(-n)(x) should equal J_n(x) for even n"
            );
        } else {
            // For odd n, J_(-n)(x) = -J_n(x)
            let neg_j_n = j_n.map(|v| -v);
            assert!(
                arrays_approx_equal(&j_neg_n, &neg_j_n, TOLERANCE * 10.0),
                "J_(-n)(x) should equal -J_n(x) for odd n"
            );
        }
    }
}

#[test]
// Previously ignored, now passes
fn test_special_functions_with_vector_args() {
    // Test functions that take vector arguments

    // Test gammainc with vector arguments
    let a = random_array_in_range(SAMPLE_SIZE, 0.5, 5.0);
    let x = random_array_in_range(SAMPLE_SIZE, 0.1, 10.0);

    // For a = 1, gammainc(1, x) = 1 - exp(-x)
    let a_ones = Array::<f64>::full(&[SAMPLE_SIZE], 1.0);
    let gammainc_ones = gammainc(&a_ones, &x).expect("Failed to compute gammainc");
    let expected = x.map(|v| 1.0 - (-v).exp());

    assert!(
        arrays_approx_equal(&gammainc_ones, &expected, 0.1),
        "gammainc(1, x) should equal 1 - exp(-x)"
    );

    // Test general case properties
    let gammainc_result = gammainc(&a, &x).expect("Failed to compute gammainc");

    // Check that all values are between 0 and 1
    for i in 0..SAMPLE_SIZE {
        let val = gammainc_result.get(&[i]).unwrap();
        assert!(
            val >= 0.0 && val <= 1.0,
            "gammainc values should be between 0 and 1"
        );
    }

    // Check that gammainc is increasing with x for fixed a
    let a_fixed = Array::<f64>::full(&[5], 2.0);
    let x_increasing = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let gammainc_increasing =
        gammainc(&a_fixed, &x_increasing).expect("Failed to compute gammainc");
    let gammainc_vec = gammainc_increasing.to_vec();

    for i in 1..5 {
        assert!(
            gammainc_vec[i] > gammainc_vec[i - 1],
            "gammainc should be increasing with x for fixed a"
        );
    }
}

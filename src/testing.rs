//! Testing Utilities for NumRS2
//!
//! This module provides comprehensive testing utilities similar to NumPy's `testing` module.
//! It includes functions for comparing arrays, checking numerical properties, and validating
//! computations with appropriate tolerances for floating-point arithmetic.

use crate::array::Array;
use crate::error::Result;
use num_traits::{Float, Zero};
use std::fmt::Debug;

/// Configuration for array comparison tolerances
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    /// Relative tolerance parameter
    pub rtol: f64,
    /// Absolute tolerance parameter
    pub atol: f64,
    /// Whether to check for equal NaN values
    pub equal_nan: bool,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-7,
            atol: 0.0,
            equal_nan: false,
        }
    }
}

/// Test result information
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Whether the test passed
    pub passed: bool,
    /// Detailed message about the test result
    pub message: String,
    /// Maximum absolute difference found
    pub max_abs_diff: Option<f64>,
    /// Maximum relative difference found
    pub max_rel_diff: Option<f64>,
    /// Number of mismatched elements
    pub mismatch_count: usize,
}

impl TestResult {
    /// Create a successful test result
    pub fn success(message: &str) -> Self {
        Self {
            passed: true,
            message: message.to_string(),
            max_abs_diff: None,
            max_rel_diff: None,
            mismatch_count: 0,
        }
    }

    /// Create a failed test result
    pub fn failure(message: &str) -> Self {
        Self {
            passed: false,
            message: message.to_string(),
            max_abs_diff: None,
            max_rel_diff: None,
            mismatch_count: 0,
        }
    }

    /// Create a detailed comparison result
    pub fn comparison_result(
        passed: bool,
        message: &str,
        max_abs_diff: f64,
        max_rel_diff: f64,
        mismatch_count: usize,
    ) -> Self {
        Self {
            passed,
            message: message.to_string(),
            max_abs_diff: Some(max_abs_diff),
            max_rel_diff: Some(max_rel_diff),
            mismatch_count,
        }
    }
}

/// Assert that arrays are equal within tolerance
///
/// This is the primary function for comparing arrays with floating-point tolerances.
///
/// # Arguments
/// * `actual` - The computed array
/// * `desired` - The expected array
/// * `config` - Tolerance configuration
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::{assert_array_almost_equal, ToleranceConfig};
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![1.0000001, 2.0000001, 3.0000001]);
/// let config = ToleranceConfig::default();
///
/// let result = assert_array_almost_equal(&a, &b, &config).unwrap();
/// assert!(result.passed);
/// ```
pub fn assert_array_almost_equal<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    config: &ToleranceConfig,
) -> Result<TestResult>
where
    T: Float + Debug + Clone,
{
    // Check shapes match
    if actual.shape() != desired.shape() {
        return Ok(TestResult::failure(&format!(
            "Arrays have different shapes: actual {:?} vs desired {:?}",
            actual.shape(),
            desired.shape()
        )));
    }

    let actual_vec = actual.to_vec();
    let desired_vec = desired.to_vec();

    let mut max_abs_diff = 0.0;
    let mut max_rel_diff = 0.0;
    let mut mismatch_count = 0;
    let mut first_mismatch: Option<(usize, T, T)> = None;

    for (i, (&a_val, &d_val)) in actual_vec.iter().zip(desired_vec.iter()).enumerate() {
        // Handle NaN cases
        if a_val.is_nan() && d_val.is_nan() {
            if config.equal_nan {
                continue;
            } else {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, a_val, d_val));
                }
                continue;
            }
        }

        if a_val.is_nan() || d_val.is_nan() {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, a_val, d_val));
            }
            continue;
        }

        // Handle infinity cases
        if a_val.is_infinite() && d_val.is_infinite() {
            if a_val.is_sign_positive() == d_val.is_sign_positive() {
                continue;
            } else {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, a_val, d_val));
                }
                continue;
            }
        }

        if a_val.is_infinite() || d_val.is_infinite() {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, a_val, d_val));
            }
            continue;
        }

        // Compute absolute and relative differences
        let abs_diff = (a_val - d_val).abs();
        let tolerance = T::from(config.atol).unwrap() + T::from(config.rtol).unwrap() * d_val.abs();

        max_abs_diff = max_abs_diff.max(abs_diff.to_f64().unwrap_or(f64::INFINITY));

        if !d_val.is_zero() {
            let rel_diff = (abs_diff / d_val.abs()).to_f64().unwrap_or(f64::INFINITY);
            max_rel_diff = max_rel_diff.max(rel_diff);
        }

        if abs_diff > tolerance {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, a_val, d_val));
            }
        }
    }

    let passed = mismatch_count == 0;
    let message = if passed {
        "Arrays are equal within tolerance".to_string()
    } else {
        match first_mismatch {
            Some((index, actual_val, desired_val)) => {
                format!(
                    "Arrays differ at index {}: actual={:?}, desired={:?}. {} elements differ (max_abs_diff={:.6e}, max_rel_diff={:.6e})",
                    index, actual_val, desired_val, mismatch_count, max_abs_diff, max_rel_diff
                )
            }
            None => format!("{} elements differ", mismatch_count),
        }
    };

    Ok(TestResult::comparison_result(
        passed,
        &message,
        max_abs_diff,
        max_rel_diff,
        mismatch_count,
    ))
}

/// Assert that arrays are exactly equal
///
/// # Arguments
/// * `actual` - The computed array
/// * `desired` - The expected array
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::assert_array_equal;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![1, 2, 3]);
///
/// let result = assert_array_equal(&a, &b).unwrap();
/// assert!(result.passed);
/// ```
pub fn assert_array_equal<T>(actual: &Array<T>, desired: &Array<T>) -> Result<TestResult>
where
    T: PartialEq + Debug + Clone,
{
    // Check shapes match
    if actual.shape() != desired.shape() {
        return Ok(TestResult::failure(&format!(
            "Arrays have different shapes: actual {:?} vs desired {:?}",
            actual.shape(),
            desired.shape()
        )));
    }

    let actual_vec = actual.to_vec();
    let desired_vec = desired.to_vec();

    for (i, (a_val, d_val)) in actual_vec.iter().zip(desired_vec.iter()).enumerate() {
        if a_val != d_val {
            return Ok(TestResult::failure(&format!(
                "Arrays differ at index {}: actual={:?}, desired={:?}",
                i, a_val, d_val
            )));
        }
    }

    Ok(TestResult::success("Arrays are exactly equal"))
}

/// Assert that all elements of an array are finite
///
/// # Arguments
/// * `array` - The array to check
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::assert_array_all_finite;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = assert_array_all_finite(&a).unwrap();
/// assert!(result.passed);
/// ```
pub fn assert_array_all_finite<T>(array: &Array<T>) -> Result<TestResult>
where
    T: Float + Debug + Clone,
{
    let data = array.to_vec();

    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Ok(TestResult::failure(&format!(
                "Array contains non-finite value {:?} at index {}",
                val, i
            )));
        }
    }

    Ok(TestResult::success("All array elements are finite"))
}

/// Assert that an array contains no NaN values
///
/// # Arguments
/// * `array` - The array to check
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::assert_array_no_nan;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = assert_array_no_nan(&a).unwrap();
/// assert!(result.passed);
/// ```
pub fn assert_array_no_nan<T>(array: &Array<T>) -> Result<TestResult>
where
    T: Float + Debug + Clone,
{
    let data = array.to_vec();

    for (i, &val) in data.iter().enumerate() {
        if val.is_nan() {
            return Ok(TestResult::failure(&format!(
                "Array contains NaN value at index {}",
                i
            )));
        }
    }

    Ok(TestResult::success("Array contains no NaN values"))
}

/// Assert that arrays have the same shape
///
/// # Arguments
/// * `actual` - The computed array
/// * `desired` - The expected array
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::assert_array_same_shape;
///
/// let a: Array<f64> = Array::zeros(&[2, 3]);
/// let b: Array<f64> = Array::ones(&[2, 3]);
///
/// let result = assert_array_same_shape(&a, &b).expect("assert_array_same_shape failed");
/// assert!(result.passed);
/// ```
pub fn assert_array_same_shape<T, U>(actual: &Array<T>, desired: &Array<U>) -> Result<TestResult>
where
    T: Clone,
    U: Clone,
{
    if actual.shape() == desired.shape() {
        Ok(TestResult::success("Arrays have the same shape"))
    } else {
        Ok(TestResult::failure(&format!(
            "Arrays have different shapes: actual {:?} vs desired {:?}",
            actual.shape(),
            desired.shape()
        )))
    }
}

/// Assert that a scalar value is approximately equal to expected value
///
/// # Arguments
/// * `actual` - The computed value
/// * `desired` - The expected value
/// * `config` - Tolerance configuration
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::{assert_scalar_almost_equal, ToleranceConfig};
///
/// let mut config = ToleranceConfig::default();
/// config.atol = 1e-4;
/// let result = assert_scalar_almost_equal(3.14159, 3.1416, &config).expect("assert_scalar_almost_equal failed");
/// assert!(result.passed);
/// ```
pub fn assert_scalar_almost_equal<T>(
    actual: T,
    desired: T,
    config: &ToleranceConfig,
) -> Result<TestResult>
where
    T: Float + Debug,
{
    // Handle NaN cases
    if actual.is_nan() && desired.is_nan() {
        if config.equal_nan {
            return Ok(TestResult::success("Both values are NaN"));
        } else {
            return Ok(TestResult::failure(
                "Both values are NaN but equal_nan is false",
            ));
        }
    }

    if actual.is_nan() || desired.is_nan() {
        return Ok(TestResult::failure(&format!(
            "Values differ: actual={:?}, desired={:?}",
            actual, desired
        )));
    }

    // Handle infinity cases
    if actual.is_infinite() && desired.is_infinite() {
        if actual.is_sign_positive() == desired.is_sign_positive() {
            return Ok(TestResult::success("Both values are the same infinity"));
        } else {
            return Ok(TestResult::failure(&format!(
                "Values are different infinities: actual={:?}, desired={:?}",
                actual, desired
            )));
        }
    }

    if actual.is_infinite() || desired.is_infinite() {
        return Ok(TestResult::failure(&format!(
            "One value is infinite: actual={:?}, desired={:?}",
            actual, desired
        )));
    }

    // Compute tolerance
    let abs_diff = (actual - desired).abs();
    let tolerance = T::from(config.atol).unwrap() + T::from(config.rtol).unwrap() * desired.abs();

    if abs_diff <= tolerance {
        Ok(TestResult::success("Values are equal within tolerance"))
    } else {
        let abs_diff_f64 = abs_diff.to_f64().unwrap_or(f64::INFINITY);
        let rel_diff = if !desired.is_zero() {
            (abs_diff / desired.abs()).to_f64().unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };

        Ok(TestResult::comparison_result(
            false,
            &format!(
                "Values differ: actual={:?}, desired={:?}, abs_diff={:.6e}, rel_diff={:.6e}",
                actual, desired, abs_diff_f64, rel_diff
            ),
            abs_diff_f64,
            rel_diff,
            1,
        ))
    }
}

/// Utility function to check if an array contains only valid finite numbers
///
/// # Arguments
/// * `array` - The array to check
///
/// # Returns
/// * `true` if all elements are finite, `false` otherwise
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::is_finite_array;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// assert!(is_finite_array(&a));
/// ```
pub fn is_finite_array<T>(array: &Array<T>) -> bool
where
    T: Float,
{
    array.to_vec().iter().all(|&x| x.is_finite())
}

/// Utility function to check if arrays are close within tolerance
///
/// This is a convenience function that returns a boolean instead of a TestResult.
///
/// # Arguments
/// * `actual` - The computed array
/// * `desired` - The expected array
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::arrays_close;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![1.000001, 2.000001, 3.000001]);
///
/// assert!(arrays_close(&a, &b, 1e-5, 1e-8));
/// ```
pub fn arrays_close<T>(actual: &Array<T>, desired: &Array<T>, rtol: f64, atol: f64) -> bool
where
    T: Float + Debug + Clone,
{
    let config = ToleranceConfig {
        rtol,
        atol,
        equal_nan: false,
    };
    match assert_array_almost_equal(actual, desired, &config) {
        Ok(result) => result.passed,
        Err(_) => false,
    }
}

/// Count the number of non-zero elements in an array
///
/// # Arguments
/// * `array` - The array to check
///
/// # Example
/// ```
/// use numrs2::prelude::*;
/// use numrs2::testing::count_nonzero;
///
/// let a = Array::from_vec(vec![0, 1, 0, 2, 3]);
/// assert_eq!(count_nonzero(&a), 3);
/// ```
pub fn count_nonzero<T>(array: &Array<T>) -> usize
where
    T: Zero + PartialEq + Clone,
{
    array.to_vec().iter().filter(|&x| !x.is_zero()).count()
}

/// Generate a summary report of test results
///
/// # Arguments
/// * `results` - Vector of test results
///
/// # Returns
/// * A formatted string containing the test summary
pub fn test_summary(results: &[TestResult]) -> String {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let mut summary = format!("Test Summary: {}/{} tests passed\n", passed, total);

    if failed > 0 {
        summary.push_str(&format!("Failed tests ({}):\n", failed));
        for (i, result) in results.iter().enumerate() {
            if !result.passed {
                summary.push_str(&format!("  Test {}: {}\n", i + 1, result.message));
            }
        }
    }

    if passed == total {
        summary.push_str("All tests passed successfully!\n");
    }

    summary
}

/// Convenience macro for running multiple test assertions
///
/// # Example
/// ```no_run
/// // This example requires the run_tests macro which is not available in doctests
/// use numrs2::prelude::*;
/// use numrs2::testing::{assert_array_equal, assert_array_almost_equal, ToleranceConfig};
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![1, 2, 3]);
/// let c = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let d = Array::from_vec(vec![1.000001, 2.000001, 3.000001]);
///
/// // Run individual assertions
/// let result1 = assert_array_equal(&a, &b).expect("equal test failed");
/// let result2 = assert_array_almost_equal(&c, &d, &ToleranceConfig::default()).expect("almost equal test failed");
///
/// assert!(result1.passed && result2.passed);
/// ```
#[macro_export]
macro_rules! run_tests {
    ($($test:expr),* $(,)?) => {
        {
            let mut results = Vec::new();
            $(
                match $test {
                    Ok(result) => results.push(result),
                    Err(e) => results.push($crate::testing::TestResult::failure(&format!("Test error: {}", e))),
                }
            )*
            results
        }
    };
}

/// Specialized tolerance configurations for common use cases
pub mod tolerances {
    use super::ToleranceConfig;

    /// Very strict tolerance for exact comparisons
    pub fn strict() -> ToleranceConfig {
        ToleranceConfig {
            rtol: 1e-15,
            atol: 1e-15,
            equal_nan: false,
        }
    }

    /// Default tolerance for most floating-point comparisons
    pub fn default() -> ToleranceConfig {
        ToleranceConfig::default()
    }

    /// Relaxed tolerance for approximate comparisons
    pub fn relaxed() -> ToleranceConfig {
        ToleranceConfig {
            rtol: 1e-5,
            atol: 1e-8,
            equal_nan: false,
        }
    }

    /// Very relaxed tolerance for rough approximations
    pub fn loose() -> ToleranceConfig {
        ToleranceConfig {
            rtol: 1e-3,
            atol: 1e-6,
            equal_nan: false,
        }
    }

    /// Allow NaN values to be considered equal
    pub fn with_nan() -> ToleranceConfig {
        ToleranceConfig {
            rtol: 1e-7,
            atol: 0.0,
            equal_nan: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_array_almost_equal() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0000001, 2.0000001, 3.0000001]);

        let config = ToleranceConfig::default();
        let result = assert_array_almost_equal(&a, &b, &config).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_assert_array_equal() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 3]);

        let result = assert_array_equal(&a, &b).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_assert_array_all_finite() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = assert_array_all_finite(&a).unwrap();
        assert!(result.passed);

        let b = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        let result = assert_array_all_finite(&b).unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn test_count_nonzero() {
        let a = Array::from_vec(vec![0, 1, 0, 2, 3]);
        assert_eq!(count_nonzero(&a), 3);
    }

    #[test]
    fn test_run_tests_macro() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 3]);

        let results = run_tests!(assert_array_equal(&a, &b));

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }
}

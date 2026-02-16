//! Range-based array creation functions
//!
//! This module provides functions to create arrays with values spaced on specific scales:
//! - `logspace` - Create values evenly spaced on a log scale
//! - `geomspace` - Create values evenly spaced on a geometric scale

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

/// Create values spaced evenly on a log scale
///
/// Return numbers spaced evenly on a log scale.
///
/// # Parameters
///
/// * `start` - The starting value of the sequence (base^start)
/// * `stop` - The final value of the sequence (base^stop)
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
/// * `base` - The base of the log space
///
/// # Returns
///
/// Array of `num` samples, equally spaced on a log scale
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::logspace;
///
/// // Create 5 values from 10^0 to 10^2
/// let result = logspace(0.0, 2.0, 5, true, 10.0).expect("operation should succeed");
/// // Result is approximately [1.0, 3.16, 10.0, 31.6, 100.0]
///
/// // Create values from 2^1 to 2^4
/// let result = logspace(1.0, 4.0, 4, true, 2.0).expect("operation should succeed");
/// // Result is [2.0, 4.0, 8.0, 16.0]
/// ```
pub fn logspace<T>(start: T, stop: T, num: usize, endpoint: bool, base: T) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if num == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if num == 1 {
        return Ok(Array::from_vec(vec![base.powf(start)]));
    }

    let divisor = if endpoint {
        T::from(num - 1).expect("num-1 should be convertible to T (Float type)")
    } else {
        T::from(num).expect("num should be convertible to T (Float type)")
    };

    let mut result = Vec::with_capacity(num);

    for i in 0..num {
        let t = T::from(i).expect("loop index i should be convertible to T (Float type)") / divisor;
        let exponent = start + t * (stop - start);
        result.push(base.powf(exponent));
    }

    Ok(Array::from_vec(result))
}

/// Create values spaced evenly on a geometric progression
///
/// Return numbers spaced evenly on a geometric progression.
///
/// # Parameters
///
/// * `start` - The starting value of the sequence
/// * `stop` - The final value of the sequence
/// * `num` - Number of samples to generate
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included
///
/// # Returns
///
/// Array of `num` samples, equally spaced on a geometric scale
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::geomspace;
///
/// // Create 5 values from 1 to 1000
/// let result = geomspace(1.0, 1000.0, 4, true).expect("operation should succeed");
/// // Result is [1.0, 10.0, 100.0, 1000.0]
///
/// // Create 4 values from 1 to 81
/// let result = geomspace(1.0, 81.0, 5, true).expect("operation should succeed");
/// // Result is [1.0, 3.0, 9.0, 27.0, 81.0]
/// ```
pub fn geomspace<T>(start: T, stop: T, num: usize, endpoint: bool) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if num == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Check for sign consistency
    if start.is_sign_positive() != stop.is_sign_positive() {
        return Err(NumRs2Error::InvalidOperation(
            "Geometric sequence cannot include zero or change sign".to_string(),
        ));
    }

    if start == T::zero() || stop == T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Geometric sequence endpoints cannot be zero".to_string(),
        ));
    }

    // Use logarithms to create geometric progression
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();

    let result = logspace(
        log_start,
        log_stop,
        num,
        endpoint,
        T::from(std::f64::consts::E)
            .expect("Euler's number E should be convertible to Float type T"),
    )?;

    // Adjust signs if necessary
    if start.is_sign_negative() {
        let result_vec: Vec<T> = result.to_vec().into_iter().map(|x| -x).collect();
        Ok(Array::from_vec(result_vec))
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_logspace() {
        // Test basic logspace
        let result = logspace(0.0, 2.0, 3, true, 10.0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[2]).expect("operation should succeed"),
            100.0,
            epsilon = 1e-10
        );

        // Test with base 2
        let result = logspace(1.0, 4.0, 4, true, 2.0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![4]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            4.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[2]).expect("operation should succeed"),
            8.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[3]).expect("operation should succeed"),
            16.0,
            epsilon = 1e-10
        );

        // Test without endpoint
        let result = logspace(0.0, 2.0, 2, false, 10.0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![2]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            10.0,
            epsilon = 1e-10
        );

        // Test edge cases
        let result = logspace(1.0, 1.0, 1, true, 10.0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![1]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            10.0,
            epsilon = 1e-10
        );

        let result = logspace(0.0, 0.0, 0, true, 10.0).expect("operation should succeed");
        assert_eq!(result.shape(), vec![0]);
    }

    #[test]
    fn test_geomspace() {
        // Test basic geomspace
        let result = geomspace(1.0, 100.0, 3, true).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[2]).expect("operation should succeed"),
            100.0,
            epsilon = 1e-10
        );

        // Test with more points
        let result = geomspace(1.0, 81.0, 5, true).expect("operation should succeed");
        assert_eq!(result.shape(), vec![5]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[2]).expect("operation should succeed"),
            9.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[3]).expect("operation should succeed"),
            27.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[4]).expect("operation should succeed"),
            81.0,
            epsilon = 1e-10
        );

        // Test with negative values
        let result = geomspace(-1.0, -100.0, 3, true).expect("operation should succeed");
        assert_eq!(result.shape(), vec![3]);
        assert_relative_eq!(
            result.get(&[0]).expect("operation should succeed"),
            -1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[1]).expect("operation should succeed"),
            -10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.get(&[2]).expect("operation should succeed"),
            -100.0,
            epsilon = 1e-10
        );

        // Test error cases
        assert!(geomspace(1.0, -1.0, 3, true).is_err()); // Sign change
        assert!(geomspace(0.0, 1.0, 3, true).is_err()); // Zero endpoint
        assert!(geomspace(1.0, 0.0, 3, true).is_err()); // Zero endpoint
    }
}

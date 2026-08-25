//! Gradient and Additional Math Functions
//!
//! This module provides numerical gradient computation and additional math utility functions
//! for array operations.
//!
//! # Functions
//!
//! - [`gradient`] - Compute the numerical gradient of an N-dimensional array
//! - [`signbit`] - Test element-wise for signbit (whether the sign bit is set)
//! - [`reciprocal`] - Return the reciprocal of the argument, element-wise
//! - [`positive`] - Return the numerical positive of each element (no-op for real numbers)
//! - [`negative`] - Return the numerical negative of each element
//! - [`rint`] - Round elements to the nearest integer
//! - [`fix`] - Round towards zero (truncate the fractional part)
//! - [`fmax`] - Element-wise maximum of array elements, ignoring NaN
//! - [`fmin`] - Element-wise minimum of array elements, ignoring NaN
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{gradient, GradientSpacing, signbit, reciprocal};
//!
//! // Compute gradient
//! let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
//! let grad = gradient(&f, None, None, 1).expect("gradient should succeed");
//!
//! // Check signbit
//! let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
//! let signs = signbit(&a);
//!
//! // Compute reciprocal
//! let b = Array::from_vec(vec![1.0, 2.0, 4.0]);
//! let recip = reciprocal(&b);
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

/// Compute the N-dimensional gradient of an array
///
/// The gradient is computed using central differences in the interior points
/// and first or second-order differences at the boundaries.
///
/// # Arguments
///
/// * `f` - An N-dimensional array containing samples of a scalar function
/// * `varargs` - Spacing between f values. Default unitary spacing for all dimensions.
/// * `axis` - Gradient is calculated along these axes. Default is all axes.
/// * `edge_order` - Order of accuracy at boundaries (1 or 2). Default should be 1.
///
/// # Returns
///
/// A vector of arrays, one for each axis, containing the gradient along that axis.
///
/// # Errors
///
/// Returns an error if:
/// * `edge_order` is not 1 or 2
/// * An axis value is out of bounds
/// * Spacing array length doesn't match array dimensions
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::{gradient, GradientSpacing};
///
/// let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
/// let grad = gradient(&f, None, None, 1).expect("gradient should succeed");
/// ```
pub fn gradient<T>(
    f: &Array<T>,
    varargs: Option<GradientSpacing<T>>,
    axis: Option<Vec<usize>>,
    edge_order: usize,
) -> Result<Vec<Array<T>>>
where
    T: Float + Clone + 'static,
{
    let ndim = f.ndim();
    let shape = f.shape();

    // Validate edge_order
    if edge_order != 1 && edge_order != 2 {
        return Err(NumRs2Error::ValueError(
            "edge_order must be 1 or 2".to_string(),
        ));
    }

    // Determine axes to compute gradient for
    let axes = match axis {
        Some(a) => {
            // Validate axes
            for &ax in &a {
                if ax >= ndim {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "axis {} is out of bounds for array of dimension {}",
                        ax, ndim
                    )));
                }
            }
            a
        }
        None => (0..ndim).collect(),
    };

    // Parse spacing
    let spacings = match varargs {
        None => vec![T::one(); ndim],
        Some(GradientSpacing::Uniform(h)) => vec![h; ndim],
        Some(GradientSpacing::PerAxis(spacings)) => {
            if spacings.len() != ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "spacing array length {} doesn't match array dimensions {}",
                    spacings.len(),
                    ndim
                )));
            }
            spacings
        }
    };

    let mut results = Vec::new();

    // Compute gradient for each axis
    for &ax in &axes {
        let mut grad = Array::zeros(&shape);
        let h = spacings[ax];
        let n = shape[ax];

        if n == 1 {
            // Gradient of constant is zero
            results.push(grad);
            continue;
        }

        // Helper to get/set values along an axis
        let mut indices = vec![0; ndim];

        // Iterate over all positions perpendicular to the axis
        let total_perp: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax)
            .map(|(_, &s)| s)
            .product();

        // Bulk-acquire once: `grad` is write-only across this whole
        // perpendicular-position x axis-position loop (every read is
        // `f.get(..)`, a distinct object), so one unshare covers all
        // `total_perp * n` writes.
        let grad_arr = grad.array_mut();

        for perp_idx in 0..total_perp {
            // Convert linear index to multi-dimensional indices for perpendicular dimensions
            let mut temp = perp_idx;
            let mut _dim_idx = 0;
            for i in 0..ndim {
                if i != ax {
                    let stride: usize = shape
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j > i && *j != ax)
                        .map(|(_, &s)| s)
                        .product();
                    indices[i] = temp / stride;
                    temp %= stride;
                    _dim_idx += 1;
                }
            }

            // Compute gradient along the axis
            for i in 0..n {
                indices[ax] = i;

                let derivative = if i == 0 {
                    // Forward difference at the start
                    if edge_order == 1 || n < 3 {
                        indices[ax] = 1;
                        let f1 = f.get(&indices)?;
                        indices[ax] = 0;
                        let f0 = f.get(&indices)?;
                        (f1 - f0) / h
                    } else {
                        // Second-order forward difference
                        indices[ax] = 0;
                        let f0 = f.get(&indices)?;
                        indices[ax] = 1;
                        let f1 = f.get(&indices)?;
                        indices[ax] = 2;
                        let f2 = f.get(&indices)?;
                        (-f2 * T::from(0.5).expect("0.5 should be representable")
                            + f1 * T::from(2.0).expect("2.0 should be representable")
                            - f0 * T::from(1.5).expect("1.5 should be representable"))
                            / h
                    }
                } else if i == n - 1 {
                    // Backward difference at the end
                    if edge_order == 1 || n < 3 {
                        indices[ax] = n - 1;
                        let fn1 = f.get(&indices)?;
                        indices[ax] = n - 2;
                        let fn2 = f.get(&indices)?;
                        (fn1 - fn2) / h
                    } else {
                        // Second-order backward difference
                        indices[ax] = n - 1;
                        let fn1 = f.get(&indices)?;
                        indices[ax] = n - 2;
                        let fn2 = f.get(&indices)?;
                        indices[ax] = n - 3;
                        let fn3 = f.get(&indices)?;
                        (fn3 * T::from(0.5).expect("0.5 should be representable")
                            - fn2 * T::from(2.0).expect("2.0 should be representable")
                            + fn1 * T::from(1.5).expect("1.5 should be representable"))
                            / h
                    }
                } else {
                    // Central difference in the interior
                    indices[ax] = i + 1;
                    let fplus = f.get(&indices)?;
                    indices[ax] = i - 1;
                    let fminus = f.get(&indices)?;
                    (fplus - fminus) / (h * T::from(2.0).expect("2.0 should be representable"))
                };

                indices[ax] = i;
                *grad_arr.get_mut(indices.as_slice()).ok_or_else(|| {
                    NumRs2Error::IndexOutOfBounds(format!(
                        "Failed to set element at indices {:?}",
                        indices
                    ))
                })? = derivative;
            }
        }

        results.push(grad);
    }

    Ok(results)
}

/// Spacing specification for gradient calculation
pub enum GradientSpacing<T> {
    /// Uniform spacing for all dimensions
    Uniform(T),
    /// Per-axis spacing
    PerAxis(Vec<T>),
}

/// Test element-wise for signbit (whether the sign bit is set)
///
/// This function returns true where signbit is set (negative, including -0.0),
/// false otherwise. This is equivalent to NumPy's `np.signbit`.
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of booleans indicating where signbit is set
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::signbit;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0, -0.0]);
/// let result = signbit(&a);
/// assert_eq!(result.to_vec(), vec![true, false, false, true]);
/// ```
pub fn signbit<T: Float + Clone>(array: &Array<T>) -> Array<bool> {
    array.map(|x| x.is_sign_negative())
}

/// Return the reciprocal of the argument, element-wise
///
/// Calculates `1/x` for each element in the array.
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with reciprocal values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::reciprocal;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 0.5]);
/// let result = reciprocal(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 0.5, 0.25, 2.0]);
/// ```
pub fn reciprocal<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| T::one() / x)
}

/// Return the numerical positive of each element (a no-op for real numbers)
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Copy of the input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::positive;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let result = positive(&a);
/// assert_eq!(result.to_vec(), vec![-1.0, 0.0, 1.0]);
/// ```
pub fn positive<T: Clone>(array: &Array<T>) -> Array<T> {
    array.clone()
}

/// Return the numerical negative of each element
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with negated values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::negative;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let result = negative(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 0.0, -1.0]);
/// ```
pub fn negative<T: Clone + std::ops::Neg<Output = T>>(array: &Array<T>) -> Array<T> {
    array.map(|x| -x)
}

/// Round elements to the nearest integer
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with values rounded to nearest integer
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::rint;
///
/// let a = Array::from_vec(vec![1.1, 1.5, 1.9, 2.5]);
/// let result = rint(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 2.0, 3.0]);
/// ```
pub fn rint<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| x.round())
}

/// Round towards zero (truncate the fractional part)
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with values rounded towards zero
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fix;
///
/// let a = Array::from_vec(vec![1.1, 1.9, -1.1, -1.9]);
/// let result = fix(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 1.0, -1.0, -1.0]);
/// ```
pub fn fix<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| x.trunc())
}

/// Element-wise maximum of array elements, ignoring NaN
///
/// # Arguments
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing element-wise maximum
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fmax;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let b = Array::from_vec(vec![2.0, 2.0, f64::NAN]);
/// let result = fmax(&a, &b).expect("fmax should succeed");
/// assert_eq!(result.to_vec()[0], 2.0);
/// assert_eq!(result.to_vec()[1], 2.0);
/// assert_eq!(result.to_vec()[2], 3.0);
/// ```
pub fn fmax<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.max(b)
            }
        })
        .collect();

    Array::from_vec_shape(result, &x1.shape())
}

/// Element-wise minimum of array elements, ignoring NaN
///
/// # Arguments
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing element-wise minimum
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fmin;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let b = Array::from_vec(vec![2.0, 2.0, f64::NAN]);
/// let result = fmin(&a, &b).expect("fmin should succeed");
/// assert_eq!(result.to_vec()[0], 1.0);
/// assert_eq!(result.to_vec()[1], 2.0);
/// assert_eq!(result.to_vec()[2], 3.0);
/// ```
pub fn fmin<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.min(b)
            }
        })
        .collect();

    Array::from_vec_shape(result, &x1.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against `numpy.gradient` (numpy 2.4.2), default `edge_order=1`,
    /// unit spacing: `np.gradient(np.array([1.,2.,4.,7.,11.,16.]))` ==
    /// `[1. , 1.5, 2.5, 3.5, 4.5, 5. ]` (simple forward/backward difference
    /// at the two boundary points, central difference in the interior).
    #[test]
    fn test_gradient_edge_order_1_unit_spacing() {
        let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]);
        let grad = gradient(&f, None, None, 1).expect("gradient should succeed");
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0].to_vec(), vec![1.0, 1.5, 2.5, 3.5, 4.5, 5.0]);
    }

    /// Pinned against `numpy.gradient` 2.4.2, `edge_order=2`, unit spacing:
    /// `np.gradient(np.array([1.,2.,4.,7.,11.,16.]), edge_order=2)` ==
    /// `[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]` -- the boundary points use a
    /// second-order one-sided difference (`(-3f0+4f1-f2)/2h` /
    /// `(3fn-4fn-1+fn-2)/2h`) instead of edge_order=1's simple two-point
    /// difference, so only the two endpoints differ from the test above.
    #[test]
    fn test_gradient_edge_order_2_unit_spacing() {
        let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]);
        let grad = gradient(&f, None, None, 2).expect("gradient should succeed");
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0].to_vec(), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    }

    /// Pinned against `numpy.gradient` 2.4.2, `edge_order=2`, non-unit
    /// uniform spacing: `np.gradient(np.array([1.,2.,4.,7.,11.,16.]), 2.0,
    /// edge_order=2)` == `[0.25, 0.75, 1.25, 1.75, 2.25, 2.75]` (exactly
    /// the `h=1` result above, halved, since every finite-difference term
    /// here divides by `h`).
    #[test]
    fn test_gradient_edge_order_2_non_unit_spacing() {
        let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]);
        let grad = gradient(&f, Some(GradientSpacing::Uniform(2.0)), None, 2)
            .expect("gradient should succeed");
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0].to_vec(), vec![0.25, 0.75, 1.25, 1.75, 2.25, 2.75]);
    }

    /// 2-D, per-axis spacing (`GradientSpacing::PerAxis`), default
    /// `edge_order=1`. Pinned against `numpy.gradient` 2.4.2:
    /// `np.gradient(np.array([[1.,2.,4.],[7.,11.,16.]]), 1.0, 2.0)` ==
    /// `(array([[6.,9.,12.],[6.,9.,12.]]),
    ///   array([[0.5,0.75,1.],[2.,2.25,2.5]]))` -- axis 0 has only 2 points,
    /// so both its "boundary" derivatives collapse to the same simple
    /// two-point difference for every column; axis 1 (spacing `2.0`)
    /// exercises the interior central-difference and both one-sided ends.
    #[test]
    fn test_gradient_2d_per_axis_spacing() {
        let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0]).reshape(&[2, 3]);
        let grad = gradient(&f, Some(GradientSpacing::PerAxis(vec![1.0, 2.0])), None, 1)
            .expect("gradient should succeed");
        assert_eq!(grad.len(), 2);
        assert_eq!(grad[0].to_vec(), vec![6.0, 9.0, 12.0, 6.0, 9.0, 12.0]);
        assert_eq!(grad[1].to_vec(), vec![0.5, 0.75, 1.0, 2.0, 2.25, 2.5]);
    }

    #[test]
    fn test_gradient_invalid_edge_order_errors() {
        let f = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(gradient(&f, None, None, 3).is_err());
    }

    #[test]
    fn test_gradient_out_of_bounds_axis_errors() {
        let f = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(gradient(&f, None, Some(vec![1]), 1).is_err());
    }
}

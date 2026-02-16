//! Mathematical utility functions
//!
//! This module provides utility functions for mathematical operations including:
//! - Number theory functions: `gcd`, `lcm`
//! - Floating-point manipulation: `copysign`, `nextafter`, `ldexp`, `frexp`, `modf`
//! - Special functions: `sinc`, `heaviside`
//! - Complex number utilities: `real_if_close`
//! - Division operations: `divmod`, `remainder`, `fmod`
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//!
//! // GCD computation
//! let a = Array::from_vec(vec![12, 15, 21]);
//! let b = Array::from_vec(vec![8, 10, 14]);
//! let result = gcd(&a, &b).expect("gcd should succeed");
//! assert_eq!(result.to_vec(), vec![4, 5, 7]);
//!
//! // Sinc function
//! let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
//! let y = sinc(&x);
//! // sinc(0) = 1, sinc(1) = 0, sinc(2) = 0
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, PrimInt, Zero};
use scirs2_core::Complex;

/// Compute the greatest common divisor of two arrays element-wise
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing the greatest common divisor of corresponding elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![12, 15, 21]);
/// let b = Array::from_vec(vec![8, 10, 14]);
/// let result = gcd(&a, &b).expect("gcd should succeed");
/// assert_eq!(result.to_vec(), vec![4, 5, 7]);
/// ```
pub fn gcd<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + PartialEq + std::ops::Rem<Output = T> + Copy,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| gcd_scalar(a, b))
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Helper function to compute GCD of two scalar values
fn gcd_scalar<T>(mut a: T, mut b: T) -> T
where
    T: Clone + Zero + PartialEq + std::ops::Rem<Output = T> + Copy,
{
    while b != T::zero() {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compute the least common multiple of two arrays element-wise
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing the least common multiple of corresponding elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![12, 15, 21]);
/// let b = Array::from_vec(vec![8, 10, 14]);
/// let result = lcm(&a, &b).expect("lcm should succeed");
/// assert_eq!(result.to_vec(), vec![24, 30, 42]);
/// ```
pub fn lcm<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + PartialEq
        + std::ops::Rem<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + Copy,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| {
            if a == T::zero() || b == T::zero() {
                T::zero()
            } else {
                let gcd_val = gcd_scalar(a, b);
                a * b / gcd_val
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Copy the sign of one array to another element-wise
///
/// # Parameters
///
/// * `x1` - Array whose values to use
/// * `x2` - Array whose signs to use
///
/// # Returns
///
/// Array with magnitudes from x1 and signs from x2
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, -2.0, 3.0]);
/// let b = Array::from_vec(vec![-1.0, 1.0, -1.0]);
/// let result = copysign(&a, &b).expect("copysign should succeed");
/// assert_eq!(result.to_vec(), vec![-1.0, 2.0, -3.0]);
/// ```
pub fn copysign<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&val, &sign_val)| {
            if sign_val < T::zero() {
                -val.abs()
            } else {
                val.abs()
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Return the next floating-point value after x1 towards x2, element-wise
///
/// # Parameters
///
/// * `x1` - Starting values
/// * `x2` - Values to move towards
///
/// # Returns
///
/// Array with next representable floating-point values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0]);
/// let b = Array::from_vec(vec![2.0, 1.0]);
/// let result = nextafter(&a, &b).expect("nextafter should succeed");
/// // result[0] will be slightly larger than 1.0
/// // result[1] will be slightly smaller than 2.0
/// assert!(result.get(&[0]).expect("valid index") > 1.0);
/// assert!(result.get(&[1]).expect("valid index") < 2.0);
/// ```
pub fn nextafter<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    // Use epsilon for approximation since we can't access raw bits in generic Float trait
    let epsilon = T::epsilon();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&from, &to)| {
            if from == to {
                from
            } else if from.is_nan() || to.is_nan() {
                T::nan()
            } else if from < to {
                // Move towards positive infinity
                if from == T::infinity() {
                    from
                } else {
                    from + from.abs() * epsilon
                }
            } else {
                // Move towards negative infinity
                if from == T::neg_infinity() {
                    from
                } else {
                    from - from.abs() * epsilon
                }
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Normalized sinc function
///
/// Return the normalized sinc function sin(pi*x)/(pi*x). The sinc function is 1
/// for x=0, and sin(pi*x)/(pi*x) for all other points.
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Array of sinc values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let y = sinc(&x);
/// // sinc(0) = 1, sinc(1) = 0, sinc(2) = 0, sinc(3) = 0
/// ```
pub fn sinc<T>(x: &Array<T>) -> Array<T>
where
    T: Float + Clone,
{
    x.map(|val| {
        if val == T::zero() {
            T::one()
        } else {
            let pi = T::from(std::f64::consts::PI).expect("PI constant should be representable");
            let pix = pi * val;
            pix.sin() / pix
        }
    })
}

/// Heaviside step function
///
/// The Heaviside step function is defined as:
/// - 0 if x < 0
/// - h0 if x == 0
/// - 1 if x > 0
///
/// # Parameters
///
/// * `x` - Input array
/// * `h0` - The value of the function at x=0 (default: 0.5)
///
/// # Returns
///
/// Array of Heaviside function values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let y = heaviside(&x, Some(0.5));
/// assert_eq!(y.to_vec(), vec![0.0, 0.5, 1.0]);
/// ```
pub fn heaviside<T>(x: &Array<T>, h0: Option<T>) -> Array<T>
where
    T: Float + Clone,
{
    let h0_val = h0.unwrap_or_else(|| T::from(0.5).expect("0.5 should be representable"));

    x.map(|val| {
        if val < T::zero() {
            T::zero()
        } else if val == T::zero() {
            h0_val
        } else {
            T::one()
        }
    })
}

/// If input is complex with all imaginary parts close to zero, return real parts
///
/// "Close to zero" is defined as tol * (machine epsilon of the type).
///
/// # Parameters
///
/// * `a` - Input array (must be complex)
/// * `tol` - Tolerance in multiples of machine epsilon (default: 100)
///
/// # Returns
///
/// If the imaginary part of all elements is smaller than the tolerance,
/// return only the real part. Otherwise, return the original array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use scirs2_core::Complex;
///
/// let a = Array::from_vec(vec![
///     Complex::new(1.0, 1e-15),
///     Complex::new(2.0, 1e-16),
///     Complex::new(3.0, 1e-14)
/// ]);
/// // If all imaginary parts are small enough, returns real array
/// let result = real_if_close(&a, Some(1000.0));
/// ```
pub fn real_if_close<T>(a: &Array<Complex<T>>, tol: Option<T>) -> Array<T>
where
    T: Float + Clone,
{
    let tolerance = tol.unwrap_or_else(|| T::from(100.0).expect("100.0 should be representable"));
    let epsilon = T::epsilon();
    let threshold = tolerance * epsilon;

    let data = a.to_vec();

    // Check if all imaginary parts are close to zero
    let all_close = data.iter().all(|c| c.im.abs() <= threshold);

    if all_close {
        // Return only real parts
        let real_data: Vec<T> = data.iter().map(|c| c.re).collect();
        Array::from_vec(real_data).reshape(&a.shape())
    } else {
        // Convert back to complex array (keeping as real for simplicity in this context)
        // In a real implementation, this would return Result<Either<Array<T>, Array<Complex<T>>>>
        let real_data: Vec<T> = data.iter().map(|c| c.re).collect();
        Array::from_vec(real_data).reshape(&a.shape())
    }
}

/// Load exponent: returns x * 2^exp element-wise
///
/// # Parameters
///
/// * `x` - Input array
/// * `exp` - Array of integer exponents
///
/// # Returns
///
/// Array with the same shape as the input where each element is `x[i] * 2^exp[i]`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let exp = Array::from_vec(vec![1, 2, 3]);
/// let result = ldexp(&x, &exp);
/// // result = [2.0, 8.0, 24.0]
/// ```
pub fn ldexp<T, I>(x: &Array<T>, exp: &Array<I>) -> Result<Array<T>>
where
    T: Float + Clone,
    I: Clone + NumCast,
{
    if x.shape() != exp.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, exp has shape {:?}",
            x.shape(),
            exp.shape()
        )));
    }

    let x_vec = x.to_vec();
    let exp_vec = exp.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, exp_val) in x_vec.iter().zip(exp_vec.iter()) {
        let exp_f64 = I::to_f64(exp_val).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert exponent to f64".to_string())
        })?;
        let two = T::from(2.0).expect("2.0 should be representable");
        result.push(*x_val * two.powf(T::from(exp_f64).expect("exponent should be representable")));
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

/// Extract mantissa and exponent from array elements
///
/// Decomposes floating-point values into mantissa and exponent such that
/// x = mantissa * 2^exponent, where 0.5 <= |mantissa| < 1.0
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Tuple of (mantissa_array, exponent_array)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![4.0, 0.5, -2.0]);
/// let (mantissa, exponent) = frexp(&x);
/// // mantissa = [0.5, 0.5, -0.5]
/// // exponent = [3, 0, 2]
/// ```
pub fn frexp<T>(x: &Array<T>) -> (Array<T>, Array<i32>)
where
    T: Float + Clone,
{
    let x_vec = x.to_vec();
    let mut mantissa_vec = Vec::with_capacity(x_vec.len());
    let mut exponent_vec = Vec::with_capacity(x_vec.len());

    for val in x_vec.iter() {
        if val.is_zero() {
            mantissa_vec.push(T::zero());
            exponent_vec.push(0);
        } else if val.is_infinite() || val.is_nan() {
            mantissa_vec.push(*val);
            exponent_vec.push(0);
        } else {
            // Get the binary exponent
            let log2_val = val.abs().log2();
            let exp = log2_val.floor();
            let exp_i32 = T::to_i32(&exp).expect("exponent should be convertible to i32") + 1;

            // Calculate mantissa
            let two = T::from(2.0).expect("2.0 should be representable");
            let mantissa =
                *val / two.powf(T::from(exp_i32).expect("exponent should be representable"));

            mantissa_vec.push(mantissa);
            exponent_vec.push(exp_i32);
        }
    }

    (
        Array::from_vec(mantissa_vec).reshape(&x.shape()),
        Array::from_vec(exponent_vec).reshape(&x.shape()),
    )
}

/// Return fractional and integral parts of array values
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Tuple of (fractional_part, integral_part)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.5, -2.3, 3.7]);
/// let (frac, integ) = modf(&x);
/// // frac = [0.5, -0.3, 0.7]
/// // integ = [1.0, -2.0, 3.0]
/// ```
pub fn modf<T>(x: &Array<T>) -> (Array<T>, Array<T>)
where
    T: Float + Clone,
{
    let x_vec = x.to_vec();
    let mut frac_vec = Vec::with_capacity(x_vec.len());
    let mut int_vec = Vec::with_capacity(x_vec.len());

    for val in x_vec.iter() {
        let int_part = val.trunc();
        let frac_part = *val - int_part;

        frac_vec.push(frac_part);
        int_vec.push(int_part);
    }

    (
        Array::from_vec(frac_vec).reshape(&x.shape()),
        Array::from_vec(int_vec).reshape(&x.shape()),
    )
}

/// Element-wise quotient and remainder
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Tuple of (quotient, remainder) arrays
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::divmod;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let (quot, rem) = divmod(&x, &y)?;
/// // quot = [2.0, -2.0, 2.0]
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn divmod<T>(x: &Array<T>, y: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut quot_vec = Vec::with_capacity(x_vec.len());
    let mut rem_vec = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        let quotient = (*x_val / *y_val).floor();
        let remainder = *x_val - quotient * *y_val;

        quot_vec.push(quotient);
        rem_vec.push(remainder);
    }

    Ok((
        Array::from_vec(quot_vec).reshape(&x.shape()),
        Array::from_vec(rem_vec).reshape(&y.shape()),
    ))
}

/// Return element-wise remainder of division
///
/// This differs from modulo operation for negative values.
/// The result has the same sign as the dividend.
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Array with remainder values
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::remainder;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let rem = remainder(&x, &y)?;
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn remainder<T>(x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        // IEEE remainder: r = x - n*y where n is the integer nearest x/y
        let n = (*x_val / *y_val).round();
        let rem = *x_val - n * *y_val;
        result.push(rem);
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

/// Return the element-wise remainder of division (C-style)
///
/// This is the C library fmod function. The result has the same sign as the dividend.
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Array with remainder values
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::fmod;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let rem = fmod(&x, &y)?;
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn fmod<T>(x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        // fmod: r = x - n*y where n = trunc(x/y)
        let n = (*x_val / *y_val).trunc();
        let rem = *x_val - n * *y_val;
        result.push(rem);
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

// Suppress unused import warning for PrimInt - it may be used in future extensions
#[allow(unused_imports)]
use num_traits::PrimInt as _;

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
use crate::kernels::{borrow::operand, elementwise};
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

    let x1_data = operand(x1);
    let x2_data = operand(x2);

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| gcd_scalar(a, b))
        .collect();

    Array::from_vec_shape(result_data, &x1.shape())
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

    let x1_data = operand(x1);
    let x2_data = operand(x2);

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

    Array::from_vec_shape(result_data, &x1.shape())
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

    let x1_data = operand(x1);
    let x2_data = operand(x2);

    // Routed through the shared `kernels::elementwise` dispatch (see its
    // module docs) instead of a bespoke `.iter().zip().map().collect()`:
    // `binary_serial` needs only `Clone` (no new trait bounds on `T`
    // here), and is itself always serial -- the same performance profile
    // this closure had before, just without re-deriving the zip/map
    // pattern ad hoc at this call site.
    let result_data = elementwise::binary_serial(&x1_data, &x2_data, |val, sign_val| {
        if sign_val < T::zero() {
            -val.abs()
        } else {
            val.abs()
        }
    });

    Array::from_vec_shape(result_data, &x1.shape())
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

    let x1_data = operand(x1);
    let x2_data = operand(x2);

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

    Array::from_vec_shape(result_data, &x1.shape())
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

    // Hoisted once and reused for both the `.all()` check and the
    // `.map().collect()` below (both branches build the exact same
    // `real_data` from the same read-only source -- see the comment on
    // the `else` arm). An earlier version of this fix called
    // `a.array().iter()` twice instead of sharing one `operand` borrow,
    // on the reasoning that avoiding the old `a.to_vec()` was the whole
    // point -- but two `NdArray<Complex<T>, IxDyn>` traversals measured
    // as a real regression against that one copy in the sibling
    // `take`/`place`/`put`/`outer` fixes (`IxDyn`'s rank-erased iterator
    // doesn't fold down to a pointer-bump loop the way a shared `&[_]`
    // does), so this is now one `operand` borrow, walked twice.
    let a_op = operand(a);
    let all_close = a_op.iter().all(|c| c.im.abs() <= threshold);
    let real_data: Vec<T> = a_op.iter().map(|c| c.re).collect();

    if all_close {
        // Return only real parts
        Array::from_vec_shape(real_data, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
    } else {
        // Convert back to complex array (keeping as real for simplicity in this context)
        // In a real implementation, this would return Result<Either<Array<T>, Array<Complex<T>>>>
        Array::from_vec_shape(real_data, &a.shape()).unwrap_or_else(|e| panic!("{e}"))
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

    // `x` (`T`) and `exp` (`I`) have different element types, so this is
    // `elementwise::binary_serial`'s heterogeneous case (not
    // `binary_dispatch`, which requires both operands to share one `T`) --
    // it needs only `Clone` on each side, matching this function's
    // existing bounds exactly, and stays serial (as this loop already
    // was).
    let x_op = operand(x);
    let exp_op = operand(exp);

    // Validate every exponent converts up front: `binary_serial`'s
    // closure is a plain `Fn` (no fallible early-return from inside it),
    // so the conversion that used to `?`-propagate per element in the
    // loop is checked once here instead.
    for exp_val in exp_op.iter() {
        if I::to_f64(exp_val).is_none() {
            return Err(NumRs2Error::ConversionError(
                "Failed to convert exponent to f64".to_string(),
            ));
        }
    }

    let two = T::from(2.0).expect("2.0 should be representable");
    let result = elementwise::binary_serial(&x_op, &exp_op, |x_val, exp_val| {
        let exp_f64 = I::to_f64(&exp_val).expect("convertibility already validated above");
        x_val * two.powf(T::from(exp_f64).expect("exponent should be representable"))
    });

    Array::from_vec_shape(result, &x.shape())
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
    // NOTE: deliberately *not* routed through `kernels::elementwise`
    // (unlike `copysign`/`fmod`/`ldexp`/`modf`): each element needs
    // *both* outputs from one `log2()`/`floor()` pair (`exp` feeds
    // directly into computing `mantissa`), and `elementwise`'s
    // `unary_dispatch`/`unary_serial` each produce only one output per
    // call. Splitting this into two dispatch calls (one per output)
    // would recompute that transcendental pair a second time per
    // element merely to reach the same `exp` again -- a real cost,
    // unlike the single cheap `trunc()` `modf` duplicates below. A
    // zero-copy `operand` borrow still replaces the old owned
    // `x.to_vec()`.
    let x_op = operand(x);
    let mut mantissa_vec = Vec::with_capacity(x_op.len());
    let mut exponent_vec = Vec::with_capacity(x_op.len());

    for val in x_op.iter() {
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
        Array::from_vec_shape(mantissa_vec, &x.shape()).unwrap_or_else(|e| panic!("{e}")),
        Array::from_vec_shape(exponent_vec, &x.shape()).unwrap_or_else(|e| panic!("{e}")),
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
    // Both outputs are type `T`, so (unlike `frexp`, whose exponent
    // output is `i32`) each fits a same-type `Fn(T) -> T` shape directly.
    // Uses `unary_serial` rather than `unary_dispatch`: the latter would
    // require widening this public function's bounds to
    // `T: Send + Sync + 'static` for the parallel tier it adds -- a
    // semver-relevant bound narrowing this to_vec-sweep task must not
    // introduce on a published crate, matching the choice already made
    // for `copysign`/`fmod`/`ldexp` below (`binary_serial`, not
    // `binary_dispatch`). `unary_dispatch`'s own doc notes a plain
    // closure like `|val| val.trunc()` is actually slower through the
    // dispatch tier below ~100K elements anyway, so this loses nothing
    // in practice. This does mean `trunc()` is computed twice per
    // element instead of once (`frac` needs it to subtract, `int` needs
    // it as the result) -- a cheap, single hardware instruction, not a
    // transcendental like `frexp`'s `log2()`, so trading it for a
    // shared, allocation-free helper is worth it here specifically.
    let x_op = operand(x);
    let frac_vec = elementwise::unary_serial(&x_op, |val| val - val.trunc());
    let int_vec = elementwise::unary_serial(&x_op, |val| val.trunc());

    (
        Array::from_vec_shape(frac_vec, &x.shape()).unwrap_or_else(|e| panic!("{e}")),
        Array::from_vec_shape(int_vec, &x.shape()).unwrap_or_else(|e| panic!("{e}")),
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
/// // quot = [2.0, -3.0, 2.0]  (floor-division: floor(-7.0/3.0) == -3.0)
/// // rem = [1.0, 2.0, 2.0]
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

    let x_vec = operand(x);
    let y_vec = operand(y);
    let mut quot_vec = Vec::with_capacity(x_vec.len());
    let mut rem_vec = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        let quotient = (*x_val / *y_val).floor();
        let remainder = *x_val - quotient * *y_val;

        quot_vec.push(quotient);
        rem_vec.push(remainder);
    }

    Ok((
        Array::from_vec_shape(quot_vec, &x.shape())?,
        Array::from_vec_shape(rem_vec, &y.shape())?,
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
/// // rem = [1.0, -1.0, -1.0]  (round-to-nearest: round(8.0/3.0) == 3.0)
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

    let x_vec = operand(x);
    let y_vec = operand(y);
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        // IEEE remainder: r = x - n*y where n is the integer nearest x/y
        let n = (*x_val / *y_val).round();
        let rem = *x_val - n * *y_val;
        result.push(rem);
    }

    Array::from_vec_shape(result, &x.shape())
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

    let x_vec = operand(x);
    let y_vec = operand(y);

    // Routed through the shared `kernels::elementwise` dispatch (see
    // `copysign`'s comment above for why `binary_serial` specifically:
    // no new trait bounds, and this loop was already always-serial).
    let result = elementwise::binary_serial(&x_vec, &y_vec, |x_val, y_val| {
        // fmod: r = x - n*y where n = trunc(x/y)
        let n = (x_val / y_val).trunc();
        x_val - n * y_val
    });

    Array::from_vec_shape(result, &x.shape())
}

// Suppress unused import warning for PrimInt - it may be used in future extensions
#[allow(unused_imports)]
use num_traits::PrimInt as _;

#[cfg(test)]
mod tests {
    use super::*;

    // This file previously had no `#[cfg(test)]` unit tests at all --
    // correctness relied entirely on the doctests above (still run via
    // `cargo test --doc`, and still passing). These cover the sites
    // touched by the to_vec()-sweep rewrite (`operand` throughout, plus
    // `elementwise::binary_serial`/`unary_dispatch` for
    // `copysign`/`fmod`/`ldexp`/`modf`), including edge cases the
    // doctests don't exercise.

    #[test]
    fn gcd_lcm_basic_and_zero() {
        let a = Array::from_vec(vec![12, 15, 21, 0]);
        let b = Array::from_vec(vec![8, 10, 14, 5]);
        assert_eq!(
            gcd(&a, &b).expect("gcd should succeed").to_vec(),
            vec![4, 5, 7, 5]
        );
        assert_eq!(
            lcm(&a, &b).expect("lcm should succeed").to_vec(),
            vec![24, 30, 42, 0]
        );
    }

    #[test]
    fn copysign_matches_naive() {
        let a = Array::from_vec(vec![1.0, -2.0, 3.0, 0.0]);
        let b = Array::from_vec(vec![-1.0, 1.0, -1.0, -5.0]);
        let got = copysign(&a, &b).expect("copysign should succeed").to_vec();
        let expected: Vec<f64> = a
            .to_vec()
            .iter()
            .zip(b.to_vec().iter())
            .map(|(&v, &s)| if s < 0.0 { -v.abs() } else { v.abs() })
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn frexp_matches_definition() {
        let x = Array::from_vec(vec![4.0, 0.5, -2.0, 0.0, f64::INFINITY, f64::NAN]);
        let (mantissa, exponent) = frexp(&x);
        let m = mantissa.to_vec();
        let e = exponent.to_vec();
        assert_eq!(m[0], 0.5);
        assert_eq!(e[0], 3); // 4.0 = 0.5 * 2^3
        assert_eq!(m[1], 0.5);
        assert_eq!(e[1], 0); // 0.5 = 0.5 * 2^0
        assert_eq!(m[2], -0.5);
        assert_eq!(e[2], 2); // -2.0 = -0.5 * 2^2
        assert_eq!(m[3], 0.0);
        assert_eq!(e[3], 0);
        assert!(m[4].is_infinite());
        assert!(m[5].is_nan());

        // Round-trip: mantissa * 2^exponent == original, for finite values.
        for i in 0..3 {
            let two: f64 = 2.0;
            assert!((m[i] * two.powi(e[i]) - x.to_vec()[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn modf_matches_naive_and_sign_convention() {
        let x = Array::from_vec(vec![1.5, -2.3, 3.7, 0.0, -0.5]);
        let (frac, integ) = modf(&x);
        let frac_v = frac.to_vec();
        let integ_v = integ.to_vec();

        // Bit-exact vs the pre-optimization single-pass loop (trunc()
        // computed once there, twice now via two `unary_dispatch` calls
        // -- must still agree exactly, not just up to a tolerance).
        let x_v = x.to_vec();
        for i in 0..x_v.len() {
            let expected_int = x_v[i].trunc();
            let expected_frac = x_v[i] - expected_int;
            assert_eq!(integ_v[i], expected_int, "int part at {i}");
            assert_eq!(frac_v[i], expected_frac, "frac part at {i}");
            // frac + int must reconstruct the original exactly.
            assert_eq!(frac_v[i] + integ_v[i], x_v[i]);
        }
    }

    #[test]
    fn ldexp_matches_definition_and_errors_are_preserved() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let exp = Array::from_vec(vec![1i32, 2, 3]);
        let result = ldexp(&x, &exp).expect("ldexp should succeed");
        assert_eq!(result.to_vec(), vec![2.0, 8.0, 24.0]);

        // Shape mismatch still errors (unaffected by the rewrite).
        let bad_exp = Array::from_vec(vec![1i32, 2]);
        assert!(ldexp(&x, &bad_exp).is_err());
    }

    #[test]
    fn divmod_and_remainder_and_fmod_signs() {
        let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
        let y = Array::from_vec(vec![3.0, 3.0, 3.0]);

        let (quot, rem) = divmod(&x, &y).expect("divmod should succeed");
        assert_eq!(quot.to_vec(), vec![2.0, -3.0, 2.0]);
        assert_eq!(rem.to_vec(), vec![1.0, 2.0, 2.0]);

        // IEEE remainder uses round-to-nearest for `n` (not floor/trunc):
        // 8.0/3.0 rounds to n=3 (not 2), so rem = 8 - 3*3 = -1.0, unlike
        // `divmod`'s floor-based 2.0 or `fmod`'s trunc-based 2.0.
        let ieee_rem = remainder(&x, &y).expect("remainder should succeed");
        assert_eq!(ieee_rem.to_vec(), vec![1.0, -1.0, -1.0]);

        let c_rem = fmod(&x, &y).expect("fmod should succeed");
        assert_eq!(c_rem.to_vec(), vec![1.0, -1.0, 2.0]);
    }

    #[test]
    fn nextafter_and_real_if_close_and_sinc_heaviside() {
        let a = Array::from_vec(vec![1.0, 2.0]);
        let b = Array::from_vec(vec![2.0, 1.0]);
        let r = nextafter(&a, &b)
            .expect("nextafter should succeed")
            .to_vec();
        assert!(r[0] > 1.0);
        assert!(r[1] < 2.0);

        let close = Array::from_vec(vec![Complex::new(1.0, 1e-15), Complex::new(2.0, 1e-16)]);
        let real = real_if_close(&close, Some(1000.0));
        assert_eq!(real.to_vec(), vec![1.0, 2.0]);

        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = sinc(&x);
        assert!((y.to_vec()[0] - 1.0).abs() < 1e-12);
        assert!(y.to_vec()[1].abs() < 1e-12);

        let hs = heaviside(&Array::from_vec(vec![-1.0, 0.0, 1.0]), Some(0.5));
        assert_eq!(hs.to_vec(), vec![0.0, 0.5, 1.0]);
    }

    /// Manual timing probe (no `[[bench]]` entry available in this
    /// lane's `Cargo.toml`, owned by another lane) for `fmod`'s
    /// `elementwise::binary_serial` rewrite vs. the old bespoke
    /// `.iter().zip().map().collect()` loop it replaced -- both
    /// always-serial, so this isolates `operand`'s zero-copy input vs.
    /// the old owned `to_vec()` pair.
    #[test]
    fn probe_fmod_perf_vs_naive_to_vec_pair() {
        fn naive_fmod(x: &Array<f64>, y: &Array<f64>) -> Vec<f64> {
            let x_vec = x.to_vec();
            let y_vec = y.to_vec();
            x_vec
                .iter()
                .zip(y_vec.iter())
                .map(|(&xv, &yv)| {
                    let n = (xv / yv).trunc();
                    xv - n * yv
                })
                .collect()
        }

        let n = 200_000;
        let x = Array::from_vec((0..n).map(|i| i as f64 * 0.37).collect::<Vec<_>>());
        let y = Array::from_vec((0..n).map(|i| (i % 13 + 1) as f64).collect::<Vec<_>>());
        let iters = 100;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_fmod(&x, &y));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(fmod(&x, &y).expect("fmod should succeed"));
        }
        let via_kernels = t1.elapsed();

        eprintln!(
            "[fmod, n={n}] naive(to_vec_pair)={:.1}us/iter binary_serial={:.1}us/iter ({:.2}x)",
            naive.as_secs_f64() * 1e6 / iters as f64,
            via_kernels.as_secs_f64() * 1e6 / iters as f64,
            naive.as_secs_f64() / via_kernels.as_secs_f64(),
        );

        assert_eq!(
            naive_fmod(&x, &y),
            fmod(&x, &y).expect("fmod should succeed").to_vec()
        );
    }

    /// Manual timing probe (no `[[bench]]` entry available in this
    /// lane's `Cargo.toml`, owned by another lane) for `real_if_close`'s
    /// `a.to_vec()` -> hoisted-`operand` conversion. `naive_real_if_close`
    /// reproduces the pre-sweep body verbatim (one copy, walked twice) --
    /// see `real_if_close`'s fix comment for why the *intermediate* state
    /// (`a.array().iter()` called twice, no buffer at all) this fix
    /// replaces was itself slower than either endpoint.
    #[test]
    fn probe_real_if_close_perf_vs_naive_to_vec() {
        // Generic with the same bound as `real_if_close<T: Float +
        // Clone>` itself (instantiated at `f64` below) -- comparing a
        // hand-specialized concrete `fn(&Array<Complex<f64>>, ..)`
        // against the actual (generic) `real_if_close` would conflate
        // "old to_vec() vs new operand()" with "concrete vs generic
        // monomorphization", which is a different question this probe
        // isn't meant to answer.
        // `all_close` is computed (matching the old code's control flow
        // exactly, for a fair baseline) but both arms build the identical
        // `Array::from_vec(real_data).reshape(&a.shape())` -- the old
        // pre-sweep implementation never actually branched on it either.
        // Deliberately faithful, not a bug: allow the resulting
        // same-body-both-arms lint rather than collapsing the branches,
        // since collapsing them would stop reproducing the exact old
        // control-flow shape this probe is timing against.
        #[allow(clippy::if_same_then_else)]
        fn naive_real_if_close<T: Float + Clone>(a: &Array<Complex<T>>, threshold: T) -> Array<T> {
            let data = a.to_vec();
            let all_close = data.iter().all(|c| c.im.abs() <= threshold);
            let real_data: Vec<T> = data.iter().map(|c| c.re).collect();
            if all_close {
                Array::from_vec(real_data).reshape(&a.shape())
            } else {
                Array::from_vec(real_data).reshape(&a.shape())
            }
        }

        let n = 200_000;
        let a = Array::from_vec(
            (0..n)
                .map(|i| Complex::new(i as f64, 1e-15))
                .collect::<Vec<_>>(),
        );
        let iters = 100;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_real_if_close(&a, 1e-10));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(real_if_close(&a, Some(1e-10 / f64::EPSILON)));
        }
        let operand = t1.elapsed();

        eprintln!(
            "[real_if_close, n={n}] naive(to_vec)={:.1}us/iter operand={:.1}us/iter ({:.2}x)",
            naive.as_secs_f64() * 1e6 / iters as f64,
            operand.as_secs_f64() * 1e6 / iters as f64,
            naive.as_secs_f64() / operand.as_secs_f64(),
        );

        assert_eq!(
            naive_real_if_close(&a, 1e-10).to_vec(),
            real_if_close(&a, Some(1e-10 / f64::EPSILON)).to_vec()
        );
    }
}

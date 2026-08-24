//! Operator overloading for Array
//!
//! This module implements arithmetic operators with automatic broadcasting:
//! - Add, Sub, Mul, Div, Rem (element-wise)
//! - Neg (unary minus)
//! - Scalar operations

use super::Array;
use crate::kernels;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

// ============================================================================
// Operator Overloading with Automatic Broadcasting
// ============================================================================

/// Addition operator with automatic broadcasting
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
/// let b = Array::from_vec(vec![10.0, 20.0, 30.0]).reshape(&[3, 1]);
/// let c = a + b; // Broadcasts to 3x3
/// assert_eq!(c.shape(), vec![3, 3]);
/// ```
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::add_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<T> Add for Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, other: Array<T>) -> Self::Output {
        self.add_broadcast(&other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '+': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Addition operator with automatic broadcasting (by reference)
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::add_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<'b, T> Add<&'b Array<T>> for &Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, other: &'b Array<T>) -> Self::Output {
        self.add_broadcast(other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '+': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Subtraction operator with automatic broadcasting
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![10.0, 20.0, 30.0]);
/// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let c = a - b;
/// assert_eq!(c.to_vec(), vec![9.0, 18.0, 27.0]);
/// ```
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::subtract_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<T> Sub for Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, other: Array<T>) -> Self::Output {
        self.subtract_broadcast(&other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '-': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Subtraction operator with automatic broadcasting (by reference)
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::subtract_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<'b, T> Sub<&'b Array<T>> for &Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, other: &'b Array<T>) -> Self::Output {
        self.subtract_broadcast(other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '-': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Multiplication operator with automatic broadcasting (element-wise)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![2.0, 3.0, 4.0]);
/// let c = a * b;
/// assert_eq!(c.to_vec(), vec![2.0, 6.0, 12.0]);
/// ```
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::multiply_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<T> Mul for Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, other: Array<T>) -> Self::Output {
        self.multiply_broadcast(&other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '*': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Multiplication operator with automatic broadcasting (by reference)
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::multiply_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<'b, T> Mul<&'b Array<T>> for &Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, other: &'b Array<T>) -> Self::Output {
        self.multiply_broadcast(other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '*': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Division operator with automatic broadcasting (element-wise)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![10.0, 20.0, 30.0]);
/// let b = Array::from_vec(vec![2.0, 4.0, 5.0]);
/// let c = a / b;
/// assert_eq!(c.to_vec(), vec![5.0, 5.0, 6.0]);
/// ```
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::divide_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<T> Div for Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, other: Array<T>) -> Self::Output {
        self.divide_broadcast(&other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '/': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Division operator with automatic broadcasting (by reference)
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]). Use [`Array::divide_broadcast`], which
/// returns a [`crate::error::Result`] instead of panicking, to handle
/// incompatible shapes without unwinding.
impl<'b, T> Div<&'b Array<T>> for &Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, other: &'b Array<T>) -> Self::Output {
        self.divide_broadcast(other).unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '/': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Remainder operator with automatic broadcasting (element-wise)
///
/// Unlike `Add`/`Sub`/`Mul`/`Div`, there is no `remainder_broadcast`
/// method on [`Array`] to delegate to -- the [`kernels::borrow::operand`] +
/// [`kernels::elementwise::binary_serial`] + [`Array::from_vec_shape`]
/// dispatch (identical in shape to `add_broadcast`'s closure body in
/// `array/operations.rs`) is inlined directly into this operator's
/// [`Array::broadcast_op`] closure instead.
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]).
impl<T> Rem for Array<T>
where
    T: Clone + Rem<Output = T>,
{
    type Output = Array<T>;

    fn rem(self, other: Array<T>) -> Self::Output {
        self.broadcast_op(&other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x % y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
        .unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '%': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

/// Remainder operator with automatic broadcasting (by reference)
///
/// # Panics
///
/// Panics if `self` and `other`'s shapes cannot be broadcast together
/// (see [`Array::broadcast_shape`]).
impl<'b, T> Rem<&'b Array<T>> for &Array<T>
where
    T: Clone + Rem<Output = T>,
{
    type Output = Array<T>;

    fn rem(self, other: &'b Array<T>) -> Self::Output {
        self.broadcast_op(other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x % y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
        .unwrap_or_else(|e| {
            panic!(
                "numrs2: cannot broadcast shapes {:?} and {:?} for '%': {e}",
                self.shape(),
                other.shape()
            )
        })
    }
}

// ============================================================================
// Scalar Broadcasting Operations
// ============================================================================

/// Add scalar to array (broadcasting)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = a + 10.0;
/// assert_eq!(b.to_vec(), vec![11.0, 12.0, 13.0]);
/// ```
impl<T> Add<T> for Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, scalar: T) -> Self::Output {
        self.add_scalar(scalar)
    }
}

impl<T> Add<T> for &Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, scalar: T) -> Self::Output {
        self.add_scalar(scalar)
    }
}

/// Subtract scalar from array (broadcasting)
impl<T> Sub<T> for Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, scalar: T) -> Self::Output {
        self.subtract_scalar(scalar)
    }
}

impl<T> Sub<T> for &Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, scalar: T) -> Self::Output {
        self.subtract_scalar(scalar)
    }
}

/// Multiply array by scalar (broadcasting)
impl<T> Mul<T> for Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, scalar: T) -> Self::Output {
        self.multiply_scalar(scalar)
    }
}

impl<T> Mul<T> for &Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, scalar: T) -> Self::Output {
        self.multiply_scalar(scalar)
    }
}

/// Divide array by scalar (broadcasting)
impl<T> Div<T> for Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, scalar: T) -> Self::Output {
        self.divide_scalar(scalar)
    }
}

impl<T> Div<T> for &Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, scalar: T) -> Self::Output {
        self.divide_scalar(scalar)
    }
}

// ============================================================================
// Negation Operator
// ============================================================================

/// Negation operator (unary minus)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, -2.0, 3.0]);
/// let b = -a;
/// assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
/// ```
impl<T> Neg for Array<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Array<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<T> Neg for &Array<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Array<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- correctness on valid (equal-shape) inputs is unchanged ----

    #[test]
    fn add_matches_known_values() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!((a.clone() + b.clone()).to_vec(), vec![11.0, 22.0, 33.0]);
        assert_eq!((&a + &b).to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn sub_matches_known_values() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!((a.clone() - b.clone()).to_vec(), vec![9.0, 18.0, 27.0]);
        assert_eq!((&a - &b).to_vec(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn mul_matches_known_values() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![2.0, 3.0, 4.0]);
        assert_eq!((a.clone() * b.clone()).to_vec(), vec![2.0, 6.0, 12.0]);
        assert_eq!((&a * &b).to_vec(), vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn div_matches_known_values() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0]);
        let b = Array::from_vec(vec![2.0, 4.0, 5.0]);
        assert_eq!((a.clone() / b.clone()).to_vec(), vec![5.0, 5.0, 6.0]);
        assert_eq!((&a / &b).to_vec(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn rem_matches_known_values() {
        let a = Array::from_vec(vec![10i64, 21, 33]);
        let b = Array::from_vec(vec![3i64, 4, 5]);
        assert_eq!((a.clone() % b.clone()).to_vec(), vec![1, 1, 3]);
        assert_eq!((&a % &b).to_vec(), vec![1, 1, 3]);
    }

    #[test]
    fn ops_broadcast_through_the_operator_not_just_the_method() {
        // a (shape [1,3], row [1,2,3]) + b (shape [3,1], column [10;20;30])
        // -> [3,3], result[i][j] = a[j] + b[i]: row-major flatten is
        // [1+10,2+10,3+10, 1+20,2+20,3+20, 1+30,2+30,3+30].
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0]).reshape(&[3, 1]);
        let c = a + b;
        assert_eq!(c.shape(), vec![3, 3]);
        assert_eq!(
            c.to_vec(),
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0]
        );
    }

    // ---- panic fall-through: diagnostic message, not an opaque ndarray one ----

    #[test]
    #[should_panic(expected = "numrs2: cannot broadcast shapes")]
    fn add_panics_with_diagnostic_message_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "numrs2: cannot broadcast shapes")]
    fn sub_panics_with_diagnostic_message_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        let _ = a - b;
    }

    #[test]
    #[should_panic(expected = "numrs2: cannot broadcast shapes")]
    fn mul_panics_with_diagnostic_message_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        let _ = a * b;
    }

    #[test]
    #[should_panic(expected = "numrs2: cannot broadcast shapes")]
    fn div_panics_with_diagnostic_message_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        let _ = a / b;
    }

    #[test]
    #[should_panic(expected = "numrs2: cannot broadcast shapes")]
    fn rem_panics_with_diagnostic_message_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1i64; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1i64; 10]).reshape(&[2, 5]);
        let _ = a % b;
    }

    #[test]
    fn add_panic_message_names_both_shapes_and_the_op_symbol() {
        // `should_panic(expected = ..)` above only proves the stable
        // prefix is there; this pins the rest of the format the "# Panics"
        // docs promise (both shapes, and which operator).
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        let result = std::panic::catch_unwind(|| a + b);
        let err = result.expect_err("incompatible shapes must panic");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "<non-string panic payload>".to_string());
        assert!(msg.contains("[2, 3]"), "message was: {msg}");
        assert!(msg.contains("[2, 5]"), "message was: {msg}");
        assert!(msg.contains("'+'"), "message was: {msg}");
    }
}

//! Operator overloading for Array
//!
//! This module implements arithmetic operators with automatic broadcasting:
//! - Add, Sub, Mul, Div, Rem (element-wise)
//! - Neg (unary minus)
//! - Scalar operations

use super::Array;
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
impl<T> Add for Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, other: Array<T>) -> Self::Output {
        // Try broadcasting, if it fails use direct addition
        self.add_broadcast(&other).unwrap_or_else(|_| {
            let result = &self.data + &other.data;
            Array { data: result }
        })
    }
}

/// Addition operator with automatic broadcasting (by reference)
impl<'b, T> Add<&'b Array<T>> for &Array<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, other: &'b Array<T>) -> Self::Output {
        self.add_broadcast(other).unwrap_or_else(|_| {
            let result = &self.data + &other.data;
            Array { data: result }
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
impl<T> Sub for Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, other: Array<T>) -> Self::Output {
        self.subtract_broadcast(&other).unwrap_or_else(|_| {
            let result = &self.data - &other.data;
            Array { data: result }
        })
    }
}

/// Subtraction operator with automatic broadcasting (by reference)
impl<'b, T> Sub<&'b Array<T>> for &Array<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, other: &'b Array<T>) -> Self::Output {
        self.subtract_broadcast(other).unwrap_or_else(|_| {
            let result = &self.data - &other.data;
            Array { data: result }
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
impl<T> Mul for Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, other: Array<T>) -> Self::Output {
        self.multiply_broadcast(&other).unwrap_or_else(|_| {
            let result = &self.data * &other.data;
            Array { data: result }
        })
    }
}

/// Multiplication operator with automatic broadcasting (by reference)
impl<'b, T> Mul<&'b Array<T>> for &Array<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, other: &'b Array<T>) -> Self::Output {
        self.multiply_broadcast(other).unwrap_or_else(|_| {
            let result = &self.data * &other.data;
            Array { data: result }
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
impl<T> Div for Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, other: Array<T>) -> Self::Output {
        self.divide_broadcast(&other).unwrap_or_else(|_| {
            let result = &self.data / &other.data;
            Array { data: result }
        })
    }
}

/// Division operator with automatic broadcasting (by reference)
impl<'b, T> Div<&'b Array<T>> for &Array<T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, other: &'b Array<T>) -> Self::Output {
        self.divide_broadcast(other).unwrap_or_else(|_| {
            let result = &self.data / &other.data;
            Array { data: result }
        })
    }
}

/// Remainder operator with automatic broadcasting (element-wise)
impl<T> Rem for Array<T>
where
    T: Clone + Rem<Output = T>,
{
    type Output = Array<T>;

    fn rem(self, other: Array<T>) -> Self::Output {
        self.broadcast_op(&other, |a, b| {
            let result = &a.data % &b.data;
            Array { data: result }
        })
        .unwrap_or_else(|_| {
            let result = &self.data % &other.data;
            Array { data: result }
        })
    }
}

/// Remainder operator with automatic broadcasting (by reference)
impl<'b, T> Rem<&'b Array<T>> for &Array<T>
where
    T: Clone + Rem<Output = T>,
{
    type Output = Array<T>;

    fn rem(self, other: &'b Array<T>) -> Self::Output {
        self.broadcast_op(other, |a, b| {
            let result = &a.data % &b.data;
            Array { data: result }
        })
        .unwrap_or_else(|_| {
            let result = &self.data % &other.data;
            Array { data: result }
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

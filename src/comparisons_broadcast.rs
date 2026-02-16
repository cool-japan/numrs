// src/comparisons_broadcast.rs
//! Comparison operators with automatic broadcasting
//!
//! This module implements comparison operators (<, <=, >, >=, ==, !=) for Array
//! with automatic broadcasting support.

use crate::array::Array;
use crate::error::Result;
use std::cmp::PartialOrd;

impl<T> Array<T>
where
    T: Clone + PartialOrd,
{
    /// Element-wise less than with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.less_than(&b)?;
    ///     assert_eq!(result.to_vec(), vec![true, false, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn less_than(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x < y).collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise less than or equal with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.less_equal(&b)?;
    ///     assert_eq!(result.to_vec(), vec![true, true, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn less_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| x <= y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise greater than with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.greater_than(&b)?;
    ///     assert_eq!(result.to_vec(), vec![false, false, true]);
    ///     Ok(())
    /// }
    /// ```
    pub fn greater_than(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x > y).collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise greater than or equal with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.greater_equal(&b)?;
    ///     assert_eq!(result.to_vec(), vec![false, true, true]);
    ///     Ok(())
    /// }
    /// ```
    pub fn greater_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| x >= y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }
}

impl<T> Array<T>
where
    T: Clone + PartialEq,
{
    /// Element-wise equality with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.equal(&b)?;
    ///     assert_eq!(result.to_vec(), vec![false, true, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| x == y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise inequality with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    ///     let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
    ///     let result = a.not_equal(&b)?;
    ///     assert_eq!(result.to_vec(), vec![true, false, true]);
    ///     Ok(())
    /// }
    /// ```
    pub fn not_equal(&self, other: &Array<T>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| x != y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }
}

// Logical operations for boolean arrays
impl Array<bool> {
    /// Element-wise logical AND with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![true, true, false, false]);
    ///     let b = Array::from_vec(vec![true, false, true, false]);
    ///     let result = a.logical_and(&b)?;
    ///     assert_eq!(result.to_vec(), vec![true, false, false, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn logical_and(&self, other: &Array<bool>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| *x && *y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise logical OR with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![true, true, false, false]);
    ///     let b = Array::from_vec(vec![true, false, true, false]);
    ///     let result = a.logical_or(&b)?;
    ///     assert_eq!(result.to_vec(), vec![true, true, true, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn logical_or(&self, other: &Array<bool>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| *x || *y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise logical XOR with broadcasting
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::error::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let a = Array::from_vec(vec![true, true, false, false]);
    ///     let b = Array::from_vec(vec![true, false, true, false]);
    ///     let result = a.logical_xor(&b)?;
    ///     assert_eq!(result.to_vec(), vec![false, true, true, false]);
    ///     Ok(())
    /// }
    /// ```
    pub fn logical_xor(&self, other: &Array<bool>) -> Result<Array<bool>> {
        self.broadcast_op(other, |a, b| {
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();
            let result: Vec<bool> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(x, y)| *x ^ *y)
                .collect();
            Array::from_vec(result).reshape(&a.shape())
        })
    }

    /// Element-wise logical NOT
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![true, false, true, false]);
    /// let result = a.logical_not();
    /// assert_eq!(result.to_vec(), vec![false, true, false, true]);
    /// ```
    pub fn logical_not(&self) -> Array<bool> {
        self.map(|x| !x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_less_than_broadcast() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let b = Array::from_vec(vec![2.0, 2.0, 2.0]).reshape(&[3, 1]);
        let result = a.less_than(&b).expect("less_than broadcast should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
    }

    #[test]
    fn test_equal_broadcast() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![2, 2, 2]);
        let result = a.equal(&b).expect("equal comparison should succeed");
        assert_eq!(result.to_vec(), vec![false, true, false]);
    }

    #[test]
    fn test_logical_and() {
        let a = Array::from_vec(vec![true, true, false, false]);
        let b = Array::from_vec(vec![true, false, true, false]);
        let result = a.logical_and(&b).expect("logical_and should succeed");
        assert_eq!(result.to_vec(), vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_not() {
        let a = Array::from_vec(vec![true, false, true]);
        let result = a.logical_not();
        assert_eq!(result.to_vec(), vec![false, true, false]);
    }
}

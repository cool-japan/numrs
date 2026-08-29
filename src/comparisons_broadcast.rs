// src/comparisons_broadcast.rs
//! Comparison operators with automatic broadcasting
//!
//! This module implements comparison operators (<, <=, >, >=, ==, !=) for Array
//! with automatic broadcasting support.

use crate::array::Array;
use crate::error::Result;
use crate::kernels;
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x < y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x <= y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x > y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x >= y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x == y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x != y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x && y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x || y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x ^ y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
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

    // ---- spot checks for every other method rewired onto
    // `kernels::borrow::operand` + `kernels::elementwise::binary_serial`
    // in this file, values matching the equivalent `np.*` calls ----

    #[test]
    fn test_less_equal_greater_greater_equal_values() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
        assert_eq!(
            a.less_equal(&b).expect("less_equal").to_vec(),
            vec![true, true, false]
        );
        assert_eq!(
            a.greater_than(&b).expect("greater_than").to_vec(),
            vec![false, false, true]
        );
        assert_eq!(
            a.greater_equal(&b).expect("greater_equal").to_vec(),
            vec![false, true, true]
        );
    }

    #[test]
    fn test_not_equal_values() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 5, 3]);
        assert_eq!(
            a.not_equal(&b).expect("not_equal").to_vec(),
            vec![false, true, false]
        );
    }

    #[test]
    fn test_greater_than_broadcast_matches_numpy() {
        // np.greater([[1],[2],[3]], [0,2,4]) -- same case as
        // `comparisons::greater`'s equivalent test, exercised here via
        // the `Array` method instead of the free function.
        let a = Array::from_vec(vec![1, 2, 3]).reshape(&[3, 1]);
        let b = Array::from_vec(vec![0, 2, 4]).reshape(&[1, 3]);
        let result = a
            .greater_than(&b)
            .expect("broadcast greater_than should succeed");
        assert_eq!(result.shape(), vec![3, 3]);
        assert_eq!(
            result.to_vec(),
            vec![true, false, false, true, false, false, true, true, false]
        );
    }

    #[test]
    fn test_logical_or_xor_values() {
        let a = Array::from_vec(vec![true, true, false, false]);
        let b = Array::from_vec(vec![true, false, true, false]);
        assert_eq!(
            a.logical_or(&b).expect("logical_or").to_vec(),
            vec![true, true, true, false]
        );
        assert_eq!(
            a.logical_xor(&b).expect("logical_xor").to_vec(),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn test_logical_and_or_xor_broadcast() {
        // np.logical_and([[True],[False]], [True, False]) etc. -- same
        // broadcast case as `comparisons::logical_and`'s test.
        let a = Array::from_vec(vec![true, false]).reshape(&[2, 1]);
        let b = Array::from_vec(vec![true, false]).reshape(&[1, 2]);
        assert_eq!(
            a.logical_and(&b).expect("logical_and broadcast").to_vec(),
            vec![true, false, false, false]
        );
        assert_eq!(
            a.logical_or(&b).expect("logical_or broadcast").to_vec(),
            vec![true, true, true, false]
        );
        assert_eq!(
            a.logical_xor(&b).expect("logical_xor broadcast").to_vec(),
            vec![false, true, true, false]
        );
    }
}

//! Arithmetic operators and element-wise comparisons for [`MaskedArray`].
//!
//! `Add`/`Sub`/`Mul`/`Div` are `std::ops` operator overloads (`&a + &b`
//! style); they always combine masks as `self.mask OR other.mask` and
//! `Div` additionally masks any element whose divisor is zero, matching
//! `numpy.ma`'s auto-masking of division-by-zero.
//!
//! The comparisons (`equal`/`not_equal`/`less_than`/`less_equal`/
//! `greater_than`/`greater_equal`) are plain inherent methods, not
//! `PartialEq`/`PartialOrd` impls: those traits are hard-coded by `std` to
//! return `bool`/`Option<Ordering>`, which cannot express "an element-wise
//! `MaskedArray<bool>` with a propagated mask" the way NumPy's `==` does
//! for a masked array. They are named after `Array<T>`'s own
//! `comparisons_broadcast` methods (which they delegate to for the data
//! comparison) rather than NumPy's short `eq`/`ne`/`lt`/`le`/`gt`/`ge` --
//! a public inherent method named `eq(&self, ..)` trips
//! `clippy::should_implement_trait` (it does not fire on the other five
//! names, but consistency and the existing `Array<T>` naming both point
//! the same way). The NumPy names are noted on each method's doc comment.

use super::MaskedArray;
use crate::error::Result;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

// Implementation of arithmetic operations for MaskedArray
// Implement Add for MaskedArray with MaskedArray
impl<T: Clone + Add<Output = T>> Add for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    /// # Panics
    ///
    /// Panics if `self` and `other` have shapes that cannot be broadcast
    /// together. `std::ops::Add` cannot return a `Result`, so there is no
    /// non-panicking version of this operator; construct the result via
    /// [`Array::add_broadcast`](crate::array::Array::add_broadcast) directly (on the `.data`/`.mask` fields) if
    /// you need to handle a shape mismatch without panicking.
    fn add(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Add the data arrays
        let result_data = match self.data.add_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to add MaskedArrays with incompatible shapes: {:?} vs {:?}",
                self.data.shape(),
                other.data.shape()
            ),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to combine masks with incompatible shapes: {:?} vs {:?}",
                self.mask.shape(),
                other.mask.shape()
            ),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement subtract for MaskedArray with MaskedArray
impl<T: Clone + Sub<Output = T>> Sub for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    /// # Panics
    ///
    /// Panics if `self` and `other` have shapes that cannot be broadcast
    /// together. `std::ops::Sub` cannot return a `Result`, so there is no
    /// non-panicking version of this operator; construct the result via
    /// [`Array::subtract_broadcast`](crate::array::Array::subtract_broadcast) directly (on the `.data`/`.mask`
    /// fields) if you need to handle a shape mismatch without panicking.
    fn sub(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Subtract the data arrays
        let result_data = match self.data.subtract_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to subtract MaskedArrays with incompatible shapes: {:?} vs {:?}",
                self.data.shape(),
                other.data.shape()
            ),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to combine masks with incompatible shapes: {:?} vs {:?}",
                self.mask.shape(),
                other.mask.shape()
            ),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement multiply for MaskedArray with MaskedArray
impl<T: Clone + Mul<Output = T>> Mul for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    /// # Panics
    ///
    /// Panics if `self` and `other` have shapes that cannot be broadcast
    /// together. `std::ops::Mul` cannot return a `Result`, so there is no
    /// non-panicking version of this operator; construct the result via
    /// [`Array::multiply_broadcast`](crate::array::Array::multiply_broadcast) directly (on the `.data`/`.mask`
    /// fields) if you need to handle a shape mismatch without panicking.
    fn mul(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Multiply the data arrays
        let result_data = match self.data.multiply_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to multiply MaskedArrays with incompatible shapes: {:?} vs {:?}",
                self.data.shape(),
                other.data.shape()
            ),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to combine masks with incompatible shapes: {:?} vs {:?}",
                self.mask.shape(),
                other.mask.shape()
            ),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement divide for MaskedArray with MaskedArray
impl<T: Clone + Div<Output = T> + PartialEq + Zero> Div for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    /// # Panics
    ///
    /// Panics if `self` and `other` have shapes that cannot be broadcast
    /// together. `std::ops::Div` cannot return a `Result`, so there is no
    /// non-panicking version of this operator; construct the result via
    /// [`Array::divide_broadcast`](crate::array::Array::divide_broadcast) directly (on the `.data`/`.mask` fields)
    /// if you need to handle a shape mismatch without panicking. Division by
    /// a zero element does not panic: that element is masked instead (see
    /// `division_mask` below).
    fn div(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Check for divisions by zero and mask them
        let zero = T::zero();
        let other_data_op = crate::kernels::borrow::operand(&other.data);
        let other_mask_op = crate::kernels::borrow::operand(&other.mask);
        let mut division_mask_vec = Vec::with_capacity(other.size());

        for (value, is_masked) in other_data_op.iter().zip(other_mask_op.iter()) {
            division_mask_vec.push(*is_masked || *value == zero);
        }

        let division_mask = crate::array::Array::from_vec_shape(division_mask_vec, &other.shape())
            .unwrap_or_else(|e| panic!("{e}"));

        // Divide the data arrays
        let result_data = match self.data.divide_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to divide MaskedArrays with incompatible shapes: {:?} vs {:?}",
                self.data.shape(),
                other.data.shape()
            ),
        };

        // Combine the masks - an element is masked if it is masked in either input or if divisor is zero
        let mask_combined = match self.mask.zip_with(&division_mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!(
                "Failed to combine masks with incompatible shapes: {:?} vs {:?}",
                self.mask.shape(),
                division_mask.shape()
            ),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

/// Element-wise comparisons, mirroring `numpy.ma`'s `eq`/`ne`/`lt`/`le`/
/// `gt`/`ge`.
///
/// Every comparison here follows the same rule as `Add`/`Sub`/`Mul`/`Div`
/// above: the comparison itself runs on the raw `.data` (regardless of
/// masking -- a masked slot's underlying value is compared like any
/// other), then the output mask is `self.mask OR other.mask` (broadcast
/// the same way `Array::zip_with` broadcasts the data comparison, via
/// `Array<T>`'s own `less_than`/`less_equal`/... in
/// `comparisons_broadcast.rs`). This matches `numpy.ma`: `x == y` for two
/// masked arrays is masked wherever *either* operand is masked,
/// independent of what the compared values happen to be.
impl<T: Clone + PartialOrd> MaskedArray<T> {
    /// Element-wise `self == other`. NumPy: `eq`.
    pub fn equal(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.equal(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }

    /// Element-wise `self != other`. NumPy: `ne`.
    pub fn not_equal(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.not_equal(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }

    /// Element-wise `self < other`. NumPy: `lt`.
    pub fn less_than(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.less_than(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }

    /// Element-wise `self <= other`. NumPy: `le`.
    pub fn less_equal(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.less_equal(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }

    /// Element-wise `self > other`. NumPy: `gt`.
    pub fn greater_than(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.greater_than(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }

    /// Element-wise `self >= other`. NumPy: `ge`.
    pub fn greater_equal(&self, other: &Self) -> Result<MaskedArray<bool>> {
        let data = self.data.greater_equal(&other.data)?;
        let mask = self.mask.zip_with(&other.mask, |a, b| a || b)?;
        Ok(MaskedArray {
            data,
            mask,
            fill_value: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    fn ma(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64> {
        let shape = vec![data.len()];
        MaskedArray {
            data: Array::from_vec_shape(data, &shape).expect("valid shape"),
            mask: Array::from_vec_shape(mask, &shape).expect("valid shape"),
            fill_value: 0.0,
        }
    }

    /// `np.ma.array([1.,2.,3.],mask=[F,F,F]) / np.ma.array([0.,2.,0.],mask=[F,F,F])`
    /// -> `[--, 1.0, --]`, mask `[True, False, True]`: a zero divisor masks
    /// the result even when neither original operand was masked.
    #[test]
    fn div_masks_division_by_zero_even_when_unmasked() {
        let a = ma(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        let b = ma(vec![0.0, 2.0, 0.0], vec![false, false, false]);
        let r = &a / &b;
        assert_eq!(r.get_mask().to_vec(), vec![true, false, true]);
        assert_eq!(r.filled(Some(-1.0)).to_vec()[1], 1.0);
    }

    #[test]
    fn sub_propagates_mask_as_or() {
        let a = ma(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let b = ma(vec![5.0, 4.0, 3.0], vec![false, false, true]);
        let r = &a - &b;
        assert_eq!(r.get_mask().to_vec(), vec![false, true, true]);
        assert_eq!(r.filled(None).to_vec()[0], -4.0);
    }

    /// `np.ma.array([1.,2.,3.,4.],mask=[F,T,F,F]) == np.ma.array([1.,5.,3.,2.],mask=[F,F,T,F])`
    /// -> raw (mask-ignoring) data comparison `[1==1, 2==5, 3==3, 4==2]`
    /// = `[True,False,True,False]`, mask (OR of the two operand masks)
    /// `[False,True,True,False]`.
    #[test]
    fn equal_propagates_mask_and_compares_raw_data() {
        let a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        let b = ma(vec![1.0, 5.0, 3.0, 2.0], vec![false, false, true, false]);
        let eq = a.equal(&b).expect("shapes match");
        assert_eq!(eq.get_mask().to_vec(), vec![false, true, true, false]);
        assert_eq!(eq.get_data().to_vec(), vec![true, false, true, false]);
    }

    #[test]
    fn less_than_propagates_mask() {
        let a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        let b = ma(vec![1.0, 5.0, 3.0, 2.0], vec![false, false, true, false]);
        let lt = a.less_than(&b).expect("shapes match");
        assert_eq!(lt.get_mask().to_vec(), vec![false, true, true, false]);
        // Raw (mask-ignoring) comparison: [1<1, 2<5, 3<3, 4<2] = [F,T,F,F].
        assert_eq!(lt.get_data().to_vec(), vec![false, true, false, false]);
    }

    #[test]
    fn not_equal_le_gt_ge_all_propagate_mask_or() {
        let a = ma(vec![1.0, 2.0], vec![true, false]);
        let b = ma(vec![1.0, 3.0], vec![false, false]);
        let expected = vec![true, false];
        assert_eq!(
            a.not_equal(&b).expect("same shape").get_mask().to_vec(),
            expected
        );
        assert_eq!(
            a.less_equal(&b).expect("same shape").get_mask().to_vec(),
            expected
        );
        assert_eq!(
            a.greater_than(&b).expect("same shape").get_mask().to_vec(),
            expected
        );
        assert_eq!(
            a.greater_equal(&b).expect("same shape").get_mask().to_vec(),
            expected
        );
    }

    #[test]
    fn comparison_shape_mismatch_is_an_error_not_a_panic() {
        let a = ma(vec![1.0, 2.0], vec![false, false]);
        let b = ma(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        assert!(a.equal(&b).is_err());
    }
}

//! `dot`/`concatenate` for [`MaskedArray`].

use super::MaskedArray;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use std::ops::{Add, Mul};

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero> MaskedArray<T> {
    /// Dot product of two 1-D `MaskedArray`s, matching the scope of its
    /// unmasked cousin ([`Array::dot`], which is likewise 1-D-only and
    /// errors on any other shape).
    ///
    /// A pairwise product is skipped whenever *either* operand is masked
    /// at that position -- exactly the mask-propagation rule `Mul` uses
    /// for element-wise multiplication -- and the skipped products are
    /// then summed like [`MaskedArray::sum`]: `Ok(None)` only when every
    /// pairwise product was skipped, `Ok(Some(_))` otherwise. `Err` is
    /// reserved for a genuine shape problem (not 1-D, or mismatched
    /// lengths), which `Option` alone cannot distinguish from "every pair
    /// was masked".
    ///
    /// Pinned against `numpy.ma`:
    /// `ma.dot(ma.array([1.,2.,3.],mask=[F,T,F]), ma.array([4.,5.,6.]))
    /// == 22.0` (`1*4 + 3*6`; the masked `2.0`@1 drops its term
    /// entirely, not just its contribution as zero-times-something).
    /// When every pairwise term is masked (e.g. both operands masked at
    /// disjoint, complementary positions so each product has exactly one
    /// masked factor), `numpy.ma` returns its masked constant; here that
    /// is `Ok(None)`.
    pub fn dot(&self, other: &Self) -> Result<Option<T>> {
        let a_shape = self.shape();
        let b_shape = other.shape();
        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "MaskedArray::dot requires 1D arrays".to_string(),
            ));
        }
        if a_shape[0] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_shape,
                actual: b_shape,
            });
        }

        let a_data = crate::kernels::borrow::operand(self.get_data());
        let a_mask = crate::kernels::borrow::operand(self.get_mask());
        let b_data = crate::kernels::borrow::operand(other.get_data());
        let b_mask = crate::kernels::borrow::operand(other.get_mask());
        let a_data: &[T] = &a_data;
        let a_mask: &[bool] = &a_mask;
        let b_data: &[T] = &b_data;
        let b_mask: &[bool] = &b_mask;

        let mut sum = T::zero();
        let mut any_valid = false;
        for ((av, am), (bv, bm)) in a_data
            .iter()
            .zip(a_mask.iter())
            .zip(b_data.iter().zip(b_mask.iter()))
        {
            if !*am && !*bm {
                sum = sum + av.clone() * bv.clone();
                any_valid = true;
            }
        }

        Ok(if any_valid { Some(sum) } else { None })
    }
}

impl<T: Clone> MaskedArray<T> {
    /// Concatenate `MaskedArray`s along a single axis, mask-preserving:
    /// the output data is `Array`'s own
    /// [`crate::array_ops::joining::concatenate`] applied to every input's
    /// `.data`, and the output mask is the exact same operation applied to
    /// every input's `.mask` -- concatenation only rearranges *which*
    /// elements sit where, so no new masking decision is needed the way a
    /// reduction's "all-masked lane" rule is.
    ///
    /// Only a single `usize` axis is supported (unlike
    /// `array_ops::joining::concatenate`'s `impl Into<AxisArg>`, which
    /// also accepts concatenating along several axes at once) -- out of
    /// scope here. The result's fill value is the first array's.
    ///
    /// Pinned against `numpy.ma`:
    /// `ma.concatenate([ma.array([1.,2.],mask=[F,T]), ma.array([3.,4.],mask=[T,F])])`
    /// has data-with-mask `[1.0, --, --, 4.0]`, mask
    /// `[False, True, True, False]`.
    pub fn concatenate(arrays: &[&MaskedArray<T>], axis: usize) -> Result<Self> {
        if arrays.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "no MaskedArrays to concatenate".to_string(),
            ));
        }

        let data_refs: Vec<&Array<T>> = arrays.iter().map(|m| m.get_data()).collect();
        let mask_refs: Vec<&Array<bool>> = arrays.iter().map(|m| m.get_mask()).collect();

        let data = crate::array_ops::joining::concatenate(&data_refs, axis)?;
        let mask = crate::array_ops::joining::concatenate(&mask_refs, axis)?;

        Ok(MaskedArray {
            data,
            mask,
            fill_value: arrays[0].get_fill_value(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ma(data: Vec<f64>, mask: Vec<bool>, shape: &[usize]) -> MaskedArray<f64> {
        MaskedArray {
            data: Array::from_vec_shape(data, shape).expect("valid shape"),
            mask: Array::from_vec_shape(mask, shape).expect("valid shape"),
            fill_value: 0.0,
        }
    }

    #[test]
    fn dot_skips_pairs_where_either_operand_is_masked() {
        let a = ma(vec![1.0, 2.0, 3.0], vec![false, true, false], &[3]);
        let b = ma(vec![4.0, 5.0, 6.0], vec![false, false, false], &[3]);
        assert_eq!(a.dot(&b).expect("1D shapes match"), Some(22.0));
    }

    /// Each position has exactly one masked factor, so every pairwise
    /// product is skipped: the whole dot product is masked (`None`), not
    /// `0.0`.
    #[test]
    fn dot_is_none_when_every_pairwise_product_is_masked() {
        let a = ma(vec![1.0, 2.0], vec![true, false], &[2]);
        let b = ma(vec![4.0, 5.0], vec![false, true], &[2]);
        assert_eq!(a.dot(&b).expect("1D shapes match"), None);
    }

    #[test]
    fn dot_rejects_non_1d_input() {
        let a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4], &[2, 2]);
        let b = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4], &[2, 2]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_rejects_mismatched_length() {
        let a = ma(vec![1.0, 2.0], vec![false, false], &[2]);
        let b = ma(vec![1.0, 2.0, 3.0], vec![false, false, false], &[3]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn concatenate_preserves_masks_from_every_input() {
        let a = ma(vec![1.0, 2.0], vec![false, true], &[2]);
        let b = ma(vec![3.0, 4.0], vec![true, false], &[2]);
        let r = MaskedArray::concatenate(&[&a, &b], 0).expect("compatible shapes");
        assert_eq!(r.shape(), vec![4]);
        assert_eq!(r.get_mask().to_vec(), vec![false, true, true, false]);
        assert_eq!(r.filled(Some(-1.0)).to_vec(), vec![1.0, -1.0, -1.0, 4.0]);
    }

    #[test]
    fn concatenate_empty_list_is_an_error() {
        let empty: Vec<&MaskedArray<f64>> = vec![];
        assert!(MaskedArray::concatenate(&empty, 0).is_err());
    }
}

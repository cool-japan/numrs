//! `any`/`all` reductions, scoped to `MaskedArray<bool>`.
//!
//! `numpy.ma.MaskedArray.any`/`.all` work on any dtype (truthiness of the
//! underlying value), but this crate has no generic "truthy" trait for an
//! arbitrary `T`, and the natural way a boolean masked array arises here
//! is already `MaskedArray<bool>` (e.g. from `ops.rs`'s `equal`/
//! `less_than`/... comparisons). Scoping `any`/`all` to `bool` avoids
//! inventing that trait for a case this crate does not otherwise need.

use super::reductions::reduce_lanes;
use super::MaskedArray;
use crate::error::Result;

impl MaskedArray<bool> {
    /// `true` if any unmasked element is `true`, honoring an optional
    /// axis and `keepdims`. A lane with zero unmasked elements is masked
    /// in the result (not `false`), matching `numpy.ma`: masked elements
    /// are excluded from consideration entirely, not treated as `False`.
    ///
    /// Pinned against `numpy.ma`: `ma.array([True,True,False], mask=[True,True,False]).any() == False`
    /// (the only unmasked element is `False`), and an all-masked array's
    /// `.any()`/`.all()` are both the masked constant, not `True`/`False`.
    pub fn any(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<bool>> {
        reduce_lanes(
            self.get_data(),
            self.get_mask(),
            axis,
            keepdims,
            |vals, masks| {
                let mut any_valid = false;
                let mut result = false;
                for (v, m) in vals.iter().zip(masks) {
                    if !*m {
                        any_valid = true;
                        result |= *v;
                    }
                }
                any_valid.then_some(result)
            },
        )
    }

    /// `true` if every unmasked element is `true`, honoring an optional
    /// axis and `keepdims`. Like [`Self::any`], a fully-masked lane is
    /// masked in the result rather than vacuously `true`.
    pub fn all(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<bool>> {
        reduce_lanes(
            self.get_data(),
            self.get_mask(),
            axis,
            keepdims,
            |vals, masks| {
                let mut any_valid = false;
                let mut result = true;
                for (v, m) in vals.iter().zip(masks) {
                    if !*m {
                        any_valid = true;
                        result &= *v;
                    }
                }
                any_valid.then_some(result)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    fn mb(data: Vec<bool>, mask: Vec<bool>, shape: &[usize]) -> MaskedArray<bool> {
        MaskedArray {
            data: Array::from_vec_shape(data, shape).expect("valid shape"),
            mask: Array::from_vec_shape(mask, shape).expect("valid shape"),
            fill_value: false,
        }
    }

    /// `ma.array([True,True,False],mask=[True,False,False]).any() == True`,
    /// `.all() == False`.
    #[test]
    fn any_all_skip_masked_elements() {
        let m = mb(vec![true, true, false], vec![true, false, false], &[3]);
        let any = m.any(None, false).expect("reduces");
        assert!(!any.get_mask().to_vec()[0]);
        assert!(any.get_data().to_vec()[0]);
        let all = m.all(None, false).expect("reduces");
        assert!(!all.get_data().to_vec()[0]);
    }

    /// `ma.array([True,True,False],mask=[True,True,False]).any() == False`:
    /// the only unmasked element is `False`.
    #[test]
    fn any_false_when_only_unmasked_element_is_false() {
        let m = mb(vec![true, true, false], vec![true, true, false], &[3]);
        let any = m.any(None, false).expect("reduces");
        assert!(!any.get_mask().to_vec()[0]);
        assert!(!any.get_data().to_vec()[0]);
    }

    /// A fully-masked array's `any`/`all` are masked, not `False`/`True`.
    #[test]
    fn any_all_fully_masked_is_masked() {
        let m = mb(vec![true, true, false], vec![true, true, true], &[3]);
        assert!(m.any(None, false).expect("reduces").get_mask().to_vec()[0]);
        assert!(m.all(None, false).expect("reduces").get_mask().to_vec()[0]);
    }

    /// `ma.array([[True,False],[False,False]], mask=[[True,False],[False,False]])`:
    /// `any(axis=0) == [False, False]`, `all(axis=0) == [False, False]`.
    #[test]
    fn any_all_axis_0() {
        let m = mb(
            vec![true, false, false, false],
            vec![true, false, false, false],
            &[2, 2],
        );
        let any = m.any(Some(0), false).expect("axis 0 valid");
        assert_eq!(any.get_data().to_vec(), vec![false, false]);
        assert_eq!(any.get_mask().to_vec(), vec![false, false]);
        let all = m.all(Some(0), false).expect("axis 0 valid");
        assert_eq!(all.get_data().to_vec(), vec![false, false]);
    }

    #[test]
    fn keepdims_preserves_ndim() {
        let m = mb(vec![true, false, true, false], vec![false; 4], &[2, 2]);
        let r = m.any(Some(1), true).expect("axis 1 valid");
        assert_eq!(r.shape(), vec![2, 1]);
    }
}

//! `argmin`/`argmax`/`cumsum`/`sort` for [`MaskedArray`].
//!
//! `argmin`/`argmax` collapse an axis to a position, like
//! [`super::reductions::reduce_lanes`]'s reductions do to a value, but
//! their failure mode is different enough (see [`Self::argmin`]'s doc
//! comment) that they walk axes directly rather than going through that
//! helper. `cumsum`/`sort` are not reductions at all -- their output has
//! the *same* shape as the input along the walked axis -- so they also
//! walk directly, reusing only the shape/stride math
//! ([`axis_lane_shape`]/[`normalize_axis`]), not the collapsing part.

use super::reductions::{axis_lane_shape, collapsed_shape, normalize_axis};
use super::MaskedArray;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

impl<T: PartialOrd + Clone> MaskedArray<T> {
    /// Position of the minimum unmasked element, honoring an optional
    /// axis and `keepdims`. Masked elements are never candidates, so the
    /// returned position always names an unmasked element.
    ///
    /// `axis: None` returns shape `[1]` regardless of `keepdims`,
    /// matching this method's unmasked cousin,
    /// `math::statistics::argmin`, whose own `axis: None` branch ignores
    /// `keepdims` the same way (unlike this module's value-producing
    /// reductions, which *do* honor `keepdims` for `axis: None` -- each
    /// function here matches its own specific cousin rather than a single
    /// blanket rule).
    ///
    /// # A fully-masked lane is an error, not `numpy.ma`'s degenerate `0`
    ///
    /// `numpy.ma.argmin`/`argmax` silently return index `0` for a
    /// fully-masked lane (`ma.array([3.,1.],mask=[True,True]).argmin() == 0`,
    /// indistinguishable from a genuine tie at index 0). This crate has no
    /// warning channel to flag that degenerate case the way NumPy does,
    /// and returning a silently-misleading position is exactly the kind
    /// of footgun this crate avoids elsewhere (see
    /// `kernels::reduce`'s module docs on why its `min`/`max` were
    /// rewritten instead of matching an upstream kernel that silently
    /// returned a wrong finite value for some `NaN` placements) -- so a
    /// fully-masked lane is an `Err` here instead. Check
    /// [`MaskedArray::count_valid`] first if you need to distinguish "no
    /// candidates" from a real answer without pattern-matching the error.
    ///
    /// Pinned against `numpy.ma`:
    /// `ma.array([3.,1.,2.,0.],mask=[F,T,F,T]).argmin() == 2` (unmasked
    /// candidates are `3.0`@0 and `2.0`@2; the masked `1.0`@1 and `0.0`@3
    /// are never considered, so the smaller *unmasked* value wins even
    /// though `0.0` is numerically smaller).
    pub fn argmin(&self, axis: Option<isize>, keepdims: bool) -> Result<Array<usize>> {
        self.arg_extreme(axis, keepdims, false)
    }

    /// Position of the maximum unmasked element. See [`Self::argmin`] for
    /// the `axis`/`keepdims`/all-masked-lane convention.
    ///
    /// Pinned against `numpy.ma`:
    /// `ma.array([3.,1.,2.,0.],mask=[F,T,F,T]).argmax() == 0`.
    pub fn argmax(&self, axis: Option<isize>, keepdims: bool) -> Result<Array<usize>> {
        self.arg_extreme(axis, keepdims, true)
    }

    fn arg_extreme(
        &self,
        axis: Option<isize>,
        keepdims: bool,
        want_max: bool,
    ) -> Result<Array<usize>> {
        let shape = self.shape();
        let ndim = shape.len();
        let data_op = crate::kernels::borrow::operand(self.get_data());
        let mask_op = crate::kernels::borrow::operand(self.get_mask());
        let data_slice: &[T] = &data_op;
        let mask_slice: &[bool] = &mask_op;

        let (out_shape, outer, axis_size, inner) = match axis {
            // Matches `math::statistics::argmin`/`argmax`'s own `axis: None`
            // branch, which also ignores `keepdims`.
            None => (vec![1], 1usize, data_slice.len(), 1usize),
            Some(ax) => {
                let ax = normalize_axis(ax, ndim)?;
                let (outer, axis_size, inner) = axis_lane_shape(&shape, ax);
                (
                    collapsed_shape(&shape, ax, keepdims),
                    outer,
                    axis_size,
                    inner,
                )
            }
        };

        let out_len = outer * inner;
        let mut out = Vec::with_capacity(out_len);
        for o in 0..outer {
            for i in 0..inner {
                let base = o * axis_size * inner + i;
                let mut best: Option<(usize, &T)> = None;
                for k in 0..axis_size {
                    let idx = base + k * inner;
                    if mask_slice[idx] {
                        continue;
                    }
                    let v = &data_slice[idx];
                    best = match best {
                        None => Some((k, v)),
                        Some((_, bv)) if (want_max && v > bv) || (!want_max && v < bv) => {
                            Some((k, v))
                        }
                        Some(b) => Some(b),
                    };
                }
                match best {
                    Some((k, _)) => out.push(k),
                    None => {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "{}: every element of this lane is masked, so there is no unmasked \
                             position to return (numpy.ma silently returns the degenerate index \
                             0 here; see MaskedArray::argmin's doc comment for why this crate \
                             errors instead)",
                            if want_max { "argmax" } else { "argmin" }
                        )));
                    }
                }
            }
        }
        Array::from_vec_shape(out, &out_shape)
    }
}

impl<T: Float> MaskedArray<T> {
    /// Cumulative sum along `axis` (the whole array, flattened in C
    /// order, when `axis` is `None`). Masked elements contribute the
    /// additive identity (`0`, i.e. they do not advance the running sum)
    /// and **stay masked at their original position** in the output --
    /// this is a scan, not a reduction, so unlike every function in
    /// [`super::reductions`], no lane ever collapses or becomes "more
    /// masked" than its input; the output mask is exactly the input mask.
    ///
    /// Pinned against `numpy.ma`:
    /// `ma.array([1.,2.,3.,4.],mask=[F,T,F,F]).cumsum()` has data
    /// `[1.0, --, 4.0, 8.0]` (the running sum skips the masked `2.0`, so
    /// it resumes from `1` rather than `3`) and mask
    /// `[False, True, False, False]` -- identical to the input mask.
    pub fn cumsum(&self, axis: Option<isize>) -> Result<MaskedArray<T>> {
        let shape = self.shape();
        let ndim = shape.len();
        let data_op = crate::kernels::borrow::operand(self.get_data());
        let mask_op = crate::kernels::borrow::operand(self.get_mask());
        let data_slice: &[T] = &data_op;
        let mask_slice: &[bool] = &mask_op;

        match axis {
            None => {
                let mut out_data = Vec::with_capacity(data_slice.len());
                let mut running = T::zero();
                for (v, m) in data_slice.iter().zip(mask_slice.iter()) {
                    if !*m {
                        running = running + *v;
                    }
                    out_data.push(running);
                }
                let out_shape = vec![data_slice.len()];
                let out_mask = mask_slice.to_vec();
                Ok(MaskedArray {
                    data: Array::from_vec_shape(out_data, &out_shape)?,
                    mask: Array::from_vec_shape(out_mask, &out_shape)?,
                    fill_value: self.fill_value,
                })
            }
            Some(ax) => {
                let ax = normalize_axis(ax, ndim)?;
                let (outer, axis_size, inner) = axis_lane_shape(&shape, ax);
                let total = outer * axis_size * inner;
                let mut out_data = vec![T::zero(); total];
                let mut out_mask = vec![false; total];
                for o in 0..outer {
                    for i in 0..inner {
                        let base = o * axis_size * inner + i;
                        let mut running = T::zero();
                        for k in 0..axis_size {
                            let idx = base + k * inner;
                            let m = mask_slice[idx];
                            if !m {
                                running = running + data_slice[idx];
                            }
                            out_data[idx] = running;
                            out_mask[idx] = m;
                        }
                    }
                }
                Ok(MaskedArray {
                    data: Array::from_vec_shape(out_data, &shape)?,
                    mask: Array::from_vec_shape(out_mask, &shape)?,
                    fill_value: self.fill_value,
                })
            }
        }
    }
}

impl<T: PartialOrd + Clone + Default> MaskedArray<T> {
    /// Sort along `axis` (the whole array, flattened, when `axis` is
    /// `None`), with masked values sorted to the end of each lane --
    /// exactly `np.ma.sort`. Unmasked elements within a lane are sorted
    /// ascending; masked elements keep their relative order among
    /// themselves (the sort is stable) but always follow every unmasked
    /// element, regardless of what values happen to sit underneath their
    /// mask.
    ///
    /// Unlike `math::aggregation::sort`, there is no `kind`/`order`
    /// parameter: this always uses a stable sort, and there are no named
    /// fields to sort by. Both are deliberately out of scope here.
    ///
    /// Pinned against `numpy.ma`:
    /// `np.ma.sort(ma.array([3.,1.,2.,0.],mask=[F,T,F,T]))` has data
    /// `[2.0, 3.0, --, --]` and mask `[False, False, True, True]`.
    pub fn sort(&self, axis: Option<isize>) -> Result<Self> {
        let shape = self.shape();
        let ndim = shape.len();
        let data_op = crate::kernels::borrow::operand(self.get_data());
        let mask_op = crate::kernels::borrow::operand(self.get_mask());
        let data_slice: &[T] = &data_op;
        let mask_slice: &[bool] = &mask_op;

        let (out_shape, outer, axis_size, inner): (Vec<usize>, usize, usize, usize) = match axis {
            None => (vec![data_slice.len()], 1, data_slice.len(), 1),
            Some(ax) => {
                let ax = normalize_axis(ax, ndim)?;
                let (outer, axis_size, inner) = axis_lane_shape(&shape, ax);
                (shape.clone(), outer, axis_size, inner)
            }
        };

        let total = outer * axis_size * inner;
        let mut out_data = vec![T::default(); total];
        let mut out_mask = vec![false; total];
        let mut lane: Vec<(T, bool)> = Vec::with_capacity(axis_size);

        for o in 0..outer {
            for i in 0..inner {
                let base = o * axis_size * inner + i;
                lane.clear();
                for k in 0..axis_size {
                    let idx = base + k * inner;
                    lane.push((data_slice[idx].clone(), mask_slice[idx]));
                }
                // Stable: unmasked elements ascending, masked elements at
                // the end (in their original relative order among
                // themselves), matching `np.ma.sort`.
                lane.sort_by(|(av, am), (bv, bm)| match (*am, *bm) {
                    (false, false) => av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal),
                    (false, true) => std::cmp::Ordering::Less,
                    (true, false) => std::cmp::Ordering::Greater,
                    (true, true) => std::cmp::Ordering::Equal,
                });
                for (k, (v, m)) in lane.iter().enumerate() {
                    let idx = base + k * inner;
                    out_data[idx] = v.clone();
                    out_mask[idx] = *m;
                }
            }
        }

        Ok(MaskedArray {
            data: Array::from_vec_shape(out_data, &out_shape)?,
            mask: Array::from_vec_shape(out_mask, &out_shape)?,
            fill_value: self.fill_value.clone(),
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
    fn argmin_argmax_skip_masked_elements() {
        let m = ma(
            vec![3.0, 1.0, 2.0, 0.0],
            vec![false, true, false, true],
            &[4],
        );
        assert_eq!(
            m.argmin(None, false).expect("has unmasked").to_vec(),
            vec![2]
        );
        assert_eq!(
            m.argmax(None, false).expect("has unmasked").to_vec(),
            vec![0]
        );
    }

    #[test]
    fn argmin_argmax_axis() {
        let m = ma(
            vec![3.0, 1.0, 0.0, 2.0],
            vec![false, true, true, false],
            &[2, 2],
        );
        assert_eq!(
            m.argmin(Some(1), false).expect("axis 1 valid").to_vec(),
            vec![0, 1]
        );
        assert_eq!(
            m.argmin(Some(0), false).expect("axis 0 valid").to_vec(),
            vec![0, 1]
        );
    }

    #[test]
    fn argmin_errors_on_fully_masked_lane() {
        let m = ma(vec![3.0, 1.0], vec![true, true], &[2]);
        assert!(m.argmin(None, false).is_err());
        assert!(m.argmax(None, false).is_err());
    }

    #[test]
    fn argmin_axis_none_ignores_keepdims_matching_free_function_cousin() {
        let m = ma(vec![3.0, 1.0, 2.0], vec![false, false, false], &[3]);
        let a = m.argmin(None, false).expect("has data");
        let b = m.argmin(None, true).expect("has data");
        assert_eq!(a.shape(), vec![1]);
        assert_eq!(b.shape(), vec![1]);
    }

    #[test]
    fn cumsum_masked_elements_contribute_identity_and_stay_masked() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![false, true, false, false],
            &[4],
        );
        let r = m.cumsum(None).expect("reduces");
        assert_eq!(r.get_mask().to_vec(), vec![false, true, false, false]);
        let vals = r.get_data().to_vec();
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[2], 4.0); // 1 + 3, the masked 2 skipped
        assert_eq!(vals[3], 8.0); // + 4
    }

    #[test]
    fn cumsum_along_axis_shape_matches_input() {
        let m = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4], &[2, 2]);
        let r = m.cumsum(Some(1)).expect("axis 1 valid");
        assert_eq!(r.shape(), vec![2, 2]);
        assert_eq!(r.get_data().to_vec(), vec![1.0, 3.0, 3.0, 7.0]);
    }

    #[test]
    fn sort_pushes_masked_values_to_the_end() {
        let m = ma(
            vec![3.0, 1.0, 2.0, 0.0],
            vec![false, true, false, true],
            &[4],
        );
        let r = m.sort(None).expect("sorts");
        // Output length always matches input length (masked slots are
        // still real, addressable `T` values under the mask -- here the
        // original masked `1.0`@1 and `0.0`@3, carried through in their
        // original relative order since the sort is stable).
        assert_eq!(r.get_data().to_vec(), vec![2.0, 3.0, 1.0, 0.0]);
        assert_eq!(&r.get_mask().to_vec()[..2], &[false, false]);
        assert_eq!(&r.get_mask().to_vec()[2..], &[true, true]);
    }

    #[test]
    fn sort_along_axis_pushes_masked_to_end_of_each_lane() {
        let m = ma(
            vec![3.0, 1.0, 0.0, 2.0],
            vec![false, true, true, false],
            &[2, 2],
        );
        let r = m.sort(Some(1)).expect("axis 1 valid");
        assert_eq!(r.get_data().to_vec()[0], 3.0);
        assert_eq!(r.get_data().to_vec()[2], 2.0);
        assert_eq!(r.get_mask().to_vec(), vec![false, true, false, true]);
    }

    #[test]
    fn sort_out_of_bounds_axis_is_an_error() {
        let m = ma(vec![1.0, 2.0], vec![false, false], &[2]);
        assert!(m.sort(Some(5)).is_err());
    }
}

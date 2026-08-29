//! Mask-aware reductions: the shared axis-walking engine, the pre-existing
//! whole-array `mean`/`sum`/`min`/`max`, and their `axis`+`keepdims`-aware
//! siblings plus `std`/`var`/`prod`/`median`/`ptp`.
//!
//! # Why `mean_axis`/`sum_axis`/`min_axis`/`max_axis` instead of adding
//! `axis`/`keepdims` parameters directly to `mean`/`sum`/`min`/`max`
//!
//! This lane's file ownership is `src/masked.rs` (now `src/masked/`)
//! only. `tests/test_masked.rs` (a different file, outside that
//! ownership, "MANY agents run concurrently on this tree: NEVER touch
//! files outside your ownership list") calls `masked.mean()`,
//! `.sum()`, `.min()`, `.max()` with **zero** arguments and treats the
//! result as `Option<T>` (it calls `unwrap` directly on `masked.mean()`
//! and asserts the result equals `3.0`, alongside
//! `assert!(all_masked.mean().is_none())`). Adding required
//! parameters to those exact methods would not compile against that file.
//! Adding new `_axis`-suffixed methods alongside the untouched originals
//! is the only change that is both non-breaking and satisfies the task's
//! "axis= on the reductions that already exist ... if missing": the
//! `_axis` siblings below accept `axis: Option<isize>` (so `axis: None`
//! covers what the originals already did) and `keepdims`, and return
//! `Result<MaskedArray<T>>` rather than `Option<T>` -- this is what makes
//! `axis: Some(_)` expressible at all, since a `Some(axis)` reduction can
//! produce a *per-lane* masked result (some lanes all-masked, others not)
//! that a single `Option<T>` cannot represent. This mirrors `numpy.ma`
//! exactly: `x.mean()` returns a scalar (`numpy.ma.masked` if `x` is fully
//! masked), but `x.mean(axis=0)` returns a `MaskedArray`.
//!
//! `std`/`var`/`prod`/`median`/`ptp`/`any`/`all`/`argmin`/`argmax` have no
//! such pre-existing zero-arg method to preserve, so they take
//! `axis`/`keepdims` directly under their plain NumPy names.

use super::MaskedArray;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};
use std::ops::{Add, Div};

// ---------------------------------------------------------------------
// Shared axis-walking engine
// ---------------------------------------------------------------------

/// Normalize a (possibly negative) axis index against `ndim`, matching
/// every other axis-taking function in this crate: `-1` is the last axis.
pub(super) fn normalize_axis(ax: isize, ndim: usize) -> Result<usize> {
    let normalized = if ax < 0 { ax + ndim as isize } else { ax };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {ax} out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
}

/// Decompose `shape` around an already-normalized axis `ax` into
/// `(outer, axis_size, inner)`: the number of lanes before `ax` (`outer`),
/// the length of the axis being walked (`axis_size`), and the number of
/// lanes after it (`inner`).
///
/// For a C-contiguous flat buffer of this shape, the elements of the lane
/// at `(o, i)` (`o < outer`, `i < inner`) sit at flat indices
/// `o * axis_size * inner + i + k * inner` for `k` in `0..axis_size`; the
/// corresponding *collapsed* (axis removed) output element sits at flat
/// index `o * inner + i`. Every axis-walking function in this module
/// (this file's `reduce_lanes` and `search.rs`'s `argmin`/`argmax`/
/// `cumsum`/`sort`) is built on this one formula, walked over
/// [`crate::kernels::borrow::operand`] -- which normalizes both `data`
/// and `mask` to their *logical*, current-shape C order first -- so this
/// is correct for a transposed/permuted `MaskedArray` too, not only a
/// freshly-constructed contiguous one.
pub(super) fn axis_lane_shape(shape: &[usize], ax: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..ax].iter().product();
    let axis_size = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product();
    (outer, axis_size, inner)
}

/// Output shape of a reduction over axis `ax` of `shape`, honoring
/// `keepdims` the same way every axis+keepdims function elsewhere in this
/// crate does (e.g. `math::statistics::argmax`): the reduced axis becomes
/// size 1 when `keepdims` is set, or is removed entirely otherwise
/// (falling back to `[1]` if that would leave an empty shape vector).
pub(super) fn collapsed_shape(shape: &[usize], ax: usize, keepdims: bool) -> Vec<usize> {
    let mut out_shape = shape.to_vec();
    if keepdims {
        out_shape[ax] = 1;
    } else {
        out_shape.remove(ax);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
    }
    out_shape
}

/// The shared engine behind every *collapsing* mask-aware reduction in
/// this module (`mean_axis`, `sum_axis`, `min_axis`, `max_axis`, `std`,
/// `var`, `prod`, `median`, `ptp`, and `bool_ops::{any, all}`).
///
/// Walks `data`/`mask` lane-by-lane along `axis` (the whole flattened
/// array is a single lane when `axis` is `None`), gathers each lane's
/// values and masks into two scratch buffers, and hands them to `f`.
/// `f` returns `Some(value)` for an unmasked result or `None` to mark
/// that lane's output masked -- deciding *when* a lane counts as masked
/// (zero valid elements, or a stricter per-reduction rule such as `var`'s
/// `valid_count <= ddof`) is entirely `f`'s call; this helper only does
/// the axis bookkeeping and mask/shape assembly. A masked output slot's
/// underlying value is `U::default()` (never `f`'s "junk" computation --
/// there isn't one, since `f` returns `None` instead of a discarded
/// value), matching this crate's masked-result convention.
///
/// `axis: None` respects `keepdims` exactly like `math::statistics::mean`/
/// `math::statistics::var` do (shape `[1; ndim]` when set, `[1]`
/// otherwise) -- *unlike* `search.rs`'s `argmin`/`argmax`, whose own
/// unmasked cousin (`math::statistics::argmin`) ignores `keepdims` for
/// `axis: None`; each function here matches its *own* specific unmasked
/// cousin rather than a single crate-wide rule.
pub(super) fn reduce_lanes<T, U>(
    data: &Array<T>,
    mask: &Array<bool>,
    axis: Option<isize>,
    keepdims: bool,
    f: impl Fn(&[T], &[bool]) -> Option<U>,
) -> Result<MaskedArray<U>>
where
    T: Clone,
    U: Clone + Default,
{
    let shape = data.shape();
    let ndim = shape.len();
    let data_op = crate::kernels::borrow::operand(data);
    let mask_op = crate::kernels::borrow::operand(mask);
    let data_slice: &[T] = &data_op;
    let mask_slice: &[bool] = &mask_op;

    let (out_shape, outer, axis_size, inner) = match axis {
        None => {
            let out_shape = if keepdims {
                vec![1; ndim.max(1)]
            } else {
                vec![1]
            };
            (out_shape, 1usize, data_slice.len(), 1usize)
        }
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
    let mut out_data: Vec<U> = Vec::with_capacity(out_len);
    let mut out_mask: Vec<bool> = Vec::with_capacity(out_len);
    let mut lane_vals: Vec<T> = Vec::with_capacity(axis_size);
    let mut lane_mask: Vec<bool> = Vec::with_capacity(axis_size);

    for o in 0..outer {
        for i in 0..inner {
            lane_vals.clear();
            lane_mask.clear();
            let base = o * axis_size * inner + i;
            for k in 0..axis_size {
                let idx = base + k * inner;
                lane_vals.push(data_slice[idx].clone());
                lane_mask.push(mask_slice[idx]);
            }
            match f(&lane_vals, &lane_mask) {
                Some(v) => {
                    out_data.push(v);
                    out_mask.push(false);
                }
                None => {
                    out_data.push(U::default());
                    out_mask.push(true);
                }
            }
        }
    }

    Ok(MaskedArray {
        data: Array::from_vec_shape(out_data, &out_shape)?,
        mask: Array::from_vec_shape(out_mask, &out_shape)?,
        fill_value: U::default(),
    })
}

// ---------------------------------------------------------------------
// Pre-existing whole-array reductions (unchanged; moved from the
// pre-split masked.rs verbatim -- see this file's module doc comment for
// why they are not extended with an `axis` parameter directly).
// ---------------------------------------------------------------------

impl<T: Clone + Add<Output = T> + Div<Output = T> + Zero + From<f64> + Into<f64>> MaskedArray<T> {
    /// Calculate the mean of unmasked elements
    ///
    /// # Returns
    ///
    /// The mean value, or None if all elements are masked
    pub fn mean(&self) -> Option<T> {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut sum = T::zero();
        let mut count = 0;

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if !*is_masked {
                sum = sum + value.clone();
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            // We need to convert to/from f64 to properly handle division
            let count_f64 = count as f64;
            let sum_f64: f64 = sum.into();
            Some(T::from(sum_f64 / count_f64))
        }
    }

    /// Calculate the sum of unmasked elements
    ///
    /// # Returns
    ///
    /// The sum, or None if all elements are masked
    pub fn sum(&self) -> Option<T> {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut sum = T::zero();
        let mut count = 0;

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if !*is_masked {
                sum = sum + value.clone();
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some(sum)
        }
    }

    /// Find the minimum value among unmasked elements
    ///
    /// # Returns
    ///
    /// The minimum value, or None if all elements are masked
    pub fn min(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut min_val = None;

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if !*is_masked {
                match min_val {
                    None => min_val = Some(value.clone()),
                    Some(ref current_min) if value < current_min => min_val = Some(value.clone()),
                    _ => {}
                }
            }
        }

        min_val
    }

    /// Find the maximum value among unmasked elements
    ///
    /// # Returns
    ///
    /// The maximum value, or None if all elements are masked
    pub fn max(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut max_val = None;

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if !*is_masked {
                match max_val {
                    None => max_val = Some(value.clone()),
                    Some(ref current_max) if value > current_max => max_val = Some(value.clone()),
                    _ => {}
                }
            }
        }

        max_val
    }
}

// ---------------------------------------------------------------------
// New axis+keepdims-aware reductions
// ---------------------------------------------------------------------

/// `T: Float` supplies everything these need generically (`Zero`, `One`,
/// `PartialOrd`, `Copy`, `NumCast`, the four arithmetic ops, `sqrt`,
/// `is_nan`) -- but *not* `Default` (unlike `Zero::zero()`, `Float` has no
/// blanket `Default` impl), which `reduce_lanes`'s `U: Default` bound
/// needs to fill a masked output slot. Hence the explicit `+ Default`
/// here; both `f32` and `f64` satisfy it trivially (`0.0`).
impl<T: Float + Default> MaskedArray<T> {
    /// `mean`, honoring an optional axis and `keepdims`. `axis: None`
    /// behaves like [`MaskedArray::mean`] but returns a 1-element
    /// `MaskedArray` (masked, not `None`, if every element is masked)
    /// instead of `Option<T>`; `axis: Some(ax)` reduces only along `ax`,
    /// masking any output lane that had zero unmasked contributors.
    ///
    /// Pinned against `numpy.ma`: for
    /// `x = ma.array([[1,--,3],[4,5,--]])`, `x.mean(axis=0) == [2.5, 5.0, 3.0]`
    /// (column 1 has one valid entry, `5.0`; column 2 has one valid entry, `3.0`).
    pub fn mean_axis(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut sum = T::zero();
            let mut count = 0usize;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    sum = sum + *v;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                Some(sum / T::from(count).expect("lane length fits in T"))
            }
        })
    }

    /// `sum`, honoring an optional axis and `keepdims`; masked elements
    /// contribute nothing (the additive identity). See [`Self::mean_axis`]
    /// for the `axis`/`keepdims`/masking convention.
    pub fn sum_axis(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut sum = T::zero();
            let mut count = 0usize;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    sum = sum + *v;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                Some(sum)
            }
        })
    }

    /// `min`, honoring an optional axis and `keepdims`.
    ///
    /// Deliberately **not** NaN-propagating: like the pre-existing
    /// whole-array [`MaskedArray::min`] (a plain `<` fold), a `NaN` among
    /// the *unmasked* elements of a lane is silently skipped rather than
    /// poisoning the lane's result, unlike `kernels::reduce::min_f64`'s
    /// `np.min` convention used elsewhere in this crate. Keeping this
    /// `_axis` sibling on the same rule as the method it extends avoids
    /// two APIs on the same type silently disagreeing about `NaN`; use
    /// [`MaskedArray::masked_invalid`] to mask `NaN`s out up front if you
    /// need them excluded.
    pub fn min_axis(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut best: Option<T> = None;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    best = Some(match best {
                        None => *v,
                        Some(cur) if *v < cur => *v,
                        Some(cur) => cur,
                    });
                }
            }
            best
        })
    }

    /// `max`, honoring an optional axis and `keepdims`. See
    /// [`Self::min_axis`] for the (deliberately non-NaN-propagating)
    /// comparison rule.
    pub fn max_axis(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut best: Option<T> = None;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    best = Some(match best {
                        None => *v,
                        Some(cur) if *v > cur => *v,
                        Some(cur) => cur,
                    });
                }
            }
            best
        })
    }

    /// Variance of unmasked elements, `ddof`-aware (divisor `n - ddof`;
    /// `ddof = 0` is this crate's population-variance convention,
    /// matching `math::var`'s default and `numpy.ma`'s own default).
    ///
    /// A lane is masked in the output not only when it has zero unmasked
    /// elements, but also whenever its unmasked count is `<= ddof` (e.g.
    /// a lane with exactly one valid element and `ddof = 1`): rather than
    /// dividing by a non-positive divisor and returning `inf`/`NaN`
    /// (`math::var`'s free-function behavior, which requires the caller
    /// to pre-check `n <= ddof` against the *whole* array since it has no
    /// per-lane concept), this crate's masked lane reduction treats "not
    /// enough unmasked data to define this statistic" as exactly the same
    /// kind of masked result as "no unmasked data at all".
    ///
    /// Pinned against `numpy.ma`: `x.var(axis=0)` (`ddof=0`) on
    /// `x = ma.array([[1,--,3],[4,5,--]])` is `[2.25, 0.0, 0.0]`.
    pub fn var(&self, axis: Option<isize>, ddof: usize, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut sum = T::zero();
            let mut count = 0usize;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    sum = sum + *v;
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let n = T::from(count).expect("lane length fits in T");
            let mean = sum / n;
            let mut sum_sq = T::zero();
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    let d = *v - mean;
                    sum_sq = sum_sq + d * d;
                }
            }
            let divisor_n = count.checked_sub(ddof)?;
            if divisor_n == 0 {
                return None;
            }
            let divisor = T::from(divisor_n).expect("divisor fits in T");
            Some(sum_sq / divisor)
        })
    }

    /// Standard deviation of unmasked elements: `sqrt(var(..))`. See
    /// [`Self::var`] for the `ddof`/masking convention.
    ///
    /// Pinned against `numpy.ma`: `x.std(axis=1, ddof=1)` on
    /// `x = ma.array([[1,--,3],[4,5,--]])` is
    /// `[1.4142135623730951, 0.7071067811865476]`.
    pub fn std(&self, axis: Option<isize>, ddof: usize, keepdims: bool) -> Result<MaskedArray<T>> {
        let variance = self.var(axis, ddof, keepdims)?;
        Ok(MaskedArray {
            data: variance.data.map(|x| x.sqrt()),
            mask: variance.mask,
            fill_value: self.fill_value,
        })
    }

    /// Product of unmasked elements, honoring an optional axis and
    /// `keepdims`; masked elements contribute nothing (the multiplicative
    /// identity `1`).
    pub fn prod(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut p = T::one();
            let mut count = 0usize;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    p = p * *v;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                Some(p)
            }
        })
    }

    /// Median of unmasked elements: the middle value of the sorted
    /// unmasked elements, or the mean of the two middle values when their
    /// count is even -- exactly `numpy.ma.median`'s definition, applied
    /// to the unmasked elements of each lane instead of the whole array.
    ///
    /// # `NaN`
    ///
    /// A `NaN` among a lane's *unmasked* elements makes that lane's median
    /// `NaN` -- **unmasked**, not a masked slot -- regardless of the
    /// `NaN`'s position, matching `numpy.ma.median` exactly (confirmed
    /// against `numpy 2.4.2` at every position in a 5- and a 4-element
    /// lane: leading, interior and trailing all give `NaN`). This needs
    /// an explicit check rather than falling out of the sort the way
    /// [`Self::mean_axis`]/[`Self::sum_axis`]/[`Self::var`]'s plain `+`
    /// folds do: `f64`/`f32`'s `partial_cmp` returns `None` for any
    /// comparison against `NaN`, and the `unwrap_or(Equal)` fallback a
    /// sort comparator is forced to supply for that case is not
    /// transitive (`NaN` compares `Equal` to *everything*, including
    /// values that are not mutually `Equal`) -- so plugging it into
    /// `sort_by` does not reliably relocate every `NaN` to a
    /// position-independent place in the output; empirically, it
    /// silently drops the propagation for a `NaN` at some positions while
    /// keeping it at others (checked exhaustively across leading,
    /// interior and trailing placements in both odd- and even-length
    /// lanes; the version of this function that shipped without this
    /// paragraph, and the check below, reproduced exactly that: `NaN` at
    /// position 0 of `[NaN,1,3,4]` was silently lost, coming back as
    /// `2.0` instead of `NaN`, while every other tested position
    /// propagated correctly).
    ///
    /// Pinned against `numpy.ma`: for
    /// `x = ma.array([[5,--,3],[2,4,--]])`, `ma.median(x, axis=1) == [4.0, 3.0]`
    /// (row 0's unmasked elements are `[5,3]`, mean `4.0`; row 1's are
    /// `[2,4]`, mean `3.0`) and `ma.median(x, axis=0) == [3.5, 4.0, 3.0]`.
    pub fn median(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut unmasked: Vec<T> = vals
                .iter()
                .zip(masks)
                .filter(|(_, m)| !**m)
                .map(|(v, _)| *v)
                .collect();
            if unmasked.is_empty() {
                return None;
            }
            if unmasked.iter().any(|v| v.is_nan()) {
                // Unmasked `NaN`, exactly like `numpy.ma.median` -- see
                // this method's `# NaN` doc section for why this check
                // has to come before the sort below, not fall out of it.
                return Some(T::nan());
            }
            unmasked.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = unmasked.len();
            if n % 2 == 1 {
                Some(unmasked[n / 2])
            } else {
                let two = T::one() + T::one();
                Some((unmasked[n / 2 - 1] + unmasked[n / 2]) / two)
            }
        })
    }

    /// Peak-to-peak range (`max - min`) of unmasked elements, honoring an
    /// optional axis and `keepdims`. Like [`Self::min_axis`]/
    /// [`Self::max_axis`], comparisons do not propagate `NaN`.
    ///
    /// Pinned against `numpy.ma`: `x.ptp(axis=1)` on
    /// `x = ma.array([[--,2],[3,4]])` is `[0.0, 1.0]` (row 0 has a single
    /// unmasked element, `2.0`, so its range is `0.0`, not masked -- a
    /// lane is only masked here when it has *zero* unmasked elements).
    pub fn ptp(&self, axis: Option<isize>, keepdims: bool) -> Result<MaskedArray<T>> {
        reduce_lanes(&self.data, &self.mask, axis, keepdims, |vals, masks| {
            let mut min_v: Option<T> = None;
            let mut max_v: Option<T> = None;
            for (v, m) in vals.iter().zip(masks) {
                if !*m {
                    min_v = Some(match min_v {
                        None => *v,
                        Some(cur) if *v < cur => *v,
                        Some(cur) => cur,
                    });
                    max_v = Some(match max_v {
                        None => *v,
                        Some(cur) if *v > cur => *v,
                        Some(cur) => cur,
                    });
                }
            }
            match (min_v, max_v) {
                (Some(mn), Some(mx)) => Some(mx - mn),
                _ => None,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    fn ma(data: Vec<f64>, mask: Vec<bool>, shape: &[usize]) -> MaskedArray<f64> {
        MaskedArray {
            data: Array::from_vec_shape(data, shape).expect("valid shape"),
            mask: Array::from_vec_shape(mask, shape).expect("valid shape"),
            fill_value: 0.0,
        }
    }

    // ---- agreement between the old Option<T> API and the new _axis API ----

    #[test]
    fn mean_axis_none_agrees_with_scalar_mean() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, true, false, true, false],
            &[5],
        );
        let scalar = m.mean().expect("has unmasked elements");
        let axis_form = m.mean_axis(None, false).expect("reduces");
        assert!(!axis_form.get_mask().to_vec()[0]);
        assert_eq!(axis_form.get_data().to_vec()[0], scalar);
    }

    #[test]
    fn sum_min_max_axis_none_agree_with_scalar_forms() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, true, false, true, false],
            &[5],
        );
        assert_eq!(
            m.sum_axis(None, false)
                .expect("reduces")
                .get_data()
                .to_vec()[0],
            m.sum().expect("has data")
        );
        assert_eq!(
            m.min_axis(None, false)
                .expect("reduces")
                .get_data()
                .to_vec()[0],
            m.min().expect("has data")
        );
        assert_eq!(
            m.max_axis(None, false)
                .expect("reduces")
                .get_data()
                .to_vec()[0],
            m.max().expect("has data")
        );
    }

    #[test]
    fn all_masked_axis_none_is_masked_not_none() {
        let m = ma(vec![1.0, 2.0, 3.0], vec![true, true, true], &[3]);
        assert!(m.mean().is_none());
        let r = m.mean_axis(None, false).expect("reduces to a masked slot");
        assert!(r.get_mask().to_vec()[0]);
    }

    // ---- pinned numpy.ma values ----

    /// `ma.array([[1,--,3],[4,5,--]]).mean(axis=0) == [2.5, 5.0, 3.0]`
    #[test]
    fn mean_axis_0_matches_numpy_ma() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let r = m.mean_axis(Some(0), false).expect("axis 0 valid");
        assert_eq!(r.get_data().to_vec(), vec![2.5, 5.0, 3.0]);
        assert_eq!(r.get_mask().to_vec(), vec![false, false, false]);
    }

    /// Same array, permuted to a genuinely non-contiguous view first (via
    /// `Array::transpose_axis`, not `MaskedArray::transpose` -- the
    /// latter is this crate's own eager, data-*copying* transpose, so it
    /// always comes back C-contiguous and would not exercise this path;
    /// see `array::manipulation::transpose`'s 2D branch, which builds a
    /// fresh buffer via `Array::from_vec_shape`). The lane walk goes
    /// through `kernels::borrow::operand`, which normalizes a
    /// non-contiguous source to logical order, so this must agree with
    /// the untransposed case.
    #[test]
    fn mean_axis_agrees_on_transposed_non_contiguous_input() {
        let base = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let m = MaskedArray {
            data: base.get_data().transpose_axis(0, 1),
            mask: base.get_mask().transpose_axis(0, 1),
            fill_value: 0.0,
        };
        assert!(!m.get_data().is_c_contiguous());
        // Column `j` of the original is row `j` of the permuted view, so
        // reducing along axis 1 of the permuted view reproduces the
        // original's axis-0 reduction.
        let r = m.mean_axis(Some(1), false).expect("axis 1 valid");
        assert_eq!(r.get_data().to_vec(), vec![2.5, 5.0, 3.0]);
    }

    /// A genuinely 3-D case (shape `[2, 3, 4]`, reducing the *middle* axis)
    /// with a scattered mask, pinned against `numpy.ma`. Every other test
    /// in this file uses `ndim <= 2`, where `outer` or `inner` in
    /// `reduce_lanes` is trivially `1`; this is the only test that
    /// exercises both being `> 1` simultaneously (`outer = 2`,
    /// `inner = 4`), which is what actually confirms the flat-index
    /// formula `o * axis_size * inner + i + k * inner` in
    /// [`axis_lane_shape`]'s doc comment, rather than merely a
    /// 2-D-shaped special case of it. Reference values from
    /// `numpy 2.4.2`:
    /// ```python
    /// data = np.arange(24.).reshape(2,3,4)
    /// mask = np.zeros((2,3,4), dtype=bool)
    /// mask[0,1,2]=True; mask[1,0,0]=True; mask[1,2,3]=True
    /// mask[0,0,3]=True; mask[0,1,3]=True; mask[0,2,3]=True  # (0,:,3) fully masked
    /// ma.array(data, mask=mask).mean(axis=1)
    /// # -> data [[4,5,6,--],[18,17,18,17]], mask [[F,F,F,T],[F,F,F,F]]
    /// ```
    #[test]
    fn mean_axis_1_matches_numpy_ma_on_a_3d_array() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let mut mask = vec![false; 24];
        // Flat C-order index for (d0, d1, d2) in shape [2,3,4] is
        // d0*12 + d1*4 + d2.
        for &(d0, d1, d2) in &[
            (0usize, 1usize, 2usize),
            (1, 0, 0),
            (1, 2, 3),
            (0, 0, 3),
            (0, 1, 3),
            (0, 2, 3),
        ] {
            mask[d0 * 12 + d1 * 4 + d2] = true;
        }
        let m = ma(data, mask, &[2, 3, 4]);
        let r = m.mean_axis(Some(1), false).expect("axis 1 valid");
        assert_eq!(r.shape(), vec![2, 4]);
        assert_eq!(
            r.get_mask().to_vec(),
            vec![false, false, false, true, false, false, false, false]
        );
        let got = r.get_data().to_vec();
        let want = [4.0, 5.0, 6.0, 0.0, 18.0, 17.0, 18.0, 17.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    /// `ma.array([1,2,3,4],mask=[T,T,F,F]).reshape(2,2).mean(axis=1) == [--, 3.5]`, mask `[T,F]`.
    #[test]
    fn mean_axis_masks_a_fully_masked_lane_but_not_others() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true, false, false],
            &[2, 2],
        );
        let r = m.mean_axis(Some(1), false).expect("axis 1 valid");
        assert_eq!(r.get_mask().to_vec(), vec![true, false]);
        assert_eq!(r.get_data().to_vec()[1], 3.5);
    }

    /// `ma.array([[1,--,3],[4,5,--]]).var(axis=0) == [2.25, 0.0, 0.0]` (`ddof=0`).
    #[test]
    fn var_axis_0_matches_numpy_ma() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let r = m.var(Some(0), 0, false).expect("axis 0 valid");
        let got = r.get_data().to_vec();
        assert!((got[0] - 2.25).abs() < 1e-12);
        assert!((got[1] - 0.0).abs() < 1e-12);
        assert!((got[2] - 0.0).abs() < 1e-12);
    }

    /// `ma.array([[1,--,3],[4,5,--]]).std(axis=1, ddof=1) == [1.4142135623730951, 0.7071067811865476]`.
    #[test]
    fn std_axis_1_ddof_1_matches_numpy_ma() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let r = m.std(Some(1), 1, false).expect("axis 1 valid");
        let got = r.get_data().to_vec();
        assert!((got[0] - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!((got[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    /// A lane with exactly one unmasked element and `ddof=1` has no
    /// degrees of freedom left: masked, not `inf`/`NaN`.
    #[test]
    fn var_masks_a_lane_with_valid_count_at_or_below_ddof() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![false, true, false, true],
            &[2, 2],
        );
        // row 0 = [1, --] (1 valid), row 1 = [3, --] (1 valid): ddof=1 leaves
        // 0 degrees of freedom in both.
        let r = m.var(Some(1), 1, false).expect("axis 1 valid");
        assert_eq!(r.get_mask().to_vec(), vec![true, true]);
    }

    /// `ma.median(ma.array([[5,--,3],[2,4,--]]), axis=1) == [4.0, 3.0]`,
    /// `axis=0 == [3.5, 4.0, 3.0]`.
    #[test]
    fn median_matches_numpy_ma() {
        let m = ma(
            vec![5.0, 1.0, 3.0, 2.0, 4.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let r1 = m.median(Some(1), false).expect("axis 1 valid");
        assert_eq!(r1.get_data().to_vec(), vec![4.0, 3.0]);
        let r0 = m.median(Some(0), false).expect("axis 0 valid");
        assert_eq!(r0.get_data().to_vec(), vec![3.5, 4.0, 3.0]);
    }

    /// An unmasked `NaN` makes the median `NaN`, at every position, in
    /// both an odd- and an even-length lane -- pinned against `numpy.ma`
    /// (`numpy 2.4.2`) at every position tested. This is the regression
    /// this method's `# NaN` doc section describes: a naive
    /// `sort_by(partial_cmp().unwrap_or(Equal))` silently loses the
    /// propagation for some (not all) of these exact positions.
    #[test]
    fn median_propagates_unmasked_nan_at_every_position() {
        let nan = f64::NAN;
        // Odd length (5): median index 2.
        for data in [
            vec![nan, 5.0, 3.0, 2.0, 4.0],
            vec![5.0, nan, 3.0, 2.0, 4.0],
            vec![5.0, 3.0, 2.0, 4.0, nan],
        ] {
            let m = ma(data.clone(), vec![false; 5], &[5]);
            let r = m.median(None, false).expect("has unmasked data");
            assert!(
                r.get_data().to_vec()[0].is_nan(),
                "data={data:?} should have produced a NaN median"
            );
            assert!(!r.get_mask().to_vec()[0], "NaN median must be unmasked");
        }
        // Even length (4): the two middle elements average.
        for data in [
            vec![nan, 1.0, 3.0, 4.0],
            vec![1.0, nan, 3.0, 4.0],
            vec![1.0, 3.0, 4.0, nan],
        ] {
            let m = ma(data.clone(), vec![false; 4], &[4]);
            let r = m.median(None, false).expect("has unmasked data");
            assert!(
                r.get_data().to_vec()[0].is_nan(),
                "data={data:?} should have produced a NaN median"
            );
        }
    }

    /// A `NaN` sitting under a *masked* element must NOT propagate: it is
    /// excluded from the lane's unmasked values entirely, the same as any
    /// other masked value, and the median is computed from the remaining
    /// unmasked (non-`NaN`) elements. `ma.median(ma.array([1.,nan,3.],mask=[F,T,F])) == 2.0`.
    #[test]
    fn median_ignores_nan_under_a_mask() {
        let m = ma(vec![1.0, f64::NAN, 3.0], vec![false, true, false], &[3]);
        let r = m.median(None, false).expect("has unmasked data");
        assert_eq!(r.get_data().to_vec()[0], 2.0);
        assert!(!r.get_mask().to_vec()[0]);
    }

    /// `ma.array([[--,2],[3,4]]).ptp(axis=1) == [0.0, 1.0]` -- a
    /// single-valid-element lane has `ptp == 0`, not masked.
    #[test]
    fn ptp_single_valid_element_lane_is_zero_not_masked() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, false, false, false],
            &[2, 2],
        );
        let r = m.ptp(Some(1), false).expect("axis 1 valid");
        assert_eq!(r.get_mask().to_vec(), vec![false, false]);
        assert_eq!(r.get_data().to_vec(), vec![0.0, 1.0]);
    }

    /// A fully-masked lane's `ptp` is masked.
    #[test]
    fn ptp_fully_masked_lane_is_masked() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![true, true, false, false],
            &[2, 2],
        );
        let r = m.ptp(Some(1), false).expect("axis 1 valid");
        assert_eq!(r.get_mask().to_vec(), vec![true, false]);
        assert_eq!(r.get_data().to_vec()[1], 1.0);
    }

    #[test]
    fn prod_skips_masked_like_identity() {
        let m = ma(
            vec![2.0, 100.0, 3.0, 4.0],
            vec![false, true, false, false],
            &[4],
        );
        let r = m.prod(None, false).expect("reduces");
        assert_eq!(r.get_data().to_vec()[0], 24.0); // 2 * 3 * 4, the masked 100 skipped
    }

    #[test]
    fn negative_axis_matches_positive_equivalent() {
        let m = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
            &[2, 3],
        );
        let pos = m.mean_axis(Some(1), false).expect("axis 1 valid");
        let neg = m.mean_axis(Some(-1), false).expect("axis -1 valid");
        assert_eq!(pos.get_data().to_vec(), neg.get_data().to_vec());
        assert_eq!(pos.get_mask().to_vec(), neg.get_mask().to_vec());
    }

    #[test]
    fn keepdims_true_keeps_reduced_axis_as_size_one() {
        let m = ma(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![false; 6], &[2, 3]);
        let r = m.sum_axis(Some(1), true).expect("axis 1 valid");
        assert_eq!(r.shape(), vec![2, 1]);
        let r_none = m.sum_axis(None, true).expect("valid");
        assert_eq!(r_none.shape(), vec![1, 1]);
    }

    #[test]
    fn out_of_bounds_axis_is_an_error() {
        let m = ma(vec![1.0, 2.0], vec![false, false], &[2]);
        assert!(m.mean_axis(Some(1), false).is_err());
        assert!(m.mean_axis(Some(-2), false).is_err());
    }
}

//! Memory-optimized array operations
//!
//! This module provides memory-efficient variants of common array operations
//! that minimize allocations through:
//! - Buffer reuse (in-place operations)
//! - Direct iterator usage (avoiding to_vec())
//! - Stack allocation for small arrays
//! - View-based operations
//!
//! # `dtype_fast_tier`: why contiguity, not `kernels::borrow::operand`
//!
//! Every reduction here (`sum_optimized`, `variance_optimized`,
//! `min_optimized`, `max_optimized`) gates its `f64`/`f32` fast tier on
//! [`Array::as_slice`] returning `Some` -- i.e. on the array actually being
//! contiguous -- rather than going through [`crate::kernels::borrow::operand`],
//! which the crate's GEMM paths use.
//!
//! The two are not interchangeable *for a reduction*. `operand()`'s
//! non-contiguous branch materializes a full `Vec<T>` copy of the input;
//! spending an `N`-element allocation to produce a single scalar is exactly
//! the `to_vec()` cost this module exists to avoid (see `sum_optimized`'s
//! own doc: "more memory-efficient than the standard `sum()` as it avoids
//! the `to_vec()` call"). A non-contiguous array therefore falls through to
//! the scalar/parallel iterator path below, which is allocation-free and
//! stride-correct. `matmul_2d`/`batched_gemm` in `array/linalg.rs` make the
//! opposite call, correctly: there the output is already `O(m*n)`, so a
//! materializing copy of an operand is amortized rather than pure overhead.
//!
//! # `population_variance`: `n`, never `n - 1`
//!
//! [`Array::variance_optimized`] and [`Array::std_optimized`] divide by `n`
//! (population variance), on every dtype and at every length.
//!
//! They must not be routed onto `scirs2_core::simd_ops::SimdUnifiedOps`'s
//! `simd_variance`/`simd_std`, which hardcode a `n - 1` (*sample*)
//! denominator and return `NaN` for `n < 2` -- see
//! `crate::kernels::reduce`'s module docs, which forbid those two entry
//! points crate-wide for this reason. The `f64`/`f32` fast tier instead
//! builds the same two-pass `sum_sq_dev / n` these methods' scalar path
//! computes, so the value is continuous across the
//! [`crate::kernels::SIMD_MIN_LEN`] boundary rather than silently switching
//! estimator at 64 elements.

use super::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels;
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::ndarray::Axis;
use scirs2_core::parallel_ops::*;
use std::ops::{Add, Mul};

/// Threshold for using parallel processing
const PARALLEL_THRESHOLD: usize = 10000;

impl<T: Clone> Array<T> {
    /// Calculate sum without allocating a Vec (uses iterator directly)
    ///
    /// This is more memory-efficient than the standard sum() as it avoids
    /// the to_vec() call.
    pub fn sum_optimized(&self) -> T
    where
        T: Add<Output = T> + Zero + Clone + 'static,
    {
        // Dtype-dispatched fast tier for a *contiguous* `f64`/`f32` array
        // (see this module's `dtype_fast_tier` note above for why
        // contiguity, rather than `kernels::borrow::operand`, gates it).
        if self.len() >= kernels::SIMD_MIN_LEN {
            if let Some(slice) = self.as_slice() {
                if let Some(a64) = kernels::cast::as_f64(slice) {
                    if let Some(out) = kernels::cast::f64_to::<T>(kernels::reduce::sum_f64(a64)) {
                        return out;
                    }
                }
                if let Some(a32) = kernels::cast::as_f32(slice) {
                    if let Some(out) = kernels::cast::f32_to::<T>(kernels::reduce::sum_f32(a32)) {
                        return out;
                    }
                }
            }
        }

        // Direct iteration without to_vec() allocation
        self.data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate product without allocating a Vec
    pub fn product_optimized(&self) -> T
    where
        T: Mul<Output = T> + One + Clone,
    {
        self.data.iter().fold(T::one(), |acc, x| acc * x.clone())
    }

    /// In-place map operation that reuses the current array's memory
    ///
    /// This modifies the array in place instead of allocating a new one.
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        for elem in self.nd_mut().iter_mut() {
            *elem = f(elem);
        }
    }

    /// Map operation that writes to a pre-allocated output buffer
    ///
    /// This avoids allocation by reusing the provided output array.
    /// Returns an error if the output shape doesn't match.
    pub fn map_to<F>(&self, f: F, output: &mut Array<T>) -> Result<()>
    where
        F: Fn(&T) -> T,
    {
        if self.shape() != output.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: output.shape(),
            });
        }

        for (src, dst) in self.data.iter().zip(output.nd_mut().iter_mut()) {
            *dst = f(src);
        }

        Ok(())
    }

    /// Sum along axis without multiple allocations
    ///
    /// This uses ndarray's built-in axis sum which is more efficient
    /// than our manual implementation.
    pub fn sum_axis_optimized(&self, axis: usize) -> Result<Self>
    where
        T: Add<Output = T> + Zero + Clone,
    {
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                self.ndim()
            )));
        }

        // Use ndarray's built-in sum_axis which is optimized
        let result = self.data.sum_axis(Axis(axis));
        Ok(Array::from_ndarray(result))
    }
}

// Optimized statistical operations
impl<T> Array<T>
where
    T: Float + Clone + Zero + NumCast + Send + Sync + 'static,
{
    /// Memory-optimized mean calculation
    ///
    /// Avoids to_vec() by iterating directly over the array.
    pub fn mean_optimized(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        let len = self.len();
        if len >= PARALLEL_THRESHOLD {
            // Use parallel reduction for large arrays
            let sum = self
                .data
                .view()
                .into_par_iter()
                .map(|&x| x)
                .reduce(|| T::zero(), |acc, x| acc + x);
            sum / T::from(len).expect("length should be representable")
        } else {
            let sum: T = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
            sum / T::from(len).expect("length should be representable")
        }
    }

    /// Memory-optimized variance calculation
    ///
    /// Uses a single-pass algorithm with direct iteration.
    pub fn variance_optimized(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        let len = self.len();

        // Dtype-dispatched fast tier: population variance (`ddof = 0`)
        // from the fused `kernels::reduce::var_*`, never `simd_variance`
        // -- see the `population_variance` note in this module's header
        // for why. The fused kernel takes one length-tier decision for
        // both of variance's passes; the separate `mean_*` +
        // `sum_sq_dev_*` calls this replaced took one each, which cost
        // more in `rayon` dispatch than it saved right at the threshold.
        if len >= kernels::SIMD_MIN_LEN {
            if let Some(slice) = self.as_slice() {
                if let Some(a64) = kernels::cast::as_f64(slice) {
                    if let Some(out) = kernels::cast::f64_to::<T>(kernels::reduce::var_f64(a64, 0))
                    {
                        return out;
                    }
                }
                if let Some(a32) = kernels::cast::as_f32(slice) {
                    if let Some(out) = kernels::cast::f32_to::<T>(kernels::reduce::var_f32(a32, 0))
                    {
                        return out;
                    }
                }
            }
        }

        let mean = self.mean_optimized();

        if len >= PARALLEL_THRESHOLD {
            let sum_sq_diff = self
                .data
                .view()
                .into_par_iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .reduce(|| T::zero(), |acc, x| acc + x);
            sum_sq_diff / T::from(len).expect("length should be representable")
        } else {
            let sum_sq_diff: T = self
                .data
                .iter()
                .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean));
            sum_sq_diff / T::from(len).expect("length should be representable")
        }
    }

    /// Memory-optimized standard deviation
    ///
    /// The square root of [`Array::variance_optimized`], and so *population*
    /// standard deviation (denominator `n`), on every dtype and every
    /// length. There is deliberately no separate dtype dispatch here: it
    /// would only duplicate the one `variance_optimized` already performs,
    /// and a second `scirs2-core` entry point (`simd_std`) would reintroduce
    /// the `n - 1` denominator this module does not use.
    pub fn std_optimized(&self) -> T {
        self.variance_optimized().sqrt()
    }

    /// Memory-optimized min calculation
    ///
    /// `None` for an empty array. Otherwise `NaN` propagates, matching
    /// `np.min`: any `NaN` anywhere in the array makes the result
    /// `Some(NaN)`. (`math::nanmin` is the `NaN`-ignoring variant.) The
    /// two tiers below agree on that rule -- the whole array is one
    /// reduction, so which tier runs must not be observable.
    pub fn min_optimized(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // Dtype-dispatched fast tier. Deliberately *not* gated on
        // `kernels::SIMD_MIN_LEN` any more: `kernels::reduce::min_*` is a
        // plain comparison fold now (it no longer calls
        // `SimdUnifiedOps::simd_min_element`, which returned wrong finite
        // values for some `NaN` placements -- see that module's docs), so
        // there is no SIMD setup cost left for a length threshold to
        // amortize, and gating on one would only mean short f64 arrays
        // took the generic tail and a *different* `NaN` rule than long
        // ones.
        if let Some(slice) = self.as_slice() {
            if let Some(a64) = kernels::cast::as_f64(slice) {
                if let Some(out) = kernels::cast::f64_to::<T>(kernels::reduce::min_f64(a64)) {
                    return Some(out);
                }
            }
            if let Some(a32) = kernels::cast::as_f32(slice) {
                if let Some(out) = kernels::cast::f32_to::<T>(kernels::reduce::min_f32(a32)) {
                    return Some(out);
                }
            }
        }

        // Generic tail (non-f64/f32 `T`, or a non-contiguous array whose
        // `as_slice()` is `None`). `saw_nan` rides alongside the extremum
        // because `<` is false for `NaN` operands and so cannot carry it.
        // The combiner is associative and commutative (min, plus a boolean
        // OR) with `(+inf, false)` as a true identity, so the parallel
        // branch does not depend on how rayon splits the work.
        let (acc, saw_nan) = if self.len() >= PARALLEL_THRESHOLD {
            self.data
                .view()
                .into_par_iter()
                .map(|&x| (x, x.is_nan()))
                .reduce(
                    || (T::infinity(), false),
                    |(a, a_nan), (b, b_nan)| (if b < a { b } else { a }, a_nan | b_nan),
                )
        } else {
            self.data
                .iter()
                .copied()
                .fold((T::infinity(), false), |(acc, saw_nan), x| {
                    (if x < acc { x } else { acc }, saw_nan | x.is_nan())
                })
        };
        Some(if saw_nan { T::nan() } else { acc })
    }

    /// Memory-optimized max calculation
    ///
    /// `None` for an empty array; `NaN` propagates otherwise. See
    /// [`Array::min_optimized`].
    pub fn max_optimized(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // Dtype-dispatched fast tier; see `min_optimized` above.
        if let Some(slice) = self.as_slice() {
            if let Some(a64) = kernels::cast::as_f64(slice) {
                if let Some(out) = kernels::cast::f64_to::<T>(kernels::reduce::max_f64(a64)) {
                    return Some(out);
                }
            }
            if let Some(a32) = kernels::cast::as_f32(slice) {
                if let Some(out) = kernels::cast::f32_to::<T>(kernels::reduce::max_f32(a32)) {
                    return Some(out);
                }
            }
        }

        let (acc, saw_nan) = if self.len() >= PARALLEL_THRESHOLD {
            self.data
                .view()
                .into_par_iter()
                .map(|&x| (x, x.is_nan()))
                .reduce(
                    || (T::neg_infinity(), false),
                    |(a, a_nan), (b, b_nan)| (if b > a { b } else { a }, a_nan | b_nan),
                )
        } else {
            self.data
                .iter()
                .copied()
                .fold((T::neg_infinity(), false), |(acc, saw_nan), x| {
                    (if x > acc { x } else { acc }, saw_nan | x.is_nan())
                })
        };
        Some(if saw_nan { T::nan() } else { acc })
    }
}

// Optimized matrix operations.
//
// Split out of the block holding `dot_optimized` below purely so
// `matmul_to` can carry the extra `+ 'static` bound that
// `kernels::gemm::gemm_2d` needs to `TypeId`-dispatch `T` onto its f64/f32
// SIMD tiers -- mirroring the same split `array/linalg.rs` makes between
// `matmul`/`matmul_2d` and `dot`. `dot_optimized` needs no dispatch and
// keeps its historical bounds unchanged.
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + 'static,
{
    /// Memory-optimized 2D matrix multiplication with pre-allocated output
    ///
    /// This version writes to a pre-allocated output buffer instead of
    /// creating a new array, reducing memory allocations.
    ///
    /// # Accumulate, not overwrite
    ///
    /// `output` is **accumulated onto**: this computes
    /// `output += self * other`, leaving whatever `output` already held
    /// in place and adding the product to it. It does *not* overwrite.
    /// Callers wanting a plain product must pass a zeroed `output` (as
    /// [`Array::zeros`] gives). This is the contract the original
    /// hand-blocked loop implemented (`*c_ij = c_ij.clone() + a_ik * b_kj`)
    /// and is preserved exactly, which is *also* why the product cannot be
    /// written straight into `output`'s buffer: `kernels::gemm::gemm_2d`
    /// has *overwrite* (`beta = 0`) semantics, so the product is formed in
    /// a temporary and then added in.
    ///
    /// That temporary is an `m * n` allocation this method did not
    /// previously make -- a real (if asymptotically free, against the
    /// `O(m*n*k)` multiply-adds it buys) concession by a module whose
    /// stated purpose is minimizing allocations.
    pub fn matmul_to(&self, other: &Self, output: &mut Self) -> Result<()> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // Validate dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "matmul requires 2D arrays".to_string(),
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![a_shape[0], b_shape[1]],
                actual: vec![a_shape[0], a_shape[1]],
            });
        }

        let expected_shape = vec![a_shape[0], b_shape[1]];
        if output.shape() != expected_shape {
            return Err(NumRs2Error::ShapeMismatch {
                expected: expected_shape,
                actual: output.shape(),
            });
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        // Form the product through the crate's dtype-dispatched GEMM
        // kernel -- f64/f32 onto `scirs2-core`'s blocked SIMD matmul
        // (row-split across threads past `kernels::GEMM_PARALLEL_MIN_FLOPS`),
        // everything else onto the same `BLOCK_SIZE = 64` blocked i-k-j
        // triple loop that used to be inlined here, but reading flat
        // slices instead of re-deriving an `IxDyn` index (and walking
        // strides) for every one of the `m*n*k` multiply-adds.
        let a = kernels::borrow::operand(self);
        let b = kernels::borrow::operand(other);
        let mut product = vec![T::zero(); m * n];
        kernels::gemm::gemm_2d(m, k, n, &a, &b, &mut product);

        // Accumulate onto `output` rather than overwriting it (see this
        // method's doc comment). `ndarray`'s `iter_mut()` walks in logical
        // row-major order for any layout, which is the order `product` is
        // laid out in, so this stays correct for a non-contiguous
        // `output` too. `mem::replace` takes the running value out by
        // move, so no `T: Copy` is needed and `output`'s contents are
        // never cloned.
        for (dst, val) in output.nd_mut().iter_mut().zip(product) {
            let acc = std::mem::replace(dst, T::zero());
            *dst = acc + val;
        }

        Ok(())
    }
}

// `dot_optimized` deliberately stays in its own impl block, on the
// *original* bounds: it needs no dtype dispatch, so there is no reason for
// it to demand `T: 'static` alongside `matmul_to` above.
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    /// Optimized dot product without to_vec()
    pub fn dot_optimized(&self, other: &Self) -> Result<T> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.len() != 1 || b_shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "dot product requires 1D arrays".to_string(),
            ));
        }

        if a_shape[0] != b_shape[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: a_shape,
                actual: b_shape,
            });
        }

        // Direct iteration without allocation
        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (a, b)| acc + a.clone() * b.clone());

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sum_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let sum = arr.sum_optimized();
        assert_abs_diff_eq!(sum, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let mean = arr.mean_optimized();
        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_optimized() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(data);

        let var = arr.variance_optimized();
        assert_abs_diff_eq!(var, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_map_inplace() {
        let data = vec![1.0, 2.0, 3.0];
        let mut arr = Array::from_vec(data);

        arr.map_inplace(|x| x * 2.0);

        let result = arr.to_vec();
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 6.0, epsilon = 1e-10);
    }

    /// The `f64` fast tier (`len >= kernels::SIMD_MIN_LEN`) must return
    /// the same *population* variance (denominator `n`) the scalar path
    /// below 64 elements returns -- **not** the `n - 1` sample variance
    /// `SimdUnifiedOps::simd_variance` would give.
    ///
    /// `test_variance_optimized` above uses 5 elements and so never
    /// reaches this tier; without this test nothing pins which estimator
    /// the dispatched path computes, and "fixing" it back to `n - 1`
    /// would look harmless.
    #[test]
    fn variance_optimized_f64_simd_tier_is_population_not_sample() {
        // 0..128: population variance of 0..N-1 is (N^2 - 1) / 12.
        let n = 128usize;
        let arr = Array::from_vec((0..n).map(|i| i as f64).collect::<Vec<_>>());
        assert!(
            arr.len() >= kernels::SIMD_MIN_LEN,
            "test must reach the dispatched f64 tier"
        );

        let population = ((n * n - 1) as f64) / 12.0;
        let sample = population * (n as f64) / ((n - 1) as f64);
        assert_abs_diff_eq!(arr.variance_optimized(), population, epsilon = 1e-9);
        assert!(
            (arr.variance_optimized() - sample).abs() > 1.0,
            "must not be the n-1 sample variance"
        );

        // std is the square root of exactly that, with no second dispatch.
        assert_abs_diff_eq!(arr.std_optimized(), population.sqrt(), epsilon = 1e-9);
    }

    /// Same estimator continuity, checked across the dispatch boundary
    /// itself: a length-63 array (scalar path) and a length-64 array
    /// (`f64` fast tier) must agree with the naive population formula.
    #[test]
    fn variance_optimized_agrees_across_simd_min_len_boundary() {
        for n in [kernels::SIMD_MIN_LEN - 1, kernels::SIMD_MIN_LEN] {
            let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.75 - 4.0).collect();
            let mean = data.iter().sum::<f64>() / n as f64;
            let expected = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            let arr = Array::from_vec(data);
            assert_abs_diff_eq!(arr.variance_optimized(), expected, epsilon = 1e-9);
        }
    }

    /// `sum_optimized`'s dispatched tier must agree with plain iteration.
    /// Sized past `kernels::PARALLEL_MIN_LEN` so the chunked parallel
    /// branch of `kernels::reduce::sum_f64` is the one under test.
    #[test]
    fn sum_optimized_f64_parallel_tier_matches_naive() {
        let data: Vec<f64> = (0..20_000).map(|i| (i as f64) * 0.125 - 7.0).collect();
        let naive: f64 = data.iter().sum();
        let arr = Array::from_vec(data);
        let got = arr.sum_optimized();
        assert!(
            (got - naive).abs() / naive.abs().max(1.0) < 1e-9,
            "got {got}, naive {naive}"
        );
    }

    /// A non-contiguous array must fall through to the stride-correct
    /// scalar path rather than reinterpreting raw memory -- the exact
    /// failure the old raw-pointer cast from a borrowed view to an
    /// `ArrayView` of element type `f64` and dimension `Ix1` produced (it
    /// reinterpreted an `IxDyn` dimension as if it were `Ix1`).
    #[test]
    fn reductions_are_correct_for_non_contiguous_f64_array() {
        // 8x9 -> transposed 9x8 view: 72 elements, and not standard
        // layout, so `as_slice()` is `None` and the generic tail runs.
        let base = Array::from_vec((0..72).map(|i| i as f64).collect::<Vec<_>>()).reshape(&[8, 9]);
        let t = base.transpose_axis(0, 1);
        assert!(!t.is_c_contiguous());
        assert!(t.len() >= kernels::SIMD_MIN_LEN);

        assert_abs_diff_eq!(
            t.sum_optimized(),
            (0..72).sum::<usize>() as f64,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(t.min_optimized().expect("non-empty"), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(t.max_optimized().expect("non-empty"), 71.0, epsilon = 1e-9);
    }

    /// `min_optimized`/`max_optimized` propagate `NaN` (NumPy's `np.min`/
    /// `np.max` rule), on every path: the `f64` and `f32` dispatch tiers,
    /// and the generic tail a non-contiguous array falls through to.
    #[test]
    fn min_max_optimized_propagate_nan_like_numpy() {
        // f64 dispatch tier, NaN in each of first/interior/last position.
        for pos in [0usize, 37, 99] {
            let mut data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.5).collect();
            data[pos] = f64::NAN;
            let arr = Array::from_vec(data);
            assert!(
                arr.min_optimized().expect("non-empty").is_nan(),
                "NaN at index {pos}"
            );
            assert!(
                arr.max_optimized().expect("non-empty").is_nan(),
                "NaN at index {pos}"
            );
        }

        // Short array: previously below the `SIMD_MIN_LEN` gate, so it
        // took the generic fold and a *different* NaN rule than a long
        // one. Same rule now.
        let short = Array::from_vec(vec![1.0f64, 2.0, f64::NAN]);
        assert!(short.min_optimized().expect("non-empty").is_nan());
        assert!(short.max_optimized().expect("non-empty").is_nan());

        // f32 dispatch tier.
        let f32_arr = Array::from_vec(vec![1.0f32, f32::NAN, 5.0]);
        assert!(f32_arr.min_optimized().expect("non-empty").is_nan());
        assert!(f32_arr.max_optimized().expect("non-empty").is_nan());

        // Generic tail: non-contiguous, so `as_slice()` is `None`.
        let base = Array::from_vec(vec![1.0f64, 2.0, f64::NAN, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let t = base.transpose_axis(0, 1);
        assert!(t.as_slice().is_none());
        assert!(t.min_optimized().expect("non-empty").is_nan());
        assert!(t.max_optimized().expect("non-empty").is_nan());
    }

    /// `min_optimized`/`max_optimized` on a contiguous array go through
    /// `kernels::reduce`; check they still find the real extremes (the old
    /// raw-pointer cast returned garbage here).
    #[test]
    fn min_max_optimized_f64_simd_tier_finds_true_extremes() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.5).collect();
        data[37] = -12345.0;
        data[81] = 98765.0;
        let arr = Array::from_vec(data);
        assert_abs_diff_eq!(
            arr.min_optimized().expect("non-empty"),
            -12345.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            arr.max_optimized().expect("non-empty"),
            98765.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_matmul_to() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
        let mut c = Array::zeros(&[2, 2]);

        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");

        // Expected result: [[19, 22], [43, 50]]
        let result = c.to_vec();
        assert_abs_diff_eq!(result[0], 19.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 43.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10);
    }

    /// `matmul_to` **accumulates** onto `output`; it does not overwrite.
    ///
    /// `test_matmul_to` above passes `Array::zeros`, under which accumulate
    /// and overwrite are indistinguishable -- so the contract that
    /// actually distinguishes them was untested, and the migration onto
    /// `kernels::gemm::gemm_2d` (which has *overwrite* semantics) is
    /// exactly the change that would silently break it.
    #[test]
    fn matmul_to_accumulates_onto_nonzero_output() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
        // Product alone is [[19, 22], [43, 50]].
        let mut c = Array::from_vec(vec![100.0, 200.0, 300.0, 400.0]).reshape(&[2, 2]);

        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");

        let result = c.to_vec();
        assert_abs_diff_eq!(result[0], 119.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 222.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 343.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 450.0, epsilon = 1e-10);

        // Calling twice accumulates twice.
        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");
        let twice = c.to_vec();
        assert_abs_diff_eq!(twice[0], 138.0, epsilon = 1e-10);
        assert_abs_diff_eq!(twice[3], 500.0, epsilon = 1e-10);
    }

    /// The generic (non-f64/f32) tier of `matmul_to` must accumulate the
    /// same way the SIMD tier does -- `gemm_generic` zeroes its own output
    /// buffer, so the accumulate has to live in `matmul_to` itself, not be
    /// inherited from whatever the kernel left behind.
    #[test]
    fn matmul_to_accumulates_on_the_generic_dtype_tier() {
        let a = Array::from_vec(vec![1i64, 2, 3, 4]).reshape(&[2, 2]);
        let b = Array::from_vec(vec![5i64, 6, 7, 8]).reshape(&[2, 2]);
        let mut c = Array::from_vec(vec![10i64, 20, 30, 40]).reshape(&[2, 2]);

        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");

        assert_eq!(c.to_vec(), vec![29, 42, 73, 90]);
    }

    /// Rectangular, larger-than-one-block shapes with a non-contiguous
    /// left operand: checks the flat-slice dispatch agrees with a naive
    /// triple loop where the old `IxDyn` `get`/`get_mut` walk did.
    #[test]
    fn matmul_to_matches_naive_for_rectangular_and_transposed_operands() {
        let (m, k, n) = (5usize, 7usize, 3usize);
        // Build A as its transpose, then flip it, so `self` is non-contiguous.
        let a_t = Array::from_vec(
            (0..k * m)
                .map(|i| i as f64 * 0.25 - 1.0)
                .collect::<Vec<_>>(),
        )
        .reshape(&[k, m]);
        let a = a_t.transpose_axis(0, 1);
        assert_eq!(a.shape(), vec![m, k]);
        assert!(!a.is_c_contiguous());

        let b = Array::from_vec(
            (0..k * n)
                .map(|i| i as f64 * -0.5 + 2.0)
                .collect::<Vec<_>>(),
        )
        .reshape(&[k, n]);

        let a_vec = a.to_vec();
        let b_vec = b.to_vec();
        let mut expected = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    expected[i * n + j] += a_vec[i * k + p] * b_vec[p * n + j];
                }
            }
        }

        let mut c = Array::zeros(&[m, n]);
        a.matmul_to(&b, &mut c).expect("matmul_to should succeed");
        for (idx, (got, want)) in c.to_vec().iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "idx={idx}: got {got}, expected {want}"
            );
        }
    }
}

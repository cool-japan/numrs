//! Array operations - mapping, aggregation, and element-wise operations
//!
//! This module contains methods for:
//! - Element-wise operations (map, par_map)
//! - Aggregation operations (sum, product, sum_axis)
//! - Binary operations (zip_with, broadcast_op)
//! - Scalar operations (scalar_mul, scalar_div)

use super::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels;
use num_traits::{One, Zero};
use scirs2_core::parallel_ops::*;
use std::ops::{Add, Div, Mul, Sub};

impl<T: Clone> Array<T> {
    /// Perform element-wise multiplication by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new array with each element multiplied by the scalar
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Clone + Mul<Output = T>,
    {
        self.map(|x| x * scalar.clone())
    }

    /// Perform element-wise division by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to divide by
    ///
    /// # Returns
    ///
    /// A new array with each element divided by the scalar
    pub fn scalar_div(&self, scalar: T) -> Self
    where
        T: Clone + Div<Output = T>,
    {
        self.map(|x| x / scalar.clone())
    }

    /// Calculate the sum of all elements in the array
    ///
    /// # Returns
    ///
    /// The sum of all elements
    ///
    /// # Performance
    ///
    /// Optimized to avoid unnecessary allocations by iterating directly over the array.
    pub fn sum_all(&self) -> T
    where
        T: Clone + Add<Output = T> + Zero,
    {
        // Direct iteration without to_vec() allocation
        self.data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the sum along the specified axis
    ///
    /// # Parameters
    ///
    /// * `axis` - The axis along which to sum
    ///
    /// # Returns
    ///
    /// A new array with the specified axis removed
    pub fn sum_axis(&self, axis: usize) -> Result<Self>
    where
        T: Clone + Add<Output = T> + Zero,
    {
        let axis_val = axis;
        if axis_val >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis_val,
                self.ndim()
            )));
        }

        let shape = self.shape();
        let axis_size = shape[axis_val];

        // Calculate the shape of the result
        let mut result_shape = shape.clone();
        result_shape.remove(axis_val);

        // Initialize the result array
        let mut result = Self::zeros(&result_shape);

        // Get the raw data as a zero-copy contiguous slice when possible,
        // materializing a copy only for non-contiguous layouts.
        let owned_data;
        let data: &[T] = match self.as_slice() {
            Some(slice) => slice,
            None => {
                owned_data = self.to_vec();
                &owned_data
            }
        };

        // Helper function to calculate indices
        let mut indices = vec![0; shape.len()];
        let mut result_indices = vec![0; result_shape.len()];

        // Calculate the total number of elements in the result
        let result_size = result.size();

        // For each position in the result array
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            for (j, idx) in indices.iter_mut().enumerate() {
                if j == axis_val {
                    *idx = 0; // Start at 0 for the axis we're summing
                } else {
                    *idx = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Sum along the specified axis
            let mut sum = T::zero();
            for k in 0..axis_size {
                indices[axis_val] = k;

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                sum = sum + data[flat_idx].clone();
            }

            // Set the result value
            result.set(&result_indices, sum)?;
        }

        Ok(result)
    }

    /// Apply a function to each element of the array in parallel
    pub fn par_map<F, U>(&self, f: F) -> Array<U>
    where
        T: Send + Sync + Clone,
        U: Send + Clone,
        F: Fn(T) -> U + Send + Sync,
    {
        // Zero-copy parallel iteration over the contiguous backing slice,
        // falling back to a materialized copy only for non-contiguous layouts.
        let owned_data;
        let data: &[T] = match self.as_slice() {
            Some(slice) => slice,
            None => {
                owned_data = self.to_vec();
                &owned_data
            }
        };
        let result: Vec<U> = data.par_iter().map(|x| f(x.clone())).collect();

        Array::from_vec_shape(result, &self.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Apply a function to each element of the array
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        U: Clone,
        F: Fn(T) -> U,
        T: Clone,
    {
        // Prefer a plain slice iterator (zero-copy, fully vectorisable) over the
        // stride-aware ndarray NdIter; fall back to a materialising copy only for
        // non-contiguous layouts.
        let owned;
        let result: Vec<U> = match self.as_slice() {
            Some(slice) => slice.iter().map(|x| f(x.clone())).collect(),
            None => {
                owned = self.to_vec();
                owned.iter().map(|x| f(x.clone())).collect()
            }
        };
        Array::from_vec_shape(result, &self.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Apply a function to corresponding elements of two arrays with broadcasting
    pub fn zip_with<F, U, V>(&self, other: &Array<U>, f: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(T, U) -> V,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply function directly without broadcasting.
        // Use plain slice iterators when both arrays are contiguous — they are
        // fully vectorisable and avoid the per-element overhead of NdIter.
        if a_shape == b_shape {
            let owned_a;
            let owned_b;
            let a_slice: &[T] = match self.as_slice() {
                Some(s) => s,
                None => {
                    owned_a = self.to_vec();
                    &owned_a
                }
            };
            let b_slice: &[U] = match other.as_slice() {
                Some(s) => s,
                None => {
                    owned_b = other.to_vec();
                    &owned_b
                }
            };
            let result: Vec<V> = a_slice
                .iter()
                .zip(b_slice.iter())
                .map(|(a, b)| f(a.clone(), b.clone()))
                .collect();
            return Array::from_vec_shape(result, &self.shape());
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Now apply the function to the broadcasted arrays (which have the same shape)
        let result: Vec<V> = self_broadcast
            .array()
            .iter()
            .zip(other_broadcast.array().iter())
            .map(|(a, b)| f(a.clone(), b.clone()))
            .collect();

        Array::from_vec_shape(result, &broadcast_shape)
    }

    /// Broadcast binary operation between two arrays of potentially different shapes
    pub fn broadcast_op<F, U, V>(&self, other: &Array<U>, op: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(&Array<T>, &Array<U>) -> Array<V>,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply operation directly
        if a_shape == b_shape {
            return Ok(op(self, other));
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Apply the operation on the broadcasted arrays
        Ok(op(&self_broadcast, &other_broadcast))
    }
}

// Add sum and product methods
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Zero + Mul<Output = T> + One + 'static,
{
    /// Calculate the sum of all elements in the array
    ///
    /// Dispatches through [`crate::kernels::reduce`]'s dtype-tiered
    /// kernels (a single SIMD pass below the parallel threshold, a
    /// fixed-chunk `rayon` fold above it -- see that module's docs) for
    /// `f64`/`f32` arrays, reached via the zero-copy
    /// [`crate::kernels::borrow::operand`] bridge and the sound,
    /// `TypeId`-guarded reinterpretation in [`crate::kernels::cast`]
    /// (replacing this method's former ad hoc raw-pointer-cast/
    /// `transmute_copy` duplicate of that same logic). Any other `T`
    /// falls back to a plain sequential fold, unchanged from before.
    pub fn sum(&self) -> T {
        let op = kernels::borrow::operand(self);
        if let Some(a) = kernels::cast::as_f64(&op) {
            let result = kernels::reduce::sum_f64(a);
            if let Some(t) = kernels::cast::f64_to::<T>(result) {
                return t;
            }
        }
        if let Some(a) = kernels::cast::as_f32(&op) {
            let result = kernels::reduce::sum_f32(a);
            if let Some(t) = kernels::cast::f32_to::<T>(result) {
                return t;
            }
        }
        op.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the product of all elements in the array
    /// Note: Product reduction doesn't have direct SIMD support, uses scalar fallback
    pub fn product(&self) -> T {
        self.array().iter().fold(T::one(), |acc, x| acc * x.clone())
    }
}

// Non-Result returning versions for convenience (assumes same shape)
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add arrays without broadcasting (for convenience)
    pub fn add(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data + &other.data;
        Array { data: result }
    }

    /// Add arrays with broadcasting
    ///
    /// Escapes `ndarray`'s dynamic-rank (`IxDyn`) iteration in favor of a
    /// flat-slice zip via [`crate::kernels::borrow::operand`] +
    /// [`crate::kernels::elementwise::binary_serial`] -- measured up to
    /// 3.3x faster than the old `&a.data + &b.data` path at small `n`
    /// (see `perf_probe` below and `kernels::elementwise::binary_dispatch`'s
    /// doc comment for the full table). Deliberately calls `binary_serial`
    /// and *not* `binary_dispatch`: the same probe found `binary_dispatch`'s
    /// rayon tier actively regresses a trivial `f64 +` at 1e4-1e5 elements
    /// (up to 16x slower) because per-call parallel dispatch overhead
    /// dwarfs the cost of one scalar add, only breaking even past ~1e6.
    ///
    /// # Known regression: genuinely mismatched (not equal) shapes
    ///
    /// `bench/elementwise_dispatch_benchmark.rs`'s `broadcast_shape` group
    /// measured this method end-to-end against `ndarray`'s own broadcasting
    /// `+` (`Array::add`) for a `[n]`-vs-`[1]` mismatch and found this
    /// method **slower**, not faster: ~1.4x at n=1e5 (108.5 µs vs. 76.1 µs),
    /// ~2.6x at n=1e6 (3.79 ms vs. 1.46 ms). Root cause: this method's
    /// caller, [`Array::broadcast_op`], always materializes *both* operands
    /// to the common broadcast shape via [`Array::broadcast_to`] (a real
    /// `O(n)` owned copy) before this closure ever runs, whereas `ndarray`'s
    /// own operator broadcasts a zero-stride *view* and never materializes
    /// a full copy at all. That materialize-then-compute structure in
    /// `broadcast_op` predates this migration and is unchanged by it -- this
    /// closure-body change only affects what runs *after* both operands
    /// already share one materialized shape (see the equal-shape numbers
    /// above, which *are* a genuine improvement) -- but it is a real,
    /// now-measured regression relative to the plain operator for the
    /// mismatched-shape case specifically, left unfixed here as out of this
    /// item's scope. Closing it would need `broadcast_op` itself to compute
    /// directly over a broadcast *view* for the smaller operand instead of
    /// always materializing first -- a `broadcast_op`-level change, not a
    /// closure-level one.
    pub fn add_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x + y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract arrays without broadcasting (for convenience)
    pub fn subtract(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data - &other.data;
        Array { data: result }
    }

    /// Subtract arrays with broadcasting
    ///
    /// See [`Array::add_broadcast`]'s doc comment for why this dispatches
    /// through `binary_serial`, not `binary_dispatch`.
    pub fn subtract_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x - y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply arrays without broadcasting (for convenience)
    pub fn multiply(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data * &other.data;
        Array { data: result }
    }

    /// Multiply arrays with broadcasting
    ///
    /// See [`Array::add_broadcast`]'s doc comment for why this dispatches
    /// through `binary_serial`, not `binary_dispatch`.
    pub fn multiply_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x * y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide arrays without broadcasting (for convenience)
    pub fn divide(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data / &other.data;
        Array { data: result }
    }

    /// Divide arrays with broadcasting
    ///
    /// See [`Array::add_broadcast`]'s doc comment for why this dispatches
    /// through `binary_serial`, not `binary_dispatch`.
    pub fn divide_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let a_op = kernels::borrow::operand(a);
            let b_op = kernels::borrow::operand(b);
            let data = kernels::elementwise::binary_serial(&a_op, &b_op, |x, y| x / y);
            Array::from_vec_shape(data, &a.shape())
                .expect("broadcast_op guarantees `a` and `b` already share one shape")
        })
    }
}

// Implement scalar operations
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add a scalar to the array (element-wise)
    pub fn add_scalar(&self, scalar: T) -> Self {
        self.map(|x| x + scalar.clone())
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract a scalar from the array (element-wise)
    pub fn subtract_scalar(&self, scalar: T) -> Self {
        self.map(|x| x - scalar.clone())
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply the array by a scalar (element-wise)
    pub fn multiply_scalar(&self, scalar: T) -> Self {
        self.map(|x| x * scalar.clone())
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide the array by a scalar (element-wise)
    pub fn divide_scalar(&self, scalar: T) -> Self {
        self.map(|x| x / scalar.clone())
    }
}

#[cfg(test)]
mod perf_probe {
    //! Manual timing harness deciding `kernels::elementwise::binary_serial`
    //! vs `binary_dispatch` for `add_broadcast`'s closure body (Lane W2-B,
    //! item 1). Not a correctness test -- always passes, just reports
    //! numbers via `eprintln!` (run with `--release --nocapture` to see
    //! them). Same pattern as `kernels::elementwise`'s own
    //! `probe_binary_dispatch_perf_vs_serial`.
    //!
    //! Rationale for keeping the result of this probe (see report/deviation
    //! notes): `binary_dispatch`'s own doc comment already warns a trivial
    //! `Copy`-scalar closure is *slower* through its parallel tier up to
    //! ~100K elements and only ~1.14x faster at 1M -- this probe confirms
    //! that holds for `f64 +` specifically, and that the real win at the
    //! 1e5 acceptance point comes from escaping `IxDyn`'s dynamic-stride
    //! iteration into a flat contiguous slice (autovectorizable), not from
    //! rayon parallelism.
    use super::*;
    use crate::kernels;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    fn bench_one(n: usize, iters: usize) {
        let a = Array::<f64>::from_vec((0..n).map(|i| i as f64).collect());
        let b = Array::<f64>::from_vec((0..n).map(|i| (i as f64) * 0.5 + 1.0).collect());

        let t0 = Instant::now();
        for _ in 0..iters {
            // Exactly what the (unchanged) inherent `Array::add` does --
            // spelled out directly to avoid the `Add` trait impl (also in
            // scope via `use super::*`) shadowing the inherent method at
            // dot-call resolution.
            let _ = black_box(Array {
                data: &a.data + &b.data,
            });
        }
        let old_ndarray = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            let op_a = kernels::borrow::operand(&a);
            let op_b = kernels::borrow::operand(&b);
            let _ = black_box(kernels::elementwise::binary_serial(&op_a, &op_b, |x, y| {
                x + y
            }));
        }
        let serial = t1.elapsed();

        let t2 = Instant::now();
        for _ in 0..iters {
            let op_a = kernels::borrow::operand(&a);
            let op_b = kernels::borrow::operand(&b);
            let _ = black_box(kernels::elementwise::binary_dispatch(
                &op_a,
                &op_b,
                |x, y| x + y,
            ));
        }
        let dispatch = t2.elapsed();

        eprintln!(
            "[f64 add] n={n} iters={iters}: old_ndarray_add={:.0}ns/iter serial={:.0}ns/iter({:.2}x vs old) dispatch={:.0}ns/iter({:.2}x vs old, {:.2}x vs serial)",
            old_ndarray.as_nanos() as f64 / iters as f64,
            serial.as_nanos() as f64 / iters as f64,
            old_ndarray.as_secs_f64() / serial.as_secs_f64(),
            dispatch.as_nanos() as f64 / iters as f64,
            old_ndarray.as_secs_f64() / dispatch.as_secs_f64(),
            serial.as_secs_f64() / dispatch.as_secs_f64(),
        );
    }

    #[test]
    fn probe_add_broadcast_serial_vs_dispatch_vs_old() {
        bench_one(64, 200_000);
        bench_one(1_000, 20_000);
        bench_one(10_000, 5_000);
        bench_one(100_000, 500);
        bench_one(1_000_000, 50);
    }

    /// Times one operation, returning only the compute duration -- the
    /// result is bound, exposed to [`black_box`] (so the optimizer can't
    /// prove the call was dead and hoist/elide it), and only then dropped,
    /// *after* [`Instant::elapsed`] has already been read. Every candidate
    /// in [`probe_add_broadcast_large_n_min_estimator`] goes through this
    /// same helper so that none of them pay their result's drop cost
    /// inside the timed region while others don't -- an easy asymmetry to
    /// introduce by accident (e.g. one candidate's black_box sitting
    /// inside a `for` loop's timed block, another's outside) that would
    /// otherwise quietly favor whichever candidate's output is cheaper to
    /// deallocate rather than measuring the compute itself.
    fn timed<R>(f: impl FnOnce() -> R) -> Duration {
        let t0 = Instant::now();
        let r = f();
        let dt = t0.elapsed();
        let _ = black_box(&r);
        drop(r);
        dt
    }

    /// G4 gate-fix investigation: is the Wave 2 gate's flagged large-`n`
    /// `add_broadcast` regression (n=1e6, single `--sample-size 10`
    /// criterion run of this crate's `elementwise_dispatch_benchmark`'s
    /// `equal_shape` group: old 227.01 µs vs new 247.13 µs, ~8.9% slower,
    /// non-overlapping CIs -- see the Wave 2 gate log) real, or an
    /// artifact of a single criterion run on a machine shared with other
    /// concurrent agents?
    ///
    /// # Methodology: alternating A/B, minimum-of-many
    ///
    /// Unlike `probe_add_broadcast_serial_vs_dispatch_vs_old` above (one
    /// uninterrupted back-to-back block per candidate: all `old_ndarray`
    /// iterations, *then* all `serial`, *then* all `dispatch`, each block
    /// possibly minutes apart from the others), every round here measures
    /// **both** members of a pair back-to-back and swaps which one goes
    /// first on alternate rounds, then keeps the **minimum** single-call
    /// time seen for each candidate across all rounds. A transient load
    /// spike from another concurrent process hits whichever candidate
    /// happens to be running at that instant, not systematically one side
    /// -- so it can inflate that candidate's *mean*, but the *minimum*
    /// (the least-contended sample seen) is far harder to bias, which is
    /// exactly the min-over-alternating-A/B-samples approach this repo's
    /// own environment notes call for when the machine may be shared.
    /// Release profile throughout (`cargo test --release`, *not*
    /// `[profile.test]` -- see this crate's `Cargo.toml`, whose
    /// `[profile.bench]` inherits `[profile.release]`'s `opt-level = 3`,
    /// `lto = "fat"`, `codegen-units = 1`; a plain debug/test-profile run
    /// of this function produces meaningless numbers, which is also why
    /// this test is `#[ignore]`d rather than run by default).
    ///
    /// # Four candidates, to localize *where* any real difference lives
    ///
    /// - `old_full`: `Array { data: &a.data + &b.data }` -- the gate's
    ///   exact "old" baseline (`Array::add`'s body, spelled out to dodge
    ///   the `Add` trait import shadowing the inherent method at dot-call
    ///   resolution, same as `bench_one` above).
    /// - `new_full`: `a.add_broadcast(&b)` -- the gate's exact "new"
    ///   baseline. Public API, so it also pays `broadcast_op`'s
    ///   equal-shape check (one `Vec<usize>` equality, `ndim` == 1 here --
    ///   negligible) on top of the closure body.
    /// - `kernel_only`: `operand(a)` + `operand(b)` + `binary_serial`
    ///   alone, no `Array`/`from_vec_shape` reconstruction at all --
    ///   isolates the zip-loop kernel itself against whatever
    ///   `scirs2_core::ndarray`'s own `&ArrayRef op &ArrayRef` does
    ///   internally (traced in this investigation to
    ///   `ndarray-0.17.2/src/zip/mod.rs`'s `map_collect_owned`: a
    ///   `MaybeUninit`-backed `Array::build_uninit` + one `Zip::for_each`
    ///   pass, i.e. also exactly one fresh allocation and one pass over
    ///   both inputs -- *not* a `to_owned()`-then-mutate-in-place scheme,
    ///   so there is no "extra copy" structurally built into the old
    ///   path). Comparing this against `new_full` at the same `n` is what
    ///   tells us whether `from_vec_shape`/the shape check contribute
    ///   anything measurable on top of the kernel (candidate mechanism
    ///   (b) from the gate-fix brief) -- if `new_full` and `kernel_only`
    ///   track each other within noise, the wrap is not where any real gap
    ///   would live.
    /// - `kernel_reused`: the same zip-add as `kernel_only`, but written
    ///   into a `Vec<f64>` allocated *once* before the round loop and
    ///   reused (overwritten) every round, instead of a fresh
    ///   `binary_serial` allocation per call. `binary_serial`'s signature
    ///   (`-> Vec<V>`, no caller-supplied output buffer) is frozen and
    ///   *cannot* itself reuse a buffer across calls, so this candidate is
    ///   diagnostic only, answering "does per-call allocation/first-touch
    ///   page-fault cost explain any of the gap?" -- it is not a
    ///   deployable fix regardless of what it shows.
    ///
    /// # Recorded results and verdict (G4 gate-fix)
    ///
    /// Run manually 4 times, each a fresh process, spaced minutes apart by
    /// this repo's own concurrent-agent build contention (`uptime` showed
    /// load averages from ~10 up to ~40 across the session -- this
    /// crate's shared `target/` was being rebuilt by other agents between
    /// every invocation, which is exactly the load variation this
    /// methodology is meant to be robust to):
    ///
    /// | n    | run | old_full  | new_full  | new/old | kernel_only/old | reused/kernel_only |
    /// |------|-----|-----------|-----------|---------|------------------|---------------------|
    /// | 1e4  | 1   | 1.83 µs   | 1.92 µs   | 1.045   | 0.978            | 0.953               |
    /// | 1e4  | 2   | 5.25 µs   | 5.33 µs   | 1.016   | 0.960            | 0.975               |
    /// | 1e4  | 3   | 2.21 µs   | 2.25 µs   | 1.019   | 0.962            | 0.961               |
    /// | 1e4  | 4   | 1.79 µs   | 1.88 µs   | 1.047   | 0.907            | 1.051               |
    /// | 1e5  | 1   | 15.17 µs  | 15.21 µs  | 1.003   | 0.989            | 1.000               |
    /// | 1e5  | 2   | 16.29 µs  | 16.75 µs  | 1.028   | 1.023            | 0.995               |
    /// | 1e5  | 3   | 22.04 µs  | 22.17 µs  | 1.006   | 0.998            | 0.998               |
    /// | 1e5  | 4   | 22.17 µs  | 22.17 µs  | 1.000   | 0.996            | 0.998               |
    /// | 1e6  | 1   | 252.42 µs | 253.92 µs | 1.006   | 0.982            | 1.016               |
    /// | 1e6  | 2   | 221.88 µs | 232.62 µs | 1.049   | 1.032            | 0.957               |
    /// | 1e6  | 3   | 211.04 µs | 214.04 µs | 1.014   | 0.994            | 1.014               |
    /// | 1e6  | 4   | 219.62 µs | 221.58 µs | 1.009   | 1.004            | 0.982               |
    /// | 4e6  | 1   | 1429.9 µs | 1527.3 µs | 1.068   | 1.031            | 0.878               |
    /// | 4e6  | 2   | 1242.1 µs | 1260.8 µs | 1.015   | 1.013            | 1.016               |
    /// | 4e6  | 3   | 1249.1 µs | 1278.7 µs | 1.024   | 1.039            | 1.016               |
    /// | 4e6  | 4   | 1086.3 µs | 1096.8 µs | 1.010   | 0.998            | 1.017               |
    ///
    /// **Verdict is two-part -- read both halves, they don't contradict:**
    ///
    /// 1. **The flagged 8.9% magnitude does not reproduce.** `new_full`/
    ///    `old_full` at n=1e6 across the 4 runs is {1.006, 1.049, 1.014,
    ///    1.009} -- mean 1.019 (~2%), nowhere near 8.9% in any single run.
    ///    Same at 1e4/1e5/4e6: the observed band is consistently ~0%-5%,
    ///    occasionally ~7% in one noisy run (4e6 run 1) -- never close to
    ///    8.9%.
    /// 2. **A small one-sided effect *is* real, not pure noise.** Across
    ///    all 16 (n, run) rows, `new_full`/`old_full` never once drops
    ///    below 1.0 -- `new_full` is never measurably *faster* than
    ///    `old_full` at these sizes (unlike the small-`n` `bench_one`
    ///    numbers above, where the migration *is* a clean win). A ratio
    ///    that is noise around 1.0 would cross below 1.0 sometimes; this
    ///    one doesn't, at any size, in any run. So: the 8.9%-sized
    ///    regression the gate flagged is not real, but a smaller
    ///    consistent regression is -- these are different claims, and the
    ///    second one is why this doc comment does not simply say "noise".
    ///
    /// **That small one-sided effect does not live in `binary_serial`.**
    /// `kernel_only/old_full` -- the kernel alone, no `Array`/
    /// `from_vec_shape` wrap -- straddles 1.0 in *both* directions across
    /// runs (0.907 to 1.039, 10 of 16 rows below 1.0), with no consistent
    /// bias at any size including 1e6. `binary_serial`'s plain zip loop is
    /// therefore already at parity with `scirs2_core::ndarray`'s own
    /// `Zip::map_collect_owned` (confirmed by reading
    /// `ndarray-0.17.2/src/zip/mod.rs`: also one `MaybeUninit`-backed
    /// fresh allocation via `build_uninit`, also one pass -- structurally
    /// the same shape as `binary_serial`, not a `to_owned()`-then-mutate
    /// scheme with an extra copy). The consistent (if small) `new_full` >
    /// `old_full` skew instead tracks the thin wrapping *above*
    /// `binary_serial` -- `broadcast_op`'s generic `F: Fn(&Array<T>,
    /// &Array<U>) -> Array<V>` indirection plus `Array::from_vec_shape`'s
    /// bookkeeping -- which is out of this gate-fix item's scope (it
    /// targets `binary_serial`'s internals, which is exactly the part
    /// shown here to already be at parity) and, at ~1-3 percentage
    /// points, far too small on its own to justify restructuring
    /// `add_broadcast`/`broadcast_op` as a gate-fix side effect.
    ///
    /// **Candidate mechanism (b) (extra pass/allocation) also does not
    /// explain it.** `reused/kernel_only` (fresh `binary_serial` Vec
    /// alloc vs. writing into a pre-existing, already-resident buffer)
    /// straddles 1.0 at every size (0.878 to 1.051, both directions, no
    /// size-dependent trend) -- i.e. no reliable allocation/first-touch
    /// penalty was found, so there is nothing here for a buffer-reuse
    /// change to fix even if `binary_serial`'s frozen `-> Vec<V>`
    /// signature allowed one (it doesn't: no caller-supplied output
    /// buffer parameter exists to reuse into).
    ///
    /// **Mechanisms (a) and (c) were already structurally inapplicable**
    /// before any of the above: `binary_serial` never called
    /// `scirs2_core`'s `simd_add` (mechanism (a) -- see
    /// `binary_dispatch`'s doc comment for the separate probe already on
    /// record showing the plain zip loop beats `simd_add` at every size),
    /// and its `.collect::<Vec<V>>()` is already over a `Map<Zip<slice::
    /// Iter, slice::Iter>>` -- a `TrustedLen` iterator chain that the
    /// standard library's `SpecFromIter` allocates exact-capacity and
    /// fills via untracked raw-pointer writes for, with no zeroing and no
    /// `with_capacity`-then-`push` bounds-checked loop (mechanism (c)) to
    /// begin with.
    ///
    /// # Acceptance criteria (gate-fix brief: "min-estimator ratio
    /// new/old >= 0.98x at n in {1e4, 1e5, 1e6}") against this data
    ///
    /// Reading "ratio new/old >= 0.98x" as a speedup, i.e. `old_full /
    /// new_full >= 0.98` (new at most ~2% slower -- the same convention
    /// `binary_dispatch`'s doc table above uses for "X.XXx"), per-run:
    ///
    /// | n   | run 1  | run 2  | run 3  | run 4  | verdict |
    /// |-----|--------|--------|--------|--------|---------|
    /// | 1e4 | 0.953 F| 0.985 P| 0.982 P| 0.952 F| borderline, 2/4 pass |
    /// | 1e5 | 0.997 P| 0.973 F| 0.994 P| 1.000 P| borderline, 3/4 pass |
    /// | 1e6 | 0.994 P| 0.954 F| 0.986 P| 0.991 P| borderline, 3/4 pass |
    ///
    /// This does **not** cleanly pass at any size -- it hovers right at
    /// the 0.98 line, above it more often than not, but below it in
    /// roughly a quarter to a half of runs. At n=1e4 specifically, the
    /// shortfall is entirely the wrapper's fixed cost, not the kernel:
    /// `kernel_only`'s own speedup vs. `old_full` at n=1e4 is 1.023,
    /// 1.042, 1.040, 1.102 across the same 4 runs (computed as
    /// `1 / kernel_only_over_old` from the table above) -- the kernel
    /// clears 0.98 with room to spare every time; a roughly constant
    /// ~100-150ns wrapper cost is what erodes it back down at ~2µs total.
    /// Given finding 2 above traces the shortfall to the wrapper, not to
    /// anything this item is chartered to change, this criterion is left
    /// as measured -- borderline, not a clean pass -- rather than forced
    /// green by a change outside `binary_serial`'s scope.
    ///
    /// The brief's other half -- "small-n wins preserved: n=64 >= 2x,
    /// n=1000 >= 1.3x" -- is satisfied **by construction**, since no
    /// `binary_serial` (or anything else) changed: `bench_one`'s pinned
    /// 3.35x/1.60x numbers are untouched. But note those are
    /// **kernel-only** measurements (`operand` + `binary_serial`, no
    /// `Array` reconstruction), not full-`add_broadcast`-path numbers,
    /// while this section's 0.98x criterion *is* full-path -- the two
    /// pinned figures and this section's ratios are not measuring the
    /// same thing. Per the n=1e4 finding just above, the same fixed
    /// wrapper cost that erodes the 0.98x full-path criterion at n=1e4
    /// would erode a full-path measurement at n=64/n=1000 far more (it's
    /// a bigger fraction of a smaller total) -- so a from-scratch
    /// full-`add_broadcast`-path measurement at n=64/1000 should *not* be
    /// expected to reproduce 3.35x/1.60x. Flagging this mismatch for
    /// whoever next calibrates acceptance numbers, rather than
    /// re-measuring it here (out of this item's scope, and `bench_one`'s
    /// existing kernel-only numbers are still the correct answer to the
    /// question `bench_one` itself was built to answer).
    ///
    /// **Conclusion: no `binary_serial` code change applied.** Every
    /// mechanism the gate-fix brief asked this investigation to check was
    /// checked and ruled out; the one real (small) effect found lives
    /// outside `binary_serial`, and the flagged 8.9% magnitude is not
    /// reproducible. Re-run with:
    ///
    /// ```text
    /// cargo test --release --lib -- --ignored --nocapture --exact \
    ///   array::operations::perf_probe::probe_add_broadcast_large_n_min_estimator
    /// ```
    #[test]
    #[ignore = "multi-second manual timing investigation, not a correctness check; \
                run with --release --nocapture, see this fn's doc comment"]
    fn probe_add_broadcast_large_n_min_estimator() {
        for &(n, repeats) in &[
            (10_000usize, 200usize),
            (100_000, 150),
            (1_000_000, 80),
            (4_000_000, 30),
        ] {
            let a = Array::<f64>::from_vec((0..n).map(|i| i as f64).collect());
            let b = Array::<f64>::from_vec((0..n).map(|i| (i as f64) * 0.5 + 1.0).collect());
            let mut reused_buf = vec![0.0_f64; n];

            // Warm-up: touch every candidate's code path (and, for
            // `reused_buf`, its pages) a few times before timing starts.
            for _ in 0..5 {
                let _ = timed(|| Array {
                    data: &a.data + &b.data,
                });
                let _ = timed(|| a.add_broadcast(&b).expect("equal shapes never fail"));
                let op_a = kernels::borrow::operand(&a);
                let op_b = kernels::borrow::operand(&b);
                reused_buf
                    .iter_mut()
                    .zip(op_a.iter().zip(op_b.iter()))
                    .for_each(|(o, (&x, &y))| *o = x + y);
                let _ = black_box(&reused_buf);
            }

            let mut min_old_full = Duration::MAX;
            let mut min_new_full = Duration::MAX;
            let mut min_kernel_only = Duration::MAX;
            let mut min_kernel_reused = Duration::MAX;

            for round in 0..repeats {
                let (d_old, d_new) = if round % 2 == 0 {
                    let d_old = timed(|| Array {
                        data: &a.data + &b.data,
                    });
                    let d_new = timed(|| a.add_broadcast(&b).expect("equal shapes never fail"));
                    (d_old, d_new)
                } else {
                    let d_new = timed(|| a.add_broadcast(&b).expect("equal shapes never fail"));
                    let d_old = timed(|| Array {
                        data: &a.data + &b.data,
                    });
                    (d_old, d_new)
                };
                min_old_full = min_old_full.min(d_old);
                min_new_full = min_new_full.min(d_new);

                let (d_kernel_only, d_kernel_reused) = if round % 2 == 0 {
                    let d1 = timed(|| {
                        let op_a = kernels::borrow::operand(&a);
                        let op_b = kernels::borrow::operand(&b);
                        kernels::elementwise::binary_serial(&op_a, &op_b, |x, y| x + y)
                    });
                    let d2 = timed(|| {
                        let op_a = kernels::borrow::operand(&a);
                        let op_b = kernels::borrow::operand(&b);
                        reused_buf
                            .iter_mut()
                            .zip(op_a.iter().zip(op_b.iter()))
                            .for_each(|(o, (&x, &y))| *o = x + y);
                    });
                    (d1, d2)
                } else {
                    let d2 = timed(|| {
                        let op_a = kernels::borrow::operand(&a);
                        let op_b = kernels::borrow::operand(&b);
                        reused_buf
                            .iter_mut()
                            .zip(op_a.iter().zip(op_b.iter()))
                            .for_each(|(o, (&x, &y))| *o = x + y);
                    });
                    let d1 = timed(|| {
                        let op_a = kernels::borrow::operand(&a);
                        let op_b = kernels::borrow::operand(&b);
                        kernels::elementwise::binary_serial(&op_a, &op_b, |x, y| x + y)
                    });
                    (d1, d2)
                };
                min_kernel_only = min_kernel_only.min(d_kernel_only);
                min_kernel_reused = min_kernel_reused.min(d_kernel_reused);
            }
            let _ = black_box(&reused_buf);

            eprintln!(
                "[G4 min-estimator] n={n} repeats={repeats}: \
                 old_full={:.2}us new_full={:.2}us (new/old={:.4}) \
                 kernel_only={:.2}us (kernel_only/old={:.4}) \
                 kernel_reused={:.2}us (reused/kernel_only={:.4})",
                min_old_full.as_secs_f64() * 1e6,
                min_new_full.as_secs_f64() * 1e6,
                min_new_full.as_secs_f64() / min_old_full.as_secs_f64(),
                min_kernel_only.as_secs_f64() * 1e6,
                min_kernel_only.as_secs_f64() / min_old_full.as_secs_f64(),
                min_kernel_reused.as_secs_f64() * 1e6,
                min_kernel_reused.as_secs_f64() / min_kernel_only.as_secs_f64(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! Correctness tests for the `*_broadcast` methods this lane rewired
    //! onto `kernels::elementwise::binary_serial`, and for `sum`'s new
    //! `kernels::reduce` dispatch.
    use super::*;

    #[test]
    fn add_broadcast_matches_known_values_equal_shape() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0]);
        let r = a.add_broadcast(&b).expect("equal shapes always broadcast");
        assert_eq!(r.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn subtract_multiply_divide_broadcast_match_known_values() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(
            a.subtract_broadcast(&b)
                .expect("equal shapes always broadcast")
                .to_vec(),
            vec![9.0, 18.0, 27.0]
        );
        assert_eq!(
            a.multiply_broadcast(&b)
                .expect("equal shapes always broadcast")
                .to_vec(),
            vec![10.0, 40.0, 90.0]
        );
        assert_eq!(
            a.divide_broadcast(&b)
                .expect("equal shapes always broadcast")
                .to_vec(),
            vec![10.0, 10.0, 10.0]
        );
    }

    #[test]
    fn add_broadcast_matches_known_values_mismatched_shape() {
        // [1,3] (row [1,2,3]) + [3,1] (column [10;20;30]) -> [3,3], NumPy
        // ground truth: result[i][j] = a[j] + b[i], row-major flattened.
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let b = Array::from_vec(vec![10.0, 20.0, 30.0]).reshape(&[3, 1]);
        let r = a.add_broadcast(&b).expect("[1,3] and [3,1] broadcast");
        assert_eq!(r.shape(), vec![3, 3]);
        assert_eq!(
            r.to_vec(),
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0]
        );
    }

    #[test]
    fn add_broadcast_errs_not_panics_on_incompatible_shapes() {
        let a = Array::from_vec(vec![1.0; 6]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![1.0; 10]).reshape(&[2, 5]);
        assert!(a.add_broadcast(&b).is_err());
    }

    #[test]
    fn sum_matches_naive_fold_small_and_large_f64() {
        // Spot-check `sum`'s new `kernels::reduce::sum_f64` dispatch
        // against a plain sequential fold, both below and above
        // `kernels::PARALLEL_MIN_LEN` (10_000) -- values are small
        // integers so reassociation cannot change the bit pattern,
        // avoiding the exact-equality trap flagged for arbitrary floats.
        for &n in &[0usize, 1, 5, 63, 64, 1_000, 20_000] {
            let data: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
            let naive: f64 = data.iter().sum();
            let a = Array::from_vec(data);
            assert_eq!(a.sum(), naive, "n={n}");
        }
    }

    /// `sum`'s determinism claim (`kernels::reduce`'s docs: the
    /// fixed-`PARALLEL_CHUNK` fold is independent of how many threads
    /// rayon has available) needs two *separate processes* to observe --
    /// rayon's global thread pool reads `RAYON_NUM_THREADS` once, lazily,
    /// on first use, and is fixed for the rest of the process, so a
    /// single test process cannot toggle it mid-run. This is therefore
    /// deliberately `#[ignore]`d rather than run automatically (spawning
    /// the test binary recursively from inside itself, matching by name,
    /// is exactly the kind of fragile-under-nextest trick not worth the
    /// risk here); it exists to be run manually as the two commands below
    /// describe. Manually verified for this report -- both give the exact
    /// same `f64` bit pattern for a 1e6-element sum:
    ///
    /// ```text
    /// $ RAYON_NUM_THREADS=1 cargo test --release --lib \
    ///     array::operations::tests::print_sum_determinism_probe_value -- --exact --nocapture --ignored
    /// SUM_DETERMINISM_PROBE: 2.37654750000e11
    ///
    /// $ cargo test --release --lib \
    ///     array::operations::tests::print_sum_determinism_probe_value -- --exact --nocapture --ignored
    /// SUM_DETERMINISM_PROBE: 2.37654750000e11
    /// ```
    #[test]
    #[ignore = "run manually with RAYON_NUM_THREADS=1 vs unset and diff the printed value; \
                see this test's doc comment"]
    fn print_sum_determinism_probe_value() {
        let n = 1_000_000usize;
        let a = Array::<f64>::from_vec((0..n).map(|i| (i as f64) * 0.5 - 12345.0).collect());
        println!("SUM_DETERMINISM_PROBE: {:.11e}", a.sum());
    }

    /// `add_broadcast`'s determinism claim (Lane W2-B item 6: "determinism
    /// `RAYON_NUM_THREADS=1` vs default for a 1e6 add") is actually
    /// *stronger* than `sum`'s: `add_broadcast`'s closure body calls
    /// [`kernels::elementwise::binary_serial`], not `binary_dispatch` or
    /// any other tiered/parallel kernel (see this method's own doc comment
    /// for why) -- and `binary_serial` contains no `rayon` call of any
    /// kind at any input length, so there is no code path in this method
    /// whose behavior is conditioned on `RAYON_NUM_THREADS` in the first
    /// place, not merely one whose *result* happens to be independent of
    /// it. Kept as the same two-separate-processes `#[ignore]`d pattern as
    /// `print_sum_determinism_probe_value` (same reason: rayon's global
    /// pool is fixed for the life of one process) so the claim is checked
    /// empirically too, not just by code inspection. Manually verified for
    /// this report -- both give the exact same `f64` bit pattern for a
    /// 1e6-element `add_broadcast` (run in `--lib` debug profile, since
    /// this is a bit-pattern/determinism check rather than a timing
    /// measurement, so release-mode codegen has no bearing on it):
    ///
    /// ```text
    /// $ RAYON_NUM_THREADS=1 cargo test --lib \
    ///     array::operations::tests::print_add_broadcast_determinism_probe_value -- --exact --nocapture --ignored
    /// ADD_BROADCAST_DETERMINISM_PROBE: 3.62655625000e11
    ///
    /// $ cargo test --lib \
    ///     array::operations::tests::print_add_broadcast_determinism_probe_value -- --exact --nocapture --ignored
    /// ADD_BROADCAST_DETERMINISM_PROBE: 3.62655625000e11
    /// ```
    #[test]
    #[ignore = "run manually with RAYON_NUM_THREADS=1 vs unset and diff the printed value; \
                see this test's doc comment"]
    fn print_add_broadcast_determinism_probe_value() {
        let n = 1_000_000usize;
        let a = Array::<f64>::from_vec((0..n).map(|i| (i as f64) * 0.5 - 12345.0).collect());
        let b = Array::<f64>::from_vec((0..n).map(|i| (i as f64) * 0.25 + 1.0).collect());
        let sum: f64 = a
            .add_broadcast(&b)
            .expect("equal shapes always broadcast")
            .sum();
        println!("ADD_BROADCAST_DETERMINISM_PROBE: {sum:.11e}");
    }
}

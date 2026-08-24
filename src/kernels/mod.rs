//! Crate-private, dtype-dispatched compute kernels.
//!
//! This module is the shared dispatch foundation for NumRS2's hot paths:
//! one tested, documented place to route elementwise, reduction, and GEMM
//! operations through instead of re-deriving ad hoc `TypeId` transmutes and
//! SIMD/parallel thresholds at every call site (as `stats/basic.rs`,
//! `array/operations_optimized.rs`, `ufuncs.rs`, etc. currently each do
//! independently).
//!
//! - [`cast`] -- the *only* type-punning `unsafe` allowed anywhere in this
//!   module: `TypeId`-guarded reinterpretation of generic slices/`Vec`s/
//!   scalars as their concrete `f64`/`f32` form.
//! - [`borrow`] -- zero-copy (when possible) access to an [`crate::array::Array`]'s
//!   data as a flat, logically-ordered slice.
//! - [`elementwise`] -- binary/unary zip-map kernels.
//! - [`reduce`] -- deterministic sum/mean/variance/min/max/product
//!   kernels. `min`/`max` propagate `NaN` the way NumPy does.
//! - [`gemm`] -- dtype-dispatched 2D matrix multiplication.
//!
//! # Call sites
//!
//! The hot-path migration is done: these kernels now back `stats::basic`'s
//! `Statistics` impl, `math::{aggregation, statistics}`'s `sum`/`mean`/
//! `var`/`std`/`prod`/`min`/`max`, `array::{operations,
//! operations_optimized, linalg, arithmetic}`, `comparisons`,
//! `comparisons_broadcast`, `masked`, `set_ops`, `simd`,
//! `linalg::vector_ops` and more -- see `bench/*_dispatch_benchmark.rs`
//! for the before/after measurements behind each conversion.
//!
//! `dead_code` stays allowed module-wide even so: this module is a
//! *complete*, symmetric API surface (both dtypes x every operation,
//! plus every threshold constant) rather than only the exact subset that
//! happens to have a caller today, so that the next conversion finds its
//! kernel already written, tested and documented instead of growing the
//! module ad hoc. Some of that surface is consequently exercised only by
//! this module's own unit tests (see precedent:
//! `src/parallel/load_balancer.rs`, `src/interpolate.rs`).
#![allow(dead_code)]

pub(crate) mod borrow;
pub(crate) mod cast;
pub(crate) mod elementwise;
pub(crate) mod gemm;
pub(crate) mod reduce;

/// Minimum slice length (in elements) below which wrapping a slice for
/// `scirs2-core`'s `SimdUnifiedOps` kernels is worth doing.
///
/// Matches the existing de-facto value used independently at three call
/// sites today: `math/elementwise.rs:18` (`SIMD_THRESHOLD`),
/// `ufuncs.rs:41` (`SIMD_THRESHOLD`), and `linalg/vector_ops.rs:22`
/// (`SIMD_THRESHOLD`) -- all three settled on 64 after profiling showed
/// that below it, the allocation/conversion overhead of routing through
/// an owned `ndarray` conversion exceeds the SIMD speedup.
///
/// This constant is defined here as the single canonical value those three
/// (and future) call sites should converge on; it is not branched on
/// directly by anything in `kernels/` itself, because every kernel in this
/// module already borrows its input via a zero-copy `ArrayView1::from(slice)`
/// (see [`borrow::operand`]) rather than the owned `to_vec`/`Array1::from_vec`
/// conversions that motivated the threshold at the original three sites --
/// there is no allocation left here to amortize below this size.
pub(crate) const SIMD_MIN_LEN: usize = 64;

/// Minimum slice length (in elements) above which a reduction or
/// elementwise kernel switches from a single-threaded pass to a chunked,
/// `rayon`-backed parallel pass.
///
/// Matches the existing de-facto value used independently at
/// `stats/basic.rs:19` (`PARALLEL_THRESHOLD`) and
/// `array/operations_optimized.rs:19` (`PARALLEL_THRESHOLD`): both settled
/// on 10,000 as the point past which spawning parallel work reliably pays
/// for its own scheduling overhead.
pub(crate) const PARALLEL_MIN_LEN: usize = 10_000;

/// Fixed chunk size (in elements) used by every parallel elementwise/
/// reduction kernel in this module once a slice clears
/// [`PARALLEL_MIN_LEN`].
///
/// The chunk size is a compile-time constant, *not* derived from the
/// number of threads available at runtime (e.g. not
/// `len / current_num_threads()`). That is deliberate: it means the
/// sequence of chunks a slice is split into -- and so the order partial
/// results are folded back together in -- is exactly the same on a
/// 2-core laptop and a 64-core server. For a reduction (`reduce.rs`) this
/// makes the summation tree, and therefore the final floating-point bit
/// pattern, reproducible across machines; for an elementwise kernel
/// (`elementwise.rs`) it makes output ordering independent of scheduling.
/// Rayon still parallelizes *across* these fixed chunks over however many
/// threads are actually available -- only the chunk boundaries themselves
/// are fixed.
pub(crate) const PARALLEL_CHUNK: usize = 8_192;

/// Minimum estimated FLOP count (`2 * m * n * k`) above which
/// [`gemm::gemm_2d`]'s f64/f32 tier splits the `M` (row) dimension across
/// threads instead of making one single-threaded call into
/// `scirs2_core::ndarray::linalg::general_mat_mul` (the pure-Rust
/// `matrixmultiply` crate, which packs both operands and has no
/// small-matrix degradation of its own).
///
/// `1 << 20` (~1.05M FLOPs) is the measured serial/parallel crossover,
/// re-validated empirically against this backend by [`gemm::gemm_2d`]'s
/// bake-off (see that function's doc comment for the full table): `80^3`
/// (1,024,000 FLOPs, just under) wins serial, `96^3` (1,769,472 FLOPs,
/// just over) wins parallel, in both `f64` and `f32`.
pub(crate) const GEMM_PARALLEL_MIN_FLOPS: usize = 1 << 20;

/// Exercises the actual cross-module call pattern the hot-path-migration
/// lanes are expected to use: extract operands via [`borrow::operand`],
/// then feed them directly into `elementwise`/`reduce`/`gemm`.
/// [`borrow::Operand`]'s `Deref<Target = [T]>` needs to coerce cleanly at
/// each of these call sites -- which is not automatically guaranteed when
/// the callee is itself generic over the slice's element type -- so this
/// confirms it rather than leaving each lane to discover (or fail to
/// discover) it independently. All three compile with a plain `&op`; none
/// needed an explicit `&*op`.
#[cfg(test)]
mod integration_tests {
    use super::{borrow, elementwise, gemm, reduce};
    use crate::array::Array;

    #[test]
    fn operand_composes_with_binary_dispatch() {
        let a = Array::from_vec(vec![1i64, 2, 3, 4]);
        let b = Array::from_vec(vec![10i64, 20, 30, 40]);
        let op_a = borrow::operand(&a);
        let op_b = borrow::operand(&b);
        let out = elementwise::binary_dispatch(&op_a, &op_b, |x, y| x + y);
        assert_eq!(out, vec![11, 22, 33, 44]);
    }

    #[test]
    fn operand_composes_with_reduce() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let op = borrow::operand(&a);
        assert_eq!(reduce::sum_f64(&op), 15.0);
    }

    #[test]
    fn operand_composes_with_gemm_2d() {
        // gemm_2d's `c: &mut [T]` is always a freshly-allocated output
        // buffer, never extracted via `operand()`: `Operand` intentionally
        // has no `DerefMut`, since a `Borrowed` operand aliases the
        // source `Array`'s own storage and nothing here has permission to
        // mutate through a *shared* borrow. Only the read-only `a`/`b`
        // operands go through `operand()`; output buffers are built
        // directly (`vec![T::zero(); n]` or similar) by the caller.
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]); // identity
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
        let op_a = borrow::operand(&a);
        let op_b = borrow::operand(&b);
        let mut c = vec![0.0f64; 4];
        gemm::gemm_2d(2, 2, 2, &op_a, &op_b, &mut c);
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn operand_composes_with_non_contiguous_source() {
        // The `Operand::Owned` (materialized-copy) branch composes the
        // same way as `Borrowed` -- Deref doesn't care which variant it
        // is -- but is worth pinning separately since it's the less
        // common path.
        let a = Array::from_vec(vec![1i64, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let t = a.transpose_axis(0, 1); // shape [3, 2], non-contiguous
        let op = borrow::operand(&t);
        let out = elementwise::unary_dispatch(&op, |x| x * 10);
        assert_eq!(out, vec![10, 40, 20, 50, 30, 60]);
    }
}

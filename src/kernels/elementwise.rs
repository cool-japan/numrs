//! Dtype-dispatched elementwise kernels: binary and unary zip/map over
//! flat slices, tiered on slice length at [`super::PARALLEL_MIN_LEN`].
//!
//! # Why "3-tier type dispatch" is a 2-tier length dispatch here
//!
//! The original design for `binary_dispatch`/`unary_dispatch` called for
//! three tiers keyed on `T`'s identity -- route to an f64-SIMD path, an
//! f32-SIMD path, or a generic serial path -- all driven by *one*
//! caller-supplied closure `F: Fn(T, T) -> T`. That is not soundly
//! implementable as written: proving `TypeId::of::<T>() ==
//! TypeId::of::<f64>()` at runtime lets [`super::cast`] soundly
//! reinterpret *data* (`&[T]` really is `&[f64]` in that branch, by the
//! definition of `TypeId` equality on `'static` types -- see `cast`'s
//! module docs), but it does not let us soundly reinterpret the *closure*
//! `f: F` as if it had signature `Fn(f64, f64) -> f64`. `F` is an opaque,
//! unnameable generic type; there is no safe operation, and `cast.rs`
//! explicitly forbids any `unsafe` one, that turns "T equals f64" into "F
//! equals `Fn(f64, f64) -> f64`" -- doing so would mean transmuting `F`
//! itself, which is unsound in general (a capturing closure's layout is
//! unspecified) and is exactly the kind of type-punning `cast.rs`'s module
//! docs reserve to *data*, never closures.
//!
//! So `binary_dispatch`/`unary_dispatch` instead tier on slice *length*
//! only (serial below [`super::PARALLEL_MIN_LEN`], parallel above, via
//! rayon's own indexed `collect()` -- see [`binary_dispatch`]'s doc
//! comment for why that and not manual `PARALLEL_CHUNK` chunking),
//! applying the caller's closure `F` uniformly at every tier -- sound for
//! every `T: Send + Sync`, not just `f64`/`f32`. Callers that already
//! hold a concretely `f64`- or `f32`-typed closure (true of most existing
//! call sites this module's callers will migrate, since they operate on
//! `Array<f64>`/`Array<f32>` directly today) should call
//! [`binary_f64`]/[`binary_f32`] directly instead, both to opt into that
//! dtype explicitly and because they pre-allocate their output buffer
//! once up front (`f64`/`f32` have a natural zero to fill it with) rather
//! than relying on rayon's collector, which is a closer match to the
//! chunked-parallel shape the original spec described for the f64/f32
//! tier specifically. This is an internal tiering decision only: it
//! changes nothing about what any of the five named functions below take
//! or return relative to the original spec.

use super::{PARALLEL_CHUNK, PARALLEL_MIN_LEN};
use scirs2_core::parallel_ops::*;

/// Single-pass `zip` + `map`: `out[i] = f(a[i], b[i])`. Always serial.
///
/// This is the base case every other kernel in this file falls back to.
/// `a` and `b` must have equal length; callers are expected to have
/// already validated shapes upstream via the public, `Result`-returning
/// API, so this only `debug_assert`s rather than re-checking on every
/// call.
pub(crate) fn binary_serial<T, U, V, F>(a: &[T], b: &[U], f: F) -> Vec<V>
where
    T: Clone,
    U: Clone,
    F: Fn(T, U) -> V,
{
    debug_assert_eq!(
        a.len(),
        b.len(),
        "binary_serial requires equal-length slices"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f(x.clone(), y.clone()))
        .collect()
}

/// `f64` elementwise binary dispatch: serial zip below
/// [`super::PARALLEL_MIN_LEN`], fixed-size chunked parallel zip above it.
pub(crate) fn binary_f64<F>(a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "binary_f64 requires equal-length slices");
    if a.len() < PARALLEL_MIN_LEN {
        return a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    }
    let mut out = vec![0.0_f64; a.len()];
    out.par_chunks_mut(PARALLEL_CHUNK)
        .zip(a.par_chunks(PARALLEL_CHUNK))
        .zip(b.par_chunks(PARALLEL_CHUNK))
        .for_each(|((out_chunk, a_chunk), b_chunk)| {
            for i in 0..out_chunk.len() {
                out_chunk[i] = f(a_chunk[i], b_chunk[i]);
            }
        });
    out
}

/// `f32` twin of [`binary_f64`].
pub(crate) fn binary_f32<F>(a: &[f32], b: &[f32], f: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "binary_f32 requires equal-length slices");
    if a.len() < PARALLEL_MIN_LEN {
        return a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    }
    let mut out = vec![0.0_f32; a.len()];
    out.par_chunks_mut(PARALLEL_CHUNK)
        .zip(a.par_chunks(PARALLEL_CHUNK))
        .zip(b.par_chunks(PARALLEL_CHUNK))
        .for_each(|((out_chunk, a_chunk), b_chunk)| {
            for i in 0..out_chunk.len() {
                out_chunk[i] = f(a_chunk[i], b_chunk[i]);
            }
        });
    out
}

/// Homogeneous-`T` elementwise binary dispatch: serial below
/// [`super::PARALLEL_MIN_LEN`], parallel above. See the module docs for
/// why this is a *length* tier, not a *dtype* tier -- callers with a
/// concretely `f64`/`f32`-typed closure should prefer
/// [`binary_f64`]/[`binary_f32`].
///
/// The parallel tier is `par_iter().zip().map().collect()`, not manual
/// `PARALLEL_CHUNK` chunking into a `Vec<Vec<T>>` followed by a flatten --
/// an earlier version of this function did exactly that, and measured
/// **7-10x slower than the plain serial path at every size from 10K to
/// 1M elements** (see `probe_binary_dispatch_perf_vs_serial`), because
/// generic `T` has no `Default`/`Zero` bound to pre-allocate a single
/// output buffer with (unlike [`binary_f64`]/[`binary_f32`], which do
/// have one and so pre-allocate `out` once and write chunks into it
/// directly), so the chunked approach paid for one `Vec<T>` allocation
/// per chunk *plus* a second full copy of every element through
/// `flatten().collect()`. Rayon's own `collect()` on an *indexed*
/// parallel iterator (this one is: both operands have a known, exact
/// length) avoids both costs -- it pre-allocates the final-size `Vec<T>`
/// once and has each worker write its share directly into the correct
/// slice range, with no `Default` bound required (that bookkeeping is
/// rayon's problem, not this function's). Determinism is not a concern
/// here the way it is in `reduce.rs`: this is an elementwise map, not a
/// reduction, so there is no partial-result combination step to
/// reassociate -- `out[i]` is always exactly `f(a[i].clone(),
/// b[i].clone())` regardless of how rayon schedules the work, and an
/// indexed `collect()` always places each result at its original index.
///
/// Even fixed, this tier is **not free** at exactly
/// [`super::PARALLEL_MIN_LEN`], and callers should not assume
/// `len >= PARALLEL_MIN_LEN` alone guarantees a win: measured against
/// plain [`binary_serial`] on this build (`probe_binary_dispatch_perf_vs_serial`),
/// a trivial closure (`i64` `+`) is *slower* through this dispatch tier
/// up to ~100K elements and only ~1.14x faster at 1M; a costlier closure
/// (a few transcendental f64 ops) is ~2x slower at 10K but ~2.1x faster
/// by 100K. The crossover point depends on per-element closure cost, not
/// just length -- `PARALLEL_MIN_LEN` was calibrated crate-wide (see
/// `mod.rs`), not specifically for this function's worst case (a single
/// trivial op on a `Copy` scalar). Callers with a cheap closure and data
/// near the threshold may come out ahead calling [`binary_serial`]
/// directly instead.
///
/// ## `f64 +` specifically (Lane W2-B / `array/operations.rs::add_broadcast`)
///
/// A second, independent probe
/// (`array::operations::perf_probe::probe_add_broadcast_serial_vs_dispatch_vs_old`)
/// measured this tier against [`binary_serial`] *and* against the
/// pre-migration baseline (`&ndarray::ArrayBase<_, IxDyn> + &..`, i.e. what
/// `add_broadcast`'s closure did before routing through this module) for
/// plain `f64` addition, release build, same-shape contiguous operands:
///
/// | n (elements) | old `IxDyn` add | [`binary_serial`] | this fn (`binary_dispatch`) |
/// |---|---|---|---|
/// | 64      | 102 ns  | 30 ns (3.35x)  | 31 ns (3.28x)   |
/// | 1,000   | 244 ns  | 152 ns (1.60x) | 136 ns (1.79x)  |
/// | 10,000  | 2.09 µs | 2.11 µs (0.99x)| 36.0 µs (0.06x) |
/// | 100,000 | 21.5 µs | 21.3 µs (1.01x)| 52.9 µs (0.41x) |
/// | 1e6     | 253 µs  | 224 µs (1.13x) | 178 µs (1.42x)  |
///
/// (speedup vs. the old `IxDyn` baseline in parentheses). Two things fall
/// out of this that are specific to a single trivial `Copy`-scalar op and
/// do **not** generalize to this function's other callers:
///
/// - The large win at small `n` is from escaping `IxDyn`'s dynamic-stride
///   per-element bookkeeping into a flat contiguous slice -- not from SIMD
///   or parallelism (both tiers are still serial below
///   `PARALLEL_MIN_LEN`, and the win is identical between `binary_serial`
///   and this fn there, as expected).
/// - At `n` right at and above `PARALLEL_MIN_LEN` (10K, 100K), `ndarray`'s
///   own contiguous-array fast path for `IxDyn` is *already* about as fast
///   per-element as a flat serial loop -- there is no autovectorization
///   gap left to close for this specific op at this size -- so this fn's
///   parallel tier pays pure rayon dispatch overhead for a per-element cost
///   (one `f64` add) far too cheap to amortize it, costing 2.4x at 100K and
///   16x at 10K. It only pulls ahead of the old baseline again at 1e6,
///   and even there [`binary_serial`] is the closer competitor, not the
///   old baseline. **`add_broadcast` therefore calls [`binary_serial`],
///   not this function**, despite this function being the original design
///   target for that call site -- see that method's doc comment.
///
/// ## Plain zip-loop vs. `scirs2_core`'s `f64::simd_add` intrinsic
///
/// `bench/elementwise_dispatch_benchmark.rs`'s
/// `elementwise_dispatch/zip_loop_vs_simd_add` group (criterion, release,
/// `--sample-size 10`) compared [`binary_serial`]'s underlying pattern
/// (`a.iter().zip(b.iter()).map(|(&x,&y)| x + y).collect::<Vec<f64>>()`,
/// contiguous `f64` slices) against calling
/// `scirs2_core::simd_ops::SimdUnifiedOps::simd_add` directly on
/// `ArrayView1`s of the same slices:
///
/// | n         | zip loop  | `simd_add` | zip-loop speedup |
/// |---|---|---|---|
/// | 64        | 110 ns    | 231 ns     | 2.1x |
/// | 1,000     | 493 ns    | 1.27 µs    | 2.6x |
/// | 10,000    | 6.7 µs    | 5.4 µs     | 0.8x (noisy, within run-to-run variance) |
/// | 100,000   | 17.7 µs   | 33.0 µs    | 1.9x |
/// | 1,000,000 | 241 µs    | 494 µs     | 2.0x |
///
/// The plain zip loop wins (or ties within noise) at every size measured.
/// `simd_add` is not a free win here despite being the "real" SIMD
/// intrinsic path: LLVM already autovectorizes the zip loop's trivial
/// `f64` add over a contiguous slice about as well by itself, and
/// `simd_add`'s own dispatch/view-construction overhead is not free
/// either. This confirms the design prediction that going out of the way
/// to route through `simd_add` would not pay for itself here -- the zip
/// loop [`binary_serial`]/[`binary_dispatch`] already use is kept as-is.
pub(crate) fn binary_dispatch<T, F>(a: &[T], b: &[T], f: F) -> Vec<T>
where
    T: 'static + Clone + Send + Sync,
    F: Fn(T, T) -> T + Sync + Send,
{
    debug_assert_eq!(
        a.len(),
        b.len(),
        "binary_dispatch requires equal-length slices"
    );
    if a.len() < PARALLEL_MIN_LEN {
        return binary_serial(a, b, f);
    }
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| f(x.clone(), y.clone()))
        .collect()
}

/// Single-pass `map`: `out[i] = f(a[i])`. Always serial.
pub(crate) fn unary_serial<T, V, F>(a: &[T], f: F) -> Vec<V>
where
    T: Clone,
    F: Fn(T) -> V,
{
    a.iter().map(|x| f(x.clone())).collect()
}

/// Homogeneous-`T` elementwise unary dispatch: serial below
/// [`super::PARALLEL_MIN_LEN`], parallel above. Same length-vs-dtype
/// tiering rationale, and same "why `par_iter().map().collect()` and not
/// manual chunking" rationale, as [`binary_dispatch`].
pub(crate) fn unary_dispatch<T, F>(a: &[T], f: F) -> Vec<T>
where
    T: 'static + Clone + Send + Sync,
    F: Fn(T) -> T + Sync + Send,
{
    if a.len() < PARALLEL_MIN_LEN {
        return unary_serial(a, f);
    }
    a.par_iter().map(|x| f(x.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_serial_zips_and_maps() {
        let a = vec![1, 2, 3];
        let b = vec![10, 20, 30];
        let out = binary_serial(&a, &b, |x, y| x + y);
        assert_eq!(out, vec![11, 22, 33]);
    }

    #[test]
    fn binary_serial_heterogeneous_types() {
        let a = vec![1_i32, 2, 3];
        let b = vec![1.5_f64, 2.5, 3.5];
        let out: Vec<f64> = binary_serial(&a, &b, |x, y| x as f64 + y);
        assert_eq!(out, vec![2.5, 4.5, 6.5]);
    }

    #[test]
    fn binary_f64_matches_naive_below_and_above_threshold() {
        for &n in &[10usize, 20_000] {
            let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
            let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y + 1.0).collect();
            let got = binary_f64(&a, &b, |x, y| x * y + 1.0);
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn probe_binary_dispatch_perf_vs_serial() {
        fn chunk_flatten_variant<T, F>(a: &[T], b: &[T], f: F) -> Vec<T>
        where
            T: 'static + Clone + Send + Sync,
            F: Fn(T, T) -> T + Sync + Send,
        {
            a.par_chunks(PARALLEL_CHUNK)
                .zip(b.par_chunks(PARALLEL_CHUNK))
                .map(|(a_chunk, b_chunk)| binary_serial(a_chunk, b_chunk, &f))
                .collect::<Vec<Vec<T>>>()
                .into_iter()
                .flatten()
                .collect()
        }

        for &n in &[10_000usize, 20_000, 100_000, 1_000_000] {
            let a: Vec<i64> = (0..n as i64).collect();
            let b: Vec<i64> = (0..n as i64).map(|i| i * 2).collect();
            let iters = if n <= 100_000 { 200 } else { 20 };

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(binary_serial(&a, &b, |x, y| x + y));
            }
            let serial = t0.elapsed();

            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(chunk_flatten_variant(&a, &b, |x, y| x + y));
            }
            let chunk_flatten = t1.elapsed();

            let t2 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(binary_dispatch(&a, &b, |x, y| x + y));
            }
            let current = t2.elapsed();

            eprintln!(
                "[trivial +] n={n} iters={iters}: serial={:.0}ns/iter chunk_flatten={:.0}ns/iter({:.2}x) current_binary_dispatch={:.0}ns/iter({:.2}x)",
                serial.as_nanos() as f64 / iters as f64,
                chunk_flatten.as_nanos() as f64 / iters as f64,
                serial.as_secs_f64() / chunk_flatten.as_secs_f64(),
                current.as_nanos() as f64 / iters as f64,
                serial.as_secs_f64() / current.as_secs_f64(),
            );
        }

        // Same comparison with a costlier per-element closure (a small
        // handful of transcendental ops), representative of a real
        // elementwise math kernel rather than a single trivial `+` --
        // trivial ops are the worst case for any parallelization scheme.
        fn costly(x: f64, y: f64) -> f64 {
            (x.sqrt() + y.sqrt()).powi(2) - (x * y).ln().abs()
        }
        for &n in &[10_000usize, 100_000] {
            let a: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
            let b: Vec<f64> = (1..=n as i64).map(|i| i as f64 * 1.5).collect();
            let iters = 200;

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(binary_serial(&a, &b, costly));
            }
            let serial = t0.elapsed();

            let t2 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(binary_dispatch(&a, &b, costly));
            }
            let current = t2.elapsed();

            eprintln!(
                "[costly transcendental] n={n} iters={iters}: serial={:.0}ns/iter current_binary_dispatch={:.0}ns/iter({:.2}x)",
                serial.as_nanos() as f64 / iters as f64,
                current.as_nanos() as f64 / iters as f64,
                serial.as_secs_f64() / current.as_secs_f64(),
            );
        }
    }

    #[test]
    fn binary_f32_matches_naive_below_and_above_threshold() {
        for &n in &[10usize, 20_000] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
            let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
            let got = binary_f32(&a, &b, |x, y| x - y);
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn binary_dispatch_matches_naive_below_and_above_threshold() {
        for &n in &[10usize, 20_000] {
            let a: Vec<i64> = (0..n as i64).collect();
            let b: Vec<i64> = (0..n as i64).map(|i| i * 2).collect();
            let expected: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
            let got = binary_dispatch(&a, &b, |x, y| x + y);
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn binary_dispatch_empty() {
        let out: Vec<i32> = binary_dispatch(&[], &[], |x, y| x + y);
        assert!(out.is_empty());
    }

    #[test]
    fn unary_serial_maps() {
        let a = vec![1, 2, 3];
        let out = unary_serial(&a, |x| x * x);
        assert_eq!(out, vec![1, 4, 9]);
    }

    #[test]
    fn unary_dispatch_matches_naive_below_and_above_threshold() {
        for &n in &[10usize, 20_000] {
            let a: Vec<i64> = (0..n as i64).collect();
            let expected: Vec<i64> = a.iter().map(|&x| x * 3 - 1).collect();
            let got = unary_dispatch(&a, |x| x * 3 - 1);
            assert_eq!(got, expected, "n={n}");
        }
    }

    #[test]
    fn unary_dispatch_empty() {
        let out: Vec<i32> = unary_dispatch(&[], |x| x + 1);
        assert!(out.is_empty());
    }
}

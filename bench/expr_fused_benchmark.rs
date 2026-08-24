//! Does `(a.expr() + b.expr() * c.expr()).eval()` actually beat
//! `&a + &(&b * &c)`?
//!
//! That is the whole point of `src/expr/owned.rs` + `src/expr/fused_eval.rs`,
//! and this bench is where the claim is checked. The fused form makes one pass
//! over three input slices and allocates one output buffer; the eager form
//! makes two passes and allocates two buffers, one of which (`b * c`) is
//! thrown away immediately.
//!
//! # Two modes
//!
//! ```text
//! cargo bench --bench expr_fused_benchmark              # criterion means
//! EXPR_AB_REPORT=1 cargo bench --bench expr_fused_benchmark   # min-over-alternating-A/B
//! ```
//!
//! The A/B report is the authoritative one. This repo's machines carry
//! background load from unrelated builds; criterion reports a *mean*, and a
//! neighbouring process descheduling one candidate inflates that candidate's
//! mean while leaving the other alone. The A/B report interleaves the
//! candidates within each round (rotating which goes first) and keeps the
//! **minimum** sample per candidate, which is far harder to bias.
//!
//! The report covers four questions:
//!
//! 1. `report_fused_vs_eager` -- the kill criterion: fused must beat eager at
//!    n >= 100_000.
//! 2. `report_wider_trees` -- the same question for a four-leaf shape, and
//!    for an eight-leaf tree that is past every specialised loop and so takes
//!    the eager fallback.
//! 3. `report_simd_alternatives` -- whether routing the fused inner loop
//!    through `scirs2_core::simd_ops::SimdUnifiedOps` (`simd_mul` + `simd_add`,
//!    or `simd_fma`) would be faster than the plain zip loop `fused_eval` uses.
//! 4. `report_build_cost` -- that building the expression tree is O(1) in the
//!    array size, i.e. `.expr()` really is an `Arc` bump and not a copy.

#![allow(clippy::result_large_err)]

use criterion::{BenchmarkId, Criterion, Throughput};
use numrs2::array::Array;
use numrs2::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::hint::black_box;
use std::time::{Duration, Instant};

const SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];

/// The size at and above which fusion must win (the kill criterion).
const KILL_CRITERION_MIN_N: usize = 100_000;

fn sample(n: usize, scale: f64, offset: f64) -> Array<f64> {
    Array::from_vec((0..n).map(|i| i as f64 * scale + offset).collect())
}

// ---------------------------------------------------------------------------
// The two candidates
// ---------------------------------------------------------------------------

/// Fused: one pass, one allocation.
fn fused(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
    (a.expr() + b.expr() * c.expr())
        .eval()
        .expect("same-shape f64 leaves always evaluate")
}

/// Eager: two passes, two allocations. Exactly what a user writes today.
fn eager(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Array<f64> {
    a + &(b * c)
}

/// The four-leaf `(a op b) op (c op d)` shape.
///
/// This is what motivated `fused_eval::zip4`: before that loop existed, the
/// shape fell to a general block interpreter that measured 0.53-0.94x against
/// eager -- i.e. "fusion" that lost. This row is how that was caught and how
/// the fix is checked.
fn fused_four_leaf(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>, d: &Array<f64>) -> Array<f64> {
    ((a.expr() + b.expr()) * (c.expr() - d.expr()))
        .eval()
        .expect("same-shape f64 leaves always evaluate")
}

/// The eager spelling of the same expression: three passes, three
/// allocations.
fn eager_four_leaf(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>, d: &Array<f64>) -> Array<f64> {
    &(a + b) * &(c - d)
}

/// An eight-leaf tree, well past every specialised loop.
///
/// This is the case the removed block interpreter was *most* suited to -- it
/// would have read 9n where eager reads and writes 21n -- and it still lost
/// (0.66-0.86x), which is why `eval()` now sends such trees to the eager
/// fallback. The row therefore checks the fallback costs nothing but the tree
/// construction: ~1.00x is the expected result.
fn fused_deep(xs: &[Array<f64>; 8]) -> Array<f64> {
    (((xs[0].expr() + xs[1].expr()) * (xs[2].expr() - xs[3].expr()))
        + ((xs[4].expr() * xs[5].expr()) - (xs[6].expr() + xs[7].expr())))
    .eval()
    .expect("same-shape f64 leaves always evaluate")
}

/// The eager spelling of the same eight-leaf tree.
fn eager_deep(xs: &[Array<f64>; 8]) -> Array<f64> {
    let left = &(&xs[0] + &xs[1]) * &(&xs[2] - &xs[3]);
    let right = &(&xs[4] * &xs[5]) - &(&xs[6] + &xs[7]);
    &left + &right
}

/// `SimdUnifiedOps` route A: `simd_mul` then `simd_add`, two `Array1`
/// allocations.
fn simd_mul_add(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let prod = f64::simd_mul(&ArrayView1::from(b), &ArrayView1::from(c));
    let out = f64::simd_add(&ArrayView1::from(a), &prod.view());
    out.to_vec()
}

/// `SimdUnifiedOps` route B: the trait's own `simd_fma`.
///
/// Note this is measured for information only -- it computes a true
/// single-rounding fused multiply-add, so `fused_eval` could not use it
/// without matching what the eager path rounds twice.
fn simd_fma_route(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    f64::simd_fma(
        &ArrayView1::from(b),
        &ArrayView1::from(c),
        &ArrayView1::from(a),
    )
    .to_vec()
}

/// The plain zip loop `fused_eval` actually runs, isolated from `Array`
/// wrapping so the SIMD comparison is like-for-like.
fn zip_loop(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((&x, &y), &z)| x + y * z)
        .collect()
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

fn bench_fused_vs_eager(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_fused/a_plus_b_times_c");
    for n in SIZES {
        let x = sample(n, 1.0, 0.5);
        let y = sample(n, -0.25, 3.0);
        let z = sample(n, 0.125, -2.0);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("fused", n), &n, |bencher, _| {
            bencher.iter(|| black_box(fused(&x, &y, &z)))
        });
        group.bench_with_input(BenchmarkId::new("eager", n), &n, |bencher, _| {
            bencher.iter(|| black_box(eager(&x, &y, &z)))
        });
    }
    group.finish();
}

fn bench_expr_node_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_fused/expr_node_build");
    for n in [1_000usize, 1_000_000] {
        let x = sample(n, 1.0, 0.5);
        let y = sample(n, -0.25, 3.0);
        let z = sample(n, 0.125, -2.0);
        group.bench_with_input(
            BenchmarkId::new("build_3_node_tree", n),
            &n,
            |bencher, _| bencher.iter(|| black_box(x.expr() + y.expr() * z.expr())),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// A/B report
// ---------------------------------------------------------------------------

fn timed<F: FnOnce() -> R, R>(f: F) -> Duration {
    let t = Instant::now();
    let out = f();
    let d = t.elapsed();
    black_box(out);
    d
}

/// Run every candidate `rounds` times, rotating which candidate goes first in
/// each round, and return the minimum observed duration of each.
fn min_rotate(rounds: usize, fs: &mut [&mut dyn FnMut() -> Duration]) -> Vec<Duration> {
    let k = fs.len();
    let mut mins = vec![Duration::MAX; k];
    for round in 0..rounds {
        for slot in 0..k {
            let idx = (slot + round) % k;
            let d = fs[idx]();
            mins[idx] = mins[idx].min(d);
        }
    }
    mins
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

fn report_fused_vs_eager() {
    println!("\n== (1) KILL CRITERION: fused vs eager, `a + b * c`, f64 ==");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>10}  {:>8}",
        "n", "fused", "eager", "speedup", "verdict"
    );

    let mut all_pass = true;
    for n in SIZES {
        let a = sample(n, 1.0, 0.5);
        let b = sample(n, -0.25, 3.0);
        let c = sample(n, 0.125, -2.0);

        for _ in 0..3 {
            black_box(fused(&a, &b, &c));
            black_box(eager(&a, &b, &c));
        }

        let rounds = if n >= 1_000_000 { 21 } else { 101 };
        let mut f = || timed(|| fused(&a, &b, &c));
        let mut e = || timed(|| eager(&a, &b, &c));
        let mins = min_rotate(rounds, &mut [&mut f, &mut e]);
        let (fused_t, eager_t) = (mins[0], mins[1]);
        let speedup = secs(eager_t) / secs(fused_t);

        let verdict = if n < KILL_CRITERION_MIN_N {
            "-"
        } else if speedup > 1.0 {
            "PASS"
        } else {
            all_pass = false;
            "FAIL"
        };
        println!(
            "{:>10}  {:>14.1?}  {:>14.1?}  {:>9.2}x  {:>8}",
            n, fused_t, eager_t, speedup, verdict
        );
    }
    println!(
        "  kill criterion (fused faster than eager at n >= {KILL_CRITERION_MIN_N}): {}",
        if all_pass { "PASS" } else { "FAIL" }
    );
}

fn report_wider_trees() {
    println!("\n== (2) four-leaf specialised shape: `(a + b) * (c - d)`, f64 ==");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>10}",
        "n", "fused", "eager", "speedup"
    );

    for n in SIZES {
        let a = sample(n, 1.0, 0.5);
        let b = sample(n, -0.25, 3.0);
        let c = sample(n, 0.125, -2.0);
        let d = sample(n, 2.0, 1.0);

        for _ in 0..3 {
            black_box(fused_four_leaf(&a, &b, &c, &d));
            black_box(eager_four_leaf(&a, &b, &c, &d));
        }

        let rounds = if n >= 1_000_000 { 21 } else { 101 };
        let mut f = || timed(|| fused_four_leaf(&a, &b, &c, &d));
        let mut e = || timed(|| eager_four_leaf(&a, &b, &c, &d));
        let mins = min_rotate(rounds, &mut [&mut f, &mut e]);
        println!(
            "{:>10}  {:>14.1?}  {:>14.1?}  {:>9.2}x",
            n,
            mins[0],
            mins[1],
            secs(mins[1]) / secs(mins[0])
        );
    }
    println!("  (this shape has its own `zip4` loop; before it did, the block");
    println!("   interpreter served it at 0.53-0.94x -- i.e. slower than eager)");

    println!("\n== (2b) past the specialised set: 8-leaf tree, f64 ==");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>10}",
        "n", "fused", "eager", "speedup"
    );
    for n in SIZES {
        let xs: [Array<f64>; 8] =
            std::array::from_fn(|k| sample(n, 0.5 + k as f64 * 0.25, 1.0 - k as f64 * 0.5));

        for _ in 0..3 {
            black_box(fused_deep(&xs));
            black_box(eager_deep(&xs));
        }

        let rounds = if n >= 1_000_000 { 21 } else { 101 };
        let mut f = || timed(|| fused_deep(&xs));
        let mut e = || timed(|| eager_deep(&xs));
        let mins = min_rotate(rounds, &mut [&mut f, &mut e]);
        println!(
            "{:>10}  {:>14.1?}  {:>14.1?}  {:>9.2}x",
            n,
            mins[0],
            mins[1],
            secs(mins[1]) / secs(mins[0])
        );
    }
    println!("  (past every specialised loop, so this runs the eager fallback:");
    println!("   ~1.00x is the target -- the extra cost is tree construction,");
    println!("   which only shows at small n)");
}

fn report_simd_alternatives() {
    println!("\n== (3) fused inner loop: plain zip vs `SimdUnifiedOps` ==");
    println!(
        "{:>10}  {:>13}  {:>13}  {:>13}  {:>13}",
        "n", "zip loop", "simd_mul+add", "simd_fma", "eager Array"
    );

    for n in SIZES {
        let a = sample(n, 1.0, 0.5);
        let b = sample(n, -0.25, 3.0);
        let c = sample(n, 0.125, -2.0);
        let (sa, sb, sc) = (
            a.as_slice().expect("contiguous"),
            b.as_slice().expect("contiguous"),
            c.as_slice().expect("contiguous"),
        );

        for _ in 0..3 {
            black_box(zip_loop(sa, sb, sc));
            black_box(simd_mul_add(sa, sb, sc));
            black_box(simd_fma_route(sa, sb, sc));
            black_box(eager(&a, &b, &c));
        }

        let rounds = if n >= 1_000_000 { 21 } else { 41 };
        let mut f0 = || timed(|| zip_loop(sa, sb, sc));
        let mut f1 = || timed(|| simd_mul_add(sa, sb, sc));
        let mut f2 = || timed(|| simd_fma_route(sa, sb, sc));
        let mut f3 = || timed(|| eager(&a, &b, &c));
        let mins = min_rotate(rounds, &mut [&mut f0, &mut f1, &mut f2, &mut f3]);
        println!(
            "{:>10}  {:>13.1?}  {:>13.1?}  {:>13.1?}  {:>13.1?}",
            n, mins[0], mins[1], mins[2], mins[3]
        );
    }
    println!("  (`simd_fma` is single-rounding, so it is informational only --");
    println!("   using it computes a different value from the eager path.)");
}

fn report_build_cost() {
    println!("\n== (4) expr_node_build: tree construction must be O(1) in n ==");
    println!(
        "{:>10}  {:>16}  {:>16}",
        "n", "build 3-node tree", "per node"
    );

    // 64 builds per sample: a single tree build is a few ns, below what one
    // `Instant` pair can resolve.
    const BATCH: usize = 64;
    let mut timings = Vec::new();
    for n in [1_000usize, 1_000_000] {
        let a = sample(n, 1.0, 0.5);
        let b = sample(n, -0.25, 3.0);
        let c = sample(n, 0.125, -2.0);

        for _ in 0..5 {
            black_box(a.expr() + b.expr() * c.expr());
        }

        let mut build = || {
            timed(|| {
                for _ in 0..BATCH {
                    black_box(a.expr() + b.expr() * c.expr());
                }
            })
        };
        let mut noop = || timed(|| black_box(0usize));
        let mins = min_rotate(201, &mut [&mut build, &mut noop]);
        let each = mins[0] / BATCH as u32;
        timings.push(each);
        println!("{:>10}  {:>16.1?}  {:>16.1?}", n, each, each / 3);
    }

    if let (Some(small), Some(large)) = (timings.first(), timings.last()) {
        let ratio = secs(*large) / secs(*small);
        println!(
            "  n=1e6 / n=1e3 build-time ratio: {ratio:.3}  (~1.0 => no O(n) work before eval())"
        );
    }

    // The 8-leaf tree from report (2b), so the fallback's small-n shortfall can
    // be attributed instead of extrapolated from the 3-node number.
    println!(
        "\n{:>10}  {:>16}  {:>16}",
        "n", "build 8-leaf tree", "eval overhead"
    );
    for n in [1_000usize, 1_000_000] {
        let xs: [Array<f64>; 8] =
            std::array::from_fn(|k| sample(n, 0.5 + k as f64 * 0.25, 1.0 - k as f64 * 0.5));

        let tree = || {
            ((xs[0].expr() + xs[1].expr()) * (xs[2].expr() - xs[3].expr()))
                + ((xs[4].expr() * xs[5].expr()) - (xs[6].expr() + xs[7].expr()))
        };
        for _ in 0..5 {
            black_box(tree());
            black_box(fused_deep(&xs));
            black_box(eager_deep(&xs));
        }

        let mut build = || {
            timed(|| {
                for _ in 0..BATCH {
                    black_box(tree());
                }
            })
        };
        let mut noop = || timed(|| black_box(0usize));
        let build_min = min_rotate(201, &mut [&mut build, &mut noop])[0] / BATCH as u32;

        // Whole-call overhead of the fallback over hand-written eager: this is
        // tree construction plus plan() + collect_leaves + the shape Vec.
        let rounds = if n >= 1_000_000 { 21 } else { 101 };
        let mut f = || timed(|| fused_deep(&xs));
        let mut e = || timed(|| eager_deep(&xs));
        let mins = min_rotate(rounds, &mut [&mut f, &mut e]);
        let overhead = mins[0].saturating_sub(mins[1]);

        println!("{:>10}  {:>16.1?}  {:>16.1?}", n, build_min, overhead);
    }
    println!("  (build = tree construction alone; eval overhead = the whole");
    println!("   `.expr()/.eval()` call minus the eager spelling, so the");
    println!("   difference between the two columns is plan() + collect_leaves)");
}

fn ab_report() {
    println!("Fused expression A/B report -- minimum over alternating samples, release profile.");
    report_fused_vs_eager();
    report_wider_trees();
    report_simd_alternatives();
    report_build_cost();
    println!();
}

fn main() {
    if std::env::var_os("EXPR_AB_REPORT").is_some() {
        ab_report();
        return;
    }

    let mut c = Criterion::default().configure_from_args();
    bench_fused_vs_eager(&mut c);
    bench_expr_node_build(&mut c);
    c.final_summary();
}

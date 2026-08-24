//! Reference tests for `Array::matmul`'s migration onto the crate's
//! dtype-dispatched GEMM kernel (`src/kernels/gemm.rs`).
//!
//! `matmul` no longer runs one hand-blocked triple loop for every dtype and
//! shape. It now picks between a float tier (`f64`/`f32`, on
//! `scirs2_core::ndarray::linalg::general_mat_mul`, i.e. `matrixmultiply`)
//! and a generic blocked fallback for every other dtype; the float tier
//! additionally splits rows across threads once the FLOP count clears
//! `kernels::GEMM_PARALLEL_MIN_FLOPS`, and the N-D batched path feeds each
//! panel through that same dispatcher via flat-index arithmetic instead of
//! per-element `IxDyn` `get`/`set`.
//!
//! Every one of those branches is a place the answer could silently change,
//! so this file pins `matmul` against an independent naive triple-loop
//! oracle across the tier and blocking boundaries: dtype (f64 / f32 / i32),
//! size (`BLOCK_SIZE = 64` and SIMD-lane multiples, ±1 either side),
//! degenerate `m`/`k`/`n = 0`, non-contiguous operands, and the batched and
//! broadcast-batched N-D shapes.
//!
//! # Why tolerances are magnitude-relative, not absolute
//!
//! The `f64`/`f32` tier hands off to `matrixmultiply`'s packed, blocked
//! kernel, and past `GEMM_PARALLEL_MIN_FLOPS` it splits the `M` dimension
//! over `current_num_threads()` chunks. Floating-point addition is not
//! associative, so a `k`-term dot product accumulated in blocked order
//! legitimately differs from this file's left-to-right `+=` oracle by a few
//! ULPs, growing with `k` and with the accumulated magnitude. (The row
//! split itself is *not* a source of that drift: it partitions `M` only, so
//! every dot product is still summed inside one chunk in the same
//! `k`-determined order -- `kernels::gemm` pins that with an exact-equality
//! unit test. What varies is blocked-vs-naive order, not core count.)
//! A fixed absolute epsilon would pass at small `k` and flake once the
//! accumulated magnitude grows. `assert_close_rel` scales the tolerance by
//! the expected value's own magnitude instead (mirroring the same helper in
//! `kernels/gemm.rs`'s unit tests), which stays meaningful for both
//! near-zero and large sums.

use numrs2::array::Array;

// ---------------------------------------------------------------------
// Oracles and helpers
// ---------------------------------------------------------------------

/// Independent naive triple-loop GEMM over flat row-major data.
///
/// Deliberately *not* blocked and not dispatched: this is the reference
/// `matmul` is checked against, so it must share no code with it.
fn naive_gemm_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// `f32` twin of [`naive_gemm_f64`].
fn naive_gemm_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// `i32` twin of [`naive_gemm_f64`], for the generic (non-SIMD) tier.
fn naive_gemm_i32(m: usize, k: usize, n: usize, a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Magnitude-relative float comparison; see this file's module docs for
/// why an absolute epsilon is the wrong tool here.
fn assert_close_rel(got: f64, expected: f64, tol: f64, ctx: impl std::fmt::Display) {
    let scale = expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= tol * scale,
        "{ctx}: got {got}, expected {expected} (relative diff {:.3e}, tol {tol:.0e})",
        (got - expected).abs() / scale,
    );
}

fn assert_close_rel_f32(got: f32, expected: f32, tol: f32, ctx: impl std::fmt::Display) {
    let scale = expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= tol * scale,
        "{ctx}: got {got}, expected {expected} (relative diff {:.3e}, tol {tol:.0e})",
        (got - expected).abs() / scale,
    );
}

/// Deterministic, non-degenerate test data. A plain `0, 1, 2, ...` ramp
/// makes many wrong index arithmetics still produce plausible-looking
/// numbers; mixing a scale and an offset (and letting values straddle
/// zero) makes a transposed or mis-strided read visibly wrong.
fn seq_f64(len: usize, scale: f64, offset: f64) -> Vec<f64> {
    (0..len).map(|i| (i as f64) * scale + offset).collect()
}

fn seq_f32(len: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * scale + offset).collect()
}

fn seq_i32(len: usize, modulus: i32, offset: i32) -> Vec<i32> {
    (0..len).map(|i| (i as i32 % modulus) + offset).collect()
}

/// The `(m, k, n)` grid every dtype is checked over.
///
/// Sampled from the boundary set `{1, 2, 31, 32, 33, 63, 64, 65, 127, 128}`
/// -- the values that straddle `kernels::gemm::gemm_generic`'s
/// `BLOCK_SIZE = 64` blocking, `matrixmultiply`'s micro-panel widths, and
/// the SIMD lane widths underneath both. Combinations deliberately mix *unequal*
/// `m`, `k`, `n`: a cubic grid cannot catch an `m`/`k`/`n` mixup, since every
/// stride multiplier is then identical.
const SIZE_GRID: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 32, 1),
    (2, 31, 33),
    (31, 32, 33),
    (32, 32, 32),
    (33, 64, 31),
    (63, 64, 65),
    (64, 64, 64),
    (65, 63, 127),
    (127, 128, 2),
    (128, 128, 128),
    (128, 1, 64),
    (64, 127, 65),
    (32, 128, 63),
    // The serial/parallel cutover. `kernels::gemm`'s float tier switches to
    // an `M`-row-split parallel call when `2*m*n*k >= GEMM_PARALLEL_MIN_FLOPS`
    // (`1 << 20` = 1,048,576). With `m = k = 64`, `2*m*k = 8192` divides that
    // exactly, so `n = 128` lands *on* the threshold and `n = 127`/`n = 129`
    // sit one step either side of it. `kernels::gemm` is `pub(crate)`, so
    // this integration test cannot name the constant -- it derives the
    // shapes in its own unit tests (`gemm_2d_matches_naive_at_the_parallel_cutover`)
    // and these three mirror them end-to-end through `Array::matmul`.
    (64, 64, 127),
    (64, 64, 128),
    (64, 64, 129),
];

// ---------------------------------------------------------------------
// 2-D, per-dtype tier
// ---------------------------------------------------------------------

#[test]
fn matmul_2d_f64_matches_naive_across_boundary_grid() {
    for &(m, k, n) in SIZE_GRID {
        let a_data = seq_f64(m * k, 0.125, -3.0);
        let b_data = seq_f64(k * n, -0.0625, 1.5);
        let expected = naive_gemm_f64(m, k, n, &a_data, &b_data);

        let a = Array::from_vec(a_data).reshape(&[m, k]);
        let b = Array::from_vec(b_data).reshape(&[k, n]);
        let c = a.matmul(&b).expect("matmul should succeed");

        assert_eq!(c.shape(), vec![m, n], "(m={m},k={k},n={n}) shape");
        for (idx, (got, want)) in c.to_vec().iter().zip(&expected).enumerate() {
            assert_close_rel(*got, *want, 1e-9, format!("(m={m},k={k},n={n}) idx={idx}"));
        }
    }
}

#[test]
fn matmul_2d_f32_matches_naive_across_boundary_grid() {
    for &(m, k, n) in SIZE_GRID {
        let a_data = seq_f32(m * k, 0.125, -3.0);
        let b_data = seq_f32(k * n, -0.0625, 1.5);
        let expected = naive_gemm_f32(m, k, n, &a_data, &b_data);

        let a = Array::from_vec(a_data).reshape(&[m, k]);
        let b = Array::from_vec(b_data).reshape(&[k, n]);
        let c = a.matmul(&b).expect("matmul should succeed");

        assert_eq!(c.shape(), vec![m, n], "(m={m},k={k},n={n}) shape");
        for (idx, (got, want)) in c.to_vec().iter().zip(&expected).enumerate() {
            assert_close_rel_f32(*got, *want, 1e-4, format!("(m={m},k={k},n={n}) idx={idx}"));
        }
    }
}

/// `i32` has no SIMD tier, so this exercises `gemm_generic` -- the blocked
/// fallback every non-`f64`/`f32` dtype lands on. Integers are exact, so
/// this is an equality assertion, which also makes it the strictest check
/// in the file that the blocking arithmetic itself is right.
#[test]
fn matmul_2d_i32_generic_tier_matches_naive_across_boundary_grid() {
    for &(m, k, n) in SIZE_GRID {
        let a_data = seq_i32(m * k, 7, -3);
        let b_data = seq_i32(k * n, 5, -2);
        let expected = naive_gemm_i32(m, k, n, &a_data, &b_data);

        let a = Array::from_vec(a_data).reshape(&[m, k]);
        let b = Array::from_vec(b_data).reshape(&[k, n]);
        let c = a.matmul(&b).expect("matmul should succeed");

        assert_eq!(c.shape(), vec![m, n], "(m={m},k={k},n={n}) shape");
        assert_eq!(c.to_vec(), expected, "(m={m},k={k},n={n})");
    }
}

// ---------------------------------------------------------------------
// Degenerate dimensions
// ---------------------------------------------------------------------

/// `k = 0` is the interesting empty case: the *output* is not empty
/// (`m * n` elements), it is all zeros -- an empty sum. A kernel that
/// skipped zero-initialization, or that returned early on "no work",
/// would hand back an uninitialized or wrongly-shaped buffer here.
#[test]
fn matmul_2d_zero_k_yields_zero_filled_output() {
    let (m, k, n) = (5usize, 0usize, 4usize);
    let a = Array::<f64>::from_vec(vec![]).reshape(&[m, k]);
    let b = Array::<f64>::from_vec(vec![]).reshape(&[k, n]);

    let c = a.matmul(&b).expect("matmul with k=0 should succeed");
    assert_eq!(c.shape(), vec![m, n]);
    assert_eq!(c.to_vec(), vec![0.0; m * n]);
}

/// `m = 0` and `n = 0` both give a genuinely empty output; the shape must
/// still be right.
#[test]
fn matmul_2d_zero_m_and_zero_n_yield_empty_outputs() {
    let a = Array::<f64>::from_vec(vec![]).reshape(&[0, 5]);
    let b = Array::from_vec(seq_f64(20, 1.0, 0.0)).reshape(&[5, 4]);
    let c = a.matmul(&b).expect("matmul with m=0 should succeed");
    assert_eq!(c.shape(), vec![0, 4]);
    assert!(c.to_vec().is_empty());

    let a2 = Array::from_vec(seq_f64(20, 1.0, 0.0)).reshape(&[5, 4]);
    let b2 = Array::<f64>::from_vec(vec![]).reshape(&[4, 0]);
    let c2 = a2.matmul(&b2).expect("matmul with n=0 should succeed");
    assert_eq!(c2.shape(), vec![5, 0]);
    assert!(c2.to_vec().is_empty());
}

/// The generic tier's own `k = 0` behavior, checked separately: it zeroes
/// `c` through a different code path than the SIMD tier does.
#[test]
fn matmul_2d_zero_k_on_generic_tier_yields_zero_filled_output() {
    let (m, k, n) = (3usize, 0usize, 6usize);
    let a = Array::<i32>::from_vec(vec![]).reshape(&[m, k]);
    let b = Array::<i32>::from_vec(vec![]).reshape(&[k, n]);

    let c = a.matmul(&b).expect("matmul with k=0 should succeed");
    assert_eq!(c.shape(), vec![m, n]);
    assert_eq!(c.to_vec(), vec![0i32; m * n]);
}

// ---------------------------------------------------------------------
// Non-contiguous (transposed) operands
// ---------------------------------------------------------------------

/// Transposed operands are the case where "read the backing buffer as a
/// flat row-major slice" is *wrong*: a transposed view's memory order is
/// not its logical order. `kernels::borrow::operand` is supposed to notice
/// (via `as_slice()` returning `None`) and materialize a logically-ordered
/// copy instead of borrowing.
///
/// A bug here is silent and produces plausible numbers, so both operands
/// are checked -- separately and together -- against an oracle fed from
/// `to_vec()` (which is independently known to respect strides).
#[test]
fn matmul_2d_handles_non_contiguous_operands() {
    let (m, k, n) = (33usize, 65usize, 31usize);

    // `a` built as its own transpose, then flipped: shape [m, k], non-contiguous.
    let a = Array::from_vec(seq_f64(k * m, 0.25, -2.0))
        .reshape(&[k, m])
        .transpose_axis(0, 1);
    // `b` likewise: shape [k, n], non-contiguous.
    let b = Array::from_vec(seq_f64(n * k, -0.125, 0.75))
        .reshape(&[n, k])
        .transpose_axis(0, 1);

    assert_eq!(a.shape(), vec![m, k]);
    assert_eq!(b.shape(), vec![k, n]);
    assert!(!a.is_c_contiguous(), "test requires a non-contiguous `a`");
    assert!(!b.is_c_contiguous(), "test requires a non-contiguous `b`");

    let expected = naive_gemm_f64(m, k, n, &a.to_vec(), &b.to_vec());

    // Both transposed.
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape(), vec![m, n]);
    for (idx, (got, want)) in c.to_vec().iter().zip(&expected).enumerate() {
        assert_close_rel(*got, *want, 1e-9, format!("both transposed idx={idx}"));
    }

    // Only `a` transposed: `b` re-materialized as a contiguous array with
    // identical logical contents, so the answer must not change.
    let b_contig = Array::from_vec(b.to_vec()).reshape(&[k, n]);
    assert!(b_contig.is_c_contiguous());
    let c_a_only = a.matmul(&b_contig).expect("matmul should succeed");
    for (idx, (got, want)) in c_a_only.to_vec().iter().zip(&expected).enumerate() {
        assert_close_rel(*got, *want, 1e-9, format!("a transposed idx={idx}"));
    }

    // Only `b` transposed.
    let a_contig = Array::from_vec(a.to_vec()).reshape(&[m, k]);
    assert!(a_contig.is_c_contiguous());
    let c_b_only = a_contig.matmul(&b).expect("matmul should succeed");
    for (idx, (got, want)) in c_b_only.to_vec().iter().zip(&expected).enumerate() {
        assert_close_rel(*got, *want, 1e-9, format!("b transposed idx={idx}"));
    }
}

/// The generic tier's non-contiguous path, exactly (integers, so exact
/// equality). Small enough to hand-check the transpose is real.
#[test]
fn matmul_2d_handles_non_contiguous_operands_on_generic_tier() {
    // [[1, 2, 3], [4, 5, 6]] transposed -> [[1, 4], [2, 5], [3, 6]] (3x2)
    let a = Array::from_vec(vec![1i32, 2, 3, 4, 5, 6])
        .reshape(&[2, 3])
        .transpose_axis(0, 1);
    assert_eq!(a.to_vec(), vec![1, 4, 2, 5, 3, 6]);
    assert!(!a.is_c_contiguous());

    // [[1, 0], [0, 1]] identity (2x2)
    let b = Array::from_vec(vec![1i32, 0, 0, 1]).reshape(&[2, 2]);
    let c = a.matmul(&b).expect("matmul should succeed");

    assert_eq!(c.shape(), vec![3, 2]);
    assert_eq!(c.to_vec(), vec![1, 4, 2, 5, 3, 6]);
}

// ---------------------------------------------------------------------
// N-D batched
// ---------------------------------------------------------------------

/// Oracle for a batched matmul: `batch` independent 2-D products, each
/// computed by the naive triple loop, concatenated in panel order.
fn naive_batched_f64(batch: usize, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(batch * m * n);
    for t in 0..batch {
        out.extend(naive_gemm_f64(
            m,
            k,
            n,
            &a[t * m * k..(t + 1) * m * k],
            &b[t * k * n..(t + 1) * k * n],
        ));
    }
    out
}

#[test]
fn matmul_3d_batched_matches_loop_of_2d_oracle() {
    // Includes a batch large enough (with panel size) to clear
    // `GEMM_PARALLEL_MIN_FLOPS` and exercise the parallel batch split,
    // and small ones that stay sequential.
    for &(batch, m, k, n) in &[
        (1usize, 4usize, 5usize, 3usize),
        (3, 8, 7, 6),
        (8, 16, 16, 16),
        (64, 16, 16, 16),
        (5, 33, 65, 31),
    ] {
        let a_data = seq_f64(batch * m * k, 0.0625, -1.25);
        let b_data = seq_f64(batch * k * n, -0.03125, 2.5);
        let expected = naive_batched_f64(batch, m, k, n, &a_data, &b_data);

        let a = Array::from_vec(a_data).reshape(&[batch, m, k]);
        let b = Array::from_vec(b_data).reshape(&[batch, k, n]);
        let c = a.matmul(&b).expect("batched matmul should succeed");

        assert_eq!(
            c.shape(),
            vec![batch, m, n],
            "(batch={batch},m={m},k={k},n={n}) shape"
        );
        for (idx, (got, want)) in c.to_vec().iter().zip(&expected).enumerate() {
            assert_close_rel(
                *got,
                *want,
                1e-9,
                format!("(batch={batch},m={m},k={k},n={n}) idx={idx}"),
            );
        }
    }
}

/// A 4-D batch (two leading axes) checked against the same flat oracle:
/// the batched path flattens *all* leading axes into one panel index, so
/// `[2, 3, m, k]` must give the same answer as `[6, m, k]`.
#[test]
fn matmul_4d_batched_flattens_leading_axes_consistently() {
    let (b0, b1, m, k, n) = (2usize, 3usize, 6usize, 5usize, 4usize);
    let batch = b0 * b1;

    let a_data = seq_f64(batch * m * k, 0.1, -0.5);
    let b_data = seq_f64(batch * k * n, -0.2, 1.0);
    let expected = naive_batched_f64(batch, m, k, n, &a_data, &b_data);

    let a4 = Array::from_vec(a_data.clone()).reshape(&[b0, b1, m, k]);
    let b4 = Array::from_vec(b_data.clone()).reshape(&[b0, b1, k, n]);
    let c4 = a4.matmul(&b4).expect("4-D batched matmul should succeed");
    assert_eq!(c4.shape(), vec![b0, b1, m, n]);

    let a3 = Array::from_vec(a_data).reshape(&[batch, m, k]);
    let b3 = Array::from_vec(b_data).reshape(&[batch, k, n]);
    let c3 = a3.matmul(&b3).expect("3-D batched matmul should succeed");

    assert_eq!(
        c4.to_vec(),
        c3.to_vec(),
        "4-D and 3-D must agree bit-for-bit"
    );
    for (idx, (got, want)) in c4.to_vec().iter().zip(&expected).enumerate() {
        assert_close_rel(*got, *want, 1e-9, format!("4-D idx={idx}"));
    }
}

/// Batched generic tier (`i32`), exact.
#[test]
fn matmul_3d_batched_generic_tier_matches_oracle() {
    let (batch, m, k, n) = (4usize, 5usize, 3usize, 6usize);
    let a_data = seq_i32(batch * m * k, 7, -3);
    let b_data = seq_i32(batch * k * n, 5, -2);

    let mut expected = Vec::with_capacity(batch * m * n);
    for t in 0..batch {
        expected.extend(naive_gemm_i32(
            m,
            k,
            n,
            &a_data[t * m * k..(t + 1) * m * k],
            &b_data[t * k * n..(t + 1) * k * n],
        ));
    }

    let a = Array::from_vec(a_data).reshape(&[batch, m, k]);
    let b = Array::from_vec(b_data).reshape(&[batch, k, n]);
    let c = a.matmul(&b).expect("batched matmul should succeed");

    assert_eq!(c.shape(), vec![batch, m, n]);
    assert_eq!(c.to_vec(), expected);
}

// ---------------------------------------------------------------------
// Broadcast batching
// ---------------------------------------------------------------------

/// `[2, 1, m, k] x [1, 3, k, n] -> [2, 3, m, n]`: neither operand's batch
/// axes match the output's, so *both* get broadcast, and every one of the
/// six output panels pairs a different `(a_i, b_j)`.
///
/// This is the case the flat-index panel arithmetic could get wrong while
/// still looking right on a non-broadcast batch: it relies on
/// `broadcast_to` having already materialized a real, standard-layout
/// `[2, 3, m, k]` array (so panel `t` really is a contiguous run), rather
/// than a stride-0 view. The oracle below indexes the *original* operands
/// by their un-broadcast indices, so it agrees only if the broadcast
/// pairing is right.
#[test]
fn matmul_broadcast_batched_pairs_panels_correctly() {
    let (m, k, n) = (4usize, 3usize, 5usize);

    let a_data = seq_f64(2 * m * k, 0.25, -1.0);
    let b_data = seq_f64(3 * k * n, -0.5, 2.0);

    let a = Array::from_vec(a_data.clone()).reshape(&[2, 1, m, k]);
    let b = Array::from_vec(b_data.clone()).reshape(&[1, 3, k, n]);

    let c = a
        .matmul(&b)
        .expect("broadcast batched matmul should succeed");
    assert_eq!(c.shape(), vec![2, 3, m, n]);

    let got = c.to_vec();
    for i in 0..2 {
        for j in 0..3 {
            let a_panel = &a_data[i * m * k..(i + 1) * m * k];
            let b_panel = &b_data[j * k * n..(j + 1) * k * n];
            let want = naive_gemm_f64(m, k, n, a_panel, b_panel);
            let base = (i * 3 + j) * m * n;
            for (idx, w) in want.iter().enumerate() {
                assert_close_rel(got[base + idx], *w, 1e-9, format!("i={i} j={j} idx={idx}"));
            }
        }
    }
}

/// The mirrored broadcast direction (`[1, 3, ...] x [2, 1, ...]` written as
/// `[3, m, k] x [2, 3, k, n]`, i.e. a *rank*-broadcast where `a` has fewer
/// batch axes than `b`) must pair panels the same way.
#[test]
fn matmul_broadcast_batched_handles_rank_mismatch() {
    let (m, k, n) = (3usize, 4usize, 2usize);

    let a_data = seq_f64(3 * m * k, 0.5, -2.0);
    let b_data = seq_f64(2 * 3 * k * n, -0.25, 1.0);

    let a = Array::from_vec(a_data.clone()).reshape(&[3, m, k]);
    let b = Array::from_vec(b_data.clone()).reshape(&[2, 3, k, n]);

    let c = a.matmul(&b).expect("rank-broadcast matmul should succeed");
    assert_eq!(c.shape(), vec![2, 3, m, n]);

    let got = c.to_vec();
    for i in 0..2 {
        for j in 0..3 {
            // `a`'s single batch axis (length 3) aligns with `b`'s *last*
            // batch axis, so `a` panel `j` pairs with `b` panel `(i, j)`.
            let a_panel = &a_data[j * m * k..(j + 1) * m * k];
            let b_panel = &b_data[(i * 3 + j) * k * n..(i * 3 + j + 1) * k * n];
            let want = naive_gemm_f64(m, k, n, a_panel, b_panel);
            let base = (i * 3 + j) * m * n;
            for (idx, w) in want.iter().enumerate() {
                assert_close_rel(got[base + idx], *w, 1e-9, format!("i={i} j={j} idx={idx}"));
            }
        }
    }
}

// ---------------------------------------------------------------------
// Algebraic invariants (oracle-independent)
// ---------------------------------------------------------------------

/// Identity and associativity hold regardless of which tier ran, so these
/// catch a class of bug a shared-shape oracle could miss.
#[test]
fn matmul_respects_identity_and_associativity() {
    let n = 65usize; // straddles BLOCK_SIZE = 64

    let mut eye = vec![0.0f64; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let identity = Array::from_vec(eye).reshape(&[n, n]);

    let a = Array::from_vec(seq_f64(n * n, 0.01, -0.5)).reshape(&[n, n]);
    let b = Array::from_vec(seq_f64(n * n, -0.02, 0.25)).reshape(&[n, n]);

    // A * I == A
    let ai = a.matmul(&identity).expect("matmul should succeed");
    for (idx, (got, want)) in ai.to_vec().iter().zip(a.to_vec()).enumerate() {
        assert_close_rel(*got, want, 1e-12, format!("A*I idx={idx}"));
    }

    // I * A == A
    let ia = identity.matmul(&a).expect("matmul should succeed");
    for (idx, (got, want)) in ia.to_vec().iter().zip(a.to_vec()).enumerate() {
        assert_close_rel(*got, want, 1e-12, format!("I*A idx={idx}"));
    }

    // (A * B) * I == A * (B * I)
    let left = a
        .matmul(&b)
        .expect("matmul should succeed")
        .matmul(&identity)
        .expect("matmul should succeed");
    let right = a
        .matmul(&b.matmul(&identity).expect("matmul should succeed"))
        .expect("matmul should succeed");
    for (idx, (got, want)) in left.to_vec().iter().zip(right.to_vec()).enumerate() {
        assert_close_rel(*got, want, 1e-9, format!("assoc idx={idx}"));
    }
}

/// Shape validation must still reject incompatible operands rather than
/// reaching the kernel with mismatched `m`/`k`/`n`.
#[test]
fn matmul_rejects_incompatible_inner_dimensions() {
    let a = Array::from_vec(seq_f64(6, 1.0, 0.0)).reshape(&[2, 3]);
    let b = Array::from_vec(seq_f64(8, 1.0, 0.0)).reshape(&[4, 2]);
    assert!(
        a.matmul(&b).is_err(),
        "3 != 4 inner dimensions must be rejected"
    );

    let a3 = Array::from_vec(seq_f64(12, 1.0, 0.0)).reshape(&[2, 2, 3]);
    let b3 = Array::from_vec(seq_f64(16, 1.0, 0.0)).reshape(&[2, 4, 2]);
    assert!(
        a3.matmul(&b3).is_err(),
        "batched 3 != 4 inner dimensions must be rejected"
    );
}

// ---------------------------------------------------------------------
// Performance evidence (contention-robust)
// ---------------------------------------------------------------------

/// Minimum-of-alternating-samples timing harness, used instead of (well,
/// alongside) `bench/matmul_dispatch_benchmark.rs`'s criterion groups.
///
/// # Why a second harness exists
///
/// Criterion reports a confidence interval around the *mean*. On a shared
/// machine -- which is what this migration was measured on, with a dozen
/// concurrent `rustc` processes and a load average above 40 on 8 cores --
/// the mean is dominated by descheduling, and the dispatched path is
/// penalized far more than the sequential baseline because it row-splits
/// across threads. Observed directly: the same 128^3 legacy loop measured
/// 369 us in one criterion run and 1.06 ms in another twenty minutes
/// later, a 2.9x swing with no code change.
///
/// The *minimum* over many alternating samples is the standard robust
/// estimator here: it approximates the least-contended execution, which is
/// the closest available proxy for the quiet-machine number. Alternating
/// A/B/A/B (rather than all-A then all-B) additionally ensures both sides
/// see the same load regime, so even a drifting background load cancels
/// out of the ratio.
///
/// Ignored by default (it is a measurement, not an assertion about
/// correctness, and it takes seconds). Run with:
///
/// ```text
/// cargo nextest run --test test_matmul_dispatch --run-ignored all \
///     matmul_perf_evidence --no-capture
/// ```
///
/// # Acceptance record for the G3 backend cutover (2026-08-24)
///
/// `matmul_perf_evidence_2d`, minimum over separate process invocations
/// (release + fat LTO, `aarch64-apple-darwin`, 8 cores): five "before" runs
/// at load average 29-37, eight "after" runs at load average 10-11. The
/// `dispatched` and `legacy` columns of any one run are measured in the
/// same process at the same moment, so the *speedup* is
/// contention-symmetric; the raw before/after columns are not, and the
/// caveat below says where that matters.
///
/// | m,k,n | before | after | legacy blocked | after vs legacy | per-run range |
/// |---|---|---|---|---|---|
/// | `8^3`        |   291 ns |   166 ns |   416 ns |  2.51x | 2.00-2.76x  |
/// | `32^3`       | 10.54 us |  1.58 us |  7.21 us |  4.55x | 4.53-4.81x  |
/// | `64^3`       | 93.33 us | 11.08 us | 41.67 us |  3.76x | 3.69-3.84x  |
/// | `128^3`      | 213.8 us |  68.2 us | 334.3 us |  4.90x | 4.52-4.96x  |
/// | `256^3`      | 529.3 us | 216.2 us | 2.747 ms | 12.70x | 9.60-12.72x |
/// | `512^3`      | 3.357 ms | 1.278 ms | 24.06 ms | 18.82x | 14.3-18.8x  |
/// | `512x64x512` | 2.190 ms | 219.3 us | 3.030 ms | 13.82x | 10.3-13.8x  |
///
/// **Where the before/after columns are and are not comparable.** The first
/// three rows are below `kernels::GEMM_PARALLEL_MIN_FLOPS`, so both the old
/// and the new backend ran single-threaded there and the in-process
/// `legacy` control moved less than 2% between the two load regimes --
/// those improvements (1.75x, 6.66x, 8.42x) are real and load-independent.
/// The last four rows run the parallel tier, which is load-sensitive on
/// both sides, so their before/after ratio is confounded by the 3x drop in
/// background load between the two measurement sessions. The
/// load-independent evidence for those rows is the `bakeoff` module below,
/// where every candidate is timed in one rotated round-robin under
/// identical load: `nd_gmm_par` beat `simd_par` (the old parallel tier) by
/// 2.66x at `128^3`, 1.45x at `256^3`, 1.86x at `512^3` and 5.90x at
/// `512x64x512`.
///
/// The `legacy` column here is the *faithful* pre-migration loop: it reads
/// its operands through `as_slice()` with a `to_vec()` fallback, as commit
/// `fc464bf` did. An earlier revision of `legacy_matmul_2d_blocked` called
/// `to_vec()` unconditionally, charging the baseline two full operand
/// copies per call that the real code never paid; every speedup measured
/// against that version was inflated, most severely at small sizes.
mod perf {
    use super::*;
    use std::time::{Duration, Instant};

    /// Faithful copy of `matmul_2d`'s pre-migration body (commit
    /// `fc464bf`): a `BLOCK_SIZE = 64` blocked i-k-j triple loop over flat
    /// operand slices. See `bench/matmul_dispatch_benchmark.rs` for the
    /// provenance note.
    ///
    /// Operands are taken through `as_slice()` with a `to_vec()` fallback,
    /// exactly as `fc464bf` did. An earlier revision of this helper called
    /// `to_vec()` unconditionally, which handicapped the "before" side with
    /// two full `m*k`/`k*n` copies per call that the real pre-migration code
    /// never paid -- inflating every reported speedup, and most severely at
    /// the small sizes where a copy is a large fraction of the total. The
    /// numbers this module prints are only an honest baseline with the
    /// zero-copy path restored.
    pub fn legacy_matmul_2d_blocked(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let mut c_data = vec![0.0f64; m * n];
        let owned_a;
        let a_data: &[f64] = match a.as_slice() {
            Some(slice) => slice,
            None => {
                owned_a = a.to_vec();
                &owned_a
            }
        };
        let owned_b;
        let b_data: &[f64] = match b.as_slice() {
            Some(slice) => slice,
            None => {
                owned_b = b.to_vec();
                &owned_b
            }
        };

        const BLOCK_SIZE: usize = 64;
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                    let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                    let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);
                    for i in i_block..i_end {
                        for k_l in k_block..k_end {
                            let a_ik = a_data[i * k + k_l];
                            for j in j_block..j_end {
                                c_data[i * n + j] += a_ik * b_data[k_l * n + j];
                            }
                        }
                    }
                }
            }
        }
        Array::from_vec(c_data).reshape(&[m, n])
    }

    /// Run `reps` alternating A/B samples, returning `(min_a, min_b)`.
    fn alternating_min<A, B, RA, RB>(reps: usize, mut a: A, mut b: B) -> (Duration, Duration)
    where
        A: FnMut() -> RA,
        B: FnMut() -> RB,
    {
        // One untimed pass each, so neither side pays first-touch page
        // faults or cold-cache costs the other has already amortized.
        std::hint::black_box(a());
        std::hint::black_box(b());

        let mut min_a = Duration::MAX;
        let mut min_b = Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            std::hint::black_box(a());
            min_a = min_a.min(t.elapsed());

            let t = Instant::now();
            std::hint::black_box(b());
            min_b = min_b.min(t.elapsed());
        }
        (min_a, min_b)
    }

    fn mat(m: usize, n: usize) -> Array<f64> {
        Array::from_vec(
            (0..m * n)
                .map(|i| (i as f64) * 0.125 - 3.0)
                .collect::<Vec<_>>(),
        )
        .reshape(&[m, n])
    }

    #[test]
    #[ignore = "performance measurement, not a correctness assertion"]
    fn matmul_perf_evidence_2d() {
        println!(
            "\n{:<16} {:>14} {:>14} {:>10}",
            "shape (m,k,n)", "dispatched", "legacy", "speedup"
        );
        for &(m, k, n, reps) in &[
            (8usize, 8usize, 8usize, 2000usize),
            (32, 32, 32, 500),
            (64, 64, 64, 200),
            (128, 128, 128, 100),
            (256, 256, 256, 40),
            (512, 512, 512, 15),
            (512, 64, 512, 30),
        ] {
            let a = mat(m, k);
            let b = mat(k, n);
            let (d, l) = alternating_min(
                reps,
                || a.matmul(&b).expect("matmul should succeed"),
                || legacy_matmul_2d_blocked(&a, &b),
            );
            println!(
                "{:<16} {:>14?} {:>14?} {:>9.2}x",
                format!("{m},{k},{n}"),
                d,
                l,
                l.as_secs_f64() / d.as_secs_f64()
            );
        }
    }

    #[test]
    #[ignore = "performance measurement, not a correctness assertion"]
    fn matmul_perf_evidence_batched() {
        println!(
            "\n{:<20} {:>14} {:>14} {:>10}",
            "batch/panel", "dispatched", "legacy_ixdyn", "speedup"
        );
        for &(batch, panel, reps) in &[
            (1usize, 16usize, 300usize),
            (8, 16, 200),
            (64, 16, 60),
            (1, 64, 60),
            (8, 64, 30),
            (64, 64, 8),
        ] {
            let a = Array::from_vec(
                (0..batch * panel * panel)
                    .map(|i| (i as f64) * 0.125 - 3.0)
                    .collect::<Vec<_>>(),
            )
            .reshape(&[batch, panel, panel]);
            let b = a.clone();

            let (d, l) = alternating_min(
                reps,
                || a.matmul(&b).expect("matmul should succeed"),
                || legacy_batched_ixdyn(&a, &b),
            );
            println!(
                "{:<20} {:>14?} {:>14?} {:>9.2}x",
                format!("batch{batch}/panel{panel}"),
                d,
                l,
                l.as_secs_f64() / d.as_secs_f64()
            );
        }
    }

    /// Faithful copy of the pre-migration N-D batched loop (commit
    /// `fc464bf`): per-element `IxDyn` `get`, and `set()` per output
    /// element.
    fn legacy_batched_ixdyn(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        use scirs2_core::ndarray::IxDyn;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let batch_shape = &a_shape[..a_shape.len() - 2];
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        let mut result = Array::<f64>::zeros(&output_shape);
        let batch_size: usize = batch_shape.iter().product();

        for batch_idx in 0..batch_size {
            let mut batch_indices = Vec::with_capacity(batch_shape.len());
            let mut temp = batch_idx;
            for &dim in batch_shape.iter().rev() {
                batch_indices.insert(0, temp % dim);
                temp /= dim;
            }
            let mut a_indices = batch_indices.clone();
            a_indices.push(0);
            a_indices.push(0);
            let mut b_indices = batch_indices.clone();
            b_indices.push(0);
            b_indices.push(0);

            for i in 0..m {
                let p = a_indices.len() - 2;
                a_indices[p] = i;
                for j in 0..n {
                    let p = b_indices.len() - 1;
                    b_indices[p] = j;
                    let mut sum = 0.0f64;
                    for l in 0..k {
                        let p = a_indices.len() - 1;
                        a_indices[p] = l;
                        let p = b_indices.len() - 2;
                        b_indices[p] = l;
                        sum += a.array().get(IxDyn(&a_indices)).expect("valid index")
                            * b.array().get(IxDyn(&b_indices)).expect("valid index");
                    }
                    let mut out = batch_indices.clone();
                    out.push(i);
                    out.push(j);
                    result.set(&out, sum).expect("valid output index");
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------
// GEMM backend bake-off (G3)
// ---------------------------------------------------------------------

/// Three-way (in practice five-way) bake-off between every GEMM backend
/// `kernels::gemm::gemm_2d` could plausibly dispatch onto, measured at the
/// *kernel* level rather than through `Array::matmul`.
///
/// `src/kernels/` is `pub(crate)`, so this integration-test crate cannot
/// call `gemm_2d`'s internal tiers directly. Each candidate is therefore
/// re-implemented here against the same flat row-major `(m, k, n)` contract
/// `gemm_2d` uses -- `blocked_serial`/`blocked_par` byte-faithful to the
/// `fc464bf` loop, `simd_serial`/`simd_par` byte-faithful to what
/// `gemm.rs` calls today -- so the comparison isolates the kernels and not
/// the `Array` wrapper's allocation and shape plumbing.
///
/// # Fairness rules
///
/// - Every candidate **writes into one preallocated, reused `&mut [f64]`**,
///   with overwrite (`beta = 0`) semantics, so no candidate is charged for
///   an output allocation another avoids. `blas_accelerated` is the one
///   unavoidable exception: its API returns an owned `Array2<F>`, so it
///   pays one `m*n` allocation plus a copy into the shared buffer. That is
///   not a handicap invented here -- it is exactly what `gemm_2d` would
///   have to pay to use it, so it belongs in the measurement.
/// - Candidates are run **round-robin within each repetition, rotating the
///   starting candidate every repetition**, so no candidate systematically
///   occupies a hotter or colder point in the machine's load cycle.
/// - The estimator is the **minimum** over repetitions, never the mean:
///   under an oversubscribed machine the mean measures the scheduler, while
///   the minimum measures the kernel.
mod bakeoff {
    use scirs2_core::ndarray::linalg::general_mat_mul;
    use scirs2_core::ndarray::{ArrayView2, ArrayViewMut2};
    use scirs2_core::parallel_ops::*;
    use std::time::{Duration, Instant};

    // -----------------------------------------------------------------
    // Candidates
    // -----------------------------------------------------------------

    /// Candidate (a): the pre-migration `BLOCK_SIZE = 64` blocked i-k-j
    /// triple loop, monomorphic on `f64` with a plain `+=` accumulate --
    /// byte-faithful to `fc464bf`'s `matmul_2d`, and deliberately *not*
    /// `kernels::gemm::gemm_generic` (which accumulates via
    /// `mem::replace` under a `T: Clone` bound, a form that can inhibit
    /// the auto-vectorization this concrete loop relies on).
    pub fn blocked_serial_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        const BLOCK_SIZE: usize = 64;
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                    let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                    let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);
                    for i in i_block..i_end {
                        for k_l in k_block..k_end {
                            let a_ik = a[i * k + k_l];
                            for j in j_block..j_end {
                                c[i * n + j] += a_ik * b[k_l * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// `f32` twin of [`blocked_serial_f64`].
    pub fn blocked_serial_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        const BLOCK_SIZE: usize = 64;
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, m);
                    let k_end = std::cmp::min(k_block + BLOCK_SIZE, k);
                    let j_end = std::cmp::min(j_block + BLOCK_SIZE, n);
                    for i in i_block..i_end {
                        for k_l in k_block..k_end {
                            let a_ik = a[i * k + k_l];
                            for j in j_block..j_end {
                                c[i * n + j] += a_ik * b[k_l * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Candidate (d): [`blocked_serial_f64`] under the same `M`-only row
    /// split `kernels::gemm::parallel_row_split_f64` already uses. Splitting
    /// rows never touches the `K` reduction, so every `C[i,j]` dot product
    /// is still accumulated in one chunk, in the same order -- only *which
    /// thread* runs it changes.
    pub fn blocked_par_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                blocked_serial_f64(rows, k, n, a_chunk, b, c_chunk);
            });
    }

    /// `f32` twin of [`blocked_par_f64`].
    pub fn blocked_par_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                blocked_serial_f32(rows, k, n, a_chunk, b, c_chunk);
            });
    }

    /// Candidate (b): `scirs2-core`'s blocked SIMD GEMM, single call.
    pub fn simd_serial_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        scirs2_core::simd_ops::simd_matrix_multiply_f64(m, k, n, 1.0, a, b, 0.0, c);
    }

    /// `f32` twin of [`simd_serial_f64`].
    pub fn simd_serial_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        scirs2_core::simd_ops::simd_matrix_multiply_f32(m, k, n, 1.0, a, b, 0.0, c);
    }

    /// Candidate (b'): what `gemm.rs` runs today above
    /// `GEMM_PARALLEL_MIN_FLOPS` -- [`simd_serial_f64`] under an `M`-only
    /// row split.
    pub fn simd_par_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                scirs2_core::simd_ops::simd_matrix_multiply_f64(
                    rows, k, n, 1.0, a_chunk, b, 0.0, c_chunk,
                );
            });
    }

    /// `f32` twin of [`simd_par_f64`].
    pub fn simd_par_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                scirs2_core::simd_ops::simd_matrix_multiply_f32(
                    rows, k, n, 1.0, a_chunk, b, 0.0, c_chunk,
                );
            });
    }

    /// Candidate (c): `scirs2_linalg::blas_accelerated::matmul`, a thin
    /// wrapper over `ndarray`'s `a.dot(b)` (i.e. the pure-Rust
    /// `matrixmultiply` crate), plus the owned-result copy `gemm_2d` would
    /// have to pay to adopt it.
    pub fn blas_acc_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        let a_view = ArrayView2::from_shape((m, k), a).expect("operand shape should match slice");
        let b_view = ArrayView2::from_shape((k, n), b).expect("operand shape should match slice");
        let out = scirs2_linalg::blas_accelerated::matmul(&a_view, &b_view)
            .expect("blas_accelerated matmul should succeed");
        for (dst, src) in c.iter_mut().zip(out.iter()) {
            *dst = *src;
        }
    }

    /// `f32` twin of [`blas_acc_f64`].
    pub fn blas_acc_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        let a_view = ArrayView2::from_shape((m, k), a).expect("operand shape should match slice");
        let b_view = ArrayView2::from_shape((k, n), b).expect("operand shape should match slice");
        let out = scirs2_linalg::blas_accelerated::matmul(&a_view, &b_view)
            .expect("blas_accelerated matmul should succeed");
        for (dst, src) in c.iter_mut().zip(out.iter()) {
            *dst = *src;
        }
    }

    /// Candidate (e): the same `matrixmultiply` kernel `blas_accelerated`
    /// reaches through `a.dot(b)`, but via `ndarray`'s
    /// `linalg::general_mat_mul` -- which writes `alpha*A*B + beta*C`
    /// straight into a caller-owned `C`, so it pays neither the `m*n`
    /// allocation nor the copy-back that `blas_accelerated`'s
    /// owned-`Array2` return forces. Reached through the mandatory
    /// `scirs2_core::ndarray` re-export, not a direct `ndarray` dependency.
    ///
    /// `k == 0` is handled explicitly: with no `K` blocks to iterate,
    /// `matrixmultiply` may leave `C` untouched rather than applying
    /// `beta`, which would break `gemm_2d`'s overwrite contract for a
    /// dirty output buffer.
    pub fn nd_gmm_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if k == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let a_view = ArrayView2::from_shape((m, k), a).expect("operand shape should match slice");
        let b_view = ArrayView2::from_shape((k, n), b).expect("operand shape should match slice");
        let mut c_view =
            ArrayViewMut2::from_shape((m, n), c).expect("output shape should match slice");
        general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c_view);
    }

    /// `f32` twin of [`nd_gmm_f64`].
    pub fn nd_gmm_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if k == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let a_view = ArrayView2::from_shape((m, k), a).expect("operand shape should match slice");
        let b_view = ArrayView2::from_shape((k, n), b).expect("operand shape should match slice");
        let mut c_view =
            ArrayViewMut2::from_shape((m, n), c).expect("output shape should match slice");
        general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c_view);
    }

    /// Candidate (f): [`nd_gmm_f64`] under the same `M`-only row split the
    /// other parallel candidates use. `matrixmultiply` is single-threaded
    /// (its optional `threading` feature is not enabled anywhere in this
    /// tree's lock file), so this is the only way it gets more than one
    /// core.
    pub fn nd_gmm_par_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                nd_gmm_f64(rows, k, n, a_chunk, b, c_chunk);
            });
    }

    /// `f32` twin of [`nd_gmm_par_f64`].
    pub fn nd_gmm_par_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if m == 0 || k == 0 || n == 0 {
            for v in c.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let chunk_rows = m.div_ceil(current_num_threads().max(1)).max(1);
        a.par_chunks(chunk_rows * k)
            .zip(c.par_chunks_mut(chunk_rows * n))
            .for_each(|(a_chunk, c_chunk)| {
                let rows = a_chunk.len() / k;
                if rows == 0 {
                    return;
                }
                nd_gmm_f32(rows, k, n, a_chunk, b, c_chunk);
            });
    }

    // -----------------------------------------------------------------
    // Harness
    // -----------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Backend {
        BlockedSerial,
        BlockedPar,
        SimdSerial,
        SimdPar,
        BlasAcc,
        NdGmm,
        NdGmmPar,
    }

    const BACKENDS: &[Backend] = &[
        Backend::BlockedSerial,
        Backend::BlockedPar,
        Backend::SimdSerial,
        Backend::SimdPar,
        Backend::BlasAcc,
        Backend::NdGmm,
        Backend::NdGmmPar,
    ];

    const BACKEND_NAMES: &[&str] = &[
        "blk_ser",
        "blk_par",
        "simd_ser",
        "simd_par",
        "blas_acc",
        "nd_gmm",
        "nd_gmm_par",
    ];

    fn run_f64(be: Backend, m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        match be {
            Backend::BlockedSerial => blocked_serial_f64(m, k, n, a, b, c),
            Backend::BlockedPar => blocked_par_f64(m, k, n, a, b, c),
            Backend::SimdSerial => simd_serial_f64(m, k, n, a, b, c),
            Backend::SimdPar => simd_par_f64(m, k, n, a, b, c),
            Backend::BlasAcc => blas_acc_f64(m, k, n, a, b, c),
            Backend::NdGmm => nd_gmm_f64(m, k, n, a, b, c),
            Backend::NdGmmPar => nd_gmm_par_f64(m, k, n, a, b, c),
        }
    }

    fn run_f32(be: Backend, m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        match be {
            Backend::BlockedSerial => blocked_serial_f32(m, k, n, a, b, c),
            Backend::BlockedPar => blocked_par_f32(m, k, n, a, b, c),
            Backend::SimdSerial => simd_serial_f32(m, k, n, a, b, c),
            Backend::SimdPar => simd_par_f32(m, k, n, a, b, c),
            Backend::BlasAcc => blas_acc_f32(m, k, n, a, b, c),
            Backend::NdGmm => nd_gmm_f32(m, k, n, a, b, c),
            Backend::NdGmmPar => nd_gmm_par_f32(m, k, n, a, b, c),
        }
    }

    /// Repetition count for a shape: enough repetitions that the fastest
    /// candidate is sampled many times at small sizes (where a single
    /// sample is dominated by timer resolution and one descheduling event),
    /// tapering to a handful at 512-cubed (where one sample is already
    /// milliseconds of uninterruptible work).
    fn reps_for(flops: f64) -> usize {
        let want = 6.0e8 / flops.max(1.0);
        want.clamp(7.0, 1500.0) as usize
    }

    fn seq_f64(len: usize, step: f64, base: f64) -> Vec<f64> {
        (0..len).map(|i| (i as f64) * step + base).collect()
    }

    fn seq_f32(len: usize, step: f32, base: f32) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * step + base).collect()
    }

    /// One shape, every backend, minimum over rotated round-robin reps.
    fn measure_f64(m: usize, k: usize, n: usize) -> Vec<Duration> {
        let a = seq_f64(m * k, 0.0125, -3.0);
        let b = seq_f64(k * n, -0.00625, 1.5);
        let mut c = vec![0.0f64; m * n];

        for &be in BACKENDS {
            run_f64(be, m, k, n, &a, &b, &mut c);
            std::hint::black_box(&c);
        }

        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
        let reps = reps_for(flops);
        let mut best = vec![Duration::MAX; BACKENDS.len()];
        for rep in 0..reps {
            for offset in 0..BACKENDS.len() {
                let idx = (offset + rep) % BACKENDS.len();
                let t = Instant::now();
                run_f64(BACKENDS[idx], m, k, n, &a, &b, &mut c);
                let dt = t.elapsed();
                std::hint::black_box(&c);
                best[idx] = best[idx].min(dt);
            }
        }
        best
    }

    /// `f32` twin of [`measure_f64`].
    fn measure_f32(m: usize, k: usize, n: usize) -> Vec<Duration> {
        let a = seq_f32(m * k, 0.0125, -3.0);
        let b = seq_f32(k * n, -0.00625, 1.5);
        let mut c = vec![0.0f32; m * n];

        for &be in BACKENDS {
            run_f32(be, m, k, n, &a, &b, &mut c);
            std::hint::black_box(&c);
        }

        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
        let reps = reps_for(flops);
        let mut best = vec![Duration::MAX; BACKENDS.len()];
        for rep in 0..reps {
            for offset in 0..BACKENDS.len() {
                let idx = (offset + rep) % BACKENDS.len();
                let t = Instant::now();
                run_f32(BACKENDS[idx], m, k, n, &a, &b, &mut c);
                let dt = t.elapsed();
                std::hint::black_box(&c);
                best[idx] = best[idx].min(dt);
            }
        }
        best
    }

    fn print_header(tag: &str) {
        print!("{tag:<10} {:>16}", "m,k,n");
        for name in BACKEND_NAMES {
            print!(" {name:>12}");
        }
        println!(" {:>12}", "winner");
    }

    fn print_row(tag: &str, m: usize, k: usize, n: usize, best: &[Duration]) {
        print!("{tag:<10} {:>16}", format!("{m},{k},{n}"));
        for d in best {
            print!(" {:>12}", d.as_nanos());
        }
        let mut win = 0usize;
        for (i, d) in best.iter().enumerate() {
            if *d < best[win] {
                win = i;
            }
        }
        println!(" {:>12}", BACKEND_NAMES[win]);
    }

    // -----------------------------------------------------------------
    // Correctness gate: every candidate must agree with a naive oracle
    // before any of its timings are trusted.
    // -----------------------------------------------------------------

    fn naive(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }
        c
    }

    /// Checks every bake-off candidate -- including
    /// `blas_accelerated`'s non-square and `k = 0` behaviour, which must be
    /// verified *before* its timings can justify a dispatch tier -- against
    /// an independent naive triple loop.
    ///
    /// Not `#[ignore]`d: it is a correctness assertion, and it is cheap.
    #[test]
    fn every_bakeoff_candidate_matches_naive_oracle() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (5, 0, 4), // k = 0: output must be all zeros
            (0, 5, 4), // m = 0: empty output
            (5, 4, 0), // n = 0: empty output
            (3, 7, 5), // non-square, all dims small
            (33, 17, 41),
            (64, 64, 64),
            (65, 32, 129),
            (128, 16, 96),
        ] {
            let a = seq_f64(m * k, 0.0125, -3.0);
            let b = seq_f64(k * n, -0.00625, 1.5);
            let expected = naive(m, k, n, &a, &b);
            for (idx, &be) in BACKENDS.iter().enumerate() {
                let mut c = vec![f64::NAN; m * n];
                run_f64(be, m, k, n, &a, &b, &mut c);
                for (i, (&got, &want)) in c.iter().zip(expected.iter()).enumerate() {
                    let scale = want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= 1e-9 * scale,
                        "{} at (m={m},k={k},n={n}) idx {i}: got {got}, want {want}",
                        BACKEND_NAMES[idx]
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // The sweeps
    // -----------------------------------------------------------------

    /// Square sizes, `f64`, spanning every tier boundary in play.
    #[test]
    #[ignore = "performance measurement, not a correctness assertion"]
    fn bakeoff_square_f64() {
        print_header("SQ64");
        for &s in &[
            8usize, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 320, 384, 512,
        ] {
            let best = measure_f64(s, s, s);
            print_row("SQ64", s, s, s, &best);
        }
    }

    /// `k`-sweep at fixed `m = n = 512`, plus the two other rectangular
    /// shapes named in the task. Isolates the small-`k` weakness: total
    /// FLOPs vary with `k`, so compare each row against the square row of
    /// the same FLOP count in `bakeoff_square_f64`, not against each other.
    #[test]
    #[ignore = "performance measurement, not a correctness assertion"]
    fn bakeoff_rectangular_f64() {
        print_header("RECT64");
        for &k in &[8usize, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512] {
            let best = measure_f64(512, k, 512);
            print_row("RECT64", 512, k, 512, &best);
        }
        for &(m, k, n) in &[
            (64usize, 512usize, 64usize),
            (256, 32, 256),
            (1024, 32, 1024),
        ] {
            let best = measure_f64(m, k, n);
            print_row("RECT64", m, k, n, &best);
        }
    }

    /// `f32` spot checks. `f32`'s micro-kernel geometry in `scirs2-core`
    /// differs from `f64`'s (a NEON vector holds 4 `f32` but 2 `f64`), so
    /// the `f64` cutovers cannot simply be assumed to transfer.
    #[test]
    #[ignore = "performance measurement, not a correctness assertion"]
    fn bakeoff_f32() {
        print_header("F32");
        for &s in &[32usize, 48, 64, 96, 128, 192, 256, 384] {
            let best = measure_f32(s, s, s);
            print_row("F32", s, s, s, &best);
        }
        for &(m, k, n) in &[(512usize, 64usize, 512usize), (512, 32, 512)] {
            let best = measure_f32(m, k, n);
            print_row("F32", m, k, n, &best);
        }
    }
}

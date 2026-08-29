//! Equivalence gate for the owned expression templates
//! (`src/expr/owned.rs` + `src/expr/fused_eval.rs`).
//!
//! The claim under test: for a randomly shaped expression tree over randomly
//! shaped operands, `ExprNode::eval()` returns **exactly** what the same
//! expression written with the crate's ordinary eager operators returns --
//! bit for bit, signed zeros and infinities included -- whether `eval()` took
//! the single-pass fused path or fell back to eager evaluation.
//!
//! The comparison is strict for every **finite, infinite and signed-zero**
//! element: those must match bit for bit, so `0.0` never passes for `-0.0`
//! and a 1-ulp slip is a failure. A `NaN` is compared *positionally*: both
//! sides must be `NaN` in the same element, but a `NaN`'s **payload and sign
//! bits are deliberately outside the claim**.
//!
//! That exception is narrow, and it is measured rather than assumed. When one
//! operation receives two *distinct* `NaN` operands, IEEE-754 §6.2.3 leaves
//! which payload propagates implementation-defined, and LLVM treats `fadd` /
//! `fmul` as commutative -- so the single fused `zip` loop and the two-pass
//! eager spelling can each keep a different operand's `NaN`, from identical
//! source-level operand order. Reduced to a 30-line standalone program
//! containing no `numrs2` code at all, `(0.0 * inf) + NaN` gives:
//!
//! | build | single fused loop | two-pass eager | agree? |
//! |---|---|---|---|
//! | `rustc -O`  | `0x7ff8000000000000` | `0xfff8000000000000` | no |
//! | `rustc -O0` | `0xfff8000000000000` | `0xfff8000000000000` | yes |
//!
//! So the divergence is an artifact of optimization that IEEE-754 explicitly
//! permits, not a defect in the evaluator -- and fixing it in the evaluator
//! would mean giving up the vectorization the fused module exists for. NumPy
//! makes no `NaN`-payload promise either.
//! `two_distinct_nans_into_one_add_may_differ_in_payload` at the bottom of
//! this file pins that counter-example (it was found by this very property,
//! at proptest seed 12825631960650228096);
//! `nan_payload_case_that_used_to_diverge` beside it keeps the older input
//! that caught a real defect back when the evaluator had one.
//!
//! Note on naming: `eval()` here is `ExprNode::eval`, a numeric array
//! evaluator; nothing in this file interprets code or strings.
//!
//! # How the trees are generated
//!
//! `proptest` supplies a `u64` seed plus a few shape knobs; a deterministic
//! SplitMix64 generator then builds, in one recursive walk, **both**
//!
//! * an `ExprNode<f64>` to hand to `eval()`, and
//! * the eager reference value, computed as the recursion unwinds with
//!   `&a + &b`, `&a * s`, `v.map(..)` and friends -- i.e. exactly the code a
//!   user would have written by hand for that tree.
//!
//! The reference is therefore *not* the evaluator's own private `eval_eager`,
//! which would make the fallback half of the test circular.
//!
//! Trees are generated in four leaf modes, chosen per tree, so both evaluation
//! paths get hit:
//!
//! | mode | leaves | path |
//! |---|---|---|
//! | `Contiguous` | same shape, standard layout | always fused |
//! | `Transposed` | same shape, all transposed views | always the fallback (once both extents exceed 1; a `[1, n]` or `[1, 1]` leaf transposes to something still in standard layout) |
//! | `MixedLayout` | per-leaf coin flip between transposed and contiguous | usually the fallback -- an all-contiguous draw legitimately fuses |
//! | `Broadcast` | per-leaf coin flip between `[1, c]` and `[r, 1]` | usually the fallback -- a draw that happens to pick one orientation for every leaf is same-shape and contiguous, so it legitimately fuses (measured: ~55% of 300 seeds at `[3, 5]` fused) |
//!
//! Rather than lean on those tendencies, `check_one` asserts the exact routing
//! rule per tree: `will_fuse()` must be true precisely when every leaf shares
//! one shape and is `as_slice`-able. The modes exist to make both outcomes
//! common, not to be the oracle.
//!
//! A generated tree has depth at most 4 and can contain every `ExprNode`
//! variant.
//!
//! **`f32` coverage is fused-path only.** The `f32` trees are produced by
//! casting an `f64` tree with `to_f32`, and that cast goes through
//! `Array::map`, which always returns a fresh contiguous array. So the `f32`
//! property test exercises the fused `f32` kernels thoroughly but never its
//! non-contiguous fallback. That is a deliberate gap: the fallback
//! (`eval_eager`) is generic in `T` and shares no code with the `f32`
//! kernels, and the `f64` tests cover it at all four modes.

#![allow(clippy::result_large_err)]

use numrs2::expr::{BinOp, ExprNode, UnaryOp};
use numrs2::prelude::*;
use proptest::prelude::*;

// `numrs2::prelude` also exports an `any` (the array reduction), so the two
// glob imports above make the name ambiguous. An explicit import wins over
// both globs and pins it to proptest's strategy constructor.
use proptest::arbitrary::any;

// ---------------------------------------------------------------------------
// Deterministic generator
// ---------------------------------------------------------------------------

/// SplitMix64: a tiny, deterministic, reproducible-from-a-seed PRNG.
///
/// Deliberately not `scirs2_core::random`: the point here is that a failing
/// case is replayable from the printed seed alone, independent of any
/// generator's internal state layout or version.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// In `0 .. n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn boolean(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// The element pool: mostly ordinary values so the arithmetic stays
/// meaningful, plus every awkward IEEE-754 value, because bitwise equality is
/// the claim being tested.
const POOL: [f64; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.25,
    3.0,
    -7.5,
    1e-300,
    1e300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MIN_POSITIVE,
    2.0,
    -2.0,
];

/// Which leaf layout a whole tree is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Same shape, standard layout: the fused path.
    Contiguous,
    /// Same shape, every leaf a transposed (non-contiguous) view.
    Transposed,
    /// Same shape, roughly half the leaves transposed.
    MixedLayout,
    /// `[1, c]` / `[r, 1]` leaves: broadcasting, so the fallback.
    Broadcast,
}

impl Mode {
    fn from_index(i: usize) -> Mode {
        match i % 4 {
            0 => Mode::Contiguous,
            1 => Mode::Transposed,
            2 => Mode::MixedLayout,
            _ => Mode::Broadcast,
        }
    }
}

/// Shared generation context: the PRNG plus the shape all leaves derive from.
struct Ctx {
    rng: Rng,
    rows: usize,
    cols: usize,
    mode: Mode,
}

impl Ctx {
    fn value(&mut self) -> f64 {
        POOL[self.rng.below(POOL.len())]
    }

    /// A fresh leaf array in this context's mode.
    ///
    /// Every mode yields leaves that combine to the tree's common
    /// `[rows, cols]` shape, so no generated tree can fail on shape grounds.
    fn leaf(&mut self) -> Array<f64> {
        let (rows, cols) = (self.rows, self.cols);
        match self.mode {
            Mode::Contiguous => self.contiguous(rows, cols),
            Mode::Transposed => self.transposed(rows, cols),
            Mode::MixedLayout => {
                if self.rng.boolean() {
                    self.transposed(rows, cols)
                } else {
                    self.contiguous(rows, cols)
                }
            }
            Mode::Broadcast => {
                if self.rng.boolean() {
                    self.contiguous(1, cols)
                } else {
                    self.contiguous(rows, 1)
                }
            }
        }
    }

    fn contiguous(&mut self, rows: usize, cols: usize) -> Array<f64> {
        let data: Vec<f64> = (0..rows * cols).map(|_| self.value()).collect();
        Array::from_vec(data).reshape(&[rows, cols])
    }

    /// A `[rows, cols]`-shaped **view** whose memory order is column-major, so
    /// `as_slice()` returns `None` and the tree cannot fuse.
    fn transposed(&mut self, rows: usize, cols: usize) -> Array<f64> {
        let base = self.contiguous(cols, rows);
        base.transpose_axis(0, 1)
    }

    fn bin_op(&mut self) -> BinOp {
        match self.rng.below(4) {
            0 => BinOp::Add,
            1 => BinOp::Sub,
            2 => BinOp::Mul,
            _ => BinOp::Div,
        }
    }

    fn unary_op(&mut self) -> UnaryOp {
        match self.rng.below(5) {
            0 => UnaryOp::Neg,
            1 => UnaryOp::Abs,
            2 => UnaryOp::Sqrt,
            3 => UnaryOp::Exp,
            _ => UnaryOp::Ln,
        }
    }
}

/// Build a tree of depth at most `budget + 1`, together with its eager value.
///
/// The second element of the pair is computed with the crate's **public eager
/// operators** as the recursion unwinds; it never touches `ExprNode::eval`.
fn build(ctx: &mut Ctx, budget: usize) -> (ExprNode<f64>, Array<f64>) {
    if budget == 0 {
        let a = ctx.leaf();
        let value = a.clone();
        return (ExprNode::Leaf(a), value);
    }

    match ctx.rng.below(6) {
        // Leaf
        0 => {
            let a = ctx.leaf();
            let value = a.clone();
            (ExprNode::Leaf(a), value)
        }
        // Binary
        1 => {
            let op = ctx.bin_op();
            let (ln, lv) = build(ctx, budget - 1);
            let (rn, rv) = build(ctx, budget - 1);
            let value = match op {
                BinOp::Add => &lv + &rv,
                BinOp::Sub => &lv - &rv,
                BinOp::Mul => &lv * &rv,
                BinOp::Div => &lv / &rv,
            };
            (ExprNode::Binary(op, Box::new(ln), Box::new(rn)), value)
        }
        // ScalarRhs
        2 => {
            let op = ctx.bin_op();
            let k = ctx.value();
            let (n, v) = build(ctx, budget - 1);
            let value = match op {
                BinOp::Add => &v + k,
                BinOp::Sub => &v - k,
                BinOp::Mul => &v * k,
                BinOp::Div => &v / k,
            };
            (ExprNode::ScalarRhs(op, Box::new(n), k), value)
        }
        // ScalarLhs
        3 => {
            let op = ctx.bin_op();
            let k = ctx.value();
            let (n, v) = build(ctx, budget - 1);
            let value = match op {
                BinOp::Add => v.map(|x| k + x),
                BinOp::Sub => v.map(|x| k - x),
                BinOp::Mul => v.map(|x| k * x),
                BinOp::Div => v.map(|x| k / x),
            };
            (ExprNode::ScalarLhs(op, k, Box::new(n)), value)
        }
        // Unary
        4 => {
            let op = ctx.unary_op();
            let (n, v) = build(ctx, budget - 1);
            let value = match op {
                UnaryOp::Neg => -&v,
                UnaryOp::Abs => v.map(f64::abs),
                UnaryOp::Sqrt => v.map(f64::sqrt),
                UnaryOp::Exp => v.map(f64::exp),
                UnaryOp::Ln => v.map(f64::ln),
            };
            (ExprNode::Unary(op, Box::new(n)), value)
        }
        // Fma -- `(a * b) + c`, two roundings, never `mul_add`
        _ => {
            let (an, av) = build(ctx, budget - 1);
            let (bn, bv) = build(ctx, budget - 1);
            let (cn, cv) = build(ctx, budget - 1);
            let value = &(&av * &bv) + &cv;
            (
                ExprNode::Fma(Box::new(an), Box::new(bn), Box::new(cn)),
                value,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Array comparison at exactly the strength the evaluator guarantees.
///
/// Every **finite, infinite and signed-zero** element must match **bit for
/// bit**, so `0.0` never passes for `-0.0` and a 1-ulp difference is a
/// failure. The one exception is a `NaN`: when *both* sides are `NaN` in the
/// same element, differing payload and sign bits are tolerated, for the
/// IEEE-754 §6.2.3 / LLVM-commutativity reason set out in this file's module
/// docs. A `NaN` facing a number is still a failure, and so is a `NaN` that
/// turns up in an element where the other side has none -- *where* the `NaN`s
/// are remains part of the claim.
///
/// Returns how many comparisons took that exemption, so it is reported rather
/// than swallowed; `#[must_use]` forces every caller to decide what to do
/// with the number instead of dropping it silently.
#[must_use]
fn assert_values_eq(got: &Array<f64>, want: &Array<f64>, what: &str) -> usize {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (g, w) = (got.to_vec(), want.to_vec());
    let mut payload_exempt = 0usize;
    for (i, (x, y)) in g.iter().zip(w.iter()).enumerate() {
        if x.to_bits() == y.to_bits() {
            continue;
        }
        if x.is_nan() && y.is_nan() {
            payload_exempt += 1;
            continue;
        }
        // Not both `NaN`, so the full-strength comparison stands. Re-stating
        // it as the original `assert_eq!` keeps the failure message -- and
        // both raw bit patterns -- exactly as it always was.
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: element {i}: fused {x} vs eager {y}"
        );
    }
    payload_exempt
}

/// Same, for `f32`: strict everywhere except a `NaN` facing a `NaN`, and
/// likewise returns the number of tolerated payload divergences.
#[must_use]
fn assert_values_eq_f32(got: &Array<f32>, want: &Array<f32>, what: &str) -> usize {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (g, w) = (got.to_vec(), want.to_vec());
    let mut payload_exempt = 0usize;
    for (i, (x, y)) in g.iter().zip(w.iter()).enumerate() {
        if x.to_bits() == y.to_bits() {
            continue;
        }
        if x.is_nan() && y.is_nan() {
            payload_exempt += 1;
            continue;
        }
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: element {i}: fused {x} vs eager {y}"
        );
    }
    payload_exempt
}

/// Every leaf array in a tree, left to right.
///
/// Re-implemented here rather than reusing the crate-internal
/// `ExprNode::collect_leaves`, so the `will_fuse` contract below is checked
/// against an independent walk of the public enum.
fn leaf_arrays(node: &ExprNode<f64>) -> Vec<&Array<f64>> {
    let mut out = Vec::new();
    fn go<'a>(node: &'a ExprNode<f64>, out: &mut Vec<&'a Array<f64>>) {
        match node {
            ExprNode::Leaf(a) => out.push(a),
            ExprNode::Binary(_, l, r) => {
                go(l, out);
                go(r, out);
            }
            ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => go(e, out),
            ExprNode::Unary(_, e) => go(e, out),
            ExprNode::Fma(a, b, c) => {
                go(a, out);
                go(b, out);
                go(c, out);
            }
        }
    }
    go(node, &mut out);
    out
}

/// The fusion precondition exactly as `src/expr/owned.rs` documents it, both
/// halves: every leaf the same shape and `as_slice`-able (the dtype half is
/// fixed at `f64` here), **and** a tree shape the specialised loops cover.
/// Everything else evaluates through the eager fallback.
fn should_fuse(node: &ExprNode<f64>) -> bool {
    fn operands_ok(node: &ExprNode<f64>) -> bool {
        let leaves = leaf_arrays(node);
        match leaves.first() {
            None => false,
            Some(first) => {
                let shape = first.shape();
                leaves
                    .iter()
                    .all(|l| l.shape() == shape && l.as_slice().is_some())
            }
        }
    }

    fn is_leaf(node: &ExprNode<f64>) -> bool {
        matches!(node, ExprNode::Leaf(_))
    }

    /// Hand-written twin of `fused_eval::is_specialized_shape`, so the
    /// routing rule is checked against an independent statement of it.
    fn shape_ok(node: &ExprNode<f64>) -> bool {
        match node {
            ExprNode::Leaf(_) => true,
            ExprNode::Binary(_, l, r) => {
                (is_leaf(l) && is_leaf(r))
                    || (matches!(&**l, ExprNode::Binary(_, a, b) if is_leaf(a) && is_leaf(b))
                        && is_leaf(r))
                    || (is_leaf(l)
                        && matches!(&**r, ExprNode::Binary(_, a, b) if is_leaf(a) && is_leaf(b)))
                    || (matches!(&**l, ExprNode::ScalarRhs(_, e, _) if is_leaf(e)) && is_leaf(r))
                    || (is_leaf(l) && matches!(&**r, ExprNode::ScalarRhs(_, e, _) if is_leaf(e)))
                    || matches!(
                        (&**l, &**r),
                        (ExprNode::Binary(_, a, b), ExprNode::Binary(_, c, d))
                            if is_leaf(a) && is_leaf(b) && is_leaf(c) && is_leaf(d)
                    )
            }
            ExprNode::Fma(a, b, c) => is_leaf(a) && is_leaf(b) && is_leaf(c),
            ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => is_leaf(e),
            ExprNode::Unary(_, e) => is_leaf(e),
        }
    }

    operands_ok(node) && shape_ok(node)
}

/// Generate one tree, evaluate it both ways, assert they agree.
///
/// Also asserts the *routing* contract: `will_fuse()` is true exactly when
/// every leaf shares one shape and is in standard layout.
///
/// Returns `(took_the_fused_path, nan_payload_exemptions)` -- the first so
/// callers can prove they exercised both paths, the second so the narrow
/// `NaN`-payload exemption `assert_values_eq` grants stays countable at the
/// call site rather than vanishing inside the helper.
fn check_one(
    seed: u64,
    rows: usize,
    cols: usize,
    mode: Mode,
    depth: usize,
) -> Result<(bool, usize)> {
    let mut ctx = Ctx {
        rng: Rng::new(seed),
        rows,
        cols,
        mode,
    };
    let (node, eager) = build(&mut ctx, depth);
    let what = format!(
        "seed={seed} mode={mode:?} shape=[{rows},{cols}] depth<={}",
        depth + 1
    );

    let fused_path = node.will_fuse();
    assert_eq!(
        fused_path,
        should_fuse(&node),
        "{what}: will_fuse() disagrees with the documented precondition"
    );

    let got = node.eval()?;
    let payload_exempt = assert_values_eq(&got, &eager, &what);
    Ok((fused_path, payload_exempt))
}

// ---------------------------------------------------------------------------
// f32: an independently hand-written eager reference
// ---------------------------------------------------------------------------

/// Rebuild an `f64` tree as an `f32` tree, leaf data cast elementwise.
fn to_f32(node: &ExprNode<f64>) -> ExprNode<f32> {
    match node {
        ExprNode::Leaf(a) => ExprNode::Leaf(a.map(|x| x as f32)),
        ExprNode::Binary(op, l, r) => {
            ExprNode::Binary(*op, Box::new(to_f32(l)), Box::new(to_f32(r)))
        }
        ExprNode::ScalarRhs(op, e, s) => ExprNode::ScalarRhs(*op, Box::new(to_f32(e)), *s as f32),
        ExprNode::ScalarLhs(op, s, e) => ExprNode::ScalarLhs(*op, *s as f32, Box::new(to_f32(e))),
        ExprNode::Unary(op, e) => ExprNode::Unary(*op, Box::new(to_f32(e))),
        ExprNode::Fma(a, b, c) => ExprNode::Fma(
            Box::new(to_f32(a)),
            Box::new(to_f32(b)),
            Box::new(to_f32(c)),
        ),
    }
}

/// Evaluate an `f32` tree with the crate's public eager operators only.
///
/// Written out by hand here rather than reusing anything in `fused_eval`, so
/// the `f32` fused kernels are checked against genuinely independent code.
fn eager_ref_f32(node: &ExprNode<f32>) -> Array<f32> {
    match node {
        ExprNode::Leaf(a) => a.clone(),
        ExprNode::Binary(op, l, r) => {
            let (lv, rv) = (eager_ref_f32(l), eager_ref_f32(r));
            match op {
                BinOp::Add => &lv + &rv,
                BinOp::Sub => &lv - &rv,
                BinOp::Mul => &lv * &rv,
                BinOp::Div => &lv / &rv,
            }
        }
        ExprNode::ScalarRhs(op, e, k) => {
            let v = eager_ref_f32(e);
            match op {
                BinOp::Add => &v + *k,
                BinOp::Sub => &v - *k,
                BinOp::Mul => &v * *k,
                BinOp::Div => &v / *k,
            }
        }
        ExprNode::ScalarLhs(op, k, e) => {
            let (v, k) = (eager_ref_f32(e), *k);
            match op {
                BinOp::Add => v.map(|x| k + x),
                BinOp::Sub => v.map(|x| k - x),
                BinOp::Mul => v.map(|x| k * x),
                BinOp::Div => v.map(|x| k / x),
            }
        }
        ExprNode::Unary(op, e) => {
            let v = eager_ref_f32(e);
            match op {
                UnaryOp::Neg => -&v,
                UnaryOp::Abs => v.map(f32::abs),
                UnaryOp::Sqrt => v.map(f32::sqrt),
                UnaryOp::Exp => v.map(f32::exp),
                UnaryOp::Ln => v.map(f32::ln),
            }
        }
        ExprNode::Fma(a, b, c) => {
            let (av, bv, cv) = (eager_ref_f32(a), eager_ref_f32(b), eager_ref_f32(c));
            &(&av * &bv) + &cv
        }
    }
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// The gate: any tree, any leaf mode, fused or fallback, matches the
    /// hand-written eager form bit for bit.
    #[test]
    fn fused_eval_matches_eager_bitwise(
        seed in any::<u64>(),
        rows in 1usize..=5,
        cols in 1usize..=7,
        mode_idx in 0usize..4,
        depth in 0usize..=3,
    ) {
        let mode = Mode::from_index(mode_idx);
        // The per-case exemption count has nowhere to accumulate -- proptest
        // cases are independent runs of this closure -- so the aggregate
        // tally lives in `strict_bitwise_sweep_covers_nan_results_too`, which
        // sweeps the same generator over a fixed set of seeds and prints it.
        let (fused, _payload_exempt) = check_one(seed, rows, cols, mode, depth)
            .expect("every generated tree has broadcast-compatible leaves");

        // The mode/path correspondence the module docs promise. Only the two
        // absolute directions are asserted: contiguous same-shape leaves
        // always fuse, and a genuinely transposed leaf never does. A
        // degenerate `[1, n]` / `[n, 1]` / `[1, 1]` shape transposes to
        // something that is *still* standard layout, so `Transposed` only
        // guarantees non-contiguity once both extents exceed 1.
        // `check_one` already asserted the exact routing rule; all that is
        // left here is the one direction that holds unconditionally.
        if mode == Mode::Transposed && rows > 1 && cols > 1 {
            prop_assert!(!fused, "a transposed leaf can never fuse");
        }
    }

    /// Same property for `f32`, against an independently written eager
    /// reference (`eager_ref_f32`).
    #[test]
    fn fused_eval_matches_eager_bitwise_f32(
        seed in any::<u64>(),
        rows in 1usize..=4,
        cols in 1usize..=6,
        mode_idx in 0usize..4,
        depth in 0usize..=3,
    ) {
        let mode = Mode::from_index(mode_idx);
        let mut ctx = Ctx { rng: Rng::new(seed), rows, cols, mode };
        let (node64, _) = build(&mut ctx, depth);
        let node = to_f32(&node64);

        let want = eager_ref_f32(&node);
        let got = node.eval().expect("broadcast-compatible leaves");
        // Same reason as above: nothing to accumulate into across independent
        // proptest cases, so the count is deliberately dropped here.
        let _payload_exempt = assert_values_eq_f32(
            &got,
            &want,
            &format!("f32 seed={seed} mode={mode:?} shape=[{rows},{cols}]"),
        );

        // No path assertion here: `to_f32` rebuilds every leaf through
        // `Array::map`, which returns a fresh contiguous array, so the leaf
        // *layout* the mode selected does not survive the cast. See the
        // module docs' "f32 coverage is fused-path only" note.
    }
}

// ---------------------------------------------------------------------------
// Seeded sweeps that proptest's deliberately tiny shapes would not reach
// ---------------------------------------------------------------------------

/// Sweeps a handful of sizes -- including 1023/1024/1025/2049, which were
/// meaningful block-boundary values under an earlier "interpret the tree in
/// 1024-element blocks" design. That third tier was measured against the
/// current design and lost (see `expr::fused_eval`'s module docs) and was
/// removed in favor of today's two-tier engine: a fixed set of specialized
/// fused shapes evaluated via monomorphic whole-array `zip` loops, falling
/// back to `eval_eager` for every other tree. Neither tier chunks, so there
/// is no longer a "boundary" here in the original sense -- but sweeping this
/// same range of sizes (including the tiny `cols=1` case) remains a
/// reasonable general guard against size-dependent indexing bugs in either
/// tier, so the sweep is kept rather than dropped.
#[test]
fn varied_size_sweep_matches_eager() -> Result<()> {
    let mut fused_seen = 0usize;
    let mut total = 0usize;
    let mut payload_exempt = 0usize;
    for &cols in &[1usize, 1023, 1024, 1025, 2049] {
        for seed in 0u64..24 {
            total += 1;
            let (fused, exempt) = check_one(seed, 1, cols, Mode::Contiguous, 3)?;
            payload_exempt += exempt;
            if fused {
                fused_seen += 1;
            }
        }
    }
    println!("varied-size sweep: NaN-payload exemptions taken: {payload_exempt}");
    assert!(
        fused_seen * 4 >= total,
        "expected the contiguous sweep to reach the fused path often, got {fused_seen}/{total}"
    );
    Ok(())
}

/// Every non-fusing mode, swept over many seeds, still matches eager -- and
/// really does drive `eval()` into the fallback.
///
/// Note the deliberate asymmetry: `Transposed` at a shape with both extents
/// above 1 can *never* fuse, so that is an equality. `MixedLayout` and
/// `Broadcast` can by chance draw a tree whose leaves happen to be all
/// contiguous (`MixedLayout`) or all the same orientation (`Broadcast`), and
/// such a tree legitimately fuses -- `check_one` already pins the exact
/// routing rule per tree, so all this sweep has to show is that the fallback
/// is genuinely reached in bulk.
#[test]
fn fallback_modes_match_eager_and_reach_the_fallback() -> Result<()> {
    for mode in [Mode::Transposed, Mode::MixedLayout, Mode::Broadcast] {
        let mut fell_back = 0usize;
        let mut total = 0usize;
        let mut payload_exempt = 0usize;
        for seed in 0u64..300 {
            let (fused, exempt) = check_one(seed, 3, 5, mode, 3)?;
            payload_exempt += exempt;
            total += 1;
            if !fused {
                fell_back += 1;
            }
        }
        println!("{mode:?}: NaN-payload exemptions taken: {payload_exempt}");
        match mode {
            Mode::Transposed => assert_eq!(
                fell_back, total,
                "{mode:?}: every tree at [3,5] must take the eager fallback"
            ),
            _ => assert!(
                fell_back * 5 >= total,
                "{mode:?}: expected the fallback to be reached in bulk, got {fell_back}/{total}"
            ),
        }
    }
    Ok(())
}

/// Depth-4 trees specifically (the gate's stated bound), all four leaf modes.
#[test]
fn depth_four_trees_match_eager() -> Result<()> {
    let mut payload_exempt = 0usize;
    for mode_idx in 0..4 {
        let mode = Mode::from_index(mode_idx);
        for seed in 1000u64..1200 {
            let (_fused, exempt) = check_one(seed, 4, 6, mode, 3)?;
            payload_exempt += exempt;
        }
    }
    println!("depth-4 sweep: NaN-payload exemptions taken: {payload_exempt}");
    Ok(())
}

/// `fuse_fma` must be a *value-preserving* rewrite: rewriting a tree and
/// evaluating it gives exactly what evaluating the un-rewritten tree gives,
/// which is exactly the eager value.
#[test]
fn fuse_fma_rewrite_preserves_values_bitwise() -> Result<()> {
    let mut payload_exempt = 0usize;
    for mode_idx in 0..4 {
        let mode = Mode::from_index(mode_idx);
        for seed in 5000u64..5200 {
            let mut ctx = Ctx {
                rng: Rng::new(seed),
                rows: 3,
                cols: 7,
                mode,
            };
            let (node, eager) = build(&mut ctx, 3);
            let plain = node.clone().eval()?;
            let rewritten = node.fuse_fma().eval()?;
            payload_exempt +=
                assert_values_eq(&plain, &eager, &format!("plain seed={seed} mode={mode:?}"));
            payload_exempt += assert_values_eq(
                &rewritten,
                &eager,
                &format!("fuse_fma seed={seed} mode={mode:?}"),
            );
        }
    }
    println!("fuse_fma sweep: NaN-payload exemptions taken: {payload_exempt}");
    Ok(())
}

/// The headline example from the module docs, at a spread of sizes --
/// 999/1024/4096 were meaningful under an earlier "interpret the tree in
/// 1024-element blocks" design's chunk boundary. That design was removed in
/// favor of today's two-tier engine (a fixed set of specialized fused
/// shapes evaluated via whole-array `zip` loops, falling back to
/// `eval_eager` otherwise -- see `varied_size_sweep_matches_eager` above and
/// `expr::fused_eval`'s module docs), so there is no longer a boundary at
/// those sizes to cross. The spread (including `n=1` and a large
/// `n=100_000`) remains a reasonable general size sweep for the canonical
/// `a + b * c` chain against the exact eager spelling users write today.
#[test]
fn canonical_chain_matches_eager_at_scale() -> Result<()> {
    for n in [1usize, 999, 1024, 4096, 100_000] {
        let a = Array::from_vec((0..n).map(|i| i as f64 * 0.5 - 3.0).collect());
        let b = Array::from_vec((0..n).map(|i| i as f64 * -0.125 + 1.0).collect());
        let c = Array::from_vec((0..n).map(|i| 1.0 / (i as f64 + 0.5)).collect());

        let e = a.expr() + b.expr() * c.expr();
        assert!(e.will_fuse(), "n={n}");
        let payload_exempt =
            assert_values_eq(&e.eval()?, &(&a + &(&b * &c)), &format!("a+b*c n={n}"));
        // Every leaf here is finite, so no operation can see a `NaN` at all,
        // let alone two distinct ones: the exemption must never fire.
        assert_eq!(
            payload_exempt, 0,
            "n={n}: finite data must compare bit for bit with no exemption"
        );
    }
    Ok(())
}

/// Building a tree performs no element work: leaves share storage with their
/// sources (an `Arc` bump), for arrays of any size.
///
/// This is the allocation-free half of the "tree construction is O(1) in n"
/// gate; the timing half is `bench/expr_fused_benchmark.rs`'s
/// `report_build_cost`.
#[test]
fn tree_construction_shares_storage_at_every_size() {
    for n in [1usize, 1_000, 1_000_000] {
        let a = Array::from_vec(vec![1.5_f64; n]);
        let b = Array::from_vec(vec![2.5_f64; n]);
        assert!(
            a.is_unique() && b.is_unique(),
            "n={n}: fresh arrays are unique"
        );

        let tree = a.expr() + b.expr() * a.expr();
        assert_eq!(tree.leaf_count(), 3);
        assert!(
            !a.is_unique(),
            "n={n}: leaf must share `a`'s buffer, not copy it"
        );
        assert!(
            !b.is_unique(),
            "n={n}: leaf must share `b`'s buffer, not copy it"
        );

        drop(tree);
        assert!(a.is_unique() && b.is_unique(), "n={n}: buffers released");
    }
}

// ---------------------------------------------------------------------------
// The one documented exception, pinned down
// ---------------------------------------------------------------------------

/// Regression test for a real defect this suite caught -- kept for its input,
/// with a narrower claim than it once made.
///
/// An earlier revision evaluated any same-shape contiguous tree through a
/// block interpreter that accumulated in place (`dst[i] = dst[i] + src[i]`).
/// For `(a * b) + c` with `a = +NaN`, `b = +inf`, `c = -NaN` -- two `NaN`
/// operands of opposite sign meeting in one commutative operation -- that
/// path produced a `NaN` in every position, as it must, but *also* differed
/// from the eager path in the result's sign bit. That interpreter is gone (it
/// also lost the A/B benchmark to plain eager evaluation -- see
/// `src/expr/fused_eval.rs`).
///
/// What this test pins today is **not** payload agreement. Both paths do
/// still agree bit for bit on this input, but only by luck of codegen: the
/// inputs feed `+NaN` and `-NaN` into one `fadd`, which is exactly the
/// underdetermined case of IEEE-754 §6.2.3 -- see
/// `two_distinct_nans_into_one_add_may_differ_in_payload` below, and this
/// file's module docs. Asserting the payloads match here would be pinning an
/// optimization level. So `assert_values_eq` grants its `NaN`-payload
/// exemption on this input too, and what survives is the real contract: for
/// the historically divergent spelling, both paths put a `NaN` in every
/// position and neither invents a number where the other has a `NaN`. The
/// input stays because it is the sharpest one the suite has for this class of
/// difference.
#[test]
fn nan_payload_case_that_used_to_diverge() -> Result<()> {
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
    assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());

    let mut payload_exempt = 0usize;
    for n in [1usize, 2, 4, 12, 1024, 5000] {
        let a = Array::from_vec(vec![f64::NAN; n]);
        let b = Array::from_vec(vec![f64::INFINITY; n]);
        let c = Array::from_vec(vec![neg_nan; n]);

        // The historically divergent spelling: an `Fma` whose first operand is
        // not a bare leaf (`a + 0.0` preserves both value and NaN sign).
        let node = ExprNode::Fma(
            Box::new(ExprNode::ScalarRhs(BinOp::Add, Box::new(a.expr()), 0.0)),
            Box::new(b.expr()),
            Box::new(c.expr()),
        );
        let got = node.eval()?;
        let want = &(&(&a + 0.0) * &b) + &c;
        payload_exempt += assert_values_eq(&got, &want, &format!("historic NaN case, n={n}"));

        // And the same expression in a shape that does take a fused loop.
        let fused_shape = a.expr() * b.expr() + c.expr();
        assert!(
            fused_shape.will_fuse(),
            "n={n}: (a*b)+c is a specialised shape"
        );
        payload_exempt += assert_values_eq(
            &fused_shape.eval()?,
            &(&(&a * &b) + &c),
            &format!("fused NaN case, n={n}"),
        );
    }
    // Reported, not asserted: it is currently 0 -- the two paths happen to
    // keep the same `NaN` here -- but a build that flipped it to non-zero
    // would still be conforming, which is the whole point of the exemption.
    println!("historic NaN case: NaN-payload exemptions taken: {payload_exempt}");
    Ok(())
}

/// The measured counter-example to the payload half of the old claim.
///
/// `fused_eval_matches_eager_bitwise` found it at proptest seed
/// 12825631960650228096 (`Contiguous`, shape `[2, 5]`, depth <= 2), where the
/// generated tree was `Fma(a, b, c)` over three contiguous leaves and element
/// 1 happened to draw `a = 0.0`, `b = inf`, `c = NaN`. Both paths compute
/// `(a * b) + c`: `0.0 * inf` is an invalid operation, so it yields a fresh
/// hardware default `NaN` (`0xfff8_0000_0000_0000` on x86-64), and *that*
/// `NaN` is then added to the pool's `f64::NAN` (`0x7ff8_0000_0000_0000`) --
/// one `fadd` handed two distinct `NaN` operands, precisely the case
/// IEEE-754 §6.2.3 leaves implementation-defined. LLVM treats `fadd` as
/// commutative, so the single fused `zip` loop kept one operand's `NaN` and
/// the two-pass eager path kept the other's, from the same source-level
/// operand order. Reproduced in a 30-line standalone program with no `numrs2`
/// code in it: divergent under `rustc -O`, identical under `-O0`. It is a
/// codegen artifact of the vectorization this module exists for, not an
/// evaluator bug, and it cannot be removed without removing the fusion.
///
/// Not even *which* side keeps *which* `NaN` is fixed: the standalone program
/// had the fused loop keep `0x7ff8…` and the eager pair keep `0xfff8…`, while
/// inside `numrs2` -- both at the failing seed and at the lengths below -- the
/// assignment is the other way round. That is a second reason the payload
/// cannot be part of the contract: there is no stable answer to promise.
///
/// So this test asserts what the contract now says and no more: **both paths
/// produce a `NaN` in every position**. Asserting payload equality here is
/// exactly the mistake being corrected -- it would pin an optimization level
/// rather than a property of the evaluator.
///
/// The lengths are not arbitrary. Sweeping `n` from 1 to 40 on this machine,
/// the two paths differ for exactly `n = 10` and `n = 11` (8 of the elements
/// in each), and agree everywhere else: at small `n` both stay scalar and
/// keep `0xfff8…`, at large `n` both vectorize cleanly and keep `0x7ff8…`,
/// and only where the vector body and its remainder split differently do the
/// two spellings come apart. Nothing about the arrays changes across that
/// sweep -- every element is the same three constants -- so the length
/// dependence is entirely in the emitted code. `10` and `11` are included so
/// the counter-example is actually exercised here rather than merely
/// described, and the surrounding lengths so the "usually identical" majority
/// is visible next to it.
#[test]
fn two_distinct_nans_into_one_add_may_differ_in_payload() -> Result<()> {
    for n in [1usize, 2, 5, 10, 11, 64, 1024, 5000] {
        let a = Array::from_vec(vec![0.0_f64; n]);
        let b = Array::from_vec(vec![f64::INFINITY; n]);
        let c = Array::from_vec(vec![f64::NAN; n]);

        // The fused spelling: `Fma` of three bare leaves is a specialised
        // shape, so this is the single `zip` loop, not the fallback.
        let node = ExprNode::Fma(Box::new(a.expr()), Box::new(b.expr()), Box::new(c.expr()));
        assert!(
            node.will_fuse(),
            "n={n}: Fma(Leaf, Leaf, Leaf) over contiguous same-shape leaves must fuse"
        );
        let fused = node.eval()?;

        // The eager spelling a user would write by hand: two passes.
        let eager = &(&a * &b) + &c;

        assert_eq!(fused.shape(), eager.shape(), "n={n}: shape");
        let (f, e) = (fused.to_vec(), eager.to_vec());
        for (i, (g, w)) in f.iter().zip(e.iter()).enumerate() {
            assert!(g.is_nan(), "n={n}: fused element {i} is {g}, expected NaN");
            assert!(w.is_nan(), "n={n}: eager element {i} is {w}, expected NaN");
        }

        // Reported, never asserted. On this machine and at this optimization
        // level it is 8 for n = 10 and n = 11 and 0 elsewhere (see the doc
        // comment); another rustc, another target CPU or `-O0` would produce
        // a different pattern, and every one of them is conforming. That is
        // exactly why the assertions above stop at "is a NaN" -- an
        // `assert_eq!` on these bits would be pinning a code generator.
        let payload_diffs = f
            .iter()
            .zip(e.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        println!("(0*inf)+NaN, n={n}: all NaN; payload/sign differs in {payload_diffs}/{n}");
    }
    Ok(())
}

/// How much of the random sweep is actually compared, how much of it lands on
/// `NaN`, and how often the `NaN`-payload exemption is taken -- reported as
/// numbers rather than asserted about, so the strength of the equivalence
/// claim is visible under `--nocapture`.
///
/// Every element is compared bit for bit, with the single exemption
/// `assert_values_eq` documents: a `NaN` facing a `NaN` may differ in payload
/// and sign. The `nan_results` tally shows the adversarial pool really does
/// drive a large fraction of results to `NaN` -- a comparison that merely
/// checked `is_nan()` everywhere would have hidden the defect
/// `nan_payload_case_that_used_to_diverge` records -- while
/// `payload_exempt` keeps the size of the exemption itself in view, so it
/// cannot quietly grow into a licence to differ anywhere else.
#[test]
fn strict_bitwise_sweep_covers_nan_results_too() -> Result<()> {
    let mut compared = 0usize;
    let mut nan_results = 0usize;
    let mut payload_exempt = 0usize;

    for mode_idx in 0..4 {
        let mode = Mode::from_index(mode_idx);
        for seed in 0u64..150 {
            let mut ctx = Ctx {
                rng: Rng::new(seed),
                rows: 3,
                cols: 5,
                mode,
            };
            let (node, eager) = build(&mut ctx, 3);
            let got = node.eval()?;
            assert_eq!(got.shape(), eager.shape(), "seed={seed} mode={mode:?}");

            for (i, (g, w)) in got.to_vec().iter().zip(eager.to_vec().iter()).enumerate() {
                if g.to_bits() != w.to_bits() {
                    if g.is_nan() && w.is_nan() {
                        // The one difference IEEE-754 §6.2.3 leaves open: the
                        // two paths agree that this element is `NaN`, and
                        // differ only in bits no one promised.
                        payload_exempt += 1;
                    } else {
                        // Anything else -- a finite/infinite/signed-zero slip,
                        // or a `NaN` facing a number -- is still a failure,
                        // reported with both bit patterns as before.
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "seed={seed} mode={mode:?} element {i}: {g} vs {w}"
                        );
                    }
                }
                compared += 1;
                if g.is_nan() {
                    nan_results += 1;
                }
            }
        }
    }

    println!(
        "bitwise comparisons: {compared}, of which NaN results: {nan_results}, \
         of which took the NaN-payload exemption: {payload_exempt}"
    );
    assert!(
        compared > 2_000,
        "expected a substantial strict sweep, got {compared}"
    );
    assert!(
        nan_results > 100,
        "the pool is meant to be adversarial; only {nan_results} NaN results"
    );
    Ok(())
}

// Note: there is no `eager_ref_f64` twin of `eager_ref_f32` above. For f32,
// a fresh eager reference has to be recomputed by walking the tree *after*
// the f64 -> f32 cast (see `to_f32`'s note near its call site), since the
// cast happens on the tree, not on a precomputed value. For f64, `build`
// above already returns that precomputed eager value directly as its second
// tuple element -- a separate recursive walker would just recompute exactly
// what `build` already produced, so an `eager_ref_f64` was dead code (an
// unused-function warning) and has been removed rather than kept unused.

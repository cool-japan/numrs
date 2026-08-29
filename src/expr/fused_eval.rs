//! The evaluator behind [`ExprNode::eval`]: a slice-based, single-pass fused
//! engine for the `f64`/`f32` tree shapes it has loops for, plus an eager
//! fallback for every other tree.
//!
//! # Two paths, one result
//!
//! [`eval`] first calls [`plan`], which answers a single question: can this
//! whole tree be evaluated by walking flat slices? It can when every leaf is
//! `f64`/`f32`, every leaf has the same shape, and every leaf is in standard
//! layout (so [`Array::as_slice`] hands back a real slice rather than
//! `None`). If any of those fails, [`eval_eager`] walks the tree calling the
//! crate's ordinary eager operations and returns a result that is
//! element-for-element what the caller would have written by hand.
//!
//! Nothing in either path reassociates or reorders a floating-point
//! operation, so the two paths agree **bit for bit on every finite, infinite
//! and signed-zero result**. In particular [`ExprNode::Fma`] and the
//! `Add(Mul(a, b), c)` shape are evaluated as `(a * b) + c` with two
//! roundings, never through `f64::mul_add`. (Rust does not enable
//! floating-point contraction, so the `x * y + z` written in the fused loops
//! below is guaranteed to compile to a multiply followed by an add.)
//!
//! It also includes **where** a `NaN` appears: an element is `NaN` on one
//! path exactly when it is `NaN` on the other. What it does **not** include
//! is a `NaN`'s **payload and sign bits**.
//!
//! That carve-out is not a hedge; it is what the measurement says, and this
//! was always a claim the source could not argue itself out of. When one
//! operation receives two *distinct* `NaN` operands, IEEE-754 §6.2.3 leaves
//! which payload propagates implementation-defined, neither Rust nor LLVM
//! specifies a choice, and LLVM treats `fadd`/`fmul` as commutative -- so a
//! single fused `zip` loop and a two-pass eager spelling may each keep a
//! different operand's `NaN` from *identical* source-level operand order.
//! Reduced to a 30-line standalone program with no `numrs2` code in it,
//! `(0.0 * inf) + NaN` -- `0.0 * inf` is invalid, yielding a fresh default
//! `NaN`, which then meets `f64::NAN` in one `fadd` -- gives:
//!
//! | build | single fused loop | two-pass eager | agree? |
//! |---|---|---|---|
//! | `rustc -O`  | `0x7ff8000000000000` | `0xfff8000000000000` | no |
//! | `rustc -O0` | `0xfff8000000000000` | `0xfff8000000000000` | yes |
//!
//! Both are conforming; the difference is an artifact of the vectorization
//! this module exists for, and closing it would mean giving that up. NumPy
//! makes no `NaN`-payload guarantee either. The equivalence suite is written
//! to exactly this contract:
//! `tests/test_expr_fused_equivalence.rs::two_distinct_nans_into_one_add_may_differ_in_payload`
//! pins the counter-example above (it was found by that suite's property
//! test, at seed 12825631960650228096), `nan_payload_case_that_used_to_diverge`
//! keeps the input that caught a genuine defect in the block interpreter
//! described below, and the property test compares every element of a few
//! thousand random trees bit for bit, granting only that one exemption and
//! counting each time it is taken.
//!
//! # Fused path: one shape of loop, and why there is only one
//!
//! 1. **Specialised**, in [`fused_specialized`]: the tree is one of the hot
//!    shapes -- `Leaf op Leaf`, `(Leaf op Leaf) op Leaf`, `Leaf op (Leaf op
//!    Leaf)`, `(Leaf op Leaf) op (Leaf op Leaf)`, `(Leaf op scalar) op Leaf`
//!    and its mirror (the axpy shape `a * 2.0 + b`), `Fma(Leaf, Leaf, Leaf)`,
//!    `Leaf op scalar`, `scalar op Leaf`, `unary(Leaf)`. The operator is
//!    matched **once, outside the loop** (the `with_binop!` macro binds a
//!    concrete closure per arm), so what actually runs is a monomorphic `zip`
//!    over 1-4 slices that LLVM autovectorizes, with exactly one output
//!    allocation. [`is_specialized_shape`] is the same recognition rule
//!    without the evaluation, and is what [`ExprNode::will_fuse`] reports.
//! 2. **Eager fallback**, [`eval_eager`]: every other tree. Not fused at all
//!    -- one intermediate array per node -- and identical results.
//!
//! There is deliberately no third, general "interpret the tree in cache-sized
//! blocks" tier, though one was written and measured first. It walked the
//! arrays in 1024-element blocks, evaluating each subtree into an L1-resident
//! scratch buffer, which reads each leaf from memory exactly once and writes
//! one output -- the traffic argument that makes fusion worth doing. It lost
//! anyway, at every size, on the A/B harness in
//! `bench/expr_fused_benchmark.rs` (speedup vs the eager spelling; below
//! 1.00x means slower):
//!
//! | tree | n = 1,000 | 10,000 | 100,000 | 1,000,000 |
//! |---|---|---|---|---|
//! | `(a + b) * (c - d)`, 4 leaves | 0.53x | 0.69x | 0.57x | 0.94x |
//! | 8-leaf, 7-operator tree       | 0.69x | 0.66x | 0.70x | 0.86x |
//!
//! The traffic it saves is real, but it spends it again inside L1: a k-leaf,
//! m-operator tree costs k block copies plus m block passes, so ~7 passes over
//! each block for the 4-leaf tree where true fusion needs one. The 8-leaf row
//! is the case most favourable to the argument -- 9n of memory traffic against
//! eager's 21n -- and it still lost, which settled the question.
//!
//! What replaced it, measured the same way:
//!
//! | tree | n = 1,000 | 10,000 | 100,000 | 1,000,000 |
//! |---|---|---|---|---|
//! | `(a + b) * (c - d)` via [`zip4`] | 1.31x | 1.79x | 1.74x | 1.71x |
//! | 8-leaf tree via [`eval_eager`]   | 0.71x | 0.95x | 0.99x | 1.02x |
//!
//! Extending the specialised set turned the 4-leaf row from a loss into a
//! 1.3-1.8x win. Everything wider goes to [`eval_eager`], which runs the eager
//! code the caller would have written, so it lands at parity from n = 10,000
//! up. The n = 1,000 shortfall is `.expr()/.eval()` call overhead against a
//! 1.8 µs kernel, and `report_build_cost` splits it rather than guessing: of
//! 707 ns total, 287 ns is building the 15-node tree and the remaining ~420 ns
//! is [`plan`] + `collect_leaves` + the shape `Vec` + [`eval_eager`]'s own
//! per-node bookkeeping. Both halves are O(1) in `n` (the same tree costs
//! 294 ns to build at n = 1,000,000), so the whole thing washes out as `n`
//! grows.
//!
//! # Measured: why a plain zip loop, and not `SimdUnifiedOps`
//!
//! The fused loops here are plain `iter().zip().map().collect()` over
//! contiguous slices, not calls into `scirs2_core::simd_ops::SimdUnifiedOps`.
//! That is a measured choice, not an omission.
//!
//! Reproduce with
//! `CARGO_INCREMENTAL=0 EXPR_AB_REPORT=1 cargo bench --bench expr_fused_benchmark`,
//! which reports the **minimum** over alternating A/B rounds rather than a
//! criterion mean (this machine carries background build load; a mean is easy
//! to bias, a minimum is not). Numbers below are one representative run of
//! five on `aarch64-apple-darwin`, release profile
//! (`lto = "fat"`, `codegen-units = 1`); the run-to-run spread was under 15%
//! and never changed a ranking. Every table in this module is a row from that
//! report; none of it is extrapolated.
//!
//! Inner-loop shootout for `a + b * c`, all four operating on the same data:
//!
//! | n | zip loop (this module) | `simd_mul`+`simd_add` | `simd_fma` | eager `&a + &(&b * &c)` |
//! |---|---|---|---|---|
//! | 1,000     | 166 ns   | 750 ns   | 625 ns   | 458 ns   |
//! | 10,000    | 2.4 µs   | 10.3 µs  | 7.0 µs   | 3.9 µs   |
//! | 100,000   | 22.5 µs  | 93.2 µs  | 73.8 µs  | 32.2 µs  |
//! | 1,000,000 | 371 µs   | 1.2 ms   | 835 µs   | 597 µs   |
//!
//! The zip loop wins everywhere, by 2.2-6x over either explicit-SIMD route.
//! This reproduces the finding `kernels::elementwise` documents for a single
//! `+`: LLVM already vectorizes a trivial arithmetic loop over a contiguous
//! slice, and `SimdUnifiedOps`' per-call view construction and dispatch is
//! pure overhead on top -- plus both SIMD routes allocate an `Array1` per
//! call, which is exactly the intermediate fusion exists to delete. Note also
//! that `simd_fma` is *not* usable here even where it is fast: it computes a
//! true single-rounding FMA, which would change results the eager path rounds
//! twice.
//!
//! # Measured: does the whole thing actually beat eager?
//!
//! The table above isolates the inner loop. End to end -- `plan`,
//! `collect_leaves`, the output allocation and `Array::from_vec_shape`
//! included -- `(a.expr() + b.expr() * c.expr()).eval()` against the eager
//! `&a + &(&b * &c)`, same harness:
//!
//! | n | fused `.eval()` | eager | speedup (range over 5 runs) |
//! |---|---|---|---|
//! | 1,000     | 416 ns  | 458 ns  | 1.00x - 1.20x |
//! | 10,000    | 3.2 µs  | 4.8 µs  | 1.38x - 1.49x |
//! | 100,000   | 22.7 µs | 31.6 µs | 1.39x - 1.51x |
//! | 1,000,000 | 363 µs  | 564 µs  | 1.45x - 1.69x |
//!
//! Fusion wins at every size and by ~1.5x once the arrays leave cache, which
//! is what the traffic argument predicts: the fused form reads 3n and writes
//! n, the eager form reads 4n and writes 2n. At n = 1,000 the margin collapses
//! to somewhere between nothing and 1.2x, because the ~83 ns of tree
//! construction is no longer negligible against a sub-microsecond kernel --
//! which is why the kill criterion was set at n >= 100,000, where the win is
//! stable across every run.
//!
//! Tree construction itself is O(1) in the array size -- 83 ns for a
//! three-node tree at both n = 1,000 and n = 1,000,000 (~27 ns/node; ratio
//! 0.97-1.02 across runs), confirming `.expr()` is an `Arc` bump, not a copy.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::{borrow, cast};
use std::any::TypeId;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::owned::{BinOp, ExprNode, UnaryOp};

// ---------------------------------------------------------------------------
// Element types the fused engine can run on
// ---------------------------------------------------------------------------

/// The dtypes the fused path supports, plus the sound `TypeId`-guarded
/// reinterpretation hooks (from [`crate::kernels::cast`]) that get a generic
/// `&[T]` in and a `Vec<T>` back out.
trait FusedElem:
    Copy
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Reinterpret `&[T]` as `&[Self]` iff `T == Self`.
    fn from_slice<T: 'static>(s: &[T]) -> Option<&[Self]>;
    /// Reinterpret `&T` as `Self` iff `T == Self`.
    fn from_scalar<T: 'static>(x: &T) -> Option<Self>;
    /// Reinterpret an owned `Vec<Self>` as `Vec<T>` iff `T == Self`.
    fn into_vec<T: 'static>(v: Vec<Self>) -> Option<Vec<T>>;

    /// `|self|`
    fn abs(self) -> Self;
    /// `sqrt(self)`
    fn sqrt(self) -> Self;
    /// `e^self`
    fn exp(self) -> Self;
    /// `ln(self)`
    fn ln(self) -> Self;
}

impl FusedElem for f64 {
    fn from_slice<T: 'static>(s: &[T]) -> Option<&[f64]> {
        cast::as_f64(s)
    }
    fn from_scalar<T: 'static>(x: &T) -> Option<f64> {
        cast::as_f64(std::slice::from_ref(x)).and_then(|s| s.first().copied())
    }
    fn into_vec<T: 'static>(v: Vec<f64>) -> Option<Vec<T>> {
        cast::vec_from_f64(v)
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
}

impl FusedElem for f32 {
    fn from_slice<T: 'static>(s: &[T]) -> Option<&[f32]> {
        cast::as_f32(s)
    }
    fn from_scalar<T: 'static>(x: &T) -> Option<f32> {
        cast::as_f32(std::slice::from_ref(x)).and_then(|s| s.first().copied())
    }
    fn into_vec<T: 'static>(v: Vec<f32>) -> Option<Vec<T>> {
        cast::vec_from_f32(v)
    }
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    fn ln(self) -> f32 {
        f32::ln(self)
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Decide whether `root` can be evaluated by the fused engine, and if so
/// return the common leaf shape and element count.
///
/// The three conditions are the ones [`super::owned`]'s module docs promise:
/// `f64`/`f32` dtype, one shape shared by every leaf (no broadcasting), and
/// every leaf in standard layout.
fn plan<T: Clone + 'static>(root: &ExprNode<T>) -> Option<(Vec<usize>, usize)> {
    if TypeId::of::<T>() != TypeId::of::<f64>() && TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    let mut leaves: Vec<&Array<T>> = Vec::new();
    root.collect_leaves(&mut leaves);
    let first = leaves.first()?;
    let shape = first.shape();
    for leaf in &leaves {
        if leaf.shape() != shape || leaf.as_slice().is_none() {
            return None;
        }
    }
    let n = first.size();
    Some((shape, n))
}

/// Whether [`eval`] will take the fused path for this tree.
///
/// Both halves must hold: [`plan`] (dtype, one common shape, standard layout)
/// and [`is_specialized_shape`] (a tree shape [`fused_specialized`] has a loop
/// for). Everything else evaluates through [`eval_eager`].
pub(super) fn will_fuse<T: Clone + 'static>(root: &ExprNode<T>) -> bool {
    plan(root).is_some() && is_specialized_shape(root)
}

// ---------------------------------------------------------------------------
// Operator dispatch: match once, outside the loop
// ---------------------------------------------------------------------------

/// Bind `$f` to a concrete, inlinable closure for `$op` and run `$body` with
/// it.
///
/// The point is *where* the `match` happens: once per `eval` call rather than
/// once per element, so each arm's `$body` monomorphizes into a loop with the
/// arithmetic inlined and no branch inside it. Requires a type parameter
/// named `S: FusedElem` to be in scope at the call site.
macro_rules! with_binop {
    ($op:expr, $f:ident, $body:expr) => {
        match $op {
            BinOp::Add => {
                let $f = |x: S, y: S| x + y;
                $body
            }
            BinOp::Sub => {
                let $f = |x: S, y: S| x - y;
                $body
            }
            BinOp::Mul => {
                let $f = |x: S, y: S| x * y;
                $body
            }
            BinOp::Div => {
                let $f = |x: S, y: S| x / y;
                $body
            }
        }
    };
}

/// `out[i] = f(x[i])`
fn map1<S: Copy, F: Fn(S) -> S>(x: &[S], f: F) -> Vec<S> {
    x.iter().map(|&a| f(a)).collect()
}

/// `out[i] = f(x[i], y[i])`
fn zip2<S: Copy, F: Fn(S, S) -> S>(x: &[S], y: &[S], f: F) -> Vec<S> {
    x.iter().zip(y.iter()).map(|(&a, &b)| f(a, b)).collect()
}

/// `out[i] = outer(inner(x[i], y[i]), z[i])`
fn zip3_left<S: Copy, FI: Fn(S, S) -> S, FO: Fn(S, S) -> S>(
    x: &[S],
    y: &[S],
    z: &[S],
    inner: FI,
    outer: FO,
) -> Vec<S> {
    x.iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((&a, &b), &c)| outer(inner(a, b), c))
        .collect()
}

/// `out[i] = outer(x[i], inner(y[i], z[i]))`
fn zip3_right<S: Copy, FI: Fn(S, S) -> S, FO: Fn(S, S) -> S>(
    x: &[S],
    y: &[S],
    z: &[S],
    inner: FI,
    outer: FO,
) -> Vec<S> {
    x.iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((&a, &b), &c)| outer(a, inner(b, c)))
        .collect()
}

/// `out[i] = outer(left(w[i], x[i]), right(y[i], z[i]))`
///
/// The four-leaf "two binary subtrees" shape -- `(a + b) * (c - d)` and
/// friends. It gets its own loop because the alternative for this shape is the
/// block interpreter that used to serve it, which measured *slower* than eager
/// (see this module's "Measured" section): seven passes over an L1 block beat
/// nothing.
fn zip4<S: Copy, FL: Fn(S, S) -> S, FR: Fn(S, S) -> S, FO: Fn(S, S) -> S>(
    w: &[S],
    x: &[S],
    y: &[S],
    z: &[S],
    left: FL,
    right: FR,
    outer: FO,
) -> Vec<S> {
    w.iter()
        .zip(x.iter())
        .zip(y.iter())
        .zip(z.iter())
        .map(|(((&a, &b), &c), &d)| outer(left(a, b), right(c, d)))
        .collect()
}

/// Apply a unary op elementwise.
fn apply_unary<S: FusedElem>(op: UnaryOp, x: &[S]) -> Vec<S> {
    match op {
        UnaryOp::Neg => map1(x, |a| -a),
        UnaryOp::Abs => map1(x, S::abs),
        UnaryOp::Sqrt => map1(x, S::sqrt),
        UnaryOp::Exp => map1(x, S::exp),
        UnaryOp::Ln => map1(x, S::ln),
    }
}

// ---------------------------------------------------------------------------
// Fused path, shape 1: hand-specialised whole-tree loops
// ---------------------------------------------------------------------------

/// The leaf's data as a flat `&[S]`, or `None` if this node is not a leaf (or
/// is not contiguous / not of dtype `S`).
fn leaf_slice<T: Clone + 'static, S: FusedElem>(node: &ExprNode<T>) -> Option<&[S]> {
    match node {
        ExprNode::Leaf(a) => S::from_slice(a.as_slice()?),
        _ => None,
    }
}

/// Recognise the hot tree shapes and run a single monomorphic loop for them.
///
/// Returns `None` for any shape not covered here, which sends the tree to
/// [`eval_eager`] instead. [`is_specialized_shape`] answers the same question
/// without evaluating anything.
fn fused_specialized<T: Clone + 'static, S: FusedElem>(root: &ExprNode<T>) -> Option<Vec<S>> {
    match root {
        ExprNode::Leaf(_) => leaf_slice::<T, S>(root).map(<[S]>::to_vec),

        ExprNode::Binary(op, l, r) => {
            // Leaf op Leaf
            if let (Some(x), Some(y)) = (leaf_slice::<T, S>(l), leaf_slice::<T, S>(r)) {
                return Some(with_binop!(*op, f, zip2(x, y, f)));
            }
            // (Leaf op Leaf) op Leaf  -- this is where `a * b + c` lands.
            if let (ExprNode::Binary(iop, ll, lr), Some(z)) = (&**l, leaf_slice::<T, S>(r)) {
                if let (Some(x), Some(y)) = (leaf_slice::<T, S>(ll), leaf_slice::<T, S>(lr)) {
                    return Some(with_binop!(
                        *iop,
                        fi,
                        with_binop!(*op, fo, zip3_left(x, y, z, fi, fo))
                    ));
                }
            }
            // Leaf op (Leaf op Leaf)
            if let (Some(x), ExprNode::Binary(iop, rl, rr)) = (leaf_slice::<T, S>(l), &**r) {
                if let (Some(y), Some(z)) = (leaf_slice::<T, S>(rl), leaf_slice::<T, S>(rr)) {
                    return Some(with_binop!(
                        *iop,
                        fi,
                        with_binop!(*op, fo, zip3_right(x, y, z, fi, fo))
                    ));
                }
            }
            // (Leaf sop k) op Leaf -- `a * 2.0 + b`, the axpy shape.
            if let (ExprNode::ScalarRhs(sop, se, k), Some(y)) = (&**l, leaf_slice::<T, S>(r)) {
                if let (Some(x), Some(k)) = (leaf_slice::<T, S>(se), S::from_scalar(k)) {
                    return Some(with_binop!(
                        *sop,
                        fs,
                        with_binop!(*op, fo, zip2(x, y, |a, b| fo(fs(a, k), b)))
                    ));
                }
            }
            // Leaf op (Leaf sop k) -- `a + b * 2.0`.
            if let (Some(x), ExprNode::ScalarRhs(sop, se, k)) = (leaf_slice::<T, S>(l), &**r) {
                if let (Some(y), Some(k)) = (leaf_slice::<T, S>(se), S::from_scalar(k)) {
                    return Some(with_binop!(
                        *sop,
                        fs,
                        with_binop!(*op, fo, zip2(x, y, |a, b| fo(a, fs(b, k))))
                    ));
                }
            }
            // (Leaf op Leaf) op (Leaf op Leaf) -- `(a + b) * (c - d)`.
            if let (ExprNode::Binary(lop, ll, lr), ExprNode::Binary(rop, rl, rr)) = (&**l, &**r) {
                if let (Some(w), Some(x), Some(y), Some(z)) = (
                    leaf_slice::<T, S>(ll),
                    leaf_slice::<T, S>(lr),
                    leaf_slice::<T, S>(rl),
                    leaf_slice::<T, S>(rr),
                ) {
                    return Some(with_binop!(
                        *lop,
                        fl,
                        with_binop!(*rop, fr, with_binop!(*op, fo, zip4(w, x, y, z, fl, fr, fo)))
                    ));
                }
            }
            None
        }

        ExprNode::Fma(a, b, c) => {
            let (x, y, z) = (
                leaf_slice::<T, S>(a)?,
                leaf_slice::<T, S>(b)?,
                leaf_slice::<T, S>(c)?,
            );
            // Two roundings, exactly as `(a * b) + c`: never `mul_add`.
            Some(zip3_left(x, y, z, |p, q| p * q, |p, q| p + q))
        }

        ExprNode::ScalarRhs(op, e, s) => {
            let x = leaf_slice::<T, S>(e)?;
            let k = S::from_scalar(s)?;
            Some(with_binop!(*op, f, map1(x, |a| f(a, k))))
        }

        ExprNode::ScalarLhs(op, s, e) => {
            let x = leaf_slice::<T, S>(e)?;
            let k = S::from_scalar(s)?;
            Some(with_binop!(*op, f, map1(x, |a| f(k, a))))
        }

        ExprNode::Unary(op, e) => {
            let x = leaf_slice::<T, S>(e)?;
            Some(apply_unary(*op, x))
        }
    }
}

// ---------------------------------------------------------------------------
// Shape recognition (what `will_fuse` reports, without evaluating)
// ---------------------------------------------------------------------------

/// Is this node a plain [`ExprNode::Leaf`]?
fn is_leaf<T>(node: &ExprNode<T>) -> bool {
    matches!(node, ExprNode::Leaf(_))
}

/// Whether [`fused_specialized`] has a loop for this tree's *shape*.
///
/// Structure only: [`plan`] has already established dtype, common shape and
/// standard layout by the time this is consulted. Kept in lockstep with
/// [`fused_specialized`]'s arms by
/// `recognizer_agrees_with_the_specialised_arms` in this module's tests.
fn is_specialized_shape<T>(root: &ExprNode<T>) -> bool {
    match root {
        ExprNode::Leaf(_) => true,

        ExprNode::Binary(_, l, r) => {
            // Leaf op Leaf
            (is_leaf(l) && is_leaf(r))
                // (Leaf op Leaf) op Leaf
                || (matches!(&**l, ExprNode::Binary(_, ll, lr) if is_leaf(ll) && is_leaf(lr))
                    && is_leaf(r))
                // Leaf op (Leaf op Leaf)
                || (is_leaf(l)
                    && matches!(&**r, ExprNode::Binary(_, rl, rr) if is_leaf(rl) && is_leaf(rr)))
                // (Leaf sop k) op Leaf
                || (matches!(&**l, ExprNode::ScalarRhs(_, se, _) if is_leaf(se)) && is_leaf(r))
                // Leaf op (Leaf sop k)
                || (is_leaf(l)
                    && matches!(&**r, ExprNode::ScalarRhs(_, se, _) if is_leaf(se)))
                // (Leaf op Leaf) op (Leaf op Leaf)
                || matches!(
                    (&**l, &**r),
                    (ExprNode::Binary(_, ll, lr), ExprNode::Binary(_, rl, rr))
                        if is_leaf(ll) && is_leaf(lr) && is_leaf(rl) && is_leaf(rr)
                )
        }

        ExprNode::Fma(a, b, c) => is_leaf(a) && is_leaf(b) && is_leaf(c),
        ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => is_leaf(e),
        ExprNode::Unary(_, e) => is_leaf(e),
    }
}

// ---------------------------------------------------------------------------
// Eager fallback
// ---------------------------------------------------------------------------

/// Elementwise `abs`/`sqrt`/`exp`/`ln` on an already-evaluated array, for the
/// eager path.
///
/// These have no generic eager counterpart in the crate, so they are
/// `f64`/`f32` only; any other dtype is a hard error rather than a silently
/// different answer.
fn unary_math_eager<T: Clone + 'static>(a: &Array<T>, op: UnaryOp) -> Result<Array<T>> {
    let shape = a.shape();
    let src = borrow::operand(a);

    if let Some(s) = cast::as_f64(&src) {
        let out = apply_unary(op, s);
        if let Some(data) = cast::vec_from_f64::<T>(out) {
            return Array::from_vec_shape(data, &shape);
        }
    }
    if let Some(s) = cast::as_f32(&src) {
        let out = apply_unary(op, s);
        if let Some(data) = cast::vec_from_f32::<T>(out) {
            return Array::from_vec_shape(data, &shape);
        }
    }
    Err(NumRs2Error::NotImplemented(format!(
        "expression unary op `{op:?}` is implemented for f64 and f32 only"
    )))
}

/// Evaluate the tree with the crate's ordinary eager operations, one
/// intermediate array per node.
///
/// This is the fallback for every tree the fused engine declines, and the
/// definition of "correct" the fused engine is checked against: each arm
/// calls exactly the method the corresponding eager operator calls
/// (`Add for &Array<T>` is `add_broadcast`, and so on).
fn eval_eager<T>(node: &ExprNode<T>) -> Result<Array<T>>
where
    T: Clone
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    match node {
        ExprNode::Leaf(a) => Ok(a.clone()),

        ExprNode::Binary(op, l, r) => {
            let lv = eval_eager(l)?;
            let rv = eval_eager(r)?;
            match op {
                BinOp::Add => lv.add_broadcast(&rv),
                BinOp::Sub => lv.subtract_broadcast(&rv),
                BinOp::Mul => lv.multiply_broadcast(&rv),
                BinOp::Div => lv.divide_broadcast(&rv),
            }
        }

        ExprNode::ScalarRhs(op, e, s) => {
            let v = eval_eager(e)?;
            Ok(match op {
                BinOp::Add => v.add_scalar(s.clone()),
                BinOp::Sub => v.subtract_scalar(s.clone()),
                BinOp::Mul => v.multiply_scalar(s.clone()),
                BinOp::Div => v.divide_scalar(s.clone()),
            })
        }

        ExprNode::ScalarLhs(op, s, e) => {
            let v = eval_eager(e)?;
            Ok(match op {
                BinOp::Add => v.map(|x| s.clone() + x),
                BinOp::Sub => v.map(|x| s.clone() - x),
                BinOp::Mul => v.map(|x| s.clone() * x),
                BinOp::Div => v.map(|x| s.clone() / x),
            })
        }

        ExprNode::Unary(UnaryOp::Neg, e) => Ok(eval_eager(e)?.map(|x| -x)),
        ExprNode::Unary(op, e) => unary_math_eager(&eval_eager(e)?, *op),

        ExprNode::Fma(a, b, c) => {
            let av = eval_eager(a)?;
            let bv = eval_eager(b)?;
            let prod = av.multiply_broadcast(&bv)?;
            let cv = eval_eager(c)?;
            prod.add_broadcast(&cv)
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Evaluate `root`, fusing when [`plan`] says the tree qualifies.
pub(super) fn eval<T>(root: &ExprNode<T>) -> Result<Array<T>>
where
    T: Clone
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    // `n` (element count) is validated by `plan` but not needed here: every
    // leaf already shares `shape` (and so, by construction, the same
    // length) by the time `fused_specialized` walks their slices.
    let Some((shape, _n)) = plan(root) else {
        return eval_eager(root);
    };

    // `plan` already proved T is f64 or f32, so exactly one of these runs and
    // the reinterpretation back to `Vec<T>` cannot fail; the `if let`s keep
    // that fact from needing an `unwrap`.
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        if let Some(out) = fused_specialized::<T, f64>(root) {
            if let Some(data) = <f64 as FusedElem>::into_vec::<T>(out) {
                return Array::from_vec_shape(data, &shape);
            }
        }
    } else if let Some(out) = fused_specialized::<T, f32>(root) {
        if let Some(data) = <f32 as FusedElem>::into_vec::<T>(out) {
            return Array::from_vec_shape(data, &shape);
        }
    }

    eval_eager(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::owned::IntoExpr;

    fn seq(n: usize, scale: f64, offset: f64) -> Array<f64> {
        Array::from_vec((0..n).map(|i| i as f64 * scale + offset).collect())
    }

    /// Full-strength bitwise comparison: `0.0` never passes for `-0.0` and a
    /// 1-ulp slip is a failure.
    ///
    /// This is *stricter* than the module's guarantee, which exempts a
    /// `NaN`'s payload and sign bits (see the module docs). Keeping it strict
    /// here is safe because every caller was audited for the one situation
    /// that makes payloads underdetermined -- a single operation receiving two
    /// **distinct** `NaN` operands -- and none of them reaches it:
    ///
    /// * every caller but one feeds only finite data (`seq`, small integer
    ///   ranges), so no `NaN` arises at all;
    /// * the exception, `special_values_match_eager`, does compute `NaN`s, but
    ///   never from two distinct ones. Its only operation with two `NaN`
    ///   operands is `f64::NAN + (2.0 * f64::NAN)`; the multiply sees exactly
    ///   one `NaN` and so returns that operand quieted, leaving the add with
    ///   two *bit-identical* `NaN`s, where IEEE-754 §6.2.3's freedom to pick
    ///   either operand cannot change a single output bit.
    ///
    /// The relaxed comparison the contract actually licenses lives in
    /// `tests/test_expr_fused_equivalence.rs::assert_values_eq`, which is
    /// where the adversarial `NaN` pool is. A new caller here that can put two
    /// distinct `NaN`s into one operation must relax this helper the same way
    /// rather than pin an optimization level.
    fn assert_bit_eq(got: &Array<f64>, want: &Array<f64>, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what}: shape");
        for (i, (g, w)) in got.to_vec().iter().zip(want.to_vec().iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: element {i}: {g} ({:#018x}) vs {w} ({:#018x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    #[test]
    fn fuses_the_canonical_chain_and_matches_eager() -> Result<()> {
        // Big enough to cross the chunk boundary in both directions.
        for n in [0usize, 1, 7, 1_023, 1_024, 1_025, 5_000] {
            let a = seq(n, 1.0, 0.5);
            let b = seq(n, -0.25, 3.0);
            let c = seq(n, 0.125, -2.0);

            let e = a.expr() + b.expr() * c.expr();
            assert!(e.will_fuse(), "n={n}");
            let fused = e.eval()?;
            let eager = &a + &(&b * &c);
            assert_bit_eq(&fused, &eager, &format!("a + b*c, n={n}"));
        }
        Ok(())
    }

    #[test]
    fn fma_node_is_two_roundings_not_mul_add() {
        // `(1 + 2^-27)^2` needs 54 significand bits, so mul-then-add loses the
        // low bit that a single-rounding `mul_add` keeps.
        let x = 1.0_f64 + 2.0_f64.powi(-27);
        let a = Array::from_vec(vec![x]);
        let b = Array::from_vec(vec![x]);
        let c = Array::from_vec(vec![-1.0_f64]);

        let node = (a.expr() * b.expr() + c.expr()).fuse_fma();
        assert!(matches!(node, ExprNode::Fma(..)));
        let got = node.eval().expect("fma eval");

        let two_roundings = x * x - 1.0;
        let one_rounding = x.mul_add(x, -1.0);
        assert_ne!(
            two_roundings.to_bits(),
            one_rounding.to_bits(),
            "test is vacuous unless these differ"
        );
        assert_eq!(got.to_vec()[0].to_bits(), two_roundings.to_bits());
    }

    #[test]
    fn unspecialised_shapes_fall_back_to_eager_and_match() -> Result<()> {
        let n = 3_000;
        let a = seq(n, 1.0, 0.5);
        let b = seq(n, -0.25, 3.0);
        let c = seq(n, 0.125, -2.0);
        let d = seq(n, 2.0, 1.0);
        let e = seq(n, -1.5, 0.25);

        // Five leaves: past every hand-specialised shape, so this evaluates
        // through `eval_eager` -- measured faster than the block interpreter
        // that used to serve this case (see the module docs).
        let tree = ((a.expr() + b.expr()) * (c.expr() - d.expr())) / e.expr();
        assert!(!tree.will_fuse(), "5 leaves is past the specialised set");
        assert!(fused_specialized::<f64, f64>(&tree).is_none());
        let fused = tree.eval()?;
        let eager = &(&(&a + &b) * &(&c - &d)) / &e;
        assert_bit_eq(&fused, &eager, "((a+b)*(c-d))/e");
        Ok(())
    }

    /// The four-leaf `(a op b) op (c op d)` shape has its own specialised
    /// loop ([`zip4`]); it must be taken, and must agree with eager.
    #[test]
    fn four_leaf_shape_is_specialised_and_matches_eager() -> Result<()> {
        let n = 3_000;
        let a = seq(n, 1.0, 0.5);
        let b = seq(n, -0.25, 3.0);
        let c = seq(n, 0.125, -2.0);
        let d = seq(n, 2.0, 1.0);

        let tree = (a.expr() + b.expr()) * (c.expr() - d.expr());
        assert!(tree.will_fuse());
        assert!(
            fused_specialized::<f64, f64>(&tree).is_some(),
            "(a+b)*(c-d) must hit the zip4 loop, not the eager fallback"
        );
        assert_bit_eq(&tree.eval()?, &(&(&a + &b) * &(&c - &d)), "(a+b)*(c-d)");

        // A different operator triple, to prove the dispatch is per-operator.
        let tree2 = (a.expr() / b.expr()) - (c.expr() * d.expr());
        assert!(fused_specialized::<f64, f64>(&tree2).is_some());
        assert_bit_eq(&tree2.eval()?, &(&(&a / &b) - &(&c * &d)), "(a/b)-(c*d)");
        Ok(())
    }

    #[test]
    fn every_specialised_shape_matches_eager() -> Result<()> {
        let n = 2_000;
        let a = seq(n, 1.0, 1.5);
        let b = seq(n, 0.5, 2.5);
        let c = seq(n, 0.25, 0.75);

        assert_bit_eq(&(a.expr() - b.expr()).eval()?, &(&a - &b), "leaf-leaf");
        assert_bit_eq(
            &((a.expr() / b.expr()) * c.expr()).eval()?,
            &(&(&a / &b) * &c),
            "chain-left",
        );
        assert_bit_eq(
            &(a.expr() - b.expr() / c.expr()).eval()?,
            &(&a - &(&b / &c)),
            "chain-right",
        );
        assert_bit_eq(&(a.expr() * 3.0).eval()?, &(&a * 3.0), "scalar-rhs");
        assert_bit_eq(&(3.0 - a.expr()).eval()?, &a.map(|x| 3.0 - x), "scalar-lhs");
        assert_bit_eq(&(-a.expr()).eval()?, &(-&a), "neg");
        assert_bit_eq(&a.expr().sqrt().eval()?, &a.map(f64::sqrt), "sqrt");
        assert_bit_eq(&a.expr().eval()?, &a, "bare leaf");
        Ok(())
    }

    #[test]
    fn non_contiguous_leaf_falls_back_and_matches() -> Result<()> {
        let a = Array::from_vec((0..12).map(|i| i as f64).collect()).reshape(&[3, 4]);
        let t = a.transpose_axis(0, 1);
        assert!(!t.is_c_contiguous());

        let e = t.expr() * t.expr() + t.expr();
        assert!(!e.will_fuse(), "non-contiguous leaf must not fuse");
        let got = e.eval()?;
        let want = &(&t * &t) + &t;
        assert_bit_eq(&got, &want, "transposed leaves");
        Ok(())
    }

    #[test]
    fn broadcast_shapes_fall_back_and_match() -> Result<()> {
        let row = Array::from_vec(vec![1.0_f64, 2.0, 3.0]).reshape(&[1, 3]);
        let col = Array::from_vec(vec![10.0_f64, 20.0, 30.0]).reshape(&[3, 1]);

        let e = row.expr() + col.expr();
        assert!(!e.will_fuse());
        let got = e.eval()?;
        assert_eq!(got.shape(), vec![3, 3]);
        assert_bit_eq(&got, &(&row + &col), "broadcast");
        Ok(())
    }

    #[test]
    fn incompatible_shapes_error_like_the_eager_op() {
        let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0_f64, 2.0]);
        let e = a.expr() + b.expr();
        assert!(!e.will_fuse());
        assert!(e.eval().is_err());
        assert!(a.add_broadcast(&b).is_err());
    }

    #[test]
    fn f32_fuses_too() -> Result<()> {
        let n = 1_500;
        let a = Array::from_vec((0..n).map(|i| i as f32 * 0.5).collect());
        let b = Array::from_vec((0..n).map(|i| i as f32 - 3.0).collect());
        // `(a * b) + b` is the zip3_left shape, specialised for f32 too.
        let e = a.expr() * b.expr() + b.expr();
        assert!(e.will_fuse());
        let got = e.eval()?;
        let want = &(&a * &b) + &b;
        for (g, w) in got.to_vec().iter().zip(want.to_vec().iter()) {
            assert_eq!(g.to_bits(), w.to_bits());
        }
        Ok(())
    }

    #[test]
    fn integer_dtype_falls_back_and_matches() -> Result<()> {
        let a = Array::from_vec(vec![1_i64, 2, 3, 4]);
        let b = Array::from_vec(vec![10_i64, 20, 30, 40]);
        let e = a.expr() * b.expr() + 5_i64.into_expr_node();
        assert!(!e.will_fuse(), "i64 has no fused path");
        assert_eq!(e.eval()?.to_vec(), vec![15, 45, 95, 165]);
        Ok(())
    }

    /// Small helper so the integer test above can build a constant leaf
    /// without scalar operator impls (those are f64/f32 only).
    trait IntoExprNode {
        fn into_expr_node(self) -> ExprNode<i64>;
    }
    impl IntoExprNode for i64 {
        fn into_expr_node(self) -> ExprNode<i64> {
            ExprNode::Leaf(Array::from_vec(vec![self; 4]))
        }
    }

    #[test]
    fn maths_op_on_integer_dtype_is_a_clean_error() {
        let a = Array::from_vec(vec![4_i64, 9]);
        let e = a.expr().sqrt();
        let err = e.eval().expect_err("sqrt on i64 must not silently succeed");
        assert!(matches!(err, NumRs2Error::NotImplemented(_)), "{err:?}");
    }

    #[test]
    fn special_values_match_eager() -> Result<()> {
        let a = Array::from_vec(vec![
            0.0_f64,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.0,
        ]);
        let b = Array::from_vec(vec![-0.0_f64, 0.0, 1.0, f64::INFINITY, 2.0, f64::NAN]);
        let c = Array::from_vec(vec![1.0_f64, -1.0, 0.0, -0.0, f64::NAN, 3.0]);

        let e = a.expr() + b.expr() * c.expr();
        assert!(e.will_fuse());
        assert_bit_eq(&e.eval()?, &(&a + &(&b * &c)), "special values");
        Ok(())
    }

    /// The signed-zero half of the guarantee, on its own and strictly: `-0.0`
    /// results must stay `-0.0` and `0.0` must stay `0.0` in the fused path.
    /// (No `NaN` is involved, so nothing here is relaxed.)
    #[test]
    fn signed_zeros_are_preserved_exactly() -> Result<()> {
        // -0.0 + -0.0 = -0.0, while -0.0 + 0.0 = +0.0.
        let a = Array::from_vec(vec![-0.0_f64, -0.0, 0.0, 0.0]);
        let b = Array::from_vec(vec![-0.0_f64, 0.0, -0.0, 0.0]);
        let c = Array::from_vec(vec![1.0_f64; 4]);

        // `a + b * c` with c == 1 keeps b's zero sign through the multiply.
        let got = (a.expr() + b.expr() * c.expr()).eval()?;
        let want = &a + &(&b * &c);
        for (i, (g, w)) in got.to_vec().iter().zip(want.to_vec().iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "element {i}: {g} vs {w} -- signed zeros must survive fusion"
            );
        }
        assert_eq!(got.to_vec()[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(got.to_vec()[3].to_bits(), 0.0_f64.to_bits());
        Ok(())
    }

    /// [`is_specialized_shape`] must agree, arm for arm, with what
    /// [`fused_specialized`] actually handles -- they are two hand-written
    /// copies of the same rule, and `will_fuse` believes the first one.
    #[test]
    fn recognizer_agrees_with_the_specialised_arms() {
        let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let l = || a.expr();

        let trees: Vec<ExprNode<f64>> = vec![
            l(),
            l() + l(),
            (l() + l()) * l(),
            l() - (l() / l()),
            (l() + l()) * (l() - l()),
            ExprNode::Fma(Box::new(l()), Box::new(l()), Box::new(l())),
            l() * 2.0,
            2.0 - l(),
            -l(),
            l().sqrt(),
            // Past the specialised set:
            ((l() + l()) * (l() - l())) / l(),
            l() * 2.0 + l(),
            ExprNode::Fma(Box::new(l() + l()), Box::new(l()), Box::new(l())),
            (l() + l()) + ((l() + l()) + l()),
            (-l()).abs(),
        ];

        for t in &trees {
            assert_eq!(
                is_specialized_shape(t),
                fused_specialized::<f64, f64>(t).is_some(),
                "recognizer disagrees with the evaluator for {t:?}"
            );
        }
    }

    #[test]
    fn deep_tree_agrees_with_eager() -> Result<()> {
        let n = 2_500;
        let a = seq(n, 1.0, 1.0);
        let b = seq(n, 0.5, 2.0);
        let c = seq(n, 0.25, 3.0);
        let d = seq(n, 0.125, 4.0);

        // Depth 5, mixed variants, deliberately not a specialised shape, so
        // this goes through `eval_eager`.
        let e = ((a.expr() * b.expr() + c.expr()) / (d.expr() - 1.0)).abs() + (2.0 * a.expr());
        assert!(
            !e.will_fuse(),
            "depth-5 mixed tree is past the specialised set"
        );
        let got = e.eval()?;

        let quotient = &(&(&a * &b) + &c) / &(&d - 1.0);
        let want = &quotient.map(f64::abs) + &a.map(|x| 2.0 * x);
        assert_bit_eq(&got, &want, "depth-5 mixed tree");
        Ok(())
    }

    #[test]
    fn empty_arrays_evaluate_to_empty() -> Result<()> {
        let a: Array<f64> = Array::from_vec(vec![]);
        let e = a.expr() + a.expr() * a.expr();
        assert!(e.will_fuse());
        let got = e.eval()?;
        assert_eq!(got.size(), 0);
        Ok(())
    }

    #[test]
    fn multi_dimensional_shape_is_preserved() -> Result<()> {
        let a = Array::from_vec((0..24).map(|i| i as f64).collect()).reshape(&[2, 3, 4]);
        // The axpy shape `(a * k) + a`, which has its own specialised loop.
        let e = a.expr() * 2.0 + a.expr();
        assert!(e.will_fuse(), "axpy shape must fuse");
        let got = e.eval()?;
        assert_eq!(got.shape(), vec![2, 3, 4]);
        assert_bit_eq(&got, &(&(&a * 2.0) + &a), "3-D shape");
        Ok(())
    }
}

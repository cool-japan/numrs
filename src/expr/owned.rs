//! Owned, lifetime-free expression templates (`ExprNode`) and their entry
//! point [`IntoExpr::expr`].
//!
//! # What this module is
//!
//! An [`ExprNode<T>`] is a small, flat, pattern-matchable tree describing an
//! elementwise computation over one or more [`Array<T>`] operands. Building
//! one costs tens of nanoseconds per node regardless of array size (measured:
//! ~27 ns/node): a leaf is an `Array<T>` obtained by [`Clone`], which under
//! the crate's `Arc`-backed copy-on-write storage is an O(1) reference-count
//! bump, not a data copy.
//!
//! No arithmetic happens until [`ExprNode::eval`] is called. For the common
//! tree shapes -- listed under "What actually fuses" below -- that call
//! evaluates the whole tree in a **single pass** over the operand slices,
//! writing exactly one output buffer. Trees past those shapes are evaluated
//! eagerly instead, at exactly the speed of the eager code the caller would
//! have written; `eval()` is never slower than writing it out by hand, but it
//! is not always faster either.
//!
//! ```
//! use numrs2::prelude::*;
//!
//! let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
//! let b = Array::from_vec(vec![10.0, 20.0, 30.0]);
//! let c = Array::from_vec(vec![2.0, 2.0, 2.0]);
//!
//! // One pass, one allocation -- `b * c` is never materialized.
//! let fused = (a.expr() + b.expr() * c.expr()).eval()?;
//! assert_eq!(fused.to_vec(), vec![21.0, 42.0, 63.0]);
//! # Ok::<(), numrs2::error::NumRs2Error>(())
//! ```
//!
//! # Honest scope
//!
//! **Eager operator syntax never fuses.** `&a + &(&b * &c)` calls the
//! ordinary [`std::ops::Add`]/[`std::ops::Mul`] impls on [`Array<T>`], which
//! evaluate immediately and allocate one intermediate array per operator.
//! Those operators are deliberately left untouched: an operator that returned
//! a lazy node instead of an `Array` would change the type of every existing
//! expression in the crate and in user code. Fusion is **opt-in** and its
//! spelling is `.expr()` … `.eval()`:
//!
//! ```text
//! eager  (2 passes, 2 allocations):  &a + &(&b * &c)
//! fused  (1 pass,  1 allocation):    (a.expr() + b.expr() * c.expr()).eval()?
//! ```
//!
//! **What actually fuses.** [`ExprNode::eval`] takes the single-pass fused
//! path when all four of these hold:
//!
//! 1. the element type `T` is `f64` or `f32`;
//! 2. every leaf has the *same* shape (no broadcasting between leaves);
//! 3. every leaf is in standard (C-contiguous) layout, i.e.
//!    [`Array::as_slice`] returns `Some`;
//! 4. the tree is one of the shapes that has a single-pass loop: up to four
//!    leaves combined by one or two binary operators (`a + b`, `a * b + c`,
//!    `a + b * c`, `(a + b) * (c - d)`), the axpy shape `a * k + b` and its
//!    mirror, an [`ExprNode::Fma`] of three leaves, a leaf with a scalar, or a
//!    unary op on a leaf.
//!
//! Condition 4 exists because a general tree interpreter was written,
//! measured, and found *slower* than plain eager evaluation at every size --
//! see the `fused_eval` module's notes for the numbers. Rather than ship a
//! "fusion" that loses, anything past the specialised shapes takes the eager
//! fallback, where it is exactly as fast as the eager code the caller would
//! have written by hand.
//!
//! **What falls back.** Any other tree -- a transposed / otherwise
//! non-contiguous leaf, leaves of different (even validly broadcastable)
//! shapes, a shape past the specialised set, or any dtype other than
//! `f64`/`f32` -- is evaluated by
//! `eval_eager`, which walks the same tree calling the crate's ordinary eager
//! operations (`add_broadcast`, `multiply_broadcast`, …). The fallback is
//! **element-for-element identical** to what the user would have got by
//! writing the eager form by hand; it is a performance fallback, never a
//! semantic one. [`ExprNode::will_fuse`] reports which path a given tree will
//! take, without evaluating it.
//!
//! **Floating point.** No rewrite performed here reassociates or reorders any
//! floating-point operation, and the [`ExprNode::Fma`] node is a *loop*
//! fusion, not an FMA *instruction*: it evaluates `a * b + c` with the two
//! separate roundings that `(a * b) + c` has, never `f64::mul_add`'s single
//! rounding. Fused and eager results are therefore **bit-identical for every
//! finite, infinite and signed-zero value** -- there is no 1-ulp wobble, no
//! reassociation, and `0.0` never turns into `-0.0`.
//!
//! The two paths also agree on **where** a `NaN` appears: an element is `NaN`
//! on one exactly when it is `NaN` on the other. A `NaN`'s **payload and sign
//! bits are not part of the guarantee**, and that is deliberate. When a single
//! operation receives two *distinct* `NaN` operands, IEEE-754 §6.2.3 leaves
//! which payload propagates implementation-defined, neither Rust nor LLVM
//! specifies a choice, and LLVM may commute the operands of an `fadd`/`fmul`
//! -- so a one-pass fused loop and a two-pass eager spelling can each keep a
//! different operand's `NaN` from identical source. This is a property that
//! has to be *measured* rather than argued from the source, and it was: in a
//! standalone program containing no `numrs2` code, `(0.0 * inf) + NaN`
//! diverges between the two loops under `rustc -O`
//! (`0x7ff8000000000000` vs `0xfff8000000000000`) and agrees under `-O0`.
//! NumPy makes no `NaN`-payload promise either. See the `fused_eval` module's
//! notes and `tests/test_expr_fused_equivalence.rs`, which compares every
//! element of a few thousand random trees bit for bit, grants only that one
//! exemption, and counts each time it is taken
//! (`two_distinct_nans_into_one_add_may_differ_in_payload` pins the
//! counter-example).
//!
//! **Deferred to 0.6.0.** A `fused!` macro (compile-time expansion of an
//! expression written in ordinary infix syntax into a fused kernel) is *not*
//! part of this module; the runtime tree here is the whole feature. Reductions
//! (`sum`, `dot`) inside an expression are likewise out of scope -- an
//! `ExprNode` always evaluates to an array of the leaves' common shape.
//!
//! # Why this could not be written before
//!
//! The pre-existing expression machinery in `src/expr/` (`ArrayExpr<'a, T>`
//! and friends) borrows its operands, so every combinator carries a lifetime
//! parameter, and `a.lazy() + b.lazy()` cannot outlive the temporaries it was
//! built from -- which is exactly why operator overloading was never wired up
//! there (see [`crate::expr`]'s "Current Status" note). Copy-on-write storage
//! removes the constraint outright: [`ExprNode::Leaf`] *owns* its `Array<T>`,
//! so `ExprNode<T>` has no lifetime parameter at all, at the cost of one
//! atomic increment per leaf.
//!
//! This module shares no evaluation code with the older `Expr`/`SimdEval`
//! machinery, whose per-element `get_flat` dispatch chain is slower than the
//! eager operators it was meant to accelerate.

use crate::array::Array;
use crate::error::Result;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A binary elementwise operator in an [`ExprNode`] tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
}

impl BinOp {
    /// The operator's infix spelling, for [`fmt::Debug`] output.
    fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
        }
    }
}

/// A unary elementwise operator in an [`ExprNode`] tree.
///
/// [`UnaryOp::Neg`] is available for every element type that implements
/// [`std::ops::Neg`]. The four floating-point maths operators are `f64`/`f32`
/// only: on any other dtype [`ExprNode::eval`] returns
/// [`crate::error::NumRs2Error::NotImplemented`] rather than silently
/// producing a different answer, because the crate has no eager elementwise
/// `sqrt`/`exp`/`ln` for non-float dtypes to be equivalent *to*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// `-a`
    Neg,
    /// `a.abs()` (`f64`/`f32` only)
    Abs,
    /// `a.sqrt()` (`f64`/`f32` only)
    Sqrt,
    /// `a.exp()` (`f64`/`f32` only)
    Exp,
    /// `a.ln()` (`f64`/`f32` only)
    Ln,
}

impl UnaryOp {
    /// The operator's name, for [`fmt::Debug`] output.
    fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Exp => "exp",
            UnaryOp::Ln => "ln",
        }
    }
}

/// One node of an owned, lifetime-free elementwise expression tree.
///
/// The enum is deliberately **flat**: every combinator is a variant of this
/// one type rather than a distinct generic wrapper struct. That is what makes
/// a rewrite such as `Add(Mul(a, b), c) -> Fma(a, b, c)` a short `match`
/// ([`ExprNode::fuse_fma`]) instead of an unwritable type-level transformation
/// -- the reason the crate's older type-tower expression templates never
/// managed to fire a single fusion.
///
/// Build one with [`IntoExpr::expr`] and the operators; evaluate it with
/// [`ExprNode::eval`].
pub enum ExprNode<T> {
    /// An operand array, owned via an O(1) copy-on-write clone.
    Leaf(Array<T>),
    /// `lhs op rhs`, both operands elementwise arrays.
    Binary(BinOp, Box<ExprNode<T>>, Box<ExprNode<T>>),
    /// `expr op scalar` (scalar on the right).
    ScalarRhs(BinOp, Box<ExprNode<T>>, T),
    /// `scalar op expr` (scalar on the left).
    ///
    /// A separate variant rather than a rewrite of [`ExprNode::ScalarRhs`],
    /// because `-` and `/` do not commute and rewriting `s / e` into
    /// `e * (1/s)` would not be bit-identical to the eager form.
    ScalarLhs(BinOp, T, Box<ExprNode<T>>),
    /// `op(expr)`.
    Unary(UnaryOp, Box<ExprNode<T>>),
    /// `(a * b) + c`, evaluated in one pass with **two** roundings.
    ///
    /// This is loop fusion, not the `fma` machine instruction: see this
    /// module's "Floating point" note.
    Fma(Box<ExprNode<T>>, Box<ExprNode<T>>, Box<ExprNode<T>>),
}

impl<T: Clone> Clone for ExprNode<T> {
    /// Cloning a tree clones its leaves, which is one `Arc` bump each -- no
    /// element data is copied.
    fn clone(&self) -> Self {
        match self {
            ExprNode::Leaf(a) => ExprNode::Leaf(a.clone()),
            ExprNode::Binary(op, l, r) => ExprNode::Binary(*op, l.clone(), r.clone()),
            ExprNode::ScalarRhs(op, e, s) => ExprNode::ScalarRhs(*op, e.clone(), s.clone()),
            ExprNode::ScalarLhs(op, s, e) => ExprNode::ScalarLhs(*op, s.clone(), e.clone()),
            ExprNode::Unary(op, e) => ExprNode::Unary(*op, e.clone()),
            ExprNode::Fma(a, b, c) => ExprNode::Fma(a.clone(), b.clone(), c.clone()),
        }
    }
}

/// Prints the tree's *structure*, never its element data: a leaf shows only
/// its shape, so debugging a million-element expression stays readable.
impl<T: fmt::Debug + Clone> fmt::Debug for ExprNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprNode::Leaf(a) => write!(f, "Leaf(shape={:?})", a.shape()),
            ExprNode::Binary(op, l, r) => {
                write!(f, "({:?} {} {:?})", l, op.symbol(), r)
            }
            ExprNode::ScalarRhs(op, e, s) => write!(f, "({:?} {} {:?})", e, op.symbol(), s),
            ExprNode::ScalarLhs(op, s, e) => write!(f, "({:?} {} {:?})", s, op.symbol(), e),
            ExprNode::Unary(op, e) => write!(f, "{}({:?})", op.name(), e),
            ExprNode::Fma(a, b, c) => write!(f, "fma({:?}, {:?}, {:?})", a, b, c),
        }
    }
}

impl<T> ExprNode<T> {
    /// Number of leaves in the tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            ExprNode::Leaf(_) => 1,
            ExprNode::Binary(_, l, r) => l.leaf_count() + r.leaf_count(),
            ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => e.leaf_count(),
            ExprNode::Unary(_, e) => e.leaf_count(),
            ExprNode::Fma(a, b, c) => a.leaf_count() + b.leaf_count() + c.leaf_count(),
        }
    }

    /// Length of the longest root-to-leaf path (a bare leaf has depth 1).
    pub fn depth(&self) -> usize {
        match self {
            ExprNode::Leaf(_) => 1,
            ExprNode::Binary(_, l, r) => 1 + l.depth().max(r.depth()),
            ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => 1 + e.depth(),
            ExprNode::Unary(_, e) => 1 + e.depth(),
            ExprNode::Fma(a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
        }
    }

    /// Append every leaf array, left to right, to `out`.
    pub(super) fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a Array<T>>) {
        match self {
            ExprNode::Leaf(a) => out.push(a),
            ExprNode::Binary(_, l, r) => {
                l.collect_leaves(out);
                r.collect_leaves(out);
            }
            ExprNode::ScalarRhs(_, e, _) | ExprNode::ScalarLhs(_, _, e) => e.collect_leaves(out),
            ExprNode::Unary(_, e) => e.collect_leaves(out),
            ExprNode::Fma(a, b, c) => {
                a.collect_leaves(out);
                b.collect_leaves(out);
                c.collect_leaves(out);
            }
        }
    }

    /// Rewrite every `Add(Mul(a, b), c)` in the tree into `Fma(a, b, c)`.
    ///
    /// This is the whole "fusion pass": ten lines of `match`, possible only
    /// because [`ExprNode`] is one flat enum.
    ///
    /// Calling it is **optional** -- [`ExprNode::eval`] recognises the
    /// un-rewritten `Add(Mul(..), ..)` shape and fuses it into the same
    /// single-pass loop either way. It exists so the canonical form can be
    /// produced and inspected (and so the rewrite is testable in isolation).
    ///
    /// Only the canonical operand order `(a * b) + c` is rewritten, never
    /// `c + (a * b)`: floating-point addition commutes in value but not
    /// necessarily in NaN *payload*, so this rewrite preserves source-level
    /// operand order exactly and changes nothing the language defines. (What
    /// the backend then does with that order is a separate matter -- LLVM is
    /// free to commute an `fadd`, which is why the module docs above put a
    /// `NaN`'s payload and sign bits outside the fused/eager guarantee. The
    /// point here is that the *rewrite* adds no divergence of its own.)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use numrs2::expr::ExprNode;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0]);
    /// let e = (a.expr() * a.expr() + a.expr()).fuse_fma();
    /// assert!(matches!(e, ExprNode::Fma(..)));
    /// ```
    pub fn fuse_fma(self) -> Self {
        match self {
            ExprNode::Binary(BinOp::Add, lhs, rhs) => match *lhs {
                ExprNode::Binary(BinOp::Mul, a, b) => ExprNode::Fma(
                    Box::new(a.fuse_fma()),
                    Box::new(b.fuse_fma()),
                    Box::new(rhs.fuse_fma()),
                ),
                other => ExprNode::Binary(
                    BinOp::Add,
                    Box::new(other.fuse_fma()),
                    Box::new(rhs.fuse_fma()),
                ),
            },
            ExprNode::Binary(op, l, r) => {
                ExprNode::Binary(op, Box::new(l.fuse_fma()), Box::new(r.fuse_fma()))
            }
            ExprNode::ScalarRhs(op, e, s) => ExprNode::ScalarRhs(op, Box::new(e.fuse_fma()), s),
            ExprNode::ScalarLhs(op, s, e) => ExprNode::ScalarLhs(op, s, Box::new(e.fuse_fma())),
            ExprNode::Unary(op, e) => ExprNode::Unary(op, Box::new(e.fuse_fma())),
            ExprNode::Fma(a, b, c) => ExprNode::Fma(
                Box::new(a.fuse_fma()),
                Box::new(b.fuse_fma()),
                Box::new(c.fuse_fma()),
            ),
            leaf => leaf,
        }
    }

    /// `self.abs()` as an expression node (`f64`/`f32` only -- see
    /// [`UnaryOp`]).
    pub fn abs(self) -> Self {
        ExprNode::Unary(UnaryOp::Abs, Box::new(self))
    }

    /// `self.sqrt()` as an expression node (`f64`/`f32` only).
    pub fn sqrt(self) -> Self {
        ExprNode::Unary(UnaryOp::Sqrt, Box::new(self))
    }

    /// `self.exp()` as an expression node (`f64`/`f32` only).
    pub fn exp(self) -> Self {
        ExprNode::Unary(UnaryOp::Exp, Box::new(self))
    }

    /// `self.ln()` as an expression node (`f64`/`f32` only).
    pub fn ln(self) -> Self {
        ExprNode::Unary(UnaryOp::Ln, Box::new(self))
    }
}

/// Turn an [`Array`] into an expression [`ExprNode::Leaf`].
///
/// Named `expr()` rather than `lazy()` because [`crate::expr::LazyEval::lazy`]
/// already exists in the prelude for the older borrowing expression templates
/// and means something different (and slower).
///
/// The receiver is `&self`, so `a.expr()` never moves `a`; the leaf holds an
/// O(1) copy-on-write clone of it.
pub trait IntoExpr<T: Clone> {
    /// Wrap `self` as an expression leaf.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    ///
    /// // Building the tree copies no element data; `a` and `b` stay usable.
    /// let e = a.expr() * 2.0 + b.expr();
    /// assert_eq!(e.eval()?.to_vec(), vec![6.0, 9.0, 12.0]);
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0]);
    /// # Ok::<(), numrs2::error::NumRs2Error>(())
    /// ```
    fn expr(&self) -> ExprNode<T>;
}

impl<T: Clone> IntoExpr<T> for Array<T> {
    fn expr(&self) -> ExprNode<T> {
        ExprNode::Leaf(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Node-to-node operators
// ---------------------------------------------------------------------------

macro_rules! impl_node_binop {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<T> $trait<ExprNode<T>> for ExprNode<T> {
            type Output = ExprNode<T>;

            fn $method(self, rhs: ExprNode<T>) -> ExprNode<T> {
                ExprNode::Binary(BinOp::$variant, Box::new(self), Box::new(rhs))
            }
        }
    };
}

impl_node_binop!(Add, add, Add);
impl_node_binop!(Sub, sub, Sub);
impl_node_binop!(Mul, mul, Mul);
impl_node_binop!(Div, div, Div);

impl<T> Neg for ExprNode<T> {
    type Output = ExprNode<T>;

    fn neg(self) -> ExprNode<T> {
        ExprNode::Unary(UnaryOp::Neg, Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// Scalar operators
// ---------------------------------------------------------------------------
//
// These must be written out concretely for `f64` and `f32`. A blanket
// `impl<T> Add<T> for ExprNode<T>` would overlap the `Add<ExprNode<T>> for
// ExprNode<T>` impl above (nothing stops `T` from itself being an
// `ExprNode<_>`), and the scalar-on-the-left direction is a foreign trait on a
// foreign type (`impl Add<ExprNode<f64>> for f64`), which the orphan rule only
// permits because the local `ExprNode` appears in the parameter list -- and
// only for a concrete self type. Two dtypes x four operators x two sides = 16
// impls, generated by the two macros below.

macro_rules! impl_scalar_rhs {
    ($ty:ty, $trait:ident, $method:ident, $variant:ident) => {
        impl $trait<$ty> for ExprNode<$ty> {
            type Output = ExprNode<$ty>;

            fn $method(self, rhs: $ty) -> ExprNode<$ty> {
                ExprNode::ScalarRhs(BinOp::$variant, Box::new(self), rhs)
            }
        }
    };
}

macro_rules! impl_scalar_lhs {
    ($ty:ty, $trait:ident, $method:ident, $variant:ident) => {
        impl $trait<ExprNode<$ty>> for $ty {
            type Output = ExprNode<$ty>;

            fn $method(self, rhs: ExprNode<$ty>) -> ExprNode<$ty> {
                ExprNode::ScalarLhs(BinOp::$variant, self, Box::new(rhs))
            }
        }
    };
}

macro_rules! impl_scalar_ops {
    ($ty:ty) => {
        impl_scalar_rhs!($ty, Add, add, Add);
        impl_scalar_rhs!($ty, Sub, sub, Sub);
        impl_scalar_rhs!($ty, Mul, mul, Mul);
        impl_scalar_rhs!($ty, Div, div, Div);
        impl_scalar_lhs!($ty, Add, add, Add);
        impl_scalar_lhs!($ty, Sub, sub, Sub);
        impl_scalar_lhs!($ty, Mul, mul, Mul);
        impl_scalar_lhs!($ty, Div, div, Div);
    };
}

impl_scalar_ops!(f64);
impl_scalar_ops!(f32);

// ---------------------------------------------------------------------------
// Evaluation entry points (the engine itself lives in `fused_eval`)
// ---------------------------------------------------------------------------

impl<T> ExprNode<T>
where
    T: Clone
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    /// Evaluate the expression, fusing the tree into a single pass when its
    /// shape and operands qualify, and evaluating it eagerly when they do not
    /// (see this module's "Honest scope"). [`ExprNode::will_fuse`] reports
    /// which, without evaluating.
    ///
    /// The operator bounds are exactly those of the eager operators this can
    /// fall back to, which is why unsigned integer dtypes (no
    /// [`std::ops::Neg`]) are not supported here -- use the eager operators
    /// for those.
    ///
    /// # Errors
    ///
    /// Returns the same errors the eager operations return -- principally
    /// [`crate::error::NumRs2Error::ShapeMismatch`] when two leaves are
    /// neither equal-shaped nor broadcast-compatible -- plus
    /// [`crate::error::NumRs2Error::NotImplemented`] when a floating-point
    /// maths [`UnaryOp`] is applied to a non-`f64`/`f32` dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
    /// let b = Array::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    ///
    /// let fused = (a.expr() * b.expr() - 1.0).eval()?;
    /// assert_eq!(fused.to_vec(), vec![-0.5, 0.0, 0.5, 1.0]);
    ///
    /// // Identical to the eager spelling, bit for bit.
    /// let eager = &(&a * &b) - 1.0;
    /// assert_eq!(fused.to_vec(), eager.to_vec());
    /// # Ok::<(), numrs2::error::NumRs2Error>(())
    /// ```
    ///
    /// A tree that cannot be fused still evaluates, via the eager fallback:
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    /// let t = a.transpose_axis(0, 1); // non-contiguous view
    ///
    /// let e = t.expr() + t.expr();
    /// assert!(!e.will_fuse());
    /// assert_eq!(e.eval()?.to_vec(), (&t + &t).to_vec());
    /// # Ok::<(), numrs2::error::NumRs2Error>(())
    /// ```
    pub fn eval(&self) -> Result<Array<T>> {
        super::fused_eval::eval(self)
    }

    /// Whether [`ExprNode::eval`] will take the single-pass fused path for
    /// this tree (`false` means it falls back to eager evaluation, with an
    /// identical result).
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
    /// assert!((a.expr() + a.expr()).will_fuse());
    ///
    /// // Different (but broadcastable) shapes fall back.
    /// let row = Array::from_vec(vec![1.0_f64, 2.0, 3.0]).reshape(&[1, 3]);
    /// let col = Array::from_vec(vec![1.0_f64, 2.0, 3.0]).reshape(&[3, 1]);
    /// assert!(!(row.expr() + col.expr()).will_fuse());
    /// ```
    pub fn will_fuse(&self) -> bool {
        super::fused_eval::will_fuse(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_wraps_array_without_moving_it() {
        let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let e = a.expr();
        assert!(matches!(e, ExprNode::Leaf(_)));
        // `a` is still usable: `expr()` takes `&self`.
        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn leaf_shares_storage_with_its_source() {
        let a = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(a.is_unique());
        let e = a.expr();
        // The leaf holds an Arc bump, not a copy.
        assert!(!a.is_unique());
        drop(e);
        assert!(a.is_unique());
    }

    #[test]
    fn operators_build_the_expected_tree() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);
        let e = a.expr() + a.expr() * a.expr();
        match &e {
            ExprNode::Binary(BinOp::Add, l, r) => {
                assert!(matches!(**l, ExprNode::Leaf(_)));
                assert!(matches!(**r, ExprNode::Binary(BinOp::Mul, _, _)));
            }
            other => panic!("unexpected tree: {other:?}"),
        }
        assert_eq!(e.leaf_count(), 3);
        assert_eq!(e.depth(), 3);
    }

    #[test]
    fn scalar_operators_build_scalar_nodes_on_both_sides() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);
        assert!(matches!(
            a.expr() + 1.0,
            ExprNode::ScalarRhs(BinOp::Add, _, _)
        ));
        assert!(matches!(
            2.0 - a.expr(),
            ExprNode::ScalarLhs(BinOp::Sub, _, _)
        ));
        let b = Array::from_vec(vec![1.0_f32, 2.0]);
        assert!(matches!(
            b.expr() / 4.0_f32,
            ExprNode::ScalarRhs(BinOp::Div, _, _)
        ));
        assert!(matches!(
            4.0_f32 / b.expr(),
            ExprNode::ScalarLhs(BinOp::Div, _, _)
        ));
    }

    #[test]
    fn neg_and_math_builders() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);
        assert!(matches!(-a.expr(), ExprNode::Unary(UnaryOp::Neg, _)));
        assert!(matches!(a.expr().abs(), ExprNode::Unary(UnaryOp::Abs, _)));
        assert!(matches!(a.expr().sqrt(), ExprNode::Unary(UnaryOp::Sqrt, _)));
        assert!(matches!(a.expr().exp(), ExprNode::Unary(UnaryOp::Exp, _)));
        assert!(matches!(a.expr().ln(), ExprNode::Unary(UnaryOp::Ln, _)));
    }

    #[test]
    fn fuse_fma_rewrites_canonical_order_only() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);

        let canonical = (a.expr() * a.expr() + a.expr()).fuse_fma();
        assert!(matches!(canonical, ExprNode::Fma(..)));

        // `c + (a * b)` is deliberately left alone.
        let swapped = (a.expr() + a.expr() * a.expr()).fuse_fma();
        assert!(matches!(swapped, ExprNode::Binary(BinOp::Add, _, _)));
    }

    #[test]
    fn fuse_fma_rewrites_nested_occurrences() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);
        // ((a*a + a) * a + a)
        let e = ((a.expr() * a.expr() + a.expr()) * a.expr() + a.expr()).fuse_fma();
        match e {
            ExprNode::Fma(x, _, _) => assert!(matches!(*x, ExprNode::Fma(..))),
            other => panic!("outer rewrite missing: {other:?}"),
        }
    }

    #[test]
    fn debug_shows_structure_not_data() {
        let a = Array::from_vec(vec![1.0_f64; 1000]);
        let s = format!("{:?}", a.expr() + a.expr() * 2.0);
        assert_eq!(s, "(Leaf(shape=[1000]) + (Leaf(shape=[1000]) * 2.0))");
    }

    #[test]
    fn clone_of_a_tree_copies_no_data() {
        let a = Array::from_vec(vec![1.0_f64; 64]);
        let e = a.expr() + a.expr();
        let f = e.clone();
        assert_eq!(f.leaf_count(), 2);
        // Four live leaves (two per tree) plus `a` itself all share one buffer.
        assert!(!a.is_unique());
    }

    #[test]
    fn depth_and_leaf_count_cover_every_variant() {
        let a = Array::from_vec(vec![1.0_f64, 2.0]);
        let e = ExprNode::Fma(
            Box::new(a.expr()),
            Box::new(ExprNode::ScalarLhs(
                BinOp::Sub,
                1.0,
                Box::new(a.expr().sqrt()),
            )),
            Box::new(ExprNode::ScalarRhs(BinOp::Div, Box::new(-a.expr()), 3.0)),
        );
        assert_eq!(e.leaf_count(), 3);
        assert_eq!(e.depth(), 4);
    }
}

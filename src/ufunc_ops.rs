//! Generic NumPy `ufunc`-method machinery: `reduce`, `accumulate`, `outer`,
//! `reduceat`, `at`, and `where=` support.
//!
//! NumPy exposes these as *methods on a ufunc object* (`np.add.reduce`,
//! `np.add.at`, ...). Rust has no equivalent late-bound "ufunc object", so
//! this module represents the operation itself as the [`UfuncOp`] enum and
//! every method below as a free function taking one as its first argument:
//! `ufunc_reduce(UfuncOp::Add, &a, ..)` plays the role of `np.add.reduce(a,
//! ..)`. An enum (rather than a bare closure) is deliberate: [`UfuncOp`]
//! carries per-operation *data* -- its identity element, its display name --
//! that a closure has no way to expose, and that [`ufunc_reduce`]'s
//! empty-array handling depends on.
//!
//! # `axis: Option<isize>` -- one crate-wide convention, not two NumPy ones
//!
//! Every axis-aware function here takes `axis: Option<isize>` and gives
//! `None` the same meaning this crate's other axis-aware reductions already
//! do (`math::sum`/`max`/`min`/`cumsum`/`cumprod`): **flatten the array
//! first, then operate on the flat result**. `Some(ax)` reduces/accumulates
//! along exactly that one axis, with negative axes counting from the end
//! (`-1` is the last axis), matching NumPy.
//!
//! This is a deliberate, single, predictable rule -- but it is worth being
//! explicit about where it lines up with real NumPy and where it does not,
//! because NumPy itself is not internally consistent here:
//!
//! - [`ufunc_reduce`]'s `axis: None` matches `np.add.reduce(a, axis=None)`
//!   *exactly* (verified against NumPy 2.4.2: `np.add.reduce(a, axis=None)`
//!   flattens and reduces every axis) -- **not** `np.add.reduce(a)`'s
//!   omitted-argument default, which is `axis=0` (reduce only the first
//!   axis). Rust's `Option<isize>` cannot distinguish "argument omitted"
//!   from "argument explicitly `None`" the way Python's calling convention
//!   can, so this crate picks the flatten meaning, matching every sibling
//!   reduction (`math::sum(&a, None, false)`, etc.) instead of ufunc
//!   `reduce`'s own single-argument default.
//! - [`ufunc_accumulate`]'s `axis: None` also flattens -- matching this
//!   crate's `cumsum`/`cumprod` (which it delegates straight to for
//!   `Add`/`Multiply`, see below), but **not** matching raw NumPy: real
//!   `np.add.accumulate(a, axis=None)` raises `ValueError: accumulate does
//!   not allow multiple axes` for any `a` with `ndim > 1` (accumulate has no
//!   flatten mode at all in NumPy; `axis=None` there means "reduce over
//!   *all* axes to find the one there is", which only exists when there is
//!   exactly one). For a 1-D array the two conventions coincide exactly, so
//!   every 1-D test below is pinned directly against
//!   `np.add.accumulate`/`np.multiply.accumulate`. For `ndim > 1` this
//!   module is strictly more capable than raw NumPy (it flattens instead of
//!   raising) rather than incompatible with it.
//!
//! # Reduce-all-axes shape: `[1]`, not a true 0-d array
//!
//! NumPy's `np.add.reduce(a)` (no axis, or a full-flatten reduce without
//! `keepdims`) returns a genuine 0-d array (`shape == ()`, a bare scalar
//! wrapped in an ndarray). This crate has no 0-d array representation in
//! active use, and [`crate::math::aggregation`]'s `sum`/`max`/`min` already
//! established the convention this module follows instead: a full,
//! non-`keepdims` reduction returns shape `[1]` (a length-1 1-D array), via
//! plain `Array::from_vec(vec![value])`. This is a deliberate, precedented
//! divergence from NumPy's own shape, not an oversight -- see this module's
//! tests for the exact shapes produced at every `keepdims`/`axis`
//! combination, each checked against what real NumPy 2.4.2 returns (with
//! this one documented exception).
//!
//! # `where=`: no broadcasting, no `out=`
//!
//! [`ufunc_where`] (and its named wrappers [`add_where`], [`subtract_where`],
//! [`multiply_where`], [`divide_where`]) implement NumPy's `where=` kwarg
//! for binary ufuncs: `out[i] = op(a[i], b[i])` if `mask[i]`, else `out[i] =
//! a[i]`. Two simplifications relative to real NumPy, both deliberate:
//!
//! - `a`, `b`, and `mask` must all have exactly the same shape -- no
//!   broadcasting. The task this module implements states the formula
//!   index-for-index (`a[i]`/`b[i]`/`mask[i]`), which is only unambiguous
//!   without broadcasting; a shape mismatch on any pair is a
//!   [`NumRs2Error::ShapeMismatch`].
//! - There is no `out=` parameter -- this module always returns a fresh
//!   [`Array`]. Real NumPy, called with `where=` and no `out=`, emits a
//!   `UserWarning` and leaves the masked-out lanes as **uninitialized
//!   memory** (verified against NumPy 2.4.2). This module instead behaves
//!   as if `out=a.clone()` had been passed -- the single most common real
//!   usage of `where=` without a fresh buffer -- so masked-out lanes keep
//!   `a`'s original values instead of being garbage. Verified to match
//!   `np.add(a, b, where=mask, out=np.copy(a))` exactly.
//!
//! # Dispatch
//!
//! [`ufunc_reduce`]'s **full** reduction (`axis: None`) for `Add`/`Multiply`
//! on `T` that is concretely `f64`/`f32` routes through
//! [`crate::kernels::reduce`]'s `sum_f64`/`sum_f32`/`prod_f64`/`prod_f32`
//! (themselves SIMD-below/parallel-above-`PARALLEL_MIN_LEN`, per that
//! module's docs) instead of a scalar fold. Every other combination --
//! `Some(axis)` reductions, `Maximum`/`Minimum` at any axis, any other `T`,
//! and every other function in this module -- uses a plain, generic fold.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::{borrow::operand, cast, reduce};
use num_traits::{Float, ToPrimitive};

// =============================================================================
// UfuncOp
// =============================================================================

/// A binary NumPy-style universal-function operation, playing the role of a
/// "ufunc object" (`np.add`, `np.maximum`, ...) for the free functions in
/// this module. See the module docs for why this is an enum rather than a
/// bare closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UfuncOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
}

impl UfuncOp {
    /// The NumPy name of this operation, used in error messages (e.g.
    /// `"zero-size array to reduction operation maximum which has no
    /// identity"`, matching NumPy's own wording for the same condition).
    pub fn name(self) -> &'static str {
        match self {
            UfuncOp::Add => "add",
            UfuncOp::Subtract => "subtract",
            UfuncOp::Multiply => "multiply",
            UfuncOp::Divide => "divide",
            UfuncOp::Maximum => "maximum",
            UfuncOp::Minimum => "minimum",
        }
    }

    /// This operation's identity element, if it has one -- `0` for `Add`,
    /// `1` for `Multiply`, matching `np.add.identity`/`np.multiply.identity`.
    /// `None` for `Subtract`/`Divide`/`Maximum`/`Minimum`, matching NumPy
    /// (`np.maximum.identity is None`): reducing an empty array with one of
    /// these and no `initial` is an error, not a silently-produced value.
    pub fn identity<T: Float>(self) -> Option<T> {
        match self {
            UfuncOp::Add => Some(T::zero()),
            UfuncOp::Multiply => Some(T::one()),
            UfuncOp::Subtract | UfuncOp::Divide | UfuncOp::Maximum | UfuncOp::Minimum => None,
        }
    }

    /// Apply this operation to two scalars.
    ///
    /// `Maximum`/`Minimum` propagate `NaN` symmetrically in either operand
    /// (`apply(NaN, x) == apply(x, NaN) == NaN`), matching NumPy's
    /// element-wise `np.maximum`/`np.minimum` (verified: `np.maximum(np.nan,
    /// 5.0)` and `np.maximum(5.0, np.nan)` are both `nan`). This is
    /// deliberately *not* implemented as `if a > b { a } else { b }`: that
    /// expression is `false` for *every* comparison involving a `NaN`
    /// (IEEE-754), so it silently returns `b` when `a` is `NaN` -- correct
    /// only when the `NaN` happens to be in the second operand.
    #[inline]
    pub fn apply<T: Float>(self, a: T, b: T) -> T {
        match self {
            UfuncOp::Add => a + b,
            UfuncOp::Subtract => a - b,
            UfuncOp::Multiply => a * b,
            UfuncOp::Divide => a / b,
            UfuncOp::Maximum => {
                if a.is_nan() || b.is_nan() {
                    T::nan()
                } else if a > b {
                    a
                } else {
                    b
                }
            }
            UfuncOp::Minimum => {
                if a.is_nan() || b.is_nan() {
                    T::nan()
                } else if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Normalize a possibly-negative NumPy-style axis against `ndim`, matching
/// the `(ndim as isize + ax) as usize` idiom used throughout this crate
/// (`math::aggregation`, `math::statistics::cumsum`), but checking the sign
/// explicitly before casting rather than relying on `isize -> usize`
/// wraparound to land out of bounds.
fn normalize_axis(ax: isize, ndim: usize) -> Result<usize> {
    let normalized = if ax < 0 { ax + ndim as isize } else { ax };
    if normalized < 0 || normalized as usize >= ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "axis {ax} out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
}

/// Convert a flat, row-major (C-order) index into per-dimension indices for
/// `shape`.
///
/// Precondition (never violated by any call site in this module): every
/// entry of `shape` is nonzero, and `flat < shape.iter().product()`. Every
/// caller here only ever iterates `flat` over `0..shape.iter().product()`,
/// so if that product is `0` (some dimension is `0`) the range is empty and
/// this function is never actually invoked -- see e.g. [`reduce_along_axis`]
/// and [`ufunc_at`].
fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        idx[i] = flat % shape[i];
        flat /= shape[i];
    }
    idx
}

/// Fold `op` over `lane`, seeded by `initial` if given, else by the lane's
/// own first element -- the shared core of every "reduce one 1-D lane"
/// operation in this module ([`ufunc_reduce`]'s per-axis lanes,
/// [`reduceat_along_axis`]'s per-segment lanes).
///
/// - `initial = Some(v)`: folds `v` with *every* element of `lane` (even if
///   `lane` is empty, in which case the result is `v` unchanged) -- matches
///   `np.add.reduce(a, initial=v)` exactly, including on an empty lane
///   (verified: `np.add.reduce(np.zeros((3, 0)), axis=1, initial=7)` gives
///   `[7, 7, 7]`).
/// - `initial = None`, `lane` non-empty: seeds from the lane's own first
///   element, matching NumPy's default reduce (this matters for
///   non-commutative/non-associative ops: `np.subtract.reduce([1,2,3])` is
///   `1 - 2 - 3 = -4`, not `0 - 1 - 2 - 3`).
/// - `initial = None`, `lane` empty: `op`'s identity element if it has one
///   (`Add` -> `0`, `Multiply` -> `1`), else an error -- matching NumPy's
///   `ValueError: zero-size array to reduction operation {name} which has
///   no identity` for `np.maximum.reduce`/`np.minimum.reduce` on an empty
///   input.
fn reduce_lane<T: Float>(
    op: UfuncOp,
    mut lane: impl Iterator<Item = T>,
    initial: Option<T>,
) -> Result<T> {
    match initial {
        Some(seed) => Ok(lane.fold(seed, |acc, x| op.apply(acc, x))),
        None => match lane.next() {
            Some(first) => Ok(lane.fold(first, |acc, x| op.apply(acc, x))),
            None => op.identity().ok_or_else(|| {
                NumRs2Error::InvalidOperation(format!(
                    "zero-size array to reduction operation {} which has no identity",
                    op.name()
                ))
            }),
        },
    }
}

// =============================================================================
// ufunc_reduce
// =============================================================================

/// Full-array (`axis: None`) reduction, with the `f64`/`f32` fast path for
/// `Add`/`Multiply` described in the module docs.
fn reduce_full<T: Float + Clone + 'static>(
    op: UfuncOp,
    data: &[T],
    initial: Option<T>,
) -> Result<T> {
    match op {
        UfuncOp::Add => {
            if let Some(s) = cast::as_f64(data) {
                let seed = initial.and_then(|v| v.to_f64()).unwrap_or(0.0);
                return Ok(cast::f64_to(seed + reduce::sum_f64(s))
                    .expect("T == f64 per cast::as_f64 match"));
            }
            if let Some(s) = cast::as_f32(data) {
                let seed = initial.and_then(|v| v.to_f32()).unwrap_or(0.0);
                return Ok(cast::f32_to(seed + reduce::sum_f32(s))
                    .expect("T == f32 per cast::as_f32 match"));
            }
        }
        UfuncOp::Multiply => {
            if let Some(s) = cast::as_f64(data) {
                let seed = initial.and_then(|v| v.to_f64()).unwrap_or(1.0);
                return Ok(cast::f64_to(seed * reduce::prod_f64(s))
                    .expect("T == f64 per cast::as_f64 match"));
            }
            if let Some(s) = cast::as_f32(data) {
                let seed = initial.and_then(|v| v.to_f32()).unwrap_or(1.0);
                return Ok(cast::f32_to(seed * reduce::prod_f32(s))
                    .expect("T == f32 per cast::as_f32 match"));
            }
        }
        UfuncOp::Subtract | UfuncOp::Divide | UfuncOp::Maximum | UfuncOp::Minimum => {}
    }
    reduce_lane(op, data.iter().copied(), initial)
}

/// Wrap a full-array reduction's scalar result per `keepdims`, matching
/// `math::aggregation::sum`'s `axis: None` convention (shape `[1]` when not
/// `keepdims`, `[1; ndim]` when `keepdims` -- see the module docs for why
/// this is `[1]` rather than NumPy's true 0-d shape).
fn wrap_full_result<T: Float + Clone>(value: T, ndim: usize, keepdims: bool) -> Result<Array<T>> {
    if keepdims {
        Array::from_vec_shape(vec![value], &vec![1usize; ndim])
    } else {
        Ok(Array::from_vec(vec![value]))
    }
}

/// `axis: Some(axis)` reduction: one call to [`reduce_lane`] per lane
/// perpendicular to `axis`, walking each lane via a fixed `axis_stride`
/// step over `a`'s own logical (C-order) flat data -- the same
/// stride-stepping shape `math::statistics::cumsum_no_out`'s `Some(axis)`
/// branch uses, generalized from a hardcoded running `+` to `op.apply`.
fn reduce_along_axis<T: Float + Clone>(
    op: UfuncOp,
    a: &Array<T>,
    axis: usize,
    keepdims: bool,
    initial: Option<T>,
) -> Result<Array<T>> {
    let shape = a.shape();
    let axis_size = shape[axis];
    let flat = operand(a);

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let axis_stride = strides[axis];

    // Every dimension except `axis`, in their original relative order --
    // this is exactly `out_shape` modulo `axis` itself being removed
    // (never-`keepdims`) or collapsed to size 1 (`keepdims`), and inserting
    // or removing a size-1 axis never changes a C-order flat enumeration,
    // so `other_shape`'s flat index doubles as `out_shape`'s flat index
    // regardless of `keepdims`.
    let other_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .collect();
    let n_outer: usize = other_shape.iter().product();

    let mut out_shape = shape.clone();
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    if out_shape.is_empty() {
        // 1-D input reduced along its only axis without keepdims: NumPy
        // returns a true 0-d array here: this crate does not, and instead
        // matches `math::aggregation::sum`/`max`/`min`'s established `[1]`
        // convention -- see the module docs.
        out_shape.push(1);
    }

    let mut result = vec![T::zero(); n_outer];
    for outer_idx in 0..n_outer {
        let other_indices = unravel_index(outer_idx, &other_shape);
        let mut base = 0usize;
        let mut oi = 0usize;
        for (i, &stride) in strides.iter().enumerate() {
            if i == axis {
                continue;
            }
            base += other_indices[oi] * stride;
            oi += 1;
        }
        let lane = (0..axis_size).map(|k| flat[base + k * axis_stride]);
        result[outer_idx] = reduce_lane(op, lane, initial)?;
    }

    Array::from_vec_shape(result, &out_shape)
}

/// `op.reduce(a, axis=axis, keepdims=keepdims, initial=initial)`: NumPy's
/// `np.add.reduce`/`np.multiply.reduce`/`np.maximum.reduce`/
/// `np.minimum.reduce`, generalized over [`UfuncOp`]. See the module docs
/// for the `axis: None` convention, the `[1]`-not-0-d shape convention, and
/// which combination gets the `f64`/`f32` fast path.
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::{ufunc_reduce, UfuncOp};
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
///
/// // np.add.reduce(a, axis=1) == [6.0, 15.0]
/// let row_sums = ufunc_reduce(UfuncOp::Add, &a, Some(1), false, None)
///     .expect("reduce should succeed");
/// assert_eq!(row_sums.to_vec(), vec![6.0, 15.0]);
///
/// // np.maximum.reduce(a, axis=0) == [4.0, 5.0, 6.0]
/// let col_max = ufunc_reduce(UfuncOp::Maximum, &a, Some(0), false, None)
///     .expect("reduce should succeed");
/// assert_eq!(col_max.to_vec(), vec![4.0, 5.0, 6.0]);
///
/// // np.maximum.reduce(np.array([]), initial=5) == 5.0 (no identity, but `initial` covers it)
/// let empty: Array<f64> = Array::from_vec(vec![]);
/// let with_initial = ufunc_reduce(UfuncOp::Maximum, &empty, None, false, Some(5.0))
///     .expect("initial makes an empty maximum.reduce succeed");
/// assert_eq!(with_initial.to_vec(), vec![5.0]);
///
/// // np.maximum.reduce(np.array([])) with no initial: ValueError (no identity)
/// assert!(ufunc_reduce(UfuncOp::Maximum, &empty, None, false, None).is_err());
/// ```
pub fn ufunc_reduce<T>(
    op: UfuncOp,
    a: &Array<T>,
    axis: Option<isize>,
    keepdims: bool,
    initial: Option<T>,
) -> Result<Array<T>>
where
    T: Float + Clone + 'static,
{
    match axis {
        None => {
            let flat = operand(a);
            let value = reduce_full(op, &flat, initial)?;
            wrap_full_result(value, a.ndim(), keepdims)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            reduce_along_axis(op, a, axis, keepdims, initial)
        }
    }
}

// =============================================================================
// ufunc_accumulate
// =============================================================================

/// `axis: None` accumulate: flatten first, then run a single cumulative
/// fold in flat order -- the non-`Add`/`Multiply` twin of
/// `math::statistics::cumsum_no_out`'s `axis: None` branch (which this
/// module delegates straight to for `Add`; see [`ufunc_accumulate`]).
fn accumulate_flat<T: Float + Clone>(op: UfuncOp, a: &Array<T>) -> Result<Array<T>> {
    if a.is_empty() {
        return Ok(a.clone());
    }
    let flat = operand(a);
    let mut result = Vec::with_capacity(flat.len());
    let mut acc = flat[0];
    result.push(acc);
    for &x in flat.iter().skip(1) {
        acc = op.apply(acc, x);
        result.push(acc);
    }
    Ok(Array::from_vec(result))
}

/// `axis: Some(axis)` accumulate: same stride-stepping shape as
/// [`reduce_along_axis`], but every partial fold along the lane is written
/// back (running accumulation) instead of only the final one.
fn accumulate_along_axis<T: Float + Clone>(
    op: UfuncOp,
    a: &Array<T>,
    axis: usize,
) -> Result<Array<T>> {
    if a.is_empty() {
        return Ok(a.clone());
    }
    let shape = a.shape();
    let axis_size = shape[axis];
    let flat = operand(a);

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let axis_stride = strides[axis];

    let other_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .collect();
    let n_outer: usize = other_shape.iter().product();

    let mut result = vec![T::zero(); flat.len()];
    for outer_idx in 0..n_outer {
        let other_indices = unravel_index(outer_idx, &other_shape);
        let mut base = 0usize;
        let mut oi = 0usize;
        for (i, &stride) in strides.iter().enumerate() {
            if i == axis {
                continue;
            }
            base += other_indices[oi] * stride;
            oi += 1;
        }
        let mut acc = flat[base];
        result[base] = acc;
        for k in 1..axis_size {
            let pos = base + k * axis_stride;
            acc = op.apply(acc, flat[pos]);
            result[pos] = acc;
        }
    }

    Array::from_vec_shape(result, &shape)
}

/// `op.accumulate(a, axis=axis)`: NumPy's `np.add.accumulate`/
/// `np.multiply.accumulate`/`np.maximum.accumulate`/
/// `np.minimum.accumulate`/etc., generalized over [`UfuncOp`]. `Add` and
/// `Multiply` delegate straight to [`crate::math::cumsum`]/
/// [`crate::math::cumprod`] rather than duplicating their (SIMD- and
/// parallel-tiered) logic; every other operation runs the generic
/// stride-stepping fold in this module. See the module docs for the
/// `axis: None` convention.
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::{ufunc_accumulate, UfuncOp};
///
/// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
///
/// // np.maximum.accumulate(a) == [1, 3, 3, 5, 5]
/// let running_max = ufunc_accumulate(UfuncOp::Maximum, &a, None)
///     .expect("accumulate should succeed");
/// assert_eq!(running_max.to_vec(), vec![1.0, 3.0, 3.0, 5.0, 5.0]);
/// ```
pub fn ufunc_accumulate<T>(op: UfuncOp, a: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Send + Sync + 'static,
{
    match op {
        UfuncOp::Add => crate::math::cumsum(a, axis, None),
        UfuncOp::Multiply => crate::math::cumprod(a, axis, None),
        UfuncOp::Subtract | UfuncOp::Divide | UfuncOp::Maximum | UfuncOp::Minimum => match axis {
            None => accumulate_flat(op, a),
            Some(ax) => {
                let axis = normalize_axis(ax, a.ndim())?;
                accumulate_along_axis(op, a, axis)
            }
        },
    }
}

// =============================================================================
// ufunc_outer
// =============================================================================

/// `op.outer(a, b)`: NumPy's `np.add.outer`/`np.multiply.outer`/etc.,
/// generalized over [`UfuncOp`] and full N-D (not just 1-D x 1-D): the
/// result has shape `a.shape() ++ b.shape()`, with
/// `out[i.., j..] = op(a[i..], b[j..])`.
///
/// Implemented as one flat nested loop over each operand's own
/// logical (C-order) elements: for every element of `a` (in its own
/// C-order), every element of `b` is visited once, in `b`'s own C-order.
/// This produces exactly the concatenated shape's C-order flat data
/// directly, with no index bookkeeping needed, because varying `b`'s
/// element fastest while holding `a`'s element fixed is precisely what
/// C-order means for a shape ending in `b.shape()`.
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::{ufunc_outer, UfuncOp};
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![10.0, 20.0]);
///
/// // np.add.outer(a, b) == [[11, 21], [12, 22], [13, 23]]
/// let out = ufunc_outer(UfuncOp::Add, &a, &b).expect("outer should succeed");
/// assert_eq!(out.shape(), vec![3, 2]);
/// assert_eq!(out.to_vec(), vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
/// ```
pub fn ufunc_outer<T>(op: UfuncOp, a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let a_flat = operand(a);
    let b_flat = operand(b);
    let mut result = Vec::with_capacity(a_flat.len() * b_flat.len());
    for &av in a_flat.iter() {
        for &bv in b_flat.iter() {
            result.push(op.apply(av, bv));
        }
    }
    let mut shape = a.shape();
    shape.extend(b.shape());
    Array::from_vec_shape(result, &shape)
}

// =============================================================================
// ufunc_reduceat
// =============================================================================

/// Shared core of [`ufunc_reduceat`]: `flat` is `shape`'s own logical
/// (C-order) flat data, and `axis` is an already-normalized (non-negative,
/// in-bounds) axis into `shape`.
///
/// Implements NumPy's exact segment rule (verified against NumPy 2.4.2):
/// for output position `i` with `j = indices[i]`, let
/// `end = indices[i + 1]` (or the axis's own size, for the last `i`). If
/// `end > j`, the output is `op.reduce(lane[j..end])` (a plain reduce, no
/// `initial`). Otherwise (`end <= j`, NumPy's documented special case for
/// "the last element, or `indices[i] >= indices[i + 1]`") the output is
/// simply `lane[j]` itself, unreduced. This module implements that special
/// case by widening the segment to `[j, j + 1)` instead of branching
/// separately: reducing a single-element lane always returns that element
/// regardless of `op` (see [`reduce_lane`]), so the two are exactly
/// equivalent and this avoids stating the rule twice.
fn reduceat_along_axis<T: Float + Clone>(
    op: UfuncOp,
    flat: &[T],
    shape: &[usize],
    axis: usize,
    indices: &[usize],
) -> Result<Array<T>> {
    let dim_size = shape[axis];
    for &j in indices {
        if j >= dim_size {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {j} out of bounds for reduceat axis of size {dim_size}"
            )));
        }
    }

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let axis_stride = strides[axis];

    let other_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .collect();
    let n_outer: usize = other_shape.iter().product();

    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let mut out_strides = vec![1usize; out_shape.len()];
    for i in (0..out_shape.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    let out_axis_stride = out_strides[axis];

    let mut result = vec![T::zero(); n_outer * indices.len()];
    for outer_idx in 0..n_outer {
        let other_indices = unravel_index(outer_idx, &other_shape);
        let mut in_base = 0usize;
        let mut out_base = 0usize;
        let mut oi = 0usize;
        for i in 0..shape.len() {
            if i == axis {
                continue;
            }
            in_base += other_indices[oi] * strides[i];
            out_base += other_indices[oi] * out_strides[i];
            oi += 1;
        }

        for (out_k, &j) in indices.iter().enumerate() {
            let end = if out_k + 1 < indices.len() {
                indices[out_k + 1]
            } else {
                dim_size
            };
            let seg_end = if end > j { end } else { j + 1 };
            let lane = (j..seg_end).map(|k| flat[in_base + k * axis_stride]);
            result[out_base + out_k * out_axis_stride] = reduce_lane(op, lane, None)?;
        }
    }

    Array::from_vec_shape(result, &out_shape)
}

/// `op.reduceat(a, indices, axis=axis)`: NumPy's `np.add.reduceat`/
/// `np.multiply.reduceat`/etc., generalized over [`UfuncOp`]. See
/// [`reduceat_along_axis`] for the exact segment rule (including the
/// `indices[i] >= indices[i + 1]` single-element special case), and the
/// module docs for the `axis: None` (flatten first) convention.
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::{ufunc_reduceat, UfuncOp};
///
/// let a = Array::from_vec((0..10).map(|i| i as f64).collect::<Vec<_>>());
///
/// // np.add.reduceat(a, [0, 4, 7]) == [6, 15, 24]
/// // (segments [0:4], [4:7], [7:] -> 0+1+2+3, 4+5+6, 7+8+9)
/// let out = ufunc_reduceat(UfuncOp::Add, &a, &[0, 4, 7], None)
///     .expect("reduceat should succeed");
/// assert_eq!(out.to_vec(), vec![6.0, 15.0, 24.0]);
///
/// // np.add.reduceat(a, [0, 4, 4, 7]) == [6, 4, 15, 24]:
/// // indices[1] >= indices[2] (4 >= 4), so segment 1 is just a[4] == 4, unreduced.
/// let out2 = ufunc_reduceat(UfuncOp::Add, &a, &[0, 4, 4, 7], None)
///     .expect("reduceat should succeed");
/// assert_eq!(out2.to_vec(), vec![6.0, 4.0, 15.0, 24.0]);
/// ```
pub fn ufunc_reduceat<T>(
    op: UfuncOp,
    a: &Array<T>,
    indices: &[usize],
    axis: Option<isize>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    match axis {
        None => {
            let flat = operand(a);
            reduceat_along_axis(op, &flat, &[flat.len()], 0, indices)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            let flat = operand(a);
            reduceat_along_axis(op, &flat, &a.shape(), axis, indices)
        }
    }
}

// =============================================================================
// ufunc_at
// =============================================================================

/// `op.at(a, indices, b)`: NumPy's `np.add.at`/`np.multiply.at`/etc.,
/// generalized over [`UfuncOp`] -- unbuffered, in-place, with
/// **repeated-index accumulation**: `ufunc_at(Add, &mut a, &[0, 0, 1], &b)`
/// adds `b[0]` then `b[1]` into `a[0]` (both, in order) and `b[2]` into
/// `a[1]`, which is exactly what `a[indices] += b` in fancy-indexing form
/// gets wrong (it computes `a[indices]` once, adds elementwise, and writes
/// back once, so a repeated index only keeps the *last* addition to that
/// position instead of accumulating every one).
///
/// `indices` selects along `a`'s axis 0; `b` must have shape `[indices.len(),
/// ..a.shape()[1..]]` (for a 1-D `a`, this is just `[indices.len()]`, matching
/// the classic 1-D example above). For each `k`, `a`'s "row" `indices[k]` has
/// `op` applied against `b`'s row `k`, element-for-element.
///
/// [`Array::array_mut`] is called exactly once, before the loop (per this
/// crate's Arc-COW `Array`, that is the one point where a shared buffer is
/// unshared) -- every per-element update below reads and writes through the
/// resulting `&mut` reference directly, rather than calling a bounds-checked
/// `get`/`set` pair (which would re-check `array_mut`'s unshare condition on
/// every element).
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::{ufunc_at, UfuncOp};
///
/// // np.add.at(a, [0, 0, 1], [10, 20, 30]) -> a == [1+10+20, 2+30, 3] == [31, 32, 3]
/// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![10.0, 20.0, 30.0]);
/// ufunc_at(UfuncOp::Add, &mut a, &[0, 0, 1], &b).expect("at should succeed");
/// assert_eq!(a.to_vec(), vec![31.0, 32.0, 3.0]);
/// ```
pub fn ufunc_at<T>(op: UfuncOp, a: &mut Array<T>, indices: &[usize], b: &Array<T>) -> Result<()>
where
    T: Float + Clone,
{
    let a_shape = a.shape();
    if a_shape.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "ufunc_at requires an array with at least 1 dimension".to_string(),
        ));
    }
    let n0 = a_shape[0];
    let rest_shape = a_shape[1..].to_vec();

    let mut expected_b_shape = vec![indices.len()];
    expected_b_shape.extend_from_slice(&rest_shape);
    if b.shape() != expected_b_shape {
        return Err(NumRs2Error::ShapeMismatch {
            expected: expected_b_shape,
            actual: b.shape(),
        });
    }
    for &idx in indices {
        if idx >= n0 {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "index {idx} out of bounds for axis 0 with size {n0}"
            )));
        }
    }

    let row_len: usize = rest_shape.iter().product();
    if row_len == 0 {
        // A degenerate zero-size trailing dimension: every "row" is empty,
        // so there is nothing to update (and unraveling below would divide
        // by that zero dimension).
        return Ok(());
    }
    let b_flat = operand(b);

    // The one hoisted unshare (see this function's doc comment): every
    // write in the loops below goes through `nd` directly.
    let nd = a.array_mut();

    let mut multi_index = vec![0usize; a_shape.len()];
    for (k, &row) in indices.iter().enumerate() {
        multi_index[0] = row;
        for r in 0..row_len {
            let rest_idx = unravel_index(r, &rest_shape);
            multi_index[1..].copy_from_slice(&rest_idx);

            let b_val = b_flat[k * row_len + r];
            match nd.get_mut(multi_index.as_slice()) {
                Some(elem) => *elem = op.apply(*elem, b_val),
                None => return Err(NumRs2Error::bulk_index_oob(&multi_index)),
            }
        }
    }
    Ok(())
}

// =============================================================================
// where= support
// =============================================================================

/// `op(a, b, where=mask)` with no `out=`: NumPy's `where=` kwarg for a
/// binary ufunc, generalized over [`UfuncOp`]. See the module docs for the
/// same-shape-only and no-broadcasting-`out=a`-instead-of-uninitialized-
/// memory simplifications relative to real NumPy.
///
/// `a`, `b`, and `mask` must all have exactly the same shape.
pub fn ufunc_where<T>(
    op: UfuncOp,
    a: &Array<T>,
    b: &Array<T>,
    mask: &Array<bool>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let a_shape = a.shape();
    if b.shape() != a_shape {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a_shape,
            actual: b.shape(),
        });
    }
    if mask.shape() != a_shape {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a_shape,
            actual: mask.shape(),
        });
    }

    let a_flat = operand(a);
    let b_flat = operand(b);
    let mask_flat = operand(mask);
    let result: Vec<T> = a_flat
        .iter()
        .zip(b_flat.iter())
        .zip(mask_flat.iter())
        .map(|((&av, &bv), &m)| if m { op.apply(av, bv) } else { av })
        .collect();
    Array::from_vec_shape(result, &a_shape)
}

/// `np.add(a, b, where=mask)` (no `out=`; see [`ufunc_where`]'s docs for the
/// `out=a` convention this implies).
///
/// # Examples
///
/// ```
/// use numrs2::array::Array;
/// use numrs2::ufunc_ops::add_where;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let b = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
/// let mask = Array::from_vec(vec![true, false, true, false]);
///
/// // np.add(a, b, where=mask, out=np.copy(a)) == [11, 2, 33, 4]
/// let out = add_where(&a, &b, &mask).expect("add_where should succeed");
/// assert_eq!(out.to_vec(), vec![11.0, 2.0, 33.0, 4.0]);
/// ```
pub fn add_where<T: Float + Clone>(
    a: &Array<T>,
    b: &Array<T>,
    mask: &Array<bool>,
) -> Result<Array<T>> {
    ufunc_where(UfuncOp::Add, a, b, mask)
}

/// `np.subtract(a, b, where=mask)` (no `out=`; see [`ufunc_where`]).
pub fn subtract_where<T: Float + Clone>(
    a: &Array<T>,
    b: &Array<T>,
    mask: &Array<bool>,
) -> Result<Array<T>> {
    ufunc_where(UfuncOp::Subtract, a, b, mask)
}

/// `np.multiply(a, b, where=mask)` (no `out=`; see [`ufunc_where`]).
pub fn multiply_where<T: Float + Clone>(
    a: &Array<T>,
    b: &Array<T>,
    mask: &Array<bool>,
) -> Result<Array<T>> {
    ufunc_where(UfuncOp::Multiply, a, b, mask)
}

/// `np.divide(a, b, where=mask)` (no `out=`; see [`ufunc_where`]).
pub fn divide_where<T: Float + Clone>(
    a: &Array<T>,
    b: &Array<T>,
    mask: &Array<bool>,
) -> Result<Array<T>> {
    ufunc_where(UfuncOp::Divide, a, b, mask)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: Vec<f64>) -> Array<f64> {
        Array::from_vec(v)
    }

    // ---- UfuncOp::apply NaN semantics ----

    #[test]
    fn maximum_minimum_apply_propagate_nan_symmetrically() {
        // np.maximum(nan, 5.0) == np.maximum(5.0, nan) == nan (both orders).
        assert!(UfuncOp::Maximum.apply(f64::NAN, 5.0).is_nan());
        assert!(UfuncOp::Maximum.apply(5.0, f64::NAN).is_nan());
        assert!(UfuncOp::Minimum.apply(f64::NAN, 5.0).is_nan());
        assert!(UfuncOp::Minimum.apply(5.0, f64::NAN).is_nan());
        // Ordinary case still behaves.
        assert_eq!(UfuncOp::Maximum.apply(2.0, 5.0), 5.0);
        assert_eq!(UfuncOp::Minimum.apply(2.0, 5.0), 2.0);
    }

    #[test]
    fn identity_matches_numpy() {
        assert_eq!(UfuncOp::Add.identity::<f64>(), Some(0.0));
        assert_eq!(UfuncOp::Multiply.identity::<f64>(), Some(1.0));
        assert_eq!(UfuncOp::Maximum.identity::<f64>(), None);
        assert_eq!(UfuncOp::Minimum.identity::<f64>(), None);
        assert_eq!(UfuncOp::Subtract.identity::<f64>(), None);
        assert_eq!(UfuncOp::Divide.identity::<f64>(), None);
    }

    // =========================================================================
    // ufunc_reduce -- pinned against numpy 2.4.2
    //
    // a = np.arange(1, 7, dtype=float64).reshape(2, 3) == [[1,2,3],[4,5,6]]
    // =========================================================================

    fn a_2x3() -> Array<f64> {
        Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])
    }

    #[test]
    fn reduce_add_axis_none_matches_numpy() {
        // np.add.reduce(a, axis=None) == 21.0 (flattened full sum)
        let a = a_2x3();
        let r = ufunc_reduce(UfuncOp::Add, &a, None, false, None).expect("reduce should succeed");
        assert_eq!(r.shape(), vec![1]);
        assert_eq!(r.to_vec(), vec![21.0]);
    }

    #[test]
    fn reduce_add_axis_none_keepdims_matches_numpy_value_this_crate_shape() {
        // np.add.reduce(a, axis=None, keepdims=True) == [[21.0]] (shape (1,1));
        // value matches, shape follows this crate's convention (see module docs).
        let a = a_2x3();
        let r = ufunc_reduce(UfuncOp::Add, &a, None, true, None).expect("reduce should succeed");
        assert_eq!(r.shape(), vec![1, 1]);
        assert_eq!(r.to_vec(), vec![21.0]);
    }

    #[test]
    fn reduce_add_axis0_and_axis1_match_numpy() {
        let a = a_2x3();
        // np.add.reduce(a, axis=0) == [5, 7, 9]
        let r0 = ufunc_reduce(UfuncOp::Add, &a, Some(0), false, None).expect("reduce ok");
        assert_eq!(r0.to_vec(), vec![5.0, 7.0, 9.0]);
        // np.add.reduce(a, axis=1) == [6, 15]
        let r1 = ufunc_reduce(UfuncOp::Add, &a, Some(1), false, None).expect("reduce ok");
        assert_eq!(r1.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_negative_axis_matches_positive_equivalent() {
        let a = a_2x3();
        // np.add.reduce(a, axis=-1) == np.add.reduce(a, axis=1) == [6, 15]
        let r_neg = ufunc_reduce(UfuncOp::Add, &a, Some(-1), false, None).expect("reduce ok");
        let r_pos = ufunc_reduce(UfuncOp::Add, &a, Some(1), false, None).expect("reduce ok");
        assert_eq!(r_neg.to_vec(), r_pos.to_vec());
        assert_eq!(r_neg.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_keepdims_matches_numpy() {
        let a = a_2x3();
        // np.add.reduce(a, axis=1, keepdims=True) == [[6.0], [15.0]], shape (2, 1)
        let r = ufunc_reduce(UfuncOp::Add, &a, Some(1), true, None).expect("reduce ok");
        assert_eq!(r.shape(), vec![2, 1]);
        assert_eq!(r.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_multiply_and_extrema_match_numpy() {
        let a = a_2x3();
        // np.multiply.reduce(a, axis=1) == [6, 120]
        let mul = ufunc_reduce(UfuncOp::Multiply, &a, Some(1), false, None).expect("ok");
        assert_eq!(mul.to_vec(), vec![6.0, 120.0]);
        // np.maximum.reduce(a, axis=0) == [4, 5, 6]
        let mx = ufunc_reduce(UfuncOp::Maximum, &a, Some(0), false, None).expect("ok");
        assert_eq!(mx.to_vec(), vec![4.0, 5.0, 6.0]);
        // np.minimum.reduce(a, axis=1) == [1, 4]
        let mn = ufunc_reduce(UfuncOp::Minimum, &a, Some(1), false, None).expect("ok");
        assert_eq!(mn.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn reduce_with_initial_matches_numpy() {
        let a = a_2x3();
        // np.add.reduce(a, axis=1, initial=100) == [106, 115]
        let add_i = ufunc_reduce(UfuncOp::Add, &a, Some(1), false, Some(100.0)).expect("ok");
        assert_eq!(add_i.to_vec(), vec![106.0, 115.0]);
        // np.multiply.reduce(a, axis=1, initial=2) == [12, 240]
        let mul_i = ufunc_reduce(UfuncOp::Multiply, &a, Some(1), false, Some(2.0)).expect("ok");
        assert_eq!(mul_i.to_vec(), vec![12.0, 240.0]);
        // np.maximum.reduce(a, axis=1, initial=0) == [3, 6]
        let max_i0 = ufunc_reduce(UfuncOp::Maximum, &a, Some(1), false, Some(0.0)).expect("ok");
        assert_eq!(max_i0.to_vec(), vec![3.0, 6.0]);
        // np.maximum.reduce(a, axis=1, initial=100) == [100, 100]
        let max_i100 = ufunc_reduce(UfuncOp::Maximum, &a, Some(1), false, Some(100.0)).expect("ok");
        assert_eq!(max_i100.to_vec(), vec![100.0, 100.0]);
    }

    #[test]
    fn reduce_empty_array_matches_numpy() {
        let empty = arr(vec![]);
        // np.add.reduce(np.array([])) == 0.0
        assert_eq!(
            ufunc_reduce(UfuncOp::Add, &empty, None, false, None)
                .expect("ok")
                .to_vec(),
            vec![0.0]
        );
        // np.multiply.reduce(np.array([])) == 1.0
        assert_eq!(
            ufunc_reduce(UfuncOp::Multiply, &empty, None, false, None)
                .expect("ok")
                .to_vec(),
            vec![1.0]
        );
        // np.maximum.reduce(np.array([])) raises ValueError (no identity)
        assert!(ufunc_reduce(UfuncOp::Maximum, &empty, None, false, None).is_err());
        assert!(ufunc_reduce(UfuncOp::Minimum, &empty, None, false, None).is_err());
        // np.add.reduce(np.array([]), initial=5) == 5.0
        assert_eq!(
            ufunc_reduce(UfuncOp::Add, &empty, None, false, Some(5.0))
                .expect("ok")
                .to_vec(),
            vec![5.0]
        );
        // np.maximum.reduce(np.array([]), initial=5) == 5.0 (initial rescues it)
        assert_eq!(
            ufunc_reduce(UfuncOp::Maximum, &empty, None, false, Some(5.0))
                .expect("ok")
                .to_vec(),
            vec![5.0]
        );
    }

    #[test]
    fn reduce_empty_axis_matches_numpy() {
        // b = np.zeros((3, 0))
        let b: Array<f64> = Array::from_vec(vec![]).reshape(&[3, 0]);
        // np.add.reduce(b, axis=1) == [0, 0, 0]
        let r = ufunc_reduce(UfuncOp::Add, &b, Some(1), false, None).expect("ok");
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 0.0]);
        // np.maximum.reduce(b, axis=1) raises (every lane is empty, no identity)
        assert!(ufunc_reduce(UfuncOp::Maximum, &b, Some(1), false, None).is_err());
        // np.add.reduce(b, axis=1, initial=7) == [7, 7, 7]
        let r_init = ufunc_reduce(UfuncOp::Add, &b, Some(1), false, Some(7.0)).expect("ok");
        assert_eq!(r_init.to_vec(), vec![7.0, 7.0, 7.0]);
        // np.maximum.reduce(b, axis=1, initial=7) == [7, 7, 7]
        let max_init = ufunc_reduce(UfuncOp::Maximum, &b, Some(1), false, Some(7.0)).expect("ok");
        assert_eq!(max_init.to_vec(), vec![7.0, 7.0, 7.0]);
        // np.add.reduce(b, axis=0) has shape (0,)
        let r0 = ufunc_reduce(UfuncOp::Add, &b, Some(0), false, None).expect("ok");
        assert_eq!(r0.shape(), vec![0]);
        // np.add.reduce(b, axis=1, keepdims=True).shape == (3, 1)
        let r1_keep = ufunc_reduce(UfuncOp::Add, &b, Some(1), true, None).expect("ok");
        assert_eq!(r1_keep.shape(), vec![3, 1]);
    }

    #[test]
    fn reduce_1d_empty_axis_keepdims_matches_numpy() {
        // np.add.reduce(np.zeros(0), axis=0, keepdims=True) == array([0.]), shape (1,).
        // This is the one shape path the other reduce tests don't touch: the
        // out_shape.is_empty() -> push(1) guard and a genuinely empty axis
        // (axis_size == 0) both firing on the very same 1-D input at once.
        let empty_1d: Array<f64> = Array::from_vec(vec![]).reshape(&[0]);
        let r = ufunc_reduce(UfuncOp::Add, &empty_1d, Some(0), true, None).expect("ok");
        assert_eq!(r.shape(), vec![1]);
        assert_eq!(r.to_vec(), vec![0.0]);
    }

    #[test]
    fn reduce_3d_negative_axis_matches_numpy() {
        // a = np.arange(24, dtype=float64).reshape(2, 3, 4)
        let a: Array<f64> =
            Array::from_vec((0..24).map(|i| i as f64).collect()).reshape(&[2, 3, 4]);
        // np.add.reduce(a, axis=-2).shape == (2, 4); values [[12,15,18,21],[48,51,54,57]]
        let r = ufunc_reduce(UfuncOp::Add, &a, Some(-2), false, None).expect("ok");
        assert_eq!(r.shape(), vec![2, 4]);
        assert_eq!(
            r.to_vec(),
            vec![12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]
        );
        // np.maximum.reduce(a, axis=-1, keepdims=True).shape == (2, 3, 1)
        let r2 = ufunc_reduce(UfuncOp::Maximum, &a, Some(-1), true, None).expect("ok");
        assert_eq!(r2.shape(), vec![2, 3, 1]);
        assert_eq!(r2.to_vec(), vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);
    }

    #[test]
    fn reduce_out_of_bounds_axis_errors() {
        let a = a_2x3();
        assert!(ufunc_reduce(UfuncOp::Add, &a, Some(2), false, None).is_err());
        assert!(ufunc_reduce(UfuncOp::Add, &a, Some(-3), false, None).is_err());
    }

    #[test]
    fn reduce_f32_fast_path_matches_f64_generic_path() {
        // Same computation in f32 (routes through the sum_f32/prod_f32 fast
        // path) and f64 (fast path too, but a different kernel) should agree
        // to f32 precision.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let a32 = Array::from_vec(data.clone());
        let r32 = ufunc_reduce(UfuncOp::Add, &a32, None, false, None).expect("ok");
        assert_eq!(r32.to_vec(), vec![15.0f32]);

        let mul32 = ufunc_reduce(UfuncOp::Multiply, &a32, None, false, None).expect("ok");
        assert_eq!(mul32.to_vec(), vec![120.0f32]);
    }

    // =========================================================================
    // ufunc_accumulate -- pinned against numpy 2.4.2
    // =========================================================================

    #[test]
    fn accumulate_add_multiply_delegate_correctly() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        // np.add.accumulate(a) == [1, 3, 6, 10]
        let add_acc = ufunc_accumulate(UfuncOp::Add, &a, None).expect("ok");
        assert_eq!(add_acc.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
        // np.multiply.accumulate(a) == [1, 2, 6, 24]
        let mul_acc = ufunc_accumulate(UfuncOp::Multiply, &a, None).expect("ok");
        assert_eq!(mul_acc.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn accumulate_maximum_minimum_match_numpy() {
        // np.maximum.accumulate([1,3,2,5,4]) == [1,3,3,5,5]
        let a = arr(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        let mx = ufunc_accumulate(UfuncOp::Maximum, &a, None).expect("ok");
        assert_eq!(mx.to_vec(), vec![1.0, 3.0, 3.0, 5.0, 5.0]);
        // np.minimum.accumulate([5,3,4,1,2]) == [5,3,3,1,1]
        let b = arr(vec![5.0, 3.0, 4.0, 1.0, 2.0]);
        let mn = ufunc_accumulate(UfuncOp::Minimum, &b, None).expect("ok");
        assert_eq!(mn.to_vec(), vec![5.0, 3.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn accumulate_subtract_divide_match_numpy() {
        // np.subtract.accumulate([10,1,2,3]) == [10, 9, 7, 4]
        let a = arr(vec![10.0, 1.0, 2.0, 3.0]);
        let sub = ufunc_accumulate(UfuncOp::Subtract, &a, None).expect("ok");
        assert_eq!(sub.to_vec(), vec![10.0, 9.0, 7.0, 4.0]);
        // np.divide.accumulate([10,1,2,3]) == [10, 10, 5, 1.66666667]
        let div = ufunc_accumulate(UfuncOp::Divide, &a, None).expect("ok");
        let got = div.to_vec();
        let want = [10.0, 10.0, 5.0, 5.0 / 3.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-9, "got {g}, want {w}");
        }
    }

    #[test]
    fn accumulate_with_axis_matches_numpy() {
        // m = [[1,5,2],[8,1,9]]
        let m = Array::from_vec(vec![1.0, 5.0, 2.0, 8.0, 1.0, 9.0]).reshape(&[2, 3]);
        // np.maximum.accumulate(m, axis=1) == [[1,5,5],[8,8,9]]
        let mx = ufunc_accumulate(UfuncOp::Maximum, &m, Some(1)).expect("ok");
        assert_eq!(mx.to_vec(), vec![1.0, 5.0, 5.0, 8.0, 8.0, 9.0]);
        // np.minimum.accumulate(m, axis=0) == [[1,5,2],[1,1,2]]
        let mn = ufunc_accumulate(UfuncOp::Minimum, &m, Some(0)).expect("ok");
        assert_eq!(mn.to_vec(), vec![1.0, 5.0, 2.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn accumulate_negative_axis_matches_positive() {
        let m = Array::from_vec(vec![1.0, 5.0, 2.0, 8.0, 1.0, 9.0]).reshape(&[2, 3]);
        let neg = ufunc_accumulate(UfuncOp::Maximum, &m, Some(-1)).expect("ok");
        let pos = ufunc_accumulate(UfuncOp::Maximum, &m, Some(1)).expect("ok");
        assert_eq!(neg.to_vec(), pos.to_vec());
    }

    // =========================================================================
    // ufunc_outer -- pinned against numpy 2.4.2
    // =========================================================================

    #[test]
    fn outer_add_multiply_match_numpy() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![10.0, 20.0]);
        // np.add.outer(a, b) == [[11,21],[12,22],[13,23]]
        let add_out = ufunc_outer(UfuncOp::Add, &a, &b).expect("ok");
        assert_eq!(add_out.shape(), vec![3, 2]);
        assert_eq!(add_out.to_vec(), vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
        // np.multiply.outer(a, b) == [[10,20],[20,40],[30,60]]
        let mul_out = ufunc_outer(UfuncOp::Multiply, &a, &b).expect("ok");
        assert_eq!(mul_out.to_vec(), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn outer_maximum_matches_numpy() {
        // np.maximum.outer([1,5,2], [3,1]) == [[3,1],[5,5],[3,2]]
        let a = arr(vec![1.0, 5.0, 2.0]);
        let b = arr(vec![3.0, 1.0]);
        let out = ufunc_outer(UfuncOp::Maximum, &a, &b).expect("ok");
        assert_eq!(out.to_vec(), vec![3.0, 1.0, 5.0, 5.0, 3.0, 2.0]);
    }

    #[test]
    fn outer_full_nd_shape_matches_numpy() {
        // a.shape=(2,3), b.shape=(2,) -> np.add.outer(a,b).shape == (2,3,2)
        let a: Array<f64> = Array::from_vec((1..7).map(|i| i as f64).collect()).reshape(&[2, 3]);
        let b = arr(vec![1.0, 2.0]);
        let out = ufunc_outer(UfuncOp::Add, &a, &b).expect("ok");
        assert_eq!(out.shape(), vec![2, 3, 2]);
        // np.add.outer(a,b) == [[[2,3],[3,4],[4,5]],[[5,6],[6,7],[7,8]]]
        assert_eq!(
            out.to_vec(),
            vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0]
        );
    }

    // =========================================================================
    // ufunc_reduceat -- pinned against numpy 2.4.2, including the tricky
    // indices[i] >= indices[i+1] single-element rule.
    // =========================================================================

    fn seq10() -> Array<f64> {
        arr((0..10).map(|i| i as f64).collect())
    }

    #[test]
    fn reduceat_basic_matches_numpy() {
        // np.add.reduceat(a, [0,4,7]) == [6, 15, 24]
        let a = seq10();
        let out = ufunc_reduceat(UfuncOp::Add, &a, &[0, 4, 7], None).expect("ok");
        assert_eq!(out.to_vec(), vec![6.0, 15.0, 24.0]);
    }

    #[test]
    fn reduceat_single_element_rule_matches_numpy() {
        let a = seq10();
        // np.add.reduceat(a, [0,4,4,7]) == [6, 4, 15, 24]
        let out1 = ufunc_reduceat(UfuncOp::Add, &a, &[0, 4, 4, 7], None).expect("ok");
        assert_eq!(out1.to_vec(), vec![6.0, 4.0, 15.0, 24.0]);
        // np.add.reduceat(a, [0,2,2,2,7]) == [1, 2, 2, 20, 24]
        let out2 = ufunc_reduceat(UfuncOp::Add, &a, &[0, 2, 2, 2, 7], None).expect("ok");
        assert_eq!(out2.to_vec(), vec![1.0, 2.0, 2.0, 20.0, 24.0]);
    }

    #[test]
    fn reduceat_last_index_runs_to_end() {
        let a = seq10();
        // np.add.reduceat(a, [len(a)-1]) == [9.0]
        let out = ufunc_reduceat(UfuncOp::Add, &a, &[9], None).expect("ok");
        assert_eq!(out.to_vec(), vec![9.0]);
    }

    #[test]
    fn reduceat_empty_indices_yields_empty_result() {
        // np.add.reduceat(a, []) == array([]), shape (0,); no segment to
        // validate or reduce, but the output-shape/from_vec_shape plumbing
        // must still accept a zero-sized axis.
        let a = seq10();
        let out =
            ufunc_reduceat(UfuncOp::Add, &a, &[], None).expect("empty indices should succeed");
        assert_eq!(out.shape(), vec![0]);
        assert!(out.to_vec().is_empty());

        // Same, but axis=None on an already-empty array (n_outer == 1 via
        // the trivial `other_shape == []` case, zero inner work).
        let empty: Array<f64> = arr(vec![]);
        let out2 =
            ufunc_reduceat(UfuncOp::Add, &empty, &[], None).expect("empty indices should succeed");
        assert_eq!(out2.shape(), vec![0]);
    }

    #[test]
    fn reduceat_out_of_bounds_index_errors() {
        let a = seq10();
        // np.add.reduceat(a, [10, ...]) raises IndexError (10 is out of bounds)
        assert!(ufunc_reduceat(UfuncOp::Add, &a, &[10], None).is_err());
    }

    #[test]
    fn reduceat_with_axis_matches_numpy() {
        // a3 = np.arange(1,13,dtype=float64).reshape(3,4)
        let a3: Array<f64> = Array::from_vec((1..13).map(|i| i as f64).collect()).reshape(&[3, 4]);
        // np.add.reduceat(a3, [0,2], axis=1) == [[3,7],[11,15],[19,23]]
        let out = ufunc_reduceat(UfuncOp::Add, &a3, &[0, 2], Some(1)).expect("ok");
        assert_eq!(out.shape(), vec![3, 2]);
        assert_eq!(out.to_vec(), vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);

        // np.multiply.reduceat(a3, [0,1,3], axis=1) == [[1,6,4],[5,42,8],[9,110,12]]
        let mul = ufunc_reduceat(UfuncOp::Multiply, &a3, &[0, 1, 3], Some(1)).expect("ok");
        assert_eq!(
            mul.to_vec(),
            vec![1.0, 6.0, 4.0, 5.0, 42.0, 8.0, 9.0, 110.0, 12.0]
        );

        // np.add.reduceat(a3, [0,2], axis=0) == [[6,8,10,12],[9,10,11,12]]
        let axis0 = ufunc_reduceat(UfuncOp::Add, &a3, &[0, 2], Some(0)).expect("ok");
        assert_eq!(axis0.shape(), vec![2, 4]);
        assert_eq!(
            axis0.to_vec(),
            vec![6.0, 8.0, 10.0, 12.0, 9.0, 10.0, 11.0, 12.0]
        );

        // np.add.reduceat(a3, [0,2], axis=-1) == same as axis=1
        let neg = ufunc_reduceat(UfuncOp::Add, &a3, &[0, 2], Some(-1)).expect("ok");
        assert_eq!(neg.to_vec(), out.to_vec());
    }

    // =========================================================================
    // ufunc_at -- pinned against numpy 2.4.2, including repeated-index
    // accumulation.
    // =========================================================================

    #[test]
    fn at_add_repeated_indices_accumulates() {
        // np.add.at(a, [0,0,1], [10,20,30]) -> a == [31, 32, 3]
        let mut a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![10.0, 20.0, 30.0]);
        ufunc_at(UfuncOp::Add, &mut a, &[0, 0, 1], &b).expect("at should succeed");
        assert_eq!(a.to_vec(), vec![31.0, 32.0, 3.0]);
    }

    #[test]
    fn at_multiply_repeated_indices_accumulates() {
        // np.multiply.at(a, [0,0,3], [2,3,5]) -> a == [1*2*3, 2, 3, 4*5] == [6,2,3,20]
        let mut a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let b = arr(vec![2.0, 3.0, 5.0]);
        ufunc_at(UfuncOp::Multiply, &mut a, &[0, 0, 3], &b).expect("at should succeed");
        assert_eq!(a.to_vec(), vec![6.0, 2.0, 3.0, 20.0]);
    }

    #[test]
    fn at_triple_repeat_at_same_index() {
        // np.add.at(a2, [0,0,0,1,1], 1.0-per-call) -> a2 == [3, 2, 0, 0, 0]
        let mut a = arr(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = arr(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        ufunc_at(UfuncOp::Add, &mut a, &[0, 0, 0, 1, 1], &b).expect("at should succeed");
        assert_eq!(a.to_vec(), vec![3.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn at_nd_repeated_indices_matches_numpy() {
        // a4 = zeros((3,2)); b4 = [[1,1],[2,2],[3,3]]
        // np.add.at(a4, [0,0,1], b4) -> [[3,3],[3,3],[0,0]]
        let mut a4: Array<f64> = Array::from_vec(vec![0.0; 6]).reshape(&[3, 2]);
        let b4 = Array::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).reshape(&[3, 2]);
        ufunc_at(UfuncOp::Add, &mut a4, &[0, 0, 1], &b4).expect("at should succeed");
        assert_eq!(a4.to_vec(), vec![3.0, 3.0, 3.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn at_out_of_bounds_index_errors() {
        // np.add.at(zeros(3), [5], [1.0]) raises IndexError
        let mut a = arr(vec![0.0, 0.0, 0.0]);
        let b = arr(vec![1.0]);
        assert!(ufunc_at(UfuncOp::Add, &mut a, &[5], &b).is_err());
    }

    #[test]
    fn at_shape_mismatch_errors() {
        let mut a = arr(vec![0.0, 0.0, 0.0]);
        let b = arr(vec![1.0, 2.0]); // wrong length for 3 indices
        assert!(ufunc_at(UfuncOp::Add, &mut a, &[0, 1, 2], &b).is_err());
    }

    // =========================================================================
    // where= support
    // =========================================================================

    #[test]
    fn where_add_subtract_multiply_divide_match_numpy_out_eq_a() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let b = arr(vec![10.0, 20.0, 30.0, 40.0]);
        let mask = Array::from_vec(vec![true, false, true, false]);

        // np.add(a,b,where=mask,out=copy(a)) == [11, 2, 33, 4]
        assert_eq!(
            add_where(&a, &b, &mask).expect("ok").to_vec(),
            vec![11.0, 2.0, 33.0, 4.0]
        );
        // np.subtract(a,b,where=mask,out=copy(a)) == [-9, 2, -27, 4]
        assert_eq!(
            subtract_where(&a, &b, &mask).expect("ok").to_vec(),
            vec![-9.0, 2.0, -27.0, 4.0]
        );
        // np.multiply(a,b,where=mask,out=copy(a)) == [10, 2, 90, 4]
        assert_eq!(
            multiply_where(&a, &b, &mask).expect("ok").to_vec(),
            vec![10.0, 2.0, 90.0, 4.0]
        );
        // np.divide(a,b,where=mask,out=copy(a)) == [0.1, 2, 0.1, 4]
        assert_eq!(
            divide_where(&a, &b, &mask).expect("ok").to_vec(),
            vec![0.1, 2.0, 0.1, 4.0]
        );
    }

    #[test]
    fn where_generic_dispatch_matches_named_wrappers() {
        let a = arr(vec![1.0, 2.0]);
        let b = arr(vec![5.0, 6.0]);
        let mask = Array::from_vec(vec![true, true]);
        assert_eq!(
            ufunc_where(UfuncOp::Add, &a, &b, &mask)
                .expect("ok")
                .to_vec(),
            add_where(&a, &b, &mask).expect("ok").to_vec()
        );
    }

    #[test]
    fn where_shape_mismatch_errors_for_b_and_for_mask() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b_wrong = arr(vec![1.0, 2.0]);
        let mask_ok = Array::from_vec(vec![true, false, true]);
        let mask_wrong = Array::from_vec(vec![true, false]);
        let b_ok = arr(vec![1.0, 2.0, 3.0]);

        assert!(ufunc_where(UfuncOp::Add, &a, &b_wrong, &mask_ok).is_err());
        assert!(ufunc_where(UfuncOp::Add, &a, &b_ok, &mask_wrong).is_err());
    }
}

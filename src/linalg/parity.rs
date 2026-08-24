//! NumPy-parity linear algebra additions that are new operations in their
//! own right, rather than extensions of an existing function: `multi_dot`'s
//! matrix-chain-order optimizer, and the tensor-shaped `solve`/`inv`
//! variants `tensorsolve`/`tensorinv`.
//!
//! Every function here shares the same bound: exactly what [`Array::matmul`],
//! [`Array::inv`], and [`Array::solve`] each require (see their impl blocks
//! in `linalg::mod`), since all three are used somewhere in this file.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::borrow::Cow;
use std::fmt::Debug;

/// Compute the dot product of two or more arrays, automatically choosing
/// the evaluation order that minimizes the total scalar-multiplication
/// count -- NumPy's `numpy.linalg.multi_dot`.
///
/// As in NumPy: if the first array is 1-D it is treated as a row vector,
/// and/or if the last array is 1-D it is treated as a column vector; every
/// other array (and either end, if it is not 1-D) must be 2-D. Whichever
/// end(s) were promoted this way are squeezed back out of the result -- if
/// *both* ends are 1-D (including the plain 2-argument case), the result
/// is fully reduced to a scalar, represented, as this crate's `squeeze`
/// already does for its own `axis: None` "would become a scalar" case, as
/// a length-1 array rather than a true 0-D one.
///
/// # Errors
/// * `InvalidOperation` if fewer than two arrays are given.
/// * `DimensionMismatch` if an array other than the first or last is not
///   2-D.
/// * `ShapeMismatch` if two adjacent arrays' inner dimensions disagree.
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::multi_dot;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let c = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
/// let result = multi_dot(&[&a, &b, &c]).expect("multi_dot should succeed");
/// let expected = a.matmul(&b).expect("matmul").matmul(&c).expect("matmul");
/// assert_eq!(result.shape(), expected.shape());
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!(
///             (result.get(&[i, j]).expect("valid") - expected.get(&[i, j]).expect("valid")).abs()
///                 < 1e-10
///         );
///     }
/// }
///
/// // A 1-D first argument is a row vector, squeezed back out of the result.
/// let v = Array::from_vec(vec![1.0, 1.0]);
/// let r = multi_dot(&[&v, &a, &b]).expect("multi_dot should succeed");
/// assert_eq!(r.shape(), vec![2]);
/// ```
pub fn multi_dot<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    if arrays.len() < 2 {
        return Err(NumRs2Error::InvalidOperation(
            "multi_dot requires at least two arrays".to_string(),
        ));
    }

    let last = arrays.len() - 1;
    let first_is_vector = arrays[0].ndim() == 1;
    let last_is_vector = arrays[last].ndim() == 1;

    // Promote a 1-D first/last operand to a row/column vector; every other
    // operand (and a first/last one that is already 2-D) must be 2-D.
    let mut mats: Vec<Array<T>> = Vec::with_capacity(arrays.len());
    for (i, &arr) in arrays.iter().enumerate() {
        let shape = arr.shape();
        match shape.len() {
            2 => mats.push(arr.clone()),
            1 if i == 0 => mats.push(arr.try_reshape(&[1, shape[0]])?),
            1 if i == last => mats.push(arr.try_reshape(&[shape[0], 1])?),
            _ => {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "multi_dot: array {} must be 2-D (only the first and last arrays may be 1-D), got shape {:?}",
                    i, shape
                )));
            }
        }
    }

    // Adjacent-dimension compatibility, checked up front so the chain-order
    // DP below never has to.
    for i in 0..mats.len() - 1 {
        if mats[i].shape()[1] != mats[i + 1].shape()[0] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![mats[i].shape()[1]],
                actual: vec![mats[i + 1].shape()[0]],
            });
        }
    }

    let n = mats.len();
    let result = if n == 2 {
        // Fast path: only one possible parenthesization, so the
        // chain-order DP below would be pure overhead.
        mats[0].matmul(&mats[1])?
    } else {
        multi_dot_chain(&mats)?
    };

    // Squeeze back out whichever end(s) were promoted above. `result` is
    // always exactly 2-D here (every `mats[i]` is, so every `matmul` in the
    // fast path or the chain evaluation below is too), with its first dim
    // `== 1` iff `first_is_vector` and its last dim `== 1` iff
    // `last_is_vector`.
    let mut final_shape = result.shape();
    if last_is_vector {
        final_shape.pop();
    }
    if first_is_vector {
        final_shape.remove(0);
    }

    if final_shape.is_empty() {
        // Both ends were vectors: fully reduced to a scalar.
        Ok(Array::from_vec(result.to_vec()))
    } else {
        result.try_reshape(&final_shape)
    }
}

/// Matrix-chain-order dynamic program: finds the parenthesization of
/// `mats[0] @ mats[1] @ ... @ mats[n-1]` (`n >= 3`; `multi_dot` handles
/// `n == 2` itself) that minimizes the total scalar-multiplication count,
/// then evaluates it. The classic textbook DP (e.g. CLRS ch. 15) --
/// mirrors what `numpy.linalg.multi_dot` itself does internally via
/// `_multi_dot_matrix_chain_order`.
fn multi_dot_chain<T>(mats: &[Array<T>]) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    let n = mats.len();

    // `dims[i]` for `i` in `1..n` is the shared "inner" dimension between
    // `mats[i-1]` and `mats[i]`; `dims[0]`/`dims[n]` are the two free
    // (non-contracted) ends of the whole product.
    let mut dims = Vec::with_capacity(n + 1);
    dims.push(mats[0].shape()[0]);
    for m in mats {
        dims.push(m.shape()[1]);
    }

    // `cost[i][j]` = minimum scalar-multiply count to compute
    // `mats[i] @ ... @ mats[j]`; `split[i][j]` = the `k` at which the
    // optimal parenthesization divides that range into
    // `(mats[i..=k]) @ (mats[k+1..=j])`. `u128` costs nothing here and
    // rules out any realistic overflow in the cost accumulation (a
    // planning cost, never an actual element count).
    let mut cost = vec![vec![0u128; n]; n];
    let mut split = vec![vec![0usize; n]; n];

    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            let mut best_cost = u128::MAX;
            let mut best_k = i;
            for k in i..j {
                let candidate = cost[i][k]
                    + cost[k + 1][j]
                    + (dims[i] as u128) * (dims[k + 1] as u128) * (dims[j + 1] as u128);
                if candidate < best_cost {
                    best_cost = candidate;
                    best_k = k;
                }
            }
            cost[i][j] = best_cost;
            split[i][j] = best_k;
        }
    }

    multi_dot_evaluate(mats, &split, 0, n - 1)
}

/// Evaluate the optimal parenthesization `multi_dot_chain` found, via
/// [`Array::matmul`] at each internal node.
fn multi_dot_evaluate<T>(
    mats: &[Array<T>],
    split: &[Vec<usize>],
    i: usize,
    j: usize,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    if i == j {
        return Ok(mats[i].clone());
    }
    let k = split[i][j];
    let left = multi_dot_evaluate(mats, split, i, k)?;
    let right = multi_dot_evaluate(mats, split, k + 1, j)?;
    left.matmul(&right)
}

/// Solve the tensor equation `a x = b` for `x` -- NumPy's
/// `numpy.linalg.tensorsolve`.
///
/// All indices of `x` are assumed summed over in the product together with
/// the rightmost indices of `a`, as in `tensordot(a, x, axes=x.ndim())`.
/// `a`'s shape must be `b.shape() + q` for some tuple `q` with
/// `prod(q) == prod(b.shape())` (`a` is "square" in that sense).
///
/// `axes`, if given, names axes of `a` to move to the end (in the given
/// order) before solving -- matching NumPy's own `axes` parameter, which
/// reorders `a` via an equivalent `moveaxis` before reshaping it down to a
/// square 2-D matrix.
///
/// # Errors
/// * `DimensionMismatch` if `b` has more dimensions than `a`, if an `axes`
///   entry is out of bounds, or if `a`'s shape is not "square" in the
///   sense above.
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensorsolve;
///
/// // For a plain 2-D square `a` and a 1-D `b`, this degenerates to an
/// // ordinary linear solve.
/// let a = Array::from_vec(vec![2.0, 0.0, 0.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![4.0, 8.0]);
/// let x = tensorsolve(&a, &b, None).expect("tensorsolve should succeed");
/// assert_eq!(x.shape(), vec![2]);
/// assert!((x.get(&[0]).expect("valid") - 2.0).abs() < 1e-10);
/// assert!((x.get(&[1]).expect("valid") - 2.0).abs() < 1e-10);
/// ```
pub fn tensorsolve<T>(a: &Array<T>, b: &Array<T>, axes: Option<&[usize]>) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    let an = a.ndim();
    let bn = b.ndim();

    if bn > an {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "tensorsolve: b has more dimensions ({}) than a ({})",
            bn, an
        )));
    }

    // Reorder the requested axes of `a` to the end, mirroring NumPy's own
    // `allaxes.remove(k); allaxes.insert(an, k)` loop over `axes` -- which
    // is exactly `moveaxis(a, axes, (an - len(axes))..an)` (each moved axis
    // ends up appended in the order given, verified against
    // `np.moveaxis`/manual-transpose agreement).
    let a_reordered: Cow<'_, Array<T>> = match axes {
        None => Cow::Borrowed(a),
        Some(axes_to_move) => {
            for &ax in axes_to_move {
                if ax >= an {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "tensorsolve: axis {} out of bounds for a {}-D array",
                        ax, an
                    )));
                }
            }
            let k = axes_to_move.len();
            let destination: Vec<usize> = (an - k..an).collect();
            Cow::Owned(crate::array_ops::axis_ops::moveaxis(
                a,
                axes_to_move,
                &destination,
            )?)
        }
    };

    let a_shape = a_reordered.shape();
    let old_shape: Vec<usize> = a_shape[bn..].to_vec();
    let prod: usize = old_shape.iter().product();

    if a_reordered.size() != prod * prod {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "tensorsolve: array with shape {:?} does not satisfy prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
            a_shape
        )));
    }

    let a_mat = a_reordered.try_reshape(&[prod, prod])?;
    let b_flat = b.try_reshape(&[prod])?;
    let x = a_mat.solve(&b_flat)?;
    x.try_reshape(&old_shape)
}

/// Compute the "inverse" of an N-dimensional array relative to the
/// `tensordot` operation -- NumPy's `numpy.linalg.tensorinv`. Up to
/// floating-point accuracy, `tensordot(tensorinv(a, ind), a, ind)` is the
/// identity tensor for that contraction.
///
/// `a`'s shape must be "square": `prod(a.shape()[..ind]) ==
/// prod(a.shape()[ind..])`. The result's shape is `a.shape()[ind..] +
/// a.shape()[..ind]`.
///
/// # Errors
/// * `InvalidOperation` if `ind == 0`.
/// * `DimensionMismatch` if `ind > a.ndim()`, or if `a`'s shape is not
///   "square" in the sense above.
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensorinv;
///
/// // For a plain 2-D square matrix, `ind = 1` degenerates to the ordinary
/// // matrix inverse.
/// let a = Array::from_vec(vec![4.0, 0.0, 0.0, 2.0]).reshape(&[2, 2]);
/// let a_inv = tensorinv(&a, 1).expect("tensorinv should succeed");
/// assert_eq!(a_inv.shape(), vec![2, 2]);
/// assert!((a_inv.get(&[0, 0]).expect("valid") - 0.25).abs() < 1e-10);
/// assert!((a_inv.get(&[1, 1]).expect("valid") - 0.5).abs() < 1e-10);
/// ```
pub fn tensorinv<T>(a: &Array<T>, ind: usize) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    if ind == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "tensorinv: ind must be a positive integer".to_string(),
        ));
    }

    let old_shape = a.shape();
    if ind > old_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "tensorinv: ind ({}) must be <= a.ndim ({})",
            ind,
            old_shape.len()
        )));
    }

    let prod_head: usize = old_shape[..ind].iter().product();
    let prod_tail: usize = old_shape[ind..].iter().product();

    if prod_head != prod_tail {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "tensorinv: array with shape {:?} is not 'square' for ind={} (prod(shape[..ind]) = {}, prod(shape[ind..]) = {})",
            old_shape, ind, prod_head, prod_tail
        )));
    }

    let reshaped = a.try_reshape(&[prod_tail, prod_head])?;
    let inv_mat = reshaped.inv()?;

    let mut inv_shape: Vec<usize> = old_shape[ind..].to_vec();
    inv_shape.extend_from_slice(&old_shape[..ind]);
    inv_mat.try_reshape(&inv_shape)
}

//! Zero-copy (when possible) operand access for [`Array`].
//!
//! Every kernel in this module (`elementwise`, `reduce`, `gemm`) operates
//! on flat slices, not on [`Array<T>`] directly. [`operand`] is the one
//! place that bridges the two: it borrows an [`Array<T>`]'s data as a
//! flat, *logically ordered* slice, at zero cost whenever possible.

use crate::array::Array;
use std::ops::Deref;

/// A borrowed-or-owned view of one kernel operand, produced by
/// [`operand`].
///
/// [`Operand::Borrowed`] is a genuine zero-copy borrow of `a`'s own
/// backing storage (the common case: `a` is contiguous).
/// [`Operand::Owned`] is a freshly materialized copy, needed only when `a`
/// is not contiguous (e.g. a permuted-axes view produced by
/// [`Array::transpose_axis`]) and so has no single contiguous slice to
/// borrow.
///
/// Deliberately `Deref`, never `DerefMut`: `operand()` only ever borrows
/// from a shared `&Array<T>`, so a `Borrowed(&'a [T])` variant aliases
/// the source array's own storage through a shared reference -- there is
/// no permission to mutate through it, and adding `DerefMut` would either
/// be unimplementable for that variant or require silently falling back
/// to a private copy on write, which is not this type's job. Kernels
/// needing an output buffer (e.g. `gemm::gemm_2d`'s `c: &mut [T]`) build
/// one directly (a fresh `Vec<T>`) rather than obtaining it from
/// `operand()`.
pub(crate) enum Operand<'a, T: Clone> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<'a, T: Clone> Deref for Operand<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Operand::Borrowed(s) => s,
            Operand::Owned(v) => v.as_slice(),
        }
    }
}

/// Borrow `a`'s data as a flat, logically-ordered [`Operand`].
///
/// Zero-copy ([`Operand::Borrowed`]) whenever `a` is contiguous
/// ([`Array::as_slice`] returns `Some`). Otherwise materializes an owned
/// copy by walking `a.array().iter()` -- which respects strides, and so
/// yields elements in the array's *logical* order for its current shape
/// -- directly, rather than through [`Array::to_vec`]. The two approaches
/// return identical data in this (non-contiguous) branch, since
/// `to_vec()` itself falls back to the same `iter().cloned().collect()`
/// for non-standard-layout arrays; going straight to `iter()` here just
/// skips `to_vec()`'s redundant leading contiguity check, since `operand`
/// already knows (from `as_slice()` returning `None`) that the array is
/// non-contiguous.
pub(crate) fn operand<T: Clone>(a: &Array<T>) -> Operand<'_, T> {
    match a.as_slice() {
        Some(s) => Operand::Borrowed(s),
        None => Operand::Owned(a.array().iter().cloned().collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_borrows_contiguous_array() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let op = operand(&a);
        assert!(matches!(op, Operand::Borrowed(_)));
        assert_eq!(&*op, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn operand_borrows_contiguous_reshaped_array() {
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let op = operand(&a);
        assert!(matches!(op, Operand::Borrowed(_)));
        assert_eq!(&*op, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn operand_materializes_non_contiguous_array_in_logical_order() {
        // transpose_axis on a 2D array produces a permuted (non-contiguous)
        // view; `operand` must return its LOGICAL order (matching
        // `to_vec()`/`iter()`), not raw memory order.
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let t = a.transpose_axis(0, 1); // shape [3, 2]
        assert!(!t.is_c_contiguous());
        let op = operand(&t);
        assert!(matches!(op, Operand::Owned(_)));
        assert_eq!(&*op, &[1, 4, 2, 5, 3, 6]);
        assert_eq!(op.to_vec(), t.to_vec());
    }

    #[test]
    fn operand_empty_array() {
        let a: Array<f64> = Array::from_vec(vec![]);
        let op = operand(&a);
        assert_eq!(op.len(), 0);
    }
}

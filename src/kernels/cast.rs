//! `TypeId`-guarded slice/`Vec`/scalar reinterpretation.
//!
//! This is the **only** module in `kernels/` allowed to type-pun via
//! `unsafe`. Every other dispatch kernel in `kernels/` that needs to
//! reinterpret a generic `&[T]`/`Vec<T>`/`T` as its concretely-typed
//! `f64`/`f32` form (to hand it to one of `scirs2-core`'s
//! `SimdUnifiedOps`/`simd_ops::matmul` routines, or to hand a computed
//! `f64`/`f32` scalar back) goes through the functions here, which
//! replace the ad hoc per-call-site `TypeId` + raw-pointer-cast idiom
//! used throughout the rest of the crate today (e.g.
//! `stats/basic.rs`'s `let ptr = x as *const T as *const f64; *ptr`,
//! `array/operations_optimized.rs`'s `mem::transmute_copy`) with one
//! centralized, tested implementation.
//!
//! # Soundness
//!
//! Every function first checks `TypeId::of::<T>() == TypeId::of::<f64>()`
//! (or `f32`). `TypeId` is defined only for `'static` types, and two
//! `'static` types compare equal under `TypeId` if and only if they are
//! the *same* type -- so a positive check is a proof, not a heuristic,
//! that `T` and `f64` (or `f32`) are the same type: identical size,
//! alignment, bit representation, and drop behavior (both `f64`/`f32` are
//! `Copy` and have no `Drop` impl). Reinterpreting a `&[T]`/`&mut
//! [T]`/`Vec<T>`/`T` as the equivalent `f64`/`f32` form under that proof
//! is sound. What is *not* covered by this guarantee -- and so must never
//! be done, here or anywhere else in `kernels/` -- is transmuting a
//! *closure* or other non-data value on the strength of a `TypeId` check
//! on an unrelated generic parameter; see `elementwise.rs`'s module docs
//! for why `binary_dispatch`/`unary_dispatch` do not attempt that.

use std::any::TypeId;
use std::mem::ManuallyDrop;

/// Reinterpret `s: &[T]` as `&[f64]` iff `T == f64`, without copying.
pub(crate) fn as_f64<T: 'static>(s: &[T]) -> Option<&[f64]> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: TypeId::of::<T>() == TypeId::of::<f64>() proves T == f64
        // (module docs). `s.as_ptr()` is therefore already a valid,
        // aligned `*const f64` pointing at `s.len()` initialized `f64`
        // values borrowed for the lifetime of `s`; reinterpreting the
        // pointer's type and rebuilding a slice of the same length over
        // the same memory changes nothing about validity or aliasing.
        Some(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f64, s.len()) })
    } else {
        None
    }
}

/// Reinterpret `s: &[T]` as `&[f32]` iff `T == f32`, without copying.
pub(crate) fn as_f32<T: 'static>(s: &[T]) -> Option<&[f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: see `as_f64`; identical reasoning with f32 in place of f64.
        Some(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len()) })
    } else {
        None
    }
}

/// Reinterpret `s: &mut [T]` as `&mut [f64]` iff `T == f64`, without
/// copying.
pub(crate) fn as_f64_mut<T: 'static>(s: &mut [T]) -> Option<&mut [f64]> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T == f64 (as in `as_f64`). `s` is a unique (`&mut`)
        // borrow of `s.len()` initialized `T` (= `f64`) values, so the
        // reinterpreted `&mut [f64]` remains the sole live reference to
        // this memory for its lifetime, exactly as `s` was.
        Some(unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f64, s.len()) })
    } else {
        None
    }
}

/// Reinterpret `s: &mut [T]` as `&mut [f32]` iff `T == f32`, without
/// copying.
pub(crate) fn as_f32_mut<T: 'static>(s: &mut [T]) -> Option<&mut [f32]> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: see `as_f64_mut`; identical reasoning with f32.
        Some(unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f32, s.len()) })
    } else {
        None
    }
}

/// Reinterpret an owned `Vec<f64>` as `Vec<T>` iff `T == f64`, reusing the
/// original allocation (no element copy).
pub(crate) fn vec_from_f64<T: 'static>(v: Vec<f64>) -> Option<Vec<T>> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let mut v = ManuallyDrop::new(v);
        let ptr = v.as_mut_ptr() as *mut T;
        let len = v.len();
        let cap = v.capacity();
        // SAFETY: T == f64 (module docs), so `ptr` was allocated by the
        // global allocator with exactly the layout `Vec::<T>::from_raw_parts`
        // expects (same size and alignment as the `Vec<f64>` it came
        // from), `len <= cap`, and the first `len` elements are
        // initialized `T` values. Wrapping the original `Vec<f64>` in
        // `ManuallyDrop` hands ownership of that allocation to the new
        // `Vec<T>` exactly once, so there is no double-free; the `else`
        // branch never touches `v`, so it drops normally there.
        Some(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    } else {
        None
    }
}

/// Reinterpret an owned `Vec<f32>` as `Vec<T>` iff `T == f32`, reusing the
/// original allocation (no element copy).
pub(crate) fn vec_from_f32<T: 'static>(v: Vec<f32>) -> Option<Vec<T>> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let mut v = ManuallyDrop::new(v);
        let ptr = v.as_mut_ptr() as *mut T;
        let len = v.len();
        let cap = v.capacity();
        // SAFETY: see `vec_from_f64`; identical reasoning with f32.
        Some(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    } else {
        None
    }
}

/// Reinterpret a scalar `f64` as `T` iff `T == f64`.
pub(crate) fn f64_to<T: 'static>(x: f64) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T == f64 (module docs), so size_of::<T>() ==
        // size_of::<f64>(); `transmute_copy` reads exactly that many
        // bytes from the valid, live reference `&x` and reinterprets them
        // as `T`, which is sound because T and f64 are the same type.
        Some(unsafe { std::mem::transmute_copy::<f64, T>(&x) })
    } else {
        None
    }
}

/// Reinterpret a scalar `f32` as `T` iff `T == f32`.
pub(crate) fn f32_to<T: 'static>(x: f32) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: see `f64_to`; identical reasoning with f32.
        Some(unsafe { std::mem::transmute_copy::<f32, T>(&x) })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_f64_round_trips_for_f64() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let s = as_f64(&data).expect("T == f64 must succeed");
        assert_eq!(s, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_f64_none_for_other_types() {
        assert!(as_f64(&[1_i32, 2, 3]).is_none());
        assert!(as_f64(&[1_u8, 2, 3]).is_none());
        assert!(as_f64(&[true, false]).is_none());
        assert!(as_f64(&[1.0_f32, 2.0]).is_none());
    }

    #[test]
    fn as_f64_empty_slice() {
        let data: Vec<f64> = vec![];
        let s = as_f64(&data).expect("T == f64 must succeed even when empty");
        assert!(s.is_empty());
    }

    #[test]
    fn as_f32_round_trips_for_f32() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let s = as_f32(&data).expect("T == f32 must succeed");
        assert_eq!(s, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_f32_none_for_other_types() {
        assert!(as_f32(&[1_i32, 2, 3]).is_none());
        assert!(as_f32(&[1.0_f64, 2.0]).is_none());
        assert!(as_f32(&[1_u8]).is_none());
    }

    #[test]
    fn as_f32_empty_slice() {
        let data: Vec<f32> = vec![];
        let s = as_f32(&data).expect("T == f32 must succeed even when empty");
        assert!(s.is_empty());
    }

    #[test]
    fn as_f64_mut_round_trips_and_writes_back() {
        let mut data = vec![1.0_f64, 2.0, 3.0];
        {
            let s = as_f64_mut(&mut data).expect("T == f64 must succeed");
            s[0] = 42.0;
        }
        assert_eq!(data, vec![42.0, 2.0, 3.0]);
    }

    #[test]
    fn as_f64_mut_none_for_other_types() {
        let mut data = vec![1_i32, 2, 3];
        assert!(as_f64_mut(&mut data).is_none());
    }

    #[test]
    fn as_f32_mut_round_trips_and_writes_back() {
        let mut data = vec![1.0_f32, 2.0, 3.0];
        {
            let s = as_f32_mut(&mut data).expect("T == f32 must succeed");
            s[0] = 42.0;
        }
        assert_eq!(data, vec![42.0, 2.0, 3.0]);
    }

    #[test]
    fn vec_from_f64_round_trips() {
        let v = vec![1.0_f64, 2.0, 3.0];
        let out: Vec<f64> = vec_from_f64(v).expect("T == f64 must succeed");
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn vec_from_f64_none_for_other_types() {
        let v = vec![1.0_f64, 2.0, 3.0];
        assert!(vec_from_f64::<i32>(v).is_none());
        let v2 = vec![1.0_f64, 2.0, 3.0];
        assert!(vec_from_f64::<u8>(v2).is_none());
    }

    #[test]
    fn vec_from_f64_empty() {
        let v: Vec<f64> = vec![];
        let out: Vec<f64> = vec_from_f64(v).expect("T == f64 must succeed even when empty");
        assert!(out.is_empty());
    }

    #[test]
    fn vec_from_f32_round_trips() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let out: Vec<f32> = vec_from_f32(v).expect("T == f32 must succeed");
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn vec_from_f32_none_for_other_types() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(vec_from_f32::<bool>(v).is_none());
    }

    #[test]
    fn vec_from_f32_empty() {
        let v: Vec<f32> = vec![];
        let out: Vec<f32> = vec_from_f32(v).expect("T == f32 must succeed even when empty");
        assert!(out.is_empty());
    }

    #[test]
    fn f64_to_round_trips() {
        let x: f64 = f64_to(3.5_f64).expect("T == f64 must succeed");
        assert_eq!(x, 3.5);
    }

    #[test]
    fn f64_to_none_for_other_types() {
        assert!(f64_to::<i32>(3.5).is_none());
        assert!(f64_to::<f32>(3.5).is_none());
    }

    #[test]
    fn f32_to_round_trips() {
        let x: f32 = f32_to(3.5_f32).expect("T == f32 must succeed");
        assert_eq!(x, 3.5);
    }

    #[test]
    fn f32_to_none_for_other_types() {
        assert!(f32_to::<i32>(3.5).is_none());
        assert!(f32_to::<f64>(3.5).is_none());
    }
}

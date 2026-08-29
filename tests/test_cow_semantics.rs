//! Black-box copy-on-write semantics for `Array<T>`.
//!
//! `Array`'s owned buffer lives behind an `Arc`, which makes `Clone` an O(1)
//! reference-count bump instead of an O(n) deep copy. That is a *representation*
//! change only: every mutating path unshares the buffer (via the single
//! `Arc::make_mut` in `src/array/core.rs`) before it writes, so no caller can
//! ever observe one handle's write through another.
//!
//! These tests go through the public API exclusively -- they never touch
//! `is_unique()` or any internal -- so they assert exactly what a user of the
//! crate is entitled to rely on: **a clone behaves like an independent array**.
//! The white-box counterparts in `src/array/core.rs`'s `cow_tests` module
//! assert the other half (that sharing really does happen, so the O(1) clone is
//! real rather than an accidental deep copy).
//!
//! Every assertion here would also have passed before the `Arc` landed. That is
//! the point: this file is a regression net for the value semantics, not a
//! description of new behaviour.

use numrs2::array::Array;
use numrs2::array_ops::creation::{may_share_memory, shares_memory};
use numrs2::masked::MaskedArray;

fn base() -> Array<f64> {
    Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

const ORIGINAL: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

// ---------------------------------------------------------------------
// set(): the canonical single-element write
// ---------------------------------------------------------------------

#[test]
fn set_on_a_clone_leaves_the_original_untouched() {
    let original = base();
    let mut copy = original.clone();

    copy.set(&[0], 99.0).expect("index 0 is in bounds");

    assert_eq!(copy.to_vec(), vec![99.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn set_on_the_original_leaves_a_clone_untouched() {
    let mut original = base();
    let copy = original.clone();

    original.set(&[5], -1.0).expect("index 5 is in bounds");

    assert_eq!(original.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, -1.0]);
    assert_eq!(copy.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn interleaved_writes_to_two_clones_never_cross() {
    let mut a = base();
    let mut b = a.clone();

    for i in 0..6 {
        a.set(&[i], 100.0 + i as f64).expect("in bounds");
        b.set(&[i], 200.0 + i as f64).expect("in bounds");
    }

    assert_eq!(a.to_vec(), vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);
    assert_eq!(b.to_vec(), vec![200.0, 201.0, 202.0, 203.0, 204.0, 205.0]);
}

#[test]
fn a_chain_of_clones_stays_mutually_independent() {
    let a = base();
    let b = a.clone();
    let mut c = b.clone();
    let d = c.clone();

    c.set(&[2], 0.0).expect("index 2 is in bounds");

    assert_eq!(c.to_vec(), vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0]);
    for (name, arr) in [("a", &a), ("b", &b), ("d", &d)] {
        assert_eq!(
            arr.to_vec(),
            ORIGINAL.to_vec(),
            "{name} must still see the original values"
        );
    }
}

// ---------------------------------------------------------------------
// as_slice_mut() / array_mut(): bulk writes through a borrowed buffer
// ---------------------------------------------------------------------

#[test]
fn as_slice_mut_writes_do_not_leak_into_a_clone() {
    let original = base();
    let mut copy = original.clone();

    let slice = copy
        .as_slice_mut()
        .expect("a freshly built 1-D array is contiguous");
    for (i, x) in slice.iter_mut().enumerate() {
        *x = -(i as f64);
    }

    assert_eq!(copy.to_vec(), vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn array_mut_writes_do_not_leak_into_a_clone() {
    let original = base();
    let mut copy = original.clone();

    copy.array_mut().fill(7.0);

    assert_eq!(copy.to_vec(), vec![7.0; 6]);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn as_slice_mut_taken_twice_keeps_writing_to_the_same_buffer() {
    // The unshare happens on the *first* mutable acquisition; a second one
    // must not silently hand back a different (re-copied) buffer and lose
    // the first batch of writes.
    let original = base();
    let mut copy = original.clone();

    copy.as_slice_mut().expect("contiguous")[0] = 10.0;
    copy.as_slice_mut().expect("contiguous")[1] = 20.0;

    assert_eq!(copy.to_vec(), vec![10.0, 20.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

// ---------------------------------------------------------------------
// map_inplace(): a whole-array in-place transform
// ---------------------------------------------------------------------

#[test]
fn map_inplace_on_a_clone_leaves_the_original_untouched() {
    let original = base();
    let mut copy = original.clone();

    copy.map_inplace(|x| x * 10.0);

    assert_eq!(copy.to_vec(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn map_inplace_on_the_original_leaves_a_clone_untouched() {
    let mut original = base();
    let copy = original.clone();

    original.map_inplace(|x| x + 1.0);

    assert_eq!(original.to_vec(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(copy.to_vec(), ORIGINAL.to_vec());
}

// ---------------------------------------------------------------------
// Output-parameter APIs: the destination must be unshared before writing
// ---------------------------------------------------------------------

#[test]
fn map_to_output_aliasing_does_not_corrupt_the_outputs_clone() {
    let source = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let scratch = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let mut output = scratch.clone();

    source
        .map_to(|x| x * 3.0, &mut output)
        .expect("shapes match");

    assert_eq!(output.to_vec(), vec![3.0, 6.0, 9.0, 12.0]);
    assert_eq!(
        scratch.to_vec(),
        vec![0.0; 4],
        "the scratch buffer's other handle must still read as zeros"
    );
}

#[test]
fn map_to_writing_into_a_clone_of_its_own_source_is_safe() {
    // `output` starts life sharing `source`'s buffer. Unsharing must happen
    // before the first write, or the reads of `source` inside `map_to` would
    // see values the loop had already overwritten.
    let source = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut output = source.clone();

    source
        .map_to(|x| x * 2.0, &mut output)
        .expect("shapes match");

    assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    assert_eq!(source.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_to_output_aliasing_does_not_corrupt_the_outputs_clone() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0])
        .try_reshape(&[2, 2])
        .expect("4 elements reshape to 2x2");
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0])
        .try_reshape(&[2, 2])
        .expect("4 elements reshape to 2x2");

    // `matmul_to` accumulates onto `output`, so it must start at zero for
    // the result to be a plain product.
    let zeros = Array::<f64>::zeros(&[2, 2]);
    let mut output = zeros.clone();

    a.matmul_to(&b, &mut output).expect("2x2 by 2x2 is valid");

    assert_eq!(output.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    assert_eq!(
        zeros.to_vec(),
        vec![0.0; 4],
        "the zero buffer's other handle must not have been accumulated into"
    );
}

#[test]
fn matmul_to_into_a_clone_of_an_operand_is_safe() {
    let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0])
        .try_reshape(&[2, 2])
        .expect("4 elements reshape to 2x2");
    let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0])
        .try_reshape(&[2, 2])
        .expect("4 elements reshape to 2x2");

    // `output` shares `b`'s buffer on entry.
    let mut output = b.clone();
    a.matmul_to(&b, &mut output).expect("2x2 by 2x2 is valid");

    // identity * b accumulated onto a copy of b == 2b.
    assert_eq!(output.to_vec(), vec![4.0, 6.0, 8.0, 10.0]);
    assert_eq!(
        b.to_vec(),
        vec![2.0, 3.0, 4.0, 5.0],
        "the operand must not have been mutated through the aliased output"
    );
}

// ---------------------------------------------------------------------
// Non-standard layouts: copy-on-write must respect *logical* order
// ---------------------------------------------------------------------

#[test]
fn an_f_layout_clone_mutates_independently_and_keeps_logical_order() {
    let c_layout = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .try_reshape(&[2, 3])
        .expect("6 elements reshape to 2x3");
    let f_layout = c_layout.to_f_layout();

    // `to_f_layout` reverses the axes, so the shape flips to 3x2 and the
    // logical (row-major) reading of the result is the transpose.
    assert_eq!(f_layout.shape(), vec![3, 2]);
    let logical_before = f_layout.to_vec();
    assert_eq!(logical_before, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let mut copy = f_layout.clone();
    copy.set(&[0, 1], -4.0).expect("index [0,1] is in bounds");

    assert_eq!(
        copy.to_vec(),
        vec![1.0, -4.0, 2.0, 5.0, 3.0, 6.0],
        "the write must land at the LOGICAL position, not a raw buffer offset"
    );
    assert_eq!(
        f_layout.to_vec(),
        logical_before,
        "the original F-layout array must be untouched"
    );
    assert_eq!(
        c_layout.to_vec(),
        ORIGINAL.to_vec(),
        "the C-layout ancestor must be untouched too"
    );
}

#[test]
fn a_permuted_axes_clone_mutates_independently_and_keeps_logical_order() {
    let original = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .try_reshape(&[2, 3])
        .expect("6 elements reshape to 2x3");
    let permuted = original.transpose_axis(0, 1);

    assert_eq!(permuted.shape(), vec![3, 2]);
    let logical_before = permuted.to_vec();
    assert_eq!(logical_before, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let mut copy = permuted.clone();
    copy.set(&[2, 0], 0.0).expect("index [2,0] is in bounds");

    assert_eq!(
        copy.to_vec(),
        vec![1.0, 4.0, 2.0, 5.0, 0.0, 6.0],
        "the write must land at the LOGICAL position under permuted strides"
    );
    assert_eq!(permuted.to_vec(), logical_before);
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn reshaping_a_clone_does_not_reshape_the_original() {
    let original = base();
    let reshaped = original
        .clone()
        .try_reshape(&[2, 3])
        .expect("6 elements reshape to 2x3");

    assert_eq!(reshaped.shape(), vec![2, 3]);
    assert_eq!(
        original.shape(),
        vec![6],
        "the source array's shape must be unchanged"
    );
    assert_eq!(original.to_vec(), ORIGINAL.to_vec());
}

// ---------------------------------------------------------------------
// MaskedArray: the aliasing pattern flagged in the COW design
// ---------------------------------------------------------------------

#[test]
fn mutating_a_clone_of_a_masked_arrays_base_leaves_the_mask_side_intact() {
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = Array::from_vec(vec![false, true, false, true, false]);

    // The MaskedArray takes its own handles on both buffers...
    let masked = MaskedArray::new(data.clone(), Some(mask.clone()), Some(0.0))
        .expect("data and mask have matching shapes");

    // ...so mutating the caller's still-live clone of the base must not be
    // visible inside the MaskedArray.
    let mut base_clone = data.clone();
    base_clone.map_inplace(|x| x * -1.0);
    let mut mask_clone = mask.clone();
    mask_clone.set(&[0], true).expect("index 0 is in bounds");

    assert_eq!(
        masked.get_data().to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        "the masked array's data must be untouched"
    );
    assert_eq!(
        masked.get_mask().to_vec(),
        vec![false, true, false, true, false],
        "the masked array's mask must be untouched"
    );
    assert_eq!(base_clone.to_vec(), vec![-1.0, -2.0, -3.0, -4.0, -5.0]);
    assert_eq!(mask_clone.to_vec(), vec![true, true, false, true, false]);
}

#[test]
fn mutating_a_masked_array_leaves_the_source_arrays_intact() {
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = Array::from_vec(vec![false, true, false, true, false]);

    let mut masked = MaskedArray::new(data.clone(), Some(mask.clone()), Some(0.0))
        .expect("data and mask have matching shapes");

    masked
        .set(&[0], 42.0, Some(true))
        .expect("index 0 is in bounds");

    assert_eq!(masked.get_data().to_vec()[0], 42.0);
    assert!(masked.get_mask().to_vec()[0]);
    assert_eq!(
        data.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        "the caller's data array must be untouched"
    );
    assert_eq!(
        mask.to_vec(),
        vec![false, true, false, true, false],
        "the caller's mask array must be untouched"
    );
}

// ---------------------------------------------------------------------
// The aliasing predicates must keep reporting clones as non-aliasing
// ---------------------------------------------------------------------

#[test]
fn a_clone_is_not_reported_as_sharing_memory() {
    // A clone shares bytes but can never observe a write made through the
    // other handle, so both NumPy-compatible predicates must say "no".
    let a = base();
    let c = a.clone();

    assert!(!may_share_memory(&a, &c));
    assert!(!shares_memory(&a, &c));

    // ...while an array still aliases itself.
    assert!(may_share_memory(&a, &a));
    assert!(shares_memory(&a, &a));
}

// ---------------------------------------------------------------------
// Sanity: reads through a shared buffer are consistent across handles
// ---------------------------------------------------------------------

#[test]
fn every_read_accessor_agrees_across_a_shared_pair() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .try_reshape(&[2, 3])
        .expect("6 elements reshape to 2x3");
    let b = a.clone();

    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.strides(), b.strides());
    assert_eq!(a.ndim(), b.ndim());
    assert_eq!(a.size(), b.size());
    assert_eq!(a.nbytes(), b.nbytes());
    assert_eq!(a.to_vec(), b.to_vec());
    assert_eq!(a.as_slice(), b.as_slice());
    assert_eq!(a.is_c_contiguous(), b.is_c_contiguous());
    assert_eq!(format!("{a}"), format!("{b}"));
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a.sum_all(), b.sum_all());
}

#[test]
fn arithmetic_between_two_handles_of_one_buffer_is_correct() {
    let a = base();
    let b = a.clone();

    // Both operands read the *same* buffer; the result must still be a
    // fresh array with the elementwise sum, and neither operand disturbed.
    let sum = a.add(&b);
    assert_eq!(sum.to_vec(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    let diff = a.subtract(&b);
    assert_eq!(diff.to_vec(), vec![0.0; 6]);

    assert_eq!(a.to_vec(), ORIGINAL.to_vec());
    assert_eq!(b.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn a_clone_survives_the_originals_drop() {
    let copy = {
        let original = base();
        let copy = original.clone();
        drop(original);
        copy
    };
    assert_eq!(copy.to_vec(), ORIGINAL.to_vec());
}

#[test]
fn a_clone_can_be_mutated_after_the_original_is_dropped() {
    let mut copy = {
        let original = base();
        original.clone()
    };
    copy.set(&[0], 1234.0).expect("index 0 is in bounds");
    assert_eq!(copy.to_vec(), vec![1234.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

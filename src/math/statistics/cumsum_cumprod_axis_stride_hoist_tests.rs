//! Regression tests for the `Some(axis)` stride hoist in `cumsum`/`cumprod`.
//!
//! Extracted verbatim from `mod.rs`, where these lived as an inline `mod` block, so that
//! every file in this module stays under the 2,000-line cap. `super` is `math::statistics`
//! here exactly as it was inline.

use super::*;
use crate::array::Array;

#[test]
fn cumsum_axis_matches_naive_2d_small() {
    // Sequential branch (n_iterations = 2, well under PARALLEL_THRESHOLD).
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let axis0 = cumsum(&a, Some(0), None).expect("cumsum should succeed");
    assert_eq!(axis0.to_vec(), vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);

    let axis1 = cumsum(&a, Some(1), None).expect("cumsum should succeed");
    assert_eq!(axis1.to_vec(), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn cumprod_axis_matches_naive_2d_small() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let axis0 = cumprod(&a, Some(0), None).expect("cumprod should succeed");
    assert_eq!(axis0.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]);

    let axis1 = cumprod(&a, Some(1), None).expect("cumprod should succeed");
    assert_eq!(axis1.to_vec(), vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
}

/// Exercises the *parallel* axis branch (`n_iterations` == `rows` == 2000, well above
/// `PARALLEL_THRESHOLD` == 1000, with `axis = 1` so `axis_stride == 1`), where the
/// flat-index base-plus-stride hoist applies inside the rayon closure -- validated here
/// against an independent, naive per-row computation sharing no code with
/// `cumsum_no_out`'s hoisted implementation.
#[test]
fn cumsum_axis_matches_naive_parallel_branch() {
    let rows = 2000usize;
    let cols = 4usize;
    let data: Vec<f64> = (0..rows * cols).map(|i| (i % 13) as f64 - 6.0).collect();
    let a = Array::from_vec(data.clone()).reshape(&[rows, cols]);

    let got = cumsum(&a, Some(1), None)
        .expect("cumsum should succeed")
        .to_vec();

    let mut expected = vec![0.0; rows * cols];
    for r in 0..rows {
        let mut running = 0.0;
        for c in 0..cols {
            running += data[r * cols + c];
            expected[r * cols + c] = running;
        }
    }
    assert_eq!(got, expected);
}

#[test]
fn cumprod_axis_matches_naive_parallel_branch() {
    let rows = 2000usize;
    let cols = 4usize;
    // Keep values close to 1 so the product doesn't overflow/underflow across 4 steps.
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| 1.0 + ((i % 7) as f64 - 3.0) * 0.01)
        .collect();
    let a = Array::from_vec(data.clone()).reshape(&[rows, cols]);

    let got = cumprod(&a, Some(1), None)
        .expect("cumprod should succeed")
        .to_vec();

    let mut expected = vec![0.0; rows * cols];
    for r in 0..rows {
        let mut running = 1.0;
        for c in 0..cols {
            running *= data[r * cols + c];
            expected[r * cols + c] = running;
        }
    }
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-9, "got {g}, expected {e}");
    }
}

/// `axis = 0` on a `[rows, cols]` array gives `axis_stride == cols` (not 1), unlike the
/// `axis = 1` cases above where `axis_stride == 1` -- this exercises the hoist with a
/// non-unit stride, still against an independent naive computation, still large enough
/// (`n_iterations == cols == 2000`) to hit the parallel branch.
#[test]
fn cumsum_axis0_matches_naive_parallel_branch() {
    let rows = 3usize;
    let cols = 2000usize;
    let data: Vec<f64> = (0..rows * cols).map(|i| (i % 11) as f64 - 5.0).collect();
    let a = Array::from_vec(data.clone()).reshape(&[rows, cols]);

    let got = cumsum(&a, Some(0), None)
        .expect("cumsum should succeed")
        .to_vec();

    let mut expected = vec![0.0; rows * cols];
    for c in 0..cols {
        let mut running = 0.0;
        for r in 0..rows {
            running += data[r * cols + c];
            expected[r * cols + c] = running;
        }
    }
    assert_eq!(got, expected);
}

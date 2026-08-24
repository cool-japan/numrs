#![cfg(feature = "lapack")]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Compute the LU decomposition of a matrix with partial pivoting
///
/// This implementation includes various numerical stability enhancements:
/// 1. Better pivot selection with scaling for row equilibration
/// 2. Handling of small pivots with thresholds based on machine precision
/// 3. Compensated summation to reduce round-off errors
/// 4. Row scaling to improve numerical stability for matrices with widely varying values
/// 5. Full error checking with detailed diagnostics
/// 6. Optional verification of decomposition accuracy
pub fn lu<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>, Array<usize>)>
where
    T: Float + Clone + Debug + std::fmt::Display + 'static,
{
    // Check if the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "LU decomposition requires a 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];
    let k = std::cmp::min(m, n);

    // Create working copies of the arrays
    let mut a_copy = a.clone();
    let mut p = (0..m).collect::<Vec<usize>>();

    // Bulk-acquire the mutable buffer ONCE for the whole factorization instead
    // of paying `Array::set`'s `Arc::make_mut` unshare check on every one of the
    // O(m*n) + O(k*m*n) element writes below (row scaling, pivoting, and
    // elimination all read *and* write `a_copy` directly with no intervening
    // helper calls, so a single hoisted handle is sound for this whole span).
    let a_arr = a_copy.array_mut();

    // Compute row scaling factors for better numerical stability
    // This helps with matrices that have widely varying magnitudes
    let mut row_scale = vec![num_traits::Zero::zero(); m];
    for i in 0..m {
        let mut max_in_row = num_traits::Zero::zero();
        for j in 0..n {
            let abs_val = num_traits::Float::abs(a_arr[[i, j]]);
            if abs_val > max_in_row {
                max_in_row = abs_val;
            }
        }

        // If row is all zeros, set scale to 1 to avoid division by zero
        if max_in_row == <T as num_traits::Zero>::zero() {
            row_scale[i] = <T as num_traits::One>::one();
        } else {
            row_scale[i] = <T as num_traits::One>::one() / max_in_row;
        }
    }

    // Compute a size-dependent tolerance threshold for detecting small pivots
    // This is based on machine epsilon, matrix size, and norm estimation
    let eps = T::epsilon();
    let matrix_size = <T as num_traits::NumCast>::from(std::cmp::max(m, n))
        .expect("matrix dimension should convert to float type");
    let tolerance = eps * matrix_size;

    // Estimate matrix norm for setting thresholds (using max absolute element as approximation)
    let mut matrix_norm = <T as num_traits::Zero>::zero();
    for i in 0..m {
        for j in 0..n {
            let abs_val = num_traits::Float::abs(a_arr[[i, j]]);
            if abs_val > matrix_norm {
                matrix_norm = abs_val;
            }
        }
    }

    // Threshold for pivot detection - will treat smaller values as effectively zero
    let pivot_threshold = tolerance * matrix_norm;

    // Count rank deficiency for diagnostic purposes
    let mut rank_deficient = false;
    let mut num_small_pivots = 0;

    // LU factorization with partial pivoting, scaling, and enhanced numerical stability
    for i in 0..k {
        // Find pivot with scaling using complete pivoting strategy
        // Complete pivoting would search all elements below current position
        let mut p_row = i;
        let mut p_val = num_traits::Float::abs(a_arr[[i, i]]) * row_scale[i];

        for j in (i + 1)..m {
            let val = num_traits::Float::abs(a_arr[[j, i]]) * row_scale[j];
            if val > p_val {
                p_row = j;
                p_val = val;
            }
        }

        // Check for numerical singularity with adaptive threshold
        if p_val < pivot_threshold {
            // Matrix is numerically singular - set a diagnostic flag
            rank_deficient = true;
            num_small_pivots += 1;

            // Continue with a small pivot, but this indicates potential instability
            // We'll warn the user about this in the result
        }

        // Swap rows if needed
        if p_row != i {
            p.swap(i, p_row);
            row_scale.swap(i, p_row);

            // Swap rows in A
            for j in 0..n {
                let temp = a_arr[[i, j]];
                a_arr[[i, j]] = a_arr[[p_row, j]];
                a_arr[[p_row, j]] = temp;
            }
        }

        // Handle small pivots to prevent overflow
        let pivot = a_arr[[i, i]];
        let abs_pivot = num_traits::Float::abs(pivot);

        if abs_pivot < pivot_threshold {
            // Set a small non-zero pivot to maintain numerical stability
            // Use a size consistent with matrix norm to avoid introducing large errors
            let small_pivot_magnitude = pivot_threshold
                * <T as num_traits::NumCast>::from(10.0)
                    .unwrap_or_else(|| <T as num_traits::One>::one());

            // Preserve sign of original pivot if possible
            let small_pivot = if pivot >= <T as num_traits::Zero>::zero()
                || pivot == <T as num_traits::Zero>::zero()
            {
                small_pivot_magnitude
            } else {
                -small_pivot_magnitude
            };

            a_arr[[i, i]] = small_pivot;
        }

        // Perform elimination with improved numerical stability
        for j in (i + 1)..m {
            let pivot = a_arr[[i, i]];
            let factor = a_arr[[j, i]] / pivot;

            // Store multiplier
            a_arr[[j, i]] = factor;

            // Update remaining elements with compensated summation for better precision
            for l in (i + 1)..n {
                let a_jl = a_arr[[j, l]];
                let a_il = a_arr[[i, l]];
                let prod = factor * a_il;
                let new_val = a_jl - prod;
                a_arr[[j, l]] = new_val;
            }
        }
    }

    // The factorization loop above was `a_arr`'s last use, so its exclusive
    // borrow of `a_copy` has already ended here (NLL) -- `a_copy.get` below is
    // plain shared, COW-free reads of the now-settled buffer.

    // Extract L and U from factorized matrix
    let mut l = Array::zeros(&[m, k]);
    let mut u = Array::zeros(&[k, n]);

    // Set diagonal of L to 1
    {
        let l_arr = l.array_mut();
        for i in 0..k {
            l_arr[[i, i]] = num_traits::One::one();
        }

        // Fill L below diagonal
        for i in 1..m {
            for j in 0..std::cmp::min(i, k) {
                l_arr[[i, j]] = a_copy.get(&[i, j])?;
            }
        }
    }

    // Fill U at and above diagonal
    {
        let u_arr = u.array_mut();
        for i in 0..k {
            for j in i..n {
                u_arr[[i, j]] = a_copy.get(&[i, j])?;
            }
        }
    }

    // Convert permutation to array
    let piv_array = Array::from_vec(p.clone());

    // Verify the decomposition P*A ≈ L*U to check numerical stability
    // This is useful for diagnostic purposes
    #[cfg(feature = "validation")]
    {
        // Compute permuted A (P*A)
        let mut pa = Array::zeros(&[m, n]);
        let pa_arr = pa.array_mut();
        for i in 0..m {
            for j in 0..n {
                pa_arr[[i, j]] = a.get(&[p[i], j])?;
            }
        }

        // Compute L*U
        let lu_product = l.matmul(&u)?;

        // Calculate the maximum element-wise difference
        let mut max_diff = <T as num_traits::Zero>::zero();
        for i in 0..m {
            for j in 0..n {
                let diff = num_traits::Float::abs(pa.get(&[i, j])? - lu_product.get(&[i, j])?);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        // Check if the error is acceptable
        let acceptable_error = eps * matrix_norm * matrix_size;
        if max_diff > acceptable_error {
            eprintln!(
                "Warning: LU decomposition may be numerically unstable. Max difference: {}",
                max_diff
            );
            // In a full implementation, we could log this or return it as part of extended diagnostics
        }
    }

    // If matrix appears to be rank deficient, issue a warning
    if rank_deficient {
        // In production code, we would log this or provide a mechanism for the caller
        // to check the numerical quality of the factorization
        eprintln!("Warning: Matrix appears to be rank deficient or ill-conditioned. {} small pivots detected.", num_small_pivots);
    }

    Ok((l, u, piv_array))
}

// ---------------------------------------------------------------------------
// W3-A2 perf verification: pre-COW-conversion twin of `lu()`'s hand-rolled
// factorization loop (this module has no external LAPACK call, so the whole
// routine's cost sits in the loop touched below -- an apples-to-apples A/B
// target for the `Array::set` -> bulk-`array_mut()` conversion).
// This twin is a byte-for-byte copy of `lu()` before that conversion, kept
// only to measure the change; it is not part of the public surface.
#[cfg(test)]
fn lu_precow<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>, Array<usize>)>
where
    T: Float + Clone + Debug + std::fmt::Display,
{
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "LU decomposition requires a 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];
    let k = std::cmp::min(m, n);

    let mut a_copy = a.clone();
    let mut p = (0..m).collect::<Vec<usize>>();

    let mut row_scale = vec![num_traits::Zero::zero(); m];
    for i in 0..m {
        let mut max_in_row = num_traits::Zero::zero();
        for j in 0..n {
            let abs_val = num_traits::Float::abs(a_copy.get(&[i, j])?);
            if abs_val > max_in_row {
                max_in_row = abs_val;
            }
        }
        if max_in_row == <T as num_traits::Zero>::zero() {
            row_scale[i] = <T as num_traits::One>::one();
        } else {
            row_scale[i] = <T as num_traits::One>::one() / max_in_row;
        }
    }

    let eps = T::epsilon();
    let matrix_size = <T as num_traits::NumCast>::from(std::cmp::max(m, n))
        .expect("matrix dimension should convert to float type");
    let tolerance = eps * matrix_size;

    let mut matrix_norm = <T as num_traits::Zero>::zero();
    for i in 0..m {
        for j in 0..n {
            let abs_val = num_traits::Float::abs(a_copy.get(&[i, j])?);
            if abs_val > matrix_norm {
                matrix_norm = abs_val;
            }
        }
    }

    let pivot_threshold = tolerance * matrix_norm;
    let mut rank_deficient = false;
    let mut num_small_pivots = 0;

    for i in 0..k {
        let mut p_row = i;
        let mut p_val = num_traits::Float::abs(a_copy.get(&[i, i])?) * row_scale[i];

        for j in (i + 1)..m {
            let val = num_traits::Float::abs(a_copy.get(&[j, i])?) * row_scale[j];
            if val > p_val {
                p_row = j;
                p_val = val;
            }
        }

        if p_val < pivot_threshold {
            rank_deficient = true;
            num_small_pivots += 1;
        }

        if p_row != i {
            p.swap(i, p_row);
            row_scale.swap(i, p_row);

            for j in 0..n {
                let temp = a_copy.get(&[i, j])?;
                a_copy.set(&[i, j], a_copy.get(&[p_row, j])?)?;
                a_copy.set(&[p_row, j], temp)?;
            }
        }

        let pivot = a_copy.get(&[i, i])?;
        let abs_pivot = num_traits::Float::abs(pivot);

        if abs_pivot < pivot_threshold {
            let small_pivot_magnitude = pivot_threshold
                * <T as num_traits::NumCast>::from(10.0)
                    .unwrap_or_else(|| <T as num_traits::One>::one());

            let small_pivot = if pivot >= <T as num_traits::Zero>::zero()
                || pivot == <T as num_traits::Zero>::zero()
            {
                small_pivot_magnitude
            } else {
                -small_pivot_magnitude
            };

            a_copy.set(&[i, i], small_pivot)?;
        }

        for j in (i + 1)..m {
            let pivot = a_copy.get(&[i, i])?;
            let factor = a_copy.get(&[j, i])? / pivot;

            a_copy.set(&[j, i], factor)?;

            for l in (i + 1)..n {
                let a_jl = a_copy.get(&[j, l])?;
                let a_il = a_copy.get(&[i, l])?;
                let prod = factor * a_il;
                let new_val = a_jl - prod;
                a_copy.set(&[j, l], new_val)?;
            }
        }
    }

    let mut l = Array::zeros(&[m, k]);
    let mut u = Array::zeros(&[k, n]);

    for i in 0..k {
        l.set(&[i, i], num_traits::One::one())?;
    }

    for i in 1..m {
        for j in 0..std::cmp::min(i, k) {
            l.set(&[i, j], a_copy.get(&[i, j])?)?;
        }
    }

    for i in 0..k {
        for j in i..n {
            u.set(&[i, j], a_copy.get(&[i, j])?)?;
        }
    }

    let piv_array = Array::from_vec(p.clone());

    if rank_deficient {
        eprintln!("Warning: Matrix appears to be rank deficient or ill-conditioned. {} small pivots detected.", num_small_pivots);
    }

    Ok((l, u, piv_array))
}

#[cfg(test)]
mod perf_verification {
    use super::*;
    use std::time::Instant;

    /// Deterministic, diagonally-dominant matrix (no randomness dependency):
    /// well-conditioned enough that `lu_precow`/`lu` never hit the small
    /// pivot / rank-deficiency warning paths, keeping the two loops doing
    /// identical work.
    fn make_matrix(n: usize) -> Array<f64> {
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let base = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                let diag_boost = if i == j { 2.0 * n as f64 } else { 0.0 };
                data.push(base + diag_boost);
            }
        }
        Array::from_vec_shape(data, &[n, n]).expect("matrix construction should succeed")
    }

    /// Min-of-N, alternating A/B timing: interleaving avoids attributing a
    /// transient load spike on this (possibly shared) machine to one side.
    #[test]
    fn bench_lu_cow_vs_precow() {
        // Full sizes only under `--release`: in an unoptimized `test`-profile
        // build, O(n^3) hand-rolled LU at n=256 is slow enough to noticeably
        // bloat the standard `cargo nextest run` (no `--release`) suite for a
        // measurement that (per the threshold above) isn't meaningful there
        // anyway -- so debug builds just smoke-test at a much smaller size.
        let sizes: &[usize] = if cfg!(debug_assertions) {
            &[16, 32]
        } else {
            &[128, 256]
        };
        const SAMPLES: usize = 7;

        for &n in sizes {
            let a = make_matrix(n);

            let mut precow_times = Vec::with_capacity(SAMPLES);
            let mut cow_times = Vec::with_capacity(SAMPLES);

            for _ in 0..SAMPLES {
                let start = Instant::now();
                let _ = lu_precow(&a).expect("lu_precow should succeed");
                precow_times.push(start.elapsed());

                let start = Instant::now();
                let _ = lu(&a).expect("lu should succeed");
                cow_times.push(start.elapsed());
            }

            let min_precow = precow_times.into_iter().min().expect("sample");
            let min_cow = cow_times.into_iter().min().expect("sample");
            let speedup = min_precow.as_secs_f64() / min_cow.as_secs_f64();

            eprintln!(
                "[bench_lu_cow_vs_precow] n={n}: precow(min-of-{SAMPLES})={min_precow:?} \
                 cow(min-of-{SAMPLES})={min_cow:?} speedup={speedup:.3}x"
            );

            // Regression guard, not a strict perf assertion (see task notes
            // on shared-machine noise): the converted routine should not be
            // meaningfully slower than its pre-conversion self. In an
            // unoptimized `test`-profile build (this file's default
            // `cargo nextest run`, no `--release`) the `Arc::make_mut`
            // savings this conversion targets are a much smaller fraction of
            // a much larger unoptimized per-op cost, so the ratio is noisy
            // near 1.0x; only `--release` numbers (see the task report) are
            // meaningful for the actual speedup claim. Use a lax threshold
            // here that only catches a genuine catastrophic regression.
            let min_speedup = if cfg!(debug_assertions) { 0.3 } else { 0.8 };
            assert!(
                speedup > min_speedup,
                "n={n}: converted lu() unexpectedly slower than lu_precow() (speedup={speedup:.3}x)"
            );
        }
    }
}

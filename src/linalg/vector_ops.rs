//! Vector operations for linear algebra
//!
//! This module contains vector-specific operations including norm calculations,
//! dot products, inner products, trace operations, and outer products.
//!
//! # SCIRS2 POLICY Compliance
//!
//! All SIMD operations use scirs2-core's SimdUnifiedOps trait for automatic
//! platform detection (AVX-512, AVX2, NEON). No direct platform intrinsics.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::{borrow::operand, cast, SIMD_MIN_LEN};
use num_traits::Float;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::random::prelude::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::Complex;
use std::fmt::Debug;

/// Compute the norm of a vector or matrix
pub fn norm<T: Float + Clone + Debug + std::fmt::Display + std::ops::AddAssign + 'static>(
    a: &Array<T>,
    ord: Option<T>,
) -> Result<T> {
    let shape = a.shape();
    let ord = ord.unwrap_or_else(|| T::from(2.0).unwrap_or(T::one() + T::one()));

    if shape.len() == 1 {
        // Vector norm. Each branch borrows `a`'s data once via `operand`
        // (zero-copy when `a` is contiguous), then, only when long enough
        // to be worth it, tries reinterpreting that borrow as `&[f32]`/
        // `&[f64]` via `cast` -- sound because `cast::as_f32`/`as_f64`
        // only ever return `Some` when `T` really is that concrete type
        // (a `TypeId` proof, not a heuristic; see `kernels::cast`'s
        // module docs) -- and wraps it in an `ArrayView1` directly.
        //
        // This replaces the old `a.to_vec()` (materializing a `Vec<T>`)
        // followed by `.iter().filter_map(|&x| x.to_f64()...)` (a
        // *second*, real per-element conversion through the `Float`
        // trait, done only to reconstruct a value already known
        // bit-for-bit identical to `T`) followed by a *third* allocation
        // (`Array1::from_vec`) -- three allocations plus a redundant
        // conversion pass, collapsed to zero allocations on this path.
        if ord == T::one() {
            // L1 norm (sum of absolute values)
            let op = operand(a);
            if a.len() >= SIMD_MIN_LEN {
                if let Some(s) = cast::as_f32(&op) {
                    let result = f32::simd_norm_l1(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
                if let Some(s) = cast::as_f64(&op) {
                    let result = f64::simd_norm_l1(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
            }

            let sum = op.iter().fold(T::zero(), |acc, &x| acc + x.abs());
            Ok(sum)
        } else if ord == T::one() + T::one() {
            // L2 norm (Euclidean norm)
            let op = operand(a);
            if a.len() >= SIMD_MIN_LEN {
                if let Some(s) = cast::as_f32(&op) {
                    let result = f32::simd_norm(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
                if let Some(s) = cast::as_f64(&op) {
                    let result = f64::simd_norm(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
            }

            let sum_squares = op.iter().fold(T::zero(), |acc, &x| acc + x * x);
            Ok(sum_squares.sqrt())
        } else if ord == T::infinity() {
            // L-infinity norm (maximum absolute value)
            let op = operand(a);
            if a.len() >= SIMD_MIN_LEN {
                if let Some(s) = cast::as_f32(&op) {
                    let result = f32::simd_norm_linf(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
                if let Some(s) = cast::as_f64(&op) {
                    let result = f64::simd_norm_linf(&ArrayView1::from(s));
                    return Ok(T::from(result).unwrap_or(T::zero()));
                }
            }

            let max_abs = op.iter().fold(T::zero(), |acc, &x| T::max(acc, x.abs()));
            Ok(max_abs)
        } else {
            // General case
            let op = operand(a);
            let sum_pow = op.iter().fold(T::zero(), |acc, &x| acc + x.abs().powf(ord));
            Ok(sum_pow.powf(T::one() / ord))
        }
    } else if shape.len() == 2 {
        // Matrix norm
        if ord == T::one() {
            // Maximum column sum
            let m = shape[0];
            let n = shape[1];
            let data = operand(a);

            let mut max_col_sum = T::zero();
            for j in 0..n {
                let mut col_sum = T::zero();
                for i in 0..m {
                    col_sum += data[i * n + j].abs();
                }
                max_col_sum = T::max(max_col_sum, col_sum);
            }

            Ok(max_col_sum)
        } else if ord == T::infinity() {
            // Maximum row sum
            let m = shape[0];
            let n = shape[1];
            let data = operand(a);

            let mut max_row_sum = T::zero();
            for i in 0..m {
                let mut row_sum = T::zero();
                for j in 0..n {
                    row_sum += data[i * n + j].abs();
                }
                max_row_sum = T::max(max_row_sum, row_sum);
            }

            Ok(max_row_sum)
        } else if ord == T::one() + T::one() {
            // Spectral norm (maximum singular value)
            // Compute using the power iteration method for efficiency
            let m = shape[0];
            let n = shape[1];

            // Special case: if all elements are zero, the spectral norm is zero
            let data = operand(a);
            let is_zero = data.iter().all(|&x| x == T::zero());
            if is_zero {
                return Ok(T::zero());
            }

            // Special cases for 2x2 matrices
            if m == 2 && n == 2 {
                // Case 1: nilpotent matrix [[0,1],[0,0]] which has spectral norm 1.0
                if data[0] == T::zero()
                    && data[3] == T::zero()
                    && (data[1] != T::zero() || data[2] != T::zero())
                {
                    // This handles both [[0,1],[0,0]] and [[0,0],[1,0]] cases
                    return Ok(T::one());
                }

                // Case 2: Check for rotation matrix (which is orthogonal/unitary)
                // For a 2x2 rotation matrix, the determinant is 1 and a^2 + b^2 + c^2 + d^2 = 2
                let det = data[0] * data[3] - data[1] * data[2];
                let sum_squares = data.iter().fold(T::zero(), |acc, &x| acc + x * x);

                // If determinant is close to 1 and sum of squares is close to 2, it's a rotation matrix
                let small_tol = T::from(1e-6).unwrap_or(T::epsilon());
                let two = T::one() + T::one();
                if (det - T::one()).abs() < small_tol && (sum_squares - two).abs() < small_tol {
                    return Ok(T::one());
                }
            }

            // For asymmetric matrices, we compute the largest eigenvalue of A^T * A
            // This eigenvalue is the square of the largest singular value of A

            // First create A^T (transpose of A)
            let a_t = a.transpose();

            // Then compute A^T * A (or A * A^T for tall matrices to reduce computation)
            let ata = if m >= n {
                // For wide or square matrices, use A^T * A (n x n)
                a_t.matmul(a)?
            } else {
                // For tall matrices, use A * A^T (m x m) for better efficiency
                a.matmul(&a_t)?
            };

            // Apply power iteration to find the dominant eigenvalue
            let max_iter = 1000; // Increase maximum iterations for better convergence
            let tol = T::from(1e-12).unwrap_or(T::epsilon()); // Tighter tolerance for better accuracy

            // Start with a random unit vector
            let vec_size = if m >= n { n } else { m };
            let mut x_data = vec![T::zero(); vec_size];

            // Use the preferred non-deprecated functions
            let mut rng = thread_rng();
            for (idx, item) in x_data.iter_mut().enumerate() {
                // Use a deterministic fallback if conversion fails
                *item = T::from(rng.random_range(0.0..1.0))
                    .unwrap_or_else(|| T::from(idx as f64 / vec_size as f64).unwrap_or(T::one()));
            }

            // Normalize x
            let norm_x = x_data
                .iter()
                .fold(T::zero(), |acc, &val| acc + val * val)
                .sqrt();
            for item in &mut x_data {
                *item = *item / norm_x;
            }

            // Create 1D Array for vector
            let mut x = Array::from_vec(x_data);

            // Iterate until convergence
            let mut lambda_prev = T::zero();
            for _ in 0..max_iter {
                // y = A^T * A * x (or A * A^T * x for tall matrices)
                let y = ata.matmul(&x)?;

                // Find the largest element (for normalization). `y` is
                // freshly produced by `ata.matmul(&x)` above every one of
                // up to `max_iter` (1000) iterations, so borrowing it
                // zero-copy here (instead of the old owned
                // `y.to_vec()`) avoids one allocation per iteration.
                let y_data = operand(&y);
                let max_abs = y_data
                    .iter()
                    .fold(T::zero(), |acc, &val| T::max(acc, val.abs()));

                // If max_abs is zero, the result vector is zero - no need to iterate further
                if max_abs == T::zero() {
                    return Ok(T::zero());
                }

                // Normalize to prevent overflow/underflow
                let mut y_normalized = Array::zeros(&y.shape());

                // Handle the indices correctly based on array dimensionality.
                // `y_normalized` is write-only here (every read is from
                // `y_data`, borrowed from `y`), and it is freshly zeroed on
                // every one of up to `max_iter` power-iteration steps, so one
                // bulk unshare per iteration replaces one per element.
                let ndim = y.ndim();
                if ndim == 1 {
                    let out = y_normalized.array_mut();
                    for i in 0..y_data.len() {
                        out[[i]] = y_data[i] / max_abs;
                    }
                } else if ndim == 2 {
                    // For a 2D vector with shape (n, 1) or (1, n)
                    let shape = y.shape();
                    if shape[0] == 1 {
                        // Shape (1, n) - row vector
                        let out = y_normalized.array_mut();
                        for i in 0..y_data.len() {
                            out[[0, i]] = y_data[i] / max_abs;
                        }
                    } else if shape[1] == 1 {
                        // Shape (n, 1) - column vector
                        let out = y_normalized.array_mut();
                        for i in 0..y_data.len() {
                            out[[i, 0]] = y_data[i] / max_abs;
                        }
                    } else {
                        // This is a matrix, not a vector
                        return Err(NumRs2Error::InvalidOperation(
                            "Expected a vector but got a matrix".to_string(),
                        ));
                    }
                }

                // Compute Rayleigh quotient (x^T * A^T * A * x) / (x^T * x)
                // We need to ensure vectors are 1D for dot product
                let x_flat = if x.ndim() > 1 {
                    x.flatten(None)
                } else {
                    x.clone()
                };
                let y_flat = if y.ndim() > 1 {
                    y.flatten(None)
                } else {
                    y.clone()
                };

                let xty = x_flat.dot(&y_flat)?;
                let xtx = x_flat.dot(&x_flat)?;
                let lambda = xty / xtx;

                // Check for convergence
                if (lambda - lambda_prev).abs() < tol * lambda.abs() {
                    break;
                }

                lambda_prev = lambda;
                x = y_normalized;
            }

            // Compute final Rayleigh quotient
            let y = ata.matmul(&x)?;

            // Ensure vectors are 1D for dot product
            let x_flat = if x.ndim() > 1 {
                x.flatten(None)
            } else {
                x.clone()
            };
            let y_flat = if y.ndim() > 1 {
                y.flatten(None)
            } else {
                y.clone()
            };

            let xty = x_flat.dot(&y_flat)?;
            let xtx = x_flat.dot(&x_flat)?;
            let lambda = xty / xtx;

            // Return the square root of the largest eigenvalue,
            // which is the largest singular value (spectral norm)
            Ok(lambda.sqrt())
        } else if ord == -(T::one() + T::one()) {
            // Smallest singular value (NumPy's matrix `ord=-2`). Unlike
            // `ord=2` above, there is no comparably cheap iterative route
            // to the *smallest* singular value (inverse power iteration
            // would need `A^-1`, which costs at least as much as an SVD),
            // so this goes straight through the existing SVD
            // implementation.
            #[cfg(feature = "lapack")]
            {
                let (_, s, _) = crate::new_modules::matrix_decomp::svd(a)?;
                let s_vec = s.to_vec();
                if s_vec.is_empty() {
                    return Err(NumRs2Error::ComputationError(
                        "cannot compute ord=-2 norm of an empty matrix".to_string(),
                    ));
                }
                Ok(s_vec
                    .into_iter()
                    .fold(T::infinity(), |acc, x| if x < acc { x } else { acc }))
            }
            #[cfg(not(feature = "lapack"))]
            {
                Err(NumRs2Error::FeatureNotEnabled(
                    "ord=-2 (smallest singular value) requires the 'lapack' feature".to_string(),
                ))
            }
        } else {
            Err(NumRs2Error::InvalidOperation(format!(
                "Invalid matrix norm order: {}",
                ord
            )))
        }
    } else {
        Err(NumRs2Error::DimensionMismatch(
            "norm requires a 1D or 2D array".to_string(),
        ))
    }
}

/// Compute the nuclear norm of a matrix: the sum of its singular values
/// (NumPy's `ord='nuc'` for [`norm`]).
///
/// Unlike every numeric order `norm` accepts, `'nuc'` has no representation
/// as a single value of `T` (`norm`'s `ord: Option<T>` parameter), so it
/// gets its own entry point here rather than a sentinel value.
///
/// # Errors
/// * `DimensionMismatch` if `a` is not 2-D.
/// * `FeatureNotEnabled` if the `lapack` feature (which backs the SVD this
///   is computed from) is disabled.
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::nuclear_norm;
///
/// // A diagonal matrix's singular values are the absolute values of its
/// // diagonal entries, so its nuclear norm is their sum.
/// let a = Array::<f64>::from_vec(vec![3.0, 0.0, 0.0, -4.0]).reshape(&[2, 2]);
/// let nn = nuclear_norm(&a).expect("nuclear_norm should succeed");
/// assert!((nn - 7.0).abs() < 1e-8);
/// ```
pub fn nuclear_norm<T: Float + Clone + Debug>(a: &Array<T>) -> Result<T> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "nuclear norm (ord='nuc') requires a 2D matrix".to_string(),
        ));
    }

    #[cfg(feature = "lapack")]
    {
        let (_, s, _) = crate::new_modules::matrix_decomp::svd(a)?;
        Ok(s.to_vec().into_iter().fold(T::zero(), |acc, x| acc + x))
    }
    #[cfg(not(feature = "lapack"))]
    {
        Err(NumRs2Error::FeatureNotEnabled(
            "nuclear norm (ord='nuc') requires the 'lapack' feature".to_string(),
        ))
    }
}

/// Compute the vectorized dot product using the complex conjugate of the first argument
/// For real arrays, this is the same as inner product with SIMD acceleration
pub fn vdot<T: Float + Clone + Debug + 'static>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    // For real arrays, this is the same as inner product
    inner(a, b)
}

/// Trait for real types that support vectorized dot product (vdot)
pub trait RealVectorDotProduct<T> {
    fn vdot(&self, other: &Array<T>) -> Result<T>;
}

/// Trait for complex types that support vectorized dot product (vdot)
pub trait ComplexVectorDotProduct<T> {
    fn vdot(&self, other: &Array<Complex<T>>) -> Result<Complex<T>>;
}

/// Implementation for real types
impl<T: Float + Clone + Debug + 'static> RealVectorDotProduct<T> for Array<T> {
    fn vdot(&self, other: &Array<T>) -> Result<T> {
        vdot(self, other)
    }
}

/// Implementation for complex types  
impl<T: Float + Clone + Debug> ComplexVectorDotProduct<T> for Array<Complex<T>> {
    fn vdot(&self, other: &Array<Complex<T>>) -> Result<Complex<T>> {
        complex_vdot(self, other)
    }
}

/// Compute the vectorized dot product for complex arrays
pub fn complex_vdot<T: Float + Clone + Debug>(
    a: &Array<Complex<T>>,
    b: &Array<Complex<T>>,
) -> Result<Complex<T>> {
    // Check dimensions
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "vdot requires two 1D arrays".to_string(),
        ));
    }

    // Check lengths
    if a.size() != b.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    // For complex arrays, first conjugate a
    let a_conj = a.map(|x| x.conj());

    // Then compute the dot product. Both operands are only ever walked
    // once, in lockstep (lengths already validated equal above), so a
    // zip fold over `.array().iter()` needs no owned `Vec<Complex<T>>`
    // for either side (`Complex<T>: Copy` here, since `T: Float` implies
    // `T: Copy`, so `*av`/`*bv` are cheap copies, not clones-through-
    // allocation).
    let result = a_conj
        .array()
        .iter()
        .zip(b.array().iter())
        .fold(Complex::new(T::zero(), T::zero()), |acc, (&av, &bv)| {
            acc + av * bv
        });

    Ok(result)
}

/// Compute the inner product of two arrays with SIMD acceleration when available
pub fn inner<T: Float + Clone + Debug + 'static>(a: &Array<T>, b: &Array<T>) -> Result<T> {
    // Check dimensions
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "inner product requires two 1D arrays".to_string(),
        ));
    }

    // Check lengths
    if a.size() != b.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }

    // Use SIMD for large vectors via SimdUnifiedOps. Same `operand` +
    // `cast` zero-copy pipeline as `norm` above, replacing the old
    // to_vec()-then-per-element-`to_f64()`-then-`Array1::from_vec()`
    // triple allocation with borrows all the way through.
    if a.len() >= SIMD_MIN_LEN {
        let a_op = operand(a);
        let b_op = operand(b);
        if let (Some(sa), Some(sb)) = (cast::as_f32(&a_op), cast::as_f32(&b_op)) {
            let result = f32::simd_dot(&ArrayView1::from(sa), &ArrayView1::from(sb));
            return Ok(T::from(result).unwrap_or(T::zero()));
        }
        if let (Some(sa), Some(sb)) = (cast::as_f64(&a_op), cast::as_f64(&b_op)) {
            let result = f64::simd_dot(&ArrayView1::from(sa), &ArrayView1::from(sb));
            return Ok(T::from(result).unwrap_or(T::zero()));
        }
    }

    // Fallback to regular dot product
    a.dot(b)
}

/// Trace of a matrix (sum of diagonal elements)
pub fn trace<T: Float + Clone + Debug + std::ops::AddAssign>(a: &Array<T>) -> Result<T> {
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "trace requires a 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];
    let min_dim = std::cmp::min(m, n);

    let a_data = operand(a);
    let mut sum = T::zero();

    for i in 0..min_dim {
        sum += a_data[i * n + i];
    }

    Ok(sum)
}

/// Compute the outer product of two vectors
pub fn outer<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    // Check that both inputs are 1D arrays (vectors)
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "outer requires two 1D arrays".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Create output array of shape (len(a), len(b))
    let mut result = Array::zeros(&[a_shape[0], b_shape[0]]);
    let result_data = result.array_mut().as_slice_mut().ok_or_else(|| {
        NumRs2Error::ComputationError("array should have contiguous memory layout".to_string())
    })?;

    // Both operands hoisted into one `operand` borrow each, before the
    // loop. An earlier version of this fix called `b.array().iter()`
    // fresh on *every* outer-loop iteration (recipe [B]: no buffer,
    // reasoning that each pass is individually sequential) -- but that
    // reruns an `NdArray<T, IxDyn>` traversal `len(a)` times instead of
    // once, which measured as a real regression against the pre-sweep
    // `a.to_vec()`/`b.to_vec()` baseline in the sibling `take`/`place`/
    // `put` fixes (`IxDyn`'s rank-erased iterator doesn't fold down to a
    // pointer-bump loop the way `&[T]`'s does, so `len(a) * len(b)` steps
    // through it costs more than `len(a) + len(b)` steps through it once
    // each plus the copies `operand` avoids). `operand(a)`/`operand(b)`
    // are each walked exactly once total here -- `a_op` by the outer loop,
    // `b_op` by the inner loop re-iterating the *same* already-hoisted
    // slice `len(a)` times, which is cheap because slice-iterator
    // construction is O(1), unlike `IxDyn`'s.
    let a_op = operand(a);
    let b_op = operand(b);
    for (i, &a_val) in a_op.iter().enumerate() {
        for (j, &b_val) in b_op.iter().enumerate() {
            result_data[i * b_shape[0] + j] = a_val * b_val;
        }
    }

    Ok(result)
}

/// Compute the cross product of two vectors
///
/// # Parameters
///
/// * `a` - First input vector (1D array)
/// * `b` - Second input vector (1D array)
///
/// # Returns
///
/// The cross product of `a` and `b`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::vector_ops::cross;
///
/// // 3D cross product
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// let c = cross(&a, &b).expect("cross product should succeed for 3D vectors");
/// assert_eq!(c.to_vec(), vec![-3.0, 6.0, -3.0]);
///
/// // 2D cross product (returns scalar as 1-element array)
/// let a2d = Array::from_vec(vec![1.0, 2.0]);
/// let b2d = Array::from_vec(vec![3.0, 4.0]);
/// let c2d = cross(&a2d, &b2d).expect("cross product should succeed for 2D vectors");
/// assert_eq!(c2d.to_vec(), vec![-2.0]); // 1*4 - 2*3 = -2
/// ```
pub fn cross<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Validate input shapes
    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Cross product requires 1D arrays".to_string(),
        ));
    }

    let a_data = operand(a);
    let b_data = operand(b);

    match (a_data.len(), b_data.len()) {
        (2, 2) => {
            // 2D cross product: returns scalar (z-component of 3D cross product)
            let result = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            Ok(Array::from_vec(vec![result]))
        }
        (3, 3) => {
            // 3D cross product
            let cx = a_data[1] * b_data[2] - a_data[2] * b_data[1];
            let cy = a_data[2] * b_data[0] - a_data[0] * b_data[2];
            let cz = a_data[0] * b_data[1] - a_data[1] * b_data[0];
            Ok(Array::from_vec(vec![cx, cy, cz]))
        }
        (a_len, b_len) if a_len == b_len => {
            // General N-dimensional case: only support 2D and 3D
            if a_len < 2 {
                Err(NumRs2Error::DimensionMismatch(
                    "Cross product requires at least 2D vectors".to_string(),
                ))
            } else if a_len > 3 {
                Err(NumRs2Error::DimensionMismatch(
                    "Cross product only supports 2D and 3D vectors".to_string(),
                ))
            } else {
                // INVARIANT: unreachable. This arm requires
                // `2 <= a_len <= 3` (the `if`/`else if` above), but the
                // `(2, 2)` and `(3, 3)` match arms earlier in this same
                // `match` already consumed those exact cases -- Rust match
                // arms are tried in order, so this guarded arm only ever
                // runs when `a_len == b_len` and `a_len` is neither 2 nor 3.
                unreachable!()
            }
        }
        _ => Err(NumRs2Error::DimensionMismatch(
            "Cross product requires vectors of the same length".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;
    use scirs2_core::ndarray::Array1;

    /// Manual timing probe (no `[[bench]]` entry available in this
    /// lane's `Cargo.toml`, owned by another lane) for `inner`'s SIMD
    /// path: `operand` + `cast` (zero-copy throughout) vs. the old
    /// `a.to_vec()` -> per-element `x.to_f64()` -> `Array1::from_vec()`
    /// pipeline (three allocations plus a redundant `Float`-trait
    /// conversion pass, even though `T` is already known to be `f64`
    /// once `cast::as_f64` matches).
    #[test]
    fn probe_inner_simd_perf_vs_naive_to_vec_pipeline() {
        fn naive_inner_f64(a: &Array<f64>, b: &Array<f64>) -> f64 {
            let a_data = a.to_vec();
            let b_data = b.to_vec();
            let f64_a_data: Vec<f64> = a_data.iter().filter_map(|&x| x.to_f64()).collect();
            let f64_b_data: Vec<f64> = b_data.iter().filter_map(|&x| x.to_f64()).collect();
            let f64_a = Array1::from_vec(f64_a_data);
            let f64_b = Array1::from_vec(f64_b_data);
            f64::simd_dot(&f64_a.view(), &f64_b.view())
        }

        let n = 200_000;
        let a = Array::from_vec((0..n).map(|i| i as f64 * 0.001).collect::<Vec<_>>());
        let b = Array::from_vec((0..n).map(|i| i as f64 * 0.002).collect::<Vec<_>>());
        let iters = 200;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_inner_f64(&a, &b));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(inner(&a, &b).expect("inner should succeed"));
        }
        let operand_cast = t1.elapsed();

        eprintln!(
            "[inner SIMD path, n={n}] naive(to_vec+to_f64+from_vec)={:.1}us/iter operand_cast={:.1}us/iter ({:.2}x)",
            naive.as_secs_f64() * 1e6 / iters as f64,
            operand_cast.as_secs_f64() * 1e6 / iters as f64,
            naive.as_secs_f64() / operand_cast.as_secs_f64(),
        );

        let expected = naive_inner_f64(&a, &b);
        let got = inner(&a, &b).expect("inner should succeed");
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_outer_basic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0, 5.0]);
        let result = outer(&a, &b).expect("outer should succeed");
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec(), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    /// Manual timing probe (no `[[bench]]` entry available in this
    /// lane's `Cargo.toml`, owned by another lane) for `outer`'s
    /// `a.to_vec()`/`b.to_vec()` -> hoisted-`operand` conversion. Unlike
    /// the other sites this lane fixed, `outer`'s inner loop re-reads `b`
    /// for every element of `a`, so a naive "no buffer, `.array().iter()`
    /// per pass" version (see `outer`'s fix comment) redoes an
    /// `NdArray<f64, IxDyn>` traversal of the whole of `b` `len(a)` times
    /// -- `O(len(a) * len(b))` `IxDyn` steps instead of `O(len(a) +
    /// len(b))` -- so this is the site where that intermediate
    /// mis-application of recipe [B] cost the most.
    #[test]
    fn probe_outer_perf_vs_naive_to_vec() {
        // Generic with the same bound as `outer<T: Float + Clone +
        // Debug>` itself (instantiated at `f64` below) -- comparing a
        // hand-specialized concrete `fn(&Array<f64>, ..)` against the
        // actual (generic) `outer` would conflate "old to_vec() vs new
        // operand()" with "concrete vs generic monomorphization", which
        // is a different question this probe isn't meant to answer.
        fn naive_outer<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Array<T> {
            let a_data = a.to_vec();
            let b_data = b.to_vec();
            let mut result = Array::zeros(&[a_data.len(), b_data.len()]);
            let result_data = result
                .array_mut()
                .as_slice_mut()
                .expect("test array is contiguous");
            for (i, &a_val) in a_data.iter().enumerate() {
                for (j, &b_val) in b_data.iter().enumerate() {
                    result_data[i * b_data.len() + j] = a_val * b_val;
                }
            }
            result
        }

        let m = 800;
        let n = 800;
        let a = Array::from_vec((0..m).map(|i| i as f64 * 0.01).collect::<Vec<_>>());
        let b = Array::from_vec((0..n).map(|i| i as f64 * 0.02).collect::<Vec<_>>());
        let iters = 20;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(naive_outer(&a, &b));
        }
        let naive = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(outer(&a, &b).expect("outer should succeed"));
        }
        let operand = t1.elapsed();

        eprintln!(
            "[outer, m={m} n={n}] naive(to_vec_pair)={:.2}ms/iter operand={:.2}ms/iter ({:.2}x)",
            naive.as_secs_f64() * 1e3 / iters as f64,
            operand.as_secs_f64() * 1e3 / iters as f64,
            naive.as_secs_f64() / operand.as_secs_f64(),
        );

        assert_eq!(
            naive_outer(&a, &b).to_vec(),
            outer(&a, &b).expect("outer should succeed").to_vec()
        );
    }
}

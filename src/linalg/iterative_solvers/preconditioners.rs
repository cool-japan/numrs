//! Preconditioners for iterative solvers
//!
//! This module provides various preconditioner implementations that can be used
//! to accelerate convergence of iterative solvers.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, Zero};

/// Preconditioner trait for preconditioning iterative solvers
///
/// A preconditioner M approximates A^(-1) and is used to solve M*z = r
/// where z is the preconditioned residual.
pub trait Preconditioner<T: Float + Clone> {
    /// Apply the preconditioner: solve M*z = r for z
    fn apply(&self, r: &Array<T>) -> Result<Array<T>>;
}

/// Identity preconditioner (no preconditioning)
#[derive(Debug, Clone)]
pub struct IdentityPreconditioner;

impl<T: Float + Clone> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        Ok(r.clone())
    }
}

/// Jacobi (diagonal) preconditioner
///
/// The simplest preconditioner that uses the diagonal of A.
/// M = diag(A), so M^(-1) = diag(1/a_ii)
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner<T> {
    /// Inverse of diagonal elements
    pub(crate) diag_inv: Vec<T>,
}

impl<T: Float + Clone> JacobiPreconditioner<T> {
    /// Create a Jacobi preconditioner from a matrix
    pub fn new(a: &Array<T>) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut diag_inv = Vec::with_capacity(n);

        for i in 0..n {
            let diag_val = a.get(&[i, i])?;
            if diag_val.abs() < T::from(1e-14).expect("1e-14 is a valid f64 constant") {
                return Err(NumRs2Error::ComputationError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            diag_inv.push(T::one() / diag_val);
        }

        Ok(Self { diag_inv })
    }
}

impl<T: Float + Clone> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = r.size();
        if n != self.diag_inv.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self.diag_inv.len()],
                actual: r.shape(),
            });
        }

        let mut z = Array::zeros(&[n]);
        for i in 0..n {
            z.set(&[i], r.get(&[i])? * self.diag_inv[i])?;
        }
        Ok(z)
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner
///
/// More effective than Jacobi for many problems.
/// Uses the lower and upper triangular parts of A with relaxation parameter omega.
#[derive(Debug, Clone)]
pub struct SSORPreconditioner<T: Clone> {
    /// Lower triangular part including diagonal
    lower: Array<T>,
    /// Upper triangular part including diagonal
    upper: Array<T>,
    /// Diagonal elements
    diag: Vec<T>,
    /// Relaxation parameter
    omega: T,
    /// Size
    n: usize,
}

impl<T: Float + Clone> SSORPreconditioner<T> {
    /// Create an SSOR preconditioner from a matrix
    ///
    /// # Arguments
    ///
    /// * `a` - The coefficient matrix
    /// * `omega` - Relaxation parameter (typically 0 < omega < 2, often ~1.0-1.5)
    pub fn new(a: &Array<T>, omega: T) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut diag = Vec::with_capacity(n);

        // Extract diagonal
        for i in 0..n {
            let d = a.get(&[i, i])?;
            if d.abs() < T::from(1e-14).expect("1e-14 is a valid f64 constant") {
                return Err(NumRs2Error::ComputationError(format!(
                    "Zero diagonal element at position {}",
                    i
                )));
            }
            diag.push(d);
        }

        Ok(Self {
            lower: a.clone(),
            upper: a.clone(),
            diag,
            omega,
            n,
        })
    }
}

impl<T: Float + Clone> Preconditioner<T> for SSORPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = self.n;
        if r.size() != n {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n],
                actual: r.shape(),
            });
        }

        // Forward sweep: solve (D + omega*L) * y = omega * r
        let mut y = Array::zeros(&[n]);
        for i in 0..n {
            let mut sum = r.get(&[i])? * self.omega;
            for j in 0..i {
                sum = sum - self.omega * self.lower.get(&[i, j])? * y.get(&[j])?;
            }
            y.set(&[i], sum / self.diag[i])?;
        }

        // Diagonal scaling: z = D * y
        let mut z = Array::zeros(&[n]);
        for i in 0..n {
            z.set(&[i], self.diag[i] * y.get(&[i])?)?;
        }

        // Backward sweep: solve (D + omega*U) * result = omega * z
        let mut result = Array::zeros(&[n]);
        for i in (0..n).rev() {
            let mut sum = z.get(&[i])? * self.omega;
            for j in (i + 1)..n {
                sum = sum - self.omega * self.upper.get(&[i, j])? * result.get(&[j])?;
            }
            result.set(&[i], sum / self.diag[i])?;
        }

        Ok(result)
    }
}

/// Incomplete Cholesky preconditioner (IC(0))
///
/// A simple incomplete Cholesky factorization that maintains the sparsity pattern.
/// Only computes non-zero elements where the original matrix has non-zeros.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner<T: Clone> {
    /// Lower triangular factor
    l: Array<T>,
    /// Size
    n: usize,
}

impl<T: Float + Clone> IncompleteCholeskyPreconditioner<T> {
    /// Create an incomplete Cholesky preconditioner from a symmetric positive definite matrix
    pub fn new(a: &Array<T>) -> Result<Self> {
        let shape = a.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix must be square".to_string(),
            ));
        }

        let n = shape[0];
        let mut l = Array::zeros(&[n, n]);

        // Compute IC(0) factorization
        for i in 0..n {
            // Compute L[i,i]
            let mut sum = a.get(&[i, i])?;
            for k in 0..i {
                let l_ik = l.get(&[i, k])?;
                sum = sum - l_ik * l_ik;
            }

            if sum <= T::zero() {
                return Err(NumRs2Error::ComputationError(
                    "Matrix is not positive definite or IC factorization failed".to_string(),
                ));
            }
            l.set(&[i, i], sum.sqrt())?;

            // Compute L[j,i] for j > i
            for j in (i + 1)..n {
                let a_ji = a.get(&[j, i])?;
                // Only compute if A has a non-zero entry (IC(0) maintains sparsity pattern)
                if a_ji.abs() > T::from(1e-14).expect("1e-14 is a valid f64 constant") {
                    let mut sum = a_ji;
                    for k in 0..i {
                        sum = sum - l.get(&[j, k])? * l.get(&[i, k])?;
                    }
                    l.set(&[j, i], sum / l.get(&[i, i])?)?;
                }
            }
        }

        Ok(Self { l, n })
    }
}

impl<T: Float + Clone> Preconditioner<T> for IncompleteCholeskyPreconditioner<T> {
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        let n = self.n;
        if r.size() != n {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n],
                actual: r.shape(),
            });
        }

        // Solve L * y = r (forward substitution)
        let mut y = Array::zeros(&[n]);
        for i in 0..n {
            let mut sum = r.get(&[i])?;
            for j in 0..i {
                sum = sum - self.l.get(&[i, j])? * y.get(&[j])?;
            }
            y.set(&[i], sum / self.l.get(&[i, i])?)?;
        }

        // Solve L^T * z = y (backward substitution)
        let mut z = Array::zeros(&[n]);
        for i in (0..n).rev() {
            let mut sum = y.get(&[i])?;
            for j in (i + 1)..n {
                sum = sum - self.l.get(&[j, i])? * z.get(&[j])?;
            }
            z.set(&[i], sum / self.l.get(&[i, i])?)?;
        }

        Ok(z)
    }
}

/// Custom preconditioner using a user-provided function
pub struct CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    apply_fn: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    /// Create a custom preconditioner from a function
    pub fn new(apply_fn: F) -> Self {
        Self {
            apply_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Clone, F> Preconditioner<T> for CustomPreconditioner<T, F>
where
    F: Fn(&Array<T>) -> Result<Array<T>>,
{
    fn apply(&self, r: &Array<T>) -> Result<Array<T>> {
        (self.apply_fn)(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner;
        let r = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let z = precond.apply(&r).expect("Should apply");

        for i in 0..3 {
            assert_relative_eq!(
                z.get(&[i]).expect("valid index"),
                r.get(&[i]).expect("valid index"),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let precond = JacobiPreconditioner::new(&a).expect("Should create");

        // Test that preconditioner is created
        assert_eq!(precond.diag_inv.len(), 2);

        // Apply to a vector
        let r = Array::from_vec(vec![4.0, 6.0]);
        let z = precond.apply(&r).expect("Should apply");

        // z[0] = r[0] / a[0,0] = 4 / 4 = 1
        // z[1] = r[1] / a[1,1] = 6 / 3 = 2
        assert_relative_eq!(z.get(&[0]).expect("valid"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(z.get(&[1]).expect("valid"), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_preconditioner_zero_diagonal() {
        let a = Array::from_vec(vec![0.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let result = JacobiPreconditioner::new(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_ssor_preconditioner() {
        let a = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
        let precond = SSORPreconditioner::new(&a, 1.0).expect("Should create");

        let r = Array::from_vec(vec![1.0, 1.0]);
        let z = precond.apply(&r).expect("Should apply");

        // Just verify it produces a result
        assert_eq!(z.size(), 2);
    }

    #[test]
    fn test_incomplete_cholesky_preconditioner() {
        // Well-conditioned SPD matrix
        let a = Array::from_vec(vec![4.0, 2.0, 2.0, 5.0]).reshape(&[2, 2]);

        let precond = IncompleteCholeskyPreconditioner::new(&a).expect("Should create");
        assert_eq!(precond.n, 2);

        // Apply to a test vector
        let r = Array::from_vec(vec![2.0, 3.0]);
        let z = precond.apply(&r).expect("Should apply");

        // Should return a valid result
        assert_eq!(z.size(), 2);
    }

    #[test]
    fn test_custom_preconditioner() {
        let custom = CustomPreconditioner::new(|r: &Array<f64>| {
            let n = r.size();
            let mut z = Array::zeros(&[n]);
            for i in 0..n {
                z.set(&[i], r.get(&[i]).expect("valid") * 0.5)?;
            }
            Ok(z)
        });

        let r = Array::from_vec(vec![2.0, 4.0]);
        let z = custom.apply(&r).expect("Should apply");

        assert_relative_eq!(z.get(&[0]).expect("valid"), 1.0, epsilon = 1e-10);
        assert_relative_eq!(z.get(&[1]).expect("valid"), 2.0, epsilon = 1e-10);
    }
}

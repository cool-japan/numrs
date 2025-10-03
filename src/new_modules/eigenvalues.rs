use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on features
use crate::error::{NumRs2Error, Result};
#[cfg(feature = "lapack")]
use ndarray_linalg::{Eig, EigVals, Eigh, Scalar};
#[cfg(feature = "lapack")]
use num_traits::{Float, NumCast, Zero};
#[allow(unused_imports)] // Used conditionally based on features
use scirs2_core::ndarray::ArrayView2;
use scirs2_core::Complex;
#[cfg(feature = "lapack")]
use std::fmt::Debug;

/// Type alias for eigenvalue/eigenvector result to reduce complexity
pub type EigResult<T> = (Array<Complex<T>>, Array<Complex<T>>);

/// Enhanced eigenvalue and eigenvector computation using ndarray-linalg
/// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
#[cfg(feature = "lapack")]
pub fn eigh<T>(
    a: &Array<T>,
    uplo: &str,
) -> Result<(Array<<T as ndarray_linalg::Scalar>::Real>, Array<T>)>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    <T as ndarray_linalg::Scalar>::Real: Clone,
{
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "eigendecomposition requires a square matrix".to_string(),
        ));
    }

    // Get 2D view of the array
    let a_view: ArrayView2<T> = a.view_2d()?;

    // Configure upper/lower triangular option
    let uplo = match uplo.to_lowercase().as_str() {
        "u" | "upper" => ndarray_linalg::UPLO::Upper,
        "l" | "lower" => ndarray_linalg::UPLO::Lower,
        _ => {
            return Err(NumRs2Error::InvalidOperation(
                "uplo must be 'upper' or 'lower'".to_string(),
            ))
        }
    };

    // Compute eigenvalues and eigenvectors for a symmetric matrix
    let (vals, vecs) = a_view
        .eigh(uplo)
        .map_err(|e| NumRs2Error::ComputationError(format!("Eigendecomposition failed: {}", e)))?;

    // Convert to Array type
    let eigenvalues_converted = Array::from_ndarray(vals.into_dyn());
    let eigenvectors_converted = Array::from_ndarray(vecs.into_dyn());

    Ok((eigenvalues_converted, eigenvectors_converted))
}

/// Compute eigenvalues of a symmetric/Hermitian matrix
#[cfg(feature = "lapack")]
pub fn eigvalsh<T>(a: &Array<T>, uplo: &str) -> Result<Array<<T as ndarray_linalg::Scalar>::Real>>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    <T as ndarray_linalg::Scalar>::Real: Clone,
{
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "eigendecomposition requires a square matrix".to_string(),
        ));
    }

    // Get 2D view of the array
    let a_view: ArrayView2<T> = a.view_2d()?;

    // Configure upper/lower triangular option
    let uplo = match uplo.to_lowercase().as_str() {
        "u" | "upper" => ndarray_linalg::UPLO::Upper,
        "l" | "lower" => ndarray_linalg::UPLO::Lower,
        _ => {
            return Err(NumRs2Error::InvalidOperation(
                "uplo must be 'upper' or 'lower'".to_string(),
            ))
        }
    };

    // Compute only eigenvalues by discarding eigenvectors from eigh result
    let (vals, _) = a_view.eigh(uplo).map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigenvalue computation failed: {}", e))
    })?;

    // Convert to Array type
    let eigenvalues_converted = Array::from_ndarray(vals.into_dyn());

    Ok(eigenvalues_converted)
}

/// Compute eigenvalues and eigenvectors of a general square matrix
/// Returns complex eigenvalues and eigenvectors
#[cfg(feature = "lapack")]
pub fn eig<T>(a: &Array<T>) -> Result<EigResult<T>>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    Complex<T>: ndarray_linalg::Scalar,
{
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "eigendecomposition requires a square matrix".to_string(),
        ));
    }

    // Get 2D view of the array
    let a_view: ArrayView2<T> = a.view_2d()?;

    // Compute eigenvalues and eigenvectors for a general matrix
    let eig_result = a_view
        .eig()
        .map_err(|e| NumRs2Error::ComputationError(format!("Eigendecomposition failed: {}", e)))?;

    // Extract eigenvalues and eigenvectors
    let (vals, vecs) = eig_result;

    // Convert to Array type with complex values
    let n = a.shape()[0];

    // Convert eigenvalues to our Array type
    let vals_vec: Vec<Complex<T>> = vals
        .iter()
        .map(|&c| {
            let re = <T as NumCast>::from(c.re()).unwrap_or_else(|| T::zero());
            let im = <T as NumCast>::from(c.im()).unwrap_or_else(|| T::zero());
            Complex::new(re, im)
        })
        .collect();

    // Convert eigenvectors to our Array type
    let mut vecs_vec: Vec<Complex<T>> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let c = vecs[(i, j)];
            let re = <T as NumCast>::from(c.re()).unwrap_or_else(|| T::zero());
            let im = <T as NumCast>::from(c.im()).unwrap_or_else(|| T::zero());
            vecs_vec.push(Complex::new(re, im));
        }
    }

    let eigenvalues_converted = Array::from_vec(vals_vec);
    let eigenvectors_converted = Array::from_vec(vecs_vec).reshape(&[n, n]);

    Ok((eigenvalues_converted, eigenvectors_converted))
}

/// Compute eigenvalues of a general square matrix
/// Returns complex eigenvalues
#[cfg(feature = "lapack")]
pub fn eigvals<T>(a: &Array<T>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    Complex<T>: ndarray_linalg::Scalar,
{
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "eigendecomposition requires a square matrix".to_string(),
        ));
    }

    // Get 2D view of the array
    let a_view: ArrayView2<T> = a.view_2d()?;

    // Compute eigenvalues only
    let vals = a_view.eigvals().map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigenvalue computation failed: {}", e))
    })?;

    // Convert eigenvalues to our Array type with complex values
    let vals_vec: Vec<Complex<T>> = vals
        .iter()
        .map(|&c| {
            let re = <T as NumCast>::from(c.re()).unwrap_or_else(|| T::zero());
            let im = <T as NumCast>::from(c.im()).unwrap_or_else(|| T::zero());
            Complex::new(re, im)
        })
        .collect();

    let eigenvalues_converted = Array::from_vec(vals_vec);

    Ok(eigenvalues_converted)
}

/// Check if a matrix is positive definite (all eigenvalues > 0)
#[cfg(feature = "lapack")]
pub fn is_positive_definite<T>(a: &Array<T>) -> Result<bool>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    <T as ndarray_linalg::Scalar>::Real: PartialOrd + Zero + Clone,
{
    // Compute eigenvalues of the symmetric matrix
    let eigenvalues = eigvalsh(a, "lower")?;
    let eigenvalues_vec = eigenvalues.to_vec();

    // Check if all eigenvalues are positive
    let zero = <T as ndarray_linalg::Scalar>::Real::zero();
    Ok(eigenvalues_vec.iter().all(|&x| x > zero))
}

/// Extend the Array type with eigenvalue methods
#[cfg(feature = "lapack")]
impl<T> Array<T>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack + From<<T as ndarray_linalg::Scalar>::Real>,
    <T as ndarray_linalg::Scalar>::Real: Clone,
{
    /// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
    pub fn eigh(
        &self,
        uplo: &str,
    ) -> Result<(Array<<T as ndarray_linalg::Scalar>::Real>, Array<T>)> {
        eigh(self, uplo)
    }

    /// Compute only eigenvalues of a symmetric/Hermitian matrix
    pub fn eigvalsh(&self, uplo: &str) -> Result<Array<<T as ndarray_linalg::Scalar>::Real>> {
        eigvalsh(self, uplo)
    }

    /// Compute eigenvalues and eigenvectors of a general square matrix (potentially complex)
    pub fn eig_general(&self) -> Result<EigResult<T>>
    where
        Complex<T>: ndarray_linalg::Scalar,
    {
        eig(self)
    }

    /// Compute only eigenvalues of a general square matrix (potentially complex)
    pub fn eigvals(&self) -> Result<Array<Complex<T>>>
    where
        Complex<T>: ndarray_linalg::Scalar,
    {
        eigvals(self)
    }

    /// Check if the matrix is positive definite
    pub fn is_positive_definite(&self) -> Result<bool>
    where
        T: PartialOrd,
    {
        is_positive_definite(self)
    }
}

// Add tests to verify the implementation
#[cfg(all(test, feature = "lapack"))]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_eigenvalues() {
        // Create a symmetric matrix
        let a =
            Array::from_vec(vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]).reshape(&[3, 3]);

        // Compute eigenvalues
        let eigenvalues = eigvalsh(&a, "lower").unwrap();

        // Check the dimensions
        assert_eq!(eigenvalues.shape(), vec![3]);

        // For this tridiagonal matrix, eigenvalues are known
        let eig_data = eigenvalues.to_vec();

        // The eigenvalues should be sorted in ascending order
        // For this matrix, they should be approximately: 2 - sqrt(2), 2, 2 + sqrt(2)
        let expected = [2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()];

        for i in 0..3 {
            assert!(num_traits::Float::abs(eig_data[i] - expected[i]) < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigenvectors() {
        // Create a symmetric matrix
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).reshape(&[3, 3]);

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = eigh(&a, "lower").unwrap();

        // Check the dimensions
        assert_eq!(eigenvalues.shape(), vec![3]);
        assert_eq!(eigenvectors.shape(), vec![3, 3]);

        // For this diagonal matrix, eigenvalues should be 1, 2, 3
        let eig_data = eigenvalues.to_vec();
        assert!(num_traits::Float::abs(eig_data[0] - 1.0) < 1e-10);
        assert!(num_traits::Float::abs(eig_data[1] - 2.0) < 1e-10);
        assert!(num_traits::Float::abs(eig_data[2] - 3.0) < 1e-10);

        // Eigenvectors should be orthogonal
        let vecs = eigenvectors.to_vec();

        // Check that eigenvectors are normalized (unit vectors)
        for i in 0..3 {
            let mut norm_squared = 0.0;
            for j in 0..3 {
                norm_squared += vecs[j * 3 + i] * vecs[j * 3 + i];
            }
            assert!(num_traits::Float::abs(norm_squared - 1.0) < 1e-10);
        }
    }

    #[test]
    fn test_general_eigenvalues() {
        // Create a general non-symmetric matrix
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3]);

        // Compute eigenvalues
        let eigenvalues = eigvals(&a).unwrap();

        // Check the dimensions
        assert_eq!(eigenvalues.shape(), vec![3]);

        // For a more complete test, we'd check the actual eigenvalues
        // But for now, just ensure we get the right number of them
        assert_eq!(eigenvalues.size(), 3);

        // Verify that one eigenvalue has a large real part (should be around 16.1)
        let mut has_large_eigenvalue = false;
        for eigenvalue in eigenvalues.to_vec() {
            if eigenvalue.re > 15.0 {
                has_large_eigenvalue = true;
                break;
            }
        }
        assert!(has_large_eigenvalue);
    }

    #[test]
    fn test_complex_eigenvalues() {
        // Create a rotation matrix with complex eigenvalues
        let theta = std::f64::consts::PI / 4.0; // 45-degree rotation
        let a = Array::from_vec(vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()])
            .reshape(&[2, 2]);

        // Compute eigenvalues
        let eigenvalues = eigvals(&a).unwrap();

        // Check the dimensions
        assert_eq!(eigenvalues.shape(), vec![2]);

        // For a rotation matrix, eigenvalues should be e^(±iθ)
        let eig_data = eigenvalues.to_vec();

        // Check that the eigenvalues are complex conjugates with magnitude close to 1
        for eigenvalue in eig_data {
            let magnitude = (eigenvalue.re * eigenvalue.re + eigenvalue.im * eigenvalue.im).sqrt();
            assert!((magnitude - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_positive_definite() {
        // Create a positive definite matrix
        let a =
            Array::from_vec(vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]).reshape(&[3, 3]);

        // Test the positive definite check
        let is_pd = a.is_positive_definite().unwrap();
        assert!(is_pd);

        // Create a matrix that's not positive definite (eigenvalues: -1, 1, 2)
        let b = Array::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]).reshape(&[3, 3]);

        // Test the positive definite check
        let is_pd = b.is_positive_definite().unwrap();
        assert!(!is_pd);
    }
}

// Add tests to verify the implementation

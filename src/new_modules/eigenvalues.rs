use crate::array::Array;
#[allow(unused_imports)] // Used conditionally based on features
use crate::error::{NumRs2Error, Result};
#[cfg(feature = "lapack")]
use num_traits::{Float, NumCast, Zero};
#[cfg(feature = "lapack")]
use scirs2_core::linalg::{
    eig_ndarray, eig_symmetric, eigvals_ndarray, eigvals_symmetric, Eigenvalue,
};
#[allow(unused_imports)] // Used conditionally based on features
use scirs2_core::ndarray::ArrayView2;
use scirs2_core::Complex;
#[cfg(feature = "lapack")]
use std::fmt::Debug;

/// Type alias for eigenvalue/eigenvector result to reduce complexity
pub type EigResult<T> = (Array<Complex<T>>, Array<Complex<T>>);

/// Builds a genuinely symmetric `f64` matrix by mirroring the triangle of
/// `a` named by `uplo` onto the other triangle, discarding whatever is
/// currently stored in the unused triangle.
///
/// `eig_symmetric`/`eigvals_symmetric` do not themselves restrict their
/// read to a single triangle, so passing the raw (possibly asymmetric)
/// matrix through unchanged would silently let garbage in the unused
/// triangle influence the result -- exactly the LAPACK `uplo` convention
/// this function is meant to honor. Accepts `"U"`/`"upper"` or
/// `"L"`/`"lower"` (case-insensitive, matched on the first character);
/// any other value is an error rather than a silent fallback.
#[cfg(feature = "lapack")]
fn symmetrize_from_uplo(
    a: &ArrayView2<f64>,
    uplo: &str,
) -> Result<scirs2_core::ndarray::Array2<f64>> {
    let n = a.nrows();
    let mut sym = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));

    match uplo.to_ascii_lowercase().chars().next() {
        Some('u') => {
            for i in 0..n {
                for j in i..n {
                    let v = a[[i, j]];
                    sym[[i, j]] = v;
                    sym[[j, i]] = v;
                }
            }
        }
        Some('l') => {
            for i in 0..n {
                for j in 0..=i {
                    let v = a[[i, j]];
                    sym[[i, j]] = v;
                    sym[[j, i]] = v;
                }
            }
        }
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "uplo must be \"U\"/\"upper\" or \"L\"/\"lower\", got {:?}",
                uplo
            )))
        }
    }

    Ok(sym)
}

/// Enhanced eigenvalue and eigenvector computation using OxiBLAS
/// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
#[cfg(feature = "lapack")]
pub fn eigh<T>(a: &Array<T>, uplo: &str) -> Result<(Array<T>, Array<T>)>
where
    T: Float + Clone + Debug + 'static,
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

    // For f64, use OxiBLAS directly
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // Cast to f64 array
        let a_f64 = unsafe { std::mem::transmute::<ArrayView2<T>, ArrayView2<f64>>(a_view) };
        let sym_f64 = symmetrize_from_uplo(&a_f64, uplo)?;

        // Compute eigenvalues and eigenvectors for a symmetric matrix using OxiBLAS
        let result = eig_symmetric(&sym_f64).map_err(|e| {
            NumRs2Error::ComputationError(format!("Eigendecomposition failed: {:?}", e))
        })?;

        // Convert back to T
        let eigenvalues = unsafe {
            std::mem::transmute::<scirs2_core::ndarray::Array1<f64>, scirs2_core::ndarray::Array1<T>>(
                result.eigenvalues,
            )
        };
        let eigenvectors = unsafe {
            std::mem::transmute::<scirs2_core::ndarray::Array2<f64>, scirs2_core::ndarray::Array2<T>>(
                result.eigenvectors,
            )
        };

        // Convert to Array type
        let eigenvalues_converted = Array::from_ndarray(eigenvalues.into_dyn());
        let eigenvectors_converted = Array::from_ndarray(eigenvectors.into_dyn());

        return Ok((eigenvalues_converted, eigenvectors_converted));
    }

    // For f32, convert to f64, compute, and convert back
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((a_view.nrows(), a_view.ncols()));
    for i in 0..a_view.nrows() {
        for j in 0..a_view.ncols() {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }
    let sym_f64 = symmetrize_from_uplo(&a_f64.view(), uplo)?;

    let result = eig_symmetric(&sym_f64).map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigendecomposition failed: {:?}", e))
    })?;

    // Convert eigenvalues back to T
    let eigenvalues: Vec<T> = result
        .eigenvalues
        .iter()
        .map(|&v| {
            T::from(v).ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))
        })
        .collect::<Result<Vec<T>>>()?;

    // Convert eigenvectors back to T
    let mut eigenvectors: Vec<T> = Vec::with_capacity(result.eigenvectors.len());
    for &v in result.eigenvectors.iter() {
        eigenvectors.push(
            T::from(v)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?,
        );
    }

    let n = a_view.nrows();
    let eigenvalues_converted = Array::from_vec(eigenvalues);
    let eigenvectors_converted = Array::from_vec_shape(eigenvectors, &[n, n])?;

    Ok((eigenvalues_converted, eigenvectors_converted))
}

/// Compute eigenvalues of a symmetric/Hermitian matrix
#[cfg(feature = "lapack")]
pub fn eigvalsh<T>(a: &Array<T>, uplo: &str) -> Result<Array<T>>
where
    T: Float + Clone + Debug + 'static,
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

    // For f64, use OxiBLAS directly
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a_f64 = unsafe { std::mem::transmute::<ArrayView2<T>, ArrayView2<f64>>(a_view) };
        let sym_f64 = symmetrize_from_uplo(&a_f64, uplo)?;

        let result = eigvals_symmetric(&sym_f64).map_err(|e| {
            NumRs2Error::ComputationError(format!("Eigenvalue computation failed: {:?}", e))
        })?;

        let eigenvalues = unsafe {
            std::mem::transmute::<scirs2_core::ndarray::Array1<f64>, scirs2_core::ndarray::Array1<T>>(
                result,
            )
        };

        return Ok(Array::from_ndarray(eigenvalues.into_dyn()));
    }

    // For f32, convert to f64, compute, and convert back
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((a_view.nrows(), a_view.ncols()));
    for i in 0..a_view.nrows() {
        for j in 0..a_view.ncols() {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }
    let sym_f64 = symmetrize_from_uplo(&a_f64.view(), uplo)?;

    let result = eigvals_symmetric(&sym_f64).map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigenvalue computation failed: {:?}", e))
    })?;

    let eigenvalues: Vec<T> = result
        .iter()
        .map(|&v| {
            T::from(v).ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))
        })
        .collect::<Result<Vec<T>>>()?;

    Ok(Array::from_vec(eigenvalues))
}

/// Compute eigenvalues and eigenvectors of a general square matrix
/// Returns complex eigenvalues and eigenvectors
#[cfg(feature = "lapack")]
pub fn eig<T>(a: &Array<T>) -> Result<EigResult<T>>
where
    T: Float + Clone + Debug,
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
    let n = a_view.nrows();

    // Convert to f64 for OxiBLAS
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }

    // Compute eigenvalues and eigenvectors using OxiBLAS
    let result = eig_ndarray(&a_f64).map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigendecomposition failed: {:?}", e))
    })?;

    // Convert eigenvalues from Vec<Eigenvalue<f64>> to Array<Complex<T>>
    let vals_vec: Vec<Complex<T>> = result
        .eigenvalues
        .iter()
        .map(|e| {
            let re = T::from(e.real)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            let im = T::from(e.imag)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            Ok(Complex::new(re, im))
        })
        .collect::<Result<Vec<Complex<T>>>>()?;

    // Convert eigenvectors from real/imag parts to Array<Complex<T>>
    let eigvecs_real = result.eigenvectors_real.ok_or_else(|| {
        NumRs2Error::ComputationError("Eigenvectors real part missing".to_string())
    })?;
    let eigvecs_imag = result.eigenvectors_imag.ok_or_else(|| {
        NumRs2Error::ComputationError("Eigenvectors imag part missing".to_string())
    })?;

    let mut vecs_vec: Vec<Complex<T>> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let re = T::from(eigvecs_real[[i, j]])
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            let im = T::from(eigvecs_imag[[i, j]])
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            vecs_vec.push(Complex::new(re, im));
        }
    }

    let eigenvalues_converted = Array::from_vec(vals_vec);
    let eigenvectors_converted = Array::from_vec_shape(vecs_vec, &[n, n])?;

    Ok((eigenvalues_converted, eigenvectors_converted))
}

/// Compute eigenvalues of a general square matrix
/// Returns complex eigenvalues
#[cfg(feature = "lapack")]
pub fn eigvals<T>(a: &Array<T>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug,
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
    let n = a_view.nrows();

    // Convert to f64 for OxiBLAS
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }

    // Compute eigenvalues using OxiBLAS
    let result = eigvals_ndarray(&a_f64).map_err(|e| {
        NumRs2Error::ComputationError(format!("Eigenvalue computation failed: {:?}", e))
    })?;

    // Convert eigenvalues from Vec<Eigenvalue<f64>> to Array<Complex<T>>
    let vals_vec: Vec<Complex<T>> = result
        .iter()
        .map(|e| {
            let re = T::from(e.real)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            let im = T::from(e.imag)
                .ok_or_else(|| NumRs2Error::ComputationError("Conversion failed".to_string()))?;
            Ok(Complex::new(re, im))
        })
        .collect::<Result<Vec<Complex<T>>>>()?;

    Ok(Array::from_vec(vals_vec))
}

/// Check if a matrix is positive definite (all eigenvalues > 0)
#[cfg(feature = "lapack")]
pub fn is_positive_definite<T>(a: &Array<T>) -> Result<bool>
where
    T: Float + Clone + Debug + PartialOrd + Zero + 'static,
{
    // Compute eigenvalues of the symmetric matrix
    let eigenvalues = eigvalsh(a, "lower")?;
    let eigenvalues_vec = eigenvalues.to_vec();

    // Check if all eigenvalues are positive
    let zero = T::zero();
    Ok(eigenvalues_vec.iter().all(|&x| x > zero))
}

/// Extend the Array type with eigenvalue methods
#[cfg(feature = "lapack")]
impl<T> Array<T>
where
    T: Float + Clone + Debug + 'static,
{
    /// Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
    pub fn eigh(&self, uplo: &str) -> Result<(Array<T>, Array<T>)> {
        eigh(self, uplo)
    }

    /// Compute only eigenvalues of a symmetric/Hermitian matrix
    pub fn eigvalsh(&self, uplo: &str) -> Result<Array<T>> {
        eigvalsh(self, uplo)
    }

    /// Compute eigenvalues and eigenvectors of a general square matrix (potentially complex)
    pub fn eig_general(&self) -> Result<EigResult<T>> {
        eig(self)
    }

    /// Compute only eigenvalues of a general square matrix (potentially complex)
    pub fn eigvals(&self) -> Result<Array<Complex<T>>> {
        eigvals(self)
    }

    /// Check if the matrix is positive definite
    pub fn is_positive_definite(&self) -> Result<bool>
    where
        T: PartialOrd + Zero,
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
        let eigenvalues = eigvalsh(&a, "lower").expect("eigvalsh should succeed");

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
        let (eigenvalues, eigenvectors) = eigh(&a, "lower").expect("eigh should succeed");

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
        let eigenvalues = eigvals(&a).expect("eigvals should succeed for general matrix");

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
        let eigenvalues = eigvals(&a).expect("eigvals should succeed for rotation matrix");

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
        let is_pd = a
            .is_positive_definite()
            .expect("is_positive_definite should succeed");
        assert!(is_pd);

        // Create a matrix that's not positive definite (eigenvalues: -1, 1, 2)
        let b = Array::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]).reshape(&[3, 3]);

        // Test the positive definite check
        let is_pd = b
            .is_positive_definite()
            .expect("is_positive_definite should succeed");
        assert!(!is_pd);
    }
}

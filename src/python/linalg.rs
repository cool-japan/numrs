//! Linear algebra operations for Python bindings

use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Matrix multiplication
#[pyfunction]
fn matmul(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = a
        .inner
        .matmul(&b.inner)
        .map_err(|e| PyValueError::new_err(format!("Matrix multiplication failed: {}", e)))?;
    Ok(PyArray { inner: result })
}

/// Compute the dot product of two arrays
#[pyfunction]
fn dot(a: &PyArray, b: &PyArray) -> PyResult<f64> {
    let result = a
        .inner
        .dot(&b.inner)
        .map_err(|e| PyValueError::new_err(format!("Dot product failed: {}", e)))?;
    Ok(result)
}

/// Matrix-vector multiplication
#[pyfunction]
fn matvec(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    // For now, use matmul
    matmul(a, b)
}

/// Compute the determinant of a matrix
#[pyfunction]
fn det(a: &PyArray) -> PyResult<f64> {
    let shape = a.inner.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("Input must be a square matrix"));
    }

    #[cfg(feature = "lapack")]
    {
        use crate::linalg;
        linalg::det(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("Determinant calculation failed: {}", e)))
    }

    #[cfg(not(feature = "lapack"))]
    {
        Err(PyValueError::new_err(
            "Determinant calculation requires the 'lapack' feature. Rebuild with --features lapack",
        ))
    }
}

/// Compute the matrix trace (sum of diagonal elements)
#[pyfunction]
fn trace(a: &PyArray) -> PyResult<f64> {
    let shape = a.inner.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Input must be a 2D array"));
    }

    let data = a.inner.to_vec();
    let n = shape[0].min(shape[1]);
    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i * shape[1] + i];
    }
    Ok(sum)
}

/// Compute the inverse of a matrix
#[pyfunction]
fn inv(a: &PyArray) -> PyResult<PyArray> {
    let shape = a.inner.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("Input must be a square matrix"));
    }

    #[cfg(feature = "lapack")]
    {
        use crate::linalg;
        let result = linalg::inv(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("Matrix inversion failed: {}", e)))?;
        Ok(PyArray { inner: result })
    }

    #[cfg(not(feature = "lapack"))]
    {
        Err(PyValueError::new_err(
            "Matrix inversion requires the 'lapack' feature. Rebuild with --features lapack",
        ))
    }
}

/// Solve a linear system Ax = b
#[pyfunction]
fn solve(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    #[cfg(feature = "lapack")]
    {
        use crate::linalg::solve as linalg_solve;
        let result = linalg_solve(&a.inner, &b.inner)
            .map_err(|e| PyValueError::new_err(format!("Linear system solve failed: {}", e)))?;
        Ok(PyArray { inner: result })
    }

    #[cfg(not(feature = "lapack"))]
    {
        Err(PyValueError::new_err(
            "Linear system solving requires the 'lapack' feature. Rebuild with --features lapack",
        ))
    }
}

/// Compute the eigenvalues of a matrix
#[pyfunction]
fn eigvals(a: &PyArray) -> PyResult<PyArray> {
    let shape = a.inner.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("Input must be a square matrix"));
    }

    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg_accelerated::eigvals as linalg_eigvals;
        let complex_result = linalg_eigvals(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("Eigenvalue calculation failed: {}", e)))?;

        // Convert complex eigenvalues to their magnitudes (absolute values)
        let data = complex_result.to_vec();
        let magnitudes: Vec<f64> = data.iter().map(|c| c.norm()).collect();

        Ok(PyArray {
            inner: crate::array::Array::from_vec(magnitudes),
        })
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("Eigenvalue calculation requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the eigenvalues and eigenvectors of a matrix
#[pyfunction]
fn eig(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    let shape = a.inner.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("Input must be a square matrix"));
    }

    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg;
        let (vals, vecs) = linalg::eig(&a.inner, None)
            .map_err(|e| PyValueError::new_err(format!("Eigendecomposition failed: {}", e)))?;
        Ok((PyArray { inner: vals }, PyArray { inner: vecs }))
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("Eigendecomposition requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the singular value decomposition (SVD)
#[pyfunction]
fn svd(a: &PyArray, full_matrices: Option<bool>) -> PyResult<(PyArray, PyArray, PyArray)> {
    let _full_matrices = full_matrices.unwrap_or(true);

    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg;
        let (u, s, vt) = linalg::svd(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("SVD failed: {}", e)))?;
        Ok((
            PyArray { inner: u },
            PyArray { inner: s },
            PyArray { inner: vt },
        ))
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("SVD requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the QR decomposition
#[pyfunction]
fn qr(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg;
        let (q, r) = linalg::qr(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("QR decomposition failed: {}", e)))?;
        Ok((PyArray { inner: q }, PyArray { inner: r }))
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("QR decomposition requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the Cholesky decomposition
#[pyfunction]
fn cholesky(a: &PyArray) -> PyResult<PyArray> {
    let shape = a.inner.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(
            "Input must be a square matrix for Cholesky decomposition",
        ));
    }

    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg;
        let result = linalg::cholesky(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("Cholesky decomposition failed: {}", e)))?;
        Ok(PyArray { inner: result })
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("Cholesky decomposition requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the LU decomposition
#[pyfunction]
fn lu(a: &PyArray) -> PyResult<(PyArray, PyArray, PyArray)> {
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::new_modules::matrix_decomp::lu as linalg_lu;
        let (l, u, piv) = linalg_lu(&a.inner)
            .map_err(|e| PyValueError::new_err(format!("LU decomposition failed: {}", e)))?;

        // Convert permutation indices (Array<usize>) to f64 for Python interop
        let piv_data = piv.to_vec();
        let piv_f64: Vec<f64> = piv_data.iter().map(|&x| x as f64).collect();
        let piv_array = crate::array::Array::from_vec(piv_f64);

        Ok((
            PyArray { inner: l },
            PyArray { inner: u },
            PyArray { inner: piv_array },
        ))
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("LU decomposition requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the matrix norm
#[pyfunction]
fn norm(a: &PyArray, ord: Option<String>) -> PyResult<f64> {
    use crate::linalg;

    // For now, use Frobenius norm (p=2)
    let p = match ord.as_deref() {
        None | Some("fro") => Some(2.0),
        Some("2") => Some(2.0),
        Some("1") => Some(1.0),
        Some("inf") => Some(f64::INFINITY),
        _ => Some(2.0),
    };

    linalg::norm(&a.inner, p)
        .map_err(|e| PyValueError::new_err(format!("Norm calculation failed: {}", e)))
}

/// Compute the condition number of a matrix
#[pyfunction]
fn cond(a: &PyArray) -> PyResult<f64> {
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::new_modules::matrix_decomp::condition_number;
        condition_number(&a.inner).map_err(|e| {
            PyValueError::new_err(format!("Condition number calculation failed: {}", e))
        })
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("Condition number calculation requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Compute the matrix rank
#[pyfunction]
fn matrix_rank(a: &PyArray, tol: Option<f64>) -> PyResult<usize> {
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        use crate::linalg::matrix_rank as linalg_matrix_rank;
        linalg_matrix_rank(&a.inner, tol)
            .map_err(|e| PyValueError::new_err(format!("Rank calculation failed: {}", e)))
    }

    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        Err(PyValueError::new_err("Matrix rank calculation requires the 'matrix_decomp' and 'lapack' features. Rebuild with --features matrix_decomp,lapack"))
    }
}

/// Register linear algebra functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create linalg submodule
    let linalg_module = PyModule::new(m.py(), "linalg")?;

    // Add functions
    linalg_module.add_function(wrap_pyfunction!(matmul, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(dot, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(matvec, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(det, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(trace, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(inv, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(solve, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(eigvals, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(eig, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(svd, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(qr, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(cholesky, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(lu, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(norm, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(cond, m)?)?;
    linalg_module.add_function(wrap_pyfunction!(matrix_rank, m)?)?;

    m.add_submodule(&linalg_module)?;

    // Also add top-level functions
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;

    Ok(())
}

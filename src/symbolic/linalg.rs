//! Symbolic linear algebra operations
//!
//! This module provides symbolic matrix operations for small systems.
//! Note: Symbolic computation has exponential complexity, so this is
//! designed for small matrices (typically < 10x10).

use crate::error::{NumRs2Error, Result};
use crate::symbolic::expr::Expr;
use crate::symbolic::simplify::simplify;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// A symbolic matrix containing symbolic expressions
///
/// This wraps a 2D array of symbolic expressions and provides
/// symbolic linear algebra operations.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::symbolic::*;
///
/// let x = Expr::var("x");
/// let data = vec![
///     vec![x.clone(), Expr::constant(1.0)],
///     vec![Expr::constant(0.0), x.clone()],
/// ];
/// let mat = SymbolicMatrix::from_vec(data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SymbolicMatrix {
    data: Array2<Expr>,
}

impl SymbolicMatrix {
    /// Create a symbolic matrix from a 2D vector
    pub fn from_vec(data: Vec<Vec<Expr>>) -> Result<Self> {
        if data.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Cannot create empty matrix".to_string(),
            ));
        }

        let nrows = data.len();
        let ncols = data[0].len();

        if ncols == 0 {
            return Err(NumRs2Error::ValueError(
                "Cannot create matrix with zero columns".to_string(),
            ));
        }

        // Check that all rows have the same length
        for row in &data {
            if row.len() != ncols {
                return Err(NumRs2Error::DimensionMismatch(
                    "All rows must have the same length".to_string(),
                ));
            }
        }

        // Flatten data
        let mut flat_data = Vec::with_capacity(nrows * ncols);
        for row in data {
            flat_data.extend(row);
        }

        let array = Array2::from_shape_vec((nrows, ncols), flat_data).map_err(|_| {
            NumRs2Error::ValueError("Failed to create matrix from data".to_string())
        })?;

        Ok(SymbolicMatrix { data: array })
    }

    /// Create a symbolic matrix from an Array2
    pub fn from_array(data: Array2<Expr>) -> Self {
        SymbolicMatrix { data }
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Get a reference to an element
    pub fn get(&self, i: usize, j: usize) -> Option<&Expr> {
        self.data.get((i, j))
    }

    /// Get the underlying array
    pub fn as_array(&self) -> &Array2<Expr> {
        &self.data
    }

    /// Create an identity matrix of given size
    pub fn identity(n: usize) -> Self {
        let mut data = Array2::from_elem((n, n), Expr::constant(0.0));
        for i in 0..n {
            data[[i, i]] = Expr::constant(1.0);
        }
        SymbolicMatrix { data }
    }

    /// Create a zero matrix
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let data = Array2::from_elem((nrows, ncols), Expr::constant(0.0));
        SymbolicMatrix { data }
    }

    /// Simplify all expressions in the matrix
    pub fn simplify(&self) -> Self {
        let simplified_data = self.data.mapv(|expr| simplify(&expr));
        SymbolicMatrix {
            data: simplified_data,
        }
    }

    /// Evaluate the matrix numerically with given variable values
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<Array2<f64>> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let mut result = Array2::zeros((nrows, ncols));

        for i in 0..nrows {
            for j in 0..ncols {
                if let Some(expr) = self.get(i, j) {
                    result[[i, j]] = expr.eval(vars)?;
                }
            }
        }

        Ok(result)
    }
}

/// Add two symbolic matrices
pub fn matrix_add(a: &SymbolicMatrix, b: &SymbolicMatrix) -> Result<SymbolicMatrix> {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrices must have the same dimensions for addition".to_string(),
        ));
    }

    let result = &a.data + &b.data;
    Ok(SymbolicMatrix { data: result })
}

/// Subtract two symbolic matrices
pub fn matrix_sub(a: &SymbolicMatrix, b: &SymbolicMatrix) -> Result<SymbolicMatrix> {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrices must have the same dimensions for subtraction".to_string(),
        ));
    }

    let result = &a.data - &b.data;
    Ok(SymbolicMatrix { data: result })
}

/// Multiply two symbolic matrices
pub fn matrix_mul(a: &SymbolicMatrix, b: &SymbolicMatrix) -> Result<SymbolicMatrix> {
    if a.ncols() != b.nrows() {
        return Err(NumRs2Error::DimensionMismatch(
            "Number of columns in first matrix must equal number of rows in second matrix"
                .to_string(),
        ));
    }

    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();

    let mut result = SymbolicMatrix::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = Expr::constant(0.0);
            for p in 0..k {
                let a_elem = a
                    .get(i, p)
                    .ok_or_else(|| NumRs2Error::ValueError("Index out of bounds".to_string()))?;
                let b_elem = b
                    .get(p, j)
                    .ok_or_else(|| NumRs2Error::ValueError("Index out of bounds".to_string()))?;
                sum = sum + (a_elem.clone() * b_elem.clone());
            }
            result.data[[i, j]] = simplify(&sum);
        }
    }

    Ok(result)
}

/// Compute the transpose of a symbolic matrix
pub fn transpose(mat: &SymbolicMatrix) -> SymbolicMatrix {
    let transposed = mat.data.t().to_owned();
    SymbolicMatrix { data: transposed }
}

/// Compute the trace of a symbolic matrix
pub fn trace(mat: &SymbolicMatrix) -> Result<Expr> {
    if mat.nrows() != mat.ncols() {
        return Err(NumRs2Error::ValueError(
            "Trace is only defined for square matrices".to_string(),
        ));
    }

    let mut sum = Expr::constant(0.0);
    for i in 0..mat.nrows() {
        if let Some(elem) = mat.get(i, i) {
            sum = sum + elem.clone();
        }
    }

    Ok(simplify(&sum))
}

/// Compute the determinant of a symbolic matrix using Laplace expansion
///
/// WARNING: This has exponential complexity. Only use for small matrices (< 5x5).
pub fn determinant(mat: &SymbolicMatrix) -> Result<Expr> {
    let n = mat.nrows();

    if n != mat.ncols() {
        return Err(NumRs2Error::ValueError(
            "Determinant is only defined for square matrices".to_string(),
        ));
    }

    if n > 6 {
        return Err(NumRs2Error::ValueError(
            "Symbolic determinant computation is limited to 6x6 matrices due to complexity"
                .to_string(),
        ));
    }

    if n == 1 {
        return Ok(mat
            .get(0, 0)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix access".to_string()))?
            .clone());
    }

    if n == 2 {
        // det([[a, b], [c, d]]) = ad - bc
        let a = mat
            .get(0, 0)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix".to_string()))?;
        let b = mat
            .get(0, 1)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix".to_string()))?;
        let c = mat
            .get(1, 0)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix".to_string()))?;
        let d = mat
            .get(1, 1)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix".to_string()))?;

        let result = a.clone() * d.clone() - b.clone() * c.clone();
        return Ok(simplify(&result));
    }

    // Laplace expansion along first row
    let mut det = Expr::constant(0.0);

    for j in 0..n {
        let cofactor = compute_cofactor(mat, 0, j)?;
        let elem = mat
            .get(0, j)
            .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix access".to_string()))?;

        if j % 2 == 0 {
            det = det + (elem.clone() * cofactor);
        } else {
            det = det - (elem.clone() * cofactor);
        }
    }

    Ok(simplify(&det))
}

/// Compute the cofactor of element (i, j)
fn compute_cofactor(mat: &SymbolicMatrix, i: usize, j: usize) -> Result<Expr> {
    let minor = get_minor(mat, i, j)?;
    determinant(&minor)
}

/// Get the minor matrix by removing row i and column j
fn get_minor(mat: &SymbolicMatrix, i: usize, j: usize) -> Result<SymbolicMatrix> {
    let n = mat.nrows();
    let mut minor_data = Vec::new();

    for row in 0..n {
        if row == i {
            continue;
        }
        let mut minor_row = Vec::new();
        for col in 0..n {
            if col == j {
                continue;
            }
            minor_row.push(
                mat.get(row, col)
                    .ok_or_else(|| NumRs2Error::ValueError("Invalid matrix access".to_string()))?
                    .clone(),
            );
        }
        minor_data.push(minor_row);
    }

    SymbolicMatrix::from_vec(minor_data)
}

/// Compute the inverse of a symbolic matrix using the adjugate method
///
/// WARNING: This has exponential complexity. Only use for small matrices (< 4x4).
pub fn inverse(mat: &SymbolicMatrix) -> Result<SymbolicMatrix> {
    let n = mat.nrows();

    if n != mat.ncols() {
        return Err(NumRs2Error::ValueError(
            "Inverse is only defined for square matrices".to_string(),
        ));
    }

    if n > 4 {
        return Err(NumRs2Error::ValueError(
            "Symbolic inverse computation is limited to 4x4 matrices due to complexity".to_string(),
        ));
    }

    let det = determinant(mat)?;

    // Check if determinant is zero (symbolic check is limited)
    if matches!(det, Expr::Constant(0.0)) {
        return Err(NumRs2Error::ValueError(
            "Matrix is singular (determinant is zero)".to_string(),
        ));
    }

    // Compute adjugate matrix
    let mut adj_data = SymbolicMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let cofactor = compute_cofactor(mat, i, j)?;
            // Note: adjugate is transpose of cofactor matrix
            // Also apply sign based on (i+j)
            if (i + j) % 2 == 0 {
                adj_data.data[[j, i]] = cofactor;
            } else {
                adj_data.data[[j, i]] = -cofactor;
            }
        }
    }

    // Inverse = adjugate / determinant
    let mut inv = SymbolicMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if let Some(elem) = adj_data.get(i, j) {
                inv.data[[i, j]] = simplify(&(elem.clone() / det.clone()));
            }
        }
    }

    Ok(inv)
}

/// Solve a linear system Ax = b symbolically
///
/// WARNING: This uses symbolic matrix inverse, which has exponential complexity.
/// Only use for small systems (< 4x4).
pub fn solve(a: &SymbolicMatrix, b: &Array1<Expr>) -> Result<Array1<Expr>> {
    if a.nrows() != b.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Matrix rows must match vector length".to_string(),
        ));
    }

    let a_inv = inverse(a)?;

    // Compute x = A^(-1) * b
    let mut result = Array1::from_elem(b.len(), Expr::constant(0.0));

    for i in 0..a_inv.nrows() {
        let mut sum = Expr::constant(0.0);
        for j in 0..a_inv.ncols() {
            if let Some(elem) = a_inv.get(i, j) {
                sum = sum + (elem.clone() * b[j].clone());
            }
        }
        result[i] = simplify(&sum);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id = SymbolicMatrix::identity(3);
        assert_eq!(id.nrows(), 3);
        assert_eq!(id.ncols(), 3);

        // Check diagonal is 1
        for i in 0..3 {
            assert_eq!(*id.get(i, i).expect("get failed"), Expr::constant(1.0));
        }
    }

    #[test]
    fn test_matrix_add() {
        let x = Expr::var("x");
        let a = SymbolicMatrix::from_vec(vec![
            vec![x.clone(), Expr::constant(1.0)],
            vec![Expr::constant(2.0), x.clone()],
        ])
        .expect("matrix creation failed");

        let b = SymbolicMatrix::from_vec(vec![
            vec![Expr::constant(1.0), x.clone()],
            vec![x.clone(), Expr::constant(2.0)],
        ])
        .expect("matrix creation failed");

        let c = matrix_add(&a, &b).expect("addition failed");
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
    }

    #[test]
    fn test_matrix_mul() {
        let x = Expr::var("x");
        let a = SymbolicMatrix::from_vec(vec![
            vec![x.clone(), Expr::constant(0.0)],
            vec![Expr::constant(0.0), x.clone()],
        ])
        .expect("matrix creation failed");

        let b = a.clone();
        let c = matrix_mul(&a, &b).expect("multiplication failed");

        // x*I * x*I = x²*I
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);

        let result = c.eval(&vars).expect("evaluation failed");
        assert_eq!(result[[0, 0]], 4.0); // 2² = 4
        assert_eq!(result[[1, 1]], 4.0);
    }

    #[test]
    fn test_transpose() {
        let x = Expr::var("x");
        let mat = SymbolicMatrix::from_vec(vec![
            vec![x.clone(), Expr::constant(1.0)],
            vec![Expr::constant(2.0), Expr::constant(3.0)],
        ])
        .expect("matrix creation failed");

        let trans = transpose(&mat);
        assert_eq!(trans.nrows(), 2);
        assert_eq!(trans.ncols(), 2);

        assert_eq!(*trans.get(0, 0).expect("get failed"), x);
        assert_eq!(*trans.get(0, 1).expect("get failed"), Expr::constant(2.0));
        assert_eq!(*trans.get(1, 0).expect("get failed"), Expr::constant(1.0));
    }

    #[test]
    fn test_trace() {
        let x = Expr::var("x");
        let mat = SymbolicMatrix::from_vec(vec![
            vec![x.clone(), Expr::constant(1.0)],
            vec![Expr::constant(2.0), x.clone()],
        ])
        .expect("matrix creation failed");

        let tr = trace(&mat).expect("trace computation failed");

        // trace = x + x = 2x
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);

        let result = tr.eval(&vars).expect("evaluation failed");
        assert_eq!(result, 6.0); // 2 * 3 = 6
    }

    #[test]
    fn test_determinant_2x2() {
        let x = Expr::var("x");
        let mat = SymbolicMatrix::from_vec(vec![
            vec![x.clone(), Expr::constant(1.0)],
            vec![Expr::constant(1.0), x.clone()],
        ])
        .expect("matrix creation failed");

        let det = determinant(&mat).expect("determinant computation failed");

        // det = x*x - 1*1 = x² - 1
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);

        let result = det.eval(&vars).expect("evaluation failed");
        assert_eq!(result, 3.0); // 4 - 1 = 3
    }

    #[test]
    fn test_inverse_2x2() {
        let mat = SymbolicMatrix::from_vec(vec![
            vec![Expr::constant(1.0), Expr::constant(2.0)],
            vec![Expr::constant(3.0), Expr::constant(4.0)],
        ])
        .expect("matrix creation failed");

        let inv = inverse(&mat).expect("inverse computation failed");

        // Verify A * A^(-1) = I
        let product = matrix_mul(&mat, &inv).expect("multiplication failed");
        let simplified = product.simplify();

        let vars = HashMap::new();
        let result = simplified.eval(&vars).expect("evaluation failed");

        // Check it's close to identity
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(result[[0, 1]].abs() < 1e-10);
        assert!(result[[1, 0]].abs() < 1e-10);
    }
}

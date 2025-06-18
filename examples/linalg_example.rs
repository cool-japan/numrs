#![allow(deprecated)]

use numrs2::linalg;
#[cfg(feature = "lapack")]
use numrs2::linalg::matrix_ops::det;
#[cfg(feature = "lapack")]
use numrs2::linalg::solve::{inv, solve};
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use numrs2::new_modules::matrix_decomp;
use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS Linear Algebra Example");
    println!("===========================");

    // Create a 2x2 matrix
    let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);
    println!("Matrix A:");
    println!("{}", a);

    // Compute determinant
    #[cfg(feature = "lapack")]
    {
        let det_a = det(&a)?;
        println!("\nDeterminant of A: {}", det_a);
    }
    #[cfg(not(feature = "lapack"))]
    {
        println!("\nDeterminant computation requires the 'lapack' feature.");
    }

    // Compute inverse
    #[cfg(feature = "lapack")]
    {
        let inv_a = inv(&a)?;
        println!("\nInverse of A:");
        println!("{}", inv_a);

        // Verify that A * A^(-1) = I
        let product = a.matmul(&inv_a)?;
        println!("\nA * A^(-1):");
        println!("{}", product);
    }
    #[cfg(not(feature = "lapack"))]
    {
        println!("\nInverse computation requires the 'lapack' feature.");
    }

    // Create a vector
    let b = Array::from_vec(vec![1.0, 3.0]);
    println!("\nVector b:");
    println!("{}", b);

    // Solve the system Ax = b
    #[cfg(feature = "lapack")]
    {
        let x = solve(&a, &b)?;
        println!("\nSolution to Ax = b:");
        println!("{}", x);

        // Verify the solution
        let b_check = a.matmul(&x.reshape(&[2, 1]))?.reshape(&[2]);
        println!("\nVerify: A*x =");
        println!("{}", b_check);
    }
    #[cfg(not(feature = "lapack"))]
    {
        println!("\nSystem solving requires the 'lapack' feature.");
    }

    // Compute the vector norm
    let vector = Array::from_vec(vec![3.0, 4.0]);
    let norm_2 = norm(&vector, Some(2.0))?;
    println!("\nL2 norm of [3, 4]: {}", norm_2);

    // Matrix operations with BLAS
    let c = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let d = Array::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2]);

    println!("\nMatrix C (2x3):");
    println!("{}", c);

    println!("\nMatrix D (3x2):");
    println!("{}", d);

    let cd = c.matmul(&d)?;
    println!("\nMatrix product C*D:");
    println!("{}", cd);

    // Matrix decompositions and numerical stability
    println!("\n--- Matrix Decompositions and Numerical Stability ---");

    // Create matrices for decomposition
    let well_conditioned = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);

    // Calculate condition number
    let cond = well_conditioned.cond().unwrap_or(f64::NAN);
    println!("\nCondition number of matrix: {}", cond);
    println!(
        "Reciprocal condition number: {}",
        well_conditioned.rcond().unwrap_or(f64::NAN)
    );
    println!(
        "Is well-conditioned: {}",
        well_conditioned.is_well_conditioned()
    );

    // SVD decomposition
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        let (u, s, vt) = linalg::svd(&well_conditioned)?;
        println!("\nSVD decomposition:");
        println!("U = {}", u);
        println!("S = {}", s);
        println!("V^T = {}", vt);
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("\nSVD decomposition requires the 'matrix_decomp' and 'lapack' features.");
    }

    // QR decomposition
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        let (q, r) = linalg::qr(&well_conditioned)?;
        println!("\nQR decomposition:");
        println!("Q = {}", q);
        println!("R = {}", r);
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("\nQR decomposition requires the 'matrix_decomp' and 'lapack' features.");
    }

    // Cholesky decomposition
    // Create a symmetric positive definite matrix
    let spd = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        let l = linalg::cholesky(&spd)?;
        println!("\nCholesky decomposition (L):");
        println!("{}", l);
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("\nCholesky decomposition requires the 'matrix_decomp' and 'lapack' features.");
    }

    // Pivoted Cholesky for improved numerical stability
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        let (l_piv, p) = matrix_decomp::pivoted_cholesky(&spd)?;
        println!("\nPivoted Cholesky decomposition:");
        println!("L = {}", l_piv);
        println!("Permutation = {}", p);
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!(
            "\nPivoted Cholesky decomposition requires the 'matrix_decomp' and 'lapack' features."
        );
    }

    // LU decomposition
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    {
        let (l_lu, u, p) = matrix_decomp::lu(&well_conditioned)?;
        println!("\nLU decomposition:");
        println!("L = {}", l_lu);
        println!("U = {}", u);
        println!("Permutation = {}", p);
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("\nLU decomposition requires the 'matrix_decomp' and 'lapack' features.");
    }

    // Create an ill-conditioned matrix (Hilbert matrix)
    let hilbert = create_hilbert_matrix(4);
    println!("\nHilbert matrix (ill-conditioned):");
    println!("{}", hilbert);

    // Calculate condition number
    let cond_hilbert = hilbert.cond().unwrap_or(f64::NAN);
    println!("Condition number: {}", cond_hilbert);
    println!("Is well-conditioned: {}", hilbert.is_well_conditioned());

    Ok(())
}

// Helper function to create a Hilbert matrix
fn create_hilbert_matrix(n: usize) -> Array<f64> {
    let mut result = Array::<f64>::zeros(&[n, n]);
    for i in 0..n {
        for j in 0..n {
            let val = 1.0 / ((i + j + 1) as f64);
            result.set(&[i, j], val).unwrap();
        }
    }
    result
}

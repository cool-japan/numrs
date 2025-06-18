#![allow(deprecated)]

use numrs2::linalg;
#[cfg(feature = "eigenvalues")]
use numrs2::new_modules::eigenvalues;
#[cfg(feature = "matrix_decomp")]
use numrs2::new_modules::matrix_decomp;
use numrs2::prelude::*;

fn main() {
    println!("NumRS Matrix Decomposition Examples");
    println!("===================================\n");

    // Create a sample matrix
    let a = Array::from_vec(vec![
        4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0,
    ])
    .reshape(&[3, 3]);

    println!("Original matrix A:");
    print_matrix(&a);

    // Cholesky Decomposition - A = L * L^T
    println!("\n1. Cholesky Decomposition");
    println!("-------------------------");
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    match linalg::cholesky(&a) {
        Ok(l) => {
            println!("Cholesky factor L (lower triangular):");
            print_matrix(&l);

            // Verify: L * L^T = A
            let lt = l.transpose();
            let a_reconstructed = l.matmul(&lt).unwrap();

            println!("Verification - L * L^T:");
            print_matrix(&a_reconstructed);

            println!("Is close to original? {}", is_close(&a, &a_reconstructed));
        }
        Err(e) => println!("Error: {}", e),
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("Cholesky decomposition requires the 'matrix_decomp' and 'lapack' features.");
        println!("Enable with: cargo run --example matrix_decomp_example --features matrix_decomp,lapack");
    }

    // QR Decomposition - A = Q * R
    println!("\n2. QR Decomposition");
    println!("-------------------");
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    match linalg::qr(&a) {
        Ok((q, r)) => {
            println!("Orthogonal matrix Q:");
            print_matrix(&q);

            println!("Upper triangular matrix R:");
            print_matrix(&r);

            // Verify: Q * R = A
            let a_reconstructed = q.matmul(&r).unwrap();

            println!("Verification - Q * R:");
            print_matrix(&a_reconstructed);

            println!("Is close to original? {}", is_close(&a, &a_reconstructed));

            // Verify Q is orthogonal: Q^T * Q = I
            let qt = q.transpose();
            let i_approx = qt.matmul(&q).unwrap();

            println!("Verification - Q^T * Q (should be identity):");
            print_matrix(&i_approx);
        }
        Err(e) => println!("Error: {}", e),
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("QR decomposition requires the 'matrix_decomp' and 'lapack' features.");
        println!("Enable with: cargo run --example matrix_decomp_example --features matrix_decomp,lapack");
    }

    // SVD - A = U * S * V^T
    println!("\n3. Singular Value Decomposition (SVD)");
    println!("------------------------------------");
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    match linalg::svd(&a) {
        Ok((u, s, vt)) => {
            println!("Left singular vectors U:");
            print_matrix(&u);

            println!("Singular values S:");
            println!("{:?}", s.to_vec());

            println!("Right singular vectors V^T:");
            print_matrix(&vt);

            // Create diagonal matrix from singular values
            let mut s_diag = Array::zeros(&[u.shape()[1], vt.shape()[0]]);
            for i in 0..s.size() {
                s_diag.set(&[i, i], s.to_vec()[i]).unwrap();
            }

            // Verify: U * S * V^T = A
            let us = u.matmul(&s_diag).unwrap();
            let a_reconstructed = us.matmul(&vt).unwrap();

            println!("Verification - U * S * V^T:");
            print_matrix(&a_reconstructed);

            println!("Is close to original? {}", is_close(&a, &a_reconstructed));
        }
        Err(e) => println!("Error: {}", e),
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("SVD requires the 'matrix_decomp' and 'lapack' features.");
        println!("Enable with: cargo run --example matrix_decomp_example --features matrix_decomp,lapack");
    }

    // Eigendecomposition - A = P * D * P^(-1)
    println!("\n4. Eigenvalue Decomposition");
    println!("--------------------------");
    #[cfg(feature = "eigenvalues")]
    match eigenvalues::eigh(&a, "lower") {
        Ok((eigenvalues, eigenvectors)) => {
            println!("Eigenvalues:");
            println!("{:?}", eigenvalues.to_vec());

            println!("Eigenvectors:");
            print_matrix(&eigenvectors);

            // Create diagonal matrix from eigenvalues
            let mut d = Array::zeros(&[eigenvectors.shape()[0], eigenvectors.shape()[0]]);
            for i in 0..eigenvalues.size() {
                d.set(&[i, i], eigenvalues.to_vec()[i]).unwrap();
            }

            // Verify: P * D * P^T = A (for symmetric matrices, P^(-1) = P^T)
            let pd = eigenvectors.matmul(&d).unwrap();
            let a_reconstructed = pd.matmul(&eigenvectors.transpose()).unwrap();

            println!("Verification - P * D * P^T:");
            print_matrix(&a_reconstructed);

            println!("Is close to original? {}", is_close(&a, &a_reconstructed));
        }
        Err(e) => println!("Error: {}", e),
    }
    #[cfg(not(feature = "eigenvalues"))]
    {
        println!("Eigenvalue decomposition requires the 'eigenvalues' feature.");
        println!("Enable with: cargo run --example matrix_decomp_example --features eigenvalues");
    }

    // LU Decomposition - A = L * U
    println!("\n5. LU Decomposition");
    println!("-------------------");
    #[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
    match matrix_decomp::lu(&a) {
        Ok((l, u, _p)) => {
            println!("Lower triangular matrix L:");
            print_matrix(&l);

            println!("Upper triangular matrix U:");
            print_matrix(&u);

            // Verify: L * U ≈ A
            let a_reconstructed = l.matmul(&u).unwrap();
            println!("Verification - L * U:");
            print_matrix(&a_reconstructed);
            println!("Is close to original? {}", is_close(&a, &a_reconstructed));
        }
        Err(e) => println!("Error: {}", e),
    }
    #[cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]
    {
        println!("LU decomposition requires the 'matrix_decomp' and 'lapack' features.");
        println!("Enable with: cargo run --example matrix_decomp_example --features matrix_decomp,lapack");
    }
}

// Helper function to print matrices in a readable format
fn print_matrix<T: Clone + std::fmt::Display>(matrix: &Array<T>) {
    let shape = matrix.shape();
    if shape.len() != 2 {
        println!("{}", matrix);
        return;
    }

    let rows = shape[0];
    let cols = shape[1];
    let data = matrix.to_vec();

    for i in 0..rows {
        print!("[ ");
        for j in 0..cols {
            print!("{:8.4} ", data[i * cols + j]);
        }
        println!("]");
    }
}

// Helper function to check if two matrices are close
fn is_close<T: Clone + num_traits::Float>(a: &Array<T>, b: &Array<T>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let epsilon = T::epsilon() * T::from(100.0).unwrap();

    for i in 0..a_data.len() {
        if (a_data[i] - b_data[i]).abs() > epsilon {
            return false;
        }
    }

    true
}

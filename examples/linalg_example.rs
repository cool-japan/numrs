use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS Linear Algebra Example");
    println!("===========================");

    // Create a 2x2 matrix
    let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);
    println!("Matrix A:");
    println!("{}", a);

    // Compute determinant
    let det_a = det(&a)?;
    println!("\nDeterminant of A: {}", det_a);

    // Compute inverse
    let inv_a = inv(&a)?;
    println!("\nInverse of A:");
    println!("{}", inv_a);

    // Verify that A * A^(-1) = I
    let product = a.matmul(&inv_a)?;
    println!("\nA * A^(-1):");
    println!("{}", product);

    // Create a vector
    let b = Array::from_vec(vec![1.0, 3.0]);
    println!("\nVector b:");
    println!("{}", b);

    // Solve the system Ax = b
    let x = solve(&a, &b)?;
    println!("\nSolution to Ax = b:");
    println!("{}", x);

    // Verify the solution
    let b_check = a.matmul(&x.reshape(&[2, 1]))?.reshape(&[2]);
    println!("\nVerify: A*x =");
    println!("{}", b_check);

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
        well_conditioned.is_well_conditioned().unwrap_or(false)
    );

    // SVD decomposition
    let (u, s, vt) = svd(&well_conditioned)?;
    println!("\nSVD decomposition:");
    println!("U = {}", u);
    println!("S = {}", s);
    println!("V^T = {}", vt);

    // QR decomposition
    let (q, r) = qr(&well_conditioned)?;
    println!("\nQR decomposition:");
    println!("Q = {}", q);
    println!("R = {}", r);

    // Cholesky decomposition
    // Create a symmetric positive definite matrix
    let spd = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
    let l = cholesky(&spd)?;
    println!("\nCholesky decomposition (L):");
    println!("{}", l);

    // Pivoted Cholesky for improved numerical stability
    let (l_piv, p) = pivoted_cholesky(&spd)?;
    println!("\nPivoted Cholesky decomposition:");
    println!("L = {}", l_piv);
    println!("Permutation = {}", p);

    // LU decomposition
    let (l_lu, u, p) = lu(&well_conditioned)?;
    println!("\nLU decomposition:");
    println!("L = {}", l_lu);
    println!("U = {}", u);
    println!("Permutation = {}", p);

    // Create an ill-conditioned matrix (Hilbert matrix)
    let hilbert = create_hilbert_matrix(4);
    println!("\nHilbert matrix (ill-conditioned):");
    println!("{}", hilbert);

    // Calculate condition number
    let cond_hilbert = hilbert.cond()?;
    println!("Condition number: {}", cond_hilbert);
    println!("Is well-conditioned: {}", hilbert.is_well_conditioned()?);

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

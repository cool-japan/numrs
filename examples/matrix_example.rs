#![allow(deprecated)]

use numrs2::prelude::*;

fn main() {
    println!("Matrix Examples");
    println!("==============\n");

    matrix_basics();
    matrix_operations();
    banded_matrix_example();
    special_matrices_example();
}

fn matrix_basics() {
    println!("1. Matrix Basics");
    println!("---------------");

    // Create a matrix from a 2D array
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let matrix = Matrix::new(array.clone()).unwrap();

    println!("Original array:\n{}", array);
    println!("Matrix created from array:\n{}", matrix);

    // Create a matrix directly
    let nested_vec = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let matrix2 = Matrix::from_nested_vec(nested_vec).unwrap();
    println!("Matrix created from nested vector:\n{}", matrix2);

    // Create specialized matrices
    let zeros = Matrix::<f64>::zeros(2, 3);
    println!("Zero matrix (2x3):\n{}", zeros);

    let ones = Matrix::<f64>::ones(2, 3);
    println!("Ones matrix (2x3):\n{}", ones);

    let identity = Matrix::<f64>::eye(3);
    println!("Identity matrix (3x3):\n{}", identity);

    // Matrix properties
    println!("Matrix shape: {:?}", matrix.shape());
    println!("Number of rows: {}", matrix.nrows());
    println!("Number of columns: {}", matrix.ncols());
    println!("Total elements: {}", matrix.size());

    // Matrix indexing
    println!("Value at (0,1): {}", matrix.get(0, 1).unwrap());

    // Matrix transpose
    let transposed = matrix.transpose();
    println!("\nTransposed matrix:\n{}", transposed);

    // Get rows and columns
    let row = matrix.row(0).unwrap();
    println!("First row:\n{}", row);

    let column = matrix.column(1).unwrap();
    println!("Second column:\n{}", column);

    println!();
}

fn matrix_operations() {
    println!("2. Matrix Operations");
    println!("------------------");

    // Create two matrices
    let a = Matrix::from_nested_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

    let b = Matrix::from_nested_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

    println!("Matrix A:\n{}", a);
    println!("Matrix B:\n{}", b);

    // Matrix addition
    let sum = &a + &b;
    println!("A + B:\n{}", sum);

    // Matrix subtraction
    let diff = &a - &b;
    println!("A - B:\n{}", diff);

    // Matrix multiplication
    let product = &a * &b;
    println!("A * B (matrix multiplication):\n{}", product);

    // Matrix-vector multiplication
    let vec = Array::from_vec(vec![1.0, 2.0]);
    let result = a.dot(&Matrix::from_vec(vec.to_vec())).unwrap();
    println!("A * [1.0, 2.0]:\n{}", result);

    // Check properties
    let square = Matrix::from_nested_vec(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ])
    .unwrap();

    println!("\nDiagonal matrix:\n{}", square);
    println!("Is square: {}", square.is_square());
    println!("Is symmetric: {}", square.is_symmetric());

    // Diagonal extraction
    let diag = square.diagonal();
    println!("Diagonal elements:\n{}", diag);

    println!();
}

fn banded_matrix_example() {
    println!("3. Banded Matrix Example");
    println!("----------------------");

    // Create a banded matrix
    let mut banded = BandedMatrix::new(5, 5, 1, 1);

    // Set values in the band
    for i in 0..5 {
        // Main diagonal
        banded.set(i, i, (i + 1) as f64).unwrap();

        // Sub-diagonal (below main)
        if i > 0 {
            banded.set(i, i - 1, -1.0).unwrap();
        }

        // Super-diagonal (above main)
        if i < 4 {
            banded.set(i, i + 1, -1.0).unwrap();
        }
    }

    println!("Banded matrix (tridiagonal):\n{}", banded);

    // Convert to full array
    let full_array = banded.to_array();
    println!("Full array representation:\n{}", full_array);

    // Matrix-vector product
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = banded.matvec(&vec).unwrap();
    println!("Banded matrix * vector: {:?}", result);

    // Create from existing array
    let array = Array::from_vec(vec![
        1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 9.0, 10.0,
    ])
    .reshape(&[4, 4]);

    let banded2 = BandedMatrix::from_array(&array, 1, 1).unwrap();
    println!("\nOriginal array:\n{}", array);
    println!("Banded representation (sub=1, super=1):\n{}", banded2);

    println!();
}

fn special_matrices_example() {
    println!("4. Special Matrices");
    println!("-----------------");

    // Import special matrix functions
    use numrs2::matrix::special;

    // Hilbert matrix
    let hilbert = special::hilbert(4);
    println!("Hilbert matrix (4x4):\n{}", hilbert);

    // Vandermonde matrix
    let vander = special::vandermonde(vec![1.0, 2.0, 3.0, 4.0], None);
    println!("\nVandermonde matrix:\n{}", vander);

    // Toeplitz matrix
    let toeplitz =
        special::toeplitz(vec![1.0, 2.0, 3.0, 4.0], Some(vec![1.0, 5.0, 6.0, 7.0])).unwrap();
    println!("\nToeplitz matrix:\n{}", toeplitz);

    // Circulant matrix
    let circulant = special::circulant(vec![1.0, 2.0, 3.0, 4.0]);
    println!("\nCirculant matrix:\n{}", circulant);

    // Companion matrix
    let companion = special::companion(vec![1.0, -3.0, 3.0, -1.0]).unwrap();
    println!(
        "\nCompanion matrix for polynomial x³ - 3x² + 3x - 1:\n{}",
        companion
    );

    // Leslie matrix (population modeling)
    let leslie = special::leslie(vec![1.2, 0.8, 0.2], vec![0.7, 0.1]).unwrap();
    println!("\nLeslie matrix for population modeling:\n{}", leslie);

    println!();
}

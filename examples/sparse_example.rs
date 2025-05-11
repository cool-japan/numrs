use numrs2::prelude::*;

fn main() {
    println!("NumRS Sparse Array Examples");
    println!("==========================\n");

    // 1. Basic sparse array operations
    println!("1. Basic Sparse Array Operations");
    println!("-------------------------------");

    // Create a sparse array
    let mut sparse = SparseArray::new(&[5, 5]);

    // Set some values
    sparse.set(&[0, 0], 1.0).unwrap();
    sparse.set(&[1, 1], 2.0).unwrap();
    sparse.set(&[2, 2], 3.0).unwrap();
    sparse.set(&[3, 3], 4.0).unwrap();
    sparse.set(&[4, 4], 5.0).unwrap();
    sparse.set(&[0, 4], 6.0).unwrap();
    sparse.set(&[4, 0], 7.0).unwrap();

    println!("Created a 5x5 sparse array with diagonal values and two off-diagonal values");
    println!("Non-zero count: {}", sparse.nnz());
    println!("Density: {:.2}%", sparse.density() * 100.0);

    // Convert to dense for display
    let dense = sparse.to_array();
    println!("\nAs dense array:");
    print_2d_array(&dense);

    // Access elements
    println!("\nAccessing elements:");
    println!("sparse[0, 0] = {}", sparse.get(&[0, 0]).unwrap());
    println!("sparse[2, 2] = {}", sparse.get(&[2, 2]).unwrap());
    println!("sparse[0, 4] = {}", sparse.get(&[0, 4]).unwrap());
    println!("sparse[1, 3] = {}", sparse.get(&[1, 3]).unwrap()); // Should be 0.0

    // 2. Sparse array arithmetic
    println!("\n2. Sparse Array Arithmetic");
    println!("-------------------------");

    // Create another sparse array
    let mut sparse2 = SparseArray::new(&[5, 5]);
    sparse2.set(&[0, 0], 5.0).unwrap();
    sparse2.set(&[1, 1], 4.0).unwrap();
    sparse2.set(&[2, 2], 3.0).unwrap();
    sparse2.set(&[3, 3], 2.0).unwrap();
    sparse2.set(&[4, 4], 1.0).unwrap();
    sparse2.set(&[1, 3], 7.0).unwrap();

    println!("Second sparse array");
    let dense2 = sparse2.to_array();
    println!("Non-zero count: {}", sparse2.nnz());
    print_2d_array(&dense2);

    // Addition
    match sparse.add(&sparse2) {
        Ok(sum) => {
            println!("\nSum of sparse arrays:");
            println!("Non-zero count: {}", sum.nnz());
            let dense_sum = sum.to_array();
            print_2d_array(&dense_sum);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Element-wise multiplication
    match sparse.multiply(&sparse2) {
        Ok(product) => {
            println!("\nElement-wise product of sparse arrays:");
            println!("Non-zero count: {}", product.nnz());
            let dense_product = product.to_array();
            print_2d_array(&dense_product);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Scalar multiplication
    let scaled = sparse.multiply_scalar(2.0);
    println!("\nSparse array multiplied by 2.0:");
    println!("Non-zero count: {}", scaled.nnz());
    let dense_scaled = scaled.to_array();
    print_2d_array(&dense_scaled);

    // 3. Sparse matrices and specialized formats
    println!("\n3. Sparse Matrix Formats");
    println!("-----------------------");

    // Create a sparse matrix
    match SparseMatrix::new(&[4, 4]) {
        Ok(mut sparse_matrix) => {
            // Set some values
            sparse_matrix.set(0, 0, 1.0).unwrap();
            sparse_matrix.set(1, 1, 2.0).unwrap();
            sparse_matrix.set(2, 2, 3.0).unwrap();
            sparse_matrix.set(3, 3, 4.0).unwrap();
            sparse_matrix.set(0, 3, 5.0).unwrap();
            sparse_matrix.set(3, 0, 6.0).unwrap();

            println!("Original sparse matrix (COO format):");
            let dense_matrix = sparse_matrix.to_array();
            print_2d_array(&dense_matrix);

            // Convert to CSR format
            match sparse_matrix.to_csr() {
                Ok(_) => {
                    println!("\nConverted to CSR (Compressed Sparse Row) format");
                    println!(
                        "This format is efficient for row operations and matrix-vector products"
                    );

                    // Verify data is still accessible
                    println!("\nVerifying elements:");
                    println!("matrix[0, 0] = {}", sparse_matrix.get(0, 0).unwrap());
                    println!("matrix[2, 2] = {}", sparse_matrix.get(2, 2).unwrap());
                    println!("matrix[0, 3] = {}", sparse_matrix.get(0, 3).unwrap());
                }
                Err(e) => println!("Error converting to CSR: {}", e),
            }

            // Convert to CSC format
            match sparse_matrix.to_csc() {
                Ok(_) => {
                    println!("\nConverted to CSC (Compressed Sparse Column) format");
                    println!("This format is efficient for column operations");

                    // Verify data is still accessible
                    println!("\nVerifying elements:");
                    println!("matrix[0, 0] = {}", sparse_matrix.get(0, 0).unwrap());
                    println!("matrix[3, 0] = {}", sparse_matrix.get(3, 0).unwrap());
                }
                Err(e) => println!("Error converting to CSC: {}", e),
            }
        }
        Err(e) => println!("Error creating sparse matrix: {}", e),
    }

    // 4. Special sparse matrix constructors
    println!("\n4. Special Sparse Matrix Constructors");
    println!("-----------------------------------");

    // Identity matrix
    match SparseMatrix::eye(4) {
        Ok(eye) => {
            println!("Sparse identity matrix (4x4):");
            println!("Non-zero count: {}", eye.nnz());
            let dense_eye = eye.to_array();
            print_2d_array(&dense_eye);
        }
        Err(e) => println!("Error creating identity matrix: {}", e),
    }

    // Diagonal matrix
    match SparseMatrix::diag(&[1.0, 2.0, 3.0, 4.0, 5.0]) {
        Ok(diag) => {
            println!("\nSparse diagonal matrix from vector [1, 2, 3, 4, 5]:");
            println!("Non-zero count: {}", diag.nnz());
            let dense_diag = diag.to_array();
            print_2d_array(&dense_diag);
        }
        Err(e) => println!("Error creating diagonal matrix: {}", e),
    }

    // 5. Sparse matrix operations
    println!("\n5. Sparse Matrix Operations");
    println!("---------------------------");

    // Create two sparse matrices for multiplication
    match SparseMatrix::new(&[3, 3]) {
        Ok(mut a) => {
            a.set(0, 0, 1.0).unwrap();
            a.set(0, 1, 2.0).unwrap();
            a.set(1, 0, 3.0).unwrap();
            a.set(1, 1, 4.0).unwrap();
            a.set(2, 2, 5.0).unwrap();

            match SparseMatrix::new(&[3, 2]) {
                Ok(mut b) => {
                    b.set(0, 0, 6.0).unwrap();
                    b.set(0, 1, 7.0).unwrap();
                    b.set(1, 0, 8.0).unwrap();
                    b.set(1, 1, 9.0).unwrap();

                    println!("Matrix A:");
                    print_2d_array(&a.to_array());

                    println!("\nMatrix B:");
                    print_2d_array(&b.to_array());

                    // Matrix multiplication
                    match a.matmul(&b) {
                        Ok(c) => {
                            println!("\nMatrix C = A * B:");
                            println!("Shape: {:?}", c.shape());
                            println!("Non-zero count: {}", c.nnz());
                            print_2d_array(&c.to_array());

                            // Verify: should be [[22, 25], [50, 57], [0, 0]]
                        }
                        Err(e) => println!("Error in matrix multiplication: {}", e),
                    }

                    // Transpose
                    match a.transpose() {
                        Ok(at) => {
                            println!("\nTranspose of A:");
                            print_2d_array(&at.to_array());
                        }
                        Err(e) => println!("Error computing transpose: {}", e),
                    }
                }
                Err(e) => println!("Error creating matrix B: {}", e),
            }
        }
        Err(e) => println!("Error creating matrix A: {}", e),
    }

    // 6. Performance benefits of sparse arrays
    println!("\n6. Performance Benefits of Sparse Arrays");
    println!("--------------------------------------");
    println!("In a large-scale application, sparse arrays provide significant memory savings");
    println!("For example, a 1,000,000 x 1,000,000 matrix with 0.001% non-zeros:");
    println!("  - Dense: Would require 8 TB of memory (8 bytes per element)");
    println!("  - Sparse: Would require ~8 GB of memory (storing only non-zeros)");
    println!("\nSparse matrices are particularly useful for:");
    println!("  - Large network/graph problems");
    println!("  - Finite element analysis");
    println!("  - Document-term matrices in text processing");
    println!("  - Machine learning with sparse features");
}

// Helper function to print 2D arrays
fn print_2d_array(array: &Array<f64>) {
    let shape = array.shape();
    if shape.len() != 2 {
        println!("Not a 2D array");
        return;
    }

    let rows = shape[0];
    let cols = shape[1];
    let data = array.to_vec();

    for i in 0..rows {
        for j in 0..cols {
            let val = data[i * cols + j];
            // Format with fixed width for better visualization
            print!("{:5.1} ", val);
        }
        println!();
    }
}

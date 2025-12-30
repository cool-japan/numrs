# NumRS Linear Algebra Module

The NumRS linear algebra module provides high-performance implementations of essential matrix operations, decompositions, and solvers. The module is designed to be both efficient (using BLAS/LAPACK bindings where appropriate) and numerically stable.

## Key Features

- Basic matrix operations (multiplication, inversion, determinant)
- Linear system solvers with various methods
- Matrix decompositions (SVD, QR, LU, Cholesky, etc.)
- Eigenvalue and eigenvector computation
- Norms and condition number calculations
- Matrix factorizations for numerical stability
- Broadcasting support for matrix operations

## Basic Operations

### Matrix-Matrix Multiplication

```rust
use numrs2::prelude::*;

// Create matrices
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

// Matrix multiplication
let c = a.matmul(&b)?;
// Or use the shorthand dot method
let c2 = a.dot(&b)?;
```

### Matrix-Vector Multiplication

```rust
use numrs2::prelude::*;

// Create matrix and vector
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let v = Array::from_vec(vec![5.0, 6.0]);

// Matrix-vector product
let result = a.matmul(&v.reshape(&[2, 1]))?.reshape(&[2]);
// Or use the convenience method
let result2 = a.dot_vec(&v)?;
```

### Matrix Inversion

```rust
use numrs2::prelude::*;
use numrs2::linalg::inv;

// Create a matrix
let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);

// Compute the inverse
let inv_a = inv(&a)?;

// Verify: A × A⁻¹ = I
let product = a.matmul(&inv_a)?;
```

### Determinant

```rust
use numrs2::prelude::*;
use numrs2::linalg::det;

// Create a matrix
let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);

// Compute the determinant
let det_a = det(&a)?;
```

## Solving Linear Systems

### Direct Solver

```rust
use numrs2::prelude::*;
use numrs2::linalg::solve;

// Create coefficient matrix and right-hand side
let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![1.0, 3.0]);

// Solve the system Ax = b
let x = solve(&a, &b)?;

// Verify the solution
let b_check = a.matmul(&x.reshape(&[2, 1]))?.reshape(&[2]);
```

### Least Squares Solver

```rust
use numrs2::prelude::*;
use numrs2::linalg::lstsq;

// Create an overdetermined system
let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0]).reshape(&[3, 2]);
let b = Array::from_vec(vec![6.0, 5.0, 7.0]);

// Solve in the least squares sense
let (x, residuals, rank, singular_values) = lstsq(&a, &b, None)?;
```

### Solving with Specific Decompositions

```rust
use numrs2::prelude::*;
use numrs2::linalg::{solve_svd, solve_qr, solve_lu, solve_cholesky};

// Create coefficient matrix and right-hand side
let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![1.0, 3.0]);

// Solve using different decompositions
let x_svd = solve_svd(&a, &b)?;
let x_qr = solve_qr(&a, &b)?;
let x_lu = solve_lu(&a, &b)?;

// For symmetric positive definite matrices
let spd = Array::from_vec(vec![4.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
let b2 = Array::from_vec(vec![1.0, 2.0]);
let x_chol = solve_cholesky(&spd, &b2)?;
```

## Matrix Decompositions

### Singular Value Decomposition (SVD)

```rust
use numrs2::prelude::*;
use numrs2::linalg::svd;

// Create a matrix
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

// Compute SVD: A = U * Σ * Vᵀ
let (u, s, vt) = svd(&a)?;

// Reconstruct the original matrix
let diag_s = Array::diag(&s);
let reconstructed = u.matmul(&diag_s)?.matmul(&vt)?;
```

### QR Decomposition

```rust
use numrs2::prelude::*;
use numrs2::linalg::qr;

// Create a matrix
let a = Array::from_vec(vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0])
       .reshape(&[3, 3]);

// Compute QR: A = Q * R
let (q, r) = qr(&a)?;

// Verify: Q is orthogonal (Q * Qᵀ = I)
let q_orth = q.matmul(&q.transpose())?;

// Verify: A = Q * R
let reconstructed = q.matmul(&r)?;
```

### LU Decomposition

```rust
use numrs2::prelude::*;
use numrs2::linalg::lu;

// Create a matrix
let a = Array::from_vec(vec![2.0, 5.0, 8.0, 7.0, 3.0, 1.0, 6.0, 4.0, 9.0]).reshape(&[3, 3]);

// Compute LU: P * A = L * U
let (l, u, p) = lu(&a)?;

// Verify: P * A = L * U
let pa = p.matmul(&a)?;
let lu_product = l.matmul(&u)?;
```

### Cholesky Decomposition

```rust
use numrs2::prelude::*;
use numrs2::linalg::cholesky;

// Create a symmetric positive definite matrix
let spd = Array::from_vec(vec![4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0]).reshape(&[3, 3]);

// Compute Cholesky: A = L * Lᵀ
let l = cholesky(&spd)?;

// Verify: A = L * Lᵀ
let reconstructed = l.matmul(&l.transpose())?;
```

### Eigenvalue Decomposition

```rust
use numrs2::prelude::*;
use numrs2::linalg::eig;

// Create a matrix
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

// Compute eigenvalues and eigenvectors
let (eigenvalues, eigenvectors) = eig(&a)?;

// For symmetric matrices, use the more efficient eigenh
let symmetric = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0]).reshape(&[2, 2]);
let (evals_sym, evecs_sym) = eigenh(&symmetric)?;
```

## Numerical Stability Analysis

### Condition Number

```rust
use numrs2::prelude::*;
use numrs2::linalg::cond;

// Create a matrix
let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0]).reshape(&[2, 2]);

// Compute the condition number (using 2-norm by default)
let cond_num = cond(&a)?;

// Check if the matrix is well-conditioned
let is_well_cond = a.is_well_conditioned()?;
```

### Creating Ill-Conditioned Matrices

```rust
use numrs2::prelude::*;

// Create a Hilbert matrix (notoriously ill-conditioned)
let hilbert = create_hilbert_matrix(5);
let cond_hilbert = hilbert.cond()?;

// Helper function
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
```

## Performance Considerations

### BLAS Integration

NumRS uses BLAS for many underlying matrix operations:

```rust
use numrs2::prelude::*;
use numrs2::blas::gemm;

// Matrix-matrix multiplication with BLAS
let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
let c = Array::zeros(&[2, 2]);

// C = α*A*B + β*C
let result = gemm(1.0, &a, &b, 0.0, &c)?;
```

### Advanced Solvers

For large-scale problems, specialized solvers can be more efficient:

```rust
use numrs2::prelude::*;
use numrs2::linalg::solve_iterative;

// Create a large sparse system
let a = create_large_sparse_matrix();
let b = Array::ones(&[a.shape()[0]]);

// Solve using an iterative method
let x = solve_iterative(&a, &b, "cg", 1e-6, 1000)?;
```

## Working with Matrices vs Arrays

NumRS provides both a general `Array` type and a specialized `Matrix` type:

```rust
use numrs2::prelude::*;

// Array-based operations
let a_array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b_array = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
let c_array = a_array.matmul(&b_array)?;

// Matrix-based operations
let a_matrix = Matrix::from_array(&a_array)?;
let b_matrix = Matrix::from_array(&b_array)?;
let c_matrix = a_matrix.multiply(&b_matrix)?;
```

## Examples

For complete examples of linear algebra operations, see:
- `linalg_example.rs`: Basic linear algebra operations
- `matrix_example.rs`: Matrix-specific operations
- `matrix_decomp_example.rs`: Matrix decomposition methods

## Implementation Details

- NumRS uses `ndarray-linalg` for many core operations
- BLAS/LAPACK bindings are used for high-performance calculations
- Operations support both CPU and (where available) GPU acceleration
- Numerical stability is carefully considered in all implementations
- Broadcasting is supported for certain operations

## BLAS/LAPACK Integration

NumRS leverages the highly-optimized BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage) libraries for many operations:

1. **BLAS Level 1**: Vector operations
   - Dot products, norms, vector scaling

2. **BLAS Level 2**: Matrix-vector operations
   - Matrix-vector multiplication (gemv)
   - Rank-1 updates (ger)

3. **BLAS Level 3**: Matrix-matrix operations
   - Matrix multiplication (gemm)
   - Triangular solvers (trsm)

4. **LAPACK**: Advanced linear algebra
   - Matrix decompositions
   - Eigenvalue problems
   - Linear system solvers

This integration allows NumRS to achieve performance comparable to NumPy/SciPy while maintaining Rust's safety guarantees.
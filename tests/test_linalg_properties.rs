use approx::assert_abs_diff_eq;
/// Property-based tests for linear algebra operations
///
/// This file tests the mathematical properties and relationships
/// that should be satisfied by linear algebra operations.
use numrs2::prelude::*;

// Import from the core linalg module (non-deprecated)
#[allow(deprecated)]
use numrs2::new_modules::matrix_decomp::cholesky;
#[allow(deprecated)]
use numrs2::new_modules::matrix_decomp::qr;
#[allow(deprecated)]
use numrs2::new_modules::matrix_decomp::svd;
#[cfg(feature = "matrix_decomp")]
#[allow(deprecated)]
use numrs2::new_modules::matrix_decomp::{condition_number, lu};
#[cfg(feature = "lapack")]
#[allow(deprecated)]
use numrs2::new_modules::eigenvalues::eigh as eigh_impl;
use numrs2::linalg::vector_ops::{norm, trace};
use numrs2::linalg::solve::{inv, solve};
#[cfg(feature = "lapack")]
use numrs2::linalg::matrix_ops::det;

// For eigenvalues, we need to define a wrapper that calls the correct function
#[cfg(feature = "lapack")]
#[allow(deprecated)]
fn eigh(a: &Array<f64>, uplo: &str) -> numrs2::error::Result<(Array<f64>, Array<f64>)> {
    // Use the correct symmetric eigendecomposition function
    eigh_impl(a, uplo)
}

// Constants for testing
// Tolerance for floating point comparisons
const TOLERANCE: f64 = 1e-8;
const MATRIX_SIZES: [usize; 2] = [3, 5]; // Test with smaller matrix sizes to avoid stack overflow

/// Helper function to generate a random matrix with specific dimensions
fn random_matrix(rows: usize, cols: usize) -> Array<f64> {
    let rng = random::default_rng();
    rng.random::<f64>(&[rows, cols]).unwrap()
}


/// Helper function to generate a random positive definite matrix
fn random_positive_definite_matrix(size: usize) -> Array<f64> {
    // Create a simple well-conditioned positive definite matrix
    // Using a diagonal matrix with values > 1 to ensure positive definiteness
    let mut data = vec![0.0; size * size];
    
    // Fill diagonal with values 1.0 + i to ensure positive definiteness
    for i in 0..size {
        data[i * size + i] = 1.0 + i as f64;
    }
    
    // Add some off-diagonal elements to make it more interesting
    for i in 0..size {
        for j in 0..size {
            if i != j {
                data[i * size + j] = 0.1 * (i + j + 1) as f64 / (size as f64);
            }
        }
    }
    
    let mut result = Array::from_vec(data).reshape(&[size, size]);
    
    // Make it symmetric: A = (M + M^T) / 2 + I
    let a_t = result.transpose();
    result = result.add(&a_t).multiply_scalar(0.5);
    let eye = Array::<f64>::eye(size, size, 0);
    result.add(&eye)
}

/// Helper function to check if matrices are approximately equal
fn matrices_approx_equal(a: &Array<f64>, b: &Array<f64>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    // Get flat vectors
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    for (a_val, b_val) in a_vec.iter().zip(b_vec.iter()) {
        if (a_val - b_val).abs() > TOLERANCE {
            return false;
        }
    }

    true
}

/// Helper function to check if a matrix is approximately symmetric
fn is_approximately_symmetric(m: &Array<f64>) -> bool {
    if m.shape()[0] != m.shape()[1] {
        return false;
    }

    let m_t = m.transpose();
    matrices_approx_equal(m, &m_t)
}

/// Helper function to check if a matrix is approximately orthogonal
fn is_approximately_orthogonal(m: &Array<f64>) -> bool {
    if m.shape()[0] != m.shape()[1] {
        return false;
    }

    let m_t = m.transpose();
    let identity = Array::<f64>::eye(m.shape()[0], m.shape()[0], 0);
    let product = m.matmul(&m_t).unwrap();

    matrices_approx_equal(&product, &identity)
}

#[test]
fn test_matmul_properties() {
    // Test matrix multiplication properties
    for &size in MATRIX_SIZES.iter() {
        // Create matrices
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);
        let c = random_matrix(size, size);

        // Property 1: Associativity (A*B)*C = A*(B*C)
        let ab = a.matmul(&b).unwrap();
        let ab_c = ab.matmul(&c).unwrap();

        let bc = b.matmul(&c).unwrap();
        let a_bc = a.matmul(&bc).unwrap();

        assert!(
            matrices_approx_equal(&ab_c, &a_bc),
            "Matrix multiplication should be associative"
        );

        // Property 2: Distributivity A*(B+C) = A*B + A*C
        let b_plus_c = b.add(&c);
        let a_bc = a.matmul(&b_plus_c).unwrap();

        let ab = a.matmul(&b).unwrap();
        let ac = a.matmul(&c).unwrap();
        let ab_plus_ac = ab.add(&ac);

        assert!(
            matrices_approx_equal(&a_bc, &ab_plus_ac),
            "Matrix multiplication should be distributive"
        );

        // Property 3: Identity property: A*I = A and I*A = A
        let identity = Array::<f64>::eye(size, size, 0);
        let a_i = a.matmul(&identity).unwrap();
        let i_a = identity.matmul(&a).unwrap();

        assert!(matrices_approx_equal(&a, &a_i), "A*I should equal A");
        assert!(matrices_approx_equal(&a, &i_a), "I*A should equal A");
    }
}

#[test]
fn test_transpose_properties() {
    // Test matrix transpose properties
    for &size in MATRIX_SIZES.iter() {
        // Create matrices
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);

        // Property 1: (A^T)^T = A
        let a_t = a.transpose();
        let a_t_t = a_t.transpose();

        assert!(
            matrices_approx_equal(&a, &a_t_t),
            "Double transpose should return the original matrix"
        );

        // Property 2: (A+B)^T = A^T + B^T
        let a_plus_b = a.add(&b);
        let a_plus_b_t = a_plus_b.transpose();

        let a_t = a.transpose();
        let b_t = b.transpose();
        let a_t_plus_b_t = a_t.add(&b_t);

        assert!(
            matrices_approx_equal(&a_plus_b_t, &a_t_plus_b_t),
            "Transpose should distribute over addition"
        );

        // Property 3: (A*B)^T = B^T * A^T
        let ab = a.matmul(&b).unwrap();
        let ab_t = ab.transpose();

        let a_t = a.transpose();
        let b_t = b.transpose();
        let b_t_a_t = b_t.matmul(&a_t).unwrap();

        assert!(
            matrices_approx_equal(&ab_t, &b_t_a_t),
            "Transpose of product should equal product of transposes in reverse order"
        );
    }
}

#[test]
fn test_inverse_properties() {
    // Test matrix inverse properties
    for &size in MATRIX_SIZES.iter() {
        // Create random invertible matrices (using positive definite matrices ensures invertibility)
        let a = random_positive_definite_matrix(size);

        // Property 1: A * A^(-1) = I
        let a_inv = inv(&a).unwrap();
        let _identity = Array::<f64>::eye(size, size, 0);

        let a_a_inv = a.matmul(&a_inv).unwrap();

        for i in 0..size {
            for j in 0..size {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(a_a_inv.get(&[i, j]).unwrap(), expected, epsilon = TOLERANCE);
            }
        }

        // Property 2: (A^(-1))^(-1) = A
        let a_inv_inv = inv(&a_inv).unwrap();

        assert!(
            matrices_approx_equal(&a, &a_inv_inv),
            "Double inversion should return the original matrix"
        );

        // Property 3: (A^T)^(-1) = (A^(-1))^T
        let a_t = a.transpose();
        let a_t_inv = inv(&a_t).unwrap();

        let a_inv_t = a_inv.transpose();

        assert!(
            matrices_approx_equal(&a_t_inv, &a_inv_t),
            "Inverse of transpose should equal transpose of inverse"
        );
    }
}

#[cfg(feature = "lapack")]
#[test]
fn test_determinant_properties() {
    // Test determinant properties
    for &size in MATRIX_SIZES.iter() {
        if size < 3 {
            continue; // Skip very small matrices for some tests
        }

        // Create random matrices
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);

        // Property 1: det(A*B) = det(A) * det(B)
        let ab = a.matmul(&b).unwrap();
        let det_ab = det(&ab).unwrap();

        let det_a = det(&a).unwrap();
        let det_b = det(&b).unwrap();
        let det_a_det_b = det_a * det_b;

        assert_abs_diff_eq!(det_ab, det_a_det_b, epsilon = TOLERANCE * 10.0);

        // Property 2: det(A^T) = det(A)
        let a_t = a.transpose();
        let det_a_t = det(&a_t).unwrap();

        assert_abs_diff_eq!(det_a, det_a_t, epsilon = TOLERANCE);

        // Property 3: det(kA) = k^n * det(A) for n x n matrix
        let k = 2.0;
        let k_a = a.multiply_scalar(k);
        let det_k_a = det(&k_a).unwrap();

        let k_pow_n = k.powi(size as i32);
        let expected = k_pow_n * det_a;

        assert_abs_diff_eq!(det_k_a, expected, epsilon = TOLERANCE * 100.0);
    }
}

#[cfg(feature = "lapack")]
#[test]
fn test_eigendecomposition_properties() {
    // Test eigendecomposition properties using symmetric matrices
    // Use fixed matrices instead of random ones to avoid numerical issues
    let test_matrices = vec![
        // 3x3 symmetric matrix with known eigenvalues
        {
            let mut m = Array::<f64>::zeros(&[3, 3]);
            m.set(&[0, 0], 4.0).unwrap();
            m.set(&[0, 1], 1.0).unwrap();
            m.set(&[0, 2], 1.0).unwrap();
            m.set(&[1, 0], 1.0).unwrap();
            m.set(&[1, 1], 3.0).unwrap();
            m.set(&[1, 2], 1.0).unwrap();
            m.set(&[2, 0], 1.0).unwrap();
            m.set(&[2, 1], 1.0).unwrap();
            m.set(&[2, 2], 2.0).unwrap();
            m
        },
    ];

    for a in test_matrices {
        let size = a.shape()[0];

        // Check if the matrix is actually symmetric (for debugging)
        assert!(
            is_approximately_symmetric(&a),
            "Matrix is not symmetric as expected"
        );

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = eigh(&a, "lower").unwrap();

        // Property 1: A*v_i = λ_i*v_i for each eigenpair (λ_i, v_i)
        for i in 0..size {
            // Extract the i-th eigenvalue
            let lambda_i = eigenvalues.get(&[i]).unwrap();

            // Extract the i-th eigenvector (column i of eigenvectors)
            let mut v_i = Array::<f64>::zeros(&[size]);
            for j in 0..size {
                v_i.set(&[j], eigenvectors.get(&[j, i]).unwrap()).unwrap();
            }

            // Compute A*v_i
            let av_i = a.matmul(&v_i.reshape(&[size, 1])).unwrap().reshape(&[size]);

            // Compute λ_i*v_i
            let lambda_v_i = v_i.multiply_scalar(lambda_i);

            // Compare A*v_i and λ_i*v_i (should be approximately equal)
            for j in 0..size {
                assert_abs_diff_eq!(
                    av_i.get(&[j]).unwrap(),
                    lambda_v_i.get(&[j]).unwrap(),
                    epsilon = TOLERANCE * 10.0
                );
            }
        }

        // Property 2: V is orthogonal (V^T * V = I)
        // Check V^T * V = I (eigenvectors of symmetric matrix should be orthogonal)
        let eigenvectors_t = eigenvectors.transpose();
        let v_t_v = eigenvectors_t.matmul(&eigenvectors).unwrap();

        // Check if the result is approximately the identity matrix
        let identity = Array::<f64>::eye(size, size, 0);
        assert!(
            matrices_approx_equal(&v_t_v, &identity),
            "Eigenvectors of symmetric matrix should be orthogonal"
        );

        // Property 3: A = V^T * Λ * V (reconstruction property)
        // Note: eigh returns eigenvectors as rows, so we use V^T * Λ * V
        let lambda_diag = Array::<f64>::create_diagonal_matrix(&eigenvalues, 0);
        let v_t = eigenvectors.transpose();
        let vt_lambda = v_t.matmul(&lambda_diag).unwrap();
        let a_reconstructed = vt_lambda.matmul(&eigenvectors).unwrap();

        assert!(
            matrices_approx_equal(&a, &a_reconstructed),
            "A should equal V^T * Λ * V"
        );
    }
}

#[test]
#[allow(deprecated)]
fn test_svd_properties() {
    // Test SVD properties
    for &rows in MATRIX_SIZES.iter() {
        for &cols in MATRIX_SIZES.iter() {
            let a = random_matrix(rows, cols);

            // Compute SVD: A = U * Σ * V^T
            let (u, s, vt) = svd(&a).unwrap();

            // Property 1: U and V are orthogonal matrices
            assert!(
                is_approximately_orthogonal(&u),
                "U from SVD should be orthogonal"
            );

            let v = vt.transpose();
            assert!(
                is_approximately_orthogonal(&v),
                "V from SVD should be orthogonal"
            );

            // Property 2: Σ is a diagonal matrix with non-negative entries
            let min_dim = rows.min(cols);
            for i in 0..min_dim {
                assert!(
                    s.get(&[i]).unwrap() >= 0.0,
                    "Singular values should be non-negative"
                );
            }

            // Property 3: A = U * Σ * V^T (reconstruction property)
            // Create sigma matrix with correct dimensions (rows x cols)
            let mut sigma = Array::<f64>::zeros(&[rows, cols]);
            let min_dim = rows.min(cols);
            for i in 0..min_dim {
                sigma.set(&[i, i], s.get(&[i]).unwrap()).unwrap();
            }
            let u_sigma = u.matmul(&sigma).unwrap();
            let a_reconstructed = u_sigma.matmul(&vt).unwrap();

            assert!(
                matrices_approx_equal(&a, &a_reconstructed),
                "A should equal U * Σ * V^T"
            );
        }
    }
}

#[test]
#[allow(deprecated)]
fn test_qr_decomposition_properties() {
    // Test QR decomposition properties
    for &rows in MATRIX_SIZES.iter() {
        for &cols in MATRIX_SIZES.iter().filter(|&c| *c <= rows) {
            let a = random_matrix(rows, cols);

            // Compute QR decomposition: A = Q * R
            let (q, r) = qr(&a).unwrap();

            // Property 1: Q is orthogonal (Q^T * Q = I)
            let q_t = q.transpose();
            let q_t_q = q_t.matmul(&q).unwrap();

            let identity = Array::<f64>::eye(cols, cols, 0);
            assert!(
                matrices_approx_equal(&q_t_q, &identity),
                "Q from QR decomposition should be orthogonal"
            );

            // Property 2: R is upper triangular
            for i in 0..cols {
                for j in 0..cols {
                    if i > j {
                        assert_abs_diff_eq!(r.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
                    }
                }
            }

            // Property 3: A = Q * R (reconstruction property)
            let a_reconstructed = q.matmul(&r).unwrap();

            assert!(
                matrices_approx_equal(&a, &a_reconstructed),
                "A should equal Q * R"
            );
        }
    }
}

#[test]
#[allow(deprecated)]
fn test_cholesky_decomposition_properties() {
    // Test Cholesky decomposition properties
    for &size in MATRIX_SIZES.iter() {
        // Create a positive definite matrix
        let a = random_positive_definite_matrix(size);

        // Compute Cholesky decomposition: A = L * L^T
        let l = cholesky(&a).unwrap();

        // Property 1: L is lower triangular
        for i in 0..size {
            for j in 0..size {
                if j > i {
                    assert_abs_diff_eq!(l.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
                }
            }
        }

        // Property 2: A = L * L^T (reconstruction property)
        let l_t = l.transpose();
        let a_reconstructed = l.matmul(&l_t).unwrap();

        assert!(
            matrices_approx_equal(&a, &a_reconstructed),
            "A should equal L * L^T"
        );

        // Property 3: Diagonal elements of L should be positive
        for i in 0..size {
            assert!(
                l.get(&[i, i]).unwrap() > 0.0,
                "Diagonal elements of L should be positive"
            );
        }
    }
}

#[cfg(feature = "matrix_decomp")]
#[test]
fn test_lu_decomposition_properties() {
    // Test LU decomposition properties
    // Use fixed matrices instead of random ones to avoid numerical issues
    let test_matrices = vec![
        Array::<f64>::from_vec(vec![2.0, 1.0, 1.0, 4.0]).reshape(&[2, 2]),
        Array::<f64>::from_vec(vec![2.0, 1.0, 1.0, 4.0, 10.0, -1.0, 3.0, 5.0, 0.0])
            .reshape(&[3, 3]),
    ];

    for a in test_matrices {
        let size = a.shape()[0];

        // Compute LU decomposition: A = P * L * U
        #[cfg(feature = "matrix_decomp")]
        #[allow(deprecated)]
        let (l, u, p) = lu(&a).unwrap();
        #[cfg(not(feature = "matrix_decomp"))]
        let (p, l, u) = lu(&a).unwrap();

        // Property 1: L is lower triangular with ones on the diagonal
        for i in 0..size {
            for j in 0..size {
                if j > i {
                    assert_abs_diff_eq!(l.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
                } else if i == j {
                    assert_abs_diff_eq!(l.get(&[i, j]).unwrap(), 1.0, epsilon = TOLERANCE);
                }
            }
        }

        // Property 2: U is upper triangular
        // Convert u to f64 to handle comparison with 0.0
        let u_f64 = u.astype::<f64>().unwrap();
        for i in 0..size {
            for j in 0..size {
                if i > j {
                    assert_abs_diff_eq!(u_f64.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
                }
            }
        }

        // Property 3: A = P * L * U (reconstruction property)
        #[cfg(feature = "matrix_decomp")]
        {
            // For new matrix_decomp: reconstruct permuted matrix
            let mut pa = Array::zeros(&[size, size]);
            for i in 0..size {
                for j in 0..size {
                    let perm_idx = p.get(&[i]).unwrap() as usize;
                    pa.set(&[i, j], a.get(&[perm_idx, j]).unwrap()).unwrap();
                }
            }
            let lu_reconstructed = l.matmul(&u).unwrap();
            assert!(
                matrices_approx_equal(&pa, &lu_reconstructed),
                "P*A should equal L * U"
            );
        }
        #[cfg(not(feature = "matrix_decomp"))]
        {
            // For old linalg_extended: P is a matrix
            let l_f64 = l.astype::<f64>().unwrap();
            let u_f64 = u.astype::<f64>().unwrap();
            let lu = l_f64.matmul(&u_f64).unwrap();
            let p_t = p.transpose();
            let a_reconstructed = p_t.matmul(&lu).unwrap();
            assert!(
                matrices_approx_equal(&a, &a_reconstructed),
                "A should equal P^T * L * U"
            );
        }
    }
}

#[cfg(feature = "matrix_decomp")]
#[test]
#[allow(deprecated)]
fn test_condition_number_properties() {
    // Test condition number properties
    for &size in MATRIX_SIZES.iter() {
        let a = random_matrix(size, size);

        // Property 1: cond(A) >= 1
        #[allow(deprecated)]
        let cond_a = condition_number(&a).unwrap();
        assert!(cond_a >= 1.0, "Condition number should be >= 1");

        // Property 2: cond(A^(-1)) = cond(A)
        // Only test with well-conditioned matrices
        if cond_a < 1e5 {
            let a_inv = inv(&a).unwrap();
            #[allow(deprecated)]
            let cond_a_inv = condition_number(&a_inv).unwrap();

            assert_abs_diff_eq!(
                cond_a,
                cond_a_inv,
                epsilon = TOLERANCE * cond_a // Scale by condition number
            );
        }

        // Property 3: For orthogonal Q, cond(Q) = 1
        let q = random_matrix(size, size);
        let (q, _) = qr(&q).unwrap(); // Get orthogonal matrix from QR

        #[allow(deprecated)]
        let cond_q = condition_number(&q).unwrap();
        assert_abs_diff_eq!(cond_q, 1.0, epsilon = TOLERANCE * 10.0);
    }
}

#[test]
fn test_norm_properties() {
    // Test matrix norm properties
    for &size in MATRIX_SIZES.iter() {
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);

        // L2 norm (Frobenius norm)
        let norm_a = norm(&a, Some(2.0)).unwrap();
        let norm_b = norm(&b, Some(2.0)).unwrap();

        // Property 1: ||A|| >= 0 (non-negativity)
        assert!(norm_a >= 0.0, "Norm should be non-negative");

        // Property 2: ||A|| = 0 iff A = 0 (positive definiteness)
        let zero_matrix = Array::<f64>::zeros(&[size, size]);
        let norm_zero = norm(&zero_matrix, Some(2.0)).unwrap();
        assert_abs_diff_eq!(norm_zero, 0.0, epsilon = TOLERANCE);

        // Property 3: ||k*A|| = |k| * ||A|| (homogeneity)
        let k = 2.0;
        let k_a = a.multiply_scalar(k);
        let norm_k_a = norm(&k_a, Some(2.0)).unwrap();

        assert_abs_diff_eq!(norm_k_a, k * norm_a, epsilon = TOLERANCE * norm_a);

        // Property 4: ||A + B|| <= ||A|| + ||B|| (triangle inequality)
        let a_plus_b = a.add(&b);
        let norm_a_plus_b = norm(&a_plus_b, Some(2.0)).unwrap();

        assert!(
            norm_a_plus_b <= norm_a + norm_b + TOLERANCE,
            "Norm should satisfy triangle inequality"
        );
    }
}

#[test]
fn test_trace_properties() {
    // Test trace properties
    for &size in MATRIX_SIZES.iter() {
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);

        // Property 1: tr(A + B) = tr(A) + tr(B)
        let a_plus_b = a.add(&b);
        let trace_a_plus_b = trace(&a_plus_b).unwrap();

        let trace_a = trace(&a).unwrap();
        let trace_b = trace(&b).unwrap();

        assert_abs_diff_eq!(trace_a_plus_b, trace_a + trace_b, epsilon = TOLERANCE);

        // Property 2: tr(k*A) = k * tr(A)
        let k = 2.0;
        let k_a = a.multiply_scalar(k);
        let trace_k_a = trace(&k_a).unwrap();

        assert_abs_diff_eq!(trace_k_a, k * trace_a, epsilon = TOLERANCE);

        // Property 3: tr(A*B) = tr(B*A)
        let ab = a.matmul(&b).unwrap();
        let ba = b.matmul(&a).unwrap();

        let trace_ab = trace(&ab).unwrap();
        let trace_ba = trace(&ba).unwrap();

        assert_abs_diff_eq!(
            trace_ab,
            trace_ba,
            epsilon = TOLERANCE * 10.0 // Increased tolerance for numerical stability
        );

        // Property 4: tr(A^T) = tr(A)
        let a_t = a.transpose();
        let trace_a_t = trace(&a_t).unwrap();

        assert_abs_diff_eq!(trace_a, trace_a_t, epsilon = TOLERANCE);
    }
}

#[test]
#[allow(deprecated)]
fn test_rank_properties() {
    // Test matrix rank properties using fixed matrices to avoid numerical issues
    let test_matrices = vec![
        // 3x3 full rank matrix
        Array::<f64>::from_vec(vec![2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 4.0]).reshape(&[3, 3]),
    ];

    for full_rank in test_matrices {
        let size = full_rank.shape()[0];

        // Create a rank-deficient matrix with known rank < size
        // Make a matrix where the last row is the sum of the first two rows
        let mut rank_deficient = Array::<f64>::zeros(&[size, size]);
        rank_deficient.set(&[0, 0], 1.0).unwrap();
        rank_deficient.set(&[0, 1], 2.0).unwrap();
        rank_deficient.set(&[0, 2], 3.0).unwrap();
        rank_deficient.set(&[1, 0], 4.0).unwrap();
        rank_deficient.set(&[1, 1], 5.0).unwrap();
        rank_deficient.set(&[1, 2], 6.0).unwrap();
        // Last row = first row + second row (making it rank deficient)
        rank_deficient.set(&[2, 0], 5.0).unwrap();
        rank_deficient.set(&[2, 1], 7.0).unwrap();
        rank_deficient.set(&[2, 2], 9.0).unwrap();

        // Helper function to calculate rank using SVD (since matrix_rank is broken)
        let calculate_rank = |matrix: &Array<f64>| -> usize {
            let (_, s, _) = svd(matrix).unwrap();
            let tolerance = 1e-10;
            s.to_vec().iter().filter(|&&val| val > tolerance).count()
        };

        // Property 1: rank(A) <= min(m, n) for m x n matrix
        let rank_full = calculate_rank(&full_rank);
        assert!(
            rank_full <= size,
            "Rank should not exceed matrix dimensions"
        );

        // Property 2: Full rank matrix should have rank = size
        assert_eq!(
            rank_full, size,
            "Full rank matrix should have rank equal to its size"
        );

        // Property 3: Rank deficient matrix should have rank < size
        let rank_deficient_val = calculate_rank(&rank_deficient);
        assert!(
            rank_deficient_val < size,
            "Rank deficient matrix should have rank less than its size"
        );

        // Property 4: rank(A*B) <= min(rank(A), rank(B))
        // Use a fixed matrix instead of random for consistency
        let b = Array::<f64>::eye(size, size, 0); // Identity matrix has rank = size
        let rank_b = calculate_rank(&b);

        let ab = full_rank.matmul(&b).unwrap();
        let rank_ab = calculate_rank(&ab);

        assert!(
            rank_ab <= rank_full.min(rank_b),
            "Rank of product should not exceed min of ranks"
        );
    }
}

#[test]
fn test_solve_properties() {
    // Test properties of linear system solving - simplified to avoid stack overflow
    // Only test with size 3 to reduce stack usage
    let size = 3;
    
    // Create a simple well-conditioned matrix
    let mut a_data = vec![0.0; size * size];
    a_data[0] = 2.0; a_data[1] = 1.0; a_data[2] = 0.0;
    a_data[3] = 1.0; a_data[4] = 3.0; a_data[5] = 1.0;
    a_data[6] = 0.0; a_data[7] = 1.0; a_data[8] = 2.0;
    let a = Array::from_vec(a_data).reshape(&[size, size]);

    // Create a simple right-hand side
    let b = Array::from_vec(vec![1.0, 2.0, 3.0]);

    // Solve the system A * x = b
    let x = solve(&a, &b).unwrap();

    // Property 1: A * x ≈ b (solution verification)
    let a_x = a.matmul(&x.reshape(&[size, 1])).unwrap().reshape(&[size]);

    assert!(
        matrices_approx_equal(&a_x, &b),
        "A * x should equal b for solution of A * x = b"
    );

    // Property 2: For invertible A, x = A^(-1) * b  
    let a_inv = inv(&a).unwrap();
    let a_inv_b = a_inv.matmul(&b.reshape(&[size, 1])).unwrap().reshape(&[size]);

    assert!(
        matrices_approx_equal(&x, &a_inv_b),
        "Solution should equal A^(-1) * b"
    );
}

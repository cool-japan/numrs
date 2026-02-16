//! Tests for symbolic linear algebra operations

use numrs2::symbolic::linalg::*;
use numrs2::symbolic::*;
use std::collections::HashMap;

#[test]
fn test_symbolic_matrix_creation() {
    let x = Expr::var("x");
    let data = vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(0.0), x.clone()],
    ];

    let mat = SymbolicMatrix::from_vec(data).expect("matrix creation failed");
    assert_eq!(mat.nrows(), 2);
    assert_eq!(mat.ncols(), 2);
}

#[test]
fn test_identity_matrix() {
    let id = SymbolicMatrix::identity(3);
    assert_eq!(id.nrows(), 3);
    assert_eq!(id.ncols(), 3);

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(*id.get(i, j).expect("get failed"), Expr::constant(1.0));
            } else {
                assert_eq!(*id.get(i, j).expect("get failed"), Expr::constant(0.0));
            }
        }
    }
}

#[test]
fn test_matrix_evaluation() {
    let x = Expr::var("x");
    let data = vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(2.0), x.clone()],
    ];

    let mat = SymbolicMatrix::from_vec(data).expect("matrix creation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = mat.eval(&vars).expect("evaluation failed");
    assert_eq!(result[[0, 0]], 3.0);
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[1, 0]], 2.0);
    assert_eq!(result[[1, 1]], 3.0);
}

#[test]
fn test_matrix_addition() {
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

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = c.eval(&vars).expect("evaluation failed");
    // [[2, 1], [2, 2]] + [[1, 2], [2, 2]] = [[3, 3], [4, 4]]
    assert_eq!(result[[0, 0]], 3.0);
    assert_eq!(result[[0, 1]], 3.0);
    assert_eq!(result[[1, 0]], 4.0);
    assert_eq!(result[[1, 1]], 4.0);
}

#[test]
fn test_matrix_subtraction() {
    let x = Expr::var("x");
    let a = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(2.0)],
        vec![Expr::constant(3.0), x.clone()],
    ])
    .expect("matrix creation failed");

    let b = SymbolicMatrix::from_vec(vec![
        vec![Expr::constant(1.0), Expr::constant(1.0)],
        vec![Expr::constant(1.0), Expr::constant(1.0)],
    ])
    .expect("matrix creation failed");

    let c = matrix_sub(&a, &b).expect("subtraction failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = c.eval(&vars).expect("evaluation failed");
    // [[3, 2], [3, 3]] - [[1, 1], [1, 1]] = [[2, 1], [2, 2]]
    assert_eq!(result[[0, 0]], 2.0);
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[1, 0]], 2.0);
    assert_eq!(result[[1, 1]], 2.0);
}

#[test]
fn test_matrix_multiplication() {
    let x = Expr::var("x");
    let a = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(0.0)],
        vec![Expr::constant(0.0), x.clone()],
    ])
    .expect("matrix creation failed");

    let b = a.clone();
    let c = matrix_mul(&a, &b).expect("multiplication failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = c.eval(&vars).expect("evaluation failed");
    // [[2, 0], [0, 2]] * [[2, 0], [0, 2]] = [[4, 0], [0, 4]]
    assert_eq!(result[[0, 0]], 4.0);
    assert_eq!(result[[0, 1]], 0.0);
    assert_eq!(result[[1, 0]], 0.0);
    assert_eq!(result[[1, 1]], 4.0);
}

#[test]
fn test_matrix_transpose() {
    let x = Expr::var("x");
    let mat = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(2.0), Expr::constant(3.0)],
    ])
    .expect("matrix creation failed");

    let trans = transpose(&mat);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let result = trans.eval(&vars).expect("evaluation failed");
    // [[5, 1], [2, 3]]^T = [[5, 2], [1, 3]]
    assert_eq!(result[[0, 0]], 5.0);
    assert_eq!(result[[0, 1]], 2.0);
    assert_eq!(result[[1, 0]], 1.0);
    assert_eq!(result[[1, 1]], 3.0);
}

#[test]
fn test_matrix_trace() {
    let x = Expr::var("x");
    let mat = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(2.0), x.clone()],
    ])
    .expect("matrix creation failed");

    let tr = trace(&mat).expect("trace computation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = tr.eval(&vars).expect("evaluation failed");
    // trace = x + x = 2x, at x=3: 6
    assert_eq!(result, 6.0);
}

#[test]
fn test_determinant_2x2_constant() {
    let mat = SymbolicMatrix::from_vec(vec![
        vec![Expr::constant(1.0), Expr::constant(2.0)],
        vec![Expr::constant(3.0), Expr::constant(4.0)],
    ])
    .expect("matrix creation failed");

    let det = determinant(&mat).expect("determinant computation failed");

    let vars = HashMap::new();
    let result = det.eval(&vars).expect("evaluation failed");
    // det = 1*4 - 2*3 = -2
    assert_eq!(result, -2.0);
}

#[test]
fn test_determinant_2x2_symbolic() {
    let x = Expr::var("x");
    let mat = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(1.0), x.clone()],
    ])
    .expect("matrix creation failed");

    let det = determinant(&mat).expect("determinant computation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = det.eval(&vars).expect("evaluation failed");
    // det = x*x - 1*1 = x² - 1, at x=2: 3
    assert_eq!(result, 3.0);
}

#[test]
fn test_determinant_3x3() {
    let mat = SymbolicMatrix::from_vec(vec![
        vec![
            Expr::constant(1.0),
            Expr::constant(2.0),
            Expr::constant(3.0),
        ],
        vec![
            Expr::constant(0.0),
            Expr::constant(1.0),
            Expr::constant(4.0),
        ],
        vec![
            Expr::constant(5.0),
            Expr::constant(6.0),
            Expr::constant(0.0),
        ],
    ])
    .expect("matrix creation failed");

    let det = determinant(&mat).expect("determinant computation failed");

    let vars = HashMap::new();
    let result = det.eval(&vars).expect("evaluation failed");
    // Calculate expected determinant manually
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    assert_eq!(result, 1.0);
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

    let vars = HashMap::new();
    let result = product.eval(&vars).expect("evaluation failed");

    // Check it's close to identity
    assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
    assert!(result[[0, 1]].abs() < 1e-10);
    assert!(result[[1, 0]].abs() < 1e-10);
}

#[test]
fn test_inverse_symbolic() {
    let x = Expr::var("x");
    let mat = SymbolicMatrix::from_vec(vec![
        vec![x.clone(), Expr::constant(0.0)],
        vec![Expr::constant(0.0), x.clone()],
    ])
    .expect("matrix creation failed");

    let inv = inverse(&mat).expect("inverse computation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = inv.eval(&vars).expect("evaluation failed");
    // inv([[2, 0], [0, 2]]) = [[0.5, 0], [0, 0.5]]
    assert_eq!(result[[0, 0]], 0.5);
    assert_eq!(result[[0, 1]], 0.0);
    assert_eq!(result[[1, 0]], 0.0);
    assert_eq!(result[[1, 1]], 0.5);
}

#[test]
fn test_solve_2x2() {
    use scirs2_core::ndarray::Array1;

    let mat = SymbolicMatrix::from_vec(vec![
        vec![Expr::constant(2.0), Expr::constant(1.0)],
        vec![Expr::constant(1.0), Expr::constant(2.0)],
    ])
    .expect("matrix creation failed");

    let b = Array1::from_vec(vec![Expr::constant(3.0), Expr::constant(3.0)]);

    let x = solve(&mat, &b).expect("solve failed");

    let vars = HashMap::new();
    let x0 = x[0].eval(&vars).expect("eval failed");
    let x1 = x[1].eval(&vars).expect("eval failed");

    // Solve [[2, 1], [1, 2]] * [x, y] = [3, 3]
    // Solution: x = 1, y = 1
    assert!((x0 - 1.0).abs() < 1e-10);
    assert!((x1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_dimension_mismatch_add() {
    let a = SymbolicMatrix::zeros(2, 2);
    let b = SymbolicMatrix::zeros(2, 3);

    let result = matrix_add(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_dimension_mismatch_mul() {
    let a = SymbolicMatrix::zeros(2, 3);
    let b = SymbolicMatrix::zeros(2, 2);

    let result = matrix_mul(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_trace_non_square() {
    let mat = SymbolicMatrix::zeros(2, 3);
    let result = trace(&mat);
    assert!(result.is_err());
}

#[test]
fn test_determinant_non_square() {
    let mat = SymbolicMatrix::zeros(2, 3);
    let result = determinant(&mat);
    assert!(result.is_err());
}

#[test]
fn test_matrix_simplification() {
    let x = Expr::var("x");
    let data = vec![
        vec![x.clone() + 0.0, x.clone() * 1.0],
        vec![x.clone() * 0.0, x.clone().pow(1.0)],
    ];

    let mat = SymbolicMatrix::from_vec(data).expect("matrix creation failed");
    let simplified = mat.simplify();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let result = simplified.eval(&vars).expect("evaluation failed");
    // After simplification: [[x, x], [0, x]]
    assert_eq!(result[[0, 0]], 5.0);
    assert_eq!(result[[0, 1]], 5.0);
    assert_eq!(result[[1, 0]], 0.0);
    assert_eq!(result[[1, 1]], 5.0);
}

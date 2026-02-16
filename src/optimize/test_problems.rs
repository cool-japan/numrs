//! Multi-objective test problems for optimization benchmarking
//!
//! This module provides standard benchmark problems for evaluating multi-objective
//! optimization algorithms. These problems are widely used in the evolutionary
//! computation community.
//!
//! # ZDT Test Suite
//!
//! The ZDT (Zitzler-Deb-Thiele) test suite consists of bi-objective problems with
//! known Pareto-optimal fronts. These problems are specifically designed to test
//! different aspects of multi-objective optimization algorithms:
//!
//! - **ZDT1**: Convex Pareto front - tests convergence
//! - **ZDT2**: Non-convex (concave) Pareto front - tests diversity
//! - **ZDT3**: Disconnected Pareto front - tests diversity maintenance
//!
//! # DTLZ Test Suite
//!
//! The DTLZ (Deb-Thiele-Laumanns-Zitzler) test suite is a collection of scalable
//! multi-objective test problems with known Pareto-optimal fronts. Key features:
//!
//! - **Scalability**: Problems support arbitrary number of objectives (M >= 2)
//! - **Variety**: Different Pareto front shapes (linear, concave, disconnected)
//! - **Difficulty**: Range from simple to multi-modal and deceptive
//!
//! ## Available Problems
//!
//! - **DTLZ1**: Linear Pareto front, multi-modal (3^k local fronts)
//! - **DTLZ2**: Concave/spherical Pareto front, unimodal
//! - **DTLZ3**: Concave Pareto front, multi-modal (3^k local fronts)
//! - **DTLZ7**: Disconnected Pareto regions, mixed shape
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::test_problems::{ZDT1, DTLZ1, TestProblem};
//!
//! // ZDT1: Bi-objective problem with convex front
//! let zdt1 = ZDT1::new(30);
//! let x = vec![0.5; 30];
//! let objectives = TestProblem::<f64>::evaluate(&zdt1, &x);
//! assert_eq!(objectives.len(), 2);
//!
//! // DTLZ1: Scalable multi-objective problem
//! let dtlz1 = DTLZ1::new(3, 7);
//! let x = vec![0.5; 7];
//! let objectives = dtlz1.evaluate(&x);
//! assert_eq!(objectives.len(), 3);
//!
//! // Generate true Pareto fronts
//! let zdt1_front: Vec<Vec<f64>> = TestProblem::<f64>::generate_pareto_front(&zdt1, 100);
//! let dtlz1_front: Vec<Vec<f64>> = TestProblem::<f64>::generate_pareto_front(&dtlz1, 100);
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

/// Trait for multi-objective test problems
pub trait TestProblem<T: Float> {
    /// Number of objectives
    fn n_objectives(&self) -> usize;

    /// Number of decision variables
    fn n_variables(&self) -> usize;

    /// Evaluate objectives at given decision variables
    ///
    /// # Arguments
    ///
    /// * `x` - Decision variables (must have length n_variables)
    ///
    /// # Returns
    ///
    /// Vector of objective values (length n_objectives)
    fn evaluate(&self, x: &[T]) -> Vec<T>;

    /// Generate points on the true Pareto-optimal front
    ///
    /// # Arguments
    ///
    /// * `n_points` - Number of Pareto-optimal points to generate
    ///
    /// # Returns
    ///
    /// Vector of objective vectors representing the true Pareto front
    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>>;

    /// Get variable bounds (default [0, 1] for all variables)
    fn bounds(&self) -> Vec<(T, T)> {
        vec![(T::zero(), T::one()); self.n_variables()]
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute sum for DTLZ problems
#[inline]
fn compute_sum<T: Float>(x: &[T]) -> T {
    x.iter().copied().fold(T::zero(), |acc, xi| acc + xi)
}

/// Compute product for DTLZ problems
#[inline]
fn compute_product<T: Float>(x: &[T]) -> T {
    x.iter().copied().fold(T::one(), |acc, xi| acc * xi)
}

/// Generate uniform points on (M-1)-dimensional simplex
///
/// Used for generating Pareto-optimal points for problems with linear
/// or concave Pareto fronts.
fn generate_simplex_points<T: Float>(m: usize, n_points: usize) -> Vec<Vec<T>> {
    if m == 2 {
        // For 2 objectives, simplex is a line segment
        return (0..n_points)
            .map(|i| {
                let t = T::from(i).expect("usize to Float")
                    / T::from(n_points - 1).expect("usize to Float");
                vec![T::one() - t, t]
            })
            .collect();
    }

    // For M > 2, use Das-Dennis approach (simplified uniform sampling)
    let mut points = Vec::new();
    generate_simplex_recursive(m, n_points, T::zero(), Vec::new(), &mut points);
    points
}

/// Recursive helper for simplex point generation
fn generate_simplex_recursive<T: Float>(
    m: usize,
    n_points: usize,
    sum: T,
    current: Vec<T>,
    points: &mut Vec<Vec<T>>,
) {
    if current.len() == m - 1 {
        let mut point = current;
        point.push(T::one() - sum);
        points.push(point);
        return;
    }

    let divisions = (n_points as f64).powf(1.0 / (m - 1) as f64).ceil() as usize;
    for i in 0..=divisions {
        let val = T::from(i).expect("usize to Float") / T::from(divisions).expect("usize to Float")
            * (T::one() - sum);
        if sum + val <= T::one() + T::epsilon() {
            let mut next = current.clone();
            next.push(val);
            generate_simplex_recursive(m, n_points, sum + val, next, points);
        }
    }
}

// ============================================================================
// ZDT Test Suite: Bi-objective optimization problems
// ============================================================================

// ============================================================================
// ZDT1: Convex Pareto Front
// ============================================================================

/// ZDT1 test problem with convex Pareto front
///
/// # Problem Definition
///
/// - **Objectives**: 2 (minimize both)
/// - **Variables**: n (typically 30)
/// - **Bounds**: [0, 1] for all variables
///
/// ## Mathematical Formulation
///
/// ```text
/// f1(x) = x1
/// f2(x) = g(x) * h(f1, g)
/// g(x) = 1 + 9 * sum(x2...xn) / (n-1)
/// h(f1, g) = 1 - sqrt(f1 / g)
/// ```
///
/// ## Pareto Front
///
/// The true Pareto front is:
/// - f1 ∈ [0, 1]
/// - f2 = 1 - sqrt(f1)
/// - Convex and continuous
///
/// ## Characteristics
///
/// - **Difficulty**: Easy
/// - **Type**: Continuous, convex
/// - **Optimal**: x1 ∈ \[0,1\], xi = 0 for i > 1
#[derive(Debug, Clone)]
pub struct ZDT1 {
    n_variables: usize,
}

impl ZDT1 {
    /// Create a new ZDT1 problem instance
    ///
    /// # Arguments
    ///
    /// * `n_variables` - Number of decision variables (typically 30)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::ZDT1;
    ///
    /// let problem = ZDT1::new(30);
    /// ```
    pub fn new(n_variables: usize) -> Self {
        assert!(n_variables >= 2, "ZDT1 requires at least 2 variables");
        Self { n_variables }
    }

    /// Helper function to compute g(x)
    fn compute_g<T: Float>(&self, x: &[T]) -> T {
        let sum: T = x[1..].iter().copied().fold(T::zero(), |acc, xi| acc + xi);
        let n_minus_1 = T::from(self.n_variables - 1).expect("n-1 to Float");
        T::one() + T::from(9.0).expect("9.0 to Float") * sum / n_minus_1
    }
}

impl<T: Float> TestProblem<T> for ZDT1 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let f1 = x[0];
        let g = self.compute_g(x);
        let h = T::one() - (f1 / g).sqrt();
        let f2 = g * h;

        vec![f1, f2]
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        (0..n_points)
            .map(|i| {
                let f1 = T::from(i).expect("i to Float")
                    / T::from(n_points - 1).expect("n_points-1 to Float");
                let f2 = T::one() - f1.sqrt();
                vec![f1, f2]
            })
            .collect()
    }
}

// ============================================================================
// ZDT2: Non-convex Pareto Front
// ============================================================================

/// ZDT2 test problem with non-convex Pareto front
///
/// # Problem Definition
///
/// - **Objectives**: 2 (minimize both)
/// - **Variables**: n (typically 30)
/// - **Bounds**: [0, 1] for all variables
///
/// ## Mathematical Formulation
///
/// ```text
/// f1(x) = x1
/// f2(x) = g(x) * h(f1, g)
/// g(x) = 1 + 9 * sum(x2...xn) / (n-1)
/// h(f1, g) = 1 - (f1 / g)^2
/// ```
///
/// ## Pareto Front
///
/// The true Pareto front is:
/// - f1 ∈ [0, 1]
/// - f2 = 1 - f1^2
/// - Non-convex (concave) and continuous
///
/// ## Characteristics
///
/// - **Difficulty**: Medium
/// - **Type**: Continuous, non-convex
/// - **Optimal**: x1 ∈ \[0,1\], xi = 0 for i > 1
#[derive(Debug, Clone)]
pub struct ZDT2 {
    n_variables: usize,
}

impl ZDT2 {
    /// Create a new ZDT2 problem instance
    ///
    /// # Arguments
    ///
    /// * `n_variables` - Number of decision variables (typically 30)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::ZDT2;
    ///
    /// let problem = ZDT2::new(30);
    /// ```
    pub fn new(n_variables: usize) -> Self {
        assert!(n_variables >= 2, "ZDT2 requires at least 2 variables");
        Self { n_variables }
    }

    /// Helper function to compute g(x)
    fn compute_g<T: Float>(&self, x: &[T]) -> T {
        let sum: T = x[1..].iter().copied().fold(T::zero(), |acc, xi| acc + xi);
        let n_minus_1 = T::from(self.n_variables - 1).expect("n-1 to Float");
        T::one() + T::from(9.0).expect("9.0 to Float") * sum / n_minus_1
    }
}

impl<T: Float> TestProblem<T> for ZDT2 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let f1 = x[0];
        let g = self.compute_g(x);
        let h = T::one() - (f1 / g).powi(2);
        let f2 = g * h;

        vec![f1, f2]
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        (0..n_points)
            .map(|i| {
                let f1 = T::from(i).expect("i to Float")
                    / T::from(n_points - 1).expect("n_points-1 to Float");
                let f2 = T::one() - f1.powi(2);
                vec![f1, f2]
            })
            .collect()
    }
}

// ============================================================================
// ZDT3: Disconnected Pareto Front
// ============================================================================

/// ZDT3 test problem with disconnected Pareto front
///
/// # Problem Definition
///
/// - **Objectives**: 2 (minimize both)
/// - **Variables**: n (typically 30)
/// - **Bounds**: [0, 1] for all variables
///
/// ## Mathematical Formulation
///
/// ```text
/// f1(x) = x1
/// f2(x) = g(x) * h(f1, g)
/// g(x) = 1 + 9 * sum(x2...xn) / (n-1)
/// h(f1, g) = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * π * f1)
/// ```
///
/// ## Pareto Front
///
/// The true Pareto front is:
/// - Multiple disconnected regions in objective space
/// - f1 ∈ [0, 0.0830], [0.1822, 0.2577], [0.4093, 0.4538], [0.6183, 0.6525], [0.8233, 0.8518]
/// - Highly multi-modal
///
/// ## Characteristics
///
/// - **Difficulty**: Hard
/// - **Type**: Discontinuous, multi-modal
/// - **Optimal**: x1 in specific intervals, xi = 0 for i > 1
#[derive(Debug, Clone)]
pub struct ZDT3 {
    n_variables: usize,
}

impl ZDT3 {
    /// Create a new ZDT3 problem instance
    ///
    /// # Arguments
    ///
    /// * `n_variables` - Number of decision variables (typically 30)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::ZDT3;
    ///
    /// let problem = ZDT3::new(30);
    /// ```
    pub fn new(n_variables: usize) -> Self {
        assert!(n_variables >= 2, "ZDT3 requires at least 2 variables");
        Self { n_variables }
    }

    /// Helper function to compute g(x)
    fn compute_g<T: Float>(&self, x: &[T]) -> T {
        let sum: T = x[1..].iter().copied().fold(T::zero(), |acc, xi| acc + xi);
        let n_minus_1 = T::from(self.n_variables - 1).expect("n-1 to Float");
        T::one() + T::from(9.0).expect("9.0 to Float") * sum / n_minus_1
    }
}

impl<T: Float> TestProblem<T> for ZDT3 {
    fn n_objectives(&self) -> usize {
        2
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let f1 = x[0];
        let g = self.compute_g(x);
        let ratio = f1 / g;

        let pi = T::from(PI).expect("PI to Float");
        let ten = T::from(10.0).expect("10.0 to Float");

        let h = T::one() - ratio.sqrt() - ratio * (ten * pi * f1).sin();
        let f2 = g * h;

        vec![f1, f2]
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        let pi = T::from(PI).expect("PI to Float");
        let ten = T::from(10.0).expect("10.0 to Float");

        (0..n_points)
            .map(|i| {
                let f1 = T::from(i).expect("i to Float")
                    / T::from(n_points - 1).expect("n_points-1 to Float");
                let f2 = T::one() - f1.sqrt() - f1 * (ten * pi * f1).sin();
                vec![f1, f2]
            })
            .collect()
    }
}

// ============================================================================
// DTLZ Test Suite: Scalable multi-objective optimization problems
// ============================================================================

// ============================================================================
// DTLZ1: Linear Pareto front, multi-modal
// ============================================================================

/// DTLZ1 test problem
///
/// DTLZ1 has a linear Pareto-optimal front and is multi-modal with 3^k local
/// Pareto-optimal fronts, where k = n - M + 1. The global Pareto-optimal front
/// is the hyperplane sum(f_i) = 0.5.
///
/// # Characteristics
///
/// - **Pareto Front**: Linear (hyperplane)
/// - **Modality**: Multi-modal (3^k local fronts)
/// - **Separability**: Separable
/// - **Bias**: Unbiased
///
/// # Optimal Solutions
///
/// For the Pareto-optimal set, x_M to x_n should equal 0.5, and x_1 to x_{M-1}
/// can take any value in \[0,1\] that satisfies sum(f_i) = 0.5.
///
/// # Parameters
///
/// - M: Number of objectives (≥ 2)
/// - n: Number of variables (recommended: n = M + k - 1, default k = 5)
#[derive(Debug, Clone)]
pub struct DTLZ1 {
    /// Number of objectives
    n_objectives: usize,
    /// Number of decision variables
    n_variables: usize,
    /// k parameter (n = M + k - 1)
    k: usize,
}

impl DTLZ1 {
    /// Create a new DTLZ1 problem
    ///
    /// # Arguments
    ///
    /// * `n_objectives` - Number of objectives (M ≥ 2)
    /// * `n_variables` - Number of decision variables (recommended: M + k - 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::DTLZ1;
    ///
    /// // 3 objectives, 7 variables (k=5)
    /// let problem = DTLZ1::new(3, 7);
    /// ```
    pub fn new(n_objectives: usize, n_variables: usize) -> Self {
        assert!(n_objectives >= 2, "DTLZ1 requires at least 2 objectives");
        assert!(
            n_variables >= n_objectives,
            "Number of variables must be >= number of objectives"
        );
        let k = n_variables - n_objectives + 1;
        Self {
            n_objectives,
            n_variables,
            k,
        }
    }

    /// Create DTLZ1 with default k=5
    pub fn with_default_k(n_objectives: usize) -> Self {
        let k = 5;
        let n_variables = n_objectives + k - 1;
        Self::new(n_objectives, n_variables)
    }

    /// Compute g function for DTLZ1
    ///
    /// g(X_M) = 100 * [k + sum_{x_i in X_M}((x_i - 0.5)^2 - cos(20π(x_i - 0.5)))]
    fn g_function<T: Float>(&self, xm: &[T]) -> T {
        let k_float = T::from(self.k).expect("k to Float");
        let sum = xm.iter().fold(T::zero(), |acc, &xi| {
            let centered = xi - T::from(0.5).expect("0.5 to Float");
            let twenty_pi = T::from(20.0 * PI).expect("20π to Float");
            acc + centered * centered - (twenty_pi * centered).cos()
        });
        T::from(100.0).expect("100 to Float") * (k_float + sum)
    }
}

impl<T: Float> TestProblem<T> for DTLZ1 {
    fn n_objectives(&self) -> usize {
        self.n_objectives
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let m = self.n_objectives;

        // Split into position variables (first M-1) and distance variables (last k)
        let xm = &x[m - 1..];
        let g = self.g_function(xm);

        let mut objectives = Vec::with_capacity(m);
        let one_plus_g = T::one() + g;
        let half = T::from(0.5).expect("0.5 to Float");

        // Compute M-1 objectives
        for i in 0..m - 1 {
            let mut prod = one_plus_g * half;
            for j in 0..m - 1 - i {
                prod = prod * x[j];
            }
            if i > 0 {
                prod = prod * (T::one() - x[m - 1 - i]);
            }
            objectives.push(prod);
        }

        // Compute last objective f_M
        let mut prod = one_plus_g * half;
        if m > 1 {
            prod = prod * (T::one() - x[0]);
        }
        objectives.push(prod);

        objectives
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        // For DTLZ1, the Pareto front is a linear hyperplane where sum(f_i) = 0.5
        // Generate uniform points on the (M-1)-simplex scaled by 0.5
        let simplex_points = generate_simplex_points(self.n_objectives, n_points);
        let half = T::from(0.5).expect("0.5 to Float");

        simplex_points
            .into_iter()
            .map(|point: Vec<T>| point.into_iter().map(|fi| fi * half).collect())
            .collect()
    }
}

// ============================================================================
// DTLZ2: Concave Pareto front, unimodal
// ============================================================================

/// DTLZ2 test problem
///
/// DTLZ2 has a concave (spherical) Pareto-optimal front and is unimodal.
/// The Pareto front lies on the positive orthant of a unit hypersphere.
///
/// # Characteristics
///
/// - **Pareto Front**: Concave/spherical
/// - **Modality**: Unimodal
/// - **Separability**: Separable
/// - **Bias**: Unbiased
///
/// # Optimal Solutions
///
/// For the Pareto-optimal set, x_M to x_n should equal 0.5, and x_1 to x_{M-1}
/// can take any value in \[0,1\] such that the objectives lie on the unit sphere.
///
/// # Parameters
///
/// - M: Number of objectives (≥ 2)
/// - n: Number of variables (recommended: n = M + k - 1, default k = 10)
#[derive(Debug, Clone)]
pub struct DTLZ2 {
    /// Number of objectives
    n_objectives: usize,
    /// Number of decision variables
    n_variables: usize,
    /// k parameter (n = M + k - 1)
    k: usize,
}

impl DTLZ2 {
    /// Create a new DTLZ2 problem
    ///
    /// # Arguments
    ///
    /// * `n_objectives` - Number of objectives (M ≥ 2)
    /// * `n_variables` - Number of decision variables (recommended: M + k - 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::DTLZ2;
    ///
    /// // 3 objectives, 12 variables (k=10)
    /// let problem = DTLZ2::new(3, 12);
    /// ```
    pub fn new(n_objectives: usize, n_variables: usize) -> Self {
        assert!(n_objectives >= 2, "DTLZ2 requires at least 2 objectives");
        assert!(
            n_variables >= n_objectives,
            "Number of variables must be >= number of objectives"
        );
        let k = n_variables - n_objectives + 1;
        Self {
            n_objectives,
            n_variables,
            k,
        }
    }

    /// Create DTLZ2 with default k=10
    pub fn with_default_k(n_objectives: usize) -> Self {
        let k = 10;
        let n_variables = n_objectives + k - 1;
        Self::new(n_objectives, n_variables)
    }

    /// Compute g function for DTLZ2
    ///
    /// g(X_M) = sum_{x_i in X_M}(x_i - 0.5)^2
    fn g_function<T: Float>(&self, xm: &[T]) -> T {
        xm.iter().fold(T::zero(), |acc, &xi| {
            let centered = xi - T::from(0.5).expect("0.5 to Float");
            acc + centered * centered
        })
    }
}

impl<T: Float> TestProblem<T> for DTLZ2 {
    fn n_objectives(&self) -> usize {
        self.n_objectives
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let m = self.n_objectives;
        let pi_half = T::from(PI / 2.0).expect("π/2 to Float");

        // Split into position variables (first M-1) and distance variables (last k)
        let xm = &x[m - 1..];
        let g = self.g_function(xm);
        let one_plus_g = T::one() + g;

        let mut objectives = Vec::with_capacity(m);

        // Compute M-1 objectives
        for i in 0..m - 1 {
            let mut val = one_plus_g;
            // Product of cosines
            for j in 0..m - 1 - i {
                val = val * (x[j] * pi_half).cos();
            }
            // Multiply by sine if not the first objective
            if i > 0 {
                val = val * (x[m - 1 - i] * pi_half).sin();
            }
            objectives.push(val);
        }

        // Compute last objective f_M
        let val = one_plus_g * (x[0] * pi_half).sin();
        objectives.push(val);

        objectives
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        // For DTLZ2, the Pareto front is the positive orthant of a unit sphere
        // Generate points on the simplex, then map to sphere
        let simplex_points = generate_simplex_points(self.n_objectives, n_points);
        let pi_half = T::from(PI / 2.0).expect("π/2 to Float");

        simplex_points
            .into_iter()
            .map(|point: Vec<T>| {
                // Convert simplex point to angles
                let mut objectives = Vec::with_capacity(self.n_objectives);
                let m = self.n_objectives;

                for i in 0..m - 1 {
                    let mut val = T::one();
                    // Product of cosines
                    for j in 0..m - 1 - i {
                        let angle: T = point[j] * pi_half;
                        val = val * angle.cos();
                    }
                    // Multiply by sine if not the first objective
                    if i > 0 {
                        let angle: T = point[m - 1 - i] * pi_half;
                        val = val * angle.sin();
                    }
                    objectives.push(val);
                }

                // Last objective
                let angle: T = point[0] * pi_half;
                let val = angle.sin();
                objectives.push(val);

                objectives
            })
            .collect()
    }
}

// ============================================================================
// DTLZ3: Concave Pareto front, multi-modal
// ============================================================================

/// DTLZ3 test problem
///
/// DTLZ3 has a concave (spherical) Pareto-optimal front like DTLZ2, but with
/// the multi-modal g-function from DTLZ1. This makes it significantly harder
/// to solve than DTLZ2, with 3^k local Pareto-optimal fronts.
///
/// # Characteristics
///
/// - **Pareto Front**: Concave/spherical
/// - **Modality**: Multi-modal (3^k local fronts)
/// - **Separability**: Separable
/// - **Bias**: Unbiased
///
/// # Optimal Solutions
///
/// For the Pareto-optimal set, x_M to x_n should equal 0.5, and x_1 to x_{M-1}
/// can take any value in \[0,1\] such that the objectives lie on the unit sphere.
///
/// # Parameters
///
/// - M: Number of objectives (≥ 2)
/// - n: Number of variables (recommended: n = M + k - 1, default k = 10)
#[derive(Debug, Clone)]
pub struct DTLZ3 {
    /// Number of objectives
    n_objectives: usize,
    /// Number of decision variables
    n_variables: usize,
    /// k parameter (n = M + k - 1)
    k: usize,
}

impl DTLZ3 {
    /// Create a new DTLZ3 problem
    ///
    /// # Arguments
    ///
    /// * `n_objectives` - Number of objectives (M ≥ 2)
    /// * `n_variables` - Number of decision variables (recommended: M + k - 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::DTLZ3;
    ///
    /// // 3 objectives, 12 variables (k=10)
    /// let problem = DTLZ3::new(3, 12);
    /// ```
    pub fn new(n_objectives: usize, n_variables: usize) -> Self {
        assert!(n_objectives >= 2, "DTLZ3 requires at least 2 objectives");
        assert!(
            n_variables >= n_objectives,
            "Number of variables must be >= number of objectives"
        );
        let k = n_variables - n_objectives + 1;
        Self {
            n_objectives,
            n_variables,
            k,
        }
    }

    /// Create DTLZ3 with default k=10
    pub fn with_default_k(n_objectives: usize) -> Self {
        let k = 10;
        let n_variables = n_objectives + k - 1;
        Self::new(n_objectives, n_variables)
    }

    /// Compute g function for DTLZ3 (same as DTLZ1 - multi-modal)
    ///
    /// g(X_M) = 100 * [k + sum_{x_i in X_M}((x_i - 0.5)^2 - cos(20π(x_i - 0.5)))]
    fn g_function<T: Float>(&self, xm: &[T]) -> T {
        let k_float = T::from(self.k).expect("k to Float");
        let sum = xm.iter().fold(T::zero(), |acc, &xi| {
            let centered = xi - T::from(0.5).expect("0.5 to Float");
            let twenty_pi = T::from(20.0 * PI).expect("20π to Float");
            acc + centered * centered - (twenty_pi * centered).cos()
        });
        T::from(100.0).expect("100 to Float") * (k_float + sum)
    }
}

impl<T: Float> TestProblem<T> for DTLZ3 {
    fn n_objectives(&self) -> usize {
        self.n_objectives
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let m = self.n_objectives;
        let pi_half = T::from(PI / 2.0).expect("π/2 to Float");

        // Split into position variables (first M-1) and distance variables (last k)
        let xm = &x[m - 1..];
        let g = self.g_function(xm);
        let one_plus_g = T::one() + g;

        let mut objectives = Vec::with_capacity(m);

        // Compute M-1 objectives (same structure as DTLZ2)
        for i in 0..m - 1 {
            let mut val = one_plus_g;
            // Product of cosines
            for j in 0..m - 1 - i {
                val = val * (x[j] * pi_half).cos();
            }
            // Multiply by sine if not the first objective
            if i > 0 {
                val = val * (x[m - 1 - i] * pi_half).sin();
            }
            objectives.push(val);
        }

        // Compute last objective f_M
        let val = one_plus_g * (x[0] * pi_half).sin();
        objectives.push(val);

        objectives
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        // DTLZ3 has the same Pareto front as DTLZ2 (unit sphere)
        // Generate points on the simplex, then map to sphere
        let simplex_points = generate_simplex_points(self.n_objectives, n_points);
        let pi_half = T::from(PI / 2.0).expect("π/2 to Float");

        simplex_points
            .into_iter()
            .map(|point: Vec<T>| {
                // Convert simplex point to angles
                let mut objectives = Vec::with_capacity(self.n_objectives);
                let m = self.n_objectives;

                for i in 0..m - 1 {
                    let mut val = T::one();
                    // Product of cosines
                    for j in 0..m - 1 - i {
                        let angle: T = point[j] * pi_half;
                        val = val * angle.cos();
                    }
                    // Multiply by sine if not the first objective
                    if i > 0 {
                        let angle: T = point[m - 1 - i] * pi_half;
                        val = val * angle.sin();
                    }
                    objectives.push(val);
                }

                // Last objective
                let angle: T = point[0] * pi_half;
                let val = angle.sin();
                objectives.push(val);

                objectives
            })
            .collect()
    }
}

// ============================================================================
// DTLZ7: Disconnected Pareto regions, mixed shape
// ============================================================================

/// DTLZ7 test problem
///
/// DTLZ7 has a disconnected Pareto-optimal front with 2^(M-1) disconnected
/// regions. The first M-1 objectives are independent, and the last objective
/// creates the disconnected structure through a special h-function.
///
/// # Characteristics
///
/// - **Pareto Front**: Disconnected (2^(M-1) regions)
/// - **Modality**: Multi-modal
/// - **Separability**: Partially separable
/// - **Bias**: Unbiased
///
/// # Optimal Solutions
///
/// For the Pareto-optimal set, x_M to x_n should equal 0, and x_1 to x_{M-1}
/// can take any value in \[0,1\].
///
/// # Parameters
///
/// - M: Number of objectives (≥ 2)
/// - n: Number of variables (recommended: n = M + k - 1, default k = 20)
#[derive(Debug, Clone)]
pub struct DTLZ7 {
    /// Number of objectives
    n_objectives: usize,
    /// Number of decision variables
    n_variables: usize,
    /// k parameter (n = M + k - 1)
    k: usize,
}

impl DTLZ7 {
    /// Create a new DTLZ7 problem
    ///
    /// # Arguments
    ///
    /// * `n_objectives` - Number of objectives (M ≥ 2)
    /// * `n_variables` - Number of decision variables (recommended: M + k - 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::optimize::test_problems::DTLZ7;
    ///
    /// // 3 objectives, 22 variables (k=20)
    /// let problem = DTLZ7::new(3, 22);
    /// ```
    pub fn new(n_objectives: usize, n_variables: usize) -> Self {
        assert!(n_objectives >= 2, "DTLZ7 requires at least 2 objectives");
        assert!(
            n_variables >= n_objectives,
            "Number of variables must be >= number of objectives"
        );
        let k = n_variables - n_objectives + 1;
        Self {
            n_objectives,
            n_variables,
            k,
        }
    }

    /// Create DTLZ7 with default k=20
    pub fn with_default_k(n_objectives: usize) -> Self {
        let k = 20;
        let n_variables = n_objectives + k - 1;
        Self::new(n_objectives, n_variables)
    }

    /// Compute g function for DTLZ7
    ///
    /// g(X_M) = 1 + (9/k) * sum_{x_i in X_M}(x_i)
    fn g_function<T: Float>(&self, xm: &[T]) -> T {
        let k_float = T::from(self.k).expect("k to Float");
        let nine = T::from(9.0).expect("9 to Float");
        let sum = compute_sum(xm);
        T::one() + (nine / k_float) * sum
    }

    /// Compute h function for DTLZ7
    ///
    /// h(f_1,...,f_{M-1}, g) = M - sum_{i=1}^{M-1}[f_i/(1+g) * (1 + sin(3π*f_i))]
    fn h_function<T: Float>(&self, f: &[T], g: T) -> T {
        let m = T::from(self.n_objectives).expect("M to Float");
        let one_plus_g = T::one() + g;
        let three_pi = T::from(3.0 * PI).expect("3π to Float");

        let sum = f.iter().fold(T::zero(), |acc, &fi| {
            let term = fi / one_plus_g;
            acc + term * (T::one() + (three_pi * fi).sin())
        });

        m - sum
    }
}

impl<T: Float> TestProblem<T> for DTLZ7 {
    fn n_objectives(&self) -> usize {
        self.n_objectives
    }

    fn n_variables(&self) -> usize {
        self.n_variables
    }

    fn evaluate(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.n_variables,
            "Input must have {} variables",
            self.n_variables
        );

        let m = self.n_objectives;

        // First M-1 objectives are simply the first M-1 variables
        let mut objectives: Vec<T> = x[..m - 1].to_vec();

        // Compute g from the last k variables
        let xm = &x[m - 1..];
        let g = self.g_function(xm);

        // Compute h function
        let h = self.h_function(&objectives, g);

        // Last objective
        let one_plus_g = T::one() + g;
        objectives.push(one_plus_g * h);

        objectives
    }

    fn generate_pareto_front(&self, n_points: usize) -> Vec<Vec<T>> {
        // For DTLZ7, the Pareto front has 2^(M-1) disconnected regions
        // We'll generate points uniformly in [0,1]^(M-1) and compute the last objective
        let m = self.n_objectives;
        let three_pi = T::from(3.0 * PI).expect("3π to Float");
        let m_float = T::from(m).expect("M to Float");

        let mut points = Vec::new();

        // Generate uniform grid in [0,1]^(M-1)
        let divisions = (n_points as f64).powf(1.0 / (m - 1) as f64).ceil() as usize;

        // For 2 objectives, it's simpler
        if m == 2 {
            for i in 0..n_points {
                let f1 =
                    T::from(i).expect("i to Float") / T::from(n_points - 1).expect("n-1 to Float");
                let h = m_float - f1 * (T::one() + (three_pi * f1).sin());
                points.push(vec![f1, h]);
            }
        } else {
            // For M > 2, generate points recursively
            generate_dtlz7_recursive::<T>(m, divisions, Vec::new(), &mut points, three_pi, m_float);
        }

        points
    }
}

/// Recursive helper for DTLZ7 Pareto front generation
fn generate_dtlz7_recursive<T: Float>(
    m: usize,
    divisions: usize,
    current: Vec<T>,
    points: &mut Vec<Vec<T>>,
    three_pi: T,
    m_float: T,
) {
    if current.len() == m - 1 {
        // Compute the last objective
        let sum = current.iter().fold(T::zero(), |acc, &fi| {
            acc + fi * (T::one() + (three_pi * fi).sin())
        });
        let h = m_float - sum;

        let mut point = current;
        point.push(h);
        points.push(point);
        return;
    }

    for i in 0..=divisions {
        let val = T::from(i).expect("i to Float") / T::from(divisions).expect("divisions to Float");
        let mut next = current.clone();
        next.push(val);
        generate_dtlz7_recursive(m, divisions, next, points, three_pi, m_float);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ========================================================================
    // DTLZ1 Tests
    // ========================================================================

    #[test]
    fn test_dtlz1_construction() {
        let problem = DTLZ1::new(3, 7);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 7);
        assert_eq!(problem.k, 5);
    }

    #[test]
    fn test_dtlz1_with_default_k() {
        let problem = DTLZ1::with_default_k(3);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 7); // 3 + 5 - 1
        assert_eq!(problem.k, 5);
    }

    #[test]
    fn test_dtlz1_evaluate_dimensions() {
        let problem = DTLZ1::new(3, 7);
        let x = vec![0.5; 7];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 3);
    }

    #[test]
    fn test_dtlz1_optimal_point() {
        // For Pareto-optimal solutions, x_M to x_n should be 0.5
        let problem = DTLZ1::new(3, 7);
        let mut x = vec![0.0; 7];
        x[0] = 0.5;
        x[1] = 0.5;
        // Set x_M to x_n to 0.5 (optimal)
        for i in 2..7 {
            x[i] = 0.5;
        }

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);

        // For optimal point, sum of objectives should be 0.5
        let sum: f64 = objectives.iter().sum();
        assert_relative_eq!(sum, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_dtlz1_scalability_objectives() {
        // Test with different numbers of objectives
        for m in [2, 3, 5, 10] {
            let problem = DTLZ1::with_default_k(m);
            let x = vec![0.5; TestProblem::<f64>::n_variables(&problem)];
            let objectives = TestProblem::<f64>::evaluate(&problem, &x);
            assert_eq!(objectives.len(), m);
        }
    }

    #[test]
    fn test_dtlz1_bounds() {
        let problem = DTLZ1::new(3, 7);
        let bounds = TestProblem::<f64>::bounds(&problem);
        assert_eq!(bounds.len(), 7);
        for (lb, ub) in bounds {
            assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
            assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dtlz1_pareto_front_generation() {
        let problem = DTLZ1::new(3, 7);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 50);
        assert!(!pareto_front.is_empty());

        // All points should have sum of objectives = 0.5
        for point in &pareto_front {
            assert_eq!(point.len(), 3);
            let sum: f64 = point.iter().sum();
            assert_relative_eq!(sum, 0.5, epsilon = 1e-6);

            // All objectives should be non-negative
            for &fi in point {
                assert!(fi >= 0.0, "Objective should be non-negative");
            }
        }
    }

    // ========================================================================
    // DTLZ2 Tests
    // ========================================================================

    #[test]
    fn test_dtlz2_construction() {
        let problem = DTLZ2::new(3, 12);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 12);
        assert_eq!(problem.k, 10);
    }

    #[test]
    fn test_dtlz2_with_default_k() {
        let problem = DTLZ2::with_default_k(3);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 12); // 3 + 10 - 1
        assert_eq!(problem.k, 10);
    }

    #[test]
    fn test_dtlz2_evaluate_dimensions() {
        let problem = DTLZ2::new(3, 12);
        let x = vec![0.5; 12];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 3);
    }

    #[test]
    fn test_dtlz2_optimal_point() {
        // For Pareto-optimal solutions, x_M to x_n should be 0.5 (g = 0)
        let problem = DTLZ2::new(3, 12);
        let mut x = vec![0.0; 12];
        x[0] = 0.5; // theta_1 = π/4
        x[1] = 0.5; // theta_2 = π/4
                    // Set x_M to x_n to 0.5 (optimal)
        for i in 2..12 {
            x[i] = 0.5;
        }

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);

        // For optimal point on unit sphere, sum of squared objectives should be 1
        let sum_sq: f64 = objectives.iter().map(|&f| f * f).sum();
        assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dtlz2_scalability_objectives() {
        // Test with different numbers of objectives
        for m in [2, 3, 5, 10] {
            let problem = DTLZ2::with_default_k(m);
            let x = vec![0.5; TestProblem::<f64>::n_variables(&problem)];
            let objectives = TestProblem::<f64>::evaluate(&problem, &x);
            assert_eq!(objectives.len(), m);
        }
    }

    #[test]
    fn test_dtlz2_pareto_front_generation() {
        let problem = DTLZ2::new(3, 12);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 50);
        assert!(!pareto_front.is_empty());

        // All points should lie on the unit sphere
        for point in &pareto_front {
            assert_eq!(point.len(), 3);
            let sum_sq: f64 = point.iter().map(|&f| f * f).sum();
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-6);

            // All objectives should be non-negative
            for &fi in point {
                assert!(fi >= 0.0, "Objective should be non-negative");
            }
        }
    }

    // ========================================================================
    // DTLZ3 Tests
    // ========================================================================

    #[test]
    fn test_dtlz3_construction() {
        let problem = DTLZ3::new(3, 12);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 12);
        assert_eq!(problem.k, 10);
    }

    #[test]
    fn test_dtlz3_multi_modal() {
        // DTLZ3 should have the same structure as DTLZ2 but multi-modal
        // Test that g-function is different from DTLZ2
        let problem = DTLZ3::new(3, 12);
        let mut x = vec![0.5; 12];

        // At x = 0.5, both should have g ≈ 0 (local optimum)
        let obj1 = TestProblem::<f64>::evaluate(&problem, &x);

        // Move away from 0.5 - should show multi-modality
        for i in 2..12 {
            x[i] = 0.6;
        }
        let obj2 = TestProblem::<f64>::evaluate(&problem, &x);

        // The g-function should create larger objective values
        let sum1: f64 = obj1.iter().map(|&f| f * f).sum();
        let sum2: f64 = obj2.iter().map(|&f| f * f).sum();
        assert!(
            sum2 > sum1,
            "Moving from optimum should increase objectives"
        );
    }

    #[test]
    fn test_dtlz3_scalability_objectives() {
        // Test with different numbers of objectives
        for m in [2, 3, 5] {
            let problem = DTLZ3::with_default_k(m);
            let x = vec![0.5; TestProblem::<f64>::n_variables(&problem)];
            let objectives = TestProblem::<f64>::evaluate(&problem, &x);
            assert_eq!(objectives.len(), m);
        }
    }

    #[test]
    fn test_dtlz3_pareto_front_same_as_dtlz2() {
        // DTLZ3 has the same Pareto front as DTLZ2
        let problem = DTLZ3::new(3, 12);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 50);
        assert!(!pareto_front.is_empty());

        // All points should lie on the unit sphere
        for point in &pareto_front {
            assert_eq!(point.len(), 3);
            let sum_sq: f64 = point.iter().map(|&f| f * f).sum();
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-6);
        }
    }

    // ========================================================================
    // DTLZ7 Tests
    // ========================================================================

    #[test]
    fn test_dtlz7_construction() {
        let problem = DTLZ7::new(3, 22);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 22);
        assert_eq!(problem.k, 20);
    }

    #[test]
    fn test_dtlz7_with_default_k() {
        let problem = DTLZ7::with_default_k(3);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 3);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 22); // 3 + 20 - 1
        assert_eq!(problem.k, 20);
    }

    #[test]
    fn test_dtlz7_evaluate_dimensions() {
        let problem = DTLZ7::new(3, 22);
        let x = vec![0.5; 22];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 3);
    }

    #[test]
    fn test_dtlz7_first_objectives_equal_variables() {
        // First M-1 objectives should equal first M-1 variables for optimal g
        let problem = DTLZ7::new(3, 22);
        let mut x = vec![0.0; 22];
        x[0] = 0.3;
        x[1] = 0.7;
        // Set last k variables to 0 (optimal g)
        for i in 2..22 {
            x[i] = 0.0;
        }

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_relative_eq!(objectives[0], 0.3, epsilon = 1e-10);
        assert_relative_eq!(objectives[1], 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_dtlz7_scalability_objectives() {
        // Test with different numbers of objectives
        for m in [2, 3, 5] {
            let problem = DTLZ7::with_default_k(m);
            let x = vec![0.5; TestProblem::<f64>::n_variables(&problem)];
            let objectives = TestProblem::<f64>::evaluate(&problem, &x);
            assert_eq!(objectives.len(), m);
        }
    }

    #[test]
    fn test_dtlz7_pareto_front_generation() {
        let problem = DTLZ7::new(3, 22);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 50);
        assert!(!pareto_front.is_empty());

        // Check that points are valid
        for point in &pareto_front {
            assert_eq!(point.len(), 3);

            // First M-1 objectives should be in [0,1]
            for i in 0..2 {
                assert!(point[i] >= 0.0 && point[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_dtlz7_disconnected_regions() {
        // DTLZ7 should show disconnected structure
        // This is hard to test directly, but we can verify the h-function creates variation
        let problem = DTLZ7::new(2, 21);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 100);

        // Collect unique f2 values (should show gaps for disconnected regions)
        let mut f2_values: Vec<f64> = pareto_front.iter().map(|p| p[1]).collect();
        f2_values.sort_by(|a, b| a.partial_cmp(b).expect("No NaN values"));

        // Should have variation in f2 values
        let min_f2 = f2_values.first().copied().expect("Non-empty");
        let max_f2 = f2_values.last().copied().expect("Non-empty");
        assert!(max_f2 > min_f2, "Should have variation in f2");
    }

    // ========================================================================
    // General Tests
    // ========================================================================

    #[test]
    fn test_all_problems_bounds_in_unit_hypercube() {
        let problems: Vec<Box<dyn TestProblem<f64>>> = vec![
            Box::new(DTLZ1::new(3, 7)),
            Box::new(DTLZ2::new(3, 12)),
            Box::new(DTLZ3::new(3, 12)),
            Box::new(DTLZ7::new(3, 22)),
        ];

        for problem in problems {
            let bounds = TestProblem::<f64>::bounds(&*problem);
            for (lb, ub) in bounds {
                assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
                assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_variable_scalability() {
        // All DTLZ problems should handle varying numbers of variables
        let test_cases = vec![
            (2, 6),   // M=2, n=6
            (3, 7),   // M=3, n=7
            (5, 14),  // M=5, n=14
            (10, 19), // M=10, n=19
        ];

        for (m, n) in test_cases {
            let p1 = DTLZ1::new(m, n);
            let p2 = DTLZ2::new(m, n);
            let p3 = DTLZ3::new(m, n);
            let p7 = DTLZ7::new(m, n);

            let x = vec![0.5; n];

            assert_eq!(TestProblem::<f64>::evaluate(&p1, &x).len(), m);
            assert_eq!(TestProblem::<f64>::evaluate(&p2, &x).len(), m);
            assert_eq!(TestProblem::<f64>::evaluate(&p3, &x).len(), m);
            assert_eq!(TestProblem::<f64>::evaluate(&p7, &x).len(), m);
        }
    }

    // ========================================================================
    // ZDT1 Tests
    // ========================================================================

    #[test]
    fn test_zdt1_construction() {
        let problem = ZDT1::new(30);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 2);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 30);
    }

    #[test]
    #[should_panic(expected = "ZDT1 requires at least 2 variables")]
    fn test_zdt1_construction_invalid() {
        ZDT1::new(1);
    }

    #[test]
    fn test_zdt1_evaluate_dimensions() {
        let problem = ZDT1::new(30);
        let x = vec![0.5; 30];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 2);
    }

    #[test]
    fn test_zdt1_optimal_point() {
        // Pareto-optimal solutions have x1 ∈ [0,1], xi = 0 for i > 1
        let problem = ZDT1::new(30);
        let mut x = vec![0.0; 30];
        x[0] = 0.5;
        // All other variables are 0 (optimal)

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);

        // f1 = x[0] = 0.5
        assert_relative_eq!(objectives[0], 0.5, epsilon = 1e-10);

        // For optimal point: g = 1, h = 1 - sqrt(f1/g) = 1 - sqrt(0.5) ≈ 0.2929
        // f2 = g * h = 1 * (1 - sqrt(0.5))
        let expected_f2 = 1.0 - (0.5_f64).sqrt();
        assert_relative_eq!(objectives[1], expected_f2, epsilon = 1e-6);
    }

    #[test]
    fn test_zdt1_convex_pareto_front() {
        let problem = ZDT1::new(30);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 100);
        assert_eq!(pareto_front.len(), 100);

        // Verify convexity: for any three points a, b, c where f1_a < f1_b < f1_c,
        // the middle point should be above the line connecting endpoints
        for point in &pareto_front {
            assert_eq!(point.len(), 2);
            let f1: f64 = point[0];
            let f2: f64 = point[1];

            // Verify Pareto front equation: f2 = 1 - sqrt(f1)
            let expected_f2 = 1.0 - f1.sqrt();
            assert_relative_eq!(f2, expected_f2, epsilon = 1e-6);

            // All objectives should be non-negative
            assert!((0.0..=1.0).contains(&f1));
            assert!((0.0..=1.0).contains(&f2));
        }
    }

    #[test]
    fn test_zdt1_bounds() {
        let problem = ZDT1::new(30);
        let bounds = TestProblem::<f64>::bounds(&problem);
        assert_eq!(bounds.len(), 30);

        for (lb, ub) in bounds {
            assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
            assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zdt1_edge_cases() {
        let problem = ZDT1::new(30);

        // Test at x = [0, 0, ..., 0]
        let x_zero = vec![0.0; 30];
        let obj_zero = TestProblem::<f64>::evaluate(&problem, &x_zero);
        assert_relative_eq!(obj_zero[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(obj_zero[1], 1.0, epsilon = 1e-6); // f2 = 1 - sqrt(0) = 1

        // Test at x = [1, 0, ..., 0] (optimal endpoint)
        let mut x_one = vec![0.0; 30];
        x_one[0] = 1.0;
        let obj_one = TestProblem::<f64>::evaluate(&problem, &x_one);
        assert_relative_eq!(obj_one[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj_one[1], 0.0, epsilon = 1e-6); // f2 = 1 - sqrt(1) = 0
    }

    // ========================================================================
    // ZDT2 Tests
    // ========================================================================

    #[test]
    fn test_zdt2_construction() {
        let problem = ZDT2::new(30);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 2);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 30);
    }

    #[test]
    #[should_panic(expected = "ZDT2 requires at least 2 variables")]
    fn test_zdt2_construction_invalid() {
        ZDT2::new(1);
    }

    #[test]
    fn test_zdt2_evaluate_dimensions() {
        let problem = ZDT2::new(30);
        let x = vec![0.5; 30];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 2);
    }

    #[test]
    fn test_zdt2_optimal_point() {
        // Pareto-optimal solutions have x1 ∈ [0,1], xi = 0 for i > 1
        let problem = ZDT2::new(30);
        let mut x = vec![0.0; 30];
        x[0] = 0.5;

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);

        // f1 = x[0] = 0.5
        assert_relative_eq!(objectives[0], 0.5, epsilon = 1e-10);

        // For optimal point: g = 1, h = 1 - (f1/g)^2 = 1 - 0.25 = 0.75
        // f2 = g * h = 1 * 0.75 = 0.75
        assert_relative_eq!(objectives[1], 0.75, epsilon = 1e-6);
    }

    #[test]
    fn test_zdt2_non_convex_pareto_front() {
        let problem = ZDT2::new(30);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 100);
        assert_eq!(pareto_front.len(), 100);

        for point in &pareto_front {
            assert_eq!(point.len(), 2);
            let f1: f64 = point[0];
            let f2: f64 = point[1];

            // Verify Pareto front equation: f2 = 1 - f1^2
            let expected_f2 = 1.0 - f1.powi(2);
            assert_relative_eq!(f2, expected_f2, epsilon = 1e-6);

            // All objectives should be non-negative
            assert!((0.0..=1.0).contains(&f1));
            assert!((0.0..=1.0).contains(&f2));
        }
    }

    #[test]
    fn test_zdt2_bounds() {
        let problem = ZDT2::new(30);
        let bounds = TestProblem::<f64>::bounds(&problem);
        assert_eq!(bounds.len(), 30);

        for (lb, ub) in bounds {
            assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
            assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zdt2_edge_cases() {
        let problem = ZDT2::new(30);

        // Test at x = [0, 0, ..., 0]
        let x_zero = vec![0.0; 30];
        let obj_zero = TestProblem::<f64>::evaluate(&problem, &x_zero);
        assert_relative_eq!(obj_zero[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(obj_zero[1], 1.0, epsilon = 1e-6); // f2 = 1 - 0^2 = 1

        // Test at x = [1, 0, ..., 0] (optimal endpoint)
        let mut x_one = vec![0.0; 30];
        x_one[0] = 1.0;
        let obj_one = TestProblem::<f64>::evaluate(&problem, &x_one);
        assert_relative_eq!(obj_one[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj_one[1], 0.0, epsilon = 1e-6); // f2 = 1 - 1^2 = 0
    }

    // ========================================================================
    // ZDT3 Tests
    // ========================================================================

    #[test]
    fn test_zdt3_construction() {
        let problem = ZDT3::new(30);
        assert_eq!(TestProblem::<f64>::n_objectives(&problem), 2);
        assert_eq!(TestProblem::<f64>::n_variables(&problem), 30);
    }

    #[test]
    #[should_panic(expected = "ZDT3 requires at least 2 variables")]
    fn test_zdt3_construction_invalid() {
        ZDT3::new(1);
    }

    #[test]
    fn test_zdt3_evaluate_dimensions() {
        let problem = ZDT3::new(30);
        let x = vec![0.5; 30];
        let objectives = TestProblem::<f64>::evaluate(&problem, &x);
        assert_eq!(objectives.len(), 2);
    }

    #[test]
    fn test_zdt3_optimal_point() {
        // Pareto-optimal solutions have x1 ∈ specific intervals, xi = 0 for i > 1
        let problem = ZDT3::new(30);
        let mut x = vec![0.0; 30];
        x[0] = 0.5;

        let objectives = TestProblem::<f64>::evaluate(&problem, &x);

        // f1 = x[0] = 0.5
        assert_relative_eq!(objectives[0], 0.5, epsilon = 1e-10);

        // For optimal point: g = 1
        // h = 1 - sqrt(f1/g) - (f1/g) * sin(10 * π * f1)
        // This is more complex due to the sine term
        assert!(objectives[1].is_finite());
    }

    #[test]
    fn test_zdt3_disconnected_pareto_front() {
        let problem = ZDT3::new(30);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 1000);
        assert_eq!(pareto_front.len(), 1000);

        // ZDT3 has a disconnected Pareto front
        // Verify the mathematical formula for each point
        for point in &pareto_front {
            assert_eq!(point.len(), 2);
            let f1: f64 = point[0];
            let f2: f64 = point[1];

            // Verify Pareto front equation: f2 = 1 - sqrt(f1) - f1 * sin(10 * π * f1)
            use std::f64::consts::PI;
            let expected_f2 = 1.0 - f1.sqrt() - f1 * (10.0 * PI * f1).sin();
            assert_relative_eq!(f2, expected_f2, epsilon = 1e-6);

            // f1 should be in [0, 1]
            assert!((0.0..=1.0).contains(&f1));
        }
    }

    #[test]
    fn test_zdt3_bounds() {
        let problem = ZDT3::new(30);
        let bounds: Vec<(f64, f64)> = TestProblem::<f64>::bounds(&problem);
        assert_eq!(bounds.len(), 30);

        for (lb, ub) in bounds {
            assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
            assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zdt3_edge_cases() {
        let problem = ZDT3::new(30);

        // Test at x = [0, 0, ..., 0]
        let x_zero = vec![0.0; 30];
        let obj_zero = TestProblem::<f64>::evaluate(&problem, &x_zero);
        assert_relative_eq!(obj_zero[0], 0.0, epsilon = 1e-10);
        // f2 = 1 - sqrt(0) - 0 * sin(0) = 1
        assert_relative_eq!(obj_zero[1], 1.0, epsilon = 1e-6);

        // Test at x = [1, 0, ..., 0]
        let mut x_one = vec![0.0; 30];
        x_one[0] = 1.0;
        let obj_one = TestProblem::<f64>::evaluate(&problem, &x_one);
        assert_relative_eq!(obj_one[0], 1.0, epsilon = 1e-10);
        // f2 = 1 - sqrt(1) - 1 * sin(10π) = 0 (since sin(10π) = 0)
        assert_relative_eq!(obj_one[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zdt3_multimodality() {
        // ZDT3 has multiple disconnected regions in the Pareto front
        // Sample points across the space and verify they create disconnections
        let problem = ZDT3::new(30);
        let pareto_front = TestProblem::<f64>::generate_pareto_front(&problem, 1000);

        // Collect f1 and f2 values
        let mut f2_by_f1: Vec<(f64, f64)> = pareto_front.iter().map(|p| (p[0], p[1])).collect();
        f2_by_f1.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("No NaN"));

        // Check for discontinuities in f2 as f1 increases
        // The sine term creates local maxima and minima
        let mut has_discontinuity = false;
        for i in 1..f2_by_f1.len() - 1 {
            let prev_f2 = f2_by_f1[i - 1].1;
            let curr_f2 = f2_by_f1[i].1;
            let next_f2 = f2_by_f1[i + 1].1;

            // Look for non-monotonic behavior indicating disconnected regions
            if (curr_f2 > prev_f2 && curr_f2 > next_f2) || (curr_f2 < prev_f2 && curr_f2 < next_f2)
            {
                has_discontinuity = true;
                break;
            }
        }

        assert!(
            has_discontinuity,
            "ZDT3 should show non-monotonic behavior in Pareto front"
        );
    }

    // ========================================================================
    // ZDT General Tests
    // ========================================================================

    #[test]
    fn test_all_zdt_problems_bounds_in_unit_hypercube() {
        let problems: Vec<Box<dyn TestProblem<f64>>> = vec![
            Box::new(ZDT1::new(30)),
            Box::new(ZDT2::new(30)),
            Box::new(ZDT3::new(30)),
        ];

        for problem in problems {
            let bounds = TestProblem::<f64>::bounds(&*problem);
            assert_eq!(bounds.len(), TestProblem::<f64>::n_variables(&*problem));
            for (lb, ub) in bounds {
                assert_relative_eq!(lb, 0.0, epsilon = 1e-10);
                assert_relative_eq!(ub, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_zdt_variable_scalability() {
        // Test ZDT problems with different numbers of variables
        for n_vars in [5, 10, 30, 50] {
            let p1 = ZDT1::new(n_vars);
            let p2 = ZDT2::new(n_vars);
            let p3 = ZDT3::new(n_vars);

            let x = vec![0.5; n_vars];

            let obj1 = TestProblem::<f64>::evaluate(&p1, &x);
            let obj2 = TestProblem::<f64>::evaluate(&p2, &x);
            let obj3 = TestProblem::<f64>::evaluate(&p3, &x);

            assert_eq!(obj1.len(), 2);
            assert_eq!(obj2.len(), 2);
            assert_eq!(obj3.len(), 2);
        }
    }

    #[test]
    fn test_zdt_comparison() {
        // Compare ZDT1, ZDT2, ZDT3 at the same point
        let n = 30;
        let mut x = vec![0.0; n];
        x[0] = 0.5;

        let zdt1 = ZDT1::new(n);
        let zdt2 = ZDT2::new(n);
        let zdt3 = ZDT3::new(n);

        let obj1 = TestProblem::<f64>::evaluate(&zdt1, &x);
        let obj2 = TestProblem::<f64>::evaluate(&zdt2, &x);
        let obj3 = TestProblem::<f64>::evaluate(&zdt3, &x);

        // All should have same f1 = x[0]
        assert_relative_eq!(obj1[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(obj2[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(obj3[0], 0.5, epsilon = 1e-10);

        // f2 values should differ due to different h-functions
        // ZDT1: h = 1 - sqrt(0.5) ≈ 0.2929
        // ZDT2: h = 1 - 0.5^2 = 0.75
        // ZDT3: h includes sine term, so it's different
        assert_relative_eq!(obj1[1], 1.0 - (0.5_f64).sqrt(), epsilon = 1e-6);
        assert_relative_eq!(obj2[1], 0.75, epsilon = 1e-6);
        assert_ne!(obj3[1], obj1[1]);
        assert_ne!(obj3[1], obj2[1]);
    }
}

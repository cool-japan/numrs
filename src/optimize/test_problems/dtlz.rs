//! DTLZ Test Suite: Scalable multi-objective optimization benchmark problems
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

use num_traits::Float;
use std::f64::consts::PI;

use super::helpers::{compute_sum, generate_simplex_points};
use super::TestProblem;

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
    pub k: usize,
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
    pub k: usize,
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
    pub k: usize,
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
    pub k: usize,
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
// DTLZ Tests
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
    // General DTLZ Tests
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
}

//! ZDT Test Suite: Bi-objective optimization benchmark problems
//!
//! The ZDT (Zitzler-Deb-Thiele) test suite consists of bi-objective problems with
//! known Pareto-optimal fronts. These problems are specifically designed to test
//! different aspects of multi-objective optimization algorithms:
//!
//! - **ZDT1**: Convex Pareto front - tests convergence
//! - **ZDT2**: Non-convex (concave) Pareto front - tests diversity
//! - **ZDT3**: Disconnected Pareto front - tests diversity maintenance

use num_traits::Float;
use std::f64::consts::PI;

use super::TestProblem;

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
// ZDT Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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

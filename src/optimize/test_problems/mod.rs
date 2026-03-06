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

use num_traits::Float;

pub mod dtlz;
pub(crate) mod helpers;
pub mod zdt;

pub use dtlz::{DTLZ1, DTLZ2, DTLZ3, DTLZ7};
pub use zdt::{ZDT1, ZDT2, ZDT3};

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

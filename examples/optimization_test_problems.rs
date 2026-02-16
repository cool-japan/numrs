//! Multi-Objective Test Problems Showcase
//!
//! This example demonstrates all available multi-objective test problems in NumRS2,
//! including the ZDT (bi-objective) and DTLZ (many-objective) suites. It showcases
//! problem characteristics, true Pareto fronts, and algorithm performance comparison.
//!
//! # Test Problem Suites
//!
//! ## ZDT Suite (Bi-objective)
//! - **ZDT1**: Convex Pareto front
//! - **ZDT2**: Non-convex (concave) Pareto front
//! - **ZDT3**: Disconnected Pareto front (5 regions)
//!
//! ## DTLZ Suite (Scalable Many-objective)
//! - **DTLZ1**: Linear Pareto front, highly multi-modal (3^k local fronts)
//! - **DTLZ2**: Concave/spherical Pareto front, unimodal
//! - **DTLZ3**: Concave Pareto front, highly multi-modal (3^k local fronts)
//! - **DTLZ7**: Disconnected Pareto regions, mixed shape
//!
//! # Topics Covered
//!
//! 1. **Problem Characteristics**: Understanding each test problem
//! 2. **True Pareto Fronts**: Generating and visualizing optimal fronts
//! 3. **Algorithm Comparison**: NSGA-II vs NSGA-III performance
//! 4. **Problem Selection**: Choosing the right problem for benchmarking
//! 5. **Validation**: Verifying algorithm correctness
//!
//! Run with: `cargo run --example optimization_test_problems --release`

#![allow(clippy::type_complexity)]

use numrs2::optimize::nsga2::{nsga2, NSGA2Config, QualityMetricsConfig};
use numrs2::optimize::nsga3::{nsga3, NSGA3Config};
use numrs2::optimize::test_problems::{TestProblem, DTLZ1, DTLZ2, DTLZ3, DTLZ7, ZDT1, ZDT2, ZDT3};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        Multi-Objective Test Problems Showcase             ║");
    println!("║                   NumRS2 v0.2.0                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example 1: ZDT test suite overview
    example1_zdt_suite()?;

    // Example 2: DTLZ test suite overview
    example2_dtlz_suite()?;

    // Example 3: Pareto front generation and validation
    example3_pareto_fronts()?;

    // Example 4: Algorithm performance comparison
    example4_algorithm_comparison()?;

    // Example 5: Problem selection guide
    example5_problem_selection_guide()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     All Test Problem Examples Completed Successfully!     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Example 1: ZDT Test Suite Overview
///
/// Demonstrates all ZDT problems and their characteristics.
fn example1_zdt_suite() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 1: ZDT Test Suite (Bi-objective Problems)");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("The ZDT suite consists of bi-objective problems designed to test");
    println!("different aspects of multi-objective optimization algorithms.\n");

    // ZDT1: Convex front
    {
        println!("ZDT1: Convex Pareto Front");
        println!("─────────────────────────");
        println!("Characteristics:");
        println!("  • Objectives:    2 (minimize both)");
        println!("  • Variables:     30 (standard)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Convex, continuous");
        println!("  • Difficulty:    Easy (good convergence test)");
        println!();

        let problem = ZDT1::new(30);
        println!("Mathematical formulation:");
        println!("  f1(x) = x1");
        println!("  g(x)  = 1 + 9 * sum(x2...x30) / 29");
        println!("  f2(x) = g(x) * [1 - sqrt(f1/g)]");
        println!();
        println!("Pareto-optimal solutions:");
        println!("  x1 ∈ [0, 1], xi = 0 for i > 1");
        println!("  Pareto front: f2 = 1 - sqrt(f1) for f1 ∈ [0, 1]");
        println!();

        // Generate and display sample Pareto front
        let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(5);
        println!("Sample true Pareto front points:");
        for (i, point) in pareto_front.iter().enumerate() {
            println!("  Point {}: f1={:.4}, f2={:.4}", i + 1, point[0], point[1]);
        }
        println!();
    }

    // ZDT2: Non-convex front
    {
        println!("ZDT2: Non-convex (Concave) Pareto Front");
        println!("───────────────────────────────────────");
        println!("Characteristics:");
        println!("  • Objectives:    2 (minimize both)");
        println!("  • Variables:     30 (standard)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Non-convex (concave), continuous");
        println!("  • Difficulty:    Moderate (tests diversity preservation)");
        println!();

        let problem = ZDT2::new(30);
        println!("Mathematical formulation:");
        println!("  f1(x) = x1");
        println!("  g(x)  = 1 + 9 * sum(x2...x30) / 29");
        println!("  f2(x) = g(x) * [1 - (f1/g)²]");
        println!();
        println!("Pareto front: f2 = 1 - f1² for f1 ∈ [0, 1]");
        println!();

        let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(5);
        println!("Sample true Pareto front points:");
        for (i, point) in pareto_front.iter().enumerate() {
            println!("  Point {}: f1={:.4}, f2={:.4}", i + 1, point[0], point[1]);
        }
        println!();
    }

    // ZDT3: Disconnected front
    {
        println!("ZDT3: Disconnected Pareto Front");
        println!("───────────────────────────────");
        println!("Characteristics:");
        println!("  • Objectives:    2 (minimize both)");
        println!("  • Variables:     30 (standard)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Disconnected (5 separate regions)");
        println!("  • Difficulty:    Hard (tests diversity in disconnected regions)");
        println!();

        let problem = ZDT3::new(30);
        println!("Mathematical formulation:");
        println!("  f1(x) = x1");
        println!("  g(x)  = 1 + 9 * sum(x2...x30) / 29");
        println!("  f2(x) = g(x) * [1 - sqrt(f1/g) - (f1/g)*sin(10π*f1)]");
        println!();
        println!("The sine term creates 5 disconnected Pareto-optimal regions.");
        println!();

        let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(10);
        println!("Sample true Pareto front points (showing disconnected regions):");
        for (i, point) in pareto_front.iter().enumerate() {
            println!("  Point {}: f1={:.4}, f2={:.4}", i + 1, point[0], point[1]);
        }
        println!();
    }

    println!("✓ ZDT suite overview completed\n");
    Ok(())
}

/// Example 2: DTLZ Test Suite Overview
///
/// Demonstrates all DTLZ problems and their characteristics.
fn example2_dtlz_suite() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 2: DTLZ Test Suite (Many-objective Problems)");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("The DTLZ suite consists of scalable multi-objective problems");
    println!("that can be configured for any number of objectives (M ≥ 2).\n");

    // DTLZ1: Linear front
    {
        println!("DTLZ1: Linear Pareto Front");
        println!("──────────────────────────");
        let n_obj = 3;
        let n_vars = n_obj + 5 - 1; // Using k=5 for example
        let problem = DTLZ1::new(n_obj, n_vars);

        println!("Characteristics:");
        println!("  • Objectives:    Scalable (M ≥ 2)");
        println!("  • Variables:     n = M + k - 1 (typically k=5)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Linear (simplex)");
        println!("  • Difficulty:    Very hard - 3^k local Pareto fronts");
        println!(
            "  • Example (M=3, k=5): {} variables, 243 local fronts",
            n_vars
        );
        println!();

        println!("Pareto-optimal solutions:");
        println!("  Sum of objectives = 0.5");
        println!("  Linear hyperplane in objective space");
        println!();

        let pareto_front = problem.generate_pareto_front(5);
        println!("Sample Pareto front (3 objectives):");
        for (i, point) in pareto_front.iter().enumerate() {
            print!("  Point {}: ", i + 1);
            for (j, &obj) in point.iter().enumerate() {
                print!("f{}={:.4}", j + 1, obj);
                if j < point.len() - 1 {
                    print!(", ");
                }
            }
            let sum: f64 = point.iter().sum();
            println!(" (sum={:.4})", sum);
        }
        println!();
    }

    // DTLZ2: Concave front
    {
        println!("DTLZ2: Concave/Spherical Pareto Front");
        println!("─────────────────────────────────────");
        let n_obj = 3;
        let n_vars = n_obj + 10 - 1;
        let problem = DTLZ2::new(n_obj, n_vars);

        println!("Characteristics:");
        println!("  • Objectives:    Scalable (M ≥ 2)");
        println!("  • Variables:     n = M + k - 1 (typically k=10)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Concave (spherical with radius 1)");
        println!("  • Difficulty:    Moderate - unimodal, good baseline");
        println!("  • Example (M=3, k=10): {} variables", n_vars);
        println!();

        println!("Pareto-optimal solutions:");
        println!("  Sum of objective squares = 1");
        println!("  Spherical surface in objective space");
        println!();

        let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(5);
        println!("Sample Pareto front (3 objectives):");
        for (i, point) in pareto_front.iter().enumerate() {
            print!("  Point {}: ", i + 1);
            for (j, &obj) in point.iter().enumerate() {
                print!("f{}={:.4}", j + 1, obj);
                if j < point.len() - 1 {
                    print!(", ");
                }
            }
            let radius_sq: f64 = point.iter().map(|&x: &f64| x * x).sum();
            println!(" (r²={:.4})", radius_sq);
        }
        println!();
    }

    // DTLZ3: Multi-modal concave
    {
        println!("DTLZ3: Multi-modal Concave Pareto Front");
        println!("───────────────────────────────────────");
        let n_obj = 3;
        let n_vars = n_obj + 10 - 1;

        println!("Characteristics:");
        println!("  • Objectives:    Scalable (M ≥ 2)");
        println!("  • Variables:     n = M + k - 1 (typically k=10)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Concave (same as DTLZ2)");
        println!("  • Difficulty:    Very hard - 3^k local fronts (like DTLZ1)");
        println!(
            "  • Example (M=3, k=10): {} variables, 59,049 local fronts!",
            n_vars
        );
        println!();

        println!("DTLZ3 = DTLZ2 shape + DTLZ1 multi-modality");
        println!("Most challenging problem in the DTLZ suite.");
        println!();
    }

    // DTLZ7: Disconnected regions
    {
        println!("DTLZ7: Disconnected Pareto Regions");
        println!("──────────────────────────────────");
        let n_obj = 3;
        let n_vars = n_obj + 20 - 1;

        println!("Characteristics:");
        println!("  • Objectives:    Scalable (M ≥ 2)");
        println!("  • Variables:     n = M + k - 1 (typically k=20)");
        println!("  • Bounds:        [0, 1] for all variables");
        println!("  • Front shape:   Disconnected, mixed");
        println!("  • Difficulty:    Hard - tests diversity in disconnected regions");
        println!(
            "  • Example (M=3, k=20): {} variables, 2^(M-1) regions",
            n_vars
        );
        println!();

        println!("The Pareto front consists of 2^(M-1) disconnected regions.");
        println!("For M=3: 4 disconnected regions in objective space.");
        println!();
    }

    println!("✓ DTLZ suite overview completed\n");
    Ok(())
}

/// Example 3: Pareto Front Generation
///
/// Demonstrates generating true Pareto fronts and their properties.
fn example3_pareto_fronts() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 3: True Pareto Front Generation and Validation");
    println!("═══════════════════════════════════════════════════════════\n");

    // Generate fronts with different resolutions
    let resolutions = vec![10, 50, 100];

    println!("ZDT1 Pareto Front Generation:");
    println!("─────────────────────────────");
    let problem = ZDT1::new(30);

    for &n_points in &resolutions {
        let start = Instant::now();
        let pareto_front = problem.generate_pareto_front(n_points);
        let duration = start.elapsed();

        println!("  {} points: generated in {:?}", n_points, duration);

        // Validate properties
        let mut min_f1 = f64::MAX;
        let mut max_f1 = f64::MIN;
        let mut min_f2 = f64::MAX;
        let mut max_f2 = f64::MIN;

        for point in &pareto_front {
            min_f1 = min_f1.min(point[0]);
            max_f1 = max_f1.max(point[0]);
            min_f2 = min_f2.min(point[1]);
            max_f2 = max_f2.max(point[1]);
        }

        println!("    f1 range: [{:.4}, {:.4}]", min_f1, max_f1);
        println!("    f2 range: [{:.4}, {:.4}]", min_f2, max_f2);
    }
    println!();

    // DTLZ2 with different objective counts
    println!("DTLZ2 Pareto Front Generation (Different Objectives):");
    println!("──────────────────────────────────────────────────────");

    for n_obj in 2..=5 {
        let n_vars = n_obj + 10 - 1;
        let problem = DTLZ2::new(n_obj, n_vars);

        let start = Instant::now();
        let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(50);
        let duration = start.elapsed();

        println!(
            "  {} objectives: {} points, {:?}",
            n_obj,
            pareto_front.len(),
            duration
        );

        // Validate spherical constraint (sum of squares = 1)
        let mut total_error = 0.0;
        for point in &pareto_front {
            let radius_sq: f64 = point.iter().map(|&x: &f64| x * x).sum();
            total_error += (radius_sq - 1.0).abs();
        }
        let avg_error = total_error / pareto_front.len() as f64;

        println!(
            "    Sphere constraint error: {:.6} (should be ~0)",
            avg_error
        );
    }
    println!();

    // Verify non-domination property
    println!("Pareto Front Validation:");
    println!("────────────────────────");
    let problem = ZDT2::new(30);
    let pareto_front: Vec<Vec<f64>> = problem.generate_pareto_front(20);

    println!("Checking non-domination property for ZDT2...");
    let mut dominated_count = 0;

    for i in 0..pareto_front.len() {
        for j in 0..pareto_front.len() {
            if i != j {
                // Check if point i dominates point j
                let dominates = pareto_front[i][0] <= pareto_front[j][0]
                    && pareto_front[i][1] <= pareto_front[j][1]
                    && (pareto_front[i][0] < pareto_front[j][0]
                        || pareto_front[i][1] < pareto_front[j][1]);

                if dominates {
                    dominated_count += 1;
                }
            }
        }
    }

    if dominated_count == 0 {
        println!("  ✓ All points are non-dominated (valid Pareto front)");
    } else {
        println!(
            "  ✗ Found {} dominated points (should be 0)",
            dominated_count
        );
    }

    println!("\n✓ Pareto front generation demonstrated successfully\n");
    Ok(())
}

/// Example 4: Algorithm Performance Comparison
///
/// Compares NSGA-II and NSGA-III on different problem types.
fn example4_algorithm_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 4: Algorithm Performance Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    struct BenchmarkResult {
        problem: String,
        algorithm: String,
        objectives: usize,
        pareto_size: usize,
        time: std::time::Duration,
        igd: Option<f64>,
    }

    let mut results = Vec::new();

    // Test 1: NSGA-II on ZDT1
    {
        println!("Testing NSGA-II on ZDT1...");
        let problem = ZDT1::new(30);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| problem.evaluate(x)[0]),
            Box::new(|x: &[f64]| problem.evaluate(x)[1]),
        ];
        let bounds = problem.bounds();
        let reference_front = problem.generate_pareto_front(100);

        let config = NSGA2Config {
            pop_size: 100,
            max_generations: 150,
            quality_metrics_config: Some(QualityMetricsConfig {
                calculate_spacing: false,
                calculate_spread: false,
                reference_front: Some(reference_front),
            }),
            ..Default::default()
        };

        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(BenchmarkResult {
            problem: "ZDT1".to_string(),
            algorithm: "NSGA-II".to_string(),
            objectives: 2,
            pareto_size: result.pareto_front.len(),
            time: duration,
            igd: result.igd,
        });
        println!("  ✓ Completed\n");
    }

    // Test 2: NSGA-II on ZDT3 (challenging)
    {
        println!("Testing NSGA-II on ZDT3...");
        let problem = ZDT3::new(30);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| problem.evaluate(x)[0]),
            Box::new(|x: &[f64]| problem.evaluate(x)[1]),
        ];
        let bounds = problem.bounds();
        let reference_front = problem.generate_pareto_front(100);

        let config = NSGA2Config {
            pop_size: 100,
            max_generations: 150,
            quality_metrics_config: Some(QualityMetricsConfig {
                calculate_spacing: false,
                calculate_spread: false,
                reference_front: Some(reference_front),
            }),
            ..Default::default()
        };

        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(BenchmarkResult {
            problem: "ZDT3".to_string(),
            algorithm: "NSGA-II".to_string(),
            objectives: 2,
            pareto_size: result.pareto_front.len(),
            time: duration,
            igd: result.igd,
        });
        println!("  ✓ Completed\n");
    }

    // Test 3: NSGA-III on DTLZ2 (3 objectives)
    {
        println!("Testing NSGA-III on DTLZ2 (3 objectives)...");
        let problem = DTLZ2::new(3, 12);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..3)
            .map(|i| {
                let prob = problem.clone();
                Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let bounds = problem.bounds();

        let config = NSGA3Config {
            pop_size: 92,
            max_generations: 150,
            n_divisions: 12,
            ..Default::default()
        };

        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(BenchmarkResult {
            problem: "DTLZ2-3".to_string(),
            algorithm: "NSGA-III".to_string(),
            objectives: 3,
            pareto_size: result.pareto_front.len(),
            time: duration,
            igd: None,
        });
        println!("  ✓ Completed\n");
    }

    // Test 4: NSGA-III on DTLZ2 (5 objectives)
    {
        println!("Testing NSGA-III on DTLZ2 (5 objectives)...");
        let problem = DTLZ2::new(5, 14);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..5)
            .map(|i| {
                let prob = problem.clone();
                Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let bounds = problem.bounds();

        let config = NSGA3Config {
            pop_size: 126,
            max_generations: 200,
            n_divisions: 6,
            ..Default::default()
        };

        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(BenchmarkResult {
            problem: "DTLZ2-5".to_string(),
            algorithm: "NSGA-III".to_string(),
            objectives: 5,
            pareto_size: result.pareto_front.len(),
            time: duration,
            igd: None,
        });
        println!("  ✓ Completed\n");
    }

    // Display comparison table
    println!("═══════════════════════════════════════════════════════════");
    println!("Performance Comparison Results");
    println!("═══════════════════════════════════════════════════════════\n");

    println!(
        "{:<12} {:<12} {:<10} {:<12} {:<12}",
        "Problem", "Algorithm", "Obj", "PF Size", "Time (sec)"
    );
    println!("{}", "─".repeat(65));

    for result in &results {
        println!(
            "{:<12} {:<12} {:<10} {:<12} {:<12.2}",
            result.problem,
            result.algorithm,
            result.objectives,
            result.pareto_size,
            result.time.as_secs_f64()
        );
    }

    println!();
    println!("Observations:");
    println!("  • NSGA-II excels at bi-objective problems (2 objectives)");
    println!("  • NSGA-III required for many-objective problems (3+ objectives)");
    println!("  • Computational cost increases with objectives");
    println!("  • ZDT3 (disconnected) more challenging than ZDT1 (convex)");

    println!("\n✓ Algorithm comparison demonstrated successfully\n");
    Ok(())
}

/// Example 5: Problem Selection Guide
///
/// Provides guidance on selecting appropriate test problems for different
/// benchmarking scenarios.
fn example5_problem_selection_guide() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 5: Test Problem Selection Guide");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Selecting the Right Test Problem:");
    println!();

    println!("FOR ALGORITHM DEVELOPMENT:");
    println!("──────────────────────────");
    println!("  1. Start with ZDT1 or DTLZ2");
    println!("     - Unimodal, moderate difficulty");
    println!("     - Good for initial development and debugging");
    println!("     - Fast convergence allows quick iteration");
    println!();
    println!("  2. Add ZDT2 for diversity testing");
    println!("     - Non-convex front tests diversity preservation");
    println!("     - Ensures algorithm handles various front shapes");
    println!();
    println!("  3. Use ZDT3 or DTLZ7 for disconnected regions");
    println!("     - Tests ability to maintain diversity in gaps");
    println!("     - Critical for real-world problems");
    println!();

    println!("FOR ALGORITHM COMPARISON:");
    println!("─────────────────────────");
    println!("  1. Use complete ZDT suite (ZDT1-3)");
    println!("     - Standard benchmark for bi-objective algorithms");
    println!("     - Well-studied with known characteristics");
    println!();
    println!("  2. Add DTLZ2 and DTLZ3 for robustness");
    println!("     - DTLZ2: Baseline performance");
    println!("     - DTLZ3: Multi-modal challenge");
    println!();
    println!("  3. Include DTLZ1 or DTLZ7 for completeness");
    println!("     - DTLZ1: Linear front + high multi-modality");
    println!("     - DTLZ7: Disconnected + mixed shape");
    println!();

    println!("FOR SCALABILITY TESTING:");
    println!("────────────────────────");
    println!("  1. Use DTLZ2 with varying objectives");
    println!("     - Test: M = 3, 5, 8, 10, 15");
    println!("     - Evaluate computational scaling");
    println!();
    println!("  2. Add DTLZ3 for stress testing");
    println!("     - Same as DTLZ2 but much harder");
    println!("     - Tests robustness under difficulty");
    println!();

    println!("RECOMMENDED MINIMAL BENCHMARK SET:");
    println!("───────────────────────────────────");
    println!("  Bi-objective:    ZDT1, ZDT2, ZDT3");
    println!("  Many-objective:  DTLZ2 (M=3,5), DTLZ3 (M=3)");
    println!("  Total runtime:   ~5-10 minutes on modern hardware");
    println!();

    println!("PROBLEM CHARACTERISTICS SUMMARY:");
    println!("────────────────────────────────");
    println!();
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "Problem", "Objectives", "Difficulty", "Tests"
    );
    println!("{}", "─".repeat(65));
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "ZDT1", "2", "Easy", "Convergence"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "ZDT2", "2", "Moderate", "Diversity"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "ZDT3", "2", "Hard", "Disconnected"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "DTLZ1", "Scalable", "Very Hard", "Multi-modal"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "DTLZ2", "Scalable", "Moderate", "Baseline"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "DTLZ3", "Scalable", "Very Hard", "Robustness"
    );
    println!(
        "{:<10} {:<15} {:<20} {:<15}",
        "DTLZ7", "Scalable", "Hard", "Mixed"
    );

    println!("\n✓ Problem selection guide completed\n");
    Ok(())
}

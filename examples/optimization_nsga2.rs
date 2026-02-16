//! NSGA-II Multi-Objective Optimization Examples
//!
//! This example demonstrates comprehensive usage of NSGA-II (Non-dominated Sorting
//! Genetic Algorithm II) for solving bi-objective optimization problems using the
//! standard ZDT (Zitzler-Deb-Thiele) test suite.
//!
//! # Topics Covered
//!
//! 1. **Basic NSGA-II Usage**: Solving ZDT1 with default configuration
//! 2. **Quality Metrics**: Using spacing, spread, hypervolume indicators
//! 3. **Convergence Analysis**: IGD and GD metrics with reference fronts
//! 4. **Problem Comparison**: Performance across ZDT1, ZDT2, ZDT3
//! 5. **Configuration Options**: Tuning algorithm parameters
//! 6. **Performance Measurement**: Timing and convergence tracking
//!
//! # Test Problems
//!
//! - **ZDT1**: Convex Pareto front - tests convergence
//! - **ZDT2**: Non-convex (concave) Pareto front - tests diversity
//! - **ZDT3**: Disconnected Pareto front - tests diversity maintenance
//!
//! # Quality Metrics
//!
//! - **Hypervolume**: Volume of objective space dominated by Pareto front
//! - **Spacing (S)**: Uniformity of distribution (lower is better)
//! - **Spread (Δ)**: Extent and uniformity of spread (lower is better)
//! - **IGD**: Inverted Generational Distance - convergence and coverage
//! - **GD**: Generational Distance - convergence to reference front
//!
//! Run with: `cargo run --example optimization_nsga2 --release`

#![allow(clippy::type_complexity)]

use numrs2::optimize::nsga2::{nsga2, HypervolumeConfig, NSGA2Config, QualityMetricsConfig};
use numrs2::optimize::test_problems::{TestProblem, ZDT1, ZDT2, ZDT3};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     NSGA-II Multi-Objective Optimization Examples         ║");
    println!("║                   NumRS2 v0.2.0                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example 1: Basic NSGA-II usage with ZDT1
    example1_basic_usage()?;

    // Example 2: Using quality metrics (spacing, spread, hypervolume)
    example2_quality_metrics()?;

    // Example 3: Convergence analysis with reference front (IGD, GD)
    example3_convergence_analysis()?;

    // Example 4: Comparing performance across different problems
    example4_problem_comparison()?;

    // Example 5: Advanced configuration and parameter tuning
    example5_advanced_configuration()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          All NSGA-II Examples Completed Successfully!      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Example 1: Basic NSGA-II Usage
///
/// Demonstrates the simplest use case of NSGA-II with default configuration
/// on the ZDT1 test problem (convex Pareto front).
fn example1_basic_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 1: Basic NSGA-II Usage (ZDT1)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create ZDT1 test problem with 30 decision variables
    let problem = ZDT1::new(30);

    // Define objective functions as closures
    // ZDT1 has two objectives: f1(x) = x[0], f2(x) = g(x) * (1 - sqrt(x[0]/g(x)))
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[0]
        }),
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[1]
        }),
    ];

    // Get variable bounds [0, 1] for all variables
    let bounds = problem.bounds();

    // Use default NSGA-II configuration
    // - Population size: 100
    // - Generations: 100
    // - Crossover rate: 0.9
    // - Mutation rate: 0.1
    let config = NSGA2Config::default();

    println!("Configuration:");
    println!("  Population size:  {}", config.pop_size);
    println!("  Max generations:  {}", config.max_generations);
    println!("  Crossover rate:   {:.2}", config.crossover_rate);
    println!("  Mutation rate:    {:.2}", config.mutation_rate);
    println!("  Variables:        {}", bounds.len());
    println!();

    // Run NSGA-II optimization
    let start = Instant::now();
    let result = nsga2(
        &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
        &bounds,
        Some(config),
    )?;
    let duration = start.elapsed();

    // Display results
    println!("Results:");
    println!("  Pareto front size:      {}", result.pareto_front.len());
    println!("  Total population size:  {}", result.population.len());
    println!("  Generations executed:   {}", result.generations);
    println!("  Execution time:         {:?}", duration);
    println!();

    // Show sample solutions from Pareto front
    println!("Sample Pareto-optimal solutions:");
    let samples = if result.pareto_front.len() > 5 {
        vec![
            0,
            result.pareto_front.len() / 4,
            result.pareto_front.len() / 2,
            3 * result.pareto_front.len() / 4,
            result.pareto_front.len() - 1,
        ]
    } else {
        (0..result.pareto_front.len()).collect()
    };

    for idx in samples {
        let individual = &result.pareto_front[idx];
        println!(
            "  Solution {}: f1={:.6}, f2={:.6}",
            idx + 1,
            individual.objectives[0],
            individual.objectives[1]
        );
    }

    println!("\n✓ Basic usage demonstrated successfully\n");
    Ok(())
}

/// Example 2: Quality Metrics
///
/// Demonstrates how to use quality metrics to evaluate the performance
/// and characteristics of the obtained Pareto front.
fn example2_quality_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 2: Quality Metrics (ZDT2)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ZDT2 has a non-convex (concave) Pareto front
    let problem = ZDT2::new(30);

    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[0]
        }),
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[1]
        }),
    ];

    let bounds = problem.bounds();

    // Configure NSGA-II with quality metrics enabled
    let config = NSGA2Config {
        pop_size: 100,
        max_generations: 150,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true, // Measure distribution uniformity
            calculate_spread: true,  // Measure extent and uniformity
            reference_front: None,   // Will add in next example
        }),
        hypervolume_config: Some(HypervolumeConfig {
            reference_point: vec![2.0, 2.0], // Must dominate all solutions
        }),
        ..Default::default()
    };

    println!("ZDT2 Problem Characteristics:");
    println!("  Type: Non-convex (concave) Pareto front");
    println!("  Tests: Diversity preservation in algorithms");
    println!();

    let start = Instant::now();
    let result = nsga2(
        &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
        &bounds,
        Some(config),
    )?;
    let duration = start.elapsed();

    println!("Optimization Results:");
    println!("  Pareto front size: {}", result.pareto_front.len());
    println!("  Execution time:    {:?}", duration);
    println!();

    // Display quality metrics
    println!("Quality Metrics:");

    if let Some(hv) = result.hypervolume {
        println!("  Hypervolume (HV):  {:.6}", hv);
        println!("    - Measures volume of objective space dominated by front");
        println!("    - Higher is better (more coverage)");
    }

    if let Some(spacing) = result.spacing {
        println!("  Spacing (S):       {:.6}", spacing);
        println!("    - Measures uniformity of distribution");
        println!("    - Lower is better (more uniform)");
        println!("    - S=0 means perfectly uniform distribution");
    }

    if let Some(spread) = result.spread {
        println!("  Spread (Δ):        {:.6}", spread);
        println!("    - Measures extent and uniformity of spread");
        println!("    - Lower is better (better spread)");
        println!("    - Δ=0 means perfect spread and uniformity");
    }

    // Analyze objective space coverage
    let mut f1_min = f64::MAX;
    let mut f1_max = f64::MIN;
    let mut f2_min = f64::MAX;
    let mut f2_max = f64::MIN;

    for individual in &result.pareto_front {
        f1_min = f1_min.min(individual.objectives[0]);
        f1_max = f1_max.max(individual.objectives[0]);
        f2_min = f2_min.min(individual.objectives[1]);
        f2_max = f2_max.max(individual.objectives[1]);
    }

    println!();
    println!("Objective Space Coverage:");
    println!("  f1 range: [{:.6}, {:.6}]", f1_min, f1_max);
    println!("  f2 range: [{:.6}, {:.6}]", f2_min, f2_max);

    println!("\n✓ Quality metrics demonstrated successfully\n");
    Ok(())
}

/// Example 3: Convergence Analysis
///
/// Demonstrates convergence analysis using IGD (Inverted Generational Distance)
/// and GD (Generational Distance) metrics with a reference Pareto front.
fn example3_convergence_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 3: Convergence Analysis (ZDT1)");
    println!("═══════════════════════════════════════════════════════════\n");

    let problem = ZDT1::new(30);

    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[0]
        }),
        Box::new(|x: &[f64]| {
            let obj = problem.evaluate(x);
            obj[1]
        }),
    ];

    let bounds = problem.bounds();

    // Generate true Pareto front for ZDT1 as reference
    println!("Generating reference Pareto front (100 points)...");
    let reference_front = problem.generate_pareto_front(100);
    println!(
        "Reference front generated: {} points",
        reference_front.len()
    );
    println!();

    // Configure NSGA-II with reference front for convergence metrics
    let config = NSGA2Config {
        pop_size: 100,
        max_generations: 200,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: true,
            reference_front: Some(reference_front.clone()),
        }),
        hypervolume_config: Some(HypervolumeConfig {
            reference_point: vec![2.0, 2.0],
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

    println!("Optimization Results:");
    println!("  Pareto front size: {}", result.pareto_front.len());
    println!("  Generations:       {}", result.generations);
    println!("  Execution time:    {:?}", duration);
    println!();

    println!("Convergence Metrics:");

    if let Some(igd) = result.igd {
        println!("  IGD (Inverted Generational Distance): {:.6}", igd);
        println!("    - Measures how well obtained front covers reference front");
        println!("    - Lower is better (better coverage)");
        println!("    - IGD=0 means perfect coverage");
    }

    if let Some(gd) = result.gd {
        println!("  GD (Generational Distance):            {:.6}", gd);
        println!("    - Measures convergence to reference front");
        println!("    - Lower is better (closer to reference)");
        println!("    - GD=0 means perfect convergence");
    }

    println!();
    println!("Diversity Metrics:");

    if let Some(spacing) = result.spacing {
        println!("  Spacing: {:.6}", spacing);
    }

    if let Some(spread) = result.spread {
        println!("  Spread:  {:.6}", spread);
    }

    if let Some(hv) = result.hypervolume {
        println!();
        println!("Coverage Metric:");
        println!("  Hypervolume: {:.6}", hv);
    }

    // Convergence interpretation
    println!();
    println!("Convergence Interpretation:");

    let convergence_quality = if let Some(igd) = result.igd {
        if igd < 0.001 {
            "Excellent convergence - front very close to reference"
        } else if igd < 0.01 {
            "Good convergence - front close to reference"
        } else if igd < 0.1 {
            "Moderate convergence - front approaching reference"
        } else {
            "Poor convergence - consider more generations"
        }
    } else {
        "No convergence metric available"
    };

    println!("  {}", convergence_quality);

    println!("\n✓ Convergence analysis demonstrated successfully\n");
    Ok(())
}

/// Example 4: Problem Comparison
///
/// Compares NSGA-II performance across different problem types:
/// ZDT1 (convex), ZDT2 (non-convex), and ZDT3 (disconnected).
fn example4_problem_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 4: Problem Comparison (ZDT1, ZDT2, ZDT3)");
    println!("═══════════════════════════════════════════════════════════\n");

    struct ProblemResult {
        name: String,
        description: String,
        pareto_size: usize,
        time: std::time::Duration,
        hypervolume: Option<f64>,
        spacing: Option<f64>,
        spread: Option<f64>,
        igd: Option<f64>,
    }

    let mut results = Vec::new();

    // Common configuration for fair comparison
    let base_config = NSGA2Config {
        pop_size: 100,
        max_generations: 150,
        hypervolume_config: Some(HypervolumeConfig {
            reference_point: vec![2.0, 2.0],
        }),
        ..Default::default()
    };

    // Test ZDT1 (Convex front)
    {
        println!("Testing ZDT1 (Convex Pareto front)...");
        let problem = ZDT1::new(30);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| problem.evaluate(x)[0]),
            Box::new(|x: &[f64]| problem.evaluate(x)[1]),
        ];
        let bounds = problem.bounds();
        let reference_front = problem.generate_pareto_front(100);

        let config = NSGA2Config {
            quality_metrics_config: Some(QualityMetricsConfig {
                calculate_spacing: true,
                calculate_spread: true,
                reference_front: Some(reference_front),
            }),
            ..base_config.clone()
        };

        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(ProblemResult {
            name: "ZDT1".to_string(),
            description: "Convex front".to_string(),
            pareto_size: result.pareto_front.len(),
            time: duration,
            hypervolume: result.hypervolume,
            spacing: result.spacing,
            spread: result.spread,
            igd: result.igd,
        });
        println!("  ✓ Completed in {:?}\n", duration);
    }

    // Test ZDT2 (Non-convex front)
    {
        println!("Testing ZDT2 (Non-convex Pareto front)...");
        let problem = ZDT2::new(30);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| problem.evaluate(x)[0]),
            Box::new(|x: &[f64]| problem.evaluate(x)[1]),
        ];
        let bounds = problem.bounds();
        let reference_front = problem.generate_pareto_front(100);

        let config = NSGA2Config {
            quality_metrics_config: Some(QualityMetricsConfig {
                calculate_spacing: true,
                calculate_spread: true,
                reference_front: Some(reference_front),
            }),
            ..base_config.clone()
        };

        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(ProblemResult {
            name: "ZDT2".to_string(),
            description: "Non-convex front".to_string(),
            pareto_size: result.pareto_front.len(),
            time: duration,
            hypervolume: result.hypervolume,
            spacing: result.spacing,
            spread: result.spread,
            igd: result.igd,
        });
        println!("  ✓ Completed in {:?}\n", duration);
    }

    // Test ZDT3 (Disconnected front)
    {
        println!("Testing ZDT3 (Disconnected Pareto front)...");
        let problem = ZDT3::new(30);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x: &[f64]| problem.evaluate(x)[0]),
            Box::new(|x: &[f64]| problem.evaluate(x)[1]),
        ];
        let bounds = problem.bounds();
        let reference_front = problem.generate_pareto_front(100);

        let config = NSGA2Config {
            quality_metrics_config: Some(QualityMetricsConfig {
                calculate_spacing: true,
                calculate_spread: true,
                reference_front: Some(reference_front),
            }),
            hypervolume_config: Some(HypervolumeConfig {
                reference_point: vec![2.0, 2.0],
            }),
            ..base_config
        };

        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(ProblemResult {
            name: "ZDT3".to_string(),
            description: "Disconnected front".to_string(),
            pareto_size: result.pareto_front.len(),
            time: duration,
            hypervolume: result.hypervolume,
            spacing: result.spacing,
            spread: result.spread,
            igd: result.igd,
        });
        println!("  ✓ Completed in {:?}\n", duration);
    }

    // Display comparison table
    println!("═══════════════════════════════════════════════════════════");
    println!("Performance Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    println!(
        "{:<8} {:<20} {:<10} {:<12} {:<10}",
        "Problem", "Description", "PF Size", "Time (ms)", "HV"
    );
    println!("{}", "─".repeat(70));

    for result in &results {
        println!(
            "{:<8} {:<20} {:<10} {:<12.2} {:<10.4}",
            result.name,
            result.description,
            result.pareto_size,
            result.time.as_secs_f64() * 1000.0,
            result.hypervolume.unwrap_or(0.0)
        );
    }

    println!();
    println!(
        "{:<8} {:<12} {:<12} {:<12}",
        "Problem", "Spacing", "Spread", "IGD"
    );
    println!("{}", "─".repeat(50));

    for result in &results {
        println!(
            "{:<8} {:<12.6} {:<12.6} {:<12.6}",
            result.name,
            result.spacing.unwrap_or(0.0),
            result.spread.unwrap_or(0.0),
            result.igd.unwrap_or(0.0)
        );
    }

    println!();
    println!("Observations:");
    println!("  • ZDT1 (convex): Typically easiest to solve with good convergence");
    println!("  • ZDT2 (non-convex): Tests diversity preservation capabilities");
    println!("  • ZDT3 (disconnected): Most challenging, tests diversity in disconnected regions");

    println!("\n✓ Problem comparison demonstrated successfully\n");
    Ok(())
}

/// Example 5: Advanced Configuration
///
/// Demonstrates advanced parameter tuning and configuration options
/// to optimize NSGA-II performance for specific problem characteristics.
fn example5_advanced_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 5: Advanced Configuration and Parameter Tuning");
    println!("═══════════════════════════════════════════════════════════\n");

    let problem = ZDT1::new(30);
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| problem.evaluate(x)[0]),
        Box::new(|x: &[f64]| problem.evaluate(x)[1]),
    ];
    let bounds = problem.bounds();
    let reference_front = problem.generate_pareto_front(100);

    // Compare different configurations
    struct ConfigTest {
        name: String,
        config: NSGA2Config<f64>,
    }

    let configs = vec![
        ConfigTest {
            name: "Default Configuration".to_string(),
            config: NSGA2Config {
                quality_metrics_config: Some(QualityMetricsConfig {
                    calculate_spacing: true,
                    calculate_spread: true,
                    reference_front: Some(reference_front.clone()),
                }),
                ..Default::default()
            },
        },
        ConfigTest {
            name: "Large Population".to_string(),
            config: NSGA2Config {
                pop_size: 200,
                max_generations: 100,
                quality_metrics_config: Some(QualityMetricsConfig {
                    calculate_spacing: true,
                    calculate_spread: true,
                    reference_front: Some(reference_front.clone()),
                }),
                ..Default::default()
            },
        },
        ConfigTest {
            name: "High Crossover Rate".to_string(),
            config: NSGA2Config {
                pop_size: 100,
                max_generations: 100,
                crossover_rate: 0.95,
                quality_metrics_config: Some(QualityMetricsConfig {
                    calculate_spacing: true,
                    calculate_spread: true,
                    reference_front: Some(reference_front.clone()),
                }),
                ..Default::default()
            },
        },
        ConfigTest {
            name: "High Mutation Rate".to_string(),
            config: NSGA2Config {
                pop_size: 100,
                max_generations: 100,
                mutation_rate: 0.2,
                quality_metrics_config: Some(QualityMetricsConfig {
                    calculate_spacing: true,
                    calculate_spread: true,
                    reference_front: Some(reference_front.clone()),
                }),
                ..Default::default()
            },
        },
    ];

    println!(
        "{:<25} {:<12} {:<12} {:<12}",
        "Configuration", "IGD", "Spacing", "Time (ms)"
    );
    println!("{}", "─".repeat(65));

    for config_test in configs {
        let start = Instant::now();
        let result = nsga2(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config_test.config),
        )?;
        let duration = start.elapsed();

        println!(
            "{:<25} {:<12.6} {:<12.6} {:<12.2}",
            config_test.name,
            result.igd.unwrap_or(0.0),
            result.spacing.unwrap_or(0.0),
            duration.as_secs_f64() * 1000.0
        );
    }

    println!();
    println!("Parameter Tuning Guidelines:");
    println!();
    println!("Population Size:");
    println!("  • Larger populations provide better diversity");
    println!("  • Recommended: 50-200 for most problems");
    println!("  • Trade-off: More evaluations per generation");
    println!();
    println!("Crossover Rate:");
    println!("  • Controls exploitation vs exploration balance");
    println!("  • Typical range: 0.8-0.95");
    println!("  • Higher values: More recombination, faster convergence");
    println!();
    println!("Mutation Rate:");
    println!("  • Maintains diversity and prevents premature convergence");
    println!("  • Typical range: 0.05-0.2");
    println!("  • Higher values: More exploration, slower convergence");
    println!();
    println!("Distribution Indices (eta_c, eta_m):");
    println!("  • Control spread of offspring around parents");
    println!("  • Higher values: Offspring closer to parents");
    println!("  • Typical range: 10-30");

    println!("\n✓ Advanced configuration demonstrated successfully\n");
    Ok(())
}

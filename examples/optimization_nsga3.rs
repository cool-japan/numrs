//! NSGA-III Many-Objective Optimization Examples
//!
//! This example demonstrates comprehensive usage of NSGA-III (Non-dominated Sorting
//! Genetic Algorithm III) for solving many-objective optimization problems (3+ objectives)
//! using the DTLZ (Deb-Thiele-Laumanns-Zitzler) test suite.
//!
//! # NSGA-III vs NSGA-II
//!
//! **NSGA-II**: Best for bi-objective problems (2 objectives)
//! - Uses crowding distance for diversity preservation
//! - Effective for 2-3 objectives
//!
//! **NSGA-III**: Designed for many-objective problems (3+ objectives)
//! - Uses reference points for diversity preservation
//! - Maintains uniform distribution through reference point association
//! - Scales well to 5, 10, or more objectives
//!
//! # Topics Covered
//!
//! 1. **Basic NSGA-III Usage**: Solving DTLZ2 with 3 objectives
//! 2. **Reference Point Generation**: Understanding Das-Dennis method
//! 3. **Scalability Analysis**: Comparing 3, 4, and 5 objective problems
//! 4. **Problem Difficulty**: DTLZ2 (unimodal) vs DTLZ3 (multi-modal)
//! 5. **Performance Tuning**: Optimizing for many-objective scenarios
//! 6. **Convergence Tracking**: Monitoring optimization progress
//!
//! # DTLZ Test Problems
//!
//! - **DTLZ1**: Linear Pareto front, multi-modal (challenging)
//! - **DTLZ2**: Concave/spherical Pareto front, unimodal (baseline)
//! - **DTLZ3**: Concave front, multi-modal (very challenging)
//! - **DTLZ7**: Disconnected Pareto regions (diversity test)
//!
//! Run with: `cargo run --example optimization_nsga3 --release`

#![allow(clippy::type_complexity)]

use numrs2::optimize::nsga3::{nsga3, NSGA3Config};
use numrs2::optimize::test_problems::{TestProblem, DTLZ2, DTLZ3};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   NSGA-III Many-Objective Optimization Examples           ║");
    println!("║                   NumRS2 v0.2.0                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example 1: Basic NSGA-III usage with 3 objectives
    example1_basic_usage()?;

    // Example 2: Understanding reference points
    example2_reference_points()?;

    // Example 3: Scalability analysis (3, 4, 5 objectives)
    example3_scalability_analysis()?;

    // Example 4: Problem difficulty comparison
    example4_problem_difficulty()?;

    // Example 5: Advanced configuration for many-objective optimization
    example5_advanced_configuration()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         All NSGA-III Examples Completed Successfully!      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Example 1: Basic NSGA-III Usage
///
/// Demonstrates the simplest use case of NSGA-III with 3 objectives
/// on the DTLZ2 test problem (concave Pareto front).
fn example1_basic_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 1: Basic NSGA-III Usage (DTLZ2, 3 Objectives)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create DTLZ2 with 3 objectives and 12 variables
    // DTLZ2: M objectives, k=10, n = M + k - 1 = 3 + 10 - 1 = 12 variables
    let n_objectives = 3;
    let n_variables = 12;
    let problem = DTLZ2::new(n_objectives, n_variables);

    println!("Problem Configuration:");
    println!("  Problem type:      DTLZ2 (Concave/Spherical Pareto front)");
    println!("  Objectives:        {}", n_objectives);
    println!("  Variables:         {}", n_variables);
    println!("  Difficulty:        Unimodal (moderate)");
    println!();

    // Define objective functions
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_objectives)
        .map(|i| {
            let prob = problem.clone();
            Box::new(move |x: &[f64]| {
                let obj = prob.evaluate(x);
                obj[i]
            }) as Box<dyn Fn(&[f64]) -> f64>
        })
        .collect();

    let bounds = problem.bounds();

    // Use default NSGA-III configuration
    // Default: pop_size=92 (suitable for 3 objectives with 12 divisions)
    let config = NSGA3Config::default();

    println!("NSGA-III Configuration:");
    println!("  Population size:   {}", config.pop_size);
    println!("  Max generations:   {}", config.max_generations);
    println!("  Crossover rate:    {:.2}", config.crossover_rate);
    println!("  Mutation rate:     {:.2}", config.mutation_rate);
    println!("  Reference divs:    {}", config.n_divisions);
    println!();

    // Calculate expected number of reference points
    // For M objectives and p divisions: C(M+p-1, p)
    let expected_ref_points =
        binomial_coefficient(n_objectives + config.n_divisions - 1, config.n_divisions);
    println!("Expected reference points: ~{}", expected_ref_points);
    println!();

    // Run NSGA-III optimization
    let start = Instant::now();
    let result = nsga3(
        &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
        &bounds,
        Some(config),
    )?;
    let duration = start.elapsed();

    // Display results
    println!("Optimization Results:");
    println!("  Pareto front size:      {}", result.pareto_front.len());
    println!("  Total population size:  {}", result.population.len());
    println!("  Generations executed:   {}", result.generations);
    println!(
        "  Reference points:       {}",
        result.reference_points.len()
    );
    println!("  Execution time:         {:?}", duration);
    println!();

    // Analyze reference point usage
    let mut used_ref_points = 0;
    for ref_point in &result.reference_points {
        if ref_point.niche_count > 0 {
            used_ref_points += 1;
        }
    }

    println!("Reference Point Analysis:");
    println!(
        "  Total reference points:  {}",
        result.reference_points.len()
    );
    println!(
        "  Used reference points:   {} ({:.1}%)",
        used_ref_points,
        (used_ref_points as f64 / result.reference_points.len() as f64) * 100.0
    );
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
        print!("  Solution {}: ", idx + 1);
        for (i, obj) in individual.objectives.iter().enumerate() {
            print!("f{}={:.4}", i + 1, obj);
            if i < individual.objectives.len() - 1 {
                print!(", ");
            }
        }
        println!();
    }

    // Verify spherical constraint for DTLZ2
    // For DTLZ2, all Pareto-optimal solutions lie on a sphere with radius 1
    println!();
    println!("Pareto Front Validation (DTLZ2 sphere constraint):");
    let mut total_radius_error = 0.0;
    for individual in &result.pareto_front {
        let radius_squared: f64 = individual.objectives.iter().map(|&x| x * x).sum();
        let radius = radius_squared.sqrt();
        total_radius_error += (radius - 1.0).abs();
    }
    let avg_radius_error = total_radius_error / result.pareto_front.len() as f64;
    println!(
        "  Average radius error: {:.6} (should be close to 0)",
        avg_radius_error
    );
    println!("  Expected radius: 1.0");

    println!("\n✓ Basic NSGA-III usage demonstrated successfully\n");
    Ok(())
}

/// Example 2: Understanding Reference Points
///
/// Demonstrates how NSGA-III uses reference points for diversity preservation
/// and how the Das-Dennis method generates uniformly distributed points.
fn example2_reference_points() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 2: Reference Point Generation and Usage");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Reference Points in NSGA-III:");
    println!("  • Generated using Das-Dennis method");
    println!("  • Uniformly distributed on unit hyperplane");
    println!("  • Guide population distribution in objective space");
    println!("  • Each solution associated with nearest reference point");
    println!("  • Selection based on niche count (preserves diversity)");
    println!();

    // Test different numbers of divisions
    let n_objectives = 3;
    let division_configs = vec![4, 8, 12, 16];

    println!("Reference Point Counts for Different Division Settings:");
    println!("(3 objectives)");
    println!();
    println!(
        "{:<12} {:<20} {:<15}",
        "Divisions", "Ref Points", "Pop Size Hint"
    );
    println!("{}", "─".repeat(50));

    for n_divs in &division_configs {
        let n_ref_points = binomial_coefficient(n_objectives + n_divs - 1, *n_divs);
        // Population size should be close to reference point count
        let pop_size_hint = ((n_ref_points as f64 / 4.0).ceil() as usize) * 4;
        println!("{:<12} {:<20} {:<15}", n_divs, n_ref_points, pop_size_hint);
    }

    println!();
    println!("Choosing Number of Divisions:");
    println!("  • More divisions = More reference points = Better coverage");
    println!("  • But also requires larger population size");
    println!("  • Typical: 8-16 divisions for 3-4 objectives");
    println!("  • Reduce divisions for 5+ objectives to keep population manageable");
    println!();

    // Run optimization with visualization of reference point usage
    let problem = DTLZ2::new(n_objectives, 12);
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_objectives)
        .map(|i| {
            let prob = problem.clone();
            Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
        })
        .collect();
    let bounds = problem.bounds();

    let config = NSGA3Config {
        n_divisions: 12,
        pop_size: 92,
        max_generations: 150,
        ..Default::default()
    };

    println!("Running NSGA-III with {} divisions...", config.n_divisions);
    let start = Instant::now();
    let result = nsga3(
        &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
        &bounds,
        Some(config),
    )?;
    let duration = start.elapsed();

    println!("  ✓ Completed in {:?}", duration);
    println!();

    // Analyze reference point distribution
    println!("Reference Point Usage Analysis:");
    let mut niche_counts: Vec<usize> = result
        .reference_points
        .iter()
        .map(|rp| rp.niche_count)
        .collect();
    niche_counts.sort_unstable();

    let total_niches: usize = niche_counts.iter().sum();
    let min_niche = niche_counts.first().copied().unwrap_or(0);
    let max_niche = niche_counts.last().copied().unwrap_or(0);
    let avg_niche = if !niche_counts.is_empty() {
        total_niches as f64 / niche_counts.len() as f64
    } else {
        0.0
    };

    println!(
        "  Total reference points:  {}",
        result.reference_points.len()
    );
    println!("  Min niche count:         {}", min_niche);
    println!("  Max niche count:         {}", max_niche);
    println!("  Average niche count:     {:.2}", avg_niche);
    println!("  Total solutions:         {}", total_niches);
    println!();

    // Show distribution of niche counts
    let empty_niches = niche_counts.iter().filter(|&&c| c == 0).count();
    let low_niches = niche_counts.iter().filter(|&&c| c > 0 && c <= 2).count();
    let med_niches = niche_counts.iter().filter(|&&c| c > 2 && c <= 5).count();
    let high_niches = niche_counts.iter().filter(|&&c| c > 5).count();

    println!("Niche Count Distribution:");
    println!(
        "  Empty (0):       {} ({:.1}%)",
        empty_niches,
        (empty_niches as f64 / result.reference_points.len() as f64) * 100.0
    );
    println!(
        "  Low (1-2):       {} ({:.1}%)",
        low_niches,
        (low_niches as f64 / result.reference_points.len() as f64) * 100.0
    );
    println!(
        "  Medium (3-5):    {} ({:.1}%)",
        med_niches,
        (med_niches as f64 / result.reference_points.len() as f64) * 100.0
    );
    println!(
        "  High (6+):       {} ({:.1}%)",
        high_niches,
        (high_niches as f64 / result.reference_points.len() as f64) * 100.0
    );
    println!();

    println!("Interpretation:");
    if empty_niches as f64 / result.reference_points.len() as f64 > 0.5 {
        println!("  ⚠ Many empty reference points - consider fewer divisions");
    } else if high_niches as f64 / result.reference_points.len() as f64 > 0.3 {
        println!("  ⚠ Many crowded reference points - consider more divisions");
    } else {
        println!("  ✓ Good distribution of solutions across reference points");
    }

    println!("\n✓ Reference point analysis demonstrated successfully\n");
    Ok(())
}

/// Example 3: Scalability Analysis
///
/// Demonstrates how NSGA-III scales with increasing number of objectives
/// by solving DTLZ2 with 3, 4, and 5 objectives.
fn example3_scalability_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 3: Scalability Analysis (3, 4, 5 Objectives)");
    println!("═══════════════════════════════════════════════════════════\n");

    struct ScalabilityResult {
        n_objectives: usize,
        n_variables: usize,
        n_divisions: usize,
        pop_size: usize,
        ref_points: usize,
        pareto_size: usize,
        time: std::time::Duration,
    }

    let mut results = Vec::new();

    // Test with 3, 4, and 5 objectives
    let objective_counts = vec![3, 4, 5];

    for &n_obj in &objective_counts {
        println!("Testing DTLZ2 with {} objectives...", n_obj);

        // DTLZ2: n = M + k - 1 where k=10
        let n_vars = n_obj + 10 - 1;
        let problem = DTLZ2::new(n_obj, n_vars);

        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_obj)
            .map(|i| {
                let prob = problem.clone();
                Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let bounds = problem.bounds();

        // Adjust divisions and population size for objective count
        // Fewer divisions for more objectives to keep population manageable
        let (n_divs, pop_size) = match n_obj {
            3 => (12, 92),
            4 => (8, 120),
            5 => (6, 126),
            _ => (8, 100),
        };

        let config = NSGA3Config {
            pop_size,
            max_generations: 200,
            n_divisions: n_divs,
            ..Default::default()
        };

        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        results.push(ScalabilityResult {
            n_objectives: n_obj,
            n_variables: n_vars,
            n_divisions: n_divs,
            pop_size,
            ref_points: result.reference_points.len(),
            pareto_size: result.pareto_front.len(),
            time: duration,
        });

        println!("  ✓ Completed in {:?}", duration);
        println!("    Pareto front: {} solutions", result.pareto_front.len());
        println!("    Reference points: {}", result.reference_points.len());
        println!();
    }

    // Display comparison table
    println!("═══════════════════════════════════════════════════════════");
    println!("Scalability Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    println!(
        "{:<10} {:<10} {:<10} {:<12} {:<10} {:<12}",
        "Objectives", "Variables", "Divisions", "Pop Size", "Ref Pts", "PF Size"
    );
    println!("{}", "─".repeat(70));

    for result in &results {
        println!(
            "{:<10} {:<10} {:<10} {:<12} {:<10} {:<12}",
            result.n_objectives,
            result.n_variables,
            result.n_divisions,
            result.pop_size,
            result.ref_points,
            result.pareto_size
        );
    }

    println!();
    println!(
        "{:<10} {:<15} {:<15}",
        "Objectives", "Time (sec)", "Time per Gen (ms)"
    );
    println!("{}", "─".repeat(45));

    for result in &results {
        let time_per_gen = result.time.as_secs_f64() * 1000.0 / 200.0;
        println!(
            "{:<10} {:<15.2} {:<15.2}",
            result.n_objectives,
            result.time.as_secs_f64(),
            time_per_gen
        );
    }

    println!();
    println!("Scalability Observations:");
    println!("  • Computational cost increases with more objectives");
    println!("  • More objectives require more reference points");
    println!("  • Divisions reduced for 5+ objectives to manage population size");
    println!("  • NSGA-III scales better than NSGA-II for many objectives");

    println!("\n✓ Scalability analysis demonstrated successfully\n");
    Ok(())
}

/// Example 4: Problem Difficulty Comparison
///
/// Compares NSGA-III performance on problems with different difficulty levels:
/// DTLZ2 (unimodal, easy) vs DTLZ3 (multi-modal, hard).
fn example4_problem_difficulty() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 4: Problem Difficulty (DTLZ2 vs DTLZ3)");
    println!("═══════════════════════════════════════════════════════════\n");

    let n_objectives = 3;
    let n_variables = 12;

    struct ProblemResult {
        name: String,
        description: String,
        pareto_size: usize,
        time: std::time::Duration,
        avg_rank: f64,
    }

    let mut results = Vec::new();

    // Common configuration
    let config = NSGA3Config {
        pop_size: 92,
        max_generations: 250,
        n_divisions: 12,
        ..Default::default()
    };

    // Test DTLZ2 (Unimodal - Easy)
    {
        println!("Testing DTLZ2 (Unimodal, Easy)...");
        let problem = DTLZ2::new(n_objectives, n_variables);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_objectives)
            .map(|i| {
                let prob = problem.clone();
                Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let bounds = problem.bounds();

        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config.clone()),
        )?;
        let duration = start.elapsed();

        let avg_rank = result
            .population
            .iter()
            .map(|ind| ind.rank as f64)
            .sum::<f64>()
            / result.population.len() as f64;

        results.push(ProblemResult {
            name: "DTLZ2".to_string(),
            description: "Unimodal, concave front".to_string(),
            pareto_size: result.pareto_front.len(),
            time: duration,
            avg_rank,
        });
        println!("  ✓ Completed in {:?}", duration);
        println!(
            "    Pareto front: {} solutions\n",
            result.pareto_front.len()
        );
    }

    // Test DTLZ3 (Multi-modal - Hard)
    {
        println!("Testing DTLZ3 (Multi-modal, Hard)...");
        let problem = DTLZ3::new(n_objectives, n_variables);
        let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_objectives)
            .map(|i| {
                let prob = problem.clone();
                Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
            })
            .collect();
        let bounds = problem.bounds();

        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config),
        )?;
        let duration = start.elapsed();

        let avg_rank = result
            .population
            .iter()
            .map(|ind| ind.rank as f64)
            .sum::<f64>()
            / result.population.len() as f64;

        results.push(ProblemResult {
            name: "DTLZ3".to_string(),
            description: "Multi-modal, concave front".to_string(),
            pareto_size: result.pareto_front.len(),
            time: duration,
            avg_rank,
        });
        println!("  ✓ Completed in {:?}", duration);
        println!(
            "    Pareto front: {} solutions\n",
            result.pareto_front.len()
        );
    }

    // Display comparison
    println!("═══════════════════════════════════════════════════════════");
    println!("Problem Difficulty Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    println!(
        "{:<10} {:<30} {:<10} {:<12} {:<10}",
        "Problem", "Description", "PF Size", "Time (sec)", "Avg Rank"
    );
    println!("{}", "─".repeat(82));

    for result in &results {
        println!(
            "{:<10} {:<30} {:<10} {:<12.2} {:<10.2}",
            result.name,
            result.description,
            result.pareto_size,
            result.time.as_secs_f64(),
            result.avg_rank
        );
    }

    println!();
    println!("Problem Characteristics:");
    println!();
    println!("DTLZ2 (Unimodal):");
    println!("  • Single global Pareto-optimal front");
    println!("  • No local fronts or deceptive regions");
    println!("  • Moderate difficulty - good baseline");
    println!();
    println!("DTLZ3 (Multi-modal):");
    println!("  • 3^k local Pareto-optimal fronts (k=10 → 59,049 local fronts!)");
    println!("  • Very challenging for optimization algorithms");
    println!("  • Tests ability to escape local optima");
    println!("  • May require more generations or larger population");

    println!("\n✓ Problem difficulty comparison demonstrated successfully\n");
    Ok(())
}

/// Example 5: Advanced Configuration
///
/// Demonstrates advanced parameter tuning for many-objective optimization
/// and best practices for NSGA-III configuration.
fn example5_advanced_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 5: Advanced Configuration for Many Objectives");
    println!("═══════════════════════════════════════════════════════════\n");

    let n_objectives = 4;
    let n_variables = 13;
    let problem = DTLZ2::new(n_objectives, n_variables);

    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..n_objectives)
        .map(|i| {
            let prob = problem.clone();
            Box::new(move |x: &[f64]| prob.evaluate(x)[i]) as Box<dyn Fn(&[f64]) -> f64>
        })
        .collect();
    let bounds = problem.bounds();

    struct ConfigTest {
        name: String,
        config: NSGA3Config<f64>,
    }

    let configs = vec![
        ConfigTest {
            name: "Default (4 obj)".to_string(),
            config: NSGA3Config {
                pop_size: 120,
                max_generations: 150,
                n_divisions: 8,
                ..Default::default()
            },
        },
        ConfigTest {
            name: "More Divisions".to_string(),
            config: NSGA3Config {
                pop_size: 165,
                max_generations: 150,
                n_divisions: 10,
                ..Default::default()
            },
        },
        ConfigTest {
            name: "High Crossover".to_string(),
            config: NSGA3Config {
                pop_size: 120,
                max_generations: 150,
                n_divisions: 8,
                crossover_rate: 0.95,
                ..Default::default()
            },
        },
        ConfigTest {
            name: "High Mutation".to_string(),
            config: NSGA3Config {
                pop_size: 120,
                max_generations: 150,
                n_divisions: 8,
                mutation_rate: 0.15,
                ..Default::default()
            },
        },
    ];

    println!(
        "{:<20} {:<12} {:<12} {:<15}",
        "Configuration", "PF Size", "Ref Points", "Time (sec)"
    );
    println!("{}", "─".repeat(65));

    for config_test in configs {
        let start = Instant::now();
        let result = nsga3(
            &objectives.iter().map(|f| f.as_ref()).collect::<Vec<_>>(),
            &bounds,
            Some(config_test.config),
        )?;
        let duration = start.elapsed();

        println!(
            "{:<20} {:<12} {:<12} {:<15.2}",
            config_test.name,
            result.pareto_front.len(),
            result.reference_points.len(),
            duration.as_secs_f64()
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("NSGA-III Configuration Guidelines");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Number of Divisions:");
    println!("  • Controls reference point density");
    println!("  • More objectives → Fewer divisions needed");
    println!("  • Recommended:");
    println!("    - 3 objectives:  10-16 divisions");
    println!("    - 4-5 objectives: 6-10 divisions");
    println!("    - 6+ objectives:  4-6 divisions");
    println!();

    println!("Population Size:");
    println!("  • Should match or slightly exceed reference point count");
    println!("  • Formula: C(M+p-1, p) where M=objectives, p=divisions");
    println!("  • Larger populations improve diversity");
    println!("  • Trade-off with computational cost");
    println!();

    println!("Generations:");
    println!("  • Many-objective problems need more generations");
    println!("  • Recommended: 200-400 for 3-5 objectives");
    println!("  • Multi-modal problems (DTLZ3) need even more");
    println!();

    println!("Crossover and Mutation:");
    println!("  • Similar guidelines to NSGA-II");
    println!("  • Crossover rate: 0.9-0.95");
    println!("  • Mutation rate: 0.1-0.2");
    println!("  • Higher mutation for multi-modal problems");

    println!("\n✓ Advanced configuration demonstrated successfully\n");
    Ok(())
}

/// Helper function to calculate binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

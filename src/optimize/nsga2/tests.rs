//! Tests for the NSGA-II implementation.

use super::*;
use approx::assert_relative_eq;

type ObjectiveFns = Vec<Box<dyn Fn(&[f64]) -> f64>>;

#[test]
fn test_nsga2_simple_biobjective() {
    // Simple bi-objective: minimize f1 = x^2, f2 = (x-2)^2
    // Pareto front: x in [0, 2]
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 50,
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.pareto_front.len() <= 50);

    // Check that solutions are in expected range
    for ind in &result.pareto_front {
        assert!(ind.variables[0] >= 0.0);
        assert!(ind.variables[0] <= 3.0);
    }
}

#[test]
fn test_nsga2_dominance() {
    let a = vec![1.0, 2.0];
    let b = vec![2.0, 3.0];
    let c = vec![1.5, 1.5];

    assert!(dominates(&a, &b)); // a dominates b (better in all objectives)
    assert!(!dominates(&b, &a));
    assert!(!dominates(&a, &c)); // Neither dominates
    assert!(!dominates(&c, &a));
}

#[test]
fn test_nsga2_crowding_distance() {
    let mut population = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![0.0, 4.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![4.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
        },
    ];

    crowding_distance_assignment(&mut population, 2);

    // Boundary solutions should have infinite distance
    assert!(
        population[0].crowding_distance.is_infinite()
            || population[2].crowding_distance.is_infinite()
    );
}

#[test]
fn test_nsga2_invalid_objectives() {
    let objectives: ObjectiveFns = vec![Box::new(|x: &[f64]| x[0])];

    let bounds = vec![(0.0, 1.0)];

    let result = nsga2(&objectives, &bounds, None);
    assert!(result.is_err());
}

#[test]
fn test_nsga2_invalid_pop_size() {
    let objectives: ObjectiveFns =
        vec![Box::new(|x: &[f64]| x[0]), Box::new(|x: &[f64]| 1.0 - x[0])];

    let bounds = vec![(0.0, 1.0)];

    let config = NSGA2Config {
        pop_size: 3, // Invalid: must be even
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config));
    assert!(result.is_err());
}

#[test]
fn test_nsga2_three_objectives() {
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)),
    ];

    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
}

// Hypervolume tests
#[test]
fn test_hypervolume_2d_simple() {
    // Simple 2D case with known result
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
    let reference = vec![4.0, 4.0];

    let hv = calculate_hypervolume(&front, &reference).expect("Hypervolume should succeed");

    // Hypervolume calculation using sweep-line algorithm (right to left):
    // Point (3,1): width = 4-3 = 1, height = 4-1 = 3, area = 3
    // Point (2,2): width = 3-2 = 1, height = 4-2 = 2, area = 2
    // Point (1,3): width = 2-1 = 1, height = 4-3 = 1, area = 1
    // Total hypervolume = 3 + 2 + 1 = 6.0
    assert_relative_eq!(hv, 6.0, epsilon = 1e-6);
}

#[test]
fn test_hypervolume_2d_single_point() {
    let front = vec![vec![1.0, 2.0]];
    let reference = vec![5.0, 6.0];

    let hv = calculate_hypervolume(&front, &reference).expect("Hypervolume should succeed");

    // Expected: (5-1)*(6-2) = 4*4 = 16
    assert_relative_eq!(hv, 16.0, epsilon = 1e-6);
}

#[test]
fn test_hypervolume_3d_simple() {
    let front = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
    let reference = vec![3.0, 3.0, 3.0];

    let hv = calculate_hypervolume(&front, &reference).expect("Hypervolume should succeed");

    // This should be greater than zero
    assert!(hv > 0.0);
}

#[test]
fn test_hypervolume_4d() {
    let front = vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]];
    let reference = vec![3.0, 3.0, 3.0, 3.0];

    let hv = calculate_hypervolume(&front, &reference).expect("Hypervolume should succeed");

    // This should be greater than zero
    assert!(hv > 0.0);
}

#[test]
fn test_hypervolume_empty_front() {
    let front: Vec<Vec<f64>> = vec![];
    let reference = vec![5.0, 5.0];

    let result = calculate_hypervolume(&front, &reference);
    assert!(result.is_err());
}

#[test]
fn test_hypervolume_invalid_reference_point() {
    // Reference point doesn't dominate all points
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let reference = vec![1.5, 1.5]; // Doesn't dominate (2.0, 1.0)

    let result = calculate_hypervolume(&front, &reference);
    assert!(result.is_err());
}

#[test]
fn test_hypervolume_dimension_mismatch() {
    let front = vec![vec![1.0, 2.0]];
    let reference = vec![5.0, 5.0, 5.0]; // Wrong dimension

    let result = calculate_hypervolume(&front, &reference);
    assert!(result.is_err());
}

#[test]
fn test_nsga2_with_hypervolume() {
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 50,
        hypervolume_config: Some(HypervolumeConfig {
            reference_point: vec![10.0, 10.0],
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.hypervolume.is_some());
    assert!(result.hypervolume.expect("Hypervolume should exist") > 0.0);
}

#[test]
fn test_hypervolume_monotonicity() {
    // Adding a dominated point should not decrease hypervolume
    let front1 = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let front2 = vec![vec![1.0, 2.0], vec![1.5, 1.5], vec![2.0, 1.0]];
    let reference = vec![3.0, 3.0];

    let hv1 = calculate_hypervolume(&front1, &reference).expect("HV1 should succeed");
    let hv2 = calculate_hypervolume(&front2, &reference).expect("HV2 should succeed");

    // HV2 should be >= HV1 (adding a point should not decrease hypervolume)
    assert!(hv2 >= hv1);
}

#[test]
fn test_hypervolume_2d_collinear_points() {
    // Test with collinear points (shouldn't break the algorithm)
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
    let reference = vec![4.0, 4.0];

    let hv = calculate_hypervolume(&front, &reference).expect("Hypervolume should succeed");
    assert!(hv > 0.0);
}

// Pareto frontier validation tests
#[test]
fn test_is_pareto_optimal_true() {
    let solution = vec![1.0, 2.0];
    let front = vec![vec![2.0, 1.0], vec![1.5, 1.5]];

    assert!(is_pareto_optimal(&solution, &front));
}

#[test]
fn test_is_pareto_optimal_false() {
    let solution = vec![2.0, 2.0];
    let front = vec![
        vec![1.0, 1.0], // Dominates solution
        vec![1.5, 1.5],
    ];

    assert!(!is_pareto_optimal(&solution, &front));
}

#[test]
fn test_is_pareto_optimal_empty_front() {
    let solution = vec![1.0, 2.0];
    let front: Vec<Vec<f64>> = vec![];

    assert!(is_pareto_optimal(&solution, &front));
}

#[test]
fn test_validate_pareto_front_valid() {
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    assert!(validate_pareto_front(&front).is_ok());
}

#[test]
fn test_validate_pareto_front_invalid_dominated() {
    let front = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0], // Dominated by (1.0, 2.0)
    ];

    assert!(validate_pareto_front(&front).is_err());
}

#[test]
fn test_validate_pareto_front_empty() {
    let front: Vec<Vec<f64>> = vec![];

    assert!(validate_pareto_front(&front).is_err());
}

#[test]
fn test_validate_pareto_front_dimension_mismatch() {
    let front = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0, 0.5], // Different dimension
    ];

    assert!(validate_pareto_front(&front).is_err());
}

#[test]
fn test_validate_pareto_front_single_solution() {
    let front = vec![vec![1.0, 2.0]];

    assert!(validate_pareto_front(&front).is_ok());
}

#[test]
fn test_extract_non_dominated_simple() {
    let solutions = vec![
        vec![1.0, 3.0], // Non-dominated
        vec![2.0, 2.0], // Non-dominated
        vec![3.0, 1.0], // Non-dominated
        vec![2.5, 2.5], // Dominated by (2.0, 2.0)
    ];

    let front = extract_non_dominated(&solutions);
    assert_eq!(front.len(), 3);

    // Verify the front is valid
    assert!(validate_pareto_front(&front).is_ok());
}

#[test]
fn test_extract_non_dominated_all_dominated() {
    let solutions = vec![
        vec![1.0, 1.0], // Non-dominated
        vec![2.0, 2.0], // Dominated
        vec![3.0, 3.0], // Dominated
    ];

    let front = extract_non_dominated(&solutions);
    assert_eq!(front.len(), 1);
    assert_eq!(front[0], vec![1.0, 1.0]);
}

#[test]
fn test_extract_non_dominated_none_dominated() {
    let solutions = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let front = extract_non_dominated(&solutions);
    assert_eq!(front.len(), 3);
}

#[test]
fn test_extract_non_dominated_empty() {
    let solutions: Vec<Vec<f64>> = vec![];

    let front = extract_non_dominated(&solutions);
    assert!(front.is_empty());
}

#[test]
fn test_extract_non_dominated_single() {
    let solutions = vec![vec![1.0, 2.0]];

    let front = extract_non_dominated(&solutions);
    assert_eq!(front.len(), 1);
    assert_eq!(front[0], vec![1.0, 2.0]);
}

#[test]
fn test_extract_non_dominated_three_objectives() {
    let solutions = vec![
        vec![1.0, 2.0, 3.0], // Non-dominated
        vec![2.0, 1.0, 3.0], // Non-dominated
        vec![2.0, 2.0, 2.0], // Non-dominated
        vec![3.0, 3.0, 3.0], // Dominated
    ];

    let front = extract_non_dominated(&solutions);
    assert_eq!(front.len(), 3);
}

// Diversity metrics tests
#[test]
fn test_euclidean_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];

    let dist = metrics::euclidean_distance(&a, &b);
    assert_relative_eq!(dist, 5.0, epsilon = 1e-6);
}

#[test]
fn test_euclidean_distance_same_point() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];

    let dist = metrics::euclidean_distance(&a, &b);
    assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
}

#[test]
fn test_calculate_spacing_uniform() {
    // Perfectly uniform distribution
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let spacing = calculate_spacing(&front).expect("Spacing should succeed");
    assert!(spacing >= 0.0);
    // For uniform distribution, spacing should be relatively small
    assert!(spacing < 1.0);
}

#[test]
fn test_calculate_spacing_clustered() {
    // Clustered points (should have higher spacing)
    let front = vec![vec![1.0, 1.0], vec![1.1, 1.1], vec![5.0, 5.0]];

    let spacing = calculate_spacing(&front).expect("Spacing should succeed");
    assert!(spacing > 0.0);
}

#[test]
fn test_calculate_spacing_too_few_points() {
    let front = vec![vec![1.0, 2.0]];

    assert!(calculate_spacing(&front).is_err());
}

#[test]
fn test_calculate_spacing_dimension_mismatch() {
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0, 0.5]];

    assert!(calculate_spacing(&front).is_err());
}

#[test]
fn test_calculate_spacing_two_points() {
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    let spacing = calculate_spacing(&front).expect("Spacing should succeed");
    assert_relative_eq!(spacing, 0.0, epsilon = 1e-6);
}

#[test]
fn test_find_extreme_points() {
    let front = vec![vec![1.0, 5.0], vec![3.0, 2.0], vec![5.0, 1.0]];

    let extremes = metrics::find_extreme_points_pub(&front);
    assert_eq!(extremes.len(), 2);
    assert_eq!(extremes[0], vec![1.0, 5.0]); // Min in obj 0
    assert_eq!(extremes[1], vec![5.0, 1.0]); // Min in obj 1
}

#[test]
fn test_find_extreme_points_empty() {
    let front: Vec<Vec<f64>> = vec![];

    let extremes = metrics::find_extreme_points_pub(&front);
    assert!(extremes.is_empty());
}

#[test]
fn test_calculate_spread_uniform() {
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let spread = calculate_spread(&front, None).expect("Spread should succeed");
    assert!(spread >= 0.0);
}

#[test]
fn test_calculate_spread_with_extremes() {
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let extremes = vec![vec![0.0, 4.0], vec![4.0, 0.0]];

    let spread = calculate_spread(&front, Some(&extremes)).expect("Spread should succeed");
    assert!(spread >= 0.0);
}

#[test]
fn test_calculate_spread_too_few_points() {
    let front = vec![vec![1.0, 2.0]];

    assert!(calculate_spread(&front, None).is_err());
}

#[test]
fn test_calculate_spread_dimension_mismatch() {
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0, 0.5]];

    assert!(calculate_spread(&front, None).is_err());
}

#[test]
fn test_calculate_spread_two_points() {
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    let spread = calculate_spread(&front, None).expect("Spread should succeed");
    assert!(spread >= 0.0);
}

#[test]
fn test_diversity_metrics_three_objectives() {
    let front = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 1.0, 3.0],
        vec![2.0, 2.0, 2.0],
    ];

    let spacing = calculate_spacing(&front).expect("Spacing should succeed");
    assert!(spacing >= 0.0);

    let spread = calculate_spread(&front, None).expect("Spread should succeed");
    assert!(spread >= 0.0);
}

// Convergence metrics tests
#[test]
fn test_min_distance_to_front() {
    let point = vec![1.5, 1.5];
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    let dist = metrics::min_distance_to_front_pub(&point, &front);
    // Distance to (1.0, 2.0) = sqrt((0.5)^2 + (0.5)^2) = sqrt(0.5) ~ 0.707
    assert_relative_eq!(dist, 0.7071067811865476, epsilon = 1e-6);
}

#[test]
fn test_min_distance_to_front_exact_match() {
    let point = vec![1.0, 2.0];
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    let dist = metrics::min_distance_to_front_pub(&point, &front);
    assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
}

#[test]
fn test_calculate_igd_perfect_convergence() {
    // Identical fronts should have IGD = 0
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let igd = calculate_igd(&front, &front).expect("IGD should succeed");
    assert_relative_eq!(igd, 0.0, epsilon = 1e-6);
}

#[test]
fn test_calculate_igd_partial_convergence() {
    let obtained = vec![vec![1.5, 2.5], vec![2.5, 1.5]];

    let reference = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let igd = calculate_igd(&obtained, &reference).expect("IGD should succeed");
    assert!(igd > 0.0);
    assert!(igd < 2.0); // Should be reasonable
}

#[test]
fn test_calculate_igd_empty_obtained() {
    let obtained: Vec<Vec<f64>> = vec![];
    let reference = vec![vec![1.0, 2.0]];

    assert!(calculate_igd(&obtained, &reference).is_err());
}

#[test]
fn test_calculate_igd_empty_reference() {
    let obtained = vec![vec![1.0, 2.0]];
    let reference: Vec<Vec<f64>> = vec![];

    assert!(calculate_igd(&obtained, &reference).is_err());
}

#[test]
fn test_calculate_igd_dimension_mismatch() {
    let obtained = vec![vec![1.0, 2.0]];
    let reference = vec![vec![1.0, 2.0, 3.0]];

    assert!(calculate_igd(&obtained, &reference).is_err());
}

#[test]
fn test_calculate_gd_perfect_convergence() {
    // Identical fronts should have GD = 0
    let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let gd = calculate_gd(&front, &front, None).expect("GD should succeed");
    assert_relative_eq!(gd, 0.0, epsilon = 1e-6);
}

#[test]
fn test_calculate_gd_partial_convergence() {
    let obtained = vec![vec![1.5, 2.5], vec![2.5, 1.5]];

    let reference = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let gd = calculate_gd(&obtained, &reference, None).expect("GD should succeed");
    assert!(gd > 0.0);
    assert!(gd < 2.0); // Should be reasonable
}

#[test]
fn test_calculate_gd_with_custom_p() {
    let obtained = vec![vec![1.5, 2.5], vec![2.5, 1.5]];

    let reference = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let gd1 = calculate_gd(&obtained, &reference, Some(1.0)).expect("GD p=1 should succeed");
    let gd2 = calculate_gd(&obtained, &reference, Some(2.0)).expect("GD p=2 should succeed");

    // Both should be positive
    assert!(gd1 > 0.0);
    assert!(gd2 > 0.0);
}

#[test]
fn test_calculate_gd_empty_obtained() {
    let obtained: Vec<Vec<f64>> = vec![];
    let reference = vec![vec![1.0, 2.0]];

    assert!(calculate_gd(&obtained, &reference, None).is_err());
}

#[test]
fn test_calculate_gd_empty_reference() {
    let obtained = vec![vec![1.0, 2.0]];
    let reference: Vec<Vec<f64>> = vec![];

    assert!(calculate_gd(&obtained, &reference, None).is_err());
}

#[test]
fn test_calculate_gd_invalid_p() {
    let obtained = vec![vec![1.0, 2.0]];
    let reference = vec![vec![1.0, 2.0]];

    assert!(calculate_gd(&obtained, &reference, Some(-1.0)).is_err());
    assert!(calculate_gd(&obtained, &reference, Some(0.0)).is_err());
}

#[test]
fn test_calculate_gd_dimension_mismatch() {
    let obtained = vec![vec![1.0, 2.0]];
    let reference = vec![vec![1.0, 2.0, 3.0]];

    assert!(calculate_gd(&obtained, &reference, None).is_err());
}

#[test]
fn test_convergence_metrics_three_objectives() {
    let obtained = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 1.0, 3.0],
        vec![2.0, 2.0, 2.0],
    ];

    let reference = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 1.0, 3.0],
        vec![2.0, 2.0, 2.0],
    ];

    let igd = calculate_igd(&obtained, &reference).expect("IGD should succeed");
    assert_relative_eq!(igd, 0.0, epsilon = 1e-6);

    let gd = calculate_gd(&obtained, &reference, None).expect("GD should succeed");
    assert_relative_eq!(gd, 0.0, epsilon = 1e-6);
}

#[test]
fn test_igd_vs_gd_relationship() {
    // Test that IGD and GD measure different aspects
    let obtained = vec![vec![1.5, 2.5]];

    let reference = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let igd = calculate_igd(&obtained, &reference).expect("IGD should succeed");
    let gd = calculate_gd(&obtained, &reference, None).expect("GD should succeed");

    // Both should be positive but may differ
    assert!(igd > 0.0);
    assert!(gd > 0.0);

    // IGD typically larger when obtained front has fewer points than reference
    assert!(igd > gd);
}

// Enhanced Pareto front extraction tests
#[test]
fn test_extract_pareto_front_basic() {
    let population = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![4.0],
            objectives: vec![4.0, 4.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    let front = extract_pareto_front(&population);
    assert_eq!(front.len(), 3);

    // Should be sorted by first objective
    assert!(front[0].objectives[0] <= front[1].objectives[0]);
    assert!(front[1].objectives[0] <= front[2].objectives[0]);
}

#[test]
fn test_extract_pareto_front_empty() {
    let population: Vec<Individual<f64>> = vec![];

    let front = extract_pareto_front(&population);
    assert!(front.is_empty());
}

#[test]
fn test_extract_pareto_front_all_rank0() {
    let population = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
    ];

    let front = extract_pareto_front(&population);
    assert_eq!(front.len(), 2);
}

#[test]
fn test_extract_pareto_front_no_rank0() {
    let population = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 1,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    let front = extract_pareto_front(&population);
    assert!(front.is_empty());
}

#[test]
fn test_extract_front_objectives() {
    let population = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![4.0, 4.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    let objectives = extract_front_objectives(&population);
    assert_eq!(objectives.len(), 2);
    assert_eq!(objectives[0], vec![1.0, 3.0]);
    assert_eq!(objectives[1], vec![2.0, 2.0]);
}

#[test]
fn test_extract_front_objectives_empty() {
    let population: Vec<Individual<f64>> = vec![];

    let objectives = extract_front_objectives(&population);
    assert!(objectives.is_empty());
}

#[test]
fn test_sort_front_by_objective_first() {
    let mut front = vec![
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
    ];

    sort_front_by_objective(&mut front, 0);

    assert_eq!(front[0].objectives[0], 1.0);
    assert_eq!(front[1].objectives[0], 2.0);
    assert_eq!(front[2].objectives[0], 3.0);
}

#[test]
fn test_sort_front_by_objective_second() {
    let mut front = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
    ];

    sort_front_by_objective(&mut front, 1);

    assert_eq!(front[0].objectives[1], 1.0);
    assert_eq!(front[1].objectives[1], 2.0);
    assert_eq!(front[2].objectives[1], 3.0);
}

#[test]
fn test_filter_dominated_solutions_mixed() {
    let individuals = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 3.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    let filtered = filter_dominated_solutions(&individuals);

    // (3.0, 3.0) is dominated by both others
    assert_eq!(filtered.len(), 2);

    // All filtered should have rank 0
    for ind in &filtered {
        assert_eq!(ind.rank, 0);
    }
}

#[test]
fn test_filter_dominated_solutions_none_dominated() {
    let individuals = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
    ];

    let filtered = filter_dominated_solutions(&individuals);
    assert_eq!(filtered.len(), 3);
}

#[test]
fn test_filter_dominated_solutions_all_dominated_by_one() {
    let individuals = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 1.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 2.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 3.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    let filtered = filter_dominated_solutions(&individuals);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].objectives, vec![1.0, 1.0]);
}

#[test]
fn test_filter_dominated_solutions_empty() {
    let individuals: Vec<Individual<f64>> = vec![];

    let filtered = filter_dominated_solutions(&individuals);
    assert!(filtered.is_empty());
}

// Quality metrics integration tests
#[test]
fn test_nsga2_with_quality_metrics() {
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: true,
            reference_front: None,
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.spacing.is_some());
    assert!(result.spread.is_some());
    assert!(result.igd.is_none()); // No reference front
    assert!(result.gd.is_none()); // No reference front
}

#[test]
fn test_nsga2_with_reference_front() {
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    // Generate reference front
    let mut reference_front = Vec::new();
    for i in 0..10 {
        let x = i as f64 * 0.2;
        reference_front.push(vec![x * x, (x - 2.0).powi(2)]);
    }

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: true,
            reference_front: Some(reference_front),
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.spacing.is_some());
    assert!(result.spread.is_some());
    assert!(result.igd.is_some());
    assert!(result.gd.is_some());

    // Verify metrics are non-negative
    assert!(result.spacing.expect("Spacing should exist") >= 0.0);
    assert!(result.spread.expect("Spread should exist") >= 0.0);
    assert!(result.igd.expect("IGD should exist") >= 0.0);
    assert!(result.gd.expect("GD should exist") >= 0.0);
}

#[test]
fn test_nsga2_backward_compatibility() {
    // Test that NSGA-II works without quality metrics config
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.spacing.is_none());
    assert!(result.spread.is_none());
    assert!(result.igd.is_none());
    assert!(result.gd.is_none());
}

#[test]
fn test_nsga2_selective_metrics() {
    // Test enabling only specific metrics
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: false,
            reference_front: None,
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.spacing.is_some());
    assert!(result.spread.is_none());
    assert!(result.igd.is_none());
    assert!(result.gd.is_none());
}

#[test]
fn test_nsga2_quality_metrics_three_objectives() {
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)),
    ];

    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: true,
            reference_front: None,
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.spacing.is_some());
    assert!(result.spread.is_some());
}

// Comprehensive integration tests
#[test]
fn test_quality_metrics_edge_case_single_point() {
    // Test that quality metrics handle single-point fronts correctly
    let front = vec![vec![1.0, 2.0]];

    // Spacing and spread require at least 2 points
    assert!(calculate_spacing(&front).is_err());
    assert!(calculate_spread(&front, None).is_err());

    // But IGD/GD should work
    let reference = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    assert!(calculate_igd(&front, &reference).is_ok());
    assert!(calculate_gd(&front, &reference, None).is_ok());
}

#[test]
fn test_quality_metrics_edge_case_two_points() {
    let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    // All metrics should work
    let spacing = calculate_spacing(&front).expect("Spacing should work with 2 points");
    let spread = calculate_spread(&front, None).expect("Spread should work with 2 points");

    assert!(spacing >= 0.0);
    assert!(spread >= 0.0);

    // For 2 points, spacing should be 0 (perfectly uniform)
    assert_relative_eq!(spacing, 0.0, epsilon = 1e-6);
}

#[test]
fn test_quality_metrics_comparison() {
    // Test that metrics distinguish between good and bad fronts
    let good_front = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 2.0],
        vec![5.0, 1.0],
    ];

    let bad_front = vec![vec![1.0, 5.0], vec![1.1, 4.9], vec![5.0, 1.0]];

    let good_spacing = calculate_spacing(&good_front).expect("Good spacing should succeed");
    let bad_spacing = calculate_spacing(&bad_front).expect("Bad spacing should succeed");

    // Good front should have better (lower) spacing
    assert!(good_spacing < bad_spacing);
}

#[test]
fn test_all_metrics_combined() {
    // Comprehensive test of all metrics together
    let obtained = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

    let reference = vec![
        vec![1.0, 3.0],
        vec![1.5, 2.5],
        vec![2.0, 2.0],
        vec![2.5, 1.5],
        vec![3.0, 1.0],
    ];

    // Diversity metrics
    let spacing = calculate_spacing(&obtained).expect("Spacing should succeed");
    let spread = calculate_spread(&obtained, None).expect("Spread should succeed");

    // Convergence metrics
    let igd = calculate_igd(&obtained, &reference).expect("IGD should succeed");
    let gd = calculate_gd(&obtained, &reference, None).expect("GD should succeed");

    // All should be non-negative
    assert!(spacing >= 0.0);
    assert!(spread >= 0.0);
    assert!(igd >= 0.0);
    assert!(gd >= 0.0);

    // IGD and GD should be small for similar fronts
    assert!(igd < 1.0);
    assert!(gd < 1.0);
}

#[test]
fn test_nsga2_with_all_quality_features() {
    // Test with hypervolume + quality metrics
    let objectives: ObjectiveFns = vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ];

    let bounds = vec![(0.0, 3.0)];

    let mut reference_front = Vec::new();
    for i in 0..10 {
        let x = i as f64 * 0.2;
        reference_front.push(vec![x * x, (x - 2.0).powi(2)]);
    }

    let config = NSGA2Config {
        pop_size: 50,
        max_generations: 30,
        hypervolume_config: Some(HypervolumeConfig {
            reference_point: vec![10.0, 10.0],
        }),
        quality_metrics_config: Some(QualityMetricsConfig {
            calculate_spacing: true,
            calculate_spread: true,
            reference_front: Some(reference_front),
        }),
        ..Default::default()
    };

    let result = nsga2(&objectives, &bounds, Some(config)).expect("NSGA-II should succeed");

    // All metrics should be present
    assert!(result.hypervolume.is_some());
    assert!(result.spacing.is_some());
    assert!(result.spread.is_some());
    assert!(result.igd.is_some());
    assert!(result.gd.is_some());

    // All should be non-negative
    assert!(result.hypervolume.expect("HV should exist") > 0.0);
    assert!(result.spacing.expect("Spacing should exist") >= 0.0);
    assert!(result.spread.expect("Spread should exist") >= 0.0);
    assert!(result.igd.expect("IGD should exist") >= 0.0);
    assert!(result.gd.expect("GD should exist") >= 0.0);
}

#[test]
fn test_metric_consistency_across_runs() {
    // Test that metrics are consistent for the same front
    let front = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 2.0],
        vec![5.0, 1.0],
    ];

    let spacing1 = calculate_spacing(&front).expect("Spacing 1 should succeed");
    let spacing2 = calculate_spacing(&front).expect("Spacing 2 should succeed");

    let spread1 = calculate_spread(&front, None).expect("Spread 1 should succeed");
    let spread2 = calculate_spread(&front, None).expect("Spread 2 should succeed");

    // Should get identical results
    assert_relative_eq!(spacing1, spacing2, epsilon = 1e-10);
    assert_relative_eq!(spread1, spread2, epsilon = 1e-10);
}

#[test]
fn test_metric_validation_with_extraction_functions() {
    // Test that extraction functions work well with metrics
    let population = vec![
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 5.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![2.0, 4.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![3.0, 3.0],
            rank: 0,
            crowding_distance: 0.0,
        },
        Individual {
            variables: vec![4.0],
            objectives: vec![4.0, 4.0],
            rank: 1,
            crowding_distance: 0.0,
        },
    ];

    // Extract front
    let front = extract_pareto_front(&population);
    assert_eq!(front.len(), 3);

    // Extract objectives
    let objectives = extract_front_objectives(&population);
    assert_eq!(objectives.len(), 3);

    // Calculate metrics on extracted objectives
    let spacing = calculate_spacing(&objectives).expect("Spacing should succeed");
    let spread = calculate_spread(&objectives, None).expect("Spread should succeed");

    assert!(spacing >= 0.0);
    assert!(spread >= 0.0);
}

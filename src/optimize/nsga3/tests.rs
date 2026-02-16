#![allow(clippy::type_complexity)]

use super::reference_points::*;
use super::*;
use approx::assert_relative_eq;

#[test]
fn test_nsga3_basic_three_objectives() {
    // Simple three-objective problem
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)),
    ];

    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];

    let config = NSGA3Config {
        pop_size: 50,
        max_generations: 10,
        n_divisions: 6,
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should succeed");

    assert!(!result.pareto_front.is_empty());
    assert!(result.pareto_front.len() <= 50);
    assert!(!result.reference_points.is_empty());
}

#[test]
fn test_nsga3_invalid_objectives() {
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![Box::new(|x: &[f64]| x[0])];

    let bounds = vec![(0.0, 1.0)];

    let result = nsga3(&objectives, &bounds, None);
    assert!(result.is_err());
}

#[test]
fn test_nsga3_invalid_pop_size() {
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> =
        vec![Box::new(|x: &[f64]| x[0]), Box::new(|x: &[f64]| 1.0 - x[0])];

    let bounds = vec![(0.0, 1.0)];

    let config = NSGA3Config {
        pop_size: 2, // Too small
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config));
    assert!(result.is_err());
}

#[test]
fn test_dominance() {
    let a = vec![1.0, 2.0];
    let b = vec![2.0, 3.0];
    let c = vec![1.5, 1.5];

    assert!(dominates(&a, &b)); // a dominates b
    assert!(!dominates(&b, &a));
    assert!(!dominates(&a, &c)); // Neither dominates
    assert!(!dominates(&c, &a));
}

#[test]
fn test_euclidean_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];

    let distance = euclidean_distance(&a, &b).expect("Distance calculation should succeed");
    assert_relative_eq!(distance, 5.0, epsilon = 1e-6);
}

#[test]
fn test_reference_point_generation_basic() {
    let points =
        generate_reference_points::<f64>(3, 5).expect("Reference point generation should succeed");

    assert!(!points.is_empty());

    // Check that each point has 3 dimensions
    for point in &points {
        assert_eq!(point.position.len(), 3);
    }

    // For M=3, p=5: C(3+5-1, 5) = C(7, 5) = 21 points
    assert_eq!(points.len(), 21);
}

#[test]
fn test_reference_points_two_objectives() {
    // For M=2, p=10: C(2+10-1, 10) = C(11, 10) = 11 points
    let points =
        generate_reference_points::<f64>(2, 10).expect("Reference point generation should succeed");

    assert_eq!(points.len(), 11);

    // All points should be normalized (approximately unit length)
    for point in &points {
        let norm: f64 = point.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_reference_points_three_objectives() {
    // For M=3, p=12: C(3+12-1, 12) = C(14, 12) = 91 points
    let points =
        generate_reference_points::<f64>(3, 12).expect("Reference point generation should succeed");

    assert_eq!(points.len(), 91);

    // Verify normalization
    for point in &points {
        let norm: f64 = point.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_reference_points_five_objectives() {
    // For M=5, p=4: C(5+4-1, 4) = C(8, 4) = 70 points
    let points =
        generate_reference_points::<f64>(5, 4).expect("Reference point generation should succeed");

    assert_eq!(points.len(), 70);

    // All points should have 5 dimensions
    for point in &points {
        assert_eq!(point.position.len(), 5);
    }
}

#[test]
fn test_reference_points_uniform_distribution() {
    let points =
        generate_reference_points::<f64>(3, 6).expect("Reference point generation should succeed");

    // Check that reference points cover different regions
    // At least one point should have each objective as dominant
    let mut has_dominant = [false; 3];

    for point in &points {
        for obj in 0..3 {
            // Check if this objective is dominant (> 0.7 after normalization)
            if point.position[obj] > 0.7 {
                has_dominant[obj] = true;
            }
        }
    }

    // Verify all objectives have at least one dominant point
    assert!(
        has_dominant.iter().all(|&x| x),
        "All objectives should have dominant reference points"
    );
}

#[test]
fn test_reference_points_invalid_inputs() {
    // Zero objectives
    let result = generate_reference_points::<f64>(0, 5);
    assert!(result.is_err());

    // Zero divisions
    let result = generate_reference_points::<f64>(3, 0);
    assert!(result.is_err());
}

#[test]
fn test_reference_points_edge_cases() {
    // Single division
    let points = generate_reference_points::<f64>(3, 1).expect("Should work with single division");
    // For M=3, p=1: C(3, 1) = 3 points
    assert_eq!(points.len(), 3);

    // Two objectives, single division
    let points =
        generate_reference_points::<f64>(2, 1).expect("Should work with 2 objectives, 1 division");
    // For M=2, p=1: C(2, 1) = 2 points
    assert_eq!(points.len(), 2);
}

#[test]
fn test_vector_norm() {
    let v = vec![3.0, 4.0];
    let norm = vector_norm(&v).expect("Norm calculation should succeed");
    assert_relative_eq!(norm, 5.0, epsilon = 1e-10);

    let v = vec![1.0, 1.0, 1.0];
    let norm = vector_norm(&v).expect("Norm calculation should succeed");
    assert_relative_eq!(norm, 3.0_f64.sqrt(), epsilon = 1e-10);

    // Empty vector should error
    let v: Vec<f64> = vec![];
    let result = vector_norm(&v);
    assert!(result.is_err());
}

#[test]
fn test_perpendicular_distance_basic() {
    // Point [1, 0] to reference direction [1, 0] should have zero perpendicular distance
    let point = vec![1.0, 0.0];
    let ref_dir = vec![1.0, 0.0];
    let dist =
        perpendicular_distance(&point, &ref_dir).expect("Distance calculation should succeed");
    assert_relative_eq!(dist, 0.0, epsilon = 1e-10);

    // Point [1, 1] to reference direction [1, 0] should have perpendicular distance 1
    let point = vec![1.0, 1.0];
    let ref_dir = vec![1.0, 0.0];
    let dist =
        perpendicular_distance(&point, &ref_dir).expect("Distance calculation should succeed");
    assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
}

#[test]
fn test_perpendicular_distance_3d() {
    // Point [1, 1, 0] to reference direction [1, 0, 0]
    // Projection is [1, 0, 0], so perpendicular is [0, 1, 0] with distance 1
    let point = vec![1.0, 1.0, 0.0];
    let ref_dir = vec![1.0, 0.0, 0.0];
    let dist =
        perpendicular_distance(&point, &ref_dir).expect("Distance calculation should succeed");
    assert_relative_eq!(dist, 1.0, epsilon = 1e-10);

    // Point [1, 1, 1] to diagonal reference [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    let point = vec![1.0, 1.0, 1.0];
    let sqrt3 = 3.0_f64.sqrt();
    let ref_dir = vec![1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3];
    let dist =
        perpendicular_distance(&point, &ref_dir).expect("Distance calculation should succeed");
    // Point is already on the reference line, so distance should be ~0
    assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
}

#[test]
fn test_perpendicular_distance_perpendicular_point() {
    // Point [0, 1] perpendicular to reference [1, 0]
    let point = vec![0.0, 1.0];
    let ref_dir = vec![1.0, 0.0];
    let dist =
        perpendicular_distance(&point, &ref_dir).expect("Distance calculation should succeed");
    assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
}

#[test]
fn test_perpendicular_distance_dimension_mismatch() {
    let point = vec![1.0, 2.0];
    let ref_dir = vec![1.0, 0.0, 0.0]; // Different dimension
    let result = perpendicular_distance(&point, &ref_dir);
    assert!(result.is_err());
}

#[test]
fn test_scalar_projection() {
    // Project [3, 4] onto [1, 0]
    let point = vec![3.0, 4.0];
    let ref_dir = vec![1.0, 0.0];
    let proj = scalar_projection(&point, &ref_dir).expect("Projection should succeed");
    assert_relative_eq!(proj, 3.0, epsilon = 1e-10);

    // Project [1, 1] onto normalized diagonal [1/sqrt(2), 1/sqrt(2)]
    let point = vec![1.0, 1.0];
    let sqrt2 = 2.0_f64.sqrt();
    let ref_dir = vec![1.0 / sqrt2, 1.0 / sqrt2];
    let proj = scalar_projection(&point, &ref_dir).expect("Projection should succeed");
    assert_relative_eq!(proj, sqrt2, epsilon = 1e-10);
}

#[test]
fn test_find_closest_reference_point() {
    // Create reference points at [1,0,0], [0,1,0], [0,0,1]
    let ref_points = vec![
        ReferencePoint {
            position: vec![1.0, 0.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 1.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 0.0, 1.0],
            niche_count: 0,
        },
    ];

    // Point close to first reference direction
    let point = vec![0.9, 0.1, 0.1];
    let (idx, _dist) =
        find_closest_reference_point(&point, &ref_points).expect("Should find closest reference");
    assert_eq!(idx, 0);

    // Point close to second reference direction
    let point = vec![0.1, 0.9, 0.1];
    let (idx, _dist) =
        find_closest_reference_point(&point, &ref_points).expect("Should find closest reference");
    assert_eq!(idx, 1);

    // Point close to third reference direction
    let point = vec![0.1, 0.1, 0.9];
    let (idx, _dist) =
        find_closest_reference_point(&point, &ref_points).expect("Should find closest reference");
    assert_eq!(idx, 2);
}

#[test]
fn test_compute_ideal_point() {
    let individuals = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![1.0, 5.0, 3.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![2.0, 3.0, 4.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.5, 4.0, 2.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
    ];

    let ideal = compute_ideal_point(&individuals, 3);
    assert_relative_eq!(ideal[0], 0.5, epsilon = 1e-10);
    assert_relative_eq!(ideal[1], 3.0, epsilon = 1e-10);
    assert_relative_eq!(ideal[2], 2.0, epsilon = 1e-10);
}

#[test]
fn test_compute_nadir_point() {
    let individuals = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![1.0, 5.0, 3.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![2.0, 3.0, 4.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.5, 4.0, 2.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
    ];

    let nadir = compute_nadir_point(&individuals, 3);
    assert_relative_eq!(nadir[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(nadir[1], 5.0, epsilon = 1e-10);
    assert_relative_eq!(nadir[2], 4.0, epsilon = 1e-10);
}

#[test]
fn test_normalize_objectives() {
    let mut individuals = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![1.0, 5.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![3.0, 1.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
    ];

    normalize_objectives(&mut individuals, 2).expect("Normalization should succeed");

    // After normalization with ideal=[1,1] and nadir=[3,5]:
    // First individual: (1-1)/(3-1) = 0, (5-1)/(5-1) = 1
    assert_relative_eq!(individuals[0].objectives[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(individuals[0].objectives[1], 1.0, epsilon = 1e-10);

    // Second individual: (3-1)/(3-1) = 1, (1-1)/(5-1) = 0
    assert_relative_eq!(individuals[1].objectives[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(individuals[1].objectives[1], 0.0, epsilon = 1e-10);
}

#[test]
fn test_normalize_objectives_degenerate() {
    // All solutions have same value in one objective
    let mut individuals = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![1.0, 5.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![1.0, 3.0],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        },
    ];

    normalize_objectives(&mut individuals, 2).expect("Should handle degenerate case");

    // First objective has no variation (all 1.0), should become 0 after normalization
    assert_relative_eq!(individuals[0].objectives[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(individuals[1].objectives[0], 0.0, epsilon = 1e-10);

    // Second objective varies from 3 to 5, should be normalized
    assert_relative_eq!(individuals[1].objectives[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(individuals[0].objectives[1], 1.0, epsilon = 1e-10);
}

#[test]
fn test_update_niche_counts() {
    let mut ref_points = vec![
        ReferencePoint {
            position: vec![1.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 1.0],
            niche_count: 0,
        },
    ];

    let population = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![0.8, 0.2],
            rank: 0,
            reference_point_index: Some(0),
            perpendicular_distance: 0.1,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![0.7, 0.3],
            rank: 0,
            reference_point_index: Some(0),
            perpendicular_distance: 0.2,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.2, 0.8],
            rank: 0,
            reference_point_index: Some(1),
            perpendicular_distance: 0.15,
        },
    ];

    update_niche_counts(&population, &mut ref_points);

    assert_eq!(ref_points[0].niche_count, 2);
    assert_eq!(ref_points[1].niche_count, 1);
}

#[test]
fn test_niche_preservation_selection_basic() {
    let ref_points = vec![
        ReferencePoint {
            position: vec![1.0, 0.0],
            niche_count: 1, // Already has 1 member
        },
        ReferencePoint {
            position: vec![0.0, 1.0],
            niche_count: 0, // Empty niche
        },
    ];

    let candidates = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![0.8, 0.2],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.1,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![0.2, 0.8],
            rank: 1,
            reference_point_index: Some(1),
            perpendicular_distance: 0.15,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.3, 0.7],
            rank: 1,
            reference_point_index: Some(1),
            perpendicular_distance: 0.2,
        },
    ];

    let selected =
        niche_preservation_selection(candidates, &ref_points, 2).expect("Selection should succeed");

    assert_eq!(selected.len(), 2);

    // First selection should be from niche 1 (empty niche)
    // Should pick the one with smallest perpendicular distance in niche 1
    assert_eq!(selected[0].reference_point_index, Some(1));
    assert_relative_eq!(selected[0].perpendicular_distance, 0.15, epsilon = 1e-10);

    // Second selection could be from either niche (both have count 1 now)
    // Should pick the one with smallest perpendicular distance overall
}

#[test]
fn test_niche_preservation_selection_all_same_niche() {
    let ref_points = vec![
        ReferencePoint {
            position: vec![1.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 1.0],
            niche_count: 0,
        },
    ];

    // All candidates associated with same reference point
    let candidates = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![0.9, 0.1],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.3,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![0.8, 0.2],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.1,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.85, 0.15],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.2,
        },
    ];

    let selected =
        niche_preservation_selection(candidates, &ref_points, 2).expect("Selection should succeed");

    assert_eq!(selected.len(), 2);

    // Should select the two with smallest perpendicular distances
    assert_relative_eq!(selected[0].perpendicular_distance, 0.1, epsilon = 1e-10);
    assert_relative_eq!(selected[1].perpendicular_distance, 0.2, epsilon = 1e-10);
}

#[test]
fn test_niche_preservation_selection_empty_candidates() {
    let ref_points = vec![ReferencePoint {
        position: vec![1.0, 0.0],
        niche_count: 0,
    }];

    let candidates: Vec<Individual<f64>> = vec![];

    let result = niche_preservation_selection(candidates, &ref_points, 2);
    assert!(result.is_err());
}

#[test]
fn test_niche_preservation_selection_more_k_than_candidates() {
    let ref_points = vec![ReferencePoint {
        position: vec![1.0, 0.0],
        niche_count: 0,
    }];

    let candidates = vec![Individual {
        variables: vec![0.0],
        objectives: vec![0.8, 0.2],
        rank: 1,
        reference_point_index: Some(0),
        perpendicular_distance: 0.1,
    }];

    // Request 5 but only 1 available
    let selected = niche_preservation_selection(candidates, &ref_points, 5)
        .expect("Should select all available");

    assert_eq!(selected.len(), 1);
}

#[test]
fn test_niche_preservation_diversity() {
    // Create reference points at corners
    let ref_points = vec![
        ReferencePoint {
            position: vec![1.0, 0.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 1.0, 0.0],
            niche_count: 0,
        },
        ReferencePoint {
            position: vec![0.0, 0.0, 1.0],
            niche_count: 0,
        },
    ];

    // Create candidates clustered near first reference point
    let candidates = vec![
        Individual {
            variables: vec![0.0],
            objectives: vec![0.9, 0.05, 0.05],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.05,
        },
        Individual {
            variables: vec![1.0],
            objectives: vec![0.85, 0.1, 0.05],
            rank: 1,
            reference_point_index: Some(0),
            perpendicular_distance: 0.1,
        },
        Individual {
            variables: vec![2.0],
            objectives: vec![0.1, 0.8, 0.1],
            rank: 1,
            reference_point_index: Some(1),
            perpendicular_distance: 0.08,
        },
        Individual {
            variables: vec![3.0],
            objectives: vec![0.1, 0.1, 0.8],
            rank: 1,
            reference_point_index: Some(2),
            perpendicular_distance: 0.12,
        },
    ];

    let selected =
        niche_preservation_selection(candidates, &ref_points, 3).expect("Selection should succeed");

    assert_eq!(selected.len(), 3);

    // Should select one from each niche for diversity
    let ref_indices: Vec<Option<usize>> = selected
        .iter()
        .map(|ind| ind.reference_point_index)
        .collect();

    // Verify diversity - should have individuals from different niches
    assert!(ref_indices.contains(&Some(0)));
    assert!(ref_indices.contains(&Some(1)));
    assert!(ref_indices.contains(&Some(2)));
}

#[test]
fn test_nsga3_complete_four_objectives() {
    // Four-objective problem to test many-objective optimization
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)),
        Box::new(|x: &[f64]| (x[0] - 0.5).powi(2) + (x[1] + 0.5).powi(2)),
    ];

    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];

    let config = NSGA3Config {
        pop_size: 50,
        max_generations: 20,
        n_divisions: 4,
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config))
        .expect("NSGA-III should succeed for 4 objectives");

    assert!(!result.pareto_front.is_empty(), "Should find Pareto front");
    assert!(
        result.pareto_front.len() <= 50,
        "Pareto front should fit in population"
    );
    assert!(
        !result.reference_points.is_empty(),
        "Should have reference points"
    );
    assert_eq!(result.generations, 20, "Should complete all generations");

    // Verify all solutions are within bounds
    for ind in &result.pareto_front {
        assert!(ind.variables[0] >= 0.0 && ind.variables[0] <= 2.0);
        assert!(ind.variables[1] >= 0.0 && ind.variables[1] <= 2.0);
    }
}

#[test]
fn test_nsga3_vs_nsga2_three_objectives() {
    // Compare NSGA-III with three objectives (where both algorithms work)
    // NSGA-III should maintain better diversity

    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)),
    ];

    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

    // Increased population size and generations for more robust convergence
    let config = NSGA3Config {
        pop_size: 100,
        max_generations: 50,
        n_divisions: 8,
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should succeed");

    assert!(!result.pareto_front.is_empty());

    // Check diversity: reference points should be distributed
    let unique_ref_points: std::collections::HashSet<usize> = result
        .pareto_front
        .iter()
        .filter_map(|ind| ind.reference_point_index)
        .collect();

    // Should have solutions associated with multiple reference points
    // With increased population and generations, this should be reliable
    assert!(
        unique_ref_points.len() > 1,
        "Should have diversity across reference points (got {} unique points)",
        unique_ref_points.len()
    );
}

#[test]
fn test_nsga3_five_objectives_scalability() {
    // Test with 5 objectives to verify many-objective optimization
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| x[2]),
        Box::new(|x: &[f64]| (x[0] + x[1] + x[2]) / 3.0),
        Box::new(|x: &[f64]| (x[0] - x[1]).powi(2) + (x[1] - x[2]).powi(2)),
    ];

    let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

    let config = NSGA3Config {
        pop_size: 100,
        max_generations: 10,
        n_divisions: 3, // Small number of divisions for 5 objectives
        ..Default::default()
    };

    let result =
        nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should handle 5 objectives");

    assert!(!result.pareto_front.is_empty());
    assert!(result.pareto_front.len() <= 100);

    // Verify solutions have all 5 objectives evaluated
    for ind in &result.pareto_front {
        assert_eq!(ind.objectives.len(), 5);
    }
}

#[test]
fn test_nsga3_convergence() {
    // Test that NSGA-III converges toward Pareto front over generations
    // Using a simple bi-objective problem with known Pareto front

    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| 1.0 - x[0].sqrt()),
    ];

    let bounds = vec![(0.0, 1.0)];

    let config = NSGA3Config {
        pop_size: 50,
        max_generations: 50,
        n_divisions: 10,
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config)).expect("NSGA-III should converge");

    assert!(!result.pareto_front.is_empty());

    // Check that solutions are close to true Pareto front: f2 = (1 - sqrt(f1))
    let mut on_pareto_front = 0;
    for ind in &result.pareto_front {
        let f1 = ind.objectives[0];
        let f2 = ind.objectives[1];
        let expected_f2 = 1.0 - f1.sqrt();

        // Check if close to true Pareto front
        if (f2 - expected_f2).abs() < 0.2 {
            on_pareto_front += 1;
        }
    }

    // At least some solutions should be close to true Pareto front
    assert!(
        on_pareto_front > result.pareto_front.len() / 4,
        "Should have converged toward Pareto front"
    );
}

#[test]
fn test_environmental_selection_integration() {
    // Test the complete environmental selection process
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| x[0] + x[1]),
    ];

    // Create a population larger than target
    let mut population = Vec::new();
    for i in 0..20 {
        let x = i as f64 / 20.0;
        let y = (20 - i) as f64 / 20.0;
        population.push(Individual {
            variables: vec![x, y],
            objectives: vec![x, y, x + y],
            rank: 0,
            reference_point_index: None,
            perpendicular_distance: 0.0,
        });
    }

    let ref_points =
        generate_reference_points::<f64>(3, 5).expect("Reference point generation should succeed");

    let selected = environmental_selection(population, &ref_points, 10, 3)
        .expect("Environmental selection should succeed");

    assert_eq!(selected.len(), 10);

    // All selected individuals should have reference point associations
    for ind in &selected {
        assert!(ind.reference_point_index.is_some());
    }
}

#[test]
fn test_reference_points_layered_basic() {
    // Test layered reference point generation for 6-8 objectives
    let points = generate_reference_points_layered::<f64>(6, 3, 2)
        .expect("Layered generation should succeed");

    assert!(!points.is_empty());

    // All points should have 6 dimensions
    for point in &points {
        assert_eq!(point.position.len(), 6);
    }

    // All points should be normalized (approximately unit length)
    for point in &points {
        let norm: f64 = point.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_reference_points_layered_reduces_count() {
    // Layered approach should generate fewer points than full Das-Dennis
    let layered = generate_reference_points_layered::<f64>(7, 3, 2)
        .expect("Layered generation should succeed");

    let full = generate_reference_points::<f64>(7, 6).expect("Full Das-Dennis should succeed");

    // Layered should generate significantly fewer points
    assert!(
        layered.len() < full.len(),
        "Layered ({}) should have fewer points than full Das-Dennis ({})",
        layered.len(),
        full.len()
    );
}

#[test]
fn test_reference_points_random_basic() {
    // Test random reference point generation
    let n_points = 100;
    let points = generate_reference_points_random::<f64>(10, n_points)
        .expect("Random generation should succeed");

    assert_eq!(points.len(), n_points);

    // All points should have 10 dimensions
    for point in &points {
        assert_eq!(point.position.len(), 10);
    }

    // All points should be normalized
    for point in &points {
        let norm: f64 = point.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_reference_points_random_coverage() {
    // Random points should cover the objective space reasonably
    let points =
        generate_reference_points_random::<f64>(3, 100).expect("Random generation should succeed");

    // Check that at least some points are dominant in each objective
    let mut has_dominant = [false; 3];

    for point in &points {
        for obj in 0..3 {
            if point.position[obj] > 0.5 {
                has_dominant[obj] = true;
            }
        }
    }

    // With 100 random points, all objectives should have some dominant points
    assert!(
        has_dominant.iter().all(|&x| x),
        "Random points should cover all objectives"
    );
}

#[test]
fn test_reference_points_adaptive_small() {
    // For small objective counts (1-5), should use Das-Dennis
    let points_3obj = generate_reference_points_adaptive::<f64>(3, 12)
        .expect("Adaptive generation should succeed");

    let expected_3obj = generate_reference_points::<f64>(3, 12).expect("Das-Dennis should succeed");

    assert_eq!(
        points_3obj.len(),
        expected_3obj.len(),
        "For 3 objectives, adaptive should match Das-Dennis"
    );
}

#[test]
fn test_reference_points_adaptive_medium() {
    // For medium objective counts (6-8), should use layered approach
    let points_7obj = generate_reference_points_adaptive::<f64>(7, 12)
        .expect("Adaptive generation should succeed");

    // Should generate fewer points than full Das-Dennis with 12 divisions
    // Layered uses max(4) outer and 6 inner divisions
    assert!(
        !points_7obj.is_empty(),
        "Should generate reference points for 7 objectives"
    );

    // Verify all points are normalized
    for point in &points_7obj {
        let norm: f64 = point.position.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_reference_points_adaptive_large() {
    // For large objective counts (9+), should use random sampling
    let points_10obj = generate_reference_points_adaptive::<f64>(10, 12)
        .expect("Adaptive generation should succeed");

    // Should generate approximately n_divisions^2 points (capped at 1000)
    // For n_divisions=12: min(12*12, 1000) = 144 points
    assert_eq!(points_10obj.len(), 144);

    // All points should have 10 dimensions
    for point in &points_10obj {
        assert_eq!(point.position.len(), 10);
    }
}

#[test]
fn test_reference_points_adaptive_performance() {
    // Test that adaptive selection significantly reduces generation time for many objectives
    use std::time::Instant;

    // 8 objectives with adaptive should be much faster than old approach
    let start = Instant::now();
    let points_adaptive = generate_reference_points_adaptive::<f64>(8, 12)
        .expect("Adaptive generation should succeed");
    let duration_adaptive = start.elapsed();

    // Verify points were generated
    assert!(!points_adaptive.is_empty());

    // Duration should be reasonable (< 1 second for 8 objectives)
    assert!(
        duration_adaptive.as_millis() < 1000,
        "Adaptive generation for 8 objectives should complete in <1s, took {}ms",
        duration_adaptive.as_millis()
    );
}

#[test]
fn test_nsga3_with_adaptive_eight_objectives() {
    // Test NSGA-III with 8 objectives using adaptive reference points
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|x: &[f64]| x[0]),
        Box::new(|x: &[f64]| x[1]),
        Box::new(|x: &[f64]| x[2]),
        Box::new(|x: &[f64]| x[3]),
        Box::new(|x: &[f64]| (x[0] + x[1]) / 2.0),
        Box::new(|x: &[f64]| (x[2] + x[3]) / 2.0),
        Box::new(|x: &[f64]| (x[0] + x[2]) / 2.0),
        Box::new(|x: &[f64]| (x[1] + x[3]) / 2.0),
    ];

    let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

    let config = NSGA3Config {
        pop_size: 100,
        max_generations: 10, // Small number for fast test
        n_divisions: 4,      // Reduced divisions for 8 objectives
        ..Default::default()
    };

    let result = nsga3(&objectives, &bounds, Some(config))
        .expect("NSGA-III should handle 8 objectives with adaptive method");

    assert!(!result.pareto_front.is_empty());
    assert!(result.pareto_front.len() <= 100);

    // Verify all solutions have 8 objectives evaluated
    for ind in &result.pareto_front {
        assert_eq!(ind.objectives.len(), 8);
    }
}

#[test]
fn test_new_reference_methods_invalid_inputs() {
    // Test error handling for invalid inputs in new methods
    let result = generate_reference_points_layered::<f64>(0, 3, 2);
    assert!(result.is_err());

    let result = generate_reference_points_random::<f64>(0, 100);
    assert!(result.is_err());

    let result = generate_reference_points_random::<f64>(5, 0);
    assert!(result.is_err());

    let result = generate_reference_points_adaptive::<f64>(0, 12);
    assert!(result.is_err());
}

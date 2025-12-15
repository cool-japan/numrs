//! Randomized Linear Algebra Example
//!
//! This example demonstrates NumRS2's randomized linear algebra capabilities,
//! including randomized SVD, random projections, and fast low-rank approximations.
//!
//! Run with: cargo run --example randomized_linalg_example

use numrs2::linalg::randomized::*;
use numrs2::math;
use numrs2::prelude::*;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== NumRS2 Randomized Linear Algebra Examples ===\n");

    // Example 1: Basic Matrix Creation for Testing
    println!("1. Creating Test Matrix");
    println!("-----------------------");

    // Create a test matrix
    let m = 100;
    let n = 80;

    let a_data = math::linspace(1.0, (m * n) as f64, m * n);
    let a = a_data.reshape(&[m, n]);

    println!("Matrix size: {}x{}", m, n);
    println!("First few elements: {:?}", &a.to_vec()[0..5]);
    println!("Total elements: {}\n", a.size());

    // Example 2: Random Projections for Dimensionality Reduction
    println!("2. Random Projections for Dimensionality Reduction");
    println!("--------------------------------------------------");

    // High-dimensional data (using smaller sizes for faster execution)
    let high_dim = 500;
    let n_samples = 50;
    let data = math::linspace(1.0, (n_samples * high_dim) as f64, n_samples * high_dim);
    let high_dim_data = data.reshape(&[n_samples, high_dim]);

    println!(
        "Original data: {} samples in {} dimensions",
        n_samples, high_dim
    );

    // Project to lower dimension
    let target_dim = 50;

    // Gaussian random projection
    let start = std::time::Instant::now();
    let projected_gaussian =
        random_projection(&high_dim_data, target_dim, ProjectionType::Gaussian)?;
    let gaussian_time = start.elapsed();

    println!("\nGaussian Random Projection:");
    println!("  Projected to {} dimensions", target_dim);
    println!("  Projection time: {:?}", gaussian_time);
    println!("  Output shape: {:?}", projected_gaussian.shape());

    // Sparse random projection (faster for very high dimensions)
    let start = std::time::Instant::now();
    let projected_sparse = random_projection(&high_dim_data, target_dim, ProjectionType::Sparse)?;
    let sparse_time = start.elapsed();

    println!("\nSparse Random Projection:");
    println!("  Projected to {} dimensions", target_dim);
    println!("  Projection time: {:?}", sparse_time);
    println!("  Output shape: {:?}", projected_sparse.shape());

    // Rademacher random projection (balanced ±1 values)
    let start = std::time::Instant::now();
    let projected_rademacher =
        random_projection(&high_dim_data, target_dim, ProjectionType::Rademacher)?;
    let rademacher_time = start.elapsed();

    println!("\nRademacher Random Projection:");
    println!("  Projected to {} dimensions", target_dim);
    println!("  Projection time: {:?}", rademacher_time);
    println!("  Output shape: {:?}\n", projected_rademacher.shape());

    // Example 3: Randomized Range Finder
    println!("3. Randomized Range Finder");
    println!("--------------------------");

    // Create a matrix to find the range of (using smaller size for faster execution)
    let matrix_size = 100;
    let matrix_data = math::linspace(
        1.0,
        (matrix_size * matrix_size) as f64,
        matrix_size * matrix_size,
    );
    let matrix = matrix_data.reshape(&[matrix_size, matrix_size]);

    println!("Matrix size: {}x{}", matrix_size, matrix_size);

    let range_rank = 20;
    let q = randomized_range_finder(&matrix, range_rank, 1)?;

    println!("Computed orthonormal basis Q for range approximation");
    println!("  Q shape: {:?}", q.shape());
    println!("  Approximation rank: {}\n", range_rank);

    // Example 4: Low-Rank Approximation for Data Compression
    println!("4. Low-Rank Approximation for Data Compression");
    println!("-----------------------------------------------");

    // Create a "data matrix" (e.g., image-like data)
    let img_height = 100;
    let img_width = 100;
    let img_data = math::linspace(0.0, 255.0, img_height * img_width);
    let image = img_data.reshape(&[img_height, img_width]);

    println!("Image size: {}x{}", img_height, img_width);
    println!("Original storage: {} floats", img_height * img_width);

    // Approximate with different ranks
    for approx_rank in &[5, 10, 20, 50] {
        let approx_matrix = randomized_low_rank_approximation(&image, *approx_rank, 2)?;

        // For visualization, show approximation quality
        println!(
            "  Rank {}: approximated {}x{} matrix",
            approx_rank,
            approx_matrix.shape()[0],
            approx_matrix.shape()[1]
        );
    }
    println!();

    // Example 5: Timing Random Projections
    println!("5. Random Projection Performance");
    println!("---------------------------------");

    let test_samples = 200;
    let test_features = 1000;
    let test_data = math::linspace(
        1.0,
        (test_samples * test_features) as f64,
        test_samples * test_features,
    );
    let test_matrix = test_data.reshape(&[test_samples, test_features]);

    println!("Matrix size: {}x{}", test_samples, test_features);

    let proj_dim = 100;
    let start = std::time::Instant::now();
    let _projected = random_projection(&test_matrix, proj_dim, ProjectionType::Sparse)?;
    let projection_time = start.elapsed();

    println!(
        "Projected to {} dimensions in {:?}",
        proj_dim, projection_time
    );
    println!(
        "  Dimensionality reduction: {}x\n",
        test_features / proj_dim
    );

    // Example 6: Low-Rank Range Finding
    println!("6. Randomized Range Finder");
    println!("--------------------------");

    // Create a matrix for range finding
    let range_m = 100;
    let range_n = 80;
    let range_data = math::linspace(1.0, (range_m * range_n) as f64, range_m * range_n);
    let range_matrix = range_data.reshape(&[range_m, range_n]);

    println!("Matrix size: {}x{}", range_m, range_n);

    let target_rank = 20;
    let start = std::time::Instant::now();
    let q_basis = randomized_range_finder(&range_matrix, target_rank, 2)?;
    let range_time = start.elapsed();

    println!("Computed orthonormal basis Q in {:?}", range_time);
    println!("  Q shape: {:?}", q_basis.shape());
    println!("  Approximation rank: {}\n", target_rank);

    // Example 7: Johnson-Lindenstrauss Lemma Application
    println!("7. Johnson-Lindenstrauss Random Projection");
    println!("-------------------------------------------");

    // High-dimensional points (using smaller sizes for faster execution)
    let n_points = 100;
    let original_dim = 2000;

    println!(
        "Original: {} points in {} dimensions",
        n_points, original_dim
    );

    // JL lemma: for ε=0.3, target dimension ≥ O(log(n)/ε²)
    let epsilon = 0.3;
    let jl_target_dim = (4.0 * (n_points as f64).ln() / (epsilon * epsilon)).ceil() as usize;

    println!(
        "JL lemma with ε={}: target dimension ≥ {}",
        epsilon, jl_target_dim
    );
    println!("Using target dimension: {}", jl_target_dim);

    let jl_data = math::linspace(
        1.0,
        (n_points * original_dim) as f64,
        n_points * original_dim,
    );
    let points = jl_data.reshape(&[n_points, original_dim]);

    let projected_jl = random_projection(&points, jl_target_dim, ProjectionType::Gaussian)?;

    println!("Projected shape: {:?}", projected_jl.shape());
    println!(
        "Dimensionality reduction: {}x",
        original_dim / jl_target_dim
    );
    println!(
        "Approximate pairwise distances preserved within {}%\n",
        epsilon * 100.0
    );

    println!("=== Examples Complete ===");
    Ok(())
}

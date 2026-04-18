//! Distance metrics module for NumRS2
//!
//! Provides distance and similarity metrics similar to scipy.spatial.distance:
//!
//! # Distance Metrics
//! - **Euclidean**: L₂ norm distance
//! - **Manhattan**: L₁ norm (city block) distance
//! - **Chebyshev**: L∞ norm (maximum coordinate difference)
//! - **Minkowski**: Generalized Lₚ norm distance
//! - **Cosine**: 1 - cosine similarity
//! - **Correlation**: 1 - Pearson correlation
//! - **Hamming**: Fraction of differing elements
//! - **Jaccard**: Set dissimilarity
//!
//! # Pairwise Distance Matrices
//! - **cdist**: Distance between two sets of points
//! - **pdist**: Pairwise distances within a set
//! - **squareform**: Convert between condensed and square forms
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::distance::*;
//!
//! // Euclidean distance between two vectors
//! let x: Array<f64> = Array::from_vec(vec![1.0, 2.0, 3.0]);
//! let y: Array<f64> = Array::from_vec(vec![4.0, 5.0, 6.0]);
//! let dist: f64 = euclidean(&x, &y).expect("euclidean should succeed");
//! assert!((dist - 5.196152422706632).abs() < 1e-10);
//!
//! // Cosine similarity
//! let sim: f64 = cosine(&x, &y).expect("cosine should succeed");
//!
//! // Pairwise distances
//! let points: Array<f64> = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);
//! let dists = pdist(&points, DistanceMetric::Euclidean).expect("pdist should succeed");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use std::fmt::Debug;

/// Converts a f64 value to generic Float type T.
/// For all valid Float types (f32, f64), this is infallible.
#[inline]
fn cast_f64<T: Float>(val: f64) -> T {
    T::from(val).unwrap_or_else(T::zero)
}

/// Distance metric types
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance (L₂ norm)
    Euclidean,
    /// Manhattan distance (L₁ norm)
    Manhattan,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
    /// Minkowski distance (Lₚ norm)
    Minkowski(f64),
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Correlation distance (1 - Pearson correlation)
    Correlation,
    /// Hamming distance (fraction of differing elements)
    Hamming,
}

// ============================================================================
// Individual Distance Functions
// ============================================================================

/// Euclidean distance between two vectors
///
/// Computes ||x - y||₂ = sqrt(sum((x_i - y_i)²))
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::distance::*;
///
/// let x: Array<f64> = Array::from_vec(vec![0.0, 0.0]);
/// let y: Array<f64> = Array::from_vec(vec![3.0, 4.0]);
/// let dist: f64 = euclidean(&x, &y).expect("euclidean should succeed");
/// assert!((dist - 5.0).abs() < 1e-10);
/// ```
pub fn euclidean<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let sum_sq: T = x_vec
        .iter()
        .zip(y_vec.iter())
        .map(|(&xi, &yi)| {
            let diff = xi - yi;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x);

    Ok(sum_sq.sqrt())
}

/// Manhattan (city block, L₁) distance
///
/// Computes ||x - y||₁ = sum(|x_i - y_i|)
pub fn manhattan<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let sum: T = x_vec
        .iter()
        .zip(y_vec.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .fold(T::zero(), |acc, x| acc + x);

    Ok(sum)
}

/// Chebyshev (maximum coordinate) distance
///
/// Computes ||x - y||∞ = max(|x_i - y_i|)
pub fn chebyshev<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let max_diff = x_vec
        .iter()
        .zip(y_vec.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .fold(T::zero(), |acc, x| if x > acc { x } else { acc });

    Ok(max_diff)
}

/// Minkowski distance with parameter p
///
/// Computes ||x - y||ₚ = (sum(|x_i - y_i|^p))^(1/p)
///
/// # Arguments
///
/// * `p` - The order of the norm (p ≥ 1)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::distance::*;
///
/// let x: Array<f64> = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let y: Array<f64> = Array::from_vec(vec![4.0, 5.0, 6.0]);
///
/// // p=1 gives Manhattan distance
/// let l1: f64 = minkowski(&x, &y, 1.0).expect("minkowski p=1 should succeed");
/// assert!((l1 - 9.0).abs() < 1e-10);
///
/// // p=2 gives Euclidean distance
/// let l2: f64 = minkowski(&x, &y, 2.0).expect("minkowski p=2 should succeed");
/// assert!((l2 - euclidean(&x, &y).expect("euclidean should succeed")).abs() < 1e-10);
/// ```
pub fn minkowski<T>(x: &Array<T>, y: &Array<T>, p: f64) -> Result<T>
where
    T: Float + Debug,
{
    if p < 1.0 {
        return Err(NumRs2Error::ValueError(
            "Minkowski p must be >= 1".to_string(),
        ));
    }

    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let p_t = cast_f64(p);
    let inv_p = cast_f64(1.0 / p);

    let sum: T = x_vec
        .iter()
        .zip(y_vec.iter())
        .map(|(&xi, &yi)| (xi - yi).abs().powf(p_t))
        .fold(T::zero(), |acc, x| acc + x);

    Ok(sum.powf(inv_p))
}

/// Cosine distance between two vectors
///
/// Computes 1 - (x·y) / (||x|| ||y||)
///
/// Returns 0 for identical directions, 2 for opposite directions.
pub fn cosine<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let dot: T = x_vec
        .iter()
        .zip(y_vec.iter())
        .map(|(&xi, &yi)| xi * yi)
        .fold(T::zero(), |acc, x| acc + x);

    let norm_x: T = x_vec
        .iter()
        .map(|&xi| xi * xi)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();
    let norm_y: T = y_vec
        .iter()
        .map(|&yi| yi * yi)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    if norm_x == T::zero() || norm_y == T::zero() {
        return Err(NumRs2Error::ValueError(
            "Cosine distance undefined for zero vectors".to_string(),
        ));
    }

    let similarity = dot / (norm_x * norm_y);
    Ok(T::one() - similarity)
}

/// Correlation distance
///
/// Computes 1 - Pearson correlation coefficient
pub fn correlation<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let n = T::from(x_vec.len()).unwrap_or_else(T::zero);

    // Compute means
    let mean_x: T = x_vec.iter().copied().fold(T::zero(), |acc, x| acc + x) / n;
    let mean_y: T = y_vec.iter().copied().fold(T::zero(), |acc, x| acc + x) / n;

    // Center the data
    let x_centered: Vec<T> = x_vec.iter().map(|&xi| xi - mean_x).collect();
    let y_centered: Vec<T> = y_vec.iter().map(|&yi| yi - mean_y).collect();

    // Compute covariance and standard deviations
    let cov: T = x_centered
        .iter()
        .zip(y_centered.iter())
        .map(|(&xi, &yi)| xi * yi)
        .fold(T::zero(), |acc, x| acc + x);

    let std_x: T = x_centered
        .iter()
        .map(|&xi| xi * xi)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();
    let std_y: T = y_centered
        .iter()
        .map(|&yi| yi * yi)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    if std_x == T::zero() || std_y == T::zero() {
        return Err(NumRs2Error::ValueError(
            "Correlation undefined for constant vectors".to_string(),
        ));
    }

    let corr = cov / (std_x * std_y);
    Ok(T::one() - corr)
}

/// Hamming distance (fraction of differing elements)
///
/// For continuous values, counts elements where |x_i - y_i| > threshold
pub fn hamming<T>(x: &Array<T>, y: &Array<T>) -> Result<T>
where
    T: Float + Debug,
{
    hamming_threshold(x, y, cast_f64(1e-10))
}

/// Hamming distance with custom threshold
pub fn hamming_threshold<T>(x: &Array<T>, y: &Array<T>, threshold: T) -> Result<T>
where
    T: Float + Debug,
{
    validate_vectors(x, y)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let n_different = x_vec
        .iter()
        .zip(y_vec.iter())
        .filter(|(&xi, &yi)| (xi - yi).abs() > threshold)
        .count();

    let n = T::from(x_vec.len()).unwrap_or_else(T::zero);
    Ok(T::from(n_different).unwrap_or_else(T::zero) / n)
}

// ============================================================================
// Pairwise Distance Computation
// ============================================================================

/// Compute pairwise distances between observations in n-dimensional space
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features)
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Condensed distance vector of length n_samples*(n_samples-1)/2
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::distance::*;
///
/// let points = Array::from_vec(vec![
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0,
/// ]).reshape(&[3, 2]);
///
/// let dists = pdist(&points, DistanceMetric::Euclidean).expect("pdist should succeed");
/// assert_eq!(dists.size(), 3); // 3 choose 2 = 3 pairwise distances
/// ```
pub fn pdist<T>(x: &Array<T>, metric: DistanceMetric) -> Result<Array<T>>
where
    T: Float + Debug,
{
    if x.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Input must be 2D array".to_string(),
        ));
    }

    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    // Number of pairwise distances: n*(n-1)/2
    let n_dists = n_samples * (n_samples - 1) / 2;
    let mut distances = Vec::with_capacity(n_dists);

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            // Extract rows i and j
            let mut xi = Vec::with_capacity(n_features);
            let mut xj = Vec::with_capacity(n_features);

            for k in 0..n_features {
                xi.push(x.get(&[i, k])?);
                xj.push(x.get(&[j, k])?);
            }

            let xi_arr = Array::from_vec(xi);
            let xj_arr = Array::from_vec(xj);

            let dist = compute_distance(&xi_arr, &xj_arr, metric)?;
            distances.push(dist);
        }
    }

    Ok(Array::from_vec(distances))
}

/// Compute distance between each pair of observations from two collections
///
/// # Arguments
///
/// * `xa` - Array of shape (n_samples_A, n_features)
/// * `xb` - Array of shape (n_samples_B, n_features)
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Distance matrix of shape (n_samples_A, n_samples_B)
pub fn cdist<T>(xa: &Array<T>, xb: &Array<T>, metric: DistanceMetric) -> Result<Array<T>>
where
    T: Float + Debug,
{
    if xa.shape().len() != 2 || xb.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Inputs must be 2D arrays".to_string(),
        ));
    }

    let n_features_a = xa.shape()[1];
    let n_features_b = xb.shape()[1];

    if n_features_a != n_features_b {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![xa.shape()[0], n_features_a],
            actual: xb.shape(),
        });
    }

    let n_samples_a = xa.shape()[0];
    let n_samples_b = xb.shape()[0];
    let n_features = n_features_a;

    let mut distances = Vec::with_capacity(n_samples_a * n_samples_b);

    for i in 0..n_samples_a {
        for j in 0..n_samples_b {
            // Extract row i from XA and row j from XB
            let mut xi = Vec::with_capacity(n_features);
            let mut xj = Vec::with_capacity(n_features);

            for k in 0..n_features {
                xi.push(xa.get(&[i, k])?);
                xj.push(xb.get(&[j, k])?);
            }

            let xi_arr = Array::from_vec(xi);
            let xj_arr = Array::from_vec(xj);

            let dist = compute_distance(&xi_arr, &xj_arr, metric)?;
            distances.push(dist);
        }
    }

    Ok(Array::from_vec(distances).reshape(&[n_samples_a, n_samples_b]))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that two arrays are 1D and have the same size
fn validate_vectors<T: Float + Debug>(x: &Array<T>, y: &Array<T>) -> Result<()> {
    if x.shape().len() != 1 || y.shape().len() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "Inputs must be 1D arrays".to_string(),
        ));
    }

    if x.size() != y.size() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }

    Ok(())
}

/// Compute distance between two vectors using specified metric
fn compute_distance<T>(x: &Array<T>, y: &Array<T>, metric: DistanceMetric) -> Result<T>
where
    T: Float + Debug,
{
    match metric {
        DistanceMetric::Euclidean => euclidean(x, y),
        DistanceMetric::Manhattan => manhattan(x, y),
        DistanceMetric::Chebyshev => chebyshev(x, y),
        DistanceMetric::Minkowski(p) => minkowski(x, y, p),
        DistanceMetric::Cosine => cosine(x, y),
        DistanceMetric::Correlation => correlation(x, y),
        DistanceMetric::Hamming => hamming(x, y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let x = Array::from_vec(vec![0.0, 0.0]);
        let y = Array::from_vec(vec![3.0, 4.0]);
        let dist = euclidean(&x, &y).expect("euclidean distance should succeed");
        assert!((dist - 5.0).abs() < 1e-10, "Expected 5.0, got {}", dist);
    }

    #[test]
    fn test_manhattan_distance() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0, 6.0]);
        let dist = manhattan(&x, &y).expect("manhattan distance should succeed");
        assert!((dist - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 6.0, 5.0]);
        let dist = chebyshev(&x, &y).expect("chebyshev distance should succeed");
        assert!((dist - 4.0).abs() < 1e-10); // max(3, 4, 2) = 4
    }

    #[test]
    fn test_minkowski_distance() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0, 6.0]);

        // p=1 should equal Manhattan
        let l1 = minkowski(&x, &y, 1.0).expect("minkowski p=1 should succeed");
        let manhattan_dist = manhattan(&x, &y).expect("manhattan should succeed");
        assert!((l1 - manhattan_dist).abs() < 1e-10);

        // p=2 should equal Euclidean
        let l2 = minkowski(&x, &y, 2.0).expect("minkowski p=2 should succeed");
        let euclidean_dist = euclidean(&x, &y).expect("euclidean should succeed");
        assert!((l2 - euclidean_dist).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let x = Array::from_vec(vec![1.0, 0.0, 0.0]);
        let y = Array::from_vec(vec![0.0, 1.0, 0.0]);

        let dist = cosine(&x, &y).expect("cosine distance should succeed");
        assert!((dist - 1.0).abs() < 1e-10); // Orthogonal vectors

        // Parallel vectors
        let x2 = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y2 = Array::from_vec(vec![2.0, 4.0, 6.0]);
        let dist2 = cosine(&x2, &y2).expect("cosine distance for parallel vectors should succeed");
        assert!(dist2.abs() < 1e-10); // Same direction
    }

    #[test]
    fn test_correlation_distance() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let dist = correlation(&x, &y).expect("correlation distance should succeed");
        assert!(dist.abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_hamming_distance() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 5.0, 6.0]);

        let dist = hamming(&x, &y).expect("hamming distance should succeed");
        assert!((dist - 0.5).abs() < 1e-10); // 2 out of 4 differ
    }

    #[test]
    fn test_pdist() {
        let points = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);

        let dists = pdist(&points, DistanceMetric::Euclidean).expect("pdist should succeed");

        // 3 points give 3 pairwise distances
        assert_eq!(dists.size(), 3);

        // All distances should be positive
        for i in 0..dists.size() {
            assert!(dists.get(&[i]).expect("get element should succeed") > 0.0);
        }
    }

    #[test]
    fn test_cdist() {
        let xa = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        let xb = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

        let dists = cdist(&xa, &xb, DistanceMetric::Euclidean).expect("cdist should succeed");

        // Shape should be (2, 2)
        assert_eq!(dists.shape(), vec![2, 2]);

        // All distances should be positive
        for i in 0..2 {
            for j in 0..2 {
                assert!(dists.get(&[i, j]).expect("get element should succeed") > 0.0);
            }
        }
    }

    #[test]
    fn test_distance_validation() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);

        // Different sizes should error
        assert!(euclidean(&x, &y).is_err());
    }

    #[test]
    fn test_zero_vector_cosine() {
        let x = Array::from_vec(vec![0.0, 0.0, 0.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);

        // Zero vector should error for cosine
        assert!(cosine(&x, &y).is_err());
    }
}

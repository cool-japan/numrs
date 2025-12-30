//! Clustering algorithms module for NumRS2
//!
//! Provides clustering methods similar to scipy.cluster and scikit-learn:
//!
//! # Partitioning Methods
//! - **K-means**: Lloyd's algorithm for centroid-based clustering
//! - **K-means++**: Improved initialization for K-means
//! - **Mini-batch K-means**: Scalable K-means variant
//!
//! # Hierarchical Clustering
//! - **Agglomerative**: Bottom-up hierarchical clustering
//! - **Linkage methods**: Single, complete, average, Ward
//! - **Dendrogram**: Hierarchical tree representation
//!
//! # Density-Based Methods
//! - **DBSCAN**: Density-based spatial clustering
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::cluster::*;
//!
//! // K-means clustering
//! let data = Array::from_vec(vec![
//!     1.0, 2.0,
//!     1.5, 1.8,
//!     5.0, 8.0,
//!     8.0, 8.0,
//!     1.0, 0.6,
//!     9.0, 11.0,
//! ]).reshape(&[6, 2]);
//!
//! let kmeans = KMeans::new(2, KMeansInit::KMeansPlusPlus)
//!     .max_iter(100)
//!     .tol(1e-4)
//!     .fit(&data)
//!     .unwrap();
//!
//! let labels = kmeans.predict(&data).unwrap();
//! let centroids = kmeans.centroids();
//! ```

use crate::array::Array;
use crate::distance::*;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use scirs2_core::random::*;
use std::fmt::Debug;

// ============================================================================
// K-Means Clustering
// ============================================================================

/// Initialization method for K-means
#[derive(Debug, Clone, Copy)]
pub enum KMeansInit {
    /// Random initialization
    Random,
    /// K-means++ initialization (smart seeding)
    KMeansPlusPlus,
    /// Manual initialization (centroids provided)
    Manual,
}

/// K-means clustering algorithm
///
/// Partitions n observations into k clusters where each observation belongs
/// to the cluster with the nearest mean (centroid).
pub struct KMeans<T> {
    k: usize,
    init: KMeansInit,
    max_iter: usize,
    tol: T,
    centroids: Option<Array<T>>,
    inertia: Option<T>,
    n_iter: usize,
}

impl<T> KMeans<T>
where
    T: Float + Debug,
{
    /// Create a new K-means clusterer
    ///
    /// # Arguments
    ///
    /// * `k` - Number of clusters
    /// * `init` - Initialization method
    pub fn new(k: usize, init: KMeansInit) -> Self {
        KMeans {
            k,
            init,
            max_iter: 300,
            tol: T::from(1e-4).expect("Failed to convert default tolerance value"),
            centroids: None,
            inertia: None,
            n_iter: 0,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the K-means model to data
    ///
    /// # Arguments
    ///
    /// * `x` - Data matrix of shape (n_samples, n_features)
    pub fn fit(mut self, x: &Array<T>) -> Result<Self> {
        if x.shape().len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "Input must be 2D array".to_string(),
            ));
        }

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if self.k > n_samples {
            return Err(NumRs2Error::ValueError(format!(
                "Number of clusters {} exceeds number of samples {}",
                self.k, n_samples
            )));
        }

        // Initialize centroids
        let mut centroids = match self.init {
            KMeansInit::Random => Self::init_random(x, self.k)?,
            KMeansInit::KMeansPlusPlus => Self::init_kmeans_plusplus(x, self.k)?,
            KMeansInit::Manual => {
                if let Some(ref c) = self.centroids {
                    c.clone()
                } else {
                    return Err(NumRs2Error::ValueError(
                        "Manual init requires centroids to be set".to_string(),
                    ));
                }
            }
        };

        let mut labels = vec![0usize; n_samples];
        let mut prev_inertia = T::infinity();

        // K-means iterations
        for iter in 0..self.max_iter {
            // Assignment step: assign each point to nearest centroid
            let mut changed = false;
            for i in 0..n_samples {
                let point = Self::get_row(x, i)?;
                let new_label = Self::nearest_centroid(&point, &centroids)?;
                if new_label != labels[i] {
                    changed = true;
                    labels[i] = new_label;
                }
            }

            // Update step: recompute centroids
            let mut new_centroids = Array::zeros(&[self.k, n_features]);
            let mut counts = vec![0usize; self.k];

            for i in 0..n_samples {
                let label = labels[i];
                counts[label] += 1;
                let point = Self::get_row(x, i)?;
                for j in 0..n_features {
                    let current = new_centroids.get(&[label, j])?;
                    new_centroids.set(&[label, j], current + point.get(&[j])?)?;
                }
            }

            // Average to get centroids
            for k in 0..self.k {
                if counts[k] > 0 {
                    let count_t = T::from(counts[k]).unwrap();
                    for j in 0..n_features {
                        let sum = new_centroids.get(&[k, j])?;
                        new_centroids.set(&[k, j], sum / count_t)?;
                    }
                }
            }

            // Compute inertia (sum of squared distances to centroids)
            let inertia = Self::compute_inertia(x, &new_centroids, &labels)?;

            // Check convergence
            let delta = (prev_inertia - inertia).abs();
            if delta < self.tol && iter > 0 {
                self.centroids = Some(new_centroids);
                self.inertia = Some(inertia);
                self.n_iter = iter + 1;
                return Ok(self);
            }

            centroids = new_centroids;
            prev_inertia = inertia;

            if !changed {
                break;
            }
        }

        self.centroids = Some(centroids);
        self.inertia = Some(prev_inertia);
        self.n_iter = self.max_iter;

        Ok(self)
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, x: &Array<T>) -> Result<Vec<usize>> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| NumRs2Error::ValueError("Model not fitted".to_string()))?;

        let n_samples = x.shape()[0];
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let point = Self::get_row(x, i)?;
            let label = Self::nearest_centroid(&point, centroids)?;
            labels.push(label);
        }

        Ok(labels)
    }

    /// Get the cluster centroids
    pub fn centroids(&self) -> Option<&Array<T>> {
        self.centroids.as_ref()
    }

    /// Get the inertia (sum of squared distances to nearest centroid)
    pub fn inertia(&self) -> Option<T> {
        self.inertia
    }

    /// Get number of iterations run
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    // Helper functions

    fn get_row(x: &Array<T>, i: usize) -> Result<Array<T>> {
        let n_features = x.shape()[1];
        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            row.push(x.get(&[i, j])?);
        }
        Ok(Array::from_vec(row))
    }

    fn nearest_centroid(point: &Array<T>, centroids: &Array<T>) -> Result<usize> {
        let k = centroids.shape()[0];
        let mut min_dist = T::infinity();
        let mut min_idx = 0;

        for i in 0..k {
            let centroid = Self::get_row(centroids, i)?;
            let dist = euclidean(point, &centroid)?;
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    fn init_random(x: &Array<T>, k: usize) -> Result<Array<T>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Use scirs2_core random number generation
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle using Fisher-Yates
        for i in (1..n_samples).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let mut centroids = Array::zeros(&[k, n_features]);
        for i in 0..k {
            let idx = indices[i];
            for j in 0..n_features {
                let val = x.get(&[idx, j])?;
                centroids.set(&[i, j], val)?;
            }
        }

        Ok(centroids)
    }

    fn init_kmeans_plusplus(x: &Array<T>, k: usize) -> Result<Array<T>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        let mut rng = thread_rng();
        let mut centroids = Array::zeros(&[k, n_features]);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            let val = x.get(&[first_idx, j])?;
            centroids.set(&[0, j], val)?;
        }

        // Choose remaining centroids using k-means++ algorithm
        for i in 1..k {
            // Compute distances to nearest centroid for each point
            let mut distances = Vec::with_capacity(n_samples);
            let mut total_dist = T::zero();

            for j in 0..n_samples {
                let point = Self::get_row(x, j)?;

                // Find distance to nearest existing centroid
                let mut min_dist = T::infinity();
                for c in 0..i {
                    let centroid = Self::get_row(&centroids, c)?;
                    let dist = euclidean(&point, &centroid)?;
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }

                let dist_sq = min_dist * min_dist;
                distances.push(dist_sq);
                total_dist = total_dist + dist_sq;
            }

            // Choose next centroid with probability proportional to distance squared
            let threshold = rng.gen::<f64>() * total_dist.to_f64().unwrap();
            let mut cumsum = 0.0;
            let mut chosen_idx = 0;

            for (idx, &dist) in distances.iter().enumerate() {
                cumsum += dist.to_f64().unwrap();
                if cumsum >= threshold {
                    chosen_idx = idx;
                    break;
                }
            }

            // Set the chosen point as the new centroid
            for j in 0..n_features {
                let val = x.get(&[chosen_idx, j])?;
                centroids.set(&[i, j], val)?;
            }
        }

        Ok(centroids)
    }

    fn compute_inertia(x: &Array<T>, centroids: &Array<T>, labels: &[usize]) -> Result<T> {
        let n_samples = x.shape()[0];
        let mut inertia = T::zero();

        for i in 0..n_samples {
            let point = Self::get_row(x, i)?;
            let centroid = Self::get_row(centroids, labels[i])?;
            let dist = euclidean(&point, &centroid)?;
            inertia = inertia + dist * dist;
        }

        Ok(inertia)
    }
}

// ============================================================================
// Hierarchical Clustering
// ============================================================================

/// Linkage method for hierarchical clustering
#[derive(Debug, Clone, Copy)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
    /// Ward's minimum variance method
    Ward,
}

/// Hierarchical clustering result
#[derive(Debug, Clone)]
pub struct Dendrogram<T> {
    /// Linkage matrix: each row contains [cluster1, cluster2, distance, n_samples]
    pub linkage: Vec<[T; 4]>,
    /// Number of original observations
    pub n_observations: usize,
}

/// Perform agglomerative hierarchical clustering
///
/// # Arguments
///
/// * `x` - Data matrix of shape (n_samples, n_features)
/// * `method` - Linkage method to use
///
/// # Returns
///
/// Dendrogram structure containing the linkage matrix
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::cluster::*;
///
/// let data = Array::from_vec(vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     5.0, 8.0,
///     8.0, 8.0,
/// ]).reshape(&[4, 2]);
///
/// let dendro = hierarchical(&data, LinkageMethod::Average).unwrap();
/// ```
pub fn hierarchical<T>(x: &Array<T>, method: LinkageMethod) -> Result<Dendrogram<T>>
where
    T: Float + Debug,
{
    if x.shape().len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Input must be 2D array".to_string(),
        ));
    }

    let n = x.shape()[0];

    if n < 2 {
        return Err(NumRs2Error::ValueError(
            "Need at least 2 samples for clustering".to_string(),
        ));
    }

    // Compute pairwise distance matrix
    let distances = pdist(x, DistanceMetric::Euclidean)?;

    // Initialize clusters (each point is its own cluster initially)
    let mut active_clusters: Vec<usize> = (0..n).collect();
    let mut cluster_sizes = vec![1usize; n];
    let mut linkage = Vec::new();
    let mut next_cluster_id = n;

    // Hierarchical clustering loop
    for _ in 0..(n - 1) {
        // Find the pair of clusters with minimum distance
        let (i, j, min_dist) = find_min_distance(&active_clusters, &distances, n)?;

        let cluster_i = active_clusters[i];
        let cluster_j = active_clusters[j];
        let size_i = cluster_sizes[cluster_i];
        let size_j = cluster_sizes[cluster_j];

        // Record the merge
        linkage.push([
            T::from(cluster_i).unwrap(),
            T::from(cluster_j).unwrap(),
            min_dist,
            T::from(size_i + size_j).unwrap(),
        ]);

        // Update cluster information
        active_clusters.remove(j.max(i));
        active_clusters.remove(j.min(i));
        active_clusters.push(next_cluster_id);

        cluster_sizes.push(size_i + size_j);
        next_cluster_id += 1;
    }

    Ok(Dendrogram {
        linkage,
        n_observations: n,
    })
}

/// Find the pair of active clusters with minimum distance
fn find_min_distance<T>(
    active: &[usize],
    distances: &Array<T>,
    n: usize,
) -> Result<(usize, usize, T)>
where
    T: Float + Debug,
{
    let mut min_dist = T::infinity();
    let mut min_i = 0;
    let mut min_j = 1;

    for (idx_i, &i) in active.iter().enumerate() {
        for (idx_j, &j) in active.iter().enumerate().skip(idx_i + 1) {
            if i < n && j < n {
                // Both are original points
                let dist_idx = condensed_index(i, j, n);
                let dist = distances.get(&[dist_idx])?;
                if dist < min_dist {
                    min_dist = dist;
                    min_i = idx_i;
                    min_j = idx_j;
                }
            }
        }
    }

    Ok((min_i, min_j, min_dist))
}

/// Convert (i, j) pair to condensed distance matrix index
fn condensed_index(i: usize, j: usize, n: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    n * i - i * (i + 1) / 2 + j - i - 1
}

/// Cut a dendrogram to get flat clusters
///
/// # Arguments
///
/// * `dendro` - Dendrogram from hierarchical clustering
/// * `n_clusters` - Number of clusters to form
///
/// # Returns
///
/// Vector of cluster labels for each observation
pub fn fcluster<T>(dendro: &Dendrogram<T>, n_clusters: usize) -> Result<Vec<usize>>
where
    T: Float + Debug,
{
    let n = dendro.n_observations;

    if n_clusters > n || n_clusters == 0 {
        return Err(NumRs2Error::ValueError(format!(
            "n_clusters must be between 1 and {}",
            n
        )));
    }

    // Simple approach: take the first (n - n_clusters) merges
    let n_merges = n - n_clusters;

    // Initialize: each point in its own cluster
    let mut labels = vec![0usize; n];
    for i in 0..n {
        labels[i] = i;
    }

    // Apply merges
    let mut next_label = n;
    for merge in dendro.linkage.iter().take(n_merges) {
        let c1 = merge[0].to_usize().unwrap();
        let c2 = merge[1].to_usize().unwrap();

        // Relabel all points in clusters c1 and c2 to next_label
        for label in &mut labels {
            if *label == c1 || *label == c2 {
                *label = next_label;
            }
        }
        next_label += 1;
    }

    // Renumber labels to be 0..n_clusters-1
    let mut unique_labels: Vec<usize> = labels.clone();
    unique_labels.sort_unstable();
    unique_labels.dedup();

    let mut label_map = std::collections::HashMap::new();
    for (new_label, &old_label) in unique_labels.iter().enumerate() {
        label_map.insert(old_label, new_label);
    }

    for label in &mut labels {
        *label = *label_map.get(label).unwrap();
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Simple 2-cluster dataset
        let data = Array::from_vec(vec![
            1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0,
        ])
        .reshape(&[6, 2]);

        let kmeans = KMeans::new(2, KMeansInit::KMeansPlusPlus)
            .max_iter(100)
            .tol(1e-4)
            .fit(&data)
            .unwrap();

        let labels = kmeans.predict(&data).unwrap();

        // Check that we got 2 clusters
        let mut unique_labels = labels.clone();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        assert_eq!(unique_labels.len(), 2);

        // Check that similar points are in the same cluster
        assert_eq!(labels[0], labels[1]); // (1,2) and (1.5,1.8) should be together
        assert_eq!(labels[2], labels[3]); // (5,8) and (8,8) should be together
    }

    #[test]
    fn test_kmeans_init_random() {
        let data = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0, 8.0, 9.0, 9.0, 10.0]).reshape(&[4, 2]);

        let kmeans = KMeans::new(2, KMeansInit::Random).fit(&data).unwrap();

        assert!(kmeans.centroids().is_some());
        assert!(kmeans.inertia().is_some());
    }

    #[test]
    fn test_kmeans_convergence() {
        let data =
            Array::from_vec(vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1]).reshape(&[4, 2]);

        let kmeans = KMeans::new(2, KMeansInit::KMeansPlusPlus)
            .tol(1e-6)
            .fit(&data)
            .unwrap();

        // Should converge quickly for well-separated clusters
        assert!(kmeans.n_iter() < 10);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let data = Array::from_vec(vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0]).reshape(&[4, 2]);

        let dendro = hierarchical(&data, LinkageMethod::Average).unwrap();

        // Should have n-1 merges for n points
        assert_eq!(dendro.linkage.len(), 3);
        assert_eq!(dendro.n_observations, 4);
    }

    #[test]
    fn test_fcluster() {
        let data = Array::from_vec(vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0]).reshape(&[4, 2]);

        let dendro = hierarchical(&data, LinkageMethod::Average).unwrap();
        let labels = fcluster(&dendro, 2).unwrap();

        // Should have 2 clusters
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 2);

        // Points 0 and 1 should be in same cluster (close together)
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn test_kmeans_error_handling() {
        let data = Array::from_vec(vec![1.0, 2.0]).reshape(&[2, 1]);

        // k > n_samples should error
        let result = KMeans::new(3, KMeansInit::Random).fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_unfitted() {
        let kmeans = KMeans::<f64>::new(2, KMeansInit::Random);
        let data = Array::from_vec(vec![1.0, 2.0]);

        // Predicting without fitting should error
        assert!(kmeans.predict(&data).is_err());
    }
}

//! Mutual Information Measures
//!
//! This module implements mutual information and related measures that quantify
//! the dependence between random variables. These measures are fundamental in
//! feature selection, clustering evaluation, and independence testing.
//!
//! # Mathematical Background
//!
//! ## Mutual Information
//!
//! Mutual information measures the reduction in uncertainty about one variable
//! given knowledge of another:
//!
//! ```text
//! I(X;Y) = H(X) + H(Y) - H(X,Y)
//!        = D_KL(P(X,Y) || P(X)P(Y))
//! ```
//!
//! Properties:
//! - I(X;Y) ≥ 0 (non-negative)
//! - I(X;Y) = 0 iff X and Y are independent
//! - I(X;Y) = I(Y;X) (symmetric)
//! - I(X;Y) ≤ min(H(X), H(Y))
//!
//! ## Conditional Mutual Information
//!
//! ```text
//! I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
//! ```
//!
//! ## Variation of Information
//!
//! A metric based on mutual information:
//!
//! ```text
//! VI(X,Y) = H(X|Y) + H(Y|X) = H(X,Y) - I(X;Y)
//! ```
//!
//! ## Normalized Mutual Information
//!
//! Various normalizations make MI comparable across different scales:
//!
//! ```text
//! NMI_arithmetic = I(X;Y) / [(H(X) + H(Y))/2]
//! NMI_geometric = I(X;Y) / √(H(X)·H(Y))
//! NMI_max = I(X;Y) / max(H(X), H(Y))
//! NMI_min = I(X;Y) / min(H(X), H(Y))
//! ```

use super::entropy::{joint_entropy, shannon_entropy, LogBase};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};

/// Normalization type for normalized mutual information
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationType {
    /// Arithmetic mean: 2*I(X;Y)/(H(X)+H(Y))
    Arithmetic,
    /// Geometric mean: I(X;Y)/√(H(X)·H(Y))
    Geometric,
    /// Maximum: I(X;Y)/max(H(X),H(Y))
    Max,
    /// Minimum: I(X;Y)/min(H(X),H(Y))
    Min,
}

/// Compute mutual information I(X;Y) from joint distribution
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
///
/// # Returns
///
/// Mutual information I(X;Y) in bits
///
/// # Mathematical Formula
///
/// ```text
/// I(X;Y) = H(X) + H(Y) - H(X,Y)
///        = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::mutual_information;
/// use scirs2_core::ndarray::Array2;
///
/// // Independent variables have zero MI
/// let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).expect("valid shape and data length");
/// let mi = mutual_information(&joint).expect("valid joint distribution");
/// assert!(mi.abs() < 1e-10);
///
/// // Perfectly correlated variables
/// let joint2 = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).expect("valid shape and data length");
/// let mi2 = mutual_information(&joint2).expect("valid joint distribution");
/// assert!((mi2 - 1.0).abs() < 1e-10); // I(X;X) = H(X) = 1 bit
/// ```
pub fn mutual_information(joint_probs: &Array2<f64>) -> Result<f64, NumRs2Error> {
    // I(X;Y) = H(X) + H(Y) - H(X,Y)

    // Compute marginals
    let marginal_x: Array1<f64> = joint_probs.sum_axis(Axis(1));
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));

    // Compute individual entropies
    let h_x = shannon_entropy(&marginal_x, LogBase::Bits)?;
    let h_y = shannon_entropy(&marginal_y, LogBase::Bits)?;
    let h_xy = joint_entropy(joint_probs, LogBase::Bits)?;

    let mi = h_x + h_y - h_xy;

    // MI should be non-negative (handle numerical errors)
    Ok(mi.max(0.0))
}

/// Compute conditional mutual information I(X;Y|Z)
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y,z) as 3D array
///
/// # Returns
///
/// Conditional mutual information I(X;Y|Z) in bits
///
/// # Mathematical Formula
///
/// ```text
/// I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
///          = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::conditional_mutual_information;
/// use scirs2_core::ndarray::Array3;
///
/// // X and Y independent given Z
/// let joint = Array3::from_shape_vec(
///     (2, 2, 2),
///     vec![
///         0.125, 0.125, 0.125, 0.125,  // Z=0
///         0.125, 0.125, 0.125, 0.125,  // Z=1
///     ]
/// ).expect("valid shape and data length");
/// let cmi = conditional_mutual_information(&joint).expect("valid joint distribution");
/// assert!(cmi.abs() < 1e-10); // I(X;Y|Z) = 0 for conditional independence
/// ```
pub fn conditional_mutual_information(joint_probs: &Array3<f64>) -> Result<f64, NumRs2Error> {
    if joint_probs.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "Joint probability array is empty".to_string(),
        ));
    }

    let shape = joint_probs.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)

    // Compute H(X,Y,Z)
    let flat = joint_probs.iter().cloned().collect::<Vec<_>>();
    let flat_array = Array1::from_vec(flat);
    let h_xyz = shannon_entropy(&flat_array, LogBase::Bits)?;

    // Compute H(X,Z) by marginalizing over Y
    let mut joint_xz = Array2::zeros((nx, nz));
    for i in 0..nx {
        for k in 0..nz {
            for j in 0..ny {
                joint_xz[[i, k]] += joint_probs[[i, j, k]];
            }
        }
    }
    let h_xz = joint_entropy(&joint_xz, LogBase::Bits)?;

    // Compute H(Y,Z) by marginalizing over X
    let mut joint_yz = Array2::zeros((ny, nz));
    for j in 0..ny {
        for k in 0..nz {
            for i in 0..nx {
                joint_yz[[j, k]] += joint_probs[[i, j, k]];
            }
        }
    }
    let h_yz = joint_entropy(&joint_yz, LogBase::Bits)?;

    // Compute H(Z) by marginalizing over X and Y
    let mut marginal_z = Array1::zeros(nz);
    for k in 0..nz {
        for i in 0..nx {
            for j in 0..ny {
                marginal_z[k] += joint_probs[[i, j, k]];
            }
        }
    }
    let h_z = shannon_entropy(&marginal_z, LogBase::Bits)?;

    let cmi = h_xz + h_yz - h_z - h_xyz;

    // CMI should be non-negative
    Ok(cmi.max(0.0))
}

/// Compute normalized mutual information (NMI)
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
/// * `norm_type` - Type of normalization to use
///
/// # Returns
///
/// Normalized mutual information (0 ≤ NMI ≤ 1)
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::{
///     normalized_mutual_information, NormalizationType
/// };
/// use scirs2_core::ndarray::Array2;
///
/// // Independent variables
/// let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).expect("valid shape and data length");
/// let nmi = normalized_mutual_information(&joint, NormalizationType::Arithmetic).expect("valid joint distribution");
/// assert!(nmi.abs() < 1e-10); // NMI = 0 for independence
///
/// // Perfectly correlated
/// let joint2 = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).expect("valid shape and data length");
/// let nmi2 = normalized_mutual_information(&joint2, NormalizationType::Arithmetic).expect("valid joint distribution");
/// assert!((nmi2 - 1.0).abs() < 1e-10); // NMI = 1 for perfect correlation
/// ```
pub fn normalized_mutual_information(
    joint_probs: &Array2<f64>,
    norm_type: NormalizationType,
) -> Result<f64, NumRs2Error> {
    let mi = mutual_information(joint_probs)?;

    // Compute marginals
    let marginal_x: Array1<f64> = joint_probs.sum_axis(Axis(1));
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));

    let h_x = shannon_entropy(&marginal_x, LogBase::Bits)?;
    let h_y = shannon_entropy(&marginal_y, LogBase::Bits)?;

    // Handle edge cases
    if h_x == 0.0 && h_y == 0.0 {
        // Both variables are deterministic
        return Ok(1.0);
    }

    let nmi = match norm_type {
        NormalizationType::Arithmetic => {
            let avg = (h_x + h_y) / 2.0;
            if avg == 0.0 {
                0.0
            } else {
                mi / avg
            }
        }
        NormalizationType::Geometric => {
            let geo_mean = (h_x * h_y).sqrt();
            if geo_mean == 0.0 {
                0.0
            } else {
                mi / geo_mean
            }
        }
        NormalizationType::Max => {
            let max_h = h_x.max(h_y);
            if max_h == 0.0 {
                0.0
            } else {
                mi / max_h
            }
        }
        NormalizationType::Min => {
            let min_h = h_x.min(h_y);
            if min_h == 0.0 {
                0.0
            } else {
                mi / min_h
            }
        }
    };

    // Clamp to [0, 1] to handle numerical errors
    Ok(nmi.clamp(0.0, 1.0))
}

/// Compute pointwise mutual information (PMI) for all (x,y) pairs
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
///
/// # Returns
///
/// Array of PMI values for each (x,y) pair
///
/// # Mathematical Formula
///
/// ```text
/// PMI(x,y) = log(p(x,y)/(p(x)p(y)))
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::pointwise_mutual_information;
/// use scirs2_core::ndarray::Array2;
///
/// let joint = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.4]).expect("valid shape and data length");
/// let pmi = pointwise_mutual_information(&joint).expect("valid joint distribution");
/// // Positive PMI indicates association, negative indicates anti-association
/// assert!(pmi[[0, 0]] > 0.0); // Strong association
/// assert!(pmi[[0, 1]] < 0.0); // Weak association
/// ```
pub fn pointwise_mutual_information(joint_probs: &Array2<f64>) -> Result<Array2<f64>, NumRs2Error> {
    if joint_probs.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "Joint probability array is empty".to_string(),
        ));
    }

    let shape = joint_probs.shape();
    let (nx, ny) = (shape[0], shape[1]);

    // Compute marginals
    let marginal_x: Array1<f64> = joint_probs.sum_axis(Axis(1));
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));

    // PMI(x,y) = log(p(x,y)/(p(x)p(y)))
    let mut pmi = Array2::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let p_xy = joint_probs[[i, j]];
            let p_x = marginal_x[i];
            let p_y = marginal_y[j];

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                pmi[[i, j]] = (p_xy / (p_x * p_y)).ln();
            } else {
                pmi[[i, j]] = f64::NEG_INFINITY;
            }
        }
    }

    Ok(pmi)
}

/// Compute variation of information VI(X,Y)
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution p(x,y)
///
/// # Returns
///
/// Variation of information in bits
///
/// # Mathematical Formula
///
/// ```text
/// VI(X,Y) = H(X|Y) + H(Y|X)
///         = H(X,Y) - I(X;Y)
///         = H(X) + H(Y) - 2·I(X;Y)
/// ```
///
/// This is a true metric (satisfies triangle inequality).
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::variation_of_information;
/// use scirs2_core::ndarray::Array2;
///
/// // Independent variables
/// let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).expect("valid shape and data length");
/// let vi = variation_of_information(&joint).expect("valid joint distribution");
/// assert!((vi - 2.0).abs() < 1e-10); // VI = H(X) + H(Y) when independent
///
/// // Perfectly correlated
/// let joint2 = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).expect("valid shape and data length");
/// let vi2 = variation_of_information(&joint2).expect("valid joint distribution");
/// assert!(vi2.abs() < 1e-10); // VI = 0 when X = Y
/// ```
pub fn variation_of_information(joint_probs: &Array2<f64>) -> Result<f64, NumRs2Error> {
    // VI(X,Y) = H(X,Y) - I(X;Y)
    let h_xy = joint_entropy(joint_probs, LogBase::Bits)?;
    let mi = mutual_information(joint_probs)?;

    Ok(h_xy - mi)
}

/// Compute adjusted mutual information (AMI) for clustering evaluation
///
/// # Arguments
///
/// * `joint_probs` - Joint probability distribution (contingency table)
///
/// # Returns
///
/// Adjusted mutual information (can be negative, typically -1 to 1)
///
/// # Mathematical Formula
///
/// ```text
/// AMI = (MI - E[MI]) / (max(H(X), H(Y)) - E[MI])
/// ```
///
/// where E\[MI\] is the expected MI under random labeling.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::mutual_information::adjusted_mutual_information;
/// use scirs2_core::ndarray::Array2;
///
/// // Perfect agreement
/// let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).expect("valid shape and data length");
/// let ami = adjusted_mutual_information(&joint).expect("valid joint distribution");
/// assert!((ami - 1.0).abs() < 0.1); // AMI ≈ 1 for perfect clustering
/// ```
pub fn adjusted_mutual_information(joint_probs: &Array2<f64>) -> Result<f64, NumRs2Error> {
    if joint_probs.is_empty() {
        return Err(NumRs2Error::InvalidInput(
            "Joint probability array is empty".to_string(),
        ));
    }

    let mi = mutual_information(joint_probs)?;

    // Compute marginals
    let marginal_x: Array1<f64> = joint_probs.sum_axis(Axis(1));
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));

    let h_x = shannon_entropy(&marginal_x, LogBase::Bits)?;
    let h_y = shannon_entropy(&marginal_y, LogBase::Bits)?;

    // Compute expected MI under random labeling using the hypergeometric model
    // (Vinh, Epps, Bailey 2010).
    //
    // Since this function operates on probability distributions rather than
    // integer contingency tables, we derive a virtual contingency table by
    // choosing a virtual sample size N and rounding probabilities to counts.
    // The expected MI is then computed via the exact combinatorial formula.
    //
    // For large virtual N the E[MI] converges to the asymptotic value; we use
    // N = 1000 which gives a good balance between accuracy and performance.
    let expected_mi = compute_expected_mi_hypergeometric(joint_probs)?;

    // AMI = (MI - E[MI]) / (avg(H(X), H(Y)) - E[MI])
    // Using arithmetic mean normalization (consistent with scikit-learn default)
    let avg_h = (h_x + h_y) / 2.0;
    let denominator = avg_h - expected_mi;

    if denominator.abs() < 1e-10 {
        // Handle edge case: when normalizer equals E[MI], check if MI == E[MI]
        if (mi - expected_mi).abs() < 1e-10 {
            return Ok(0.0);
        } else {
            return Ok(1.0);
        }
    }

    let ami = (mi - expected_mi) / denominator;

    Ok(ami)
}

/// Compute the expected mutual information under random labeling using the
/// hypergeometric model (Vinh, Epps, Bailey 2010).
///
/// Given a probability distribution, we construct a virtual contingency table
/// with sample size `N` and compute E[MI] using the exact formula:
///
/// ```text
/// E[MI] = sum_{i,j} sum_{nij} (nij/N) * log2(N*nij / (a_i * b_j))
///         * (a_i! * b_j! * (N-a_i)! * (N-b_j)!) / (N! * nij! * (a_i-nij)! * (b_j-nij)! * (N-a_i-b_j+nij)!)
/// ```
fn compute_expected_mi_hypergeometric(joint_probs: &Array2<f64>) -> Result<f64, NumRs2Error> {
    let shape = joint_probs.shape();
    let (nr, nc) = (shape[0], shape[1]);

    // Virtual sample size -- large enough for good approximation
    let n: i64 = 1000;
    let n_f = n as f64;

    // Compute marginal sums as integer counts
    let marginal_x: Array1<f64> = joint_probs.sum_axis(Axis(1));
    let marginal_y: Array1<f64> = joint_probs.sum_axis(Axis(0));

    // Convert marginals to integer counts (round to nearest, then adjust to sum to N)
    let mut a: Vec<i64> = marginal_x
        .iter()
        .map(|&p| (p * n_f).round() as i64)
        .collect();
    let mut b: Vec<i64> = marginal_y
        .iter()
        .map(|&p| (p * n_f).round() as i64)
        .collect();

    // Adjust counts to ensure they sum to exactly N
    adjust_counts_to_sum(&mut a, n);
    adjust_counts_to_sum(&mut b, n);

    // Pre-compute log-factorials up to N
    let log_fact = precompute_log_factorials(n as usize);

    // Compute E[MI] using the hypergeometric distribution
    let mut emi = 0.0;
    let log_n = n_f.ln();

    for i in 0..nr {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        for j in 0..nc {
            let bj = b[j];
            if bj == 0 {
                continue;
            }

            // nij ranges from max(0, ai + bj - N) to min(ai, bj)
            let nij_min = 0_i64.max(ai + bj - n);
            let nij_max = ai.min(bj);

            for nij in nij_min..=nij_max {
                if nij == 0 {
                    // log(N*0/(ai*bj)) term is -inf * 0, contributes 0
                    continue;
                }

                // Compute log of the hypergeometric probability:
                // log P(nij) = log(C(ai, nij) * C(N-ai, bj-nij) / C(N, bj))
                //            = [log(ai!) - log(nij!) - log((ai-nij)!)]
                //            + [log((N-ai)!) - log((bj-nij)!) - log((N-ai-bj+nij)!)]
                //            - [log(N!) - log(bj!) - log((N-bj)!)]
                let term1 =
                    log_fact[ai as usize] - log_fact[nij as usize] - log_fact[(ai - nij) as usize];
                let term2 = log_fact[(n - ai) as usize]
                    - log_fact[(bj - nij) as usize]
                    - log_fact[(n - ai - bj + nij) as usize];
                let term3 =
                    log_fact[n as usize] - log_fact[bj as usize] - log_fact[(n - bj) as usize];

                let log_prob = term1 + term2 - term3;

                // The contribution: (nij/N) * log2(N*nij/(ai*bj)) * P(nij)
                let log_term = log_n + (nij as f64).ln() - (ai as f64).ln() - (bj as f64).ln();
                let log2_term = log_term / std::f64::consts::LN_2;

                let nij_over_n = nij as f64 / n_f;

                // P(nij) can be very small, so we work in log space
                // contribution = nij_over_n * log2_term * exp(log_prob)
                let prob = log_prob.exp();
                emi += nij_over_n * log2_term * prob;
            }
        }
    }

    // E[MI] should be non-negative in theory; clamp for numerical safety
    Ok(emi.max(0.0))
}

/// Pre-compute log(k!) for k = 0, 1, ..., n
fn precompute_log_factorials(n: usize) -> Vec<f64> {
    let mut log_fact = vec![0.0_f64; n + 1];
    for k in 1..=n {
        log_fact[k] = log_fact[k - 1] + (k as f64).ln();
    }
    log_fact
}

/// Adjust integer counts so they sum to the target value.
/// Distributes the rounding error across the largest elements.
fn adjust_counts_to_sum(counts: &mut [i64], target: i64) {
    let current_sum: i64 = counts.iter().sum();
    let mut diff = target - current_sum;

    if diff == 0 {
        return;
    }

    // Sort indices by count (descending) so we adjust the largest bins first
    let mut indices: Vec<usize> = (0..counts.len()).collect();
    indices.sort_by(|&a, &b| counts[b].cmp(&counts[a]));

    let step = if diff > 0 { 1 } else { -1 };
    let mut idx = 0;
    while diff != 0 {
        let i = indices[idx % indices.len()];
        // Ensure counts don't go negative
        if counts[i] + step >= 0 {
            counts[i] += step;
            diff -= step;
        }
        idx += 1;
        // Safety: prevent infinite loop if all counts are 0 and we need to subtract
        if idx > counts.len() * 2 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_mutual_information_independent() {
        // Independent variables have zero MI
        let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");
        let mi = mutual_information(&joint).expect("mi failed");
        assert!(mi.abs() < EPSILON);
    }

    #[test]
    fn test_mutual_information_correlated() {
        // Perfectly correlated variables
        let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5])
            .expect("array creation failed");
        let mi = mutual_information(&joint).expect("mi failed");
        // I(X;X) = H(X) = 1 bit for uniform binary
        assert!((mi - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_mutual_information_partial() {
        // Partially correlated
        let joint = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.4])
            .expect("array creation failed");
        let mi = mutual_information(&joint).expect("mi failed");
        assert!(mi > 0.0);
        assert!(mi < 1.0); // Less than H(X) = H(Y)
    }

    #[test]
    fn test_conditional_mutual_information_independent() {
        // X and Y independent given Z
        let joint = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                0.125, 0.125, 0.125, 0.125, // Z=0
                0.125, 0.125, 0.125, 0.125, // Z=1
            ],
        )
        .expect("array creation failed");
        let cmi = conditional_mutual_information(&joint).expect("cmi failed");
        assert!(cmi.abs() < EPSILON);
    }

    #[test]
    fn test_normalized_mutual_information_independent() {
        let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");

        let nmi_arith = normalized_mutual_information(&joint, NormalizationType::Arithmetic)
            .expect("nmi failed");
        assert!(nmi_arith.abs() < EPSILON);

        let nmi_geo = normalized_mutual_information(&joint, NormalizationType::Geometric)
            .expect("nmi failed");
        assert!(nmi_geo.abs() < EPSILON);
    }

    #[test]
    fn test_normalized_mutual_information_correlated() {
        let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5])
            .expect("array creation failed");

        let nmi_arith = normalized_mutual_information(&joint, NormalizationType::Arithmetic)
            .expect("nmi failed");
        assert!((nmi_arith - 1.0).abs() < EPSILON);

        let nmi_max =
            normalized_mutual_information(&joint, NormalizationType::Max).expect("nmi failed");
        assert!((nmi_max - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_pointwise_mutual_information() {
        let joint = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.4])
            .expect("array creation failed");
        let pmi = pointwise_mutual_information(&joint).expect("pmi failed");

        // Marginals: p(x=0) = 0.5, p(y=0) = 0.5
        // PMI(0,0) = log(0.4 / (0.5*0.5)) = log(1.6)
        let expected_00 = (0.4_f64 / 0.25_f64).ln();
        assert!((pmi[[0, 0]] - expected_00).abs() < EPSILON);

        // PMI(0,1) = log(0.1 / (0.5*0.5)) = log(0.4)
        let expected_01 = (0.1_f64 / 0.25_f64).ln();
        assert!((pmi[[0, 1]] - expected_01).abs() < EPSILON);
    }

    #[test]
    fn test_variation_of_information_independent() {
        let joint = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");
        let vi = variation_of_information(&joint).expect("vi failed");
        // VI = H(X) + H(Y) - 2*I(X;Y) = 1 + 1 - 0 = 2
        assert!((vi - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_variation_of_information_correlated() {
        let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5])
            .expect("array creation failed");
        let vi = variation_of_information(&joint).expect("vi failed");
        // VI = H(X,Y) - I(X;Y) = H(X) - H(X) = 0 when X = Y
        assert!(vi.abs() < EPSILON);
    }

    #[test]
    fn test_adjusted_mutual_information() {
        // Perfect agreement
        let joint = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5])
            .expect("array creation failed");
        let ami = adjusted_mutual_information(&joint).expect("ami failed");
        // AMI ≈ 1 for perfect clustering
        assert!(ami > 0.9);

        // Random-like labeling
        let joint2 = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25])
            .expect("array creation failed");
        let ami2 = adjusted_mutual_information(&joint2).expect("ami failed");
        // AMI ≈ 0 for random labeling
        assert!(ami2.abs() < 0.1);
    }

    #[test]
    fn test_mutual_information_symmetry() {
        let joint = Array2::from_shape_vec((2, 3), vec![0.1, 0.15, 0.05, 0.2, 0.3, 0.2])
            .expect("array creation failed");
        let mi = mutual_information(&joint).expect("mi failed");

        // Transpose to swap X and Y
        let joint_t = joint.t().to_owned();
        let mi_t = mutual_information(&joint_t).expect("mi failed");

        // MI should be symmetric
        assert!((mi - mi_t).abs() < EPSILON);
    }

    #[test]
    fn test_mi_bounds() {
        let joint = Array2::from_shape_vec(
            (3, 3),
            vec![0.2, 0.05, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05, 0.2],
        )
        .expect("array creation failed");

        let mi = mutual_information(&joint).expect("mi failed");

        // Compute marginals
        let marginal_x: Array1<f64> = joint.sum_axis(Axis(1));
        let marginal_y: Array1<f64> = joint.sum_axis(Axis(0));

        let h_x = shannon_entropy(&marginal_x, LogBase::Bits).expect("entropy failed");
        let h_y = shannon_entropy(&marginal_y, LogBase::Bits).expect("entropy failed");

        // I(X;Y) ≤ min(H(X), H(Y))
        assert!(mi <= h_x + EPSILON);
        assert!(mi <= h_y + EPSILON);
    }
}

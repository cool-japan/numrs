//! Divergence and Distance Measures
//!
//! This module implements various divergence and distance measures between
//! probability distributions. These measures quantify the dissimilarity between
//! distributions and are fundamental in machine learning, statistics, and signal processing.
//!
//! # Mathematical Background
//!
//! ## Kullback-Leibler Divergence
//!
//! The KL divergence measures the information loss when Q is used to approximate P:
//!
//! ```text
//! D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
//! ```
//!
//! Properties:
//! - D_KL(P||Q) ≥ 0 (non-negative)
//! - D_KL(P||Q) = 0 iff P = Q
//! - D_KL(P||Q) ≠ D_KL(Q||P) (not symmetric)
//!
//! ## Jensen-Shannon Divergence
//!
//! A symmetrized and smoothed version of KL divergence:
//!
//! ```text
//! JSD(P||Q) = [D_KL(P||M) + D_KL(Q||M)]/2
//! where M = (P + Q)/2
//! ```
//!
//! Properties:
//! - 0 ≤ JSD(P||Q) ≤ log(2) (bounded)
//! - JSD(P||Q) = JSD(Q||P) (symmetric)
//! - √JSD is a metric
//!
//! ## Hellinger Distance
//!
//! A metric based on the Hellinger integral:
//!
//! ```text
//! H(P,Q) = √(1 - BC(P,Q)) = √(1 - Σ√(p(x)q(x)))
//! ```
//!
//! Properties:
//! - 0 ≤ H(P,Q) ≤ 1 (bounded)
//! - H(P,Q) = H(Q,P) (symmetric)
//! - Satisfies triangle inequality (metric)
//!
//! ## f-Divergences
//!
//! A general family of divergences:
//!
//! ```text
//! D_f(P||Q) = Σ q(x) f(p(x)/q(x))
//! ```
//!
//! Different choices of f yield different divergences (KL, Chi-squared, etc.)

use super::validate_distribution;
use crate::error::NumRs2Error;
use scirs2_core::ndarray::Array1;

/// Compute Kullback-Leibler divergence D_KL(P||Q)
///
/// # Arguments
///
/// * `p` - True probability distribution P
/// * `q` - Approximating probability distribution Q
///
/// # Returns
///
/// KL divergence D_KL(P||Q) in nats
///
/// # Mathematical Formula
///
/// ```text
/// D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
/// ```
///
/// Note: Returns infinity if any q(x) = 0 where p(x) > 0.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::kl_divergence;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let d = kl_divergence(&p, &q).expect("valid probability distributions");
/// assert!(d.abs() < 1e-10); // D_KL(P||P) = 0
///
/// let q2 = Array1::from_vec(vec![0.6, 0.4]);
/// let d2 = kl_divergence(&p, &q2).expect("valid probability distributions");
/// assert!(d2 > 0.0); // D_KL(P||Q) > 0 when P != Q
/// ```
pub fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
    let mut divergence = 0.0;
    for i in 0..p_norm.len() {
        let pi = p_norm[i];
        let qi = q_norm[i];

        if pi > 0.0 {
            if qi == 0.0 {
                // p(x) > 0 but q(x) = 0 => divergence is infinite
                return Ok(f64::INFINITY);
            }
            divergence += pi * (pi / qi).ln();
        }
    }

    Ok(divergence)
}

/// Compute Jensen-Shannon divergence JSD(P||Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Jensen-Shannon divergence in nats
///
/// # Mathematical Formula
///
/// ```text
/// JSD(P||Q) = [D_KL(P||M) + D_KL(Q||M)]/2
/// where M = (P + Q)/2
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::jensen_shannon_divergence;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let jsd = jensen_shannon_divergence(&p, &q).expect("valid probability distributions");
/// assert!(jsd.abs() < 1e-10); // JSD(P,P) = 0
///
/// let q2 = Array1::from_vec(vec![0.6, 0.4]);
/// let jsd2 = jensen_shannon_divergence(&p, &q2).expect("valid probability distributions");
/// assert!(jsd2 > 0.0 && jsd2 < 2_f64.ln()); // 0 < JSD < log(2)
/// ```
pub fn jensen_shannon_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // M = (P + Q)/2
    let m = (&p_norm + &q_norm) / 2.0;

    // JSD(P||Q) = [D_KL(P||M) + D_KL(Q||M)]/2
    let d_pm = kl_divergence(&p_norm, &m)?;
    let d_qm = kl_divergence(&q_norm, &m)?;

    Ok((d_pm + d_qm) / 2.0)
}

/// Compute Bhattacharyya coefficient BC(P,Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Bhattacharyya coefficient (0 ≤ BC ≤ 1)
///
/// # Mathematical Formula
///
/// ```text
/// BC(P,Q) = Σ √(p(x)q(x))
/// ```
///
/// BC = 1 when P = Q, BC = 0 when P and Q have disjoint support.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::bhattacharyya_coefficient;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let bc = bhattacharyya_coefficient(&p, &q).expect("valid probability distributions");
/// assert!((bc - 1.0).abs() < 1e-10); // BC(P,P) = 1
///
/// let p2 = Array1::from_vec(vec![1.0, 0.0]);
/// let q2 = Array1::from_vec(vec![0.0, 1.0]);
/// let bc2 = bhattacharyya_coefficient(&p2, &q2).expect("valid probability distributions");
/// assert!(bc2.abs() < 1e-10); // BC = 0 for disjoint support
/// ```
pub fn bhattacharyya_coefficient(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // BC(P,Q) = Σ √(p(x)q(x))
    let bc: f64 = p_norm
        .iter()
        .zip(q_norm.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum();

    Ok(bc)
}

/// Compute Bhattacharyya distance D_B(P,Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Bhattacharyya distance (0 ≤ D_B < ∞)
///
/// # Mathematical Formula
///
/// ```text
/// D_B(P,Q) = -log(BC(P,Q)) = -log(Σ √(p(x)q(x)))
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::bhattacharyya_distance;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let d = bhattacharyya_distance(&p, &q).expect("valid probability distributions");
/// assert!(d.abs() < 1e-10); // D_B(P,P) = 0
/// ```
pub fn bhattacharyya_distance(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    let bc = bhattacharyya_coefficient(p, q)?;

    if bc <= 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(-bc.ln())
    }
}

/// Compute Hellinger distance H(P,Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Hellinger distance (0 ≤ H ≤ 1)
///
/// # Mathematical Formula
///
/// ```text
/// H(P,Q) = √(1 - BC(P,Q)) = √(1 - Σ√(p(x)q(x)))
/// ```
///
/// This is a true metric satisfying the triangle inequality.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::hellinger_distance;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let h = hellinger_distance(&p, &q).expect("valid probability distributions");
/// assert!(h.abs() < 1e-10); // H(P,P) = 0
///
/// let p2 = Array1::from_vec(vec![1.0, 0.0]);
/// let q2 = Array1::from_vec(vec![0.0, 1.0]);
/// let h2 = hellinger_distance(&p2, &q2).expect("valid probability distributions");
/// assert!((h2 - 1.0).abs() < 1e-10); // H = 1 for disjoint support
/// ```
pub fn hellinger_distance(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    let bc = bhattacharyya_coefficient(p, q)?;

    // H(P,Q) = √(1 - BC(P,Q))
    let h_squared = (1.0 - bc).max(0.0); // Clamp to avoid negative due to numerical errors
    Ok(h_squared.sqrt())
}

/// Compute total variation distance TV(P,Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Total variation distance (0 ≤ TV ≤ 1)
///
/// # Mathematical Formula
///
/// ```text
/// TV(P,Q) = (1/2) Σ |p(x) - q(x)|
/// ```
///
/// This is a true metric and represents the maximum difference in probability
/// of any event under the two distributions.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::total_variation_distance;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let tv = total_variation_distance(&p, &q).expect("valid probability distributions");
/// assert!(tv.abs() < 1e-10); // TV(P,P) = 0
///
/// let p2 = Array1::from_vec(vec![1.0, 0.0]);
/// let q2 = Array1::from_vec(vec![0.0, 1.0]);
/// let tv2 = total_variation_distance(&p2, &q2).expect("valid probability distributions");
/// assert!((tv2 - 1.0).abs() < 1e-10); // TV = 1 for disjoint support
/// ```
pub fn total_variation_distance(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // TV(P,Q) = (1/2) Σ |p(x) - q(x)|
    let tv: f64 = p_norm
        .iter()
        .zip(q_norm.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
        / 2.0;

    Ok(tv)
}

/// Compute Chi-squared divergence χ²(P||Q)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Chi-squared divergence (0 ≤ χ² < ∞)
///
/// # Mathematical Formula
///
/// ```text
/// χ²(P||Q) = Σ (p(x) - q(x))² / q(x)
/// ```
///
/// This is an f-divergence with f(t) = (t-1)².
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::chi_squared_divergence;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.5, 0.5]);
/// let chi2 = chi_squared_divergence(&p, &q).expect("valid probability distributions");
/// assert!(chi2.abs() < 1e-10); // χ²(P,P) = 0
/// ```
pub fn chi_squared_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, NumRs2Error> {
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // χ²(P||Q) = Σ (p(x) - q(x))² / q(x)
    let mut chi2 = 0.0;
    for i in 0..p_norm.len() {
        let pi = p_norm[i];
        let qi = q_norm[i];

        if qi == 0.0 && pi > 0.0 {
            return Ok(f64::INFINITY);
        }

        if qi > 0.0 {
            let diff = pi - qi;
            chi2 += (diff * diff) / qi;
        }
    }

    Ok(chi2)
}

/// Compute f-divergence with custom convex function f
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
/// * `f` - Convex function f: ℝ₊ → ℝ with f(1) = 0
///
/// # Returns
///
/// f-divergence D_f(P||Q)
///
/// # Mathematical Formula
///
/// ```text
/// D_f(P||Q) = Σ q(x) f(p(x)/q(x))
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::information_theory::divergence::f_divergence;
/// use scirs2_core::ndarray::Array1;
///
/// let p = Array1::from_vec(vec![0.5, 0.5]);
/// let q = Array1::from_vec(vec![0.6, 0.4]);
///
/// // KL divergence: f(t) = t*log(t)
/// let d_kl = f_divergence(&p, &q, |t| if t > 0.0 { t * t.ln() } else { 0.0 }).expect("valid probability distributions");
/// assert!(d_kl > 0.0);
///
/// // Chi-squared: f(t) = (t-1)²
/// let d_chi2 = f_divergence(&p, &q, |t| (t - 1.0).powi(2)).expect("valid probability distributions");
/// assert!(d_chi2 > 0.0);
/// ```
pub fn f_divergence<F>(p: &Array1<f64>, q: &Array1<f64>, f: F) -> Result<f64, NumRs2Error>
where
    F: Fn(f64) -> f64,
{
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Probability arrays must have same length: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    validate_distribution(&p.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    validate_distribution(&q.view()).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    let p_norm =
        super::normalize_distribution(p).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;
    let q_norm =
        super::normalize_distribution(q).map_err(|e| NumRs2Error::ValueError(e.to_string()))?;

    // D_f(P||Q) = Σ q(x) f(p(x)/q(x))
    let mut divergence = 0.0;
    for i in 0..p_norm.len() {
        let pi = p_norm[i];
        let qi = q_norm[i];

        if qi > 0.0 {
            let ratio = pi / qi;
            divergence += qi * f(ratio);
        } else if pi > 0.0 {
            // q(x) = 0 but p(x) > 0: handle based on limit behavior of f
            return Ok(f64::INFINITY);
        }
    }

    Ok(divergence)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_kl_divergence_identical() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let d = kl_divergence(&p, &q).expect("kl divergence failed");
        assert!(d.abs() < EPSILON); // D_KL(P||P) = 0
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.6, 0.4]);
        let d = kl_divergence(&p, &q).expect("kl divergence failed");
        assert!(d > 0.0); // D_KL(P||Q) > 0 when P != Q

        // KL divergence is not symmetric
        let d_reverse = kl_divergence(&q, &p).expect("kl divergence failed");
        assert!((d - d_reverse).abs() > 1e-6);
    }

    #[test]
    fn test_kl_divergence_disjoint() {
        let p = Array1::from_vec(vec![1.0, 0.0]);
        let q = Array1::from_vec(vec![0.0, 1.0]);
        let d = kl_divergence(&p, &q).expect("kl divergence failed");
        assert!(d.is_infinite()); // p(x) > 0 but q(x) = 0
    }

    #[test]
    fn test_jensen_shannon_divergence() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let jsd = jensen_shannon_divergence(&p, &q).expect("jsd failed");
        assert!(jsd.abs() < EPSILON); // JSD(P,P) = 0

        let q2 = Array1::from_vec(vec![0.6, 0.4]);
        let jsd2 = jensen_shannon_divergence(&p, &q2).expect("jsd failed");
        assert!(jsd2 > 0.0);
        assert!(jsd2 < 2_f64.ln()); // JSD is bounded by log(2)

        // JSD is symmetric
        let jsd_reverse = jensen_shannon_divergence(&q2, &p).expect("jsd failed");
        assert!((jsd2 - jsd_reverse).abs() < EPSILON);
    }

    #[test]
    fn test_bhattacharyya_coefficient() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let bc = bhattacharyya_coefficient(&p, &q).expect("bc failed");
        assert!((bc - 1.0).abs() < EPSILON); // BC(P,P) = 1

        let p2 = Array1::from_vec(vec![1.0, 0.0]);
        let q2 = Array1::from_vec(vec![0.0, 1.0]);
        let bc2 = bhattacharyya_coefficient(&p2, &q2).expect("bc failed");
        assert!(bc2.abs() < EPSILON); // BC = 0 for disjoint support

        // BC is symmetric
        let bc_reverse = bhattacharyya_coefficient(&q2, &p2).expect("bc failed");
        assert!((bc2 - bc_reverse).abs() < EPSILON);
    }

    #[test]
    fn test_bhattacharyya_distance() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let d = bhattacharyya_distance(&p, &q).expect("bhattacharyya distance failed");
        assert!(d.abs() < EPSILON); // D_B(P,P) = 0

        let p2 = Array1::from_vec(vec![1.0, 0.0]);
        let q2 = Array1::from_vec(vec![0.0, 1.0]);
        let d2 = bhattacharyya_distance(&p2, &q2).expect("bhattacharyya distance failed");
        assert!(d2.is_infinite()); // D_B = ∞ for disjoint support
    }

    #[test]
    fn test_hellinger_distance() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let h = hellinger_distance(&p, &q).expect("hellinger distance failed");
        assert!(h.abs() < EPSILON); // H(P,P) = 0

        let p2 = Array1::from_vec(vec![1.0, 0.0]);
        let q2 = Array1::from_vec(vec![0.0, 1.0]);
        let h2 = hellinger_distance(&p2, &q2).expect("hellinger distance failed");
        assert!((h2 - 1.0).abs() < EPSILON); // H = 1 for disjoint support

        // Hellinger is symmetric
        let h_reverse = hellinger_distance(&q2, &p2).expect("hellinger distance failed");
        assert!((h2 - h_reverse).abs() < EPSILON);

        // Hellinger is bounded by 1
        let p3 = Array1::from_vec(vec![0.7, 0.3]);
        let q3 = Array1::from_vec(vec![0.2, 0.8]);
        let h3 = hellinger_distance(&p3, &q3).expect("hellinger distance failed");
        assert!((0.0..=1.0).contains(&h3));
    }

    #[test]
    fn test_total_variation_distance() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let tv = total_variation_distance(&p, &q).expect("tv distance failed");
        assert!(tv.abs() < EPSILON); // TV(P,P) = 0

        let p2 = Array1::from_vec(vec![1.0, 0.0]);
        let q2 = Array1::from_vec(vec![0.0, 1.0]);
        let tv2 = total_variation_distance(&p2, &q2).expect("tv distance failed");
        assert!((tv2 - 1.0).abs() < EPSILON); // TV = 1 for disjoint support

        // TV is symmetric
        let tv_reverse = total_variation_distance(&q2, &p2).expect("tv distance failed");
        assert!((tv2 - tv_reverse).abs() < EPSILON);

        // TV is bounded by 1
        let p3 = Array1::from_vec(vec![0.7, 0.3]);
        let q3 = Array1::from_vec(vec![0.2, 0.8]);
        let tv3 = total_variation_distance(&p3, &q3).expect("tv distance failed");
        assert!((0.0..=1.0).contains(&tv3));
    }

    #[test]
    fn test_chi_squared_divergence() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.5, 0.5]);
        let chi2 = chi_squared_divergence(&p, &q).expect("chi2 divergence failed");
        assert!(chi2.abs() < EPSILON); // χ²(P,P) = 0

        let p2 = Array1::from_vec(vec![0.6, 0.4]);
        let q2 = Array1::from_vec(vec![0.5, 0.5]);
        let chi2_2 = chi_squared_divergence(&p2, &q2).expect("chi2 divergence failed");
        assert!(chi2_2 > 0.0); // χ²(P,Q) > 0 when P != Q

        // Manual calculation: χ²(P||Q) = Σ (p-q)²/q
        // = (0.6-0.5)²/0.5 + (0.4-0.5)²/0.5 = 0.01/0.5 + 0.01/0.5 = 0.04
        assert!((chi2_2 - 0.04).abs() < EPSILON);
    }

    #[test]
    fn test_f_divergence_kl() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.6, 0.4]);

        // KL divergence via f-divergence: f(t) = t*log(t)
        let d_kl = f_divergence(&p, &q, |t| if t > 0.0 { t * t.ln() } else { 0.0 })
            .expect("f-divergence failed");

        let d_kl_direct = kl_divergence(&p, &q).expect("kl divergence failed");

        assert!((d_kl - d_kl_direct).abs() < EPSILON);
    }

    #[test]
    fn test_f_divergence_chi_squared() {
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.6, 0.4]);

        // Chi-squared via f-divergence: f(t) = (t-1)²
        let d_chi2 = f_divergence(&p, &q, |t| (t - 1.0).powi(2)).expect("f-divergence failed");

        let d_chi2_direct = chi_squared_divergence(&p, &q).expect("chi2 divergence failed");

        assert!((d_chi2 - d_chi2_direct).abs() < EPSILON);
    }

    #[test]
    fn test_divergence_errors() {
        // Dimension mismatch
        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.3, 0.3, 0.4]);
        assert!(kl_divergence(&p, &q).is_err());
        assert!(jensen_shannon_divergence(&p, &q).is_err());
        assert!(hellinger_distance(&p, &q).is_err());

        // Negative probability
        let negative = Array1::from_vec(vec![0.5, -0.1]);
        let valid = Array1::from_vec(vec![0.5, 0.5]);
        assert!(kl_divergence(&negative, &valid).is_err());
    }
}

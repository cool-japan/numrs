//! # Multiresolution Analysis (MRA)
//!
//! This module implements multiresolution analysis, which decomposes signals
//! into multiple resolution levels using wavelets. It's particularly useful
//! for signal denoising and feature extraction.
//!
//! ## Mathematical Foundation
//!
//! MRA provides a hierarchical decomposition of L²(ℝ) into nested subspaces:
//!
//! ```text
//! ... ⊂ V₂ ⊂ V₁ ⊂ V₀ ⊂ V₋₁ ⊂ V₋₂ ⊂ ...
//! ```
//!
//! where each space V_j can be decomposed as:
//!
//! ```text
//! V_j = V_{j+1} ⊕ W_{j+1}
//! ```
//!
//! - V_j: approximation space at level j
//! - W_j: detail space at level j (orthogonal complement)
//!
//! ## Denoising
//!
//! Wavelet-based denoising uses thresholding of detail coefficients:
//!
//! 1. Decompose signal using DWT
//! 2. Apply threshold to detail coefficients
//! 3. Reconstruct signal from thresholded coefficients

use super::dwt::waverec_mode;
use super::{wavedec, ExtensionMode, Wavelet, WaveletError, WaveletResult};

/// Threshold type for wavelet denoising
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    /// Hard thresholding: set to zero if |x| < threshold
    Hard,
    /// Soft thresholding: shrink toward zero by threshold amount
    Soft,
    /// Garrote thresholding: non-negative garrote
    Garrote,
}

impl ThresholdType {
    /// Apply threshold to a single coefficient
    pub fn apply(&self, x: f64, threshold: f64) -> f64 {
        match self {
            ThresholdType::Hard => {
                if x.abs() < threshold {
                    0.0
                } else {
                    x
                }
            }
            ThresholdType::Soft => {
                if x.abs() < threshold {
                    0.0
                } else {
                    x.signum() * (x.abs() - threshold)
                }
            }
            ThresholdType::Garrote => {
                if x.abs() < threshold {
                    0.0
                } else {
                    x - (threshold * threshold / x)
                }
            }
        }
    }

    /// Apply threshold to a vector of coefficients
    pub fn apply_vec(&self, coeffs: &[f64], threshold: f64) -> Vec<f64> {
        coeffs.iter().map(|&x| self.apply(x, threshold)).collect()
    }
}

/// Multiresolution analysis structure
///
/// Contains approximation and detail coefficients at multiple levels,
/// providing access to signal representation at different scales.
pub struct MultiresolutionAnalysis {
    /// Final approximation coefficients
    pub approximation: Vec<f64>,
    /// Detail coefficients at each level (finest to coarsest)
    pub details: Vec<Vec<f64>>,
    /// Original signal length
    pub original_length: usize,
    /// Decomposition levels
    pub levels: usize,
    /// Extension mode used for decomposition
    mode: ExtensionMode,
}

impl MultiresolutionAnalysis {
    /// Create MRA from signal
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `wavelet` - Wavelet to use
    /// * `levels` - Number of decomposition levels
    /// * `mode` - Boundary extension mode
    pub fn from_signal(
        signal: &[f64],
        wavelet: &dyn Wavelet,
        levels: usize,
        mode: ExtensionMode,
    ) -> WaveletResult<Self> {
        let original_length = signal.len();
        let (approximation, details) = wavedec(signal, wavelet, levels, mode)?;

        Ok(Self {
            approximation,
            details,
            original_length,
            levels,
            mode,
        })
    }

    /// Reconstruct signal from MRA
    ///
    /// # Arguments
    ///
    /// * `wavelet` - Wavelet used for decomposition
    pub fn reconstruct(&self, wavelet: &dyn Wavelet) -> WaveletResult<Vec<f64>> {
        waverec_mode(
            &self.approximation,
            &self.details,
            wavelet,
            self.original_length,
            self.mode,
        )
    }

    /// Reconstruct signal using only selected levels
    ///
    /// # Arguments
    ///
    /// * `wavelet` - Wavelet used for decomposition
    /// * `selected_levels` - Levels to include (1-indexed, 1 = finest detail)
    pub fn reconstruct_selected(
        &self,
        wavelet: &dyn Wavelet,
        selected_levels: &[usize],
    ) -> WaveletResult<Vec<f64>> {
        // Create modified details with zeros for non-selected levels
        let mut modified_details = Vec::with_capacity(self.details.len());

        for (i, detail) in self.details.iter().enumerate() {
            let level = i + 1;
            if selected_levels.contains(&level) {
                modified_details.push(detail.clone());
            } else {
                modified_details.push(vec![0.0; detail.len()]);
            }
        }

        waverec_mode(
            &self.approximation,
            &modified_details,
            wavelet,
            self.original_length,
            self.mode,
        )
    }

    /// Reconstruct signal from approximation only (low-pass filtered)
    ///
    /// # Arguments
    ///
    /// * `wavelet` - Wavelet used for decomposition
    pub fn reconstruct_approximation(&self, wavelet: &dyn Wavelet) -> WaveletResult<Vec<f64>> {
        let zero_details: Vec<Vec<f64>> = self.details.iter().map(|d| vec![0.0; d.len()]).collect();

        waverec_mode(
            &self.approximation,
            &zero_details,
            wavelet,
            self.original_length,
            self.mode,
        )
    }

    /// Reconstruct signal from details only (high-pass filtered)
    ///
    /// # Arguments
    ///
    /// * `wavelet` - Wavelet used for decomposition
    pub fn reconstruct_details(&self, wavelet: &dyn Wavelet) -> WaveletResult<Vec<f64>> {
        let zero_approx = vec![0.0; self.approximation.len()];
        waverec_mode(
            &zero_approx,
            &self.details,
            wavelet,
            self.original_length,
            self.mode,
        )
    }

    /// Apply threshold to detail coefficients
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold value
    /// * `threshold_type` - Type of thresholding
    pub fn threshold_details(&mut self, threshold: f64, threshold_type: ThresholdType) {
        for detail in &mut self.details {
            *detail = threshold_type.apply_vec(detail, threshold);
        }
    }

    /// Apply different thresholds to each level
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Threshold for each level (must match number of levels)
    /// * `threshold_type` - Type of thresholding
    pub fn threshold_details_multilevel(
        &mut self,
        thresholds: &[f64],
        threshold_type: ThresholdType,
    ) -> WaveletResult<()> {
        if thresholds.len() != self.details.len() {
            return Err(WaveletError::InvalidLength(format!(
                "Expected {} thresholds, got {}",
                self.details.len(),
                thresholds.len()
            )));
        }

        for (detail, &threshold) in self.details.iter_mut().zip(thresholds.iter()) {
            *detail = threshold_type.apply_vec(detail, threshold);
        }

        Ok(())
    }

    /// Compute energy at each level
    pub fn level_energies(&self) -> Vec<f64> {
        let mut energies = Vec::with_capacity(self.levels + 1);

        // Approximation energy
        let approx_energy: f64 = self.approximation.iter().map(|x| x * x).sum();
        energies.push(approx_energy);

        // Detail energies
        for detail in &self.details {
            let energy: f64 = detail.iter().map(|x| x * x).sum();
            energies.push(energy);
        }

        energies
    }

    /// Compute percentage of energy at each level
    pub fn energy_distribution(&self) -> Vec<f64> {
        let energies = self.level_energies();
        let total_energy: f64 = energies.iter().sum();

        if total_energy == 0.0 {
            return vec![0.0; energies.len()];
        }

        energies.iter().map(|e| 100.0 * e / total_energy).collect()
    }
}

/// Denoise a signal using wavelet thresholding
///
/// # Arguments
///
/// * `signal` - Noisy input signal
/// * `wavelet` - Wavelet to use
/// * `levels` - Number of decomposition levels
/// * `threshold` - Threshold value (or None for universal threshold)
/// * `threshold_type` - Type of thresholding
/// * `mode` - Boundary extension mode
///
/// # Returns
///
/// Denoised signal
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{denoise_signal, WaveletType, ThresholdType, ExtensionMode};
///
/// let noisy_signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Daubechies(4).create()?;
/// let denoised = denoise_signal(
///     &noisy_signal,
///     &wavelet,
///     3,
///     None,  // Use universal threshold
///     ThresholdType::Soft,
///     ExtensionMode::Symmetric
/// )?;
/// ```
pub fn denoise_signal(
    signal: &[f64],
    wavelet: &dyn Wavelet,
    levels: usize,
    threshold: Option<f64>,
    threshold_type: ThresholdType,
    mode: ExtensionMode,
) -> WaveletResult<Vec<f64>> {
    let mut mra = MultiresolutionAnalysis::from_signal(signal, wavelet, levels, mode)?;

    let threshold_value = threshold.unwrap_or_else(|| universal_threshold(signal, levels));

    mra.threshold_details(threshold_value, threshold_type);
    mra.reconstruct(wavelet)
}

/// Compute universal threshold (Donoho & Johnstone)
///
/// The universal threshold is defined as:
///
/// ```text
/// λ = σ√(2 log n)
/// ```
///
/// where σ is the noise standard deviation (estimated from finest detail level)
/// and n is the signal length.
pub fn universal_threshold(signal: &[f64], _levels: usize) -> f64 {
    let n = signal.len() as f64;
    let sigma = estimate_noise_sigma(signal);
    sigma * (2.0 * n.ln()).sqrt()
}

/// Estimate noise standard deviation using Median Absolute Deviation (MAD)
///
/// The noise sigma is estimated from the signal as:
///
/// ```text
/// σ = MAD / 0.6745
/// ```
///
/// where MAD is the median absolute deviation from the median.
pub fn estimate_noise_sigma(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let mut sorted = signal.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut deviations: Vec<f64> = signal.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if deviations.len().is_multiple_of(2) {
        let mid = deviations.len() / 2;
        (deviations[mid - 1] + deviations[mid]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    mad / 0.6745
}

/// Apply VisuShrink threshold (universal threshold)
pub fn visushrink_threshold(signal: &[f64], levels: usize) -> f64 {
    universal_threshold(signal, levels)
}

/// Apply SureShrink threshold (Stein's Unbiased Risk Estimate)
///
/// Minimizes the exact SURE risk for soft thresholding,
///
/// ```text
/// SURE(t) = n - 2 * #{|x_i| <= t} + sum_i min(x_i^2, t^2)
/// ```
///
/// over the candidate thresholds `t = |x_(i)|` (the sorted absolute
/// coefficient magnitudes). Donoho & Johnstone (1995) show the minimizer
/// of SURE(t) over all `t >= 0` is attained at one of these `n` order
/// statistics, so checking exactly those candidates is exact (not a
/// heuristic approximation).
pub fn sureshrink_threshold(detail_coeffs: &[f64]) -> f64 {
    if detail_coeffs.is_empty() {
        return 0.0;
    }

    // Sort squared magnitudes ascending: sorted_sq[i] = x_(i)^2, the i-th
    // order statistic. Candidate threshold at index i is t = sqrt(sorted_sq[i]).
    let mut sorted_sq: Vec<f64> = detail_coeffs.iter().map(|&x| x * x).collect();
    sorted_sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_sq.len();
    let n_f64 = n as f64;

    // O(n) sweep: `prefix_sum` accumulates sum_{j<=i} x_(j)^2 so that
    // sum_i min(x_i^2, t^2) = prefix_sum + (n - below_or_eq) * t^2
    // is computed in O(1) per candidate (overall O(n log n) from the sort).
    let mut prefix_sum = 0.0;
    let mut min_risk = f64::INFINITY;
    let mut best_threshold = 0.0;

    for (i, &t2) in sorted_sq.iter().enumerate() {
        prefix_sum += t2;
        let below_or_eq = (i + 1) as f64; // #{|x_j| <= t}, t = sqrt(t2)
        let above = n_f64 - below_or_eq; // #{|x_j| > t}, each contributes t^2

        let sum_min = prefix_sum + above * t2;
        let risk = n_f64 - 2.0 * below_or_eq + sum_min;

        if risk < min_risk {
            min_risk = risk;
            best_threshold = t2.sqrt();
        }
    }

    best_threshold
}

/// Apply BayesShrink threshold
///
/// Minimizes Bayesian risk with a Generalized Gaussian prior.
pub fn bayesshrink_threshold(detail_coeffs: &[f64]) -> f64 {
    if detail_coeffs.is_empty() {
        return 0.0;
    }

    let sigma_noise = estimate_noise_sigma(detail_coeffs);
    let sigma_signal_squared: f64 = detail_coeffs.iter().map(|x| x * x).sum::<f64>()
        / detail_coeffs.len() as f64
        - sigma_noise * sigma_noise;

    if sigma_signal_squared <= 0.0 {
        return sigma_noise * (2.0 * (detail_coeffs.len() as f64).ln()).sqrt();
    }

    (sigma_noise * sigma_noise) / sigma_signal_squared.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::wavelets::WaveletType;

    #[test]
    fn test_threshold_type_hard() {
        let threshold = 1.5;
        assert_eq!(ThresholdType::Hard.apply(2.0, threshold), 2.0);
        assert_eq!(ThresholdType::Hard.apply(1.0, threshold), 0.0);
        assert_eq!(ThresholdType::Hard.apply(-2.0, threshold), -2.0);
    }

    #[test]
    fn test_threshold_type_soft() {
        let threshold = 1.0;
        assert_eq!(ThresholdType::Soft.apply(2.0, threshold), 1.0);
        assert_eq!(ThresholdType::Soft.apply(0.5, threshold), 0.0);
        assert_eq!(ThresholdType::Soft.apply(-2.0, threshold), -1.0);
    }

    #[test]
    fn test_threshold_type_garrote() {
        let threshold = 1.0;
        let result = ThresholdType::Garrote.apply(2.0, threshold);
        assert!(result > 0.0 && result < 2.0);

        let result = ThresholdType::Garrote.apply(0.5, threshold);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_threshold_apply_vec() {
        let coeffs = vec![0.5, 1.5, -2.5, 0.8];
        let threshold = 1.0;

        let result = ThresholdType::Hard.apply_vec(&coeffs, threshold);
        assert_eq!(result, vec![0.0, 1.5, -2.5, 0.0]);

        let result = ThresholdType::Soft.apply_vec(&coeffs, threshold);
        assert_eq!(result, vec![0.0, 0.5, -1.5, 0.0]);
    }

    #[test]
    fn test_mra_from_signal() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        assert_eq!(mra.levels, 3);
        assert_eq!(mra.original_length, 8);
        assert_eq!(mra.details.len(), 3);
    }

    #[test]
    fn test_mra_reconstruct() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let reconstructed = mra
            .reconstruct(wavelet.as_ref())
            .expect("Reconstruction failed");

        assert_eq!(reconstructed.len(), signal.len());
        for (i, (&orig, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_mra_reconstruct_selected() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        // Reconstruct with only level 1 details
        let reconstructed = mra
            .reconstruct_selected(wavelet.as_ref(), &[1])
            .expect("Reconstruction failed");

        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_mra_reconstruct_approximation() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            2,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let low_pass = mra
            .reconstruct_approximation(wavelet.as_ref())
            .expect("Reconstruction failed");

        assert_eq!(low_pass.len(), signal.len());
    }

    #[test]
    fn test_mra_reconstruct_details() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            2,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let high_pass = mra
            .reconstruct_details(wavelet.as_ref())
            .expect("Reconstruction failed");

        assert_eq!(high_pass.len(), signal.len());
    }

    #[test]
    fn test_mra_threshold_details() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mut mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            2,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        mra.threshold_details(1.0, ThresholdType::Hard);

        let reconstructed = mra
            .reconstruct(wavelet.as_ref())
            .expect("Reconstruction failed");

        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_mra_level_energies() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let energies = mra.level_energies();
        assert_eq!(energies.len(), 4); // 1 approx + 3 details

        // All energies should be non-negative
        for &energy in &energies {
            assert!(energy >= 0.0);
        }
    }

    #[test]
    fn test_mra_energy_distribution() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            2,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let distribution = mra.energy_distribution();
        assert_eq!(distribution.len(), 3);

        // Sum of percentages should be approximately 100
        let total: f64 = distribution.iter().sum();
        assert!((total - 100.0).abs() < 1e-8);
    }

    #[test]
    fn test_denoise_signal() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Daubechies(2)
            .create()
            .expect("Failed to create wavelet");

        let denoised = denoise_signal(
            &signal,
            wavelet.as_ref(),
            2,
            Some(0.5),
            ThresholdType::Soft,
            ExtensionMode::Periodic,
        )
        .expect("Denoising failed");

        assert_eq!(denoised.len(), signal.len());
    }

    #[test]
    fn test_universal_threshold() {
        // Signal with variability so MAD > 0
        let signal: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        let threshold = universal_threshold(&signal, 3);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_estimate_noise_sigma() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = estimate_noise_sigma(&signal);
        assert!(sigma >= 0.0);
    }

    #[test]
    fn test_estimate_noise_sigma_empty() {
        let signal: Vec<f64> = vec![];
        let sigma = estimate_noise_sigma(&signal);
        assert_eq!(sigma, 0.0);
    }

    #[test]
    fn test_visushrink_threshold() {
        // Signal with variability so MAD > 0
        let signal: Vec<f64> = (0..64).map(|i| (i as f64) * 0.1).collect();
        let threshold = visushrink_threshold(&signal, 3);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_sureshrink_threshold() {
        let coeffs = vec![0.1, 0.5, 1.0, 2.0, 0.3, 1.5];
        let threshold = sureshrink_threshold(&coeffs);
        assert!(threshold >= 0.0);
    }

    #[test]
    fn test_bayesshrink_threshold() {
        let coeffs = vec![0.1, 0.5, 1.0, 2.0, 0.3, 1.5];
        let threshold = bayesshrink_threshold(&coeffs);
        assert!(threshold >= 0.0);
    }

    #[test]
    fn test_threshold_multilevel() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mut mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let thresholds = vec![0.5, 1.0, 1.5];
        mra.threshold_details_multilevel(&thresholds, ThresholdType::Soft)
            .expect("Multi-level thresholding failed");
    }

    #[test]
    fn test_threshold_multilevel_invalid_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mut mra = MultiresolutionAnalysis::from_signal(
            &signal,
            wavelet.as_ref(),
            3,
            ExtensionMode::Periodic,
        )
        .expect("MRA creation failed");

        let thresholds = vec![0.5, 1.0]; // Wrong length
        let result = mra.threshold_details_multilevel(&thresholds, ThresholdType::Soft);
        assert!(result.is_err());
    }
}

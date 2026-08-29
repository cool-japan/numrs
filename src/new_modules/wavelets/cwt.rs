//! # Continuous Wavelet Transform (CWT)
//!
//! This module implements the Continuous Wavelet Transform for time-frequency
//! analysis of signals. Unlike the DWT, the CWT provides a continuous representation
//! of how signal frequencies change over time.
//!
//! ## Mathematical Definition
//!
//! The CWT of a signal x(t) with respect to a wavelet ψ is:
//!
//! ```text
//! CWT(a,b) = (1/√a) ∫ x(t) ψ*((t-b)/a) dt
//! ```
//!
//! where:
//! - `a` is the scale parameter (inversely related to frequency)
//! - `b` is the translation parameter (time shift)
//! - `ψ*` is the complex conjugate of the wavelet
//!
//! ## Wavelets
//!
//! This implementation provides three common continuous wavelets:
//!
//! - **Morlet**: Complex sinusoid modulated by Gaussian envelope
//! - **Mexican Hat (Ricker)**: Second derivative of Gaussian
//! - **Paul**: Complex wavelet with good time localization

use super::{WaveletError, WaveletResult};
use std::f64::consts::PI;

/// Trait for continuous wavelet functions
pub trait ContinuousWavelet: Send + Sync {
    /// Compute the wavelet function value at time t for scale a
    ///
    /// # Arguments
    ///
    /// * `t` - Time parameter
    /// * `scale` - Scale parameter (a > 0)
    ///
    /// # Returns
    ///
    /// Complex value of the wavelet function
    fn psi(&self, t: f64, scale: f64) -> (f64, f64);

    /// Get the wavelet name
    fn name(&self) -> &str;

    /// Get the center frequency of the wavelet
    fn center_frequency(&self) -> f64;

    /// Check if the wavelet is complex-valued
    fn is_complex(&self) -> bool;
}

/// Morlet wavelet
///
/// The Morlet wavelet is a complex sinusoid modulated by a Gaussian:
///
/// ```text
/// ψ(t) = π^(-1/4) exp(iω₀t) exp(-t²/2)
/// ```
///
/// where ω₀ is the central frequency (typically 5 or 6).
pub struct MorletWavelet {
    omega0: f64,
}

impl MorletWavelet {
    /// Create a new Morlet wavelet with specified central frequency
    ///
    /// # Arguments
    ///
    /// * `omega0` - Central frequency (default: 6.0)
    pub fn new(omega0: f64) -> WaveletResult<Self> {
        if omega0 <= 0.0 {
            return Err(WaveletError::InvalidWavelet(
                "Omega0 must be positive".to_string(),
            ));
        }
        Ok(Self { omega0 })
    }

    /// Create a default Morlet wavelet (omega0 = 6.0)
    pub fn default_wavelet() -> Self {
        Self { omega0: 6.0 }
    }
}

impl ContinuousWavelet for MorletWavelet {
    fn psi(&self, t: f64, scale: f64) -> (f64, f64) {
        let t_scaled = t / scale;
        let envelope = PI.powf(-0.25) * (-0.5 * t_scaled * t_scaled).exp() / scale.sqrt();

        let phase = self.omega0 * t_scaled;
        let real = envelope * phase.cos();
        let imag = envelope * phase.sin();

        (real, imag)
    }

    fn name(&self) -> &str {
        "morlet"
    }

    fn center_frequency(&self) -> f64 {
        self.omega0 / (2.0 * PI)
    }

    fn is_complex(&self) -> bool {
        true
    }
}

/// Mexican Hat (Ricker) wavelet
///
/// The Mexican Hat wavelet is the second derivative of a Gaussian:
///
/// ```text
/// ψ(t) = (2/√3) π^(-1/4) (1 - t²) exp(-t²/2)
/// ```
///
/// This is a real-valued wavelet with good time-frequency localization.
pub struct MexicanHatWavelet;

impl ContinuousWavelet for MexicanHatWavelet {
    fn psi(&self, t: f64, scale: f64) -> (f64, f64) {
        let t_scaled = t / scale;
        let t2 = t_scaled * t_scaled;
        let coefficient = 2.0 / (3.0_f64.sqrt()) * PI.powf(-0.25);
        let value = coefficient * (1.0 - t2) * (-0.5 * t2).exp() / scale.sqrt();

        (value, 0.0)
    }

    fn name(&self) -> &str {
        "mexican_hat"
    }

    fn center_frequency(&self) -> f64 {
        0.25
    }

    fn is_complex(&self) -> bool {
        false
    }
}

/// Paul wavelet
///
/// The Paul wavelet is a complex wavelet defined in the frequency domain:
///
/// ```text
/// ψ(ω) = (2^m i^m m!) / √(π(2m)!) ω^m exp(-ω)  for ω > 0
/// ψ(ω) = 0  for ω ≤ 0
/// ```
///
/// where m is the order (typically 4).
pub struct PaulWavelet {
    order: u32,
}

impl PaulWavelet {
    /// Create a new Paul wavelet with specified order
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the wavelet (default: 4)
    pub fn new(order: u32) -> WaveletResult<Self> {
        if order == 0 {
            return Err(WaveletError::InvalidWavelet(
                "Order must be positive".to_string(),
            ));
        }
        Ok(Self { order })
    }

    /// Create a default Paul wavelet (order = 4)
    pub fn default_wavelet() -> Self {
        Self { order: 4 }
    }

    fn factorial(n: u32) -> f64 {
        (1..=n).map(|i| i as f64).product()
    }
}

impl ContinuousWavelet for PaulWavelet {
    fn psi(&self, t: f64, scale: f64) -> (f64, f64) {
        let t_scaled = t / scale;
        let m = self.order as f64;

        // Compute normalization constant
        let numerator = 2_f64.powf(m) * Self::factorial(self.order);
        let denominator = (PI * Self::factorial(2 * self.order)).sqrt();
        let norm = numerator / denominator / scale.sqrt();

        // Compute wavelet value in time domain.
        //
        // NOTE: an abandoned first attempt at this lived here as
        // `(1.0 - (0.0, 1.0).0 * t_scaled).powi(...)` -- `(0.0, 1.0).0` is
        // just the tuple's first field (`0.0`), not the imaginary unit, so
        // that expression always evaluated to exactly `1.0` regardless of
        // `t_scaled`/`self.order` and was never used below. Removed as dead
        // code; the real computation is the polar-form approach beneath.
        //
        // Un-investigated while removing the above (out of scope for this
        // pass, flagged for the wavelets owner): the sign of `phase`. Target
        // is `1/(1 - i*t_scaled)^(m+1)`; by this reviewer's derivation
        // `1 - i*t_scaled` has argument `-atan(t_scaled)`, so its `(m+1)`
        // power has argument `-(m+1)*atan(t_scaled)`, making the
        // *reciprocal*'s argument `+(m+1)*atan(t_scaled)` -- the opposite
        // sign from the `phase` below. Numeric check at `t_scaled=1, m=0`:
        // `1/(1-i) = 0.5+0.5i`, but this function's `real`/`imag` below
        // evaluate to `0.5`/`-0.5` (imaginary part negated). Neither of this
        // struct's two tests (`test_paul_wavelet_creation`,
        // `test_paul_invalid_order`) calls `.psi()`, so nothing currently
        // exercises this value either way.
        let complex_denom = (1.0 + t_scaled * t_scaled).powf((self.order as f64 + 1.0) / 2.0);
        let phase = -(self.order as f64 + 1.0) * t_scaled.atan();

        let real = norm * phase.cos() / complex_denom;
        let imag = norm * phase.sin() / complex_denom;

        (real, imag)
    }

    fn name(&self) -> &str {
        "paul"
    }

    fn center_frequency(&self) -> f64 {
        self.order as f64 / (2.0 * PI)
    }

    fn is_complex(&self) -> bool {
        true
    }
}

/// Result structure for CWT
///
/// Contains the CWT coefficients and metadata about scales and time positions.
#[derive(Debug, Clone)]
pub struct CWTResult {
    /// CWT coefficients: coefficients\[scale_idx\]\[time_idx\] = (real, imag)
    pub coefficients: Vec<Vec<(f64, f64)>>,
    /// Scale values used
    pub scales: Vec<f64>,
    /// Time positions
    pub times: Vec<f64>,
    /// Sampling period
    pub dt: f64,
}

impl CWTResult {
    /// Get the magnitude of coefficients at all scales and times
    pub fn magnitude(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(re, im)| (re * re + im * im).sqrt())
                    .collect()
            })
            .collect()
    }

    /// Get the phase of coefficients at all scales and times
    pub fn phase(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|row| row.iter().map(|&(re, im)| im.atan2(re)).collect())
            .collect()
    }

    /// Get the power (squared magnitude) of coefficients
    pub fn power(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|row| row.iter().map(|&(re, im)| re * re + im * im).collect())
            .collect()
    }

    /// Convert scales to frequencies
    ///
    /// # Arguments
    ///
    /// * `center_freq` - Center frequency of the wavelet
    pub fn scales_to_frequencies(&self, center_freq: f64) -> Vec<f64> {
        self.scales
            .iter()
            .map(|&scale| center_freq / (scale * self.dt))
            .collect()
    }
}

/// Compute Continuous Wavelet Transform
///
/// Performs CWT of a signal using the specified wavelet at multiple scales.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Continuous wavelet to use
/// * `scales` - Array of scale values (a > 0)
/// * `dt` - Sampling period (default: 1.0)
///
/// # Returns
///
/// `CWTResult` containing coefficients and metadata
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::cwt::{cwt, MorletWavelet};
///
/// let signal = vec![0.0; 100];
/// let wavelet = MorletWavelet::default_wavelet();
/// let scales: Vec<f64> = (1..50).map(|i| i as f64).collect();
/// let result = cwt(&signal, &wavelet, &scales, 1.0)?;
/// let magnitude = result.magnitude();
/// ```
pub fn cwt(
    signal: &[f64],
    wavelet: &dyn ContinuousWavelet,
    scales: &[f64],
    dt: f64,
) -> WaveletResult<CWTResult> {
    let n = signal.len();
    if n == 0 {
        return Err(WaveletError::InvalidLength(
            "Signal must not be empty".to_string(),
        ));
    }

    if scales.is_empty() {
        return Err(WaveletError::InvalidScale(
            "Scales array must not be empty".to_string(),
        ));
    }

    // Validate scales
    for &scale in scales {
        if scale <= 0.0 {
            return Err(WaveletError::InvalidScale(format!(
                "All scales must be positive, got {}",
                scale
            )));
        }
    }

    if dt <= 0.0 {
        return Err(WaveletError::InvalidScale(format!(
            "Sampling period must be positive, got {}",
            dt
        )));
    }

    // Compute time positions
    let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();

    // Allocate coefficient matrix
    let mut coefficients = vec![vec![(0.0, 0.0); n]; scales.len()];

    // Compute CWT for each scale
    for (scale_idx, &scale) in scales.iter().enumerate() {
        for time_idx in 0..times.len() {
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;

            // Convolve with wavelet
            for (k, &x_k) in signal.iter().enumerate() {
                let tau = (k as f64 - time_idx as f64) * dt;
                let (psi_real, psi_imag) = wavelet.psi(tau, scale);

                // Complex conjugate for correlation
                sum_real += x_k * psi_real;
                sum_imag += x_k * (-psi_imag);
            }

            coefficients[scale_idx][time_idx] = (sum_real * dt, sum_imag * dt);
        }
    }

    Ok(CWTResult {
        coefficients,
        scales: scales.to_vec(),
        times,
        dt,
    })
}

/// Generate logarithmically-spaced scales
///
/// # Arguments
///
/// * `min_scale` - Minimum scale
/// * `max_scale` - Maximum scale
/// * `num_scales` - Number of scales
///
/// # Returns
///
/// Vector of logarithmically-spaced scales
pub fn logspace_scales(
    min_scale: f64,
    max_scale: f64,
    num_scales: usize,
) -> WaveletResult<Vec<f64>> {
    if min_scale <= 0.0 || max_scale <= 0.0 {
        return Err(WaveletError::InvalidScale(
            "Scales must be positive".to_string(),
        ));
    }

    if min_scale >= max_scale {
        return Err(WaveletError::InvalidScale(
            "min_scale must be less than max_scale".to_string(),
        ));
    }

    if num_scales == 0 {
        return Err(WaveletError::InvalidScale(
            "num_scales must be positive".to_string(),
        ));
    }

    let log_min = min_scale.ln();
    let log_max = max_scale.ln();
    let step = (log_max - log_min) / (num_scales - 1) as f64;

    let scales: Vec<f64> = (0..num_scales)
        .map(|i| (log_min + i as f64 * step).exp())
        .collect();

    Ok(scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morlet_wavelet_creation() {
        let wavelet = MorletWavelet::new(6.0).expect("Failed to create Morlet");
        assert_eq!(wavelet.name(), "morlet");
        assert!(wavelet.is_complex());
        assert!(wavelet.center_frequency() > 0.0);

        let wavelet = MorletWavelet::default_wavelet();
        assert_eq!(wavelet.name(), "morlet");
    }

    #[test]
    fn test_morlet_invalid_omega() {
        let result = MorletWavelet::new(0.0);
        assert!(result.is_err());

        let result = MorletWavelet::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mexican_hat_wavelet() {
        let wavelet = MexicanHatWavelet;
        assert_eq!(wavelet.name(), "mexican_hat");
        assert!(!wavelet.is_complex());
        assert!(wavelet.center_frequency() > 0.0);

        let (real, imag) = wavelet.psi(0.0, 1.0);
        assert!(real > 0.0);
        assert_eq!(imag, 0.0);
    }

    #[test]
    fn test_paul_wavelet_creation() {
        let wavelet = PaulWavelet::new(4).expect("Failed to create Paul");
        assert_eq!(wavelet.name(), "paul");
        assert!(wavelet.is_complex());

        let wavelet = PaulWavelet::default_wavelet();
        assert_eq!(wavelet.name(), "paul");
    }

    #[test]
    fn test_paul_invalid_order() {
        let result = PaulWavelet::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cwt_basic() {
        let signal = vec![1.0; 16];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0, 2.0, 4.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0).expect("CWT failed");

        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.coefficients[0].len(), 16);
        assert_eq!(result.scales.len(), 3);
        assert_eq!(result.times.len(), 16);
    }

    #[test]
    fn test_cwt_mexican_hat() {
        let signal = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let wavelet = MexicanHatWavelet;
        let scales = vec![0.5, 1.0, 2.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0).expect("CWT failed");

        assert_eq!(result.coefficients.len(), 3);
        let magnitude = result.magnitude();
        assert_eq!(magnitude.len(), 3);
        assert_eq!(magnitude[0].len(), 5);
    }

    #[test]
    fn test_cwt_empty_signal() {
        let signal: Vec<f64> = vec![];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cwt_empty_scales() {
        let signal = vec![1.0; 10];
        let wavelet = MorletWavelet::default_wavelet();
        let scales: Vec<f64> = vec![];

        let result = cwt(&signal, &wavelet, &scales, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cwt_invalid_scale() {
        let signal = vec![1.0; 10];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0, -2.0, 3.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cwt_invalid_dt() {
        let signal = vec![1.0; 10];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0];

        let result = cwt(&signal, &wavelet, &scales, 0.0);
        assert!(result.is_err());

        let result = cwt(&signal, &wavelet, &scales, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cwt_result_magnitude() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let wavelet = MexicanHatWavelet;
        let scales = vec![1.0, 2.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0).expect("CWT failed");

        let magnitude = result.magnitude();
        assert_eq!(magnitude.len(), 2);
        assert_eq!(magnitude[0].len(), 5);

        // All magnitudes should be non-negative
        for row in &magnitude {
            for &val in row {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_cwt_result_power() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0];

        let result = cwt(&signal, &wavelet, &scales, 1.0).expect("CWT failed");

        let power = result.power();
        let magnitude = result.magnitude();

        // Power should be magnitude squared
        for (p_row, m_row) in power.iter().zip(magnitude.iter()) {
            for (&p, &m) in p_row.iter().zip(m_row.iter()) {
                assert!((p - m * m).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_scales_to_frequencies() {
        let signal = vec![1.0; 10];
        let wavelet = MorletWavelet::default_wavelet();
        let scales = vec![1.0, 2.0, 4.0];

        let result = cwt(&signal, &wavelet, &scales, 0.1).expect("CWT failed");

        let frequencies = result.scales_to_frequencies(wavelet.center_frequency());
        assert_eq!(frequencies.len(), 3);

        // Frequencies should decrease as scales increase
        assert!(frequencies[0] > frequencies[1]);
        assert!(frequencies[1] > frequencies[2]);
    }

    #[test]
    fn test_logspace_scales() {
        let scales = logspace_scales(1.0, 100.0, 10).expect("Failed to generate scales");

        assert_eq!(scales.len(), 10);
        assert!((scales[0] - 1.0).abs() < 1e-10);
        assert!((scales[9] - 100.0).abs() < 1e-8);

        // Check logarithmic spacing
        for i in 1..scales.len() {
            assert!(scales[i] > scales[i - 1]);
        }
    }

    #[test]
    fn test_logspace_scales_invalid() {
        assert!(logspace_scales(0.0, 10.0, 10).is_err());
        assert!(logspace_scales(10.0, 1.0, 10).is_err());
        assert!(logspace_scales(1.0, 10.0, 0).is_err());
    }
}

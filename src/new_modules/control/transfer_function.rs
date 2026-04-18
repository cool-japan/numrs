//! Transfer Function Representation
//!
//! This module provides transfer function representation and analysis for control systems.
//!
//! A transfer function represents a linear time-invariant (LTI) system in the frequency domain:
//! ```text
//! H(s) = N(s)/D(s) = (b_n*s^n + ... + b_1*s + b_0)/(a_m*s^m + ... + a_1*s + a_0)
//! ```
//!
//! For discrete systems, s is replaced with z.

use super::{ControlError, ControlResult, FrequencyResponse, SystemType};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;
use std::f64::consts::PI;

/// Transfer function representation: H(s) = N(s)/D(s)
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator polynomial coefficients (highest degree first)
    numerator: Array1<f64>,
    /// Denominator polynomial coefficients (highest degree first)
    denominator: Array1<f64>,
    /// System type (continuous or discrete)
    system_type: SystemType,
}

impl TransferFunction {
    /// Create a new transfer function
    ///
    /// # Arguments
    /// * `numerator` - Numerator coefficients in descending order of powers
    /// * `denominator` - Denominator coefficients in descending order of powers
    ///
    /// # Example
    /// ```
    /// use numrs2::new_modules::control::TransferFunction;
    ///
    /// // H(s) = (s + 2)/(s^2 + 3s + 2)
    /// let num = vec![1.0, 2.0];  // s + 2
    /// let den = vec![1.0, 3.0, 2.0];  // s^2 + 3s + 2
    /// let tf = TransferFunction::new(num, den).expect("valid transfer function");
    /// ```
    pub fn new(numerator: Vec<f64>, denominator: Vec<f64>) -> ControlResult<Self> {
        if numerator.is_empty() || denominator.is_empty() {
            return Err(ControlError::InvalidPolynomial(
                "Coefficients cannot be empty".to_string(),
            ));
        }

        // Check if denominator is all zeros
        if denominator.iter().all(|&x| x.abs() < 1e-15) {
            return Err(ControlError::InvalidPolynomial(
                "Denominator cannot be all zeros".to_string(),
            ));
        }

        Ok(Self {
            numerator: Array1::from_vec(numerator),
            denominator: Array1::from_vec(denominator),
            system_type: SystemType::Continuous,
        })
    }

    /// Create a discrete-time transfer function
    pub fn new_discrete(
        numerator: Vec<f64>,
        denominator: Vec<f64>,
        sample_time: f64,
    ) -> ControlResult<Self> {
        if sample_time <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Sample time must be positive".to_string(),
            ));
        }

        let mut tf = Self::new(numerator, denominator)?;
        tf.system_type = SystemType::Discrete {
            sample_time: (sample_time * 1_000_000.0) as u64,
        };
        Ok(tf)
    }

    /// Get the numerator coefficients
    pub fn numerator(&self) -> &Array1<f64> {
        &self.numerator
    }

    /// Get the denominator coefficients
    pub fn denominator(&self) -> &Array1<f64> {
        &self.denominator
    }

    /// Get the system type
    pub fn system_type(&self) -> SystemType {
        self.system_type
    }

    /// Evaluate the transfer function at a complex frequency
    ///
    /// For continuous systems: H(s)
    /// For discrete systems: H(z)
    pub fn eval(&self, s: Complex64) -> Complex64 {
        let num_val = eval_polynomial(&self.numerator, s);
        let den_val = eval_polynomial(&self.denominator, s);
        num_val / den_val
    }

    /// Compute the zeros of the transfer function (roots of numerator)
    pub fn zeros(&self) -> ControlResult<Vec<Complex64>> {
        find_polynomial_roots(&self.numerator)
    }

    /// Compute the poles of the transfer function (roots of denominator)
    pub fn poles(&self) -> ControlResult<Vec<Complex64>> {
        find_polynomial_roots(&self.denominator)
    }

    /// Compute the DC gain (H(0) for continuous, H(1) for discrete)
    pub fn dc_gain(&self) -> f64 {
        match self.system_type {
            SystemType::Continuous => {
                // DC gain = H(0): evaluate at s=0
                // For polynomials in descending power order, only the constant term
                // (last coefficient) is non-zero when s=0, since all s^k terms vanish.
                let num_const = self.numerator.iter().last().copied().unwrap_or(0.0);
                let den_const = self.denominator.iter().last().copied().unwrap_or(1.0);
                num_const / den_const
            }
            SystemType::Discrete { .. } => {
                // DC gain = H(1): evaluate at z=1
                // For H(z) at z=1, all z^k = 1, so sum of coefficients is correct
                self.eval(Complex64::new(1.0, 0.0)).re
            }
        }
    }

    /// Compute frequency response at given frequencies
    ///
    /// # Arguments
    /// * `frequencies` - Angular frequencies in rad/s (for continuous) or normalized frequencies (for discrete)
    ///
    /// # Returns
    /// FrequencyResponse containing magnitude and phase at each frequency
    pub fn frequency_response(&self, frequencies: &[f64]) -> ControlResult<FrequencyResponse> {
        let mut magnitudes = Vec::with_capacity(frequencies.len());
        let mut phases = Vec::with_capacity(frequencies.len());

        for &omega in frequencies {
            let s = match self.system_type {
                SystemType::Continuous => Complex64::new(0.0, omega),
                SystemType::Discrete { sample_time } => {
                    let ts = sample_time as f64 / 1_000_000.0;
                    Complex64::new(0.0, omega * ts).exp()
                }
            };

            let h = self.eval(s);
            magnitudes.push(h.norm());
            phases.push(h.arg());
        }

        FrequencyResponse::new(
            Array1::from_vec(frequencies.to_vec()),
            Array1::from_vec(magnitudes),
            Array1::from_vec(phases),
        )
    }

    /// Compute frequency response over a logarithmic frequency range
    ///
    /// # Arguments
    /// * `start` - Starting frequency (rad/s)
    /// * `end` - Ending frequency (rad/s)
    /// * `num_points` - Number of points
    pub fn bode_response(
        &self,
        start: f64,
        end: f64,
        num_points: usize,
    ) -> ControlResult<FrequencyResponse> {
        if start <= 0.0 || end <= start {
            return Err(ControlError::InvalidParameters(
                "Invalid frequency range".to_string(),
            ));
        }

        let log_start = start.ln();
        let log_end = end.ln();
        let step = (log_end - log_start) / (num_points - 1) as f64;

        let frequencies: Vec<f64> = (0..num_points)
            .map(|i| (log_start + i as f64 * step).exp())
            .collect();

        self.frequency_response(&frequencies)
    }

    /// Series connection: H(s) = H1(s) * H2(s)
    pub fn series(&self, other: &TransferFunction) -> ControlResult<TransferFunction> {
        if self.system_type != other.system_type {
            return Err(ControlError::InvalidSystem(
                "Cannot connect systems of different types".to_string(),
            ));
        }

        let num = convolve_polynomials(&self.numerator, &other.numerator);
        let den = convolve_polynomials(&self.denominator, &other.denominator);

        let mut result = TransferFunction::new(num.to_vec(), den.to_vec())?;
        result.system_type = self.system_type;
        Ok(result)
    }

    /// Parallel connection: H(s) = H1(s) + H2(s)
    pub fn parallel(&self, other: &TransferFunction) -> ControlResult<TransferFunction> {
        if self.system_type != other.system_type {
            return Err(ControlError::InvalidSystem(
                "Cannot connect systems of different types".to_string(),
            ));
        }

        // N(s) = N1(s)*D2(s) + N2(s)*D1(s)
        let n1d2 = convolve_polynomials(&self.numerator, &other.denominator);
        let n2d1 = convolve_polynomials(&other.numerator, &self.denominator);
        let num = add_polynomials(&n1d2, &n2d1);

        // D(s) = D1(s)*D2(s)
        let den = convolve_polynomials(&self.denominator, &other.denominator);

        let mut result = TransferFunction::new(num.to_vec(), den.to_vec())?;
        result.system_type = self.system_type;
        Ok(result)
    }

    /// Feedback connection: H(s) = G(s)/(1 + G(s)*H(s))
    ///
    /// # Arguments
    /// * `feedback_tf` - Feedback transfer function H(s)
    /// * `negative` - If true, uses negative feedback (default), otherwise positive
    pub fn feedback(
        &self,
        feedback_tf: &TransferFunction,
        negative: bool,
    ) -> ControlResult<TransferFunction> {
        if self.system_type != feedback_tf.system_type {
            return Err(ControlError::InvalidSystem(
                "Cannot connect systems of different types".to_string(),
            ));
        }

        // G(s)*H(s)
        let gh = self.series(feedback_tf)?;

        // 1 + G(s)*H(s) or 1 - G(s)*H(s)
        let one = Array1::from_vec(vec![1.0]);
        let one_tf = TransferFunction::new(vec![1.0], vec![1.0])?;

        let denominator_tf = if negative {
            one_tf.parallel(&gh)?
        } else {
            // For positive feedback: 1 - G(s)*H(s)
            let mut neg_gh = gh.clone();
            neg_gh.numerator = -neg_gh.numerator;
            one_tf.parallel(&neg_gh)?
        };

        // G(s) / (1 + G(s)*H(s))
        let num = convolve_polynomials(&self.numerator, &denominator_tf.denominator);
        let den = convolve_polynomials(&self.denominator, &denominator_tf.numerator);

        let mut result = TransferFunction::new(num.to_vec(), den.to_vec())?;
        result.system_type = self.system_type;
        Ok(result)
    }

    /// Compute gain margin and phase margin
    ///
    /// Returns (gain_margin_db, phase_margin_deg, gain_crossover_freq, phase_crossover_freq)
    pub fn stability_margins(&self) -> ControlResult<(f64, f64, f64, f64)> {
        // Generate frequency response
        let response = self.bode_response(0.001, 1000.0, 1000)?;

        // Find phase crossover frequency (where phase = -180 deg)
        let mut phase_crossover_freq = 0.0;
        let mut gain_margin_db = f64::INFINITY;

        for i in 0..response.phase.len() {
            let phase_deg = response.phase[i].to_degrees();
            if phase_deg <= -180.0 {
                phase_crossover_freq = response.frequencies[i];
                gain_margin_db = -response.magnitude_db()[i];
                break;
            }
        }

        // Find gain crossover frequency (where |H(jw)| = 1, i.e., 0 dB)
        let mut gain_crossover_freq = 0.0;
        let mut phase_margin_deg = 0.0;

        for i in 0..response.magnitude.len() {
            if response.magnitude[i] <= 1.0 {
                gain_crossover_freq = response.frequencies[i];
                phase_margin_deg = 180.0 + response.phase[i].to_degrees();
                break;
            }
        }

        Ok((
            gain_margin_db,
            phase_margin_deg,
            gain_crossover_freq,
            phase_crossover_freq,
        ))
    }
}

/// Evaluate polynomial at a complex point
fn eval_polynomial(coeffs: &Array1<f64>, x: Complex64) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    let mut power = Complex64::new(1.0, 0.0);

    // Evaluate from lowest to highest degree
    for &coeff in coeffs.iter().rev() {
        result += Complex64::new(coeff, 0.0) * power;
        power *= x;
    }

    result
}

/// Find roots of a polynomial using Durand-Kerner method
fn find_polynomial_roots(coeffs: &Array1<f64>) -> ControlResult<Vec<Complex64>> {
    // Remove leading zeros
    let mut start_idx = 0;
    for (i, &c) in coeffs.iter().enumerate() {
        if c.abs() > 1e-15 {
            start_idx = i;
            break;
        }
    }

    if start_idx == coeffs.len() - 1 {
        // Constant polynomial
        return Ok(Vec::new());
    }

    let degree = coeffs.len() - start_idx - 1;
    if degree == 0 {
        return Ok(Vec::new());
    }

    // For linear polynomial
    if degree == 1 {
        let root = Complex64::new(-coeffs[start_idx + 1] / coeffs[start_idx], 0.0);
        return Ok(vec![root]);
    }

    // For quadratic polynomial
    if degree == 2 {
        let a = coeffs[start_idx];
        let b = coeffs[start_idx + 1];
        let c = coeffs[start_idx + 2];
        return Ok(solve_quadratic(a, b, c));
    }

    // For higher degree polynomials, use Durand-Kerner method
    durand_kerner(coeffs, start_idx, degree)
}

/// Solve quadratic equation ax^2 + bx + c = 0
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<Complex64> {
    let discriminant = b * b - 4.0 * a * c;

    if discriminant >= 0.0 {
        let sqrt_disc = discriminant.sqrt();
        vec![
            Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0),
            Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0),
        ]
    } else {
        let real_part = -b / (2.0 * a);
        let imag_part = (-discriminant).sqrt() / (2.0 * a);
        vec![
            Complex64::new(real_part, imag_part),
            Complex64::new(real_part, -imag_part),
        ]
    }
}

/// Durand-Kerner root finding algorithm
fn durand_kerner(
    coeffs: &Array1<f64>,
    start_idx: usize,
    degree: usize,
) -> ControlResult<Vec<Complex64>> {
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    // Normalize coefficients
    let leading_coeff = coeffs[start_idx];
    let normalized: Vec<f64> = coeffs
        .iter()
        .skip(start_idx)
        .map(|&c| c / leading_coeff)
        .collect();

    // Initialize roots on unit circle
    let mut roots: Vec<Complex64> = (0..degree)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / degree as f64;
            Complex64::new(0.4 * theta.cos(), 0.4 * theta.sin())
        })
        .collect();

    // Iterate
    for _ in 0..MAX_ITER {
        let mut converged = true;
        let old_roots = roots.clone();

        for i in 0..degree {
            let mut p = Complex64::new(0.0, 0.0);
            let mut x_power = Complex64::new(1.0, 0.0);

            for &coeff in normalized.iter().rev() {
                p += Complex64::new(coeff, 0.0) * x_power;
                x_power *= roots[i];
            }

            let mut denom = Complex64::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom *= roots[i] - old_roots[j];
                }
            }

            if denom.norm() > 1e-15 {
                roots[i] -= p / denom;
            }

            if (roots[i] - old_roots[i]).norm() > TOL {
                converged = false;
            }
        }

        if converged {
            break;
        }
    }

    Ok(roots)
}

/// Convolve two polynomials (multiply)
fn convolve_polynomials(p1: &Array1<f64>, p2: &Array1<f64>) -> Array1<f64> {
    let len = p1.len() + p2.len() - 1;
    let mut result = vec![0.0; len];

    for i in 0..p1.len() {
        for j in 0..p2.len() {
            result[i + j] += p1[i] * p2[j];
        }
    }

    Array1::from_vec(result)
}

/// Add two polynomials (padding with zeros if needed)
fn add_polynomials(p1: &Array1<f64>, p2: &Array1<f64>) -> Array1<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    let offset1 = max_len - p1.len();
    let offset2 = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[i + offset1] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[i + offset2] += val;
    }

    Array1::from_vec(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_function_creation() {
        let tf = TransferFunction::new(vec![1.0, 2.0], vec![1.0, 3.0, 2.0]);
        assert!(tf.is_ok());
    }

    #[test]
    fn test_invalid_transfer_function() {
        let tf = TransferFunction::new(vec![], vec![1.0, 2.0]);
        assert!(tf.is_err());

        let tf = TransferFunction::new(vec![1.0], vec![0.0, 0.0]);
        assert!(tf.is_err());
    }

    #[test]
    fn test_eval_polynomial() {
        // p(x) = 1*x^2 + 2*x + 3
        let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = Complex64::new(2.0, 0.0);
        let result = eval_polynomial(&coeffs, x);

        // 1*4 + 2*2 + 3 = 11
        assert!((result.re - 11.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_dc_gain() {
        // H(s) = 2/(s + 1)
        let tf = TransferFunction::new(vec![2.0], vec![1.0, 1.0])
            .expect("test: valid transfer function");
        let dc_gain = tf.dc_gain();
        assert!((dc_gain - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_connection() {
        // H1(s) = 1/(s+1), H2(s) = 1/(s+2)
        let h1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0])
            .expect("test: valid transfer function");
        let h2 = TransferFunction::new(vec![1.0], vec![1.0, 2.0])
            .expect("test: valid transfer function");

        let series = h1.series(&h2).expect("test: valid series connection");

        // Result should be 1/((s+1)(s+2)) = 1/(s^2 + 3s + 2)
        assert_eq!(series.numerator.len(), 1);
        assert_eq!(series.denominator.len(), 3);
        assert!((series.denominator[0] - 1.0).abs() < 1e-10);
        assert!((series.denominator[1] - 3.0).abs() < 1e-10);
        assert!((series.denominator[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_roots_real() {
        // x^2 - 3x + 2 = 0, roots are 1 and 2
        let roots = solve_quadratic(1.0, -3.0, 2.0);
        assert_eq!(roots.len(), 2);

        let root_values: Vec<f64> = roots.iter().map(|r| r.re).collect();
        assert!(root_values.contains(&1.0) || (root_values[0] - 1.0).abs() < 1e-10);
        assert!(root_values.contains(&2.0) || (root_values[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_roots_complex() {
        // x^2 + 1 = 0, roots are ±i
        let roots = solve_quadratic(1.0, 0.0, 1.0);
        assert_eq!(roots.len(), 2);

        assert!(roots[0].re.abs() < 1e-10);
        assert!((roots[0].im.abs() - 1.0).abs() < 1e-10);
        assert!(roots[1].re.abs() < 1e-10);
        assert!((roots[1].im.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_convolution() {
        // (x + 1)(x + 2) = x^2 + 3x + 2
        let p1 = Array1::from_vec(vec![1.0, 1.0]);
        let p2 = Array1::from_vec(vec![1.0, 2.0]);
        let result = convolve_polynomials(&p1, &p2);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_response() {
        // H(s) = 1/(s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0])
            .expect("test: valid transfer function");
        let freqs = vec![0.0, 1.0, 10.0];
        let response = tf
            .frequency_response(&freqs)
            .expect("test: valid frequency response");

        assert_eq!(response.frequencies.len(), 3);
        assert_eq!(response.magnitude.len(), 3);
        assert_eq!(response.phase.len(), 3);

        // At ω=0, |H(j0)| = 1
        assert!((response.magnitude[0] - 1.0).abs() < 1e-10);
    }
}

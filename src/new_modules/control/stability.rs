//! Stability Analysis Tools
//!
//! This module provides various stability analysis tools for control systems including:
//! - Routh-Hurwitz stability criterion
//! - Nyquist stability analysis
//! - Bode plot generation
//! - Root locus analysis
//! - Gain and phase margins
//! - Lyapunov stability

use super::{ControlError, ControlResult, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;
use std::f64::consts::PI;

/// Stability margins (gain margin and phase margin)
#[derive(Debug, Clone)]
pub struct StabilityMargins {
    /// Gain margin in dB
    pub gain_margin_db: f64,
    /// Phase margin in degrees
    pub phase_margin_deg: f64,
    /// Gain crossover frequency (rad/s)
    pub gain_crossover_freq: f64,
    /// Phase crossover frequency (rad/s)
    pub phase_crossover_freq: f64,
}

/// Bode plot data
#[derive(Debug, Clone)]
pub struct BodePlotData {
    /// Frequencies (rad/s)
    pub frequencies: Array1<f64>,
    /// Magnitude (dB)
    pub magnitude_db: Array1<f64>,
    /// Phase (degrees)
    pub phase_deg: Array1<f64>,
}

/// Nyquist plot data
#[derive(Debug, Clone)]
pub struct NyquistPlotData {
    /// Real part of H(jω)
    pub real: Array1<f64>,
    /// Imaginary part of H(jω)
    pub imag: Array1<f64>,
    /// Frequencies at which points were computed
    pub frequencies: Array1<f64>,
}

/// Routh-Hurwitz stability criterion
///
/// Tests whether all roots of a polynomial have negative real parts (for continuous systems)
#[derive(Debug)]
pub struct RouthHurwitz {
    /// Routh array
    table: Vec<Vec<f64>>,
}

impl RouthHurwitz {
    /// Create a Routh-Hurwitz table from polynomial coefficients
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients in descending order of powers
    ///
    /// # Example
    /// For polynomial s^3 + 2s^2 + 3s + 4, use coeffs = [1.0, 2.0, 3.0, 4.0]
    pub fn new(coeffs: &[f64]) -> ControlResult<Self> {
        if coeffs.is_empty() {
            return Err(ControlError::InvalidPolynomial(
                "Coefficient array cannot be empty".to_string(),
            ));
        }

        // Remove leading zeros
        let mut start_idx = 0;
        for (i, &c) in coeffs.iter().enumerate() {
            if c.abs() > 1e-15 {
                start_idx = i;
                break;
            }
        }

        let clean_coeffs = &coeffs[start_idx..];
        if clean_coeffs.is_empty() {
            return Err(ControlError::InvalidPolynomial(
                "All coefficients are zero".to_string(),
            ));
        }

        let n = clean_coeffs.len();
        let num_rows = n;

        let mut table = Vec::new();

        // First row: even-indexed coefficients
        let mut row1 = Vec::new();
        for i in (0..n).step_by(2) {
            row1.push(clean_coeffs[i]);
        }
        table.push(row1);

        // Second row: odd-indexed coefficients
        let mut row2 = Vec::new();
        for i in (1..n).step_by(2) {
            row2.push(clean_coeffs[i]);
        }
        // Pad with zeros if needed
        while row2.len() < table[0].len() {
            row2.push(0.0);
        }
        table.push(row2);

        // Subsequent rows
        for i in 2..num_rows {
            let mut new_row = Vec::new();
            let prev1 = &table[i - 1];
            let prev2 = &table[i - 2];

            let num_cols = prev1.len().max(prev2.len()) - 1;

            for j in 0..num_cols {
                let a = if i - 1 < table.len() && j + 1 < table[i - 1].len() {
                    table[i - 1][j + 1]
                } else {
                    0.0
                };

                let b = if i - 2 < table.len() && j + 1 < table[i - 2].len() {
                    table[i - 2][j + 1]
                } else {
                    0.0
                };

                let denominator = if i - 1 < table.len() && !table[i - 1].is_empty() {
                    table[i - 1][0]
                } else {
                    return Err(ControlError::NumericalError(
                        "Division by zero in Routh table".to_string(),
                    ));
                };

                if denominator.abs() < 1e-15 {
                    // Handle zero in first column by replacing with small epsilon
                    new_row.push(1e-10);
                } else {
                    let numerator = if i - 2 < table.len() && !table[i - 2].is_empty() {
                        table[i - 2][0]
                    } else {
                        0.0
                    };

                    let value = (denominator * b - numerator * a) / denominator;
                    new_row.push(value);
                }
            }

            if new_row.is_empty() {
                break;
            }

            table.push(new_row);
        }

        Ok(Self { table })
    }

    /// Get the Routh table
    pub fn table(&self) -> &[Vec<f64>] {
        &self.table
    }

    /// Check if the system is stable
    ///
    /// Returns true if all entries in the first column have the same sign
    pub fn is_stable(&self) -> bool {
        if self.table.is_empty() {
            return false;
        }

        let mut sign_changes = 0;
        let mut prev_sign = 0;

        for row in &self.table {
            if row.is_empty() {
                continue;
            }

            let value = row[0];
            if value.abs() < 1e-15 {
                continue;
            }

            let curr_sign = if value > 0.0 { 1 } else { -1 };

            if prev_sign != 0 && curr_sign != prev_sign {
                sign_changes += 1;
            }

            prev_sign = curr_sign;
        }

        sign_changes == 0
    }

    /// Count the number of unstable poles (roots in right half-plane)
    pub fn count_unstable_poles(&self) -> usize {
        if self.table.is_empty() {
            return 0;
        }

        let mut sign_changes = 0;
        let mut prev_sign = 0;

        for row in &self.table {
            if row.is_empty() {
                continue;
            }

            let value = row[0];
            if value.abs() < 1e-15 {
                continue;
            }

            let curr_sign = if value > 0.0 { 1 } else { -1 };

            if prev_sign != 0 && curr_sign != prev_sign {
                sign_changes += 1;
            }

            prev_sign = curr_sign;
        }

        sign_changes
    }
}

/// Generate Bode plot data
///
/// # Arguments
/// * `tf` - Transfer function
/// * `freq_range` - (start_freq, end_freq) in rad/s
/// * `num_points` - Number of frequency points (logarithmically spaced)
pub fn bode_plot(
    tf: &TransferFunction,
    freq_range: (f64, f64),
    num_points: usize,
) -> ControlResult<BodePlotData> {
    let (start_freq, end_freq) = freq_range;

    if start_freq <= 0.0 || end_freq <= start_freq {
        return Err(ControlError::InvalidParameters(
            "Invalid frequency range".to_string(),
        ));
    }

    if num_points == 0 {
        return Err(ControlError::InvalidParameters(
            "Number of points must be positive".to_string(),
        ));
    }

    let response = tf.bode_response(start_freq, end_freq, num_points)?;

    Ok(BodePlotData {
        frequencies: response.frequencies.clone(),
        magnitude_db: response.magnitude_db(),
        phase_deg: response.phase_deg(),
    })
}

/// Generate Nyquist plot data
///
/// # Arguments
/// * `tf` - Transfer function
/// * `freq_range` - (start_freq, end_freq) in rad/s
/// * `num_points` - Number of frequency points
pub fn nyquist_plot(
    tf: &TransferFunction,
    freq_range: (f64, f64),
    num_points: usize,
) -> ControlResult<NyquistPlotData> {
    let (start_freq, end_freq) = freq_range;

    if start_freq < 0.0 || end_freq <= start_freq {
        return Err(ControlError::InvalidParameters(
            "Invalid frequency range".to_string(),
        ));
    }

    if num_points == 0 {
        return Err(ControlError::InvalidParameters(
            "Number of points must be positive".to_string(),
        ));
    }

    // Linear spacing for Nyquist plot
    let step = (end_freq - start_freq) / (num_points - 1) as f64;
    let frequencies: Vec<f64> = (0..num_points)
        .map(|i| start_freq + i as f64 * step)
        .collect();

    let mut real_parts = Vec::with_capacity(num_points);
    let mut imag_parts = Vec::with_capacity(num_points);

    for &omega in &frequencies {
        let s = Complex64::new(0.0, omega);
        let h = tf.eval(s);
        real_parts.push(h.re);
        imag_parts.push(h.im);
    }

    Ok(NyquistPlotData {
        real: Array1::from_vec(real_parts),
        imag: Array1::from_vec(imag_parts),
        frequencies: Array1::from_vec(frequencies),
    })
}

/// Compute stability margins
pub fn stability_margins(tf: &TransferFunction) -> ControlResult<StabilityMargins> {
    let (gain_margin_db, phase_margin_deg, gain_crossover_freq, phase_crossover_freq) =
        tf.stability_margins()?;

    Ok(StabilityMargins {
        gain_margin_db,
        phase_margin_deg,
        gain_crossover_freq,
        phase_crossover_freq,
    })
}

/// Check BIBO (Bounded Input Bounded Output) stability
///
/// A system is BIBO stable if all poles have negative real parts (continuous)
/// or magnitude less than 1 (discrete)
pub fn is_bibo_stable(tf: &TransferFunction) -> ControlResult<bool> {
    let poles = tf.poles()?;

    match tf.system_type() {
        super::SystemType::Continuous => Ok(poles.iter().all(|&p| p.re < 0.0)),
        super::SystemType::Discrete { .. } => Ok(poles.iter().all(|&p: &Complex64| p.norm() < 1.0)),
    }
}

/// Root locus data point
#[derive(Debug, Clone)]
pub struct RootLocusPoint {
    /// Gain value
    pub gain: f64,
    /// Closed-loop poles at this gain
    pub poles: Vec<Complex64>,
}

/// Generate root locus data
///
/// Computes closed-loop poles for a range of gain values
///
/// # Arguments
/// * `open_loop_tf` - Open-loop transfer function G(s)
/// * `gain_range` - (min_gain, max_gain)
/// * `num_points` - Number of gain values to compute
pub fn root_locus(
    open_loop_tf: &TransferFunction,
    gain_range: (f64, f64),
    num_points: usize,
) -> ControlResult<Vec<RootLocusPoint>> {
    let (min_gain, max_gain) = gain_range;

    if min_gain < 0.0 || max_gain <= min_gain {
        return Err(ControlError::InvalidParameters(
            "Invalid gain range".to_string(),
        ));
    }

    if num_points == 0 {
        return Err(ControlError::InvalidParameters(
            "Number of points must be positive".to_string(),
        ));
    }

    let step = (max_gain - min_gain) / (num_points - 1) as f64;
    let mut results = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let k = min_gain + i as f64 * step;

        // Closed-loop: T(s) = K*G(s)/(1 + K*G(s))
        // Poles are roots of: D(s) + K*N(s) = 0
        let num = open_loop_tf.numerator();
        let den = open_loop_tf.denominator();

        // Characteristic equation: den + k*num = 0
        let char_poly = add_polynomials_scaled(den, num, k);

        // Find roots
        let poles = find_polynomial_roots_simple(&char_poly)?;

        results.push(RootLocusPoint { gain: k, poles });
    }

    Ok(results)
}

/// Solve Lyapunov equation: A*X + X*A' = -Q
///
/// Used for stability analysis. If Q is positive definite and there exists
/// a unique positive definite solution X, the system is stable.
pub fn solve_lyapunov(a: &Array2<f64>, q: &Array2<f64>) -> ControlResult<Array2<f64>> {
    let n = a.nrows();

    if a.ncols() != n {
        return Err(ControlError::DimensionMismatch {
            expected: "square A matrix".to_string(),
            actual: format!("{}×{}", n, a.ncols()),
        });
    }

    if q.nrows() != n || q.ncols() != n {
        return Err(ControlError::DimensionMismatch {
            expected: format!("Q matrix {}×{}", n, n),
            actual: format!("{}×{}", q.nrows(), q.ncols()),
        });
    }

    // Use Bartels-Stewart algorithm (simplified version)
    // For small matrices, we can use direct method

    // Vectorize: A ⊗ I + I ⊗ A^T
    let n2 = n * n;
    let mut m = Array2::zeros((n2, n2));

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let row = i * n + j;
                    let col = k * n + l;

                    // A ⊗ I term
                    if j == l {
                        m[[row, col]] += a[[i, k]];
                    }

                    // I ⊗ A^T term
                    if i == k {
                        m[[row, col]] += a[[l, j]];
                    }
                }
            }
        }
    }

    // Vectorize -Q
    let mut q_vec = Array1::zeros(n2);
    for i in 0..n {
        for j in 0..n {
            q_vec[i * n + j] = -q[[i, j]];
        }
    }

    // Solve M * x_vec = q_vec
    let x_vec = solve_linear_system(&m, &q_vec)?;

    // Reshape back to matrix
    let mut x = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = x_vec[i * n + j];
        }
    }

    Ok(x)
}

/// Add two polynomials with scaling: p1 + k*p2
fn add_polynomials_scaled(p1: &Array1<f64>, p2: &Array1<f64>, k: f64) -> Array1<f64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0.0; max_len];

    let offset1 = max_len - p1.len();
    let offset2 = max_len - p2.len();

    for (i, &val) in p1.iter().enumerate() {
        result[i + offset1] += val;
    }

    for (i, &val) in p2.iter().enumerate() {
        result[i + offset2] += k * val;
    }

    Array1::from_vec(result)
}

/// Find polynomial roots (simplified version for root locus)
fn find_polynomial_roots_simple(coeffs: &Array1<f64>) -> ControlResult<Vec<Complex64>> {
    // Remove leading zeros
    let mut start_idx = 0;
    for (i, &c) in coeffs.iter().enumerate() {
        if c.abs() > 1e-15 {
            start_idx = i;
            break;
        }
    }

    if start_idx >= coeffs.len() - 1 {
        return Ok(Vec::new());
    }

    let degree = coeffs.len() - start_idx - 1;

    if degree == 0 {
        return Ok(Vec::new());
    }

    if degree == 1 {
        let root = Complex64::new(-coeffs[start_idx + 1] / coeffs[start_idx], 0.0);
        return Ok(vec![root]);
    }

    if degree == 2 {
        let a = coeffs[start_idx];
        let b = coeffs[start_idx + 1];
        let c = coeffs[start_idx + 2];

        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            return Ok(vec![
                Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0),
                Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0),
            ]);
        } else {
            let real_part = -b / (2.0 * a);
            let imag_part = (-discriminant).sqrt() / (2.0 * a);
            return Ok(vec![
                Complex64::new(real_part, imag_part),
                Complex64::new(real_part, -imag_part),
            ]);
        }
    }

    // For higher degrees, use companion matrix eigenvalue method
    companion_matrix_eigenvalues(coeffs, start_idx, degree)
}

/// Compute eigenvalues of companion matrix (alternative root finding)
fn companion_matrix_eigenvalues(
    coeffs: &Array1<f64>,
    start_idx: usize,
    degree: usize,
) -> ControlResult<Vec<Complex64>> {
    // Normalize coefficients
    let leading = coeffs[start_idx];
    let normalized: Vec<f64> = coeffs
        .iter()
        .skip(start_idx + 1)
        .map(|&c| -c / leading)
        .collect();

    // Build companion matrix
    let mut companion = Array2::zeros((degree, degree));

    // First row is the normalized coefficients
    for j in 0..degree {
        companion[[0, j]] = normalized[j];
    }

    // Sub-diagonal is identity
    for i in 1..degree {
        companion[[i, i - 1]] = 1.0;
    }

    // Compute eigenvalues using simplified QR
    simple_eigenvalues(&companion)
}

/// Simplified eigenvalue computation
fn simple_eigenvalues(a: &Array2<f64>) -> ControlResult<Vec<Complex64>> {
    let n = a.nrows();
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    let mut ak = a.clone();

    for _ in 0..MAX_ITER {
        let (q, r) = simple_qr(&ak)?;
        let ak_new = matrix_mult_simple(&r, &q)?;

        let mut converged = true;
        for i in 0..n {
            if (ak_new[[i, i]] - ak[[i, i]]).abs() > TOL {
                converged = false;
                break;
            }
        }

        ak = ak_new;
        if converged {
            break;
        }
    }

    // Extract eigenvalues
    let mut eigenvalues = Vec::new();
    let mut i = 0;
    while i < n {
        if i < n - 1 && ak[[i + 1, i]].abs() > TOL {
            // 2x2 block
            let a_val = ak[[i, i]];
            let b = ak[[i, i + 1]];
            let c = ak[[i + 1, i]];
            let d = ak[[i + 1, i + 1]];

            let trace = a_val + d;
            let det = a_val * d - b * c;
            let disc = trace * trace - 4.0 * det;

            if disc < 0.0 {
                let real = trace / 2.0;
                let imag = (-disc).sqrt() / 2.0;
                eigenvalues.push(Complex64::new(real, imag));
                eigenvalues.push(Complex64::new(real, -imag));
            } else {
                let sqrt_d = disc.sqrt();
                eigenvalues.push(Complex64::new((trace + sqrt_d) / 2.0, 0.0));
                eigenvalues.push(Complex64::new((trace - sqrt_d) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigenvalues.push(Complex64::new(ak[[i, i]], 0.0));
            i += 1;
        }
    }

    Ok(eigenvalues)
}

/// Simple QR decomposition
fn simple_qr(a: &Array2<f64>) -> ControlResult<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();

        for i in 0..j {
            let q_col = q.column(i);
            let dot: f64 = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[[i, j]] = dot;

            for k in 0..m {
                v[k] -= r[[i, j]] * q[[k, i]];
            }
        }

        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[[j, j]] = norm;

        if norm > 1e-15 {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }

    Ok((q, r))
}

/// Simple matrix multiplication
fn matrix_mult_simple(a: &Array2<f64>, b: &Array2<f64>) -> ControlResult<Array2<f64>> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    if n != p {
        return Err(ControlError::DimensionMismatch {
            expected: format!("{}", n),
            actual: format!("{}", p),
        });
    }

    let mut result = Array2::zeros((m, q));
    for i in 0..m {
        for j in 0..q {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> ControlResult<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n {
        return Err(ControlError::LinAlgError(
            "Matrix must be square".to_string(),
        ));
    }

    if b.len() != n {
        return Err(ControlError::DimensionMismatch {
            expected: format!("{}", n),
            actual: format!("{}", b.len()),
        });
    }

    // Create augmented matrix
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();

        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(ControlError::LinAlgError("Singular matrix".to_string()));
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_routh_hurwitz_stable() {
        // s^3 + 2s^2 + 3s + 4 (stable)
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let rh = RouthHurwitz::new(&coeffs).unwrap();
        assert!(rh.is_stable());
    }

    #[test]
    fn test_routh_hurwitz_unstable() {
        // s^3 - 2s^2 + 3s + 4 (unstable due to negative coefficient)
        let coeffs = vec![1.0, -2.0, 3.0, 4.0];
        let rh = RouthHurwitz::new(&coeffs).unwrap();
        assert!(!rh.is_stable());
    }

    #[test]
    fn test_bibo_stability() {
        // Stable system: poles at -1, -2
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0]).unwrap();
        let stable = is_bibo_stable(&tf).unwrap();
        assert!(stable);
    }

    #[test]
    fn test_bode_plot_generation() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let bode = bode_plot(&tf, (0.1, 100.0), 50).unwrap();

        assert_eq!(bode.frequencies.len(), 50);
        assert_eq!(bode.magnitude_db.len(), 50);
        assert_eq!(bode.phase_deg.len(), 50);
    }

    #[test]
    fn test_nyquist_plot_generation() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let nyquist = nyquist_plot(&tf, (0.0, 10.0), 100).unwrap();

        assert_eq!(nyquist.real.len(), 100);
        assert_eq!(nyquist.imag.len(), 100);
        assert_eq!(nyquist.frequencies.len(), 100);
    }

    #[test]
    fn test_solve_linear_system() {
        // 2x + 3y = 8
        // 4x + 5y = 14
        // Solution: x = 1, y = 2
        let a = array![[2.0, 3.0], [4.0, 5.0]];
        let b = array![8.0, 14.0];

        let x = solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }
}

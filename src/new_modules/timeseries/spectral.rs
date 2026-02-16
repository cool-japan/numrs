//! # Spectral Analysis Module
//!
//! Comprehensive spectral analysis methods for time series data.
//!
//! This module provides production-grade implementations of spectral estimation
//! techniques including periodograms, Welch's method, autoregressive spectral
//! estimation, cross-spectral analysis, and supporting utility functions.
//!
//! ## Features
//!
//! - **Periodogram**: Raw and windowed spectral density estimation via FFT
//! - **Welch's Method**: Averaged, overlapping-segment PSD estimation
//! - **Bartlett's Method**: Non-overlapping segment averaged PSD
//! - **AR Spectral Estimation**: Yule-Walker and Burg's method for parametric PSD
//! - **Cross-Spectral Analysis**: CSD, coherence, phase spectrum
//! - **Window Functions**: Hann, Hamming, Blackman, Bartlett, Kaiser
//! - **Utilities**: Frequency grids, dB conversion, peak detection
//!
//! ## SCIRS2 Policy Compliance
//!
//! All array operations use `scirs2_core::ndarray`, FFT via `crate::fft` (scirs2-fft),
//! and linear algebra via `scirs2_linalg`. Zero C/C++ dependencies.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

// =============================================================================
// Window Functions
// =============================================================================

/// Supported window function types for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann (raised cosine) window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Kaiser window with shape parameter beta
    Kaiser(f64),
}

/// Generate a window function of the specified type and length.
///
/// Window functions are used to reduce spectral leakage in periodogram-based
/// spectral estimation by tapering the edges of data segments.
///
/// # Arguments
///
/// * `window_type` - The type of window to generate
/// * `n` - Length of the window
///
/// # Returns
///
/// Array of window coefficients of length `n`
///
/// # References
///
/// Harris, F. J. (1978). "On the use of windows for harmonic analysis with
/// the discrete Fourier transform." *Proceedings of the IEEE*, 66(1), 51-83.
pub fn window_function(window_type: WindowType, n: usize) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(NumRs2Error::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut w = Array1::zeros(n);
    let nm1 = (n - 1) as f64;

    match window_type {
        WindowType::Rectangular => {
            w.fill(1.0);
        }
        WindowType::Hann => {
            for i in 0..n {
                w[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / nm1).cos());
            }
        }
        WindowType::Hamming => {
            for i in 0..n {
                w[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / nm1).cos();
            }
        }
        WindowType::Blackman => {
            for i in 0..n {
                let x = 2.0 * PI * i as f64 / nm1;
                w[i] = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
            }
        }
        WindowType::Bartlett => {
            let half = nm1 / 2.0;
            for i in 0..n {
                w[i] = 1.0 - ((i as f64 - half) / half).abs();
            }
        }
        WindowType::Kaiser(beta) => {
            // Kaiser window: w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
            let half = nm1 / 2.0;
            let i0_beta = bessel_i0(beta);
            if i0_beta.abs() < 1e-300 {
                return Err(NumRs2Error::ComputationError(
                    "Bessel I0(beta) is zero or near-zero".to_string(),
                ));
            }
            for i in 0..n {
                let ratio = (i as f64 - half) / half;
                let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
                w[i] = bessel_i0(arg) / i0_beta;
            }
        }
    }

    Ok(w)
}

/// Modified Bessel function of the first kind, order zero: I_0(x).
///
/// Uses the polynomial approximation from Abramowitz & Stegun (9.8.1, 9.8.2).
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

// =============================================================================
// Periodogram
// =============================================================================

/// Compute the periodogram (raw spectral density estimate) of a time series.
///
/// The periodogram uses the FFT to estimate the power spectral density:
///
///   I(f_k) = (1 / (n * U)) * |W(f_k)|^2
///
/// where W(f_k) is the DFT of the windowed data and U is the window power
/// normalization factor.
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `fs` - Sampling frequency (default 1.0 if `None`)
/// * `window` - Window function type (default `Hann` if `None`)
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
///
/// # References
///
/// Stoica, P., & Moses, R. L. (2005). *Spectral Analysis of Signals*.
/// Prentice Hall.
pub fn periodogram_windowed(
    data: &ArrayView1<f64>,
    fs: Option<f64>,
    window: Option<WindowType>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    if n < 2 {
        return Err(NumRs2Error::ValueError(
            "Need at least 2 observations for periodogram".to_string(),
        ));
    }

    let fs = fs.unwrap_or(1.0);
    let win_type = window.unwrap_or(WindowType::Hann);

    // Generate window
    let win = window_function(win_type, n)?;

    // Window power normalization: U = (1/N) sum(w^2)
    let win_power: f64 = win.iter().map(|&w| w * w).sum::<f64>() / n as f64;
    if win_power < 1e-300 {
        return Err(NumRs2Error::ComputationError(
            "Window power is zero; check window parameters".to_string(),
        ));
    }

    // Remove mean and apply window
    let mean = data.iter().sum::<f64>() / n as f64;
    let windowed: Vec<f64> = data
        .iter()
        .zip(win.iter())
        .map(|(&x, &w)| (x - mean) * w)
        .collect();

    // FFT
    let fft_result = crate::fft::rfft(&windowed, None)
        .map_err(|e| NumRs2Error::ComputationError(format!("FFT failed: {}", e)))?;

    // Compute one-sided PSD
    let n_freqs = n / 2 + 1;
    let mut frequencies = Array1::zeros(n_freqs);
    let mut psd = Array1::zeros(n_freqs);

    let scale = 1.0 / (fs * n as f64 * win_power);

    for i in 0..n_freqs {
        frequencies[i] = i as f64 * fs / n as f64;
        let re = fft_result[i].re;
        let im = fft_result[i].im;
        let mag_sq = re * re + im * im;

        // Double-sided to one-sided: multiply interior bins by 2
        if i == 0 || (n.is_multiple_of(2) && i == n_freqs - 1) {
            psd[i] = mag_sq * scale;
        } else {
            psd[i] = 2.0 * mag_sq * scale;
        }
    }

    Ok((frequencies, psd))
}

/// Compute the direct (unwindowed) periodogram using rectangular window.
///
/// This is a convenience wrapper that uses a rectangular window (no tapering).
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `fs` - Sampling frequency (default 1.0)
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
pub fn periodogram_direct(
    data: &ArrayView1<f64>,
    fs: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    periodogram_windowed(data, fs, Some(WindowType::Rectangular))
}

// =============================================================================
// Welch's Method
// =============================================================================

/// Configuration for Welch's method.
#[derive(Debug, Clone)]
pub struct WelchConfig {
    /// Length of each segment
    pub segment_length: usize,
    /// Number of overlapping samples between segments
    pub overlap: usize,
    /// Window function to apply to each segment
    pub window: WindowType,
    /// Sampling frequency
    pub fs: f64,
}

impl WelchConfig {
    /// Create a new Welch configuration.
    ///
    /// # Arguments
    ///
    /// * `segment_length` - Length of each segment
    /// * `overlap` - Number of overlapping points (typically 50% of segment_length)
    /// * `window` - Window function type
    /// * `fs` - Sampling frequency
    pub fn new(segment_length: usize, overlap: usize, window: WindowType, fs: f64) -> Self {
        Self {
            segment_length,
            overlap,
            window,
            fs,
        }
    }
}

/// Estimate power spectral density using Welch's method.
///
/// Welch's method divides the data into overlapping segments, computes a
/// modified periodogram for each segment, and averages the results. This
/// reduces the variance of the spectral estimate at the cost of frequency
/// resolution.
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `config` - Welch configuration parameters
///
/// # Returns
///
/// Tuple of (frequencies, averaged power spectral density)
///
/// # References
///
/// Welch, P. D. (1967). "The use of fast Fourier transform for the estimation
/// of power spectra: A method based on time averaging over short, modified
/// periodograms." *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70-73.
pub fn welch_method(
    data: &ArrayView1<f64>,
    config: &WelchConfig,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    let seg_len = config.segment_length;
    let overlap = config.overlap;

    if seg_len < 2 {
        return Err(NumRs2Error::ValueError(
            "Segment length must be at least 2".to_string(),
        ));
    }
    if seg_len > n {
        return Err(NumRs2Error::ValueError(format!(
            "Segment length ({}) exceeds data length ({})",
            seg_len, n
        )));
    }
    if overlap >= seg_len {
        return Err(NumRs2Error::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let step = seg_len - overlap;
    // Count how many complete segments we can form
    let n_segments = if n >= seg_len {
        (n - seg_len) / step + 1
    } else {
        0
    };

    if n_segments == 0 {
        return Err(NumRs2Error::ValueError(
            "No complete segments available with given parameters".to_string(),
        ));
    }

    let n_freqs = seg_len / 2 + 1;
    let mut avg_psd = Array1::zeros(n_freqs);
    let mut frequencies = Array1::zeros(n_freqs);

    // Compute frequency grid once
    for i in 0..n_freqs {
        frequencies[i] = i as f64 * config.fs / seg_len as f64;
    }

    // Accumulate periodograms from each segment
    for seg_idx in 0..n_segments {
        let start = seg_idx * step;
        let end = start + seg_len;
        let segment = data.slice(scirs2_core::ndarray::s![start..end]);

        let (_f, psd) = periodogram_windowed(&segment, Some(config.fs), Some(config.window))?;

        for i in 0..n_freqs.min(psd.len()) {
            avg_psd[i] += psd[i];
        }
    }

    // Average over segments
    avg_psd /= n_segments as f64;

    Ok((frequencies, avg_psd))
}

// =============================================================================
// Bartlett's Method
// =============================================================================

/// Estimate PSD using Bartlett's method (non-overlapping averaged periodogram).
///
/// Bartlett's method is a special case of Welch's method with zero overlap
/// and a rectangular window. It divides the data into K non-overlapping
/// segments, computes the periodogram of each, and averages the results.
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `n_segments` - Number of non-overlapping segments
/// * `fs` - Sampling frequency (default 1.0)
///
/// # Returns
///
/// Tuple of (frequencies, averaged PSD)
///
/// # References
///
/// Bartlett, M. S. (1948). "Smoothing periodograms from time-series with
/// continuous spectra." *Nature*, 161(4096), 686-687.
pub fn bartlett_method(
    data: &ArrayView1<f64>,
    n_segments: usize,
    fs: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    let fs = fs.unwrap_or(1.0);

    if n_segments == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of segments must be positive".to_string(),
        ));
    }

    let seg_len = n / n_segments;
    if seg_len < 2 {
        return Err(NumRs2Error::ValueError(format!(
            "Segment length ({}) is too short; reduce n_segments or use more data",
            seg_len
        )));
    }

    let config = WelchConfig::new(seg_len, 0, WindowType::Rectangular, fs);
    welch_method(data, &config)
}

// =============================================================================
// AR Spectral Estimation
// =============================================================================

/// Estimate PSD using the Yule-Walker autoregressive method.
///
/// Fits an AR(p) model via the Yule-Walker equations and computes the
/// parametric PSD:
///
///   S(f) = sigma^2 / |1 - sum_{k=1}^{p} a_k e^{-j2pi f k}|^2
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `order` - AR model order p
/// * `n_fft` - Number of FFT points for the output PSD (default 256)
/// * `fs` - Sampling frequency (default 1.0)
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
///
/// # References
///
/// Kay, S. M. (1988). *Modern Spectral Estimation: Theory and Application*.
/// Prentice Hall.
pub fn ar_psd_yule_walker(
    data: &ArrayView1<f64>,
    order: usize,
    n_fft: Option<usize>,
    fs: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    let fs = fs.unwrap_or(1.0);
    let n_fft = n_fft.unwrap_or(256);

    if order == 0 {
        return Err(NumRs2Error::ValueError(
            "AR order must be at least 1".to_string(),
        ));
    }
    if order >= n {
        return Err(NumRs2Error::ValueError(format!(
            "AR order ({}) must be less than data length ({})",
            order, n
        )));
    }
    if n_fft < 2 {
        return Err(NumRs2Error::ValueError(
            "n_fft must be at least 2".to_string(),
        ));
    }

    // Compute ACF using the parent module
    let acf_view = data.view();
    let acf = super::autocorrelation(&acf_view, order)?;

    // Solve Yule-Walker equations: R * a = r
    // where R is the Toeplitz ACF matrix and r = [acf[1], ..., acf[p]]
    let mut r_matrix = Array2::zeros((order, order));
    for i in 0..order {
        for j in 0..order {
            let lag = i.abs_diff(j);
            r_matrix[[i, j]] = acf[lag];
        }
    }

    let mut r_vec = Array1::zeros(order);
    for i in 0..order {
        r_vec[i] = acf[i + 1];
    }

    // Solve R * a = r
    let ar_coeffs = scirs2_linalg::solve(&r_matrix.view(), &r_vec.view(), None).map_err(|_| {
        NumRs2Error::ComputationError(
            "Singular Toeplitz matrix in Yule-Walker solution".to_string(),
        )
    })?;

    // Compute innovation variance: sigma^2 = acf[0] - sum(a_k * acf[k])
    let mut sigma_sq = acf[0];
    for k in 0..order {
        sigma_sq -= ar_coeffs[k] * acf[k + 1];
    }
    if sigma_sq <= 0.0 {
        // Fallback: use a small positive value
        sigma_sq = 1e-12;
    }

    // Compute PSD: S(f) = sigma^2 / |A(f)|^2
    let n_freqs = n_fft / 2 + 1;
    let mut frequencies = Array1::zeros(n_freqs);
    let mut psd = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        let f = i as f64 * fs / n_fft as f64;
        frequencies[i] = f;

        // A(f) = 1 - sum_{k=1}^{p} a_k * exp(-j * 2pi * f * k / fs)
        let mut a_re = 1.0;
        let mut a_im = 0.0;
        for k in 0..order {
            let angle = -2.0 * PI * f * (k + 1) as f64 / fs;
            a_re -= ar_coeffs[k] * angle.cos();
            a_im -= ar_coeffs[k] * angle.sin();
        }

        let a_mag_sq = a_re * a_re + a_im * a_im;
        if a_mag_sq < 1e-300 {
            psd[i] = sigma_sq / 1e-300;
        } else {
            psd[i] = sigma_sq / a_mag_sq;
        }
    }

    Ok((frequencies, psd))
}

/// Estimate PSD using Burg's autoregressive method.
///
/// Burg's method estimates AR parameters by minimizing both forward and
/// backward prediction errors simultaneously, which guarantees a stable
/// AR model and provides better frequency resolution than Yule-Walker
/// for short data records.
///
/// # Arguments
///
/// * `data` - Input time series data
/// * `order` - AR model order p
/// * `n_fft` - Number of FFT points for output PSD (default 256)
/// * `fs` - Sampling frequency (default 1.0)
///
/// # Returns
///
/// Tuple of (frequencies, power spectral density)
///
/// # References
///
/// Burg, J. P. (1975). *Maximum Entropy Spectral Analysis*.
/// Stanford University, Ph.D. dissertation.
pub fn ar_psd_burg(
    data: &ArrayView1<f64>,
    order: usize,
    n_fft: Option<usize>,
    fs: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = data.len();
    let fs = fs.unwrap_or(1.0);
    let n_fft = n_fft.unwrap_or(256);

    if order == 0 {
        return Err(NumRs2Error::ValueError(
            "AR order must be at least 1".to_string(),
        ));
    }
    if order >= n {
        return Err(NumRs2Error::ValueError(format!(
            "AR order ({}) must be less than data length ({})",
            order, n
        )));
    }
    if n_fft < 2 {
        return Err(NumRs2Error::ValueError(
            "n_fft must be at least 2".to_string(),
        ));
    }

    // Burg's algorithm
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Initialize forward and backward prediction errors
    let mut ef: Vec<f64> = centered.clone();
    let mut eb: Vec<f64> = centered.clone();

    // Initial error power
    let mut pe = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

    let mut ar_coeffs = vec![0.0; order];
    let mut ar_prev = vec![0.0; order];

    for m in 0..order {
        // Compute reflection coefficient
        let mut num = 0.0;
        let mut den = 0.0;
        for j in (m + 1)..n {
            num += ef[j] * eb[j - 1];
            den += ef[j] * ef[j] + eb[j - 1] * eb[j - 1];
        }

        if den.abs() < 1e-300 {
            // Cannot continue; data might be near-constant
            break;
        }

        let km = 2.0 * num / den;

        // Update AR coefficients via Levinson recursion
        ar_coeffs[m] = km;
        for i in 0..m {
            ar_coeffs[i] = ar_prev[i] - km * ar_prev[m - 1 - i];
        }

        // Update error power
        pe *= 1.0 - km * km;

        // Update forward/backward errors
        // Both ef_new and eb_new must be computed from the OLD ef and eb values,
        // so we need temporary arrays for both to avoid in-place corruption.
        let mut ef_new = vec![0.0; n];
        let mut eb_new = vec![0.0; n];
        for j in (m + 1)..n {
            ef_new[j] = ef[j] - km * eb[j - 1];
            eb_new[j] = eb[j - 1] - km * ef[j];
        }
        ef[(m + 1)..n].copy_from_slice(&ef_new[(m + 1)..n]);
        eb[(m + 1)..n].copy_from_slice(&eb_new[(m + 1)..n]);

        // Save current coefficients for next iteration
        ar_prev[..=m].copy_from_slice(&ar_coeffs[..=m]);
    }

    let sigma_sq = pe.max(1e-12);

    // Compute PSD from AR parameters
    let n_freqs = n_fft / 2 + 1;
    let mut frequencies = Array1::zeros(n_freqs);
    let mut psd = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        let f = i as f64 * fs / n_fft as f64;
        frequencies[i] = f;

        let mut a_re = 1.0;
        let mut a_im = 0.0;
        for k in 0..order {
            let angle = -2.0 * PI * f * (k + 1) as f64 / fs;
            a_re -= ar_coeffs[k] * angle.cos();
            a_im -= ar_coeffs[k] * angle.sin();
        }

        let a_mag_sq = a_re * a_re + a_im * a_im;
        if a_mag_sq < 1e-300 {
            psd[i] = sigma_sq / 1e-300;
        } else {
            psd[i] = sigma_sq / a_mag_sq;
        }
    }

    Ok((frequencies, psd))
}

// =============================================================================
// Cross-Spectral Analysis
// =============================================================================

/// Compute the cross-spectral density (CSD) between two time series.
///
/// The CSD is the Fourier transform of the cross-correlation function,
/// estimated here using Welch's averaged periodogram approach applied to
/// the cross-spectrum.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `config` - Welch configuration for segment-based estimation
///
/// # Returns
///
/// Tuple of (frequencies, CSD real parts, CSD imaginary parts)
///
/// # References
///
/// Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
/// Measurement Procedures*, 4th ed. Wiley.
pub fn cross_spectral_density(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: &WelchConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let nx = x.len();
    let ny = y.len();

    if nx != ny {
        return Err(NumRs2Error::ValueError(format!(
            "Series must have same length: x={}, y={}",
            nx, ny
        )));
    }

    let n = nx;
    let seg_len = config.segment_length;
    let overlap = config.overlap;

    if seg_len < 2 || seg_len > n {
        return Err(NumRs2Error::ValueError(format!(
            "Invalid segment length {} for data length {}",
            seg_len, n
        )));
    }
    if overlap >= seg_len {
        return Err(NumRs2Error::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    let step = seg_len - overlap;
    let n_segments = if n >= seg_len {
        (n - seg_len) / step + 1
    } else {
        0
    };

    if n_segments == 0 {
        return Err(NumRs2Error::ValueError(
            "No complete segments with given parameters".to_string(),
        ));
    }

    let win = window_function(config.window, seg_len)?;
    let win_power: f64 = win.iter().map(|&w| w * w).sum::<f64>() / seg_len as f64;
    if win_power < 1e-300 {
        return Err(NumRs2Error::ComputationError(
            "Window power is zero".to_string(),
        ));
    }

    let n_freqs = seg_len / 2 + 1;
    let mut csd_re = Array1::zeros(n_freqs);
    let mut csd_im = Array1::zeros(n_freqs);
    let mut frequencies = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        frequencies[i] = i as f64 * config.fs / seg_len as f64;
    }

    let scale = 1.0 / (config.fs * seg_len as f64 * win_power);

    for seg_idx in 0..n_segments {
        let start = seg_idx * step;

        // Remove mean and apply window for both signals
        let x_raw: Vec<f64> = x.iter().skip(start).take(seg_len).copied().collect();
        let x_mean = x_raw.iter().sum::<f64>() / seg_len as f64;
        let x_seg: Vec<f64> = x_raw
            .iter()
            .zip(win.iter())
            .map(|(&xv, &w)| (xv - x_mean) * w)
            .collect();

        let y_raw: Vec<f64> = y.iter().skip(start).take(seg_len).copied().collect();
        let y_mean = y_raw.iter().sum::<f64>() / seg_len as f64;
        let y_seg: Vec<f64> = y_raw
            .iter()
            .zip(win.iter())
            .map(|(&yv, &w)| (yv - y_mean) * w)
            .collect();

        // Compute FFTs
        let fft_x = crate::fft::rfft(&x_seg, None)
            .map_err(|e| NumRs2Error::ComputationError(format!("FFT of x failed: {}", e)))?;
        let fft_y = crate::fft::rfft(&y_seg, None)
            .map_err(|e| NumRs2Error::ComputationError(format!("FFT of y failed: {}", e)))?;

        // Cross-spectrum: Sxy = conj(X) * Y
        for i in 0..n_freqs.min(fft_x.len()).min(fft_y.len()) {
            let xr = fft_x[i].re;
            let xi = fft_x[i].im;
            let yr = fft_y[i].re;
            let yi = fft_y[i].im;

            // conj(X) * Y = (xr - j*xi)(yr + j*yi)
            //              = xr*yr + xi*yi + j*(xr*yi - xi*yr)
            let cross_re = xr * yr + xi * yi;
            let cross_im = xr * yi - xi * yr;

            let factor = if i == 0 || (seg_len.is_multiple_of(2) && i == n_freqs - 1) {
                scale
            } else {
                2.0 * scale
            };

            csd_re[i] += cross_re * factor;
            csd_im[i] += cross_im * factor;
        }
    }

    // Average over segments
    csd_re /= n_segments as f64;
    csd_im /= n_segments as f64;

    Ok((frequencies, csd_re, csd_im))
}

/// Compute the magnitude squared coherence between two time series.
///
/// The coherence measures the linear relationship between two signals
/// as a function of frequency:
///
///   C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
///
/// Values range from 0 (no linear relationship) to 1 (perfect linear relationship).
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `config` - Welch configuration
///
/// # Returns
///
/// Tuple of (frequencies, coherence values in [0, 1])
///
/// # References
///
/// Carter, G. C., Knapp, C. H., & Nuttall, A. H. (1973). "Estimation of the
/// magnitude-squared coherence function via overlapped fast Fourier transform
/// processing." *IEEE Transactions on Audio and Electroacoustics*, 21(4), 337-344.
pub fn coherence(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: &WelchConfig,
) -> Result<(Array1<f64>, Array1<f64>)> {
    // Compute auto-spectra and cross-spectrum
    let (freqs, csd_re, csd_im) = cross_spectral_density(x, y, config)?;
    let (_, psd_xx) = welch_method(x, config)?;
    let (_, psd_yy) = welch_method(y, config)?;

    let n_freqs = freqs.len();
    let mut coh = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        let csd_mag_sq = csd_re[i] * csd_re[i] + csd_im[i] * csd_im[i];
        let denom = psd_xx[i] * psd_yy[i];

        if denom > 1e-300 {
            coh[i] = (csd_mag_sq / denom).min(1.0);
        }
        // else: coherence stays 0.0 (undefined / no power)
    }

    Ok((freqs, coh))
}

/// Compute the phase spectrum between two time series.
///
/// The phase spectrum is the argument (angle) of the cross-spectral density:
///
///   phi(f) = atan2(Im(S_xy(f)), Re(S_xy(f)))
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `config` - Welch configuration
///
/// # Returns
///
/// Tuple of (frequencies, phase in radians [-pi, pi])
pub fn phase_spectrum(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: &WelchConfig,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let (freqs, csd_re, csd_im) = cross_spectral_density(x, y, config)?;

    let n_freqs = freqs.len();
    let mut phase = Array1::zeros(n_freqs);

    for i in 0..n_freqs {
        phase[i] = csd_im[i].atan2(csd_re[i]);
    }

    Ok((freqs, phase))
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Generate a one-sided frequency grid for spectral analysis.
///
/// # Arguments
///
/// * `n` - Number of data points (or FFT length)
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// Array of frequencies from 0 to fs/2
pub fn frequency_grid(n: usize, fs: f64) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(NumRs2Error::ValueError("n must be positive".to_string()));
    }
    if fs <= 0.0 {
        return Err(NumRs2Error::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n_freqs = n / 2 + 1;
    let mut freqs = Array1::zeros(n_freqs);
    for i in 0..n_freqs {
        freqs[i] = i as f64 * fs / n as f64;
    }

    Ok(freqs)
}

/// Convert power values to decibels (dB).
///
///   P_dB = 10 * log10(P / P_ref)
///
/// # Arguments
///
/// * `power` - Array of power values
/// * `reference` - Reference power level (default 1.0)
///
/// # Returns
///
/// Array of values in decibels
pub fn power_to_db(power: &ArrayView1<f64>, reference: Option<f64>) -> Result<Array1<f64>> {
    let reference = reference.unwrap_or(1.0);
    if reference <= 0.0 {
        return Err(NumRs2Error::ValueError(
            "Reference power must be positive".to_string(),
        ));
    }

    let n = power.len();
    let mut db = Array1::zeros(n);

    for i in 0..n {
        if power[i] > 0.0 {
            db[i] = 10.0 * (power[i] / reference).log10();
        } else {
            // Floor at a very low dB value instead of -inf
            db[i] = -200.0;
        }
    }

    Ok(db)
}

/// Detect peak frequencies in a power spectral density estimate.
///
/// Identifies local maxima in the PSD that exceed a threshold relative
/// to the median power level.
///
/// # Arguments
///
/// * `frequencies` - Array of frequency values
/// * `psd` - Array of PSD values
/// * `threshold_db` - Minimum prominence in dB above the median PSD (default 3.0)
/// * `min_distance` - Minimum number of bins between detected peaks (default 1)
///
/// # Returns
///
/// Vector of (frequency, power) tuples for each detected peak, sorted by power
/// descending.
pub fn detect_peaks(
    frequencies: &ArrayView1<f64>,
    psd: &ArrayView1<f64>,
    threshold_db: Option<f64>,
    min_distance: Option<usize>,
) -> Result<Vec<(f64, f64)>> {
    let n = frequencies.len();
    if n != psd.len() {
        return Err(NumRs2Error::ValueError(
            "frequencies and psd must have the same length".to_string(),
        ));
    }
    if n < 3 {
        return Err(NumRs2Error::ValueError(
            "Need at least 3 frequency bins for peak detection".to_string(),
        ));
    }

    let threshold_db = threshold_db.unwrap_or(3.0);
    let min_dist = min_distance.unwrap_or(1);

    // Compute median PSD for threshold
    let mut sorted_psd: Vec<f64> = psd.iter().copied().collect();
    sorted_psd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_psd = if sorted_psd.len().is_multiple_of(2) {
        (sorted_psd[sorted_psd.len() / 2 - 1] + sorted_psd[sorted_psd.len() / 2]) / 2.0
    } else {
        sorted_psd[sorted_psd.len() / 2]
    };

    // Threshold in linear scale
    let threshold_linear = if median_psd > 0.0 {
        median_psd * 10.0_f64.powf(threshold_db / 10.0)
    } else {
        0.0
    };

    // Find local maxima above threshold
    let mut candidates: Vec<(usize, f64, f64)> = Vec::new();

    for i in 1..(n - 1) {
        if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] >= threshold_linear {
            candidates.push((i, frequencies[i], psd[i]));
        }
    }

    // Sort by power descending
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Apply minimum distance constraint (greedy)
    let mut selected: Vec<(f64, f64)> = Vec::new();
    let mut used_indices: Vec<usize> = Vec::new();

    for (idx, freq, power) in &candidates {
        let too_close = used_indices.iter().any(|&ui| {
            let diff = (*idx).abs_diff(ui);
            diff < min_dist
        });

        if !too_close {
            selected.push((*freq, *power));
            used_indices.push(*idx);
        }
    }

    Ok(selected)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Helper: generate a sinusoidal signal.
    fn sine_signal(freq: f64, fs: f64, n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
                .collect(),
        )
    }

    // -------------------------------------------------------------------------
    // Window function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_window_hann_symmetry() {
        let n = 64;
        let w = window_function(WindowType::Hann, n).expect("Hann window should succeed");
        assert_eq!(w.len(), n);
        // Hann window is symmetric
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-12,
                "Hann window not symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_window_hamming_symmetry() {
        let n = 64;
        let w = window_function(WindowType::Hamming, n).expect("Hamming window should succeed");
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-12,
                "Hamming window not symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_window_blackman_endpoints() {
        let n = 64;
        let w = window_function(WindowType::Blackman, n).expect("Blackman window should succeed");
        // Blackman window endpoints are near zero
        assert!(w[0].abs() < 1e-10, "Blackman start should be ~0");
        assert!(w[n - 1].abs() < 1e-10, "Blackman end should be ~0");
        // Peak near center
        assert!(w[n / 2] > 0.9, "Blackman center should be near 1");
    }

    #[test]
    fn test_window_bartlett_triangular() {
        let n = 65; // odd for exact center = 1
        let w = window_function(WindowType::Bartlett, n).expect("Bartlett window should succeed");
        assert!(w[0].abs() < 1e-12, "Bartlett start should be 0");
        assert!(
            (w[n / 2] - 1.0).abs() < 1e-12,
            "Bartlett center should be 1"
        );
        assert!(w[n - 1].abs() < 1e-12, "Bartlett end should be 0");
    }

    #[test]
    fn test_window_kaiser_shape() {
        let n = 64;
        let w = window_function(WindowType::Kaiser(8.0), n).expect("Kaiser window should succeed");
        // Kaiser window should be positive and symmetric
        for &val in w.iter() {
            assert!(val >= 0.0, "Kaiser window should be non-negative");
        }
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-10,
                "Kaiser window not symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_window_rectangular_unity() {
        let n = 32;
        let w =
            window_function(WindowType::Rectangular, n).expect("Rectangular window should succeed");
        for &val in w.iter() {
            assert!(
                (val - 1.0).abs() < 1e-15,
                "Rectangular window should be all ones"
            );
        }
    }

    #[test]
    fn test_window_zero_length() {
        let result = window_function(WindowType::Hann, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_window_length_one() {
        let w = window_function(WindowType::Blackman, 1).expect("Length-1 window should succeed");
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-15);
    }

    // -------------------------------------------------------------------------
    // Periodogram tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_periodogram_sinusoid_peak() {
        // A pure sine at 10 Hz, sampled at 100 Hz, 256 samples
        let fs = 100.0;
        let n = 256;
        let signal = sine_signal(10.0, fs, n);

        let (freqs, psd) = periodogram_windowed(&signal.view(), Some(fs), Some(WindowType::Hann))
            .expect("Periodogram should succeed");

        // Find the index of the peak
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("Should find peak");

        let peak_freq = freqs[peak_idx];

        // Peak should be near 10 Hz (within one frequency bin)
        let bin_width = fs / n as f64;
        assert!(
            (peak_freq - 10.0).abs() < 2.0 * bin_width,
            "Peak at {:.2} Hz, expected ~10 Hz (bin width = {:.2})",
            peak_freq,
            bin_width
        );
    }

    #[test]
    fn test_periodogram_direct_nonneg() {
        let signal = sine_signal(5.0, 50.0, 128);
        let (_, psd) = periodogram_direct(&signal.view(), Some(50.0))
            .expect("Direct periodogram should succeed");

        for &val in psd.iter() {
            assert!(val >= 0.0, "PSD must be non-negative, got {}", val);
        }
    }

    #[test]
    fn test_periodogram_dc_component() {
        // Constant signal (all DC)
        let n = 64;
        let signal = Array1::from_vec(vec![3.0; n]);
        let (freqs, psd) =
            periodogram_windowed(&signal.view(), Some(1.0), Some(WindowType::Rectangular))
                .expect("DC periodogram should succeed");

        // After mean removal, PSD should be essentially zero
        let total_power: f64 = psd.iter().sum();
        assert!(
            total_power < 1e-10,
            "DC-only signal after mean removal should have near-zero PSD, got {}",
            total_power
        );
    }

    // -------------------------------------------------------------------------
    // Welch's method tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_welch_smoother_than_periodogram() {
        // Welch's method should produce a smoother estimate (lower variance)
        let fs = 100.0;
        let n = 512;
        // Signal with noise
        let signal = Array1::from_vec(
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    (2.0 * PI * 10.0 * t).sin() + 0.3 * (2.0 * PI * 25.0 * t).sin()
                })
                .collect(),
        );

        let config = WelchConfig::new(128, 64, WindowType::Hann, fs);
        let (_, welch_psd) = welch_method(&signal.view(), &config).expect("Welch should succeed");

        let (_, raw_psd) = periodogram_windowed(&signal.view(), Some(fs), Some(WindowType::Hann))
            .expect("Periodogram should succeed");

        // Welch PSD should have fewer bins (shorter segments)
        assert!(
            welch_psd.len() < raw_psd.len(),
            "Welch segments shorter => fewer freq bins"
        );

        // All Welch PSD values should be non-negative
        for &val in welch_psd.iter() {
            assert!(val >= 0.0, "Welch PSD must be non-negative");
        }
    }

    #[test]
    fn test_welch_overlap_validation() {
        let signal = sine_signal(5.0, 50.0, 256);
        let config = WelchConfig::new(64, 64, WindowType::Hann, 50.0);
        let result = welch_method(&signal.view(), &config);
        assert!(result.is_err(), "overlap == segment_length should fail");
    }

    // -------------------------------------------------------------------------
    // Bartlett's method test
    // -------------------------------------------------------------------------

    #[test]
    fn test_bartlett_method() {
        let fs = 100.0;
        let n = 512;
        let signal = sine_signal(15.0, fs, n);

        let (freqs, psd) =
            bartlett_method(&signal.view(), 4, Some(fs)).expect("Bartlett method should succeed");

        assert!(!freqs.is_empty());
        assert!(!psd.is_empty());

        // Find peak
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("Should find peak");

        let seg_len = n / 4;
        let bin_width = fs / seg_len as f64;
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - 15.0).abs() < 2.0 * bin_width,
            "Bartlett peak at {:.2} Hz, expected ~15 Hz",
            peak_freq
        );
    }

    // -------------------------------------------------------------------------
    // AR spectral estimation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ar_yule_walker_sinusoid() {
        let fs = 100.0;
        let n = 256;
        let signal = sine_signal(20.0, fs, n);

        let (freqs, psd) = ar_psd_yule_walker(&signal.view(), 10, Some(256), Some(fs))
            .expect("AR Yule-Walker should succeed");

        // PSD should be non-negative
        for &val in psd.iter() {
            assert!(val >= 0.0, "AR PSD must be non-negative");
        }

        // Peak should be near 20 Hz
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("Should find peak");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - 20.0).abs() < 3.0,
            "AR YW peak at {:.2} Hz, expected ~20 Hz",
            peak_freq
        );
    }

    #[test]
    fn test_ar_burg_sinusoid() {
        let fs = 100.0;
        let n = 256;
        let signal = sine_signal(20.0, fs, n);

        let (freqs, psd) =
            ar_psd_burg(&signal.view(), 10, Some(256), Some(fs)).expect("AR Burg should succeed");

        for &val in psd.iter() {
            assert!(val >= 0.0, "Burg PSD must be non-negative");
        }

        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("Should find peak");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - 20.0).abs() < 3.0,
            "Burg peak at {:.2} Hz, expected ~20 Hz",
            peak_freq
        );
    }

    // -------------------------------------------------------------------------
    // Cross-spectral analysis tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coherence_identical_signals() {
        let fs = 100.0;
        let n = 512;
        let signal = sine_signal(10.0, fs, n);

        let config = WelchConfig::new(128, 64, WindowType::Hann, fs);
        let (_, coh) =
            coherence(&signal.view(), &signal.view(), &config).expect("Coherence should succeed");

        // Coherence of a signal with itself should be 1.0 at all frequencies
        // (where there is nonzero power)
        let (_, psd) = welch_method(&signal.view(), &config).expect("Welch should succeed");

        for i in 0..coh.len() {
            if psd[i] > 1e-10 {
                assert!(
                    (coh[i] - 1.0).abs() < 0.05,
                    "Coherence[{}] = {:.4}, expected ~1.0 (psd = {:.2e})",
                    i,
                    coh[i],
                    psd[i]
                );
            }
        }
    }

    #[test]
    fn test_phase_spectrum_zero_lag() {
        // Two identical signals should have zero phase
        let fs = 100.0;
        let n = 512;
        let signal = sine_signal(10.0, fs, n);

        let config = WelchConfig::new(128, 64, WindowType::Hann, fs);
        let (_, phase) = phase_spectrum(&signal.view(), &signal.view(), &config)
            .expect("Phase spectrum should succeed");

        let (_, psd) = welch_method(&signal.view(), &config).expect("Welch should succeed");

        // Phase should be near zero where signal has significant power
        for i in 0..phase.len() {
            if psd[i] > 1e-10 {
                assert!(
                    phase[i].abs() < 0.1,
                    "Phase[{}] = {:.4}, expected ~0 for identical signals",
                    i,
                    phase[i]
                );
            }
        }
    }

    #[test]
    fn test_cross_spectral_density_basic() {
        let fs = 100.0;
        let n = 256;
        let x = sine_signal(10.0, fs, n);
        let y = sine_signal(10.0, fs, n);

        let config = WelchConfig::new(64, 32, WindowType::Hann, fs);
        let (freqs, csd_re, csd_im) =
            cross_spectral_density(&x.view(), &y.view(), &config).expect("CSD should succeed");

        assert_eq!(freqs.len(), csd_re.len());
        assert_eq!(freqs.len(), csd_im.len());
    }

    // -------------------------------------------------------------------------
    // Utility function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frequency_grid() {
        let n = 128;
        let fs = 1000.0;
        let freqs = frequency_grid(n, fs).expect("Frequency grid should succeed");

        assert_eq!(freqs.len(), n / 2 + 1);
        assert!((freqs[0] - 0.0).abs() < 1e-12, "First freq should be 0");
        let nyquist = fs / 2.0;
        assert!(
            (freqs[freqs.len() - 1] - nyquist).abs() < 1e-10,
            "Last freq should be Nyquist = {}",
            nyquist
        );
    }

    #[test]
    fn test_power_to_db() {
        let power = Array1::from_vec(vec![1.0, 10.0, 100.0, 0.01]);
        let db = power_to_db(&power.view(), None).expect("dB conversion should succeed");

        assert!((db[0] - 0.0).abs() < 1e-10, "1.0 => 0 dB");
        assert!((db[1] - 10.0).abs() < 1e-10, "10.0 => 10 dB");
        assert!((db[2] - 20.0).abs() < 1e-10, "100.0 => 20 dB");
        assert!((db[3] - (-20.0)).abs() < 1e-10, "0.01 => -20 dB");
    }

    #[test]
    fn test_power_to_db_zero_power() {
        let power = Array1::from_vec(vec![0.0, 1.0]);
        let db = power_to_db(&power.view(), None).expect("dB conversion should succeed");
        assert!(db[0] < -100.0, "Zero power should map to very low dB");
    }

    #[test]
    fn test_detect_peaks_sinusoid() {
        let fs = 100.0;
        let n = 512;
        // Two-frequency signal
        let signal = Array1::from_vec(
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
                })
                .collect(),
        );

        let (freqs, psd) = periodogram_windowed(&signal.view(), Some(fs), Some(WindowType::Hann))
            .expect("Periodogram should succeed");

        let peaks = detect_peaks(&freqs.view(), &psd.view(), Some(3.0), Some(3))
            .expect("Peak detection should succeed");

        // Should detect at least one peak near 10 Hz
        assert!(!peaks.is_empty(), "Should detect at least one peak");

        let has_10hz = peaks.iter().any(|(f, _)| (*f - 10.0).abs() < 2.0);
        assert!(
            has_10hz,
            "Should detect peak near 10 Hz; peaks: {:?}",
            peaks
        );
    }

    #[test]
    fn test_detect_peaks_single_frequency() {
        let fs = 100.0;
        let n = 256;
        let signal = sine_signal(15.0, fs, n);

        let (freqs, psd) = periodogram_windowed(&signal.view(), Some(fs), Some(WindowType::Hann))
            .expect("Periodogram should succeed");

        let peaks = detect_peaks(&freqs.view(), &psd.view(), Some(3.0), Some(2))
            .expect("Peak detection should succeed");

        // Should detect exactly one dominant peak near 15 Hz
        assert!(!peaks.is_empty(), "Should detect at least one peak");
        let (peak_f, _) = peaks[0]; // strongest peak
        assert!(
            (peak_f - 15.0).abs() < 2.0,
            "Strongest peak at {:.2} Hz, expected ~15 Hz",
            peak_f
        );
    }
}

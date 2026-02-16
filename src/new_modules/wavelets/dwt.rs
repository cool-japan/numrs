//! # Discrete Wavelet Transform (DWT)
//!
//! This module implements the fast Discrete Wavelet Transform (DWT) algorithm
//! and its inverse for 1D signals. The DWT decomposes a signal into approximation
//! and detail coefficients at multiple levels.
//!
//! ## Algorithm
//!
//! The DWT uses a pair of filters (low-pass and high-pass) to decompose a signal:
//!
//! ```text
//! Approximation: a[k] = Σ h[n-2k] x[n]  (low-pass filter, downsample by 2)
//! Detail:        d[k] = Σ g[n-2k] x[n]  (high-pass filter, downsample by 2)
//! ```
//!
//! Multi-level decomposition recursively applies DWT to approximation coefficients:
//!
//! ```text
//! Level 1: x -> (a₁, d₁)
//! Level 2: a₁ -> (a₂, d₂)
//! Level 3: a₂ -> (a₃, d₃)
//! ```
//!
//! ## Boundary Handling
//!
//! Different extension modes handle signal boundaries:
//! - **Periodic**: wrap-around (circular convolution)
//! - **Symmetric**: mirror extension (reflect at boundaries)
//! - **ZeroPad**: pad with zeros
//! - **Smooth**: replicate edge values

use super::{ExtensionMode, Wavelet, WaveletError, WaveletResult};

/// Perform single-level 1D Discrete Wavelet Transform
///
/// Decomposes a signal into approximation and detail coefficients.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use for decomposition
/// * `mode` - Boundary extension mode
///
/// # Returns
///
/// A tuple `(approximation, detail)` where each has length `ceil(n/2)`
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{dwt_1d, WaveletType, ExtensionMode};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Haar.create()?;
/// let (approx, detail) = dwt_1d(&signal, &wavelet, ExtensionMode::Periodic)?;
/// ```
pub fn dwt_1d(
    signal: &[f64],
    wavelet: &dyn Wavelet,
    mode: ExtensionMode,
) -> WaveletResult<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    if n == 0 {
        return Err(WaveletError::InvalidLength(
            "Signal must not be empty".to_string(),
        ));
    }

    let filter_len = wavelet.filter_len();
    if n < filter_len {
        return Err(WaveletError::InvalidLength(format!(
            "Signal length {} is less than filter length {}",
            n, filter_len
        )));
    }

    let dec_lo = wavelet.dec_lo();
    let dec_hi = wavelet.dec_hi();

    // Output length after downsampling
    let output_len = n.div_ceil(2);

    let mut approx = vec![0.0; output_len];
    let mut detail = vec![0.0; output_len];

    match mode {
        ExtensionMode::Periodic => {
            // Circular convolution for periodic mode guarantees perfect reconstruction
            for k in 0..output_len {
                let mut sum_lo = 0.0;
                let mut sum_hi = 0.0;
                for (i, (&lo, &hi)) in dec_lo.iter().zip(dec_hi.iter()).enumerate() {
                    let idx = (2 * k + i) % n;
                    sum_lo += signal[idx] * lo;
                    sum_hi += signal[idx] * hi;
                }
                approx[k] = sum_lo;
                detail[k] = sum_hi;
            }
        }
        _ => {
            // For non-periodic modes, use signal extension
            let extended = extend_signal(signal, filter_len, mode);
            for k in 0..output_len {
                let pos = 2 * k;
                let mut sum_lo = 0.0;
                let mut sum_hi = 0.0;
                for (i, (&lo, &hi)) in dec_lo.iter().zip(dec_hi.iter()).enumerate() {
                    let idx = pos + i;
                    if idx < extended.len() {
                        sum_lo += extended[idx] * lo;
                        sum_hi += extended[idx] * hi;
                    }
                }
                approx[k] = sum_lo;
                detail[k] = sum_hi;
            }
        }
    }

    Ok((approx, detail))
}

/// Perform single-level 1D Inverse Discrete Wavelet Transform
///
/// Reconstructs a signal from approximation and detail coefficients.
///
/// # Arguments
///
/// * `approx` - Approximation coefficients
/// * `detail` - Detail coefficients
/// * `wavelet` - Wavelet used for decomposition
/// * `output_len` - Expected output length (original signal length)
///
/// # Returns
///
/// Reconstructed signal of length `output_len`
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{dwt_1d, idwt_1d, WaveletType, ExtensionMode};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Haar.create()?;
/// let (approx, detail) = dwt_1d(&signal, &wavelet, ExtensionMode::Periodic)?;
/// let reconstructed = idwt_1d(&approx, &detail, &wavelet, signal.len())?;
/// ```
pub fn idwt_1d(
    approx: &[f64],
    detail: &[f64],
    wavelet: &dyn Wavelet,
    output_len: usize,
) -> WaveletResult<Vec<f64>> {
    // Default to Periodic mode for backward compatibility
    idwt_1d_mode(approx, detail, wavelet, output_len, ExtensionMode::Periodic)
}

/// Perform single-level 1D Inverse Discrete Wavelet Transform with specified mode
///
/// Reconstructs a signal from approximation and detail coefficients using the
/// same extension mode that was used for decomposition.
///
/// # Arguments
///
/// * `approx` - Approximation coefficients
/// * `detail` - Detail coefficients
/// * `wavelet` - Wavelet used for decomposition
/// * `output_len` - Expected output length (original signal length)
/// * `mode` - Boundary extension mode (must match the mode used in dwt_1d)
pub fn idwt_1d_mode(
    approx: &[f64],
    detail: &[f64],
    wavelet: &dyn Wavelet,
    output_len: usize,
    mode: ExtensionMode,
) -> WaveletResult<Vec<f64>> {
    if approx.len() != detail.len() {
        return Err(WaveletError::FilterMismatch(format!(
            "Approximation and detail must have same length: {} vs {}",
            approx.len(),
            detail.len()
        )));
    }

    if approx.is_empty() {
        return Err(WaveletError::InvalidLength(
            "Coefficients must not be empty".to_string(),
        ));
    }

    let coeff_len = approx.len();
    let filter_len = wavelet.filter_len();

    // For orthogonal wavelets, the adjoint of the forward transform
    // (convolve + downsample) is (upsample + accumulate with same filters).
    // This is equivalent to using dec_lo/dec_hi for reconstruction when the
    // forward transform used dec_lo/dec_hi for analysis.
    let dec_lo = wavelet.dec_lo();
    let dec_hi = wavelet.dec_hi();

    match mode {
        ExtensionMode::Periodic => {
            // Circular accumulation: exact adjoint of circular convolution
            let mut signal = vec![0.0; output_len];
            for (k, (&a, &d)) in approx.iter().zip(detail.iter()).enumerate() {
                for (i, (&lo, &hi)) in dec_lo.iter().zip(dec_hi.iter()).enumerate() {
                    let idx = (2 * k + i) % output_len;
                    signal[idx] += a * lo + d * hi;
                }
            }
            Ok(signal)
        }
        _ => {
            // For non-periodic modes: upsample and accumulate with same filters
            // then extract the valid portion
            let extended_len = 2 * coeff_len + filter_len - 1;
            let mut recon_ext = vec![0.0; extended_len];

            for (k, (&a, &d)) in approx.iter().zip(detail.iter()).enumerate() {
                let pos = 2 * k;
                for (i, (&lo, &hi)) in dec_lo.iter().zip(dec_hi.iter()).enumerate() {
                    if pos + i < extended_len {
                        recon_ext[pos + i] += a * lo + d * hi;
                    }
                }
            }

            // Extract the signal portion: the valid region starts after
            // the filter delay and matches the original signal length
            let mut signal = Vec::with_capacity(output_len);
            for i in 0..output_len {
                if i < recon_ext.len() {
                    signal.push(recon_ext[i]);
                } else {
                    signal.push(0.0);
                }
            }
            Ok(signal)
        }
    }
}

/// Multi-level wavelet decomposition
///
/// Recursively decomposes a signal into multiple levels of approximation
/// and detail coefficients.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `level` - Number of decomposition levels
/// * `mode` - Boundary extension mode
///
/// # Returns
///
/// A tuple `(approx, details)` where:
/// - `approx` is the final approximation coefficients at level `level`
/// - `details` is a vector containing detail coefficients from level 1 to `level`
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{wavedec, WaveletType, ExtensionMode};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Daubechies(4).create()?;
/// let (approx, details) = wavedec(&signal, &wavelet, 3, ExtensionMode::Symmetric)?;
/// // details[0] = level 1, details[1] = level 2, details[2] = level 3
/// ```
pub fn wavedec(
    signal: &[f64],
    wavelet: &dyn Wavelet,
    level: usize,
    mode: ExtensionMode,
) -> WaveletResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if level == 0 {
        return Err(WaveletError::InvalidLevel(
            "Decomposition level must be at least 1".to_string(),
        ));
    }

    let max_level = max_decomposition_level(signal.len(), wavelet.filter_len());
    if level > max_level {
        return Err(WaveletError::InvalidLevel(format!(
            "Requested level {} exceeds maximum {} for signal length {} and filter length {}",
            level,
            max_level,
            signal.len(),
            wavelet.filter_len()
        )));
    }

    let mut approx = signal.to_vec();
    let mut details = Vec::with_capacity(level);

    for _ in 0..level {
        let (new_approx, detail) = dwt_1d(&approx, wavelet, mode)?;
        details.push(detail);
        approx = new_approx;
    }

    Ok((approx, details))
}

/// Multi-level wavelet reconstruction
///
/// Reconstructs a signal from multi-level decomposition coefficients.
///
/// # Arguments
///
/// * `approx` - Final approximation coefficients
/// * `details` - Vector of detail coefficients (from finest to coarsest level)
/// * `wavelet` - Wavelet used for decomposition
/// * `original_len` - Original signal length
///
/// # Returns
///
/// Reconstructed signal
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{wavedec, waverec, WaveletType, ExtensionMode};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Daubechies(4).create()?;
/// let (approx, details) = wavedec(&signal, &wavelet, 3, ExtensionMode::Symmetric)?;
/// let reconstructed = waverec(&approx, &details, &wavelet, signal.len())?;
/// ```
pub fn waverec(
    approx: &[f64],
    details: &[Vec<f64>],
    wavelet: &dyn Wavelet,
    original_len: usize,
) -> WaveletResult<Vec<f64>> {
    waverec_mode(
        approx,
        details,
        wavelet,
        original_len,
        ExtensionMode::Periodic,
    )
}

/// Multi-level wavelet reconstruction with specified extension mode
///
/// Reconstructs a signal from multi-level decomposition coefficients using
/// the same extension mode that was used for decomposition.
///
/// # Arguments
///
/// * `approx` - Final approximation coefficients
/// * `details` - Vector of detail coefficients (from finest to coarsest level)
/// * `wavelet` - Wavelet used for decomposition
/// * `original_len` - Original signal length
/// * `mode` - Boundary extension mode (must match the mode used in wavedec)
pub fn waverec_mode(
    approx: &[f64],
    details: &[Vec<f64>],
    wavelet: &dyn Wavelet,
    original_len: usize,
    mode: ExtensionMode,
) -> WaveletResult<Vec<f64>> {
    if details.is_empty() {
        return Ok(approx.to_vec());
    }

    let mut signal = approx.to_vec();

    // Reconstruct from coarsest to finest level (reverse order)
    for detail in details.iter().rev() {
        let expected_len = signal.len() * 2;
        signal = idwt_1d_mode(&signal, detail, wavelet, expected_len, mode)?;
    }

    // Trim to original length if needed
    if signal.len() > original_len {
        signal.truncate(original_len);
    }

    Ok(signal)
}

/// Compute maximum decomposition level for a signal
///
/// The maximum level is determined by how many times the signal can be
/// halved while remaining at least as long as the filter:
/// ```text
/// max_level = floor(log₂(n / filter_len)) + 1
/// ```
/// This ensures the signal length at each level is >= filter_len.
pub fn max_decomposition_level(signal_len: usize, filter_len: usize) -> usize {
    if signal_len < filter_len || filter_len == 0 {
        return 0;
    }
    // At level j, the signal has length ceil(signal_len / 2^j)
    // We need ceil(signal_len / 2^j) >= filter_len
    let mut level = 0;
    let mut current_len = signal_len;
    while current_len >= filter_len {
        level += 1;
        current_len = current_len.div_ceil(2);
    }
    level
}

/// Extend signal according to the specified mode
fn extend_signal(signal: &[f64], filter_len: usize, mode: ExtensionMode) -> Vec<f64> {
    let n = signal.len();
    let extension = filter_len.saturating_sub(1);

    match mode {
        ExtensionMode::Periodic => {
            // Periodic extension (wrap-around)
            let extended_len = n + extension;
            let mut extended = Vec::with_capacity(extended_len);
            extended.extend_from_slice(signal);
            for i in 0..extension {
                extended.push(signal[i % n]);
            }
            extended
        }
        ExtensionMode::Symmetric => {
            // Symmetric extension (mirror at boundaries)
            let mut extended = Vec::with_capacity(n + 2 * extension);

            // Left extension
            for i in (0..extension.min(n)).rev() {
                extended.push(signal[i]);
            }

            // Original signal
            extended.extend_from_slice(signal);

            // Right extension
            for i in 0..extension.min(n) {
                let idx = n - 1 - i;
                extended.push(signal[idx]);
            }

            extended
        }
        ExtensionMode::ZeroPad => {
            // Zero-padding extension
            let mut extended = Vec::with_capacity(n + extension);
            extended.extend_from_slice(signal);
            extended.resize(n + extension, 0.0);
            extended
        }
        ExtensionMode::Smooth => {
            // Smooth extension (replicate edge values)
            let mut extended = Vec::with_capacity(n + extension);
            extended.extend_from_slice(signal);
            if let Some(&last) = signal.last() {
                extended.resize(n + extension, last);
            }
            extended
        }
    }
}

/// Downsample signal by factor of 2 (keep even indices)
#[allow(dead_code)]
fn downsample_2(signal: &[f64]) -> Vec<f64> {
    signal.iter().step_by(2).copied().collect()
}

/// Upsample signal by factor of 2 (insert zeros between samples)
#[allow(dead_code)]
fn upsample_2(signal: &[f64]) -> Vec<f64> {
    let mut upsampled = Vec::with_capacity(2 * signal.len());
    for &val in signal {
        upsampled.push(val);
        upsampled.push(0.0);
    }
    upsampled
}

/// Convolve signal with filter
#[allow(dead_code)]
fn convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = filter.len();
    let output_len = n + m - 1;
    let mut result = vec![0.0; output_len];

    for i in 0..n {
        for (j, &filt_val) in filter.iter().enumerate() {
            result[i + j] += signal[i] * filt_val;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::wavelets::WaveletType;

    #[test]
    fn test_dwt_idwt_roundtrip_haar() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");
        let mode = ExtensionMode::Periodic;

        let (approx, detail) = dwt_1d(&signal, wavelet.as_ref(), mode).expect("DWT failed");

        let reconstructed = idwt_1d_mode(&approx, &detail, wavelet.as_ref(), signal.len(), mode)
            .expect("IDWT failed");

        assert_eq!(reconstructed.len(), signal.len());
        for (i, (&orig, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_dwt_idwt_roundtrip_db4() {
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let wavelet = WaveletType::Daubechies(4)
            .create()
            .expect("Failed to create wavelet");
        let mode = ExtensionMode::Periodic;

        let (approx, detail) = dwt_1d(&signal, wavelet.as_ref(), mode).expect("DWT failed");

        let reconstructed = idwt_1d_mode(&approx, &detail, wavelet.as_ref(), signal.len(), mode)
            .expect("IDWT failed");

        assert_eq!(reconstructed.len(), signal.len());
        for (i, (&orig, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-8,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_dwt_output_length() {
        let signal = vec![1.0; 10];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let (approx, detail) =
            dwt_1d(&signal, wavelet.as_ref(), ExtensionMode::Periodic).expect("DWT failed");

        assert_eq!(approx.len(), 5);
        assert_eq!(detail.len(), 5);
    }

    #[test]
    fn test_dwt_odd_length() {
        let signal = vec![1.0; 9];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let (approx, detail) =
            dwt_1d(&signal, wavelet.as_ref(), ExtensionMode::Periodic).expect("DWT failed");

        assert_eq!(approx.len(), 5);
        assert_eq!(detail.len(), 5);
    }

    #[test]
    fn test_wavedec_waverec_roundtrip() {
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let wavelet = WaveletType::Daubechies(2)
            .create()
            .expect("Failed to create wavelet");
        let level = 3;
        let mode = ExtensionMode::Periodic;

        let (approx, details) =
            wavedec(&signal, wavelet.as_ref(), level, mode).expect("wavedec failed");

        assert_eq!(details.len(), level);

        let reconstructed = waverec_mode(&approx, &details, wavelet.as_ref(), signal.len(), mode)
            .expect("waverec failed");

        assert_eq!(reconstructed.len(), signal.len());
        for (i, (&orig, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-8,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_max_decomposition_level() {
        assert_eq!(max_decomposition_level(16, 2), 4);
        assert_eq!(max_decomposition_level(32, 4), 4);
        assert_eq!(max_decomposition_level(64, 8), 4);
        assert_eq!(max_decomposition_level(128, 2), 7);
    }

    #[test]
    fn test_extension_modes() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let filter_len = 4;

        let periodic = extend_signal(&signal, filter_len, ExtensionMode::Periodic);
        assert_eq!(periodic.len(), 7);
        assert_eq!(&periodic[4..7], &[1.0, 2.0, 3.0]);

        let symmetric = extend_signal(&signal, filter_len, ExtensionMode::Symmetric);
        assert!(symmetric.len() >= signal.len());

        let zeropad = extend_signal(&signal, filter_len, ExtensionMode::ZeroPad);
        assert_eq!(zeropad.len(), 7);
        assert_eq!(&zeropad[4..7], &[0.0, 0.0, 0.0]);

        let smooth = extend_signal(&signal, filter_len, ExtensionMode::Smooth);
        assert_eq!(smooth.len(), 7);
        assert_eq!(&smooth[4..7], &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_dwt_empty_signal() {
        let signal: Vec<f64> = vec![];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let result = dwt_1d(&signal, wavelet.as_ref(), ExtensionMode::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_dwt_short_signal() {
        let signal = vec![1.0];
        let wavelet = WaveletType::Daubechies(4)
            .create()
            .expect("Failed to create wavelet");

        let result = dwt_1d(&signal, wavelet.as_ref(), ExtensionMode::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_idwt_mismatched_lengths() {
        let approx = vec![1.0, 2.0, 3.0];
        let detail = vec![1.0, 2.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let result = idwt_1d(&approx, &detail, wavelet.as_ref(), 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavedec_invalid_level() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let result = wavedec(&signal, wavelet.as_ref(), 0, ExtensionMode::Periodic);
        assert!(result.is_err());

        let result = wavedec(&signal, wavelet.as_ref(), 100, ExtensionMode::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_downsample_upsample() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let downsampled = downsample_2(&signal);
        assert_eq!(downsampled, vec![1.0, 3.0, 5.0]);

        let upsampled = upsample_2(&downsampled);
        assert_eq!(upsampled, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_convolve() {
        let signal = vec![1.0, 2.0, 3.0];
        let filter = vec![0.5, 0.5];
        let result = convolve(&signal, &filter);
        assert_eq!(result.len(), 4);
        assert_eq!(result, vec![0.5, 1.5, 2.5, 1.5]);
    }
}

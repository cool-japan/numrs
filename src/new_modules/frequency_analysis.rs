//! Advanced frequency domain analysis and spectral methods
//!
//! This module provides comprehensive frequency domain analysis capabilities
//! including power spectral density estimation, coherence analysis, and
//! advanced spectral methods for signal characterization.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::fft::FFT;
use crate::new_modules::signal_processing::SignalProcessor;
use num_traits::{Float, NumCast, Zero};
use scirs2_core::Complex;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Frequency domain analysis engine
pub struct FrequencyAnalyzer;

impl FrequencyAnalyzer {
    /// Estimate Power Spectral Density using Welch's method
    ///
    /// # Parameters
    /// * `signal` - Input signal
    /// * `nperseg` - Length of each segment (default: 256)
    /// * `noverlap` - Number of points to overlap between segments
    /// * `window` - Windowing function to apply
    /// * `nfft` - Length of FFT used (default: nperseg)
    /// * `detrend` - Whether to detrend each segment
    /// * `scaling` - 'density' or 'spectrum'
    pub fn welch<T>(
        signal: &Array<T>,
        nperseg: Option<usize>,
        noverlap: Option<usize>,
        window: &str,
        nfft: Option<usize>,
        detrend: bool,
        scaling: PSDScaling,
    ) -> Result<WelchResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();

        let nperseg = nperseg.unwrap_or(256.min(n));
        let noverlap = noverlap.unwrap_or(nperseg / 2);
        let nfft = nfft.unwrap_or(nperseg);

        if noverlap >= nperseg {
            return Err(NumRs2Error::InvalidOperation(
                "noverlap must be less than nperseg".to_string(),
            ));
        }

        let step = nperseg - noverlap;
        let n_segments = if n >= nperseg {
            (n - noverlap) / step
        } else {
            1
        };

        if n_segments == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Signal too short for segmentation".to_string(),
            ));
        }

        // Generate window function
        let window_values = Self::generate_window_function(nperseg, window)?;
        let window_power: f64 = window_values
            .iter()
            .map(|&w: &T| w.into())
            .map(|w: f64| w * w)
            .sum();

        let mut psd_accumulator = vec![T::zero(); nfft / 2 + 1];
        let mut segments_processed = 0;

        for i in 0..n_segments {
            let start = i * step;
            let end = (start + nperseg).min(n);

            if end - start < nperseg {
                continue; // Skip incomplete segments
            }

            // Extract segment
            let mut segment: Vec<T> = signal_data[start..end].to_vec();

            // Detrend if requested
            if detrend {
                let segment_array = Array::from_vec(segment.clone());
                let detrended = SignalProcessor::detrend(&segment_array)?;
                segment = detrended.to_vec();
            }

            // Apply window
            for (j, &window_val) in window_values.iter().enumerate() {
                segment[j] = segment[j] * window_val;
            }

            // Zero-pad to nfft length if needed
            if nperseg < nfft {
                segment.resize(nfft, T::zero());
            }

            // Compute FFT
            let segment_array = Array::from_vec(segment);
            let fft_result = FFT::fft(&segment_array)?;
            let fft_data = fft_result.to_vec();

            // Compute power spectral density for this segment
            let n_freqs = nfft / 2 + 1;
            for k in 0..n_freqs {
                let power = if k == 0 || (nfft.is_multiple_of(2) && k == nfft / 2) {
                    // DC and Nyquist components (if present) are not doubled
                    fft_data[k].norm_sqr()
                } else {
                    // Other components are doubled (since we only keep positive frequencies)
                    <T as NumCast>::from(2.0).expect("2.0 should convert to float type")
                        * fft_data[k].norm_sqr()
                };

                psd_accumulator[k] = psd_accumulator[k] + power;
            }

            segments_processed += 1;
        }

        if segments_processed == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "No segments processed".to_string(),
            ));
        }

        // Average over segments and normalize
        let segments_f = <T as NumCast>::from(segments_processed as f64).unwrap_or(T::one());
        let sample_rate = T::one(); // Assume normalized frequency

        for psd_val in &mut psd_accumulator {
            *psd_val = *psd_val / segments_f;

            // Apply scaling
            match scaling {
                PSDScaling::Density => {
                    // Scale by sampling frequency and window power
                    *psd_val = *psd_val
                        / (sample_rate * <T as NumCast>::from(window_power).unwrap_or(T::one()));
                }
                PSDScaling::Spectrum => {
                    // Scale by window power only
                    *psd_val = *psd_val / <T as NumCast>::from(window_power).unwrap_or(T::one());
                }
            }
        }

        // Generate frequency axis
        let freqs = FFT::rfftfreq(nfft, T::one() / sample_rate)?;

        Ok(WelchResult {
            frequencies: freqs,
            psd: Array::from_vec(psd_accumulator),
        })
    }

    /// Compute coherence between two signals
    pub fn coherence<T>(
        signal1: &Array<T>,
        signal2: &Array<T>,
        nperseg: Option<usize>,
        noverlap: Option<usize>,
        window: &str,
        nfft: Option<usize>,
    ) -> Result<CoherenceResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if signal1.shape() != signal2.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Signals must have the same length".to_string(),
            ));
        }

        let signal1_data = signal1.to_vec();
        let signal2_data = signal2.to_vec();
        let n = signal1_data.len();

        let nperseg = nperseg.unwrap_or(256.min(n));
        let noverlap = noverlap.unwrap_or(nperseg / 2);
        let nfft = nfft.unwrap_or(nperseg);

        let step = nperseg - noverlap;
        let n_segments = if n >= nperseg {
            (n - noverlap) / step
        } else {
            1
        };

        // Generate window function
        let window_values = Self::generate_window_function(nperseg, window)?;

        let mut psd1_accumulator = vec![Complex::<T>::zero(); nfft / 2 + 1];
        let mut psd2_accumulator = vec![Complex::<T>::zero(); nfft / 2 + 1];
        let mut cross_psd_accumulator = vec![Complex::<T>::zero(); nfft / 2 + 1];
        let mut segments_processed = 0;

        for i in 0..n_segments {
            let start = i * step;
            let end = (start + nperseg).min(n);

            if end - start < nperseg {
                continue;
            }

            // Extract and window segments
            let mut segment1: Vec<T> = signal1_data[start..end].to_vec();
            let mut segment2: Vec<T> = signal2_data[start..end].to_vec();

            for (j, &window_val) in window_values.iter().enumerate() {
                segment1[j] = segment1[j] * window_val;
                segment2[j] = segment2[j] * window_val;
            }

            // Zero-pad if needed
            if nperseg < nfft {
                segment1.resize(nfft, T::zero());
                segment2.resize(nfft, T::zero());
            }

            // Compute FFTs
            let fft1 = FFT::fft(&Array::from_vec(segment1))?;
            let fft2 = FFT::fft(&Array::from_vec(segment2))?;
            let fft1_data = fft1.to_vec();
            let fft2_data = fft2.to_vec();

            // Accumulate PSDs and cross-PSD
            let n_freqs = nfft / 2 + 1;
            for k in 0..n_freqs {
                let f1 = fft1_data[k];
                let f2 = fft2_data[k];

                psd1_accumulator[k] =
                    psd1_accumulator[k] + Complex::<T>::new(f1.norm_sqr(), T::zero());
                psd2_accumulator[k] =
                    psd2_accumulator[k] + Complex::<T>::new(f2.norm_sqr(), T::zero());
                cross_psd_accumulator[k] = cross_psd_accumulator[k] + f1 * f2.conj();
            }

            segments_processed += 1;
        }

        if segments_processed == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "No segments processed".to_string(),
            ));
        }

        // Compute coherence: |Pxy|^2 / (Pxx * Pyy)
        let mut coherence_values = Vec::with_capacity(nfft / 2 + 1);

        for k in 0..(nfft / 2 + 1) {
            let psd1 = psd1_accumulator[k].re;
            let psd2 = psd2_accumulator[k].re;
            let cross_psd_mag_sq = cross_psd_accumulator[k].norm_sqr();

            let coherence = if psd1 > T::zero() && psd2 > T::zero() {
                cross_psd_mag_sq / (psd1 * psd2)
            } else {
                T::zero()
            };

            coherence_values.push(coherence);
        }

        // Generate frequency axis
        let freqs = FFT::rfftfreq(nfft, T::one())?;

        Ok(CoherenceResult {
            frequencies: freqs,
            coherence: Array::from_vec(coherence_values),
        })
    }

    /// Compute Cross-Power Spectral Density between two signals
    pub fn cross_spectral_density<T>(
        signal1: &Array<T>,
        signal2: &Array<T>,
        nperseg: Option<usize>,
        noverlap: Option<usize>,
        window: &str,
        nfft: Option<usize>,
    ) -> Result<CrossSpectralResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if signal1.shape() != signal2.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Signals must have the same length".to_string(),
            ));
        }

        let signal1_data = signal1.to_vec();
        let signal2_data = signal2.to_vec();
        let n = signal1_data.len();

        let nperseg = nperseg.unwrap_or(256.min(n));
        let noverlap = noverlap.unwrap_or(nperseg / 2);
        let nfft = nfft.unwrap_or(nperseg);

        let step = nperseg - noverlap;
        let n_segments = if n >= nperseg {
            (n - noverlap) / step
        } else {
            1
        };

        let window_values = Self::generate_window_function(nperseg, window)?;
        let window_power: f64 = window_values
            .iter()
            .map(|&w: &T| w.into())
            .map(|w: f64| w * w)
            .sum();

        let mut cross_psd_accumulator = vec![Complex::<T>::zero(); nfft / 2 + 1];
        let mut segments_processed = 0;

        for i in 0..n_segments {
            let start = i * step;
            let end = (start + nperseg).min(n);

            if end - start < nperseg {
                continue;
            }

            let mut segment1: Vec<T> = signal1_data[start..end].to_vec();
            let mut segment2: Vec<T> = signal2_data[start..end].to_vec();

            // Apply window
            for (j, &window_val) in window_values.iter().enumerate() {
                segment1[j] = segment1[j] * window_val;
                segment2[j] = segment2[j] * window_val;
            }

            // Zero-pad if needed
            if nperseg < nfft {
                segment1.resize(nfft, T::zero());
                segment2.resize(nfft, T::zero());
            }

            // Compute FFTs
            let fft1 = FFT::fft(&Array::from_vec(segment1))?;
            let fft2 = FFT::fft(&Array::from_vec(segment2))?;
            let fft1_data = fft1.to_vec();
            let fft2_data = fft2.to_vec();

            // Accumulate cross-PSD
            let n_freqs = nfft / 2 + 1;
            for k in 0..n_freqs {
                cross_psd_accumulator[k] =
                    cross_psd_accumulator[k] + fft1_data[k] * fft2_data[k].conj();
            }

            segments_processed += 1;
        }

        if segments_processed == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "No segments processed".to_string(),
            ));
        }

        // Average and normalize
        let segments_f = <T as NumCast>::from(segments_processed as f64).unwrap_or(T::one());
        let sample_rate = T::one();
        let window_norm = <T as NumCast>::from(window_power).unwrap_or(T::one());

        for cpsd_val in &mut cross_psd_accumulator {
            *cpsd_val = *cpsd_val / Complex::<T>::new(segments_f, T::zero());
            *cpsd_val = *cpsd_val / Complex::<T>::new(sample_rate * window_norm, T::zero());
        }

        let freqs = FFT::rfftfreq(nfft, T::one())?;

        Ok(CrossSpectralResult {
            frequencies: freqs,
            cross_psd: Array::from_vec(cross_psd_accumulator),
        })
    }

    /// Compute periodogram (direct method for PSD estimation)
    pub fn periodogram<T>(
        signal: &Array<T>,
        window: Option<&str>,
        scaling: PSDScaling,
    ) -> Result<PeriodogramResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let mut signal_data = signal.to_vec();
        let n = signal_data.len();

        // Apply window if specified
        let window_power = if let Some(window_type) = window {
            let window_values = Self::generate_window_function(n, window_type)?;
            let window_power: f64 = window_values
                .iter()
                .map(|&w: &T| w.into())
                .map(|w: f64| w * w)
                .sum();

            for (i, &window_val) in window_values.iter().enumerate() {
                signal_data[i] = signal_data[i] * window_val;
            }

            window_power
        } else {
            n as f64 // Rectangular window
        };

        // Compute FFT
        let windowed_signal = Array::from_vec(signal_data);
        let fft_result = FFT::fft(&windowed_signal)?;
        let fft_data = fft_result.to_vec();

        // Compute periodogram
        let n_freqs = n / 2 + 1;
        let mut periodogram_values = Vec::with_capacity(n_freqs);
        let sample_rate = T::one();

        for k in 0..n_freqs {
            let power = if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                fft_data[k].norm_sqr()
            } else {
                <T as NumCast>::from(2.0).expect("2.0 should convert to float type")
                    * fft_data[k].norm_sqr()
            };

            let scaled_power = match scaling {
                PSDScaling::Density => {
                    power / (sample_rate * <T as NumCast>::from(window_power).unwrap_or(T::one()))
                }
                PSDScaling::Spectrum => {
                    power / <T as NumCast>::from(window_power).unwrap_or(T::one())
                }
            };

            periodogram_values.push(scaled_power);
        }

        let freqs = FFT::rfftfreq(n, T::one())?;

        Ok(PeriodogramResult {
            frequencies: freqs,
            psd: Array::from_vec(periodogram_values),
        })
    }

    /// Compute multitaper spectral estimation
    pub fn multitaper<T>(
        signal: &Array<T>,
        bandwidth: T,
        n_tapers: usize,
    ) -> Result<MultitaperResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();

        if n_tapers == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Number of tapers must be positive".to_string(),
            ));
        }

        // Generate DPSS (Discrete Prolate Spheroidal Sequences) tapers together
        // with their concentration eigenvalues lambda_m.
        let (tapers, eigenvalues) = Self::generate_dpss_tapers(n, bandwidth, n_tapers)?;

        let mut psd_accumulator = vec![T::zero(); n / 2 + 1];
        // Weight each eigenspectrum by its concentration eigenvalue (the
        // standard eigenvalue-weighted multitaper estimate). Tapers with higher
        // spectral concentration contribute more to the final estimate.
        let mut weight_sum = T::zero();

        for (taper, &lambda) in tapers.iter().zip(eigenvalues.iter()) {
            // Apply taper to signal
            let mut tapered_signal = Vec::with_capacity(n);
            for (i, &sig_val) in signal_data.iter().enumerate() {
                tapered_signal.push(sig_val * taper[i]);
            }

            // Compute periodogram for this taper (the m-th eigenspectrum)
            let tapered_array = Array::from_vec(tapered_signal);
            let periodogram = Self::periodogram(&tapered_array, None, PSDScaling::Density)?;
            let periodogram_data = periodogram.psd.to_vec();

            // Accumulate the eigenvalue-weighted eigenspectra
            for (i, &psd_val) in periodogram_data.iter().enumerate() {
                psd_accumulator[i] = psd_accumulator[i] + lambda * psd_val;
            }
            weight_sum = weight_sum + lambda;
        }

        // Normalize by the total weight (falls back to uniform averaging if the
        // eigenvalues are degenerate/non-positive for any reason).
        let normalizer = if weight_sum > T::zero() {
            weight_sum
        } else {
            <T as NumCast>::from(n_tapers as f64).unwrap_or(T::one())
        };
        for psd_val in &mut psd_accumulator {
            *psd_val = *psd_val / normalizer;
        }

        let freqs = FFT::rfftfreq(n, T::one())?;

        Ok(MultitaperResult {
            frequencies: freqs,
            psd: Array::from_vec(psd_accumulator),
            eigenvalues: Array::from_vec(eigenvalues),
        })
    }

    /// Generate DPSS (Discrete Prolate Spheroidal Sequences / Slepian) tapers.
    ///
    /// The DPSS tapers `v_0, v_1, ..., v_{K-1}` of length `N` for the
    /// time-half-bandwidth product `NW` (with half-bandwidth `W = bandwidth`,
    /// expressed as a fraction of the sampling frequency) are the eigenvectors,
    /// ordered by *descending* eigenvalue, of the symmetric tridiagonal matrix
    /// `T` defined by Slepian (1978):
    ///
    /// ```text
    ///   diagonal:     d_k = ((N - 1 - 2k) / 2)^2 * cos(2*pi*W),   k = 0..N-1
    ///   off-diagonal: e_k = (k * (N - k)) / 2,                    k = 1..N-1
    /// ```
    ///
    /// The eigenvectors of `T` are exactly the DPSS (they coincide with the
    /// eigenvectors of the sinc concentration kernel), but `T` is far better
    /// conditioned. We solve the eigenproblem with a symmetric tridiagonal
    /// QL algorithm with implicit shifts (`tql2`, EISPACK), which returns all
    /// eigenpairs simultaneously with high accuracy. The tapers are then sorted
    /// so that the most concentrated sequence comes first, normalized to unit
    /// energy, and given the standard sign convention.
    ///
    /// This function returns the tapers together with their associated
    /// concentration eigenvalues `lambda_m` (the energy fraction of taper `m`
    /// inside the band `[-W, W]`), computed from the sinc kernel.
    fn generate_dpss_tapers<T>(
        n: usize,
        bandwidth: T,
        n_tapers: usize,
    ) -> Result<(Vec<Vec<T>>, Vec<T>)>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if n == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "DPSS length n must be positive".to_string(),
            ));
        }
        if n_tapers > n {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Number of tapers ({}) cannot exceed signal length ({})",
                n_tapers, n
            )));
        }

        // Half-bandwidth W as a fraction of the sampling frequency.
        let w_frac = bandwidth.into();
        if w_frac <= 0.0 || w_frac >= 0.5 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "DPSS bandwidth (W) must lie in (0, 0.5), got {}",
                w_frac
            )));
        }

        // Trivial single-sample case: the only taper is the constant 1.
        if n == 1 {
            let tapers = vec![vec![<T as NumCast>::from(1.0).unwrap_or(T::one())]];
            let eigenvalues = vec![<T as NumCast>::from(2.0 * w_frac).unwrap_or(T::zero())];
            return Ok((tapers, eigenvalues));
        }

        // Build the symmetric tridiagonal matrix T (Slepian formulation).
        let cos_factor = (2.0 * PI * w_frac).cos();
        let mut diag = vec![0.0_f64; n];
        // Sub-diagonal of length n-1 (e[i] couples row i and row i+1).
        let mut sub = vec![0.0_f64; n];

        for k in 0..n {
            let centered = (n as f64 - 1.0 - 2.0 * k as f64) / 2.0;
            diag[k] = centered * centered * cos_factor;
        }
        // Off-diagonal e_k = k*(N-k)/2 for k = 1..N-1; store it as the
        // sub-diagonal entry coupling rows (k-1) and k, i.e. sub[k].
        for k in 1..n {
            sub[k] = (k as f64) * (n as f64 - k as f64) / 2.0;
        }

        // Solve the symmetric tridiagonal eigenproblem: eigenvalues in `eigvals`
        // (ascending), eigenvectors as columns of `z` (n x n).
        let (eigvals, z) = Self::symmetric_tridiagonal_eig(&diag, &sub)?;

        // Order eigenpairs by DESCENDING eigenvalue of T: the most concentrated
        // DPSS correspond to the largest eigenvalues of T.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            eigvals[b]
                .partial_cmp(&eigvals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tapers: Vec<Vec<T>> = Vec::with_capacity(n_tapers);
        for (taper_index, &col) in order.iter().take(n_tapers).enumerate() {
            // Extract eigenvector (column `col` of z).
            let mut taper_f = vec![0.0_f64; n];
            for row in 0..n {
                taper_f[row] = z[row * n + col];
            }

            // Normalize to unit energy.
            let norm = taper_f.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in &mut taper_f {
                    *value /= norm;
                }
            }

            // Sign convention (Percival & Walden, after the SciPy convention):
            // even-order tapers have a positive sum; odd-order tapers have a
            // positive initial slope (first half-sum dominates).
            if taper_index % 2 == 0 {
                let sum: f64 = taper_f.iter().sum();
                if sum < 0.0 {
                    for value in &mut taper_f {
                        *value = -*value;
                    }
                }
            } else {
                // Odd-order taper: enforce a positive initial slope by checking
                // the first-moment (skewness) about the series centre.
                let skew: f64 = (0..n)
                    .map(|i| (i as f64 - (n as f64 - 1.0) / 2.0) * taper_f[i])
                    .sum();
                if skew < 0.0 {
                    for value in &mut taper_f {
                        *value = -*value;
                    }
                }
            }

            let taper: Vec<T> = taper_f
                .iter()
                .map(|&x| <T as NumCast>::from(x).unwrap_or(T::zero()))
                .collect();
            tapers.push(taper);
        }

        // Compute the true concentration eigenvalues lambda_m by applying the
        // sinc concentration kernel S, where S_{ij} = sin(2*pi*W*(i-j)) /
        // (pi*(i-j)) and S_{ii} = 2W, to each taper: lambda_m = v_m^T S v_m.
        let mut eigenvalues: Vec<T> = Vec::with_capacity(tapers.len());
        for taper in &tapers {
            let taper_f: Vec<f64> = taper.iter().map(|&x| x.into()).collect();
            let mut lambda = 0.0_f64;
            for i in 0..n {
                // Diagonal contribution.
                lambda += taper_f[i] * (2.0 * w_frac) * taper_f[i];
                // Off-diagonal (symmetric) contributions.
                for j in (i + 1)..n {
                    let diff = (i as f64) - (j as f64);
                    let kernel = (2.0 * PI * w_frac * diff).sin() / (PI * diff);
                    lambda += 2.0 * taper_f[i] * kernel * taper_f[j];
                }
            }
            eigenvalues.push(<T as NumCast>::from(lambda).unwrap_or(T::zero()));
        }

        Ok((tapers, eigenvalues))
    }

    /// Symmetric tridiagonal eigensolver using the QL algorithm with implicit
    /// shifts (`tql2`, adapted from EISPACK / Numerical Recipes).
    ///
    /// # Arguments
    ///
    /// * `diagonal` - the `n` diagonal elements `d_0..d_{n-1}`.
    /// * `subdiagonal` - the sub-diagonal elements stored so that
    ///   `subdiagonal[k]` (for `k = 1..n-1`) couples rows `k-1` and `k`;
    ///   `subdiagonal[0]` is ignored.
    ///
    /// # Returns
    ///
    /// `(eigenvalues, eigenvectors)` where `eigenvalues` is sorted ascending and
    /// `eigenvectors` is an `n x n` matrix stored row-major; column `j` holds the
    /// eigenvector for `eigenvalues[j]`.
    fn symmetric_tridiagonal_eig(
        diagonal: &[f64],
        subdiagonal: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = diagonal.len();

        // d holds the eigenvalues (initially the diagonal); e holds the
        // sub-diagonal shifted so that e[i] couples rows i and i+1, with
        // e[n-1] = 0 as the algorithm expects.
        let mut d = diagonal.to_vec();
        let mut e = vec![0.0_f64; n];
        if n > 1 {
            e[..n - 1].copy_from_slice(&subdiagonal[1..n]);
        }

        // z is initialised to the identity (we accumulate the full eigenvectors).
        let mut z = vec![0.0_f64; n * n];
        for i in 0..n {
            z[i * n + i] = 1.0;
        }

        if n == 1 {
            return Ok((d, z));
        }

        // Maximum sweeps per eigenvalue before declaring non-convergence.
        let max_iter = 50;

        for l in 0..n {
            let mut iter = 0usize;
            loop {
                // Look for a single small sub-diagonal element e[m] to split off
                // a converged eigenvalue at position l.
                let mut m = l;
                while m < n - 1 {
                    let dd = d[m].abs() + d[m + 1].abs();
                    if (e[m].abs() + dd) == dd {
                        break;
                    }
                    m += 1;
                }

                if m == l {
                    // e[l] is negligible: d[l] is an eigenvalue; move on.
                    break;
                }

                if iter >= max_iter {
                    return Err(NumRs2Error::InvalidOperation(
                        "DPSS tridiagonal QL iteration failed to converge".to_string(),
                    ));
                }
                iter += 1;

                // Form the implicit Wilkinson shift (eigenvalue of trailing 2x2).
                let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                let mut r = g.hypot(1.0);
                // g = d[m] - shift, with the shift chosen for stability.
                let signed_r = if g >= 0.0 { r.abs() } else { -r.abs() };
                g = d[m] - d[l] + e[l] / (g + signed_r);

                let mut s = 1.0_f64;
                let mut c = 1.0_f64;
                let mut p = 0.0_f64;
                // Tracks whether an underflow recovery occurred (restart sweep).
                let mut underflow = false;

                // Givens plane rotations, chasing the bulge from m-1 down to l.
                let mut i = m as isize - 1;
                while i >= l as isize {
                    let idx = i as usize;
                    let mut f = s * e[idx];
                    let b = c * e[idx];
                    r = f.hypot(g);
                    e[idx + 1] = r;

                    if r == 0.0 {
                        // Underflow: recover and restart this sweep (NR tqli).
                        d[idx + 1] -= p;
                        e[m] = 0.0;
                        underflow = true;
                        break;
                    }

                    s = f / r;
                    c = g / r;
                    g = d[idx + 1] - p;
                    r = (d[idx] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[idx + 1] = g + p;
                    g = c * r - b;

                    // Accumulate this rotation into the eigenvector matrix.
                    for k in 0..n {
                        f = z[k * n + idx + 1];
                        z[k * n + idx + 1] = s * z[k * n + idx] + c * f;
                        z[k * n + idx] = c * z[k * n + idx] - s * f;
                    }

                    i -= 1;
                }

                if underflow {
                    // Restart the search/iteration for this l.
                    continue;
                }

                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }

        // Sort eigenpairs ascending by eigenvalue (selection sort, swapping the
        // corresponding eigenvector columns) for a deterministic ordering.
        for ii in 0..n {
            let mut min_idx = ii;
            for jj in (ii + 1)..n {
                if d[jj] < d[min_idx] {
                    min_idx = jj;
                }
            }
            if min_idx != ii {
                d.swap(ii, min_idx);
                for row in 0..n {
                    z.swap(row * n + ii, row * n + min_idx);
                }
            }
        }

        Ok((d, z))
    }

    /// Generate window function
    pub fn generate_window_function<T>(n: usize, window_type: &str) -> Result<Vec<T>>
    where
        T: Float + Clone + Debug + From<f64>,
    {
        match window_type.to_lowercase().as_str() {
            "hann" | "hanning" => {
                let window: Vec<T> = (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.5 * (1.0 - arg.cos())).unwrap_or(T::zero())
                    })
                    .collect();
                Ok(window)
            }
            "hamming" => {
                let window: Vec<T> = (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.54 - 0.46 * arg.cos()).unwrap_or(T::zero())
                    })
                    .collect();
                Ok(window)
            }
            "blackman" => {
                let window: Vec<T> = (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos())
                            .unwrap_or(T::zero())
                    })
                    .collect();
                Ok(window)
            }
            "bartlett" => {
                let window: Vec<T> = (0..n)
                    .map(|i| {
                        let val = if n == 1 {
                            1.0
                        } else {
                            2.0 / (n - 1) as f64 * (i as f64 - (n - 1) as f64 / 2.0).abs()
                        };
                        <T as NumCast>::from(1.0 - val).unwrap_or(T::zero())
                    })
                    .collect();
                Ok(window)
            }
            "rectangular" | "boxcar" => Ok(vec![<T as NumCast>::from(1.0).unwrap_or(T::zero()); n]),
            "kaiser" => {
                // Simplified Kaiser window (beta = 8.6)
                let beta = 8.6;
                let window: Vec<T> = (0..n)
                    .map(|i| {
                        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                        let val = Self::modified_bessel_i0(beta * (1.0 - x * x).sqrt())
                            / Self::modified_bessel_i0(beta);
                        <T as NumCast>::from(val).unwrap_or(T::zero())
                    })
                    .collect();
                Ok(window)
            }
            _ => Err(NumRs2Error::InvalidOperation(format!(
                "Unknown window type: {}",
                window_type
            ))),
        }
    }

    /// Modified Bessel function of the first kind (order 0)
    fn modified_bessel_i0(x: f64) -> f64 {
        let t = x / 3.75;
        if x.abs() < 3.75 {
            let t2 = t * t;
            1.0 + 3.5156229 * t2
                + 3.0899424 * t2 * t2
                + 1.2067492 * t2 * t2 * t2
                + 0.2659732 * t2 * t2 * t2 * t2
                + 0.0360768 * t2 * t2 * t2 * t2 * t2
                + 0.0045813 * t2 * t2 * t2 * t2 * t2 * t2
        } else {
            let inv_t = 1.0 / t;
            (x.abs().exp() / x.abs().sqrt())
                * (0.39894228 + 0.01328592 * inv_t + 0.00225319 * inv_t * inv_t
                    - 0.00157565 * inv_t * inv_t * inv_t
                    + 0.00916281 * inv_t * inv_t * inv_t * inv_t
                    - 0.02057706 * inv_t * inv_t * inv_t * inv_t * inv_t)
        }
    }
}

/// Power Spectral Density scaling options
#[derive(Debug, Clone, Copy)]
pub enum PSDScaling {
    /// Power spectral density [V^2/Hz]
    Density,
    /// Power spectrum [V^2]
    Spectrum,
}

/// Result of Welch's method PSD estimation
#[derive(Debug)]
pub struct WelchResult<T: Clone> {
    pub frequencies: Array<T>,
    pub psd: Array<T>,
}

/// Result of coherence analysis
#[derive(Debug)]
pub struct CoherenceResult<T: Clone> {
    pub frequencies: Array<T>,
    pub coherence: Array<T>,
}

/// Result of cross-spectral density analysis
#[derive(Debug)]
pub struct CrossSpectralResult<T: Clone> {
    pub frequencies: Array<T>,
    pub cross_psd: Array<Complex<T>>,
}

/// Result of periodogram analysis
#[derive(Debug)]
pub struct PeriodogramResult<T: Clone> {
    pub frequencies: Array<T>,
    pub psd: Array<T>,
}

/// Result of multitaper spectral estimation
#[derive(Debug)]
pub struct MultitaperResult<T: Clone> {
    pub frequencies: Array<T>,
    pub psd: Array<T>,
    pub eigenvalues: Array<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_welch_method() {
        // Create a test signal: sinusoid + noise
        let n = 1024;
        let mut signal = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let freq_signal = (2.0 * PI * 10.0 * t).sin(); // 10 Hz signal
            let noise = 0.1 * (2.0 * PI * 50.0 * t).sin(); // 50 Hz noise
            signal.push(freq_signal + noise);
        }

        let input = Array::from_vec(signal);
        let result = FrequencyAnalyzer::welch(
            &input,
            Some(256),
            Some(128),
            "hann",
            Some(256),
            false,
            PSDScaling::Density,
        )
        .expect("Welch PSD estimation should succeed");

        // Check that we get reasonable frequency resolution
        assert_eq!(result.frequencies.shape()[0], 129); // 256/2 + 1
        assert_eq!(result.psd.shape()[0], 129);

        // PSD values should be positive
        let psd_data = result.psd.to_vec();
        for &val in &psd_data {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_periodogram() {
        // Create a simple sinusoid
        let n = 128;
        let mut signal = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            signal.push((2.0 * PI * 5.0 * t).sin()); // 5 Hz signal
        }

        let input = Array::from_vec(signal);
        let result = FrequencyAnalyzer::periodogram(&input, Some("hann"), PSDScaling::Density)
            .expect("Periodogram computation should succeed");

        assert_eq!(result.frequencies.shape()[0], 65); // 128/2 + 1
        assert_eq!(result.psd.shape()[0], 65);

        // Find peak frequency
        let psd_data = result.psd.to_vec();
        let freq_data = result.frequencies.to_vec();

        let max_idx = psd_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("PSD data should have at least one element");

        // Peak should be around 5 Hz (allowing for discretization)
        let peak_freq = freq_data[max_idx];
        assert!((peak_freq - 5.0 / n as f64).abs() < 0.1);
    }

    #[test]
    fn test_coherence() {
        // Create two correlated signals
        let n = 256;
        let mut signal1 = Vec::with_capacity(n);
        let mut signal2 = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let base_signal = (2.0 * PI * 8.0 * t).sin();
            signal1.push(base_signal + 0.1 * (2.0 * PI * 20.0 * t).sin());
            signal2.push(base_signal + 0.1 * (2.0 * PI * 25.0 * t).sin());
        }

        let input1 = Array::from_vec(signal1);
        let input2 = Array::from_vec(signal2);

        let result =
            FrequencyAnalyzer::coherence(&input1, &input2, Some(64), Some(32), "hann", Some(64))
                .expect("Coherence computation should succeed");

        assert_eq!(result.frequencies.shape()[0], 33); // 64/2 + 1
        assert_eq!(result.coherence.shape()[0], 33);

        // Coherence values should be between 0 and 1
        let coherence_data = result.coherence.to_vec();
        for &val in &coherence_data {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_window_functions() {
        let n = 64;

        // Test Hann window
        let hann = FrequencyAnalyzer::generate_window_function::<f64>(n, "hann")
            .expect("Hann window generation should succeed");
        assert_eq!(hann.len(), n);
        assert_relative_eq!(hann[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(hann[n - 1], 0.0, epsilon = 1e-10);

        // Test Hamming window
        let hamming = FrequencyAnalyzer::generate_window_function::<f64>(n, "hamming")
            .expect("Hamming window generation should succeed");
        assert_eq!(hamming.len(), n);

        // Test rectangular window
        let rectangular = FrequencyAnalyzer::generate_window_function::<f64>(n, "rectangular")
            .expect("Rectangular window generation should succeed");
        assert_eq!(rectangular.len(), n);
        for &val in &rectangular {
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }

        // Test Blackman window
        let blackman = FrequencyAnalyzer::generate_window_function::<f64>(n, "blackman")
            .expect("Blackman window generation should succeed");
        assert_eq!(blackman.len(), n);
        assert_relative_eq!(blackman[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(blackman[n - 1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_spectral_density() {
        // Create two signals with known relationship
        let n = 128;
        let mut signal1 = Vec::with_capacity(n);
        let mut signal2 = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            let sig1 = (2.0 * PI * 4.0 * t).sin();
            let sig2 = (2.0 * PI * 4.0 * t + PI / 4.0).sin(); // Phase shifted
            signal1.push(sig1);
            signal2.push(sig2);
        }

        let input1 = Array::from_vec(signal1);
        let input2 = Array::from_vec(signal2);

        let result = FrequencyAnalyzer::cross_spectral_density(
            &input1,
            &input2,
            Some(64),
            Some(32),
            "hann",
            Some(64),
        )
        .expect("Cross spectral density computation should succeed");

        assert_eq!(result.frequencies.shape()[0], 33); // 64/2 + 1
        assert_eq!(result.cross_psd.shape()[0], 33);

        // Cross-PSD should be complex
        let cross_psd_data = result.cross_psd.to_vec();
        assert!(!cross_psd_data.is_empty());
    }

    #[test]
    fn test_multitaper() {
        // Create a test signal
        let n = 128;
        let mut signal = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64;
            signal.push((2.0 * PI * 6.0 * t).sin() + 0.1 * (2.0 * PI * 15.0 * t).sin());
        }

        let input = Array::from_vec(signal);
        let result = FrequencyAnalyzer::multitaper(&input, 0.1, 3)
            .expect("Multitaper estimation should succeed");

        assert_eq!(result.frequencies.shape()[0], 65); // 128/2 + 1
        assert_eq!(result.psd.shape()[0], 65);
        assert_eq!(result.eigenvalues.shape()[0], 3);

        // PSD values should be positive
        let psd_data = result.psd.to_vec();
        for &val in &psd_data {
            assert!(val >= 0.0);
        }
    }
}

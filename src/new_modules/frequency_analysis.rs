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

        // Generate DPSS (Discrete Prolate Spheroidal Sequences) tapers
        let tapers = Self::generate_dpss_tapers(n, bandwidth, n_tapers)?;

        let mut psd_accumulator = vec![T::zero(); n / 2 + 1];

        for taper in &tapers {
            // Apply taper to signal
            let mut tapered_signal = Vec::with_capacity(n);
            for (i, &sig_val) in signal_data.iter().enumerate() {
                tapered_signal.push(sig_val * taper[i]);
            }

            // Compute periodogram for this taper
            let tapered_array = Array::from_vec(tapered_signal);
            let periodogram = Self::periodogram(&tapered_array, None, PSDScaling::Density)?;
            let periodogram_data = periodogram.psd.to_vec();

            // Accumulate
            for (i, &psd_val) in periodogram_data.iter().enumerate() {
                psd_accumulator[i] = psd_accumulator[i] + psd_val;
            }
        }

        // Average over tapers
        let n_tapers_f = <T as NumCast>::from(n_tapers as f64).unwrap_or(T::one());
        for psd_val in &mut psd_accumulator {
            *psd_val = *psd_val / n_tapers_f;
        }

        let freqs = FFT::rfftfreq(n, T::one())?;

        Ok(MultitaperResult {
            frequencies: freqs,
            psd: Array::from_vec(psd_accumulator),
            eigenvalues: Array::from_vec(vec![T::one(); n_tapers]), // Simplified
        })
    }

    /// Generate DPSS tapers (simplified implementation)
    fn generate_dpss_tapers<T>(n: usize, bandwidth: T, n_tapers: usize) -> Result<Vec<Vec<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let mut tapers = Vec::with_capacity(n_tapers);
        let nw = bandwidth.into() * n as f64 / 2.0;

        // Simplified DPSS generation (in practice, this would use eigenvalue decomposition)
        for k in 0..n_tapers {
            let mut taper = Vec::with_capacity(n);

            for i in 0..n {
                let t = (i as f64 - (n as f64 - 1.0) / 2.0) / (n as f64 / 2.0);
                let w = nw / (n as f64 / 2.0);

                // Simplified taper (not true DPSS)
                let val = if t.abs() < w {
                    let arg = PI * t / w;
                    if arg.abs() < 1e-10 {
                        1.0
                    } else {
                        arg.sin() / arg
                    }
                } else {
                    0.0
                };

                taper.push(<T as NumCast>::from(val * (k as f64 + 1.0).cos()).unwrap_or(T::zero()));
            }

            // Normalize
            let norm: f64 = taper
                .iter()
                .map(|&x| x.into())
                .map(|x: f64| x * x)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for taper_val in &mut taper {
                    *taper_val = *taper_val / <T as NumCast>::from(norm).unwrap_or(T::one());
                }
            }

            tapers.push(taper);
        }

        Ok(tapers)
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

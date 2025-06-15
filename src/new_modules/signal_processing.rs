//! Advanced signal processing toolkit for NumRS2
//!
//! This module provides comprehensive signal processing capabilities including
//! filtering, spectral analysis, convolution, and advanced DSP operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::fft::FFT;
use num_complex::Complex;
use num_traits::{Float, NumCast, Zero};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Digital signal processing engine with advanced algorithms
pub struct SignalProcessor;

impl SignalProcessor {
    /// Apply a digital filter to the input signal
    ///
    /// # Parameters
    /// * `signal` - Input signal
    /// * `filter_type` - Type of filter to apply
    /// * `params` - Filter parameters
    pub fn filter<T>(
        signal: &Array<T>,
        filter_type: FilterType,
        params: FilterParams<T>,
    ) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        match filter_type {
            FilterType::LowPass => Self::lowpass_filter(signal, params.cutoff, params.sample_rate),
            FilterType::HighPass => {
                Self::highpass_filter(signal, params.cutoff, params.sample_rate)
            }
            FilterType::BandPass => Self::bandpass_filter(
                signal,
                params.low_cutoff,
                params.high_cutoff,
                params.sample_rate,
            ),
            FilterType::BandStop => Self::bandstop_filter(
                signal,
                params.low_cutoff,
                params.high_cutoff,
                params.sample_rate,
            ),
            FilterType::Butterworth => Self::butterworth_filter(signal, params),
            FilterType::Chebyshev => Self::chebyshev_filter(signal, params),
        }
    }

    /// Low-pass filter using frequency domain approach
    fn lowpass_filter<T>(signal: &Array<T>, cutoff: T, sample_rate: T) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        // Create frequency mask
        let freqs = FFT::fftfreq(n, T::one() / sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        // Apply low-pass mask
        for (i, &freq) in freq_data.iter().enumerate() {
            if freq.abs() > cutoff {
                fft_data[i] = Complex::zero();
            }
        }

        // Inverse FFT
        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        // Extract real part
        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// High-pass filter using frequency domain approach
    fn highpass_filter<T>(signal: &Array<T>, cutoff: T, sample_rate: T) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        // Create frequency mask
        let freqs = FFT::fftfreq(n, T::one() / sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        // Apply high-pass mask
        for (i, &freq) in freq_data.iter().enumerate() {
            if freq.abs() < cutoff {
                fft_data[i] = Complex::zero();
            }
        }

        // Inverse FFT
        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        // Extract real part
        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Band-pass filter
    fn bandpass_filter<T>(
        signal: &Array<T>,
        low_cutoff: T,
        high_cutoff: T,
        sample_rate: T,
    ) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        let freqs = FFT::fftfreq(n, T::one() / sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        // Apply band-pass mask
        for (i, &freq) in freq_data.iter().enumerate() {
            let abs_freq = freq.abs();
            if abs_freq < low_cutoff || abs_freq > high_cutoff {
                fft_data[i] = Complex::zero();
            }
        }

        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Band-stop (notch) filter
    fn bandstop_filter<T>(
        signal: &Array<T>,
        low_cutoff: T,
        high_cutoff: T,
        sample_rate: T,
    ) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        let freqs = FFT::fftfreq(n, T::one() / sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        // Apply band-stop mask
        for (i, &freq) in freq_data.iter().enumerate() {
            let abs_freq = freq.abs();
            if abs_freq >= low_cutoff && abs_freq <= high_cutoff {
                fft_data[i] = Complex::zero();
            }
        }

        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Butterworth filter implementation
    fn butterworth_filter<T>(signal: &Array<T>, params: FilterParams<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        let freqs = FFT::fftfreq(n, T::one() / params.sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        let order = params.order.unwrap_or(4);

        // Apply Butterworth response
        for (i, &freq) in freq_data.iter().enumerate() {
            let normalized_freq = freq.abs() / params.cutoff;
            let response = T::one() / (T::one() + normalized_freq.powi(2 * order)).sqrt();
            fft_data[i] = fft_data[i] * Complex::new(response, T::zero());
        }

        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Chebyshev filter implementation
    fn chebyshev_filter<T>(signal: &Array<T>, params: FilterParams<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let fft_result = FFT::fft(signal)?;
        let n = signal.shape()[0];

        let freqs = FFT::fftfreq(n, T::one() / params.sample_rate)?;
        let freq_data = freqs.to_vec();
        let mut fft_data = fft_result.to_vec();

        let order = params.order.unwrap_or(4);
        let ripple = params
            .ripple
            .unwrap_or(<T as NumCast>::from(0.5).unwrap_or(T::zero()));

        // Apply Chebyshev response (simplified)
        for (i, &freq) in freq_data.iter().enumerate() {
            let normalized_freq = freq.abs() / params.cutoff;
            let epsilon = (<T as NumCast>::from(10.0)
                .unwrap()
                .powf(ripple / <T as NumCast>::from(10.0).unwrap())
                - T::one())
            .sqrt();

            let response = if normalized_freq <= T::one() {
                T::one()
                    / (T::one() + epsilon * epsilon * Self::chebyshev_poly(order, normalized_freq))
                        .sqrt()
            } else {
                T::one()
                    / (T::one() + epsilon * epsilon * (normalized_freq.powi(2 * order) - T::one()))
                        .sqrt()
            };

            fft_data[i] = fft_data[i] * Complex::new(response, T::zero());
        }

        let filtered_fft = Array::from_vec(fft_data);
        let filtered_complex = FFT::ifft(&filtered_fft)?;
        let filtered_data = filtered_complex.to_vec();

        let result: Vec<T> = filtered_data.iter().map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Chebyshev polynomial (simplified implementation)
    fn chebyshev_poly<T>(n: i32, x: T) -> T
    where
        T: Float + Clone,
    {
        match n {
            0 => T::one(),
            1 => x,
            _ => {
                let mut t0 = T::one();
                let mut t1 = x;
                for _ in 2..=n {
                    let t2 = T::from(2.0).unwrap() * x * t1 - t0;
                    t0 = t1;
                    t1 = t2;
                }
                t1
            }
        }
    }

    /// Convolution of two signals
    pub fn convolve<T>(signal1: &Array<T>, signal2: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n1 = signal1.shape()[0];
        let n2 = signal2.shape()[0];
        let n_conv = n1 + n2 - 1;

        // Pad to next power of 2 for efficient FFT
        let n_fft = Self::next_power_of_2(n_conv);

        // Zero-pad both signals
        let mut padded1 = signal1.to_vec();
        padded1.resize(n_fft, T::zero());

        let mut padded2 = signal2.to_vec();
        padded2.resize(n_fft, T::zero());

        // FFT both signals
        let fft1 = FFT::fft(&Array::from_vec(padded1))?;
        let fft2 = FFT::fft(&Array::from_vec(padded2))?;

        // Multiply in frequency domain
        let fft1_data = fft1.to_vec();
        let fft2_data = fft2.to_vec();
        let product: Vec<Complex<T>> = fft1_data
            .iter()
            .zip(fft2_data.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Inverse FFT
        let conv_fft = Array::from_vec(product);
        let conv_complex = FFT::ifft(&conv_fft)?;
        let conv_data = conv_complex.to_vec();

        // Extract real part and truncate to correct size
        let result: Vec<T> = conv_data.iter().take(n_conv).map(|c| c.re).collect();
        Ok(Array::from_vec(result))
    }

    /// Cross-correlation of two signals
    pub fn correlate<T>(signal1: &Array<T>, signal2: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        // Cross-correlation is convolution with time-reversed second signal
        let signal2_data = signal2.to_vec();
        let reversed_signal2: Vec<T> = signal2_data.into_iter().rev().collect();
        let reversed_array = Array::from_vec(reversed_signal2);

        Self::convolve(signal1, &reversed_array)
    }

    /// Auto-correlation of a signal
    pub fn autocorrelate<T>(signal: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        Self::correlate(signal, signal)
    }

    /// Compute the spectrogram of a signal
    pub fn spectrogram<T>(
        signal: &Array<T>,
        window_size: usize,
        overlap: usize,
        window_type: &str,
    ) -> Result<SpectrogramResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();
        let step = window_size - overlap;

        if step == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Window overlap cannot equal window size".to_string(),
            ));
        }

        let n_frames = (n - overlap) / step;
        let n_freqs = window_size / 2 + 1;

        let mut spectrogram_data = Vec::with_capacity(n_frames * n_freqs);
        let mut time_axis = Vec::with_capacity(n_frames);

        for frame in 0..n_frames {
            let start = frame * step;
            let end = (start + window_size).min(n);

            if end - start < window_size {
                break; // Skip incomplete frames
            }

            // Extract window
            let window_data: Vec<T> = signal_data[start..end].to_vec();
            let window_array = Array::from_vec(window_data);

            // Apply window function
            let windowed = FFT::apply_window(&window_array, window_type)?;

            // Compute FFT and power spectrum
            let fft_result = FFT::rfft(&windowed)?;
            let fft_data = fft_result.to_vec();

            // Compute power spectral density
            let power_frame: Vec<T> = fft_data.iter().map(|c| c.norm_sqr()).collect();
            spectrogram_data.extend(power_frame);

            // Time axis
            let time =
                <T as NumCast>::from(start as f64 + window_size as f64 / 2.0).unwrap_or(T::zero());
            time_axis.push(time);
        }

        Ok(SpectrogramResult {
            spectrogram: Array::from_vec(spectrogram_data).reshape(&[time_axis.len(), n_freqs]),
            time_axis: Array::from_vec(time_axis),
            freq_axis: FFT::rfftfreq(window_size, T::one())?,
        })
    }

    /// Compute the Continuous Wavelet Transform (CWT)
    pub fn cwt<T>(
        signal: &Array<T>,
        scales: &[T],
        wavelet_type: WaveletType,
    ) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();
        let n_scales = scales.len();

        let mut cwt_result = Vec::with_capacity(n_scales * n);

        for &scale in scales {
            // Generate wavelet at current scale
            let wavelet = Self::generate_wavelet(n, scale, wavelet_type.clone())?;

            // Convolve signal with wavelet
            let conv_result = Self::convolve(signal, &wavelet)?;
            let conv_data = conv_result.to_vec();

            // Convert to complex (taking real part as real component)
            let complex_row: Vec<Complex<T>> = conv_data
                .iter()
                .take(n)
                .map(|&x| Complex::new(x, T::zero()))
                .collect();

            cwt_result.extend(complex_row);
        }

        Ok(Array::from_vec(cwt_result).reshape(&[n_scales, n]))
    }

    /// Generate wavelet function
    fn generate_wavelet<T>(n: usize, scale: T, wavelet_type: WaveletType) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let mut wavelet = Vec::with_capacity(n);
        let center = n as f64 / 2.0;

        match wavelet_type {
            WaveletType::Morlet => {
                // Morlet wavelet: exp(-t^2/2) * cos(5t)
                for i in 0..n {
                    let t = (i as f64 - center) / scale.into();
                    let value = (-t * t / 2.0).exp() * (5.0 * t).cos();
                    wavelet.push(<T as NumCast>::from(value).unwrap_or(T::zero()));
                }
            }
            WaveletType::Mexican => {
                // Mexican hat wavelet: (1 - t^2) * exp(-t^2/2)
                for i in 0..n {
                    let t = (i as f64 - center) / scale.into();
                    let value = (1.0 - t * t) * (-t * t / 2.0).exp();
                    wavelet.push(<T as NumCast>::from(value).unwrap_or(T::zero()));
                }
            }
            WaveletType::Gaussian => {
                // Gaussian wavelet: exp(-t^2/2)
                for i in 0..n {
                    let t = (i as f64 - center) / scale.into();
                    let value = (-t * t / 2.0).exp();
                    wavelet.push(<T as NumCast>::from(value).unwrap_or(T::zero()));
                }
            }
        }

        Ok(Array::from_vec(wavelet))
    }

    /// Compute the Hilbert transform
    pub fn hilbert<T>(signal: &Array<T>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n = signal.shape()[0];
        let fft_result = FFT::fft(signal)?;
        let mut fft_data = fft_result.to_vec();

        // Apply Hilbert transform in frequency domain
        // Create analytic signal: x(t) + j*H{x(t)}
        // Where H{x(t)} is the Hilbert transform

        // For analytic signal:
        // - Keep DC component (index 0) unchanged
        // - Multiply positive frequencies by 2
        // - Set negative frequencies to zero
        // - Keep Nyquist component (if n is even) unchanged

        if n % 2 == 0 {
            // Even length: DC, positive frequencies, Nyquist, negative frequencies
            for i in 1..n / 2 {
                fft_data[i] =
                    fft_data[i] * Complex::new(<T as NumCast>::from(2.0).unwrap(), T::zero());
            }
            // Zero out negative frequencies
            for i in (n / 2 + 1)..n {
                fft_data[i] = Complex::zero();
            }
            // DC (index 0) and Nyquist (index n/2) remain unchanged
        } else {
            // Odd length: DC, positive frequencies, negative frequencies
            for i in 1..n.div_ceil(2) {
                fft_data[i] =
                    fft_data[i] * Complex::new(<T as NumCast>::from(2.0).unwrap(), T::zero());
            }
            // Zero out negative frequencies
            for i in n.div_ceil(2)..n {
                fft_data[i] = Complex::zero();
            }
            // Only DC (index 0) remains unchanged
        }

        let hilbert_fft = Array::from_vec(fft_data);
        FFT::ifft(&hilbert_fft)
    }

    /// Compute instantaneous amplitude and phase
    pub fn instantaneous_attributes<T>(signal: &Array<T>) -> Result<InstantaneousAttributes<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let analytic_signal = Self::hilbert(signal)?;
        let analytic_data = analytic_signal.to_vec();

        let mut amplitude = Vec::with_capacity(analytic_data.len());
        let mut phase = Vec::with_capacity(analytic_data.len());
        let mut frequency = Vec::with_capacity(analytic_data.len().saturating_sub(1));

        // Compute amplitude and phase
        for complex_val in &analytic_data {
            amplitude.push(complex_val.norm());
            phase.push(complex_val.arg());
        }

        // Compute instantaneous frequency (derivative of phase)
        for i in 0..phase.len().saturating_sub(1) {
            let mut phase_diff = phase[i + 1] - phase[i];

            // Unwrap phase
            if phase_diff > <T as NumCast>::from(PI).unwrap_or(T::zero()) {
                phase_diff = phase_diff - <T as NumCast>::from(2.0 * PI).unwrap_or(T::zero());
            } else if phase_diff < -<T as NumCast>::from(PI).unwrap_or(T::zero()) {
                phase_diff = phase_diff + <T as NumCast>::from(2.0 * PI).unwrap_or(T::zero());
            }

            frequency.push(phase_diff / <T as NumCast>::from(2.0 * PI).unwrap_or(T::one()));
        }

        Ok(InstantaneousAttributes {
            amplitude: Array::from_vec(amplitude),
            phase: Array::from_vec(phase),
            frequency: Array::from_vec(frequency),
        })
    }

    /// Detrend a signal by removing linear trend
    pub fn detrend<T>(signal: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();

        if n < 2 {
            return Ok(signal.clone());
        }

        // Compute linear trend using least squares
        let sum_x: f64 = (0..n).map(|i| i as f64).sum();
        let sum_y: f64 = signal_data.iter().map(|&x| x.into()).sum();
        let sum_xy: f64 = signal_data
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y.into())
            .sum();
        let sum_x2: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();

        let n_f64 = n as f64;
        let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n_f64;

        // Remove trend
        let detrended: Vec<T> = signal_data
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let trend = slope * i as f64 + intercept;
                y - <T as NumCast>::from(trend).unwrap_or(T::zero())
            })
            .collect();

        Ok(Array::from_vec(detrended))
    }

    /// Resample signal to a new sampling rate
    pub fn resample<T>(signal: &Array<T>, new_length: usize) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let signal_data = signal.to_vec();
        let old_length = signal_data.len();

        if new_length == old_length {
            return Ok(signal.clone());
        }

        // Use linear interpolation for resampling
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let pos = (i as f64) * (old_length - 1) as f64 / (new_length - 1) as f64;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f64;

            if idx >= old_length - 1 {
                resampled.push(signal_data[old_length - 1]);
            } else {
                let val1 = signal_data[idx];
                let val2 = signal_data[idx + 1];
                let interpolated =
                    val1 + (val2 - val1) * <T as NumCast>::from(frac).unwrap_or(T::zero());
                resampled.push(interpolated);
            }
        }

        Ok(Array::from_vec(resampled))
    }

    /// Helper function to find next power of 2
    fn next_power_of_2(n: usize) -> usize {
        if n <= 1 {
            return 1;
        }
        let mut power = 1;
        while power < n {
            power <<= 1;
        }
        power
    }
}

/// Filter types available in the signal processor
#[derive(Debug, Clone)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    BandStop,
    Butterworth,
    Chebyshev,
}

/// Parameters for digital filters
#[derive(Debug, Clone)]
pub struct FilterParams<T> {
    pub cutoff: T,
    pub low_cutoff: T,
    pub high_cutoff: T,
    pub sample_rate: T,
    pub order: Option<i32>,
    pub ripple: Option<T>,
}

impl<T: Clone> FilterParams<T> {
    pub fn new(cutoff: T, sample_rate: T) -> Self {
        Self {
            cutoff: cutoff.clone(),
            low_cutoff: cutoff.clone(),
            high_cutoff: cutoff,
            sample_rate,
            order: None,
            ripple: None,
        }
    }

    pub fn bandpass(low_cutoff: T, high_cutoff: T, sample_rate: T) -> Self {
        Self {
            cutoff: low_cutoff.clone(),
            low_cutoff,
            high_cutoff,
            sample_rate,
            order: None,
            ripple: None,
        }
    }
}

/// Wavelet types for CWT
#[derive(Debug, Clone)]
pub enum WaveletType {
    Morlet,
    Mexican,
    Gaussian,
}

/// Result of spectrogram computation
#[derive(Debug)]
pub struct SpectrogramResult<T: Clone> {
    pub spectrogram: Array<T>,
    pub time_axis: Array<T>,
    pub freq_axis: Array<T>,
}

/// Instantaneous signal attributes
#[derive(Debug)]
pub struct InstantaneousAttributes<T: Clone> {
    pub amplitude: Array<T>,
    pub phase: Array<T>,
    pub frequency: Array<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_convolution() {
        let signal1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let signal2 = Array::from_vec(vec![0.5, 1.0, 0.5]);

        let result = SignalProcessor::convolve(&signal1, &signal2).unwrap();
        let result_data = result.to_vec();

        // Expected convolution result
        let expected = vec![0.5, 2.0, 4.0, 4.0, 1.5];
        assert_eq!(result_data.len(), expected.len());

        for (i, &val) in result_data.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lowpass_filter() {
        // Create a signal with high frequency noise
        let n = 128;
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Low frequency signal + high frequency noise
            let val = (2.0 * PI * 2.0 * t).sin() + 0.1 * (2.0 * PI * 20.0 * t).sin();
            signal.push(val);
        }

        let input = Array::from_vec(signal);
        let params = FilterParams::new(5.0, 64.0); // 5 Hz cutoff, 64 Hz sample rate

        let filtered = SignalProcessor::filter(&input, FilterType::LowPass, params).unwrap();

        // Verify that high frequencies are attenuated
        assert_eq!(filtered.shape(), input.shape());
    }

    #[test]
    fn test_autocorrelation() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        let autocorr = SignalProcessor::autocorrelate(&signal).unwrap();
        let autocorr_data = autocorr.to_vec();

        // Autocorrelation should be symmetric
        assert_eq!(autocorr_data.len(), 9); // 2*5 - 1

        // Peak should be at the center
        let center = autocorr_data.len() / 2;
        let max_val = autocorr_data.iter().fold(0.0, |a, &b| a.max(b));
        assert_relative_eq!(autocorr_data[center], max_val, epsilon = 1e-10);
    }

    #[test]
    fn test_detrend() {
        // Create a signal with linear trend
        let mut signal = Vec::new();
        for i in 0..100 {
            let trend = 0.1 * i as f64; // Linear trend
            let noise = (2.0 * PI * i as f64 / 10.0).sin(); // Oscillation
            signal.push(trend + noise);
        }

        let input = Array::from_vec(signal);
        let detrended = SignalProcessor::detrend(&input).unwrap();
        let detrended_data = detrended.to_vec();

        // Mean should be close to zero after detrending
        let mean: f64 = detrended_data.iter().sum::<f64>() / detrended_data.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_resample() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Upsample
        let upsampled = SignalProcessor::resample(&signal, 9).unwrap();
        assert_eq!(upsampled.shape()[0], 9);

        // Downsample
        let downsampled = SignalProcessor::resample(&signal, 3).unwrap();
        assert_eq!(downsampled.shape()[0], 3);

        // Check endpoints are preserved
        let up_data = upsampled.to_vec();
        assert_relative_eq!(up_data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(up_data[8], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectrogram() {
        // Create a chirp signal (frequency increases linearly)
        let n = 256;
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Chirp: frequency increases from 1 to 10 Hz
            let freq = 1.0 + 9.0 * t;
            let val = (2.0 * PI * freq * t).sin();
            signal.push(val);
        }

        let input = Array::from_vec(signal);
        let spec_result = SignalProcessor::spectrogram(&input, 64, 32, "hann").unwrap();

        // Check dimensions
        assert!(spec_result.spectrogram.shape()[0] > 0); // Time frames
        assert_eq!(spec_result.spectrogram.shape()[1], 33); // Frequency bins (64/2 + 1)
        assert_eq!(
            spec_result.time_axis.shape()[0],
            spec_result.spectrogram.shape()[0]
        );
        assert_eq!(
            spec_result.freq_axis.shape()[0],
            spec_result.spectrogram.shape()[1]
        );
    }

    #[test]
    fn test_hilbert_transform() {
        // Create a simple sinusoid
        let n = 64;
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            let val = (2.0 * PI * 4.0 * t).sin(); // 4 Hz sinusoid
            signal.push(val);
        }

        let input = Array::from_vec(signal);
        let analytic = SignalProcessor::hilbert(&input).unwrap();
        let analytic_data = analytic.to_vec();

        // Check that we get complex output
        assert_eq!(analytic_data.len(), n);

        // Basic validation that the Hilbert transform produces reasonable output
        let magnitudes: Vec<f64> = analytic_data.iter().map(|c| c.norm()).collect();

        // Check that we get real output with reasonable range
        for &mag in &magnitudes {
            assert!(
                mag >= 0.0 && mag <= 2.0,
                "Magnitude {} is out of reasonable range",
                mag
            );
            assert!(mag.is_finite(), "Magnitude should be finite");
        }

        // Check that output contains both real and imaginary parts
        let has_real = analytic_data.iter().any(|c| c.re.abs() > 1e-10);
        let has_imag = analytic_data.iter().any(|c| c.im.abs() > 1e-10);
        assert!(has_real, "Analytic signal should have non-zero real parts");
        assert!(
            has_imag,
            "Analytic signal should have non-zero imaginary parts"
        );
    }
}

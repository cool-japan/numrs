//! Advanced spectral analysis and windowing functions
//!
//! This module provides comprehensive spectral analysis capabilities including
//! advanced windowing functions, spectral peak detection, harmonic analysis,
//! and time-frequency analysis methods.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::fft::FFT;
use crate::new_modules::frequency_analysis::FrequencyAnalyzer;
use num_complex::Complex;
use num_traits::{Float, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Advanced spectral analysis engine
pub struct SpectralAnalyzer;

impl SpectralAnalyzer {
    /// Detect spectral peaks in a power spectrum
    ///
    /// # Parameters
    /// * `spectrum` - Power spectrum
    /// * `frequencies` - Corresponding frequencies
    /// * `height` - Minimum peak height
    /// * `distance` - Minimum distance between peaks (in bins)
    /// * `prominence` - Minimum peak prominence
    /// * `width` - Minimum peak width
    pub fn find_peaks<T>(
        spectrum: &Array<T>,
        frequencies: &Array<T>,
        height: Option<T>,
        distance: Option<usize>,
        prominence: Option<T>,
        width: Option<T>,
    ) -> Result<PeakResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        let spectrum_data = spectrum.to_vec();
        let freq_data = frequencies.to_vec();
        let n = spectrum_data.len();
        
        if n != freq_data.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Spectrum and frequencies must have same length".to_string()
            ));
        }
        
        let mut peak_indices = Vec::new();
        let distance = distance.unwrap_or(1);
        
        // Find local maxima
        for i in 1..n-1 {
            let current = spectrum_data[i];
            let left = spectrum_data[i-1];
            let right = spectrum_data[i+1];
            
            // Check if it's a local maximum
            if current > left && current > right {
                // Check height threshold
                if let Some(min_height) = height {
                    if current < min_height {
                        continue;
                    }
                }
                
                // Check distance constraint
                if distance > 1 {
                    let mut too_close = false;
                    for &prev_idx in &peak_indices {
                        if i.abs_diff(prev_idx) < distance {
                            too_close = true;
                            break;
                        }
                    }
                    if too_close {
                        continue;
                    }
                }
                
                // Check prominence
                if let Some(min_prominence) = prominence {
                    let prom = Self::calculate_prominence(&spectrum_data, i);
                    if prom < min_prominence {
                        continue;
                    }
                }
                
                // Check width
                if let Some(min_width) = width {
                    let peak_width = Self::calculate_peak_width(&spectrum_data, &freq_data, i);
                    if peak_width < min_width {
                        continue;
                    }
                }
                
                peak_indices.push(i);
            }
        }
        
        // Extract peak information
        let peak_frequencies: Vec<T> = peak_indices.iter().map(|&i| freq_data[i]).collect();
        let peak_heights: Vec<T> = peak_indices.iter().map(|&i| spectrum_data[i]).collect();
        let peak_prominences: Vec<T> = peak_indices.iter()
            .map(|&i| Self::calculate_prominence(&spectrum_data, i))
            .collect();
        let peak_widths: Vec<T> = peak_indices.iter()
            .map(|&i| Self::calculate_peak_width(&spectrum_data, &freq_data, i))
            .collect();
        
        Ok(PeakResult {
            peak_indices: Array::from_vec(peak_indices.iter().map(|&i| i as f64).collect()),
            peak_frequencies: Array::from_vec(peak_frequencies),
            peak_heights: Array::from_vec(peak_heights),
            peak_prominences: Array::from_vec(peak_prominences),
            peak_widths: Array::from_vec(peak_widths),
        })
    }

    /// Calculate peak prominence
    fn calculate_prominence<T>(spectrum: &[T], peak_idx: usize) -> T
    where
        T: Float + Clone,
    {
        let peak_height = spectrum[peak_idx];
        
        // Find minimum heights on both sides
        let mut left_min = peak_height;
        let mut right_min = peak_height;
        
        // Search left
        for i in (0..peak_idx).rev() {
            left_min = left_min.min(spectrum[i]);
            if spectrum[i] > peak_height {
                break;
            }
        }
        
        // Search right
        for i in (peak_idx + 1)..spectrum.len() {
            right_min = right_min.min(spectrum[i]);
            if spectrum[i] > peak_height {
                break;
            }
        }
        
        let base_height = left_min.max(right_min);
        peak_height - base_height
    }

    /// Calculate peak width at half maximum
    fn calculate_peak_width<T>(spectrum: &[T], frequencies: &[T], peak_idx: usize) -> T
    where
        T: Float + Clone,
    {
        let peak_height = spectrum[peak_idx];
        let half_height = peak_height / T::from(2.0).unwrap();
        
        // Find left half-maximum point
        let mut left_idx = peak_idx;
        for i in (0..peak_idx).rev() {
            if spectrum[i] <= half_height {
                left_idx = i;
                break;
            }
        }
        
        // Find right half-maximum point
        let mut right_idx = peak_idx;
        for i in (peak_idx + 1)..spectrum.len() {
            if spectrum[i] <= half_height {
                right_idx = i;
                break;
            }
        }
        
        if right_idx > left_idx {
            frequencies[right_idx] - frequencies[left_idx]
        } else {
            T::zero()
        }
    }

    /// Perform harmonic analysis to detect fundamental frequency and harmonics
    pub fn harmonic_analysis<T>(
        spectrum: &Array<T>,
        frequencies: &Array<T>,
        max_harmonics: usize,
        tolerance: T,
    ) -> Result<HarmonicResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        // Find peaks first
        let peaks = Self::find_peaks(spectrum, frequencies, None, Some(2), None, None)?;
        let peak_freqs = peaks.peak_frequencies.to_vec();
        let peak_heights = peaks.peak_heights.to_vec();
        
        if peak_freqs.is_empty() {
            return Ok(HarmonicResult {
                fundamental_frequency: T::zero(),
                harmonics: Array::from_vec(Vec::new()),
                harmonic_amplitudes: Array::from_vec(Vec::new()),
                total_harmonic_distortion: T::zero(),
            });
        }
        
        // Try each peak as potential fundamental
        let mut best_fundamental = T::zero();
        let mut best_score = T::zero();
        let mut best_harmonics = Vec::new();
        let mut best_amplitudes = Vec::new();
        
        for (i, &candidate_f0) in peak_freqs.iter().enumerate() {
            if candidate_f0 <= T::zero() {
                continue;
            }
            
            let mut harmonics = vec![candidate_f0];
            let mut amplitudes = vec![peak_heights[i]];
            let mut score = peak_heights[i];
            
            // Look for harmonics
            for h in 2..=max_harmonics {
                let harmonic_freq = candidate_f0 * <T as NumCast>::from(h as f64).unwrap();
                
                // Find closest peak to this harmonic frequency
                let mut closest_idx = None;
                let mut min_distance = T::infinity();
                
                for (j, &freq) in peak_freqs.iter().enumerate() {
                    let distance = (freq - harmonic_freq).abs();
                    if distance < min_distance && distance < tolerance {
                        min_distance = distance;
                        closest_idx = Some(j);
                    }
                }
                
                if let Some(idx) = closest_idx {
                    harmonics.push(peak_freqs[idx]);
                    amplitudes.push(peak_heights[idx]);
                    score = score + peak_heights[idx] / <T as NumCast>::from(h as f64).unwrap();
                }
            }
            
            if score > best_score {
                best_score = score;
                best_fundamental = candidate_f0;
                best_harmonics = harmonics;
                best_amplitudes = amplitudes;
            }
        }
        
        // Calculate Total Harmonic Distortion (THD)
        let thd = if best_amplitudes.len() > 1 {
            let fundamental_power = best_amplitudes[0] * best_amplitudes[0];
            let harmonic_power: T = best_amplitudes[1..].iter()
                .map(|&a| a * a)
                .sum();
            
            if fundamental_power > T::zero() {
                (harmonic_power / fundamental_power).sqrt()
            } else {
                T::zero()
            }
        } else {
            T::zero()
        };
        
        Ok(HarmonicResult {
            fundamental_frequency: best_fundamental,
            harmonics: Array::from_vec(best_harmonics),
            harmonic_amplitudes: Array::from_vec(best_amplitudes),
            total_harmonic_distortion: thd,
        })
    }

    /// Compute Short-Time Fourier Transform (STFT)
    pub fn stft<T>(
        signal: &Array<T>,
        window_size: usize,
        hop_size: usize,
        window_type: &str,
        zero_pad: bool,
    ) -> Result<STFTResult<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        let signal_data = signal.to_vec();
        let n = signal_data.len();
        
        let n_frames = if n >= window_size {
            (n - window_size) / hop_size + 1
        } else {
            1
        };
        
        let fft_size = if zero_pad { 
            Self::next_power_of_2(window_size * 2) 
        } else { 
            window_size 
        };
        let n_freqs = fft_size / 2 + 1;
        
        // Generate window
        let window = FrequencyAnalyzer::generate_window_function(window_size, window_type)?;
        
        let mut stft_data = Vec::with_capacity(n_frames * n_freqs);
        let mut time_axis = Vec::with_capacity(n_frames);
        
        for frame in 0..n_frames {
            let start = frame * hop_size;
            let end = (start + window_size).min(n);
            
            if end - start < window_size {
                break; // Skip incomplete frames
            }
            
            // Extract window
            let mut windowed_data: Vec<T> = signal_data[start..end].to_vec();
            
            // Apply window
            for (i, &window_val) in window.iter().enumerate() {
                windowed_data[i] = windowed_data[i] * window_val;
            }
            
            // Zero-pad if requested
            if zero_pad {
                windowed_data.resize(fft_size, T::zero());
            }
            
            // Compute FFT
            let windowed_array = Array::from_vec(windowed_data);
            let fft_result = FFT::fft(&windowed_array)?;
            let fft_data = fft_result.to_vec();
            
            // Extract positive frequencies
            stft_data.extend_from_slice(&fft_data[..n_freqs]);
            
            // Time stamp for this frame
            let time = <T as NumCast>::from(start as f64 + window_size as f64 / 2.0).unwrap_or(T::zero());
            time_axis.push(time);
        }
        
        // Generate frequency axis
        let freq_axis = FFT::rfftfreq(fft_size, T::one())?;
        
        Ok(STFTResult {
            stft: Array::from_vec(stft_data).reshape(&[time_axis.len(), n_freqs]),
            time_axis: Array::from_vec(time_axis),
            freq_axis,
        })
    }

    /// Compute instantaneous frequency using phase differences
    pub fn instantaneous_frequency<T>(complex_signal: &Array<Complex<T>>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        let signal_data = complex_signal.to_vec();
        let n = signal_data.len();
        
        if n < 2 {
            return Err(NumRs2Error::InvalidOperation(
                "Signal too short for instantaneous frequency".to_string()
            ));
        }
        
        let mut inst_freq = Vec::with_capacity(n - 1);
        
        for i in 0..n-1 {
            let phase1 = signal_data[i].arg();
            let phase2 = signal_data[i + 1].arg();
            
            // Unwrap phase difference
            let mut phase_diff = phase2 - phase1;
            let pi = <T as NumCast>::from(PI).unwrap_or(T::zero());
            let two_pi = <T as NumCast>::from(2.0 * PI).unwrap_or(T::one());
            
            if phase_diff > pi {
                phase_diff = phase_diff - two_pi;
            } else if phase_diff < -pi {
                phase_diff = phase_diff + two_pi;
            }
            
            // Convert to frequency
            let freq = phase_diff / two_pi;
            inst_freq.push(freq);
        }
        
        Ok(Array::from_vec(inst_freq))
    }

    /// Compute spectral centroid (center of mass of spectrum)
    pub fn spectral_centroid<T>(spectrum: &Array<T>, frequencies: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug + std::iter::Sum,
    {
        let spectrum_data = spectrum.to_vec();
        let freq_data = frequencies.to_vec();
        
        if spectrum_data.len() != freq_data.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Spectrum and frequencies must have same length".to_string()
            ));
        }
        
        let weighted_sum: T = spectrum_data.iter().zip(freq_data.iter())
            .map(|(&s, &f)| s * f)
            .sum();
        
        let total_power: T = spectrum_data.iter().cloned().sum();
        
        if total_power > T::zero() {
            Ok(weighted_sum / total_power)
        } else {
            Ok(T::zero())
        }
    }

    /// Compute spectral rolloff (frequency below which a specified percentage of total energy is contained)
    pub fn spectral_rolloff<T>(spectrum: &Array<T>, frequencies: &Array<T>, percentage: T) -> Result<T>
    where
        T: Float + Clone + Debug + std::iter::Sum,
    {
        let spectrum_data = spectrum.to_vec();
        let freq_data = frequencies.to_vec();
        
        if spectrum_data.len() != freq_data.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Spectrum and frequencies must have same length".to_string()
            ));
        }
        
        let total_energy: T = spectrum_data.iter().cloned().sum();
        let threshold = total_energy * percentage;
        
        let mut cumulative_energy = T::zero();
        
        for (i, &power) in spectrum_data.iter().enumerate() {
            cumulative_energy = cumulative_energy + power;
            if cumulative_energy >= threshold {
                return Ok(freq_data[i]);
            }
        }
        
        // If we reach here, return the highest frequency
        Ok(freq_data[freq_data.len() - 1])
    }

    /// Compute spectral bandwidth
    pub fn spectral_bandwidth<T>(spectrum: &Array<T>, frequencies: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug + std::iter::Sum,
    {
        let centroid = Self::spectral_centroid(spectrum, frequencies)?;
        let spectrum_data = spectrum.to_vec();
        let freq_data = frequencies.to_vec();
        
        let weighted_variance: T = spectrum_data.iter().zip(freq_data.iter())
            .map(|(&s, &f)| s * (f - centroid) * (f - centroid))
            .sum();
        
        let total_power: T = spectrum_data.iter().cloned().sum();
        
        if total_power > T::zero() {
            Ok((weighted_variance / total_power).sqrt())
        } else {
            Ok(T::zero())
        }
    }

    /// Compute spectral flatness (Wiener entropy)
    pub fn spectral_flatness<T>(spectrum: &Array<T>) -> Result<T>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        let spectrum_data = spectrum.to_vec();
        let n = spectrum_data.len();
        
        if n == 0 {
            return Ok(T::zero());
        }
        
        // Avoid log(0) by adding small epsilon
        let epsilon = <T as NumCast>::from(1e-10).unwrap_or(T::zero());
        
        let geometric_mean = {
            let log_sum: f64 = spectrum_data.iter()
                .map(|&x| (x + epsilon).into())
                .map(|x: f64| x.ln())
                .sum();
            (log_sum / n as f64).exp()
        };
        
        let arithmetic_mean: f64 = spectrum_data.iter()
            .map(|&x| x.into())
            .sum::<f64>() / n as f64;
        
        if arithmetic_mean > 0.0 {
            Ok(<T as NumCast>::from(geometric_mean / arithmetic_mean).unwrap_or(T::zero()))
        } else {
            Ok(T::zero())
        }
    }

    /// Compute Mel-frequency cepstral coefficients (MFCCs)
    pub fn mfcc<T>(
        spectrum: &Array<T>,
        sample_rate: T,
        n_mfcc: usize,
        n_mels: usize,
        fmin: T,
        fmax: T,
    ) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        let spectrum_data = spectrum.to_vec();
        let n_freqs = spectrum_data.len();
        
        // Create mel filter bank
        let mel_filters = Self::create_mel_filterbank(n_freqs, n_mels, sample_rate, fmin, fmax)?;
        
        // Apply mel filters to spectrum
        let mut mel_spectrum = Vec::with_capacity(n_mels);
        for mel_filter in &mel_filters {
            let mel_energy: T = spectrum_data.iter().zip(mel_filter.iter())
                .map(|(&s, &f)| s * f)
                .sum();
            mel_spectrum.push(mel_energy);
        }
        
        // Apply logarithm
        let epsilon = <T as NumCast>::from(1e-10).unwrap_or(T::zero());
        let log_mel_spectrum: Vec<f64> = mel_spectrum.iter()
            .map(|&x| (x + epsilon).into())
            .map(|x: f64| x.ln())
            .collect();
        
        // Apply DCT (Discrete Cosine Transform)
        let mut mfccs = Vec::with_capacity(n_mfcc);
        for k in 0..n_mfcc {
            let mut sum = 0.0;
            for (n, &log_mel) in log_mel_spectrum.iter().enumerate() {
                let arg = PI * k as f64 * (n as f64 + 0.5) / n_mels as f64;
                sum += log_mel * arg.cos();
            }
            
            let normalization = if k == 0 { 
                (1.0 / n_mels as f64).sqrt() 
            } else { 
                (2.0 / n_mels as f64).sqrt() 
            };
            
            mfccs.push(<T as NumCast>::from(sum * normalization).unwrap_or(T::zero()));
        }
        
        Ok(Array::from_vec(mfccs))
    }

    /// Create mel-frequency filter bank
    fn create_mel_filterbank<T>(
        n_freqs: usize,
        n_mels: usize,
        sample_rate: T,
        fmin: T,
        fmax: T,
    ) -> Result<Vec<Vec<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64> + std::iter::Sum,
    {
        // Convert to mel scale
        let mel_min = Self::hz_to_mel(fmin.into());
        let mel_max = Self::hz_to_mel(fmax.into());
        
        // Create mel points
        let mut mel_points = Vec::with_capacity(n_mels + 2);
        for i in 0..=n_mels + 1 {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_points.push(Self::mel_to_hz(mel));
        }
        
        // Convert mel points to frequency bins
        let freq_bin_width = sample_rate.into() / (2.0 * (n_freqs - 1) as f64);
        let mut bin_points = Vec::with_capacity(mel_points.len());
        for &freq in &mel_points {
            bin_points.push((freq / freq_bin_width).round() as usize);
        }
        
        // Create filter bank
        let mut filters = Vec::with_capacity(n_mels);
        for i in 0..n_mels {
            let mut filter = vec![T::zero(); n_freqs];
            
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];
            
            // Rising edge
            for j in left..center {
                if j < n_freqs {
                    let val = (j - left) as f64 / (center - left) as f64;
                    filter[j] = <T as NumCast>::from(val).unwrap_or(T::zero());
                }
            }
            
            // Falling edge
            for j in center..right {
                if j < n_freqs {
                    let val = (right - j) as f64 / (right - center) as f64;
                    filter[j] = <T as NumCast>::from(val).unwrap_or(T::zero());
                }
            }
            
            filters.push(filter);
        }
        
        Ok(filters)
    }

    /// Convert frequency in Hz to mel scale
    fn hz_to_mel(freq_hz: f64) -> f64 {
        2595.0 * (1.0 + freq_hz / 700.0).log10()
    }

    /// Convert mel scale to frequency in Hz
    fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
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

/// Result of peak detection
#[derive(Debug)]
pub struct PeakResult<T: Clone> {
    pub peak_indices: Array<f64>,
    pub peak_frequencies: Array<T>,
    pub peak_heights: Array<T>,
    pub peak_prominences: Array<T>,
    pub peak_widths: Array<T>,
}

/// Result of harmonic analysis
#[derive(Debug)]
pub struct HarmonicResult<T: Clone> {
    pub fundamental_frequency: T,
    pub harmonics: Array<T>,
    pub harmonic_amplitudes: Array<T>,
    pub total_harmonic_distortion: T,
}

/// Result of Short-Time Fourier Transform
#[derive(Debug)]
pub struct STFTResult<T: Clone> {
    pub stft: Array<Complex<T>>,
    pub time_axis: Array<T>,
    pub freq_axis: Array<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_peak_detection() {
        // Create a spectrum with known peaks
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut spectrum = vec![0.1; 100];
        
        // Add peaks at indices 20, 40, 60
        spectrum[20] = 1.0;
        spectrum[40] = 0.8;
        spectrum[60] = 1.2;
        
        let spectrum_array = Array::from_vec(spectrum);
        let freq_array = Array::from_vec(freqs);
        
        let peaks = SpectralAnalyzer::find_peaks(
            &spectrum_array,
            &freq_array,
            Some(0.5), // height threshold
            Some(5),   // minimum distance
            None,
            None,
        ).unwrap();
        
        let peak_freqs = peaks.peak_frequencies.to_vec();
        let peak_heights = peaks.peak_heights.to_vec();
        
        // Should find 3 peaks
        assert_eq!(peak_freqs.len(), 3);
        assert_relative_eq!(peak_freqs[0], 2.0, epsilon = 1e-10); // Index 20 * 0.1
        assert_relative_eq!(peak_freqs[1], 4.0, epsilon = 1e-10); // Index 40 * 0.1
        assert_relative_eq!(peak_freqs[2], 6.0, epsilon = 1e-10); // Index 60 * 0.1
        
        assert_relative_eq!(peak_heights[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(peak_heights[1], 0.8, epsilon = 1e-10);
        assert_relative_eq!(peak_heights[2], 1.2, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_analysis() {
        // Create a spectrum with harmonic series: 100 Hz, 200 Hz, 300 Hz
        let freqs: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect(); // 0-500 Hz
        let mut spectrum = vec![0.01; 1000];
        
        // Add harmonics (indices: 200, 400, 600 for 100, 200, 300 Hz)
        spectrum[200] = 1.0;  // 100 Hz fundamental
        spectrum[400] = 0.7;  // 200 Hz (2nd harmonic)
        spectrum[600] = 0.5;  // 300 Hz (3rd harmonic)
        
        let spectrum_array = Array::from_vec(spectrum);
        let freq_array = Array::from_vec(freqs);
        
        let result = SpectralAnalyzer::harmonic_analysis(
            &spectrum_array,
            &freq_array,
            5,    // max harmonics
            5.0,  // tolerance
        ).unwrap();
        
        // Should detect 100 Hz as fundamental
        assert_relative_eq!(result.fundamental_frequency, 100.0, epsilon = 1.0);
        
        let harmonics = result.harmonics.to_vec();
        assert!(harmonics.len() >= 3);
        
        // Check that harmonics are detected
        assert_relative_eq!(harmonics[0], 100.0, epsilon = 1.0); // Fundamental
        // Allow some tolerance for harmonic detection
        let has_200hz = harmonics.iter().any(|&f| (f - 200.0).abs() < 5.0);
        let has_300hz = harmonics.iter().any(|&f| (f - 300.0).abs() < 5.0);
        assert!(has_200hz);
        assert!(has_300hz);
    }

    #[test]
    fn test_spectral_centroid() {
        // Create a simple spectrum
        let spectrum = Array::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let freqs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let centroid = SpectralAnalyzer::spectral_centroid(&spectrum, &freqs).unwrap();
        
        // Weighted average: (1*1 + 2*2 + 3*3 + 2*4 + 1*5) / (1+2+3+2+1) = 27/9 = 3.0
        assert_relative_eq!(centroid, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_rolloff() {
        let spectrum = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let freqs = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        
        // Total energy = 10, 85% = 8.5
        // Cumulative: 1, 3, 6, 10
        // 85% threshold (8.5) is reached at the last bin
        let rolloff = SpectralAnalyzer::spectral_rolloff(&spectrum, &freqs, 0.85).unwrap();
        
        assert_relative_eq!(rolloff, 40.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_bandwidth() {
        let spectrum = Array::from_vec(vec![1.0, 1.0, 1.0]);
        let freqs = Array::from_vec(vec![1.0, 2.0, 3.0]);
        
        let bandwidth = SpectralAnalyzer::spectral_bandwidth(&spectrum, &freqs).unwrap();
        
        // For uniform spectrum, centroid = 2.0
        // Variance = (1*(1-2)^2 + 1*(2-2)^2 + 1*(3-2)^2) / 3 = 2/3
        // Bandwidth = sqrt(2/3) ≈ 0.816
        assert_relative_eq!(bandwidth, (2.0_f64/3.0).sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn test_spectral_flatness() {
        // Test with uniform spectrum (should be close to 1.0)
        let uniform_spectrum = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let flatness = SpectralAnalyzer::spectral_flatness(&uniform_spectrum).unwrap();
        assert_relative_eq!(flatness, 1.0, epsilon = 1e-2);
        
        // Test with peaked spectrum (should be less than 1.0)
        let peaked_spectrum = Array::from_vec(vec![0.1, 10.0, 0.1, 0.1]);
        let flatness_peaked = SpectralAnalyzer::spectral_flatness(&peaked_spectrum).unwrap();
        assert!(flatness_peaked < 0.5);
    }

    #[test]
    fn test_instantaneous_frequency() {
        // Create a complex sinusoid with known frequency
        let n = 64;
        let freq = 0.1; // Normalized frequency
        let mut complex_signal = Vec::with_capacity(n);
        
        for i in 0..n {
            let phase = 2.0 * PI * freq * i as f64;
            complex_signal.push(Complex::new(phase.cos(), phase.sin()));
        }
        
        let signal_array = Array::from_vec(complex_signal);
        let inst_freq = SpectralAnalyzer::instantaneous_frequency(&signal_array).unwrap();
        let inst_freq_data = inst_freq.to_vec();
        
        // All instantaneous frequencies should be close to the original frequency
        for &f in &inst_freq_data {
            assert_relative_eq!(f, freq, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_stft() {
        // Create a chirp signal (frequency increases linearly)
        let n = 256;
        let mut signal = Vec::with_capacity(n);
        
        for i in 0..n {
            let t = i as f64 / n as f64;
            let freq = 5.0 + 10.0 * t; // Frequency increases from 5 to 15 Hz
            let phase = 2.0 * PI * freq * t;
            signal.push(phase.sin());
        }
        
        let signal_array = Array::from_vec(signal);
        let stft_result = SpectralAnalyzer::stft(
            &signal_array,
            64,    // window size
            32,    // hop size
            "hann",
            false, // no zero padding
        ).unwrap();
        
        // Check dimensions
        let stft_shape = stft_result.stft.shape();
        assert!(stft_shape[0] > 0); // Time frames
        assert_eq!(stft_shape[1], 33); // Frequency bins (64/2 + 1)
        
        assert_eq!(stft_result.time_axis.shape()[0], stft_shape[0]);
        assert_eq!(stft_result.freq_axis.shape()[0], stft_shape[1]);
    }

    #[test]
    fn test_mel_hz_conversion() {
        // Test known conversions
        let freq_hz = 1000.0;
        let mel = SpectralAnalyzer::hz_to_mel(freq_hz);
        let freq_back = SpectralAnalyzer::mel_to_hz(mel);
        
        assert_relative_eq!(freq_back, freq_hz, epsilon = 1e-6);
        
        // Test that 0 Hz maps to 0 mel
        assert_relative_eq!(SpectralAnalyzer::hz_to_mel(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(SpectralAnalyzer::mel_to_hz(0.0), 0.0, epsilon = 1e-10);
    }
}
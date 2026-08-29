//! Signal Processing Example for NumRS2
//!
//! This example demonstrates comprehensive signal processing including:
//! - FFT and spectral analysis
//! - Filtering (lowpass, highpass, bandpass, bandstop)
//! - Convolution and correlation
//! - Window functions
//! - Signal generation and manipulation
//! - Frequency domain operations
//!
//! Run with: cargo run --example signal_processing

use numrs2::prelude::*;
use numrs2::random::default_rng;
use scirs2_core::Complex;
use std::f64::consts::PI;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Signal Processing Examples ===\n");

    // Example 1: FFT and Spectral Analysis
    example1_fft_analysis()?;

    // Example 2: Window Functions
    example2_window_functions()?;

    // Example 3: Filtering
    example3_filtering()?;

    // Example 4: Convolution
    example4_convolution()?;

    // Example 5: Correlation
    example5_correlation()?;

    // Example 6: Signal Generation
    example6_signal_generation()?;

    // Example 7: Frequency Domain Operations
    example7_frequency_domain()?;

    println!("\n=== All Signal Processing Examples Completed Successfully! ===");
    Ok(())
}

/// Example 1: FFT and Spectral Analysis
fn example1_fft_analysis() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: FFT and Spectral Analysis");
    println!("=====================================\n");

    // Generate a composite signal: sum of sinusoids
    let sample_rate = 1000.0; // Hz
    let duration = 1.0; // seconds
    let n = (sample_rate * duration) as usize;

    // Signal components: 50 Hz + 120 Hz + 200 Hz
    let f1 = 50.0;
    let f2 = 120.0;
    let f3 = 200.0;

    let mut signal = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let value = (2.0 * PI * f1 * t).sin()
            + 0.5 * (2.0 * PI * f2 * t).sin()
            + 0.3 * (2.0 * PI * f3 * t).sin();
        signal.push(value);
    }

    let signal_array = Array::from_vec(signal);

    println!("1.1 Signal Characteristics");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {} s", duration);
    println!("  Samples: {}", n);
    println!("  Frequency components: {} Hz, {} Hz, {} Hz", f1, f2, f3);
    println!();

    // Compute FFT
    println!("1.2 FFT Computation");
    let fft_result = FFT::fft(&signal_array)?;

    println!("  FFT size: {}", fft_result.size());
    println!();

    // Compute power spectrum
    println!("1.3 Power Spectrum");
    let power_spectrum = FFT::power_spectrum(&signal_array)?;

    // Frequency axis
    let freq_axis = FFT::fftfreq(n, 1.0 / sample_rate)?;

    // Find peaks in power spectrum (first half only - positive frequencies)
    let half_n = n / 2;
    let mut peaks = Vec::new();

    for i in 1..(half_n - 1) {
        let prev_power = power_spectrum.get(&[i - 1])?;
        let curr_power = power_spectrum.get(&[i])?;
        let next_power = power_spectrum.get(&[i + 1])?;

        // Simple peak detection
        if curr_power > prev_power && curr_power > next_power && curr_power > 0.1 {
            let freq = freq_axis.get(&[i])?;
            peaks.push((freq, curr_power));
        }
    }

    // Sort peaks by power (descending)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top frequency peaks:");
    for (i, (freq, power)) in peaks.iter().take(5).enumerate() {
        println!("    Peak {}: {:.2} Hz, power: {:.6}", i + 1, freq, power);
    }
    println!();

    // Spectral density
    println!("1.4 Power Spectral Density");
    println!("  Total power: {:.6}", power_spectrum.sum());

    // Calculate power in frequency bands
    let bands = vec![
        ("Low (0-50 Hz)", 0.0, 50.0),
        ("Mid (50-150 Hz)", 50.0, 150.0),
        ("High (150-500 Hz)", 150.0, 500.0),
    ];

    for (name, f_low, f_high) in bands {
        let mut band_power = 0.0;
        for i in 0..half_n {
            let freq = freq_axis.get(&[i])?;
            if freq >= f_low && freq < f_high {
                band_power += power_spectrum.get(&[i])?;
            }
        }
        println!("  {}: {:.6}", name, band_power);
    }
    println!();

    println!("✓ Example 1 completed\n");
    Ok(())
}

/// Example 2: Window Functions
fn example2_window_functions() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Window Functions");
    println!("===========================\n");

    let n = 100;

    // Generate test signal (single sinusoid)
    let sample_rate = 100.0;
    let frequency = 10.0;

    let mut signal = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sample_rate;
        signal.push((2.0 * PI * frequency * t).sin());
    }

    let signal_array = Array::from_vec(signal);

    println!("2.1 Available Window Functions");
    let windows = vec!["hann", "hamming", "blackman", "bartlett"];

    for window_type in windows {
        match FFT::apply_window(&signal_array, window_type) {
            Ok(windowed) => {
                let energy = windowed.to_vec().iter().map(|&x| x * x).sum::<f64>();

                println!("  {} window:", window_type);
                println!("    Signal energy after windowing: {:.6}", energy);

                // Show first 5 values
                println!("    First 5 values:");
                for i in 0..5 {
                    println!("      windowed[{}]: {:.6}", i, windowed.get(&[i])?);
                }
                println!();
            }
            Err(e) => {
                println!("  Error applying {} window: {:?}", window_type, e);
            }
        }
    }

    // Compare spectral leakage with and without windowing
    println!("2.2 Spectral Leakage Reduction");

    // Without window
    let power_no_window = FFT::power_spectrum(&signal_array)?;

    // With Hann window
    let windowed = FFT::apply_window(&signal_array, "hann")?;
    let power_with_window = FFT::power_spectrum(&windowed)?;

    println!("  No window:");
    println!("    Total power: {:.6}", power_no_window.sum());
    println!("    Peak power: {:.6}", power_no_window.max());

    println!("  Hann window:");
    println!("    Total power: {:.6}", power_with_window.sum());
    println!("    Peak power: {:.6}", power_with_window.max());
    println!("    Note: Windowing reduces spectral leakage");
    println!();

    println!("✓ Example 2 completed\n");
    Ok(())
}

/// Example 3: Filtering
fn example3_filtering() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Filtering");
    println!("====================\n");

    let rng = default_rng();

    // Generate noisy signal
    let n = 500;
    let sample_rate = 1000.0;

    // Clean signal: 50 Hz sinusoid
    let signal_freq = 50.0;
    let mut clean_signal = Vec::with_capacity(n);
    let mut noisy_signal = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        let clean = (2.0 * PI * signal_freq * t).sin();
        let noise = rng.normal(0.0, 0.3, &[1])?.get(&[0])?;
        clean_signal.push(clean);
        noisy_signal.push(clean + noise);
    }

    let clean_array = Array::from_vec(clean_signal);
    let noisy_array = Array::from_vec(noisy_signal);

    println!("3.1 Signal Characteristics");
    println!(
        "  Clean signal - mean: {:.6}, std: {:.6}",
        clean_array.mean(),
        clean_array.std()
    );
    println!(
        "  Noisy signal - mean: {:.6}, std: {:.6}",
        noisy_array.mean(),
        noisy_array.std()
    );
    println!();

    // Simple moving average filter (lowpass)
    println!("3.2 Moving Average Filter (Lowpass)");

    let window_size = 5;
    let mut filtered = Vec::with_capacity(n - window_size + 1);

    for i in 0..=(n - window_size) {
        let mut sum = 0.0;
        for j in 0..window_size {
            sum += noisy_array.get(&[i + j])?;
        }
        filtered.push(sum / window_size as f64);
    }

    let filtered_array = Array::from_vec(filtered);

    println!("  Window size: {}", window_size);
    println!(
        "  Filtered signal - mean: {:.6}, std: {:.6}",
        filtered_array.mean(),
        filtered_array.std()
    );
    println!();

    // Gaussian filter
    println!("3.3 Gaussian Filter");

    let sigma = 2.0;
    let kernel_size = 11;
    let mut gaussian_kernel = Vec::with_capacity(kernel_size);

    let center = (kernel_size - 1) as f64 / 2.0;
    for i in 0..kernel_size {
        let x = i as f64 - center;
        let weight = (-x * x / (2.0 * sigma * sigma)).exp();
        gaussian_kernel.push(weight);
    }

    // Normalize kernel
    let kernel_sum: f64 = gaussian_kernel.iter().sum();
    for val in &mut gaussian_kernel {
        *val /= kernel_sum;
    }

    println!("  Gaussian kernel (σ={}):", sigma);
    println!("    Kernel: {:?}", &gaussian_kernel[0..5]);
    println!();

    // Apply Gaussian filter (simple convolution)
    let mut gaussian_filtered = Vec::with_capacity(n - kernel_size + 1);

    for i in 0..=(n - kernel_size) {
        let mut sum = 0.0;
        for (j, &kernel_val) in gaussian_kernel.iter().enumerate() {
            sum += noisy_array.get(&[i + j])? * kernel_val;
        }
        gaussian_filtered.push(sum);
    }

    let gaussian_filtered_array = Array::from_vec(gaussian_filtered);

    println!(
        "  Gaussian filtered signal - mean: {:.6}, std: {:.6}",
        gaussian_filtered_array.mean(),
        gaussian_filtered_array.std()
    );
    println!();

    // Calculate SNR improvement
    println!("3.4 Signal-to-Noise Ratio (SNR) Improvement");

    // Calculate noise power (original noisy signal)
    let mut noise_power = 0.0;
    for i in 0..n {
        let noise = noisy_array.get(&[i])? - clean_array.get(&[i])?;
        noise_power += noise * noise;
    }
    noise_power /= n as f64;

    // Calculate residual noise power (after filtering)
    let mut residual_noise_power = 0.0;
    let filtered_len = filtered_array.size();

    for i in 0..filtered_len {
        let residual = filtered_array.get(&[i])? - clean_array.get(&[i])?;
        residual_noise_power += residual * residual;
    }
    residual_noise_power /= filtered_len as f64;

    let snr_before = 10.0 * (1.0 / noise_power).log10();
    let snr_after = 10.0 * (1.0 / residual_noise_power).log10();

    println!("  SNR before filtering: {:.2} dB", snr_before);
    println!("  SNR after filtering: {:.2} dB", snr_after);
    println!("  SNR improvement: {:.2} dB", snr_after - snr_before);
    println!();

    println!("✓ Example 3 completed\n");
    Ok(())
}

/// Example 4: Convolution
fn example4_convolution() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Convolution");
    println!("======================\n");

    // 4.1 1D Convolution
    println!("4.1 One-Dimensional Convolution");

    let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let kernel = Array::from_vec(vec![1.0, 2.0, 1.0]); // Simple averaging kernel

    println!("  Signal: {:?}", signal.to_vec());
    println!("  Kernel: {:?}", kernel.to_vec());

    // Manual convolution (valid mode)
    let signal_len = signal.size();
    let kernel_len = kernel.size();
    let output_len = signal_len - kernel_len + 1;

    let mut convolved = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            sum += signal.get(&[i + j])? * kernel.get(&[kernel_len - 1 - j])?;
        }
        convolved.push(sum);
    }

    let convolved_array = Array::from_vec(convolved);
    println!("  Convolved (valid): {:?}", convolved_array.to_vec());
    println!();

    // 4.2 Edge Detection Kernel
    println!("4.2 Edge Detection (Derivative Approximation)");

    let signal = Array::from_vec(vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]);
    let edge_kernel = Array::from_vec(vec![-1.0, 0.0, 1.0]); // Derivative kernel

    println!("  Signal: {:?}", signal.to_vec());
    println!("  Edge kernel: {:?}", edge_kernel.to_vec());

    let signal_len = signal.size();
    let kernel_len = edge_kernel.size();
    let output_len = signal_len - kernel_len + 1;

    let mut edges = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            sum += signal.get(&[i + j])? * edge_kernel.get(&[kernel_len - 1 - j])?;
        }
        edges.push(sum);
    }

    let edges_array = Array::from_vec(edges);
    println!("  Edges detected: {:?}", edges_array.to_vec());
    println!("  Note: Large values indicate edges");
    println!();

    // 4.3 Impulse Response
    println!("4.3 Impulse Response");

    let impulse = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    let filter_kernel = Array::from_vec(vec![0.25, 0.5, 0.25]); // Gaussian-like

    println!("  Impulse: {:?}", impulse.to_vec());
    println!("  Filter: {:?}", filter_kernel.to_vec());

    let signal_len = impulse.size();
    let kernel_len = filter_kernel.size();
    let output_len = signal_len - kernel_len + 1;

    let mut response = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            sum += impulse.get(&[i + j])? * filter_kernel.get(&[kernel_len - 1 - j])?;
        }
        response.push(sum);
    }

    let response_array = Array::from_vec(response);
    println!("  Impulse response: {:?}", response_array.to_vec());
    println!();

    println!("✓ Example 4 completed\n");
    Ok(())
}

/// Example 5: Correlation
fn example5_correlation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Correlation");
    println!("======================\n");

    // 5.1 Auto-correlation
    println!("5.1 Auto-correlation");

    let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let n = signal.size();

    println!("  Signal: {:?}", signal.to_vec());

    // Compute auto-correlation for lags 0 to 4
    println!("  Auto-correlation:");

    let mean = signal.mean();

    for lag in 0..5 {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            let x1 = signal.get(&[i])? - mean;
            let x2 = signal.get(&[i + lag])? - mean;
            sum += x1 * x2;
        }

        let autocorr = sum / (count as f64);
        println!("    Lag {}: {:.6}", lag, autocorr);
    }
    println!();

    // 5.2 Cross-correlation
    println!("5.2 Cross-correlation");

    let signal1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let signal2 = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]);

    println!("  Signal 1: {:?}", signal1.to_vec());
    println!("  Signal 2: {:?}", signal2.to_vec());

    let n1 = signal1.size();
    let n2 = signal2.size();

    // Compute cross-correlation (valid mode)
    let mut cross_corr = Vec::new();

    for shift in 0..=(n2 - n1) {
        let mut sum = 0.0;
        for i in 0..n1 {
            sum += signal1.get(&[i])? * signal2.get(&[shift + i])?;
        }
        cross_corr.push(sum);
    }

    let cross_corr_array = Array::from_vec(cross_corr);
    println!("  Cross-correlation: {:?}", cross_corr_array.to_vec());

    // Find maximum correlation
    let max_corr = cross_corr_array.max();
    let max_idx = cross_corr_array
        .to_vec()
        .iter()
        .position(|&x: &f64| (x - max_corr).abs() < 1e-10)
        .unwrap();

    println!("  Maximum correlation: {:.6} at lag {}", max_corr, max_idx);
    println!();

    // 5.3 Signal Alignment
    println!("5.3 Signal Alignment via Cross-correlation");

    let rng = default_rng();

    // Create template signal
    let template_len = 10;
    let mut template = Vec::with_capacity(template_len);
    for i in 0..template_len {
        let t = i as f64 / template_len as f64;
        template.push((2.0 * PI * t).sin());
    }

    // Create longer signal with template embedded at position 20
    let signal_len = 50;
    let embed_pos = 20;
    let mut long_signal = vec![0.0; signal_len];

    long_signal[embed_pos..embed_pos + template_len].copy_from_slice(&template);

    // Add noise
    for val in &mut long_signal {
        *val += rng.normal(0.0, 0.1, &[1])?.get(&[0])?;
    }

    let template_array = Array::from_vec(template);
    let signal_array = Array::from_vec(long_signal);

    // Cross-correlate to find template
    let mut correlations = Vec::new();

    for shift in 0..=(signal_len - template_len) {
        let mut sum = 0.0;
        for i in 0..template_len {
            sum += template_array.get(&[i])? * signal_array.get(&[shift + i])?;
        }
        correlations.push(sum);
    }

    let corr_array = Array::from_vec(correlations);
    let detected_pos = corr_array
        .to_vec()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("  Template embedded at position: {}", embed_pos);
    println!("  Detected position: {}", detected_pos);
    println!(
        "  Detection accuracy: {} samples",
        (detected_pos as i32 - embed_pos as i32).abs()
    );
    println!();

    println!("✓ Example 5 completed\n");
    Ok(())
}

/// Example 6: Signal Generation
fn example6_signal_generation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 6: Signal Generation");
    println!("============================\n");

    let sample_rate = 1000.0;
    let duration = 1.0;
    let n = (sample_rate * duration) as usize;

    // 6.1 Sinusoidal Signals
    println!("6.1 Sinusoidal Signals");

    let freq = 10.0;
    let amplitude = 2.0;
    let phase = PI / 4.0;

    let mut sine_wave = Vec::with_capacity(n);
    let mut cosine_wave = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        sine_wave.push(amplitude * (2.0 * PI * freq * t + phase).sin());
        cosine_wave.push(amplitude * (2.0 * PI * freq * t + phase).cos());
    }

    let sine_array = Array::from_vec(sine_wave);
    let cosine_array = Array::from_vec(cosine_wave);

    println!("  Frequency: {} Hz", freq);
    println!("  Amplitude: {}", amplitude);
    println!("  Phase: {:.4} rad ({:.1}°)", phase, phase * 180.0 / PI);
    println!(
        "  Sine wave - mean: {:.6}, std: {:.6}",
        sine_array.mean(),
        sine_array.std()
    );
    println!(
        "  Cosine wave - mean: {:.6}, std: {:.6}",
        cosine_array.mean(),
        cosine_array.std()
    );
    println!();

    // 6.2 Square Wave
    println!("6.2 Square Wave");

    let freq = 5.0;
    let mut square_wave = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        let value = if (2.0 * PI * freq * t).sin() >= 0.0 {
            1.0
        } else {
            -1.0
        };
        square_wave.push(value);
    }

    let square_array = Array::from_vec(square_wave);

    println!("  Frequency: {} Hz", freq);
    println!("  Mean: {:.6}", square_array.mean());
    println!(
        "  Min: {:.2}, Max: {:.2}",
        square_array.min(),
        square_array.max()
    );
    println!();

    // 6.3 Sawtooth Wave
    println!("6.3 Sawtooth Wave");

    let freq = 5.0;
    let mut sawtooth = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        let phase = (freq * t) % 1.0;
        sawtooth.push(2.0 * phase - 1.0);
    }

    let sawtooth_array = Array::from_vec(sawtooth);

    println!("  Frequency: {} Hz", freq);
    println!("  Mean: {:.6}", sawtooth_array.mean());
    println!(
        "  Min: {:.2}, Max: {:.2}",
        sawtooth_array.min(),
        sawtooth_array.max()
    );
    println!();

    // 6.4 Chirp Signal (Frequency Sweep)
    println!("6.4 Chirp Signal (Linear Frequency Sweep)");

    let f_start = 10.0;
    let f_end = 100.0;
    let mut chirp = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        // Linear chirp: instantaneous frequency f(t) = f_start + (f_end -
        // f_start) * t / duration; `phase` below is its closed-form time
        // integral 2*pi*∫f(t)dt, so f(t) itself need not be computed here.
        let phase = 2.0 * PI * (f_start * t + 0.5 * (f_end - f_start) * t * t / duration);
        chirp.push(phase.sin());
    }

    let chirp_array = Array::from_vec(chirp);

    println!("  Start frequency: {} Hz", f_start);
    println!("  End frequency: {} Hz", f_end);
    println!("  Duration: {} s", duration);
    println!(
        "  Chirp - mean: {:.6}, std: {:.6}",
        chirp_array.mean(),
        chirp_array.std()
    );
    println!();

    println!("✓ Example 6 completed\n");
    Ok(())
}

/// Example 7: Frequency Domain Operations
fn example7_frequency_domain() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 7: Frequency Domain Operations");
    println!("=======================================\n");

    // 7.1 Frequency Shifting
    println!("7.1 Frequency Shifting (Modulation)");

    let sample_rate = 1000.0;
    let n = 1000;

    // Original signal at 50 Hz
    let signal_freq = 50.0;
    let mut original_signal = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;
        original_signal.push((2.0 * PI * signal_freq * t).sin());
    }

    // Carrier frequency for modulation
    let carrier_freq = 200.0;
    let mut modulated = Vec::with_capacity(n);

    for (i, &signal_val) in original_signal.iter().enumerate() {
        let t = i as f64 / sample_rate;
        let carrier = (2.0 * PI * carrier_freq * t).cos();
        modulated.push(signal_val * carrier);
    }

    let original_array = Array::from_vec(original_signal);
    let modulated_array = Array::from_vec(modulated);

    println!("  Original frequency: {} Hz", signal_freq);
    println!("  Carrier frequency: {} Hz", carrier_freq);
    println!(
        "  Expected modulated frequencies: {} Hz, {} Hz",
        carrier_freq - signal_freq,
        carrier_freq + signal_freq
    );

    // Analyze with FFT
    let power_original = FFT::power_spectrum(&original_array)?;
    let power_modulated = FFT::power_spectrum(&modulated_array)?;

    println!("  Original signal power: {:.6}", power_original.sum());
    println!("  Modulated signal power: {:.6}", power_modulated.sum());
    println!();

    // 7.2 Frequency Domain Filtering
    println!("7.2 Frequency Domain Filtering");

    let rng = default_rng();

    // Create noisy signal
    let mut noisy_signal = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let signal = (2.0 * PI * 50.0 * t).sin(); // 50 Hz signal
        let noise = rng.normal(0.0, 0.5, &[1])?.get(&[0])?;
        noisy_signal.push(signal + noise);
    }

    let noisy_array = Array::from_vec(noisy_signal);

    // Transform to frequency domain
    let fft_result = FFT::fft(&noisy_array)?;

    // Create frequency domain filter (bandpass around 50 Hz)
    let freq_axis = FFT::fftfreq(n, 1.0 / sample_rate)?;
    let target_freq = 50.0;
    let bandwidth = 10.0;

    let mut filtered_fft_data = Vec::with_capacity(n);
    for i in 0..n {
        let freq = freq_axis.get(&[i])?;
        let fft_value = fft_result.get(&[i])?;

        // Bandpass filter
        if (freq.abs() - target_freq).abs() < bandwidth {
            filtered_fft_data.push(fft_value);
        } else {
            filtered_fft_data.push(Complex::new(0.0, 0.0));
        }
    }

    let filtered_fft = Array::from_vec(filtered_fft_data);

    // Inverse FFT to get filtered signal
    let filtered_signal = FFT::ifft(&filtered_fft)?;

    // Calculate SNR improvement
    let power_noisy = noisy_array.to_vec().iter().map(|&x| x * x).sum::<f64>() / n as f64;
    let power_filtered = filtered_signal
        .to_vec()
        .iter()
        .map(|c| c.norm() * c.norm())
        .sum::<f64>()
        / n as f64;

    println!("  Filter center frequency: {} Hz", target_freq);
    println!("  Filter bandwidth: {} Hz", bandwidth);
    println!("  Noisy signal power: {:.6}", power_noisy);
    println!("  Filtered signal power: {:.6}", power_filtered);
    println!();

    println!("✓ Example 7 completed\n");
    Ok(())
}

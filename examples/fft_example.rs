use numrs2::prelude::*;

fn main() {
    println!("NumRS Fast Fourier Transform (FFT) Examples");
    println!("=========================================\n");

    // 1. Basic FFT example
    println!("1. Basic FFT Example");
    println!("-------------------");

    // Create a simple signal: [1.0, 0.0, 0.0, 0.0]
    let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    println!("Input signal: {:?}", signal.to_vec());

    // Compute FFT
    match FFT::fft(&signal) {
        Ok(fft_result) => {
            println!("FFT result: ");
            for (i, val) in fft_result.to_vec().iter().enumerate() {
                println!("  bin {}: {:.4} + {:.4}i", i, val.re, val.im);
            }

            // Compute inverse FFT to recover the original signal
            match FFT::ifft(&fft_result) {
                Ok(ifft_result) => {
                    println!("\nInverse FFT (recovered signal):");
                    for (i, val) in ifft_result.to_vec().iter().enumerate() {
                        println!("  {}: {:.4} + {:.4}i", i, val.re, val.im);
                    }
                }
                Err(e) => println!("Error computing inverse FFT: {}", e),
            }
        }
        Err(e) => println!("Error computing FFT: {}", e),
    }

    // 2. Analyzing a sinusoidal signal
    println!("\n2. Analyzing a Sinusoidal Signal");
    println!("-------------------------------");

    // Create a sinusoidal signal: sin(2π * f * t)
    let sample_rate = 32.0; // Hz
    let duration = 1.0; // second
    let frequency = 5.0; // Hz
    let n = (sample_rate * duration) as usize;

    let mut signal = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin();
        signal.push(sample);
    }

    let signal_array = Array::from_vec(signal);
    println!(
        "Created a {} Hz sinusoidal signal with {} samples",
        frequency, n
    );

    // Apply a window function to reduce spectral leakage
    let windowed_signal = match FFT::apply_window(&signal_array, "hann") {
        Ok(windowed) => windowed,
        Err(e) => {
            println!("Error applying window: {}", e);
            return;
        }
    };

    // Compute FFT
    match FFT::fft(&windowed_signal) {
        Ok(_fft_result) => {
            // Compute power spectrum (|FFT|²)
            match FFT::power_spectrum(&windowed_signal) {
                Ok(power) => {
                    println!("\nPower spectrum:");
                    // Only show the first half (positive frequencies)
                    let power_data = power.to_vec();

                    // Compute frequency axis
                    match FFT::fftfreq(n, 1.0 / sample_rate) {
                        Ok(freq_axis) => {
                            let freq_data = freq_axis.to_vec();

                            // Print the first half (positive frequencies)
                            for i in 0..(n / 2) {
                                println!("  {:.2} Hz: {:.6}", freq_data[i], power_data[i]);
                            }

                            // Find and print the peak frequency
                            let mut max_power = 0.0;
                            let mut max_idx = 0;

                            for i in 0..(n / 2) {
                                if power_data[i] > max_power {
                                    max_power = power_data[i];
                                    max_idx = i;
                                }
                            }

                            println!("\nPeak frequency detected at {:.2} Hz", freq_data[max_idx]);
                            println!("Expected frequency: {:.2} Hz", frequency);
                        }
                        Err(e) => println!("Error computing frequency axis: {}", e),
                    }
                }
                Err(e) => println!("Error computing power spectrum: {}", e),
            }
        }
        Err(e) => println!("Error computing FFT: {}", e),
    }

    // 3. 2D FFT example with a simple image
    println!("\n3. 2D FFT Example");
    println!("-----------------");

    // Create a simple 8x8 "image" with a pattern
    let mut image_data = vec![0.0; 64];

    // Add a horizontal line
    for i in 24..40 {
        image_data[i] = 1.0;
    }

    let image = Array::from_vec(image_data).reshape(&[8, 8]);
    println!("Original 8x8 image (with horizontal line):");
    print_2d_array(&image);

    // Compute 2D FFT
    match FFT::fft2(&image) {
        Ok(fft_result) => {
            println!("\n2D FFT magnitude (log scale):");

            // Compute magnitude and apply log scaling for visualization
            let mut magnitude = Vec::with_capacity(64);
            for val in fft_result.to_vec() {
                // Add small value to avoid log(0)
                let mag = (val.norm() + 1e-10).ln();
                magnitude.push(mag);
            }

            let magnitude_array = Array::from_vec(magnitude).reshape(&[8, 8]);
            print_2d_array(&magnitude_array);

            // Compute inverse 2D FFT to recover the original image
            match FFT::ifft2(&fft_result) {
                Ok(ifft_result) => {
                    println!("\nInverse 2D FFT (real part, recovered image):");

                    // Extract real part
                    let mut real_part = Vec::with_capacity(64);
                    for val in ifft_result.to_vec() {
                        real_part.push(val.re);
                    }

                    let recovered = Array::from_vec(real_part).reshape(&[8, 8]);
                    print_2d_array(&recovered);
                }
                Err(e) => println!("Error computing inverse 2D FFT: {}", e),
            }
        }
        Err(e) => println!("Error computing 2D FFT: {}", e),
    }
}

// Helper function to print 2D arrays
fn print_2d_array(array: &Array<f64>) {
    let shape = array.shape();
    if shape.len() != 2 {
        println!("Not a 2D array");
        return;
    }

    let rows = shape[0];
    let cols = shape[1];
    let data = array.to_vec();

    for i in 0..rows {
        for j in 0..cols {
            let val = data[i * cols + j];
            // Format with fixed width for better visualization
            print!("{:6.2} ", val);
        }
        println!();
    }
}

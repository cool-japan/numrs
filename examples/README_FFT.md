# NumRS Fast Fourier Transform (FFT) Module

The NumRS FFT module provides efficient implementations of the Fast Fourier Transform and related algorithms for signal processing and spectral analysis. The module is designed to be both performant and easy to use, with an API that will be familiar to NumPy users.

## Key Features

- Forward and inverse FFT transformations
- Real FFT for improved performance on real-valued data
- Multi-dimensional FFT (1D, 2D, and N-D)
- Fast cosine and sine transforms
- Frequency domain filtering
- Windowing functions (Hann, Hamming, Blackman, etc.)
- Spectral analysis utilities

## Basic Usage

### 1D FFT

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, ifft};

// Create a simple signal
let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

// Compute the FFT
let spectrum = fft(&signal)?;
println!("FFT result: {}", spectrum);

// Compute the magnitude and phase
let magnitude = spectrum.abs();
let phase = spectrum.angle();
println!("Magnitude: {}", magnitude);
println!("Phase: {}", phase);

// Compute the inverse FFT to recover the original signal
let recovered = ifft(&spectrum)?;
println!("Recovered signal: {}", recovered);
```

### Real FFT

```rust
use numrs2::prelude::*;
use numrs2::fft::{rfft, irfft};

// Create a real-valued signal
let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

// Compute the real FFT (more efficient than regular FFT for real inputs)
let spectrum = rfft(&signal)?;
println!("RFFT result: {}", spectrum);

// The result has (n/2)+1 complex values (for even-length input)
println!("RFFT result length: {}", spectrum.size());

// Compute the inverse real FFT to recover the original signal
let recovered = irfft(&spectrum, Some(signal.size()))?;
println!("Recovered signal: {}", recovered);
```

### 2D FFT

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft2, ifft2};

// Create a 2D signal/image
let image = Array::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
).reshape(&[3, 3]);

// Compute the 2D FFT
let spectrum_2d = fft2(&image)?;
println!("2D FFT result: {}", spectrum_2d);

// Compute the magnitude spectrum (useful for image processing)
let magnitude_spectrum = spectrum_2d.abs();
println!("2D Magnitude Spectrum: {}", magnitude_spectrum);

// Compute the inverse 2D FFT to recover the original image
let recovered_image = ifft2(&spectrum_2d)?;
println!("Recovered image: {}", recovered_image);
```

### N-Dimensional FFT

```rust
use numrs2::prelude::*;
use numrs2::fft::{fftn, ifftn};

// Create a 3D array
let volume = Array::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
).reshape(&[2, 2, 2]);

// Compute the N-dimensional FFT
let spectrum_nd = fftn(&volume, None)?;
println!("3D FFT result: {}", spectrum_nd);

// Compute the inverse N-dimensional FFT
let recovered_volume = ifftn(&spectrum_nd, None)?;
println!("Recovered volume: {}", recovered_volume);
```

## Frequency Domain Manipulation

### Shifting the Zero Frequency Component

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, fftshift, ifftshift};

// Create a signal and compute its FFT
let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
let spectrum = fft(&signal)?;

// By default, the zero frequency component is at the beginning
println!("Original spectrum: {}", spectrum);

// Shift the zero frequency component to the center
let shifted = fftshift(&spectrum)?;
println!("Shifted spectrum: {}", shifted);

// Shift back (useful before applying inverse FFT)
let unshifted = ifftshift(&shifted)?;
println!("Unshifted spectrum: {}", unshifted);
```

### Frequency Filtering

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, ifft, fftfreq};

// Create a noisy signal (e.g., sinusoid with high-frequency noise)
let t = Array::linspace(0.0, 1.0, 100)?; // Time vector
let signal = t.sin_pi().multiply_scalar(2.0 * std::f64::consts::PI); // Base signal
let noise = rng.normal(0.0, 0.2, &[100])?; // Noise
let noisy_signal = signal.add(&noise);

// Compute the FFT
let spectrum = fft(&noisy_signal)?;

// Get frequency bins
let freqs = fftfreq(noisy_signal.size(), 1.0/100.0)?;

// Create a low-pass filter (mask high frequencies)
let mut filter = Array::ones(&[spectrum.size()]);
for (i, &freq) in freqs.to_vec().iter().enumerate() {
    if freq.abs() > 10.0 { // Cut-off frequency
        filter.set(&[i], 0.0)?;
    }
}

// Apply the filter
let filtered_spectrum = spectrum.multiply(&filter);

// Transform back to time domain
let filtered_signal = ifft(&filtered_spectrum)?;
println!("Filtered signal: {}", filtered_signal);
```

## Windowing Functions

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, window};

// Create a signal
let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]);

// Apply a Hanning window
let hann_window = window::hann(signal.size());
let windowed_signal = signal.multiply(&hann_window);
println!("Hann windowed signal: {}", windowed_signal);

// Apply a Hamming window
let hamming_window = window::hamming(signal.size());
let windowed_signal2 = signal.multiply(&hamming_window);
println!("Hamming windowed signal: {}", windowed_signal2);

// Other available windows:
// - blackman
// - bartlett
// - kaiser (with beta parameter)
// - gaussian (with standard deviation parameter)
// - rectangular (no windowing)

// Compute FFT of windowed signal for better spectral estimation
let spectrum = fft(&windowed_signal)?;
println!("FFT of windowed signal: {}", spectrum);
```

## Spectral Analysis

### Power Spectral Density

```rust
use numrs2::prelude::*;
use numrs2::fft::{fft, power_spectral_density, fftfreq};

// Create a signal
let fs = 1000.0; // Sampling frequency in Hz
let t = Array::linspace(0.0, 1.0, 1000)?; // 1 second of data
let f1 = 50.0; // First frequency component in Hz
let f2 = 120.0; // Second frequency component in Hz
let signal = t.sin_pi().multiply_scalar(2.0 * std::f64::consts::PI * f1)
  .add(&t.sin_pi().multiply_scalar(2.0 * std::f64::consts::PI * f2));

// Compute the FFT
let spectrum = fft(&signal)?;

// Compute power spectral density
let psd = power_spectral_density(&spectrum, fs)?;

// Get frequency bins
let freqs = fftfreq(signal.size(), 1.0/fs)?;

// Find peaks in PSD (simple approach)
for i in 1..(psd.size()-1) {
    if psd.get(&[i])? > psd.get(&[i-1])? && psd.get(&[i])? > psd.get(&[i+1])? {
        println!("Peak at frequency {:.2} Hz with power {:.6}", freqs.get(&[i])?, psd.get(&[i])?);
    }
}
```

### Spectrogram

```rust
use numrs2::prelude::*;
use numrs2::fft::{spectrogram, window};

// Create a chirp signal (frequency increases with time)
let fs = 1000.0; // Sampling frequency in Hz
let t = Array::linspace(0.0, 10.0, 10000)?; // 10 seconds of data
let f0 = 10.0; // Starting frequency in Hz
let f1 = 100.0; // Ending frequency in Hz
let chirp = (t.multiply_scalar(f0 + (f1 - f0) * t.divide_scalar(10.0))
    .multiply_scalar(2.0 * std::f64::consts::PI)
    .cumsum(None)?
    .divide_scalar(fs)).sin();

// Compute spectrogram with Hann window, 256 sample segments, and 128 sample overlap
let window_size = 256;
let hop_size = 128;
let window_fn = window::hann(window_size);

let (spec, frequencies, times) = spectrogram(&chirp, fs, window_size, hop_size, &window_fn)?;
println!("Spectrogram shape: {:?}", spec.shape());
println!("Frequency resolution: {:.2} Hz", frequencies.get(&[1])? - frequencies.get(&[0])?);
println!("Time resolution: {:.3} s", times.get(&[1])? - times.get(&[0])?);
```

## Fast Sine and Cosine Transforms

```rust
use numrs2::prelude::*;
use numrs2::fft::{dct, idct, dst, idst};

// Create a signal
let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

// Compute discrete cosine transform (DCT)
let dct_result = dct(&signal, Some("ortho"))?;
println!("DCT result: {}", dct_result);

// Compute inverse DCT
let idct_result = idct(&dct_result, Some("ortho"))?;
println!("Recovered signal from DCT: {}", idct_result);

// Compute discrete sine transform (DST)
let dst_result = dst(&signal, Some("ortho"))?;
println!("DST result: {}", dst_result);

// Compute inverse DST
let idst_result = idst(&dst_result, Some("ortho"))?;
println!("Recovered signal from DST: {}", idst_result);
```

## Hilbert Transform

```rust
use numrs2::prelude::*;
use numrs2::fft::hilbert;

// Create a signal
let t = Array::linspace(0.0, 1.0, 1000)?;
let signal = t.sin_pi().multiply_scalar(2.0 * std::f64::consts::PI * 5.0); // 5 Hz sine wave

// Compute Hilbert transform
let hilbert_result = hilbert(&signal)?;

// The result is complex with:
// - Real part = original signal
// - Imaginary part = 90-degree phase shifted version

// Extract amplitude envelope
let envelope = hilbert_result.abs();
println!("Signal envelope: {}", envelope);

// Extract instantaneous phase
let inst_phase = hilbert_result.angle();
println!("Instantaneous phase: {}", inst_phase);

// Extract instantaneous frequency (derivative of phase)
let inst_freq = inst_phase.diff(0)?.multiply_scalar(1000.0 / (2.0 * std::f64::consts::PI));
println!("Instantaneous frequency: {}", inst_freq);
```

## Examples

For detailed examples of FFT operations, see:
- `fft_example.rs`: Basic FFT operations and signal processing
- `spectral_analysis_example.rs`: Advanced spectral analysis techniques

## Implementation Details

- NumRS uses efficient FFT algorithms with O(n log n) complexity
- The implementation leverages bit-level optimizations where possible
- Multi-dimensional FFTs use optimized row-column decomposition
- Real FFTs exploit symmetry for improved performance
- Operations support both CPU and (where available) GPU acceleration
- The module is designed for numerical stability and accuracy
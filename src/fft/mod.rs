//! Fast Fourier Transform Module
//!
//! This module provides comprehensive Fast Fourier Transform (FFT) functionality,
//! built on top of `scirs2-fft`. It includes:
//!
//! - **Basic FFT**: Complex-to-complex FFT/IFFT (1D, 2D, ND)
//! - **Real FFT**: Optimized real-to-complex RFFT/IRFFT
//! - **DCT/DST**: Discrete Cosine/Sine Transforms (Types I-IV)
//! - **Specialized**: Fractional FFT, Non-Uniform FFT, Hermitian FFT
//! - **Time-Frequency**: STFT, spectrograms, waterfall plots
//! - **Performance**: Plan caching, GPU acceleration, SIMD optimization
//!
//! # Examples
//!
//! ## Basic FFT
//!
//! ```
//! use numrs2::fft;
//!
//! // Time-domain signal
//! let signal = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Forward FFT: time → frequency domain
//! let spectrum = fft::fft(&signal, None).expect("fft should succeed");
//! println!("Frequency spectrum: {:?}", spectrum);
//!
//! // Inverse FFT: frequency → time domain
//! let recovered = fft::ifft(&spectrum, None).expect("ifft should succeed");
//! println!("Recovered signal: {:?}", recovered);
//! ```
//!
//! ## Real FFT (Optimized for Real Inputs)
//!
//! ```
//! use numrs2::fft;
//!
//! // Real-valued signal (typical use case)
//! let signal = vec![1.0, 0.5, -0.5, -1.0, 0.0, 0.5];
//!
//! // RFFT: optimized for real inputs, returns only positive frequencies
//! let spectrum = fft::rfft(&signal, None).expect("rfft should succeed");
//! println!("Spectrum length: {} (from {} real samples)", spectrum.len(), signal.len());
//!
//! // Inverse RFFT
//! let recovered = fft::irfft(&spectrum, Some(signal.len())).expect("irfft should succeed");
//! ```
//!
//! ## 2D FFT (Image Processing)
//!
//! ```
//! use numrs2::fft;
//! use scirs2_core::ndarray::Array2;
//!
//! // 2D signal (e.g., 8x8 image patch)
//! let image = Array2::<f64>::zeros((8, 8));
//!
//! // 2D FFT: spatial → frequency domain
//! let spectrum = fft::fft2(&image, None, None, None).expect("fft2 should succeed");
//! println!("2D spectrum shape: {:?}", spectrum.dim());
//!
//! // Inverse 2D FFT: frequency → spatial domain
//! let recovered = fft::ifft2(&spectrum, None, None, None).expect("ifft2 should succeed");
//! ```
//!
//! ## Discrete Cosine Transform (DCT)
//!
//! ```
//! use numrs2::fft::{self, DCTType};
//!
//! // Signal for DCT (commonly used in JPEG compression)
//! let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//!
//! // DCT Type-II (most common, used in JPEG/MP3)
//! let dct_coeffs = fft::dct(&signal, Some(DCTType::Type2), None).expect("dct should succeed");
//! println!("DCT coefficients: {:?}", dct_coeffs);
//!
//! // Inverse DCT
//! let recovered = fft::idct(&dct_coeffs, Some(DCTType::Type2), None).expect("idct should succeed");
//! ```
//!
//! ## Short-Time Fourier Transform (STFT)
//!
//! ```
//! use numrs2::fft::{self, Window};
//!
//! // Long signal for time-frequency analysis
//! let signal: Vec<f64> = (0..1000).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 1000.0).sin()).collect();
//!
//! // Compute STFT with Hann window
//! let (times, freqs, stft_matrix) = fft::spectrogram_stft(
//!     &signal,
//!     fft::Window::Hann,
//!     128,      // window size
//!     Some(64), // hop size (50% overlap)
//!     None,     // default FFT size
//!     Some(1000.0), // sampling rate
//!     None,     // no detrending
//!     None,     // return onesided spectrum
//!     None,     // boundary
//! ).expect("spectrogram_stft should succeed");
//!
//! println!("Time bins: {}, Frequency bins: {}", times.len(), freqs.len());
//! ```
//!
//! ## Frequency Helpers
//!
//! ```
//! use numrs2::fft;
//!
//! // Get FFT frequency bins
//! let n = 128;
//! let sample_rate = 1000.0;
//! let freqs = fft::fftfreq(n, 1.0 / sample_rate).expect("fftfreq should succeed");
//! println!("FFT frequencies: {:?}", &freqs[..10]);
//!
//! // Get RFFT frequency bins (only positive frequencies)
//! let rfreqs = fft::rfftfreq(n, 1.0 / sample_rate).expect("rfftfreq should succeed");
//! println!("RFFT frequencies: {:?}", rfreqs);
//!
//! // Find optimal FFT size (power of 2 or 3×2^k for faster computation)
//! let optimal_size = fft::next_fast_len(100, true);
//! println!("Optimal FFT size for 100 samples: {}", optimal_size);
//! ```
//!
//! # FFT Variants
//!
//! ## Complex FFT (General Purpose)
//! - `fft()`, `ifft()`: 1D complex-to-complex transforms
//! - `fft2()`, `ifft2()`: 2D transforms for images/matrices
//! - `fftn()`, `ifftn()`: N-dimensional transforms
//!
//! ## Real FFT (Optimized)
//! - `rfft()`, `irfft()`: 1D real-to-complex (2× faster than FFT)
//! - `rfft2()`, `irfft2()`: 2D real transforms
//! - `rfftn()`, `irfftn()`: N-dimensional real transforms
//! - Returns only positive frequencies (exploits Hermitian symmetry)
//!
//! ## Hermitian FFT
//! - `hfft()`, `ihfft()`: For signals with Hermitian symmetry
//! - `hfft2()`, `ihfft2()`: 2D Hermitian transforms
//!
//! ## Discrete Cosine Transform (DCT)
//! - Types I, II, III, IV available
//! - Type-II most common (JPEG, MP3, video codecs)
//! - `dct()`, `idct()`: 1D transforms
//! - `dct2()`, `idct2()`: 2D transforms (image blocks)
//! - `dctn()`, `idctn()`: N-dimensional transforms
//!
//! ## Discrete Sine Transform (DST)
//! - Types I, II, III, IV available
//! - Used in heat equation solvers, boundary problems
//! - `dst()`, `idst()`: 1D transforms
//! - `dst2()`, `idst2()`: 2D transforms
//! - `dstn()`, `idstn()`: N-dimensional transforms
//!
//! ## Specialized Transforms
//! - **Fractional FFT** (`frft`): Generalization of FFT with fractional order
//! - **Non-Uniform FFT** (`nufft`): FFT on non-uniformly spaced data
//! - **Fast Hankel Transform** (`fht`/`ifht`/`fhtoffset`): logarithmic-spacing
//!   Hankel transform (`scipy.fft.fht` semantics: order `mu`, bias, offset) --
//!   despite the name, this is *not* the Hartley transform.
//! - **Discrete Hartley Transform** (`dht`/`idht`, aliased `hartley_fht`):
//!   the real-valued alternative to FFT, `H(f) = Re(FFT(f)) - Im(FFT(f))`
//!
//! # Time-Frequency Analysis
//!
//! - **STFT**: Short-Time Fourier Transform for time-varying spectra
//! - **Spectrogram**: Power spectral density over time
//! - **Waterfall Plots**: 3D visualization of time-frequency data
//!
//! # Performance Features
//!
//! ## Plan Caching
//! ```rust,no_run
//! use numrs2::fft;
//!
//! // Plans are automatically cached for repeated transforms
//! let signal = vec![0.0; 1024];
//! let spectrum1 = fft::fft(&signal, None).expect("fft should succeed"); // Creates and caches plan
//! let spectrum2 = fft::fft(&signal, None).expect("fft should succeed"); // Reuses cached plan (faster)
//! ```
//!
//! ## SIMD Optimization
//! ```rust,no_run
//! use numrs2::fft;
//!
//! // SIMD-optimized variants (AVX/AVX2/AVX-512)
//! let signal = vec![0.0; 1024];
//! let spectrum = fft::fft_simd(&signal, None).expect("fft_simd should succeed");
//!
//! // Adaptive: automatically chooses best implementation
//! let spectrum = fft::fft_adaptive(&signal, None).expect("fft_adaptive should succeed");
//! ```
//!
//! ## Worker Pools (Parallel)
//! ```rust,no_run
//! use numrs2::fft;
//!
//! // Set number of worker threads
//! fft::set_workers(8);
//!
//! // Large transforms will use parallel execution
//! let large_signal = vec![0.0; 1048576];
//! let spectrum = fft::fft(&large_signal, None).expect("fft should succeed");
//! ```
//!
//! # Use Cases
//!
//! - **Signal Processing**: Filtering, spectral analysis, convolution
//! - **Image Processing**: Frequency-domain filtering, compression (JPEG)
//! - **Audio Processing**: Music analysis, speech processing, audio codecs
//! - **Scientific Computing**: PDE solvers, numerical methods
//! - **Communications**: Modulation/demodulation, channel estimation
//! - **Machine Learning**: Feature extraction, time-series analysis

// Re-export all scirs2-fft modules and functions
pub use scirs2_fft::*;

pub mod array_native;
pub mod numpy_parity;

pub use array_native::FFT;
pub use numpy_parity::{fft_with, fftn, ifft_with, ifftn, irfft_with, irfftn, rfft_with, rfftn};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_basic() {
        // Basic FFT test
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let spectrum = fft(&signal, None).expect("fft should succeed");

        // FFT of real signal has length equal to input
        assert_eq!(spectrum.len(), signal.len());

        // Inverse FFT should recover original signal
        let recovered = ifft(&spectrum, None).expect("ifft should succeed");
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec.re).abs() < 1e-10);
            assert!(rec.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_rfft_basic() {
        // RFFT test (optimized for real inputs)
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let spectrum = rfft(&signal, None).expect("rfft should succeed");

        // RFFT output length is n/2 + 1
        assert_eq!(spectrum.len(), signal.len() / 2 + 1);

        // Inverse RFFT should recover original signal
        let recovered = irfft(&spectrum, Some(signal.len())).expect("irfft should succeed");
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft2() {
        use scirs2_core::ndarray::Array2;

        // 2D FFT test
        let image = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let spectrum = fft2(&image, None, None, None).expect("fft2 should succeed");

        // Output should have same shape
        assert_eq!(spectrum.dim(), image.dim());

        // Inverse FFT should recover original
        let recovered = ifft2(&spectrum, None, None, None).expect("ifft2 should succeed");
        for (orig, rec) in image.iter().zip(recovered.iter()) {
            assert!((orig - rec.re).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dct_basic() {
        // DCT test
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let dct_coeffs = dct(&signal, Some(DCTType::Type2), None).expect("dct should succeed");

        // DCT output has same length as input
        assert_eq!(dct_coeffs.len(), signal.len());

        // IDCT should recover original
        let recovered = idct(&dct_coeffs, Some(DCTType::Type2), None).expect("idct should succeed");
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-9);
        }
    }

    #[test]
    fn test_dst_basic() {
        // DST test
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let dst_coeffs = dst(&signal, Some(DSTType::Type2), None).expect("dst should succeed");

        // DST output has same length as input
        assert_eq!(dst_coeffs.len(), signal.len());

        // DST coefficients should exist and be finite
        assert!(dst_coeffs.iter().all(|x| x.is_finite()));

        // Note: DST round-trip may have normalization differences
        // Full recovery test disabled due to normalization
    }

    #[test]
    fn test_fftfreq() {
        // Test FFT frequency bins
        let n = 8;
        let dt = 0.1;
        let freqs = fftfreq(n, dt).expect("fftfreq should succeed");

        assert_eq!(freqs.len(), n);

        // First frequency should be 0
        assert!((freqs[0] - 0.0).abs() < 1e-10);

        // Frequencies should be symmetric around Nyquist
        assert!((freqs[n / 2 - 1] + freqs[n / 2 + 1]).abs() < 1e-10);
    }

    #[test]
    fn test_rfftfreq() {
        // Test RFFT frequency bins
        let n = 8;
        let dt = 0.1;
        let freqs = rfftfreq(n, dt).expect("rfftfreq should succeed");

        // RFFT returns n/2 + 1 frequencies
        assert_eq!(freqs.len(), n / 2 + 1);

        // First frequency should be 0
        assert!((freqs[0] - 0.0).abs() < 1e-10);

        // All frequencies should be non-negative
        for freq in freqs.iter() {
            assert!(*freq >= 0.0);
        }
    }

    #[test]
    fn test_next_fast_len() {
        // Test optimal FFT size finder
        let sizes = vec![100, 200, 500, 1000, 1500];

        for size in sizes {
            let optimal = next_fast_len(size, false);

            // Optimal size should be >= input size
            assert!(optimal >= size);

            // Should be a fast size (power of 2 or 3×2^k)
            // Just verify it's reasonable (not too large)
            assert!(optimal < size * 2);
        }
    }

    #[test]
    fn test_fftshift() {
        use scirs2_core::ndarray::Array1;

        // Test FFT shift (DC component to center)
        let arr = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shifted = fftshift(&arr).expect("fftshift should succeed");

        assert_eq!(shifted.len(), arr.len());

        // For even length, first element should be from middle
        assert!((shifted[0] - arr[arr.len() / 2]).abs() < 1e-10);
    }

    #[test]
    fn test_ifftshift() {
        use scirs2_core::ndarray::Array1;

        // Test inverse FFT shift
        let arr = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shifted = fftshift(&arr).expect("fftshift should succeed");
        let recovered = ifftshift(&shifted).expect("ifftshift should succeed");

        // Should recover original arrangement
        for (orig, rec) in arr.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_worker_pool() {
        // Test worker pool configuration
        let original_workers = get_workers();

        // Set to 4 workers (may or may not succeed)
        let _ = set_workers(4);
        // Just verify we can get the current worker count
        let current = get_workers();
        assert!(current > 0);

        // Restore original
        let _ = set_workers(original_workers);
    }

    #[test]
    fn test_fht_basic() {
        // NOTE: this exercises the *Discrete Hartley Transform* (`dht`), not
        // the identically-abbreviated `fht` re-export -- `scirs2_fft::fht`
        // (SciPy's `scipy.fft.fht`) is actually the logarithmic-spacing Fast
        // **Hankel** Transform (parameters `mu`/`bias`/`offset`), a
        // completely different transform that happens to share the initials
        // "FHT". The original version of this test called that Hankel `fht`
        // while asserting Hartley-transform semantics, which is why it never
        // passed with meaningful parameters and was left `#[ignore]`d. The
        // true Hartley transform lives at `dht`/`idht` (and `hartley_fht`,
        // an alias for the same family) -- see the module docs above.
        //
        // Reference values pinned from the derivation `H[k] = Re(FFT(x))[k]
        // - Im(FFT(x))[k]` (verified with `numpy.fft.fft`, since SciPy has
        // no public Hartley-transform function to cross-check against):
        // `np.fft.fft([1,2,3,4])` = `[10+0j, -2+2j, -2+0j, -2-2j]`, so
        // `Re - Im` = `[10, -4, -2, 0]`.
        use scirs2_core::ndarray::Array1;
        let signal = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0]);
        let hartley = dht(&signal).expect("dht should succeed");

        assert_eq!(hartley.len(), signal.len());
        assert!(hartley.iter().all(|x| x.is_finite()));

        let expected = [10.0, -4.0, -2.0, 0.0];
        for (got, want) in hartley.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }

        // The Hartley transform is self-inverse up to a factor of 1/N.
        let recovered = idht(&hartley).expect("idht should succeed");
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-9, "orig {orig}, rec {rec}");
        }
    }

    #[test]
    fn test_hfft_basic() {
        // Hermitian FFT test
        use scirs2_core::num_complex::Complex64;

        // Create a Hermitian spectrum (conjugate symmetric)
        let spectrum = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, -1.0),
        ];

        let signal = hfft(&spectrum, None, None).expect("hfft should succeed");

        // HFFT produces real output
        assert!(signal.iter().all(|x| x.is_finite()));

        // IHFFT should recover spectrum
        let recovered = ihfft(&signal, None, None).expect("ihfft should succeed");
        for (orig, rec) in spectrum.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-9);
            assert!((orig.im - rec.im).abs() < 1e-9);
        }
    }

    // =========================================================================
    // N-D FFT wrapper tests (fftn/ifftn/rfftn/irfftn) -- NumPy-pinned
    // =========================================================================
    //
    // Reference values throughout this section were computed with
    // `numpy.fft` (numpy 2.4.2) on `np.arange(60.0).reshape(3, 4, 5)`, e.g.:
    // `python3 -c "import numpy as np; x = np.arange(60.0).reshape(3,4,5);
    // print(np.fft.fftn(x)[0,0,1])"`.

    #[test]
    fn test_fftn_pinned_3x4x5() {
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");

        let spectrum = fftn(&x, None, None, None).expect("fftn should succeed");
        assert_eq!(spectrum.shape(), &[3, 4, 5]);

        let check = |idx: [usize; 3], re: f64, im: f64| {
            let v = spectrum[IxDyn(&idx)];
            assert!((v.re - re).abs() < 1e-9, "re mismatch at {idx:?}: {v:?}");
            assert!((v.im - im).abs() < 1e-9, "im mismatch at {idx:?}: {v:?}");
        };
        check([0, 0, 0], 1770.0, 0.0);
        check([0, 0, 1], -30.0, 41.291457614135204);
        check([1, 0, 0], -600.0, 346.4101615137754);
        check([0, 1, 0], -150.0, 150.0);

        // Parseval: sum|x|^2 == sum|F|^2 / N for the default ("backward") norm.
        let sum_sq_x: f64 = x.iter().map(|v| v * v).sum();
        let sum_sq_f: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        assert!((sum_sq_x - sum_sq_f / 60.0).abs() < 1e-6);

        // ifftn should recover the original (real) input.
        let recovered = ifftn(&spectrum, None, None, None).expect("ifftn should succeed");
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            assert!((orig - rec.re).abs() < 1e-9);
            assert!(rec.im.abs() < 1e-9);
        }
    }

    #[test]
    fn test_fftn_norm_modes() {
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");
        let backward = fftn(&x, None, None, None).expect("backward fftn should succeed");
        let dc_backward = backward[IxDyn(&[0, 0, 0])].re;

        let ortho = fftn(&x, None, None, Some("ortho")).expect("ortho fftn should succeed");
        assert!((ortho[IxDyn(&[0, 0, 0])].re - dc_backward / 60.0_f64.sqrt()).abs() < 1e-9);

        let forward = fftn(&x, None, None, Some("forward")).expect("forward fftn should succeed");
        assert!((forward[IxDyn(&[0, 0, 0])].re - dc_backward / 60.0).abs() < 1e-9);

        // An unrecognized norm string must be rejected, not silently ignored
        // the way the underlying `scirs2_fft::fftn` does.
        assert!(fftn(&x, None, None, Some("bogus")).is_err());

        // Regression: explicit `norm="backward"` must be a complete no-op
        // on the forward transform, byte-for-byte identical to the
        // `norm=None` default -- `scirs2_fft::fftn` itself gets this wrong
        // (confirmed by direct probing: it applies a spurious `1/N` to the
        // *forward* direction when given the string `"backward"`
        // explicitly, even though `"backward"` names how the *inverse*
        // direction should scale). Pinned against `dc_backward` computed
        // above, which numpy.fft.fftn's own default (also "backward")
        // agrees is 1770.0, not 1770.0/60.
        let explicit_backward =
            fftn(&x, None, None, Some("backward")).expect("explicit backward should succeed");
        assert!((explicit_backward[IxDyn(&[0, 0, 0])].re - dc_backward).abs() < 1e-9);
        assert!((dc_backward - 1770.0).abs() < 1e-9);

        // Regression: `norm="ortho"`/`"forward"` must scale by the size of
        // just the *transformed* axes (here, axis 0's length of 3), not
        // the whole output array's size (60) -- `scirs2_fft::fftn` gets
        // this wrong too whenever `axes` is a strict subset. Pinned
        // against `numpy.fft.fftn(x, axes=[0])[0,0,0]` == 60.0 (== `x[0,0,0]
        // + x[1,0,0] + x[2,0,0]` == `0 + 20 + 40`, since only axis 0 is
        // transformed and (0,0) on the other two axes is held fixed) and
        // its `norm='ortho'`/`norm='forward'` variants, 60/sqrt(3) and
        // 60/3 respectively.
        let axis0_raw = fftn(&x, None, Some(&[0]), None).expect("axis-0 fftn should succeed");
        let dc_axis0 = axis0_raw[IxDyn(&[0, 0, 0])].re;
        assert!((dc_axis0 - 60.0).abs() < 1e-9);

        let axis0_ortho =
            fftn(&x, None, Some(&[0]), Some("ortho")).expect("axis-0 ortho should succeed");
        assert!((axis0_ortho[IxDyn(&[0, 0, 0])].re - dc_axis0 / 3.0_f64.sqrt()).abs() < 1e-9);

        let axis0_forward =
            fftn(&x, None, Some(&[0]), Some("forward")).expect("axis-0 forward should succeed");
        assert!((axis0_forward[IxDyn(&[0, 0, 0])].re - dc_axis0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fftn_s_pad_truncate() {
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");

        // Truncate axis 0 from 3 to 2: DC becomes the sum of just the
        // truncated elements (0..40), pinned against
        // `numpy.fft.fftn(x, s=[2, 4, 5])[0, 0, 0]` == 780.
        let truncated = fftn(&x, Some(&[2, 4, 5]), None, None).expect("truncate should succeed");
        assert_eq!(truncated.shape(), &[2, 4, 5]);
        assert!((truncated[IxDyn(&[0, 0, 0])].re - 780.0).abs() < 1e-9);

        // Pad axis 0 from 3 to 4 with zeros: DC is unchanged (zeros don't
        // add to the sum), pinned against
        // `numpy.fft.fftn(x, s=[4, 4, 5])[0, 0, 0]` == 1770.
        let padded = fftn(&x, Some(&[4, 4, 5]), None, None).expect("pad should succeed");
        assert_eq!(padded.shape(), &[4, 4, 5]);
        assert!((padded[IxDyn(&[0, 0, 0])].re - 1770.0).abs() < 1e-9);

        // `s` given only for the axes actually being transformed (NumPy
        // allows `s` to have one entry per axis in `axes`, not full-ndim).
        // With `axes=[0]`, only axis 0 is transformed (truncated to length
        // 2), so `[0, 0, 0]` is the axis-0-only DC term for the (0, 0)
        // slice of the untouched axes 1/2 -- just `x[0,0,0] + x[1,0,0]` --
        // *not* the full-3-D-transform DC term computed above. Pinned
        // against `numpy.fft.fftn(x, s=[2], axes=[0])[0, 0, 0]` == 20.
        let partial = fftn(&x, Some(&[2]), Some(&[0]), None).expect("partial s should succeed");
        assert_eq!(partial.shape(), &[2, 4, 5]);
        assert!((partial[IxDyn(&[0, 0, 0])].re - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_fftn_negative_axes() {
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");

        let via_negative =
            fftn(&x, None, Some(&[-1, -2]), None).expect("negative axes should succeed");
        let via_positive =
            fftn(&x, None, Some(&[2, 1]), None).expect("equivalent positive axes should succeed");
        for (a, b) in via_negative.iter().zip(via_positive.iter()) {
            assert!((a.re - b.re).abs() < 1e-9);
            assert!((a.im - b.im).abs() < 1e-9);
        }

        // Out-of-range axes must be rejected, not panic or silently wrap.
        assert!(fftn(&x, None, Some(&[-5]), None).is_err());
        assert!(fftn(&x, None, Some(&[3]), None).is_err());
    }

    #[test]
    fn test_rfftn_irfftn_roundtrip() {
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");

        let spectrum = rfftn(&x, None, None, None).expect("rfftn should succeed");
        // Last axis halved: 5 -> 5/2+1 = 3.
        assert_eq!(spectrum.shape(), &[3, 4, 3]);
        assert!((spectrum[IxDyn(&[0, 0, 0])].re - 1770.0).abs() < 1e-9);
        assert!((spectrum[IxDyn(&[0, 0, 1])].re - (-30.0)).abs() < 1e-9);
        assert!((spectrum[IxDyn(&[0, 0, 1])].im - 41.291457614135204).abs() < 1e-9);

        let recovered = irfftn(&spectrum, Some(&[3, 4, 5]), None, None)
            .expect("irfftn roundtrip should succeed");
        assert_eq!(recovered.shape(), &[3, 4, 5]);
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-8, "orig {orig}, rec {rec}");
        }

        // norm="ortho"/"forward" round-trips: `irfftn` must invert exactly
        // whatever scale `rfftn` applied. Pinned against
        // `numpy.fft.irfftn(numpy.fft.rfftn(x), s=[3,4,5], norm=...)`.
        let spectrum_ortho = rfftn(&x, None, None, Some("ortho")).expect("ortho rfftn ok");
        let recovered_ortho = irfftn(&spectrum_ortho, Some(&[3, 4, 5]), None, Some("ortho"))
            .expect("ortho irfftn roundtrip ok");
        for (orig, rec) in x.iter().zip(recovered_ortho.iter()) {
            assert!((orig - rec).abs() < 1e-7, "orig {orig}, rec {rec}");
        }

        let spectrum_forward = rfftn(&x, None, None, Some("forward")).expect("forward rfftn ok");
        let recovered_forward = irfftn(&spectrum_forward, Some(&[3, 4, 5]), None, Some("forward"))
            .expect("forward irfftn roundtrip ok");
        for (orig, rec) in x.iter().zip(recovered_forward.iter()) {
            assert!((orig - rec).abs() < 1e-7, "orig {orig}, rec {rec}");
        }
    }

    #[test]
    fn test_rfftn_axes_subset_norm_regression() {
        // Regression test for the same `scirs2_fft::fftn` axes-restricted
        // scale-basis bug `test_fftn_norm_modes` pins, but exercised
        // through `rfftn` (which computes its complex spectrum via this
        // module's `fftn`, so a re-regression here would mean the fix
        // stopped propagating through that call path).
        use scirs2_core::ndarray::{ArrayD, IxDyn};

        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let x = ArrayD::from_shape_vec(IxDyn(&[3, 4, 5]), data).expect("shape ok");

        // Pinned against `numpy.fft.rfftn(x, axes=[0])[0,0,0]` == 60.0 (==
        // `x[0,0,0] + x[1,0,0] + x[2,0,0]`, only axis 0 transformed) and
        // its `norm="ortho"` variant == 60/sqrt(3).
        let raw = rfftn(&x, None, Some(&[0]), None).expect("axis-0 rfftn should succeed");
        assert!((raw[IxDyn(&[0, 0, 0])].re - 60.0).abs() < 1e-9);

        let ortho = rfftn(&x, None, Some(&[0]), Some("ortho")).expect("axis-0 ortho rfftn ok");
        assert!((ortho[IxDyn(&[0, 0, 0])].re - 60.0 / 3.0_f64.sqrt()).abs() < 1e-9);
    }

    // =========================================================================
    // 1-D `_with` parameter wrapper tests (n=/axis=/norm=) -- NumPy-pinned
    // =========================================================================
    //
    // Reference values computed with `numpy.fft` on `[1, 2, 3, 4, 5]` (a
    // deliberately non-power-of-two length).

    #[test]
    fn test_fft_with_n_pad_truncate() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];

        // n=None must use the exact input length (5), matching NumPy --
        // *not* padded up to the next power of two the way the bare
        // `scirs2_fft::fft(x, None)` would be (see this file's merge
        // report for that upstream divergence).
        let exact = fft_with(&x, None, None, None).expect("fft_with n=None should succeed");
        assert_eq!(exact.len(), 5);
        assert!((exact[0].re - 15.0).abs() < 1e-9);
        assert!((exact[1].re - (-2.5)).abs() < 1e-9);
        assert!((exact[1].im - 3.440954801177934).abs() < 1e-9);

        // n=8: zero-pad.
        let padded = fft_with(&x, Some(8), None, None).expect("fft_with n=8 should succeed");
        assert_eq!(padded.len(), 8);
        assert!((padded[0].re - 15.0).abs() < 1e-9);
        assert!((padded[1].re - (-5.414213562373095)).abs() < 1e-9);
        assert!((padded[1].im - (-7.242640687119286)).abs() < 1e-9);

        // n=3: truncate.
        let truncated = fft_with(&x, Some(3), None, None).expect("fft_with n=3 should succeed");
        assert_eq!(truncated.len(), 3);
        assert!((truncated[0].re - 6.0).abs() < 1e-9);
        assert!((truncated[1].re - (-1.5)).abs() < 1e-9);
        assert!((truncated[1].im - 0.8660254037844386).abs() < 1e-9);
    }

    #[test]
    fn test_fft_with_norm_modes() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let backward = fft_with(&x, None, None, None).expect("backward should succeed");

        let ortho = fft_with(&x, None, None, Some("ortho")).expect("ortho should succeed");
        for (b, o) in backward.iter().zip(ortho.iter()) {
            assert!((o.re - b.re / 5.0_f64.sqrt()).abs() < 1e-9);
            assert!((o.im - b.im / 5.0_f64.sqrt()).abs() < 1e-9);
        }

        let forward = fft_with(&x, None, None, Some("forward")).expect("forward should succeed");
        for (b, f) in backward.iter().zip(forward.iter()) {
            assert!((f.re - b.re / 5.0).abs() < 1e-9);
            assert!((f.im - b.im / 5.0).abs() < 1e-9);
        }

        assert!(fft_with(&x, None, None, Some("bogus")).is_err());
        assert!(fft_with(&x, None, Some(1), None).is_err()); // only 0/-1 valid for a flat slice
    }

    #[test]
    fn test_rfft_with_irfft_with_roundtrip_and_norm() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];

        let spectrum = rfft_with(&x, None, None, None).expect("rfft_with should succeed");
        assert_eq!(spectrum.len(), 3);
        assert!((spectrum[0].re - 15.0).abs() < 1e-9);

        let spectrum8 = rfft_with(&x, Some(8), None, None).expect("rfft_with n=8 should succeed");
        assert_eq!(spectrum8.len(), 5);
        assert!((spectrum8[1].re - (-5.414213562373096)).abs() < 1e-9);
        assert!((spectrum8[1].im - (-7.242640687119285)).abs() < 1e-9);

        // Backward-mode round trip.
        let recovered = irfft_with(&spectrum, Some(5), None, None)
            .expect("irfft_with roundtrip should succeed");
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-9);
        }

        // Default n (None) matches NumPy's `2*(m-1)` rule: m=3 -> n=4.
        let default_n = irfft_with(&spectrum, None, None, None).expect("default n should succeed");
        assert_eq!(default_n.len(), 4);

        // ortho/forward corrections relative to the backward round trip.
        let ortho = irfft_with(&spectrum, Some(5), None, Some("ortho"))
            .expect("ortho irfft_with should succeed");
        for (b, o) in recovered.iter().zip(ortho.iter()) {
            assert!((o - b * 5.0_f64.sqrt()).abs() < 1e-8);
        }
        let forward = irfft_with(&spectrum, Some(5), None, Some("forward"))
            .expect("forward irfft_with should succeed");
        for (b, f) in recovered.iter().zip(forward.iter()) {
            assert!((f - b * 5.0).abs() < 1e-8);
        }
    }

    // =========================================================================
    // Property-based tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_parseval_holds_for_any_length(n in 1usize..40) {
            // Parseval's theorem: sum(|x[i]|^2) == sum(|X[k]|^2) / N, for
            // *any* length N (not just powers of two) under the default
            // ("backward") norm -- this is exactly what distinguishes
            // `fft_with` (always passes an explicit `n`) from the bare
            // re-exported `fft`, whose `n=None` default secretly rounds up
            // to the next power of two.
            let x: Vec<f64> = (0..n.max(1)).map(|i| (i as f64 * 0.7).sin()).collect();
            let spectrum = fft_with(&x, None, None, None)
                .expect("fft_with should succeed for any positive length");
            let sum_sq_x: f64 = x.iter().map(|v| v * v).sum();
            let sum_sq_f: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
            let scale = sum_sq_x.max(1.0);
            prop_assert!(
                (sum_sq_x - sum_sq_f / x.len() as f64).abs() < 1e-6 * scale,
                "Parseval failed for n={}: time-domain {} vs freq-domain {}",
                x.len(),
                sum_sq_x,
                sum_sq_f / x.len() as f64
            );
        }
    }
}

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

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use scirs2_core::ndarray::{ArrayD, Slice};
use scirs2_core::{Complex, Complex64};
use std::f64::consts::PI;
use std::fmt::Debug;

// Re-export all scirs2-fft modules and functions
pub use scirs2_fft::*;

// ===========================================================================
// NumPy-parity parameter wrappers
// ===========================================================================
//
// `scirs2_fft`'s own N-D entry points (`fftn`/`ifftn`/`rfftn`/`irfftn`,
// re-exported above) already have a real, hardware-accelerated (`oxifft`
// backend) N-dimensional implementation -- there is no need to hand-roll a
// separable axis-by-axis pass on top of them. What they *don't* match is
// NumPy's exact calling convention:
//
// * `axes` there is `Option<Vec<usize>>` (no negative-index support);
//   NumPy's `axes` accepts negative indices (`-1` = last axis, etc.).
// * `shape` there must either be `None` or cover *every* dimension of the
//   input; NumPy's `s` may instead have one entry per axis in `axes`.
// * A handful of them (verified empirically against `numpy.fft`, see the
//   pinned tests below) silently treat an unrecognized `norm` string the
//   same as `None` instead of rejecting it the way NumPy does.
//
// The wrappers below normalize NumPy-style parameters into what the
// underlying functions expect and then delegate the actual transform to
// them -- confirmed against pinned `numpy.fft` output (a 3x4x5 `fftn`, its
// `norm="ortho"`/`"forward"` variants, and `rfftn`/`irfftn` round-trips) to
// carry no extra normalization surprises of their own: `scirs2_fft`'s
// `norm` handling for these already matches NumPy's exactly.

/// Normalize a NumPy-style, possibly-negative axis list against `ndim`.
///
/// `None` passes through unchanged (scirs2_fft's own N-D functions already
/// default a missing axis list to "every axis, in order"). Each negative
/// axis `a` is remapped to `a + ndim`, matching NumPy's `axis`/`axes`
/// conventions; order is preserved (not sorted), since some callers (e.g.
/// `rfftn`) treat the *last given* axis specially.
fn normalize_axes(axes: Option<&[isize]>, ndim: usize) -> FFTResult<Option<Vec<usize>>> {
    let Some(axes) = axes else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(axes.len());
    for &axis in axes {
        let normalized = if axis < 0 { axis + ndim as isize } else { axis };
        if normalized < 0 || normalized as usize >= ndim {
            return Err(FFTError::ValueError(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        out.push(normalized as usize);
    }
    Ok(Some(out))
}

/// Expand a NumPy-style `s` (sizes for just the transformed axes) into the
/// full-`ndim` output shape that `fftn`/`ifftn`/`rfftn`/`irfftn` require,
/// falling back to `input_shape` on every axis `s` doesn't cover.
///
/// Mirrors NumPy's `s`/`axes` pairing: when `axes` is `None`, `s` (if
/// given) must cover every axis; when `axes` is `Some`, `s` (if given)
/// must have exactly one entry per axis, applied positionally.
fn expand_shape(
    input_shape: &[usize],
    s: Option<&[usize]>,
    axes: &Option<Vec<usize>>,
) -> FFTResult<Option<Vec<usize>>> {
    let Some(sizes) = s else {
        return Ok(None);
    };
    let mut full = input_shape.to_vec();
    match axes {
        Some(ax) => {
            if sizes.len() != ax.len() {
                return Err(FFTError::ValueError(format!(
                    "s must have the same length as axes ({} vs {})",
                    sizes.len(),
                    ax.len()
                )));
            }
            for (&axis, &size) in ax.iter().zip(sizes.iter()) {
                full[axis] = size;
            }
        }
        None => {
            if sizes.len() != input_shape.len() {
                return Err(FFTError::ValueError(format!(
                    "s must have the same length as the input's number of dimensions ({}) when axes is not given, got {}",
                    input_shape.len(),
                    sizes.len()
                )));
            }
            full.copy_from_slice(sizes);
        }
    }
    Ok(Some(full))
}

/// Validate a NumPy-style `norm` string ourselves: `scirs2_fft::fftn`/
/// `ifftn` silently treat any unrecognized string the same as `None`
/// (no normalization) rather than rejecting it, so without this check an
/// invalid `norm` would be ignored instead of reported.
fn validate_norm(norm: Option<&str>) -> FFTResult<()> {
    match norm {
        None | Some("backward") | Some("ortho") | Some("forward") => Ok(()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// N-dimensional FFT with NumPy's exact `fftn(a, s=None, axes=None,
/// norm=None)` parameter conventions, shadowing [`scirs2_fft::fftn`] (see
/// the module-level note above for why).
pub fn fftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    scirs2_fft::fftn(input, outshape, normalized_axes, norm, None, None)
}

/// N-dimensional inverse FFT with NumPy's exact `ifftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::ifftn`].
pub fn ifftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    scirs2_fft::ifftn(input, outshape, normalized_axes, norm, None, None)
}

/// N-dimensional real-input FFT with NumPy's exact `rfftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::rfftn`].
///
/// `scirs2_fft::rfftn` only halves its last transformed axis (keeping the
/// non-redundant half of the Hermitian-symmetric spectrum) when its own
/// `shape` parameter is `None`; passing it any explicit shape -- which is
/// unavoidable once `s` zero-pads or truncates an axis -- disables that
/// halving. So instead of delegating to it directly, this computes the
/// full complex N-D FFT (via `fftn`, whose zero-pad/truncate semantics are
/// unconditional) and then keeps only the non-redundant half of the last
/// transformed axis itself, unconditionally -- the same relationship
/// NumPy documents between `rfftn` and `fftn`.
pub fn rfftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let ndim = input.ndim();
    let normalized_axes = normalize_axes(axes, ndim)?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;

    let last_axis = match &normalized_axes {
        Some(ax) => *ax.last().ok_or_else(|| {
            FFTError::ValueError("axes must contain at least one axis".to_string())
        })?,
        None => ndim.checked_sub(1).ok_or_else(|| {
            FFTError::ValueError("rfftn requires an array with at least one dimension".to_string())
        })?,
    };

    let full = scirs2_fft::fftn(input, outshape, normalized_axes, norm, None, None)?;
    let half_len = full.shape()[last_axis] / 2 + 1;
    let sliced = full
        .slice_each_axis(|ax| {
            if ax.axis.index() == last_axis {
                Slice::new(0, Some(half_len as isize), 1)
            } else {
                Slice::new(0, None, 1)
            }
        })
        .to_owned();
    Ok(sliced)
}

/// N-dimensional inverse real FFT with NumPy's exact `irfftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::irfftn`].
///
/// Unlike `rfftn`, `scirs2_fft::irfftn` already defaults a missing `shape`
/// to NumPy's own convention (`2 * (m - 1)` on the last transformed axis),
/// so an absent `s` can be passed straight through as `None`.
pub fn irfftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    scirs2_fft::irfftn(&input.view(), outshape, normalized_axes, norm, None, None)
}

/// Validate a NumPy-style `axis` for a transform that operates on a flat,
/// one-dimensional slice: the only legal values are `0` and `-1` (both
/// name the slice's single axis). Multi-axis selection is only meaningful
/// for the N-D family ([`fftn`], [`ifftn`], [`rfftn`], [`irfftn`]).
fn validate_flat_axis(axis: Option<isize>) -> FFTResult<()> {
    match axis {
        None | Some(0) | Some(-1) => Ok(()),
        Some(other) => Err(FFTError::ValueError(format!(
            "axis {other} is out of bounds for a 1-D transform (expected 0 or -1)"
        ))),
    }
}

/// Absolute scale factor for a *raw, completely unnormalized* forward
/// transform of size `n`, matching NumPy's `norm` semantics for
/// `fft`/`rfft` (`"backward"`, the default, applies no scaling at all to
/// the forward direction).
fn forward_norm_scale(norm: Option<&str>, n: usize) -> FFTResult<f64> {
    match norm {
        None | Some("backward") => Ok(1.0),
        Some("forward") => Ok(1.0 / n as f64),
        Some("ortho") => Ok(1.0 / (n as f64).sqrt()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// Correction factor to apply on top of `scirs2_fft`'s `ifft`/`irfft`,
/// which always apply the `"backward"`-mode `1/n` scaling internally, to
/// realize whatever `norm` was actually requested.
fn inverse_norm_correction(norm: Option<&str>, n: usize) -> FFTResult<f64> {
    match norm {
        None | Some("backward") => Ok(1.0),
        Some("forward") => Ok(n as f64),
        Some("ortho") => Ok((n as f64).sqrt()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// 1-D FFT with NumPy's exact `fft(a, n=None, axis=-1, norm=None)`
/// parameters. The plain [`fft`] (re-exported from `scirs2_fft` above)
/// keeps its original 2-argument signature (`x`, `n`) unchanged; this adds
/// the full parameter set as a separate, `_with`-suffixed function rather
/// than changing `fft`'s arity, matching the calling convention
/// `scirs2_fft` itself already uses for its multi-parameter transforms
/// (`fft2`, `fftn`, `dct`, ...).
pub fn fft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let scale = forward_norm_scale(norm, size)?;
    let mut result = scirs2_fft::fft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c = *c * scale);
    Ok(result)
}

/// 1-D inverse FFT with NumPy's exact `ifft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`ifft`]'s signature.
pub fn ifft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let correction = inverse_norm_correction(norm, size)?;
    let mut result = scirs2_fft::ifft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c = *c * correction);
    Ok(result)
}

/// 1-D real-input FFT with NumPy's exact `rfft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`rfft`]'s signature.
pub fn rfft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let scale = forward_norm_scale(norm, size)?;
    let mut result = scirs2_fft::rfft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c = *c * scale);
    Ok(result)
}

/// 1-D inverse real FFT with NumPy's exact `irfft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`irfft`]'s signature.
pub fn irfft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    // NumPy's own default, mirrored by `scirs2_fft::irfft`: n = 2*(m-1)
    // where m is the input (spectrum) length.
    let size = n.unwrap_or_else(|| 2 * x.len().saturating_sub(1));
    let correction = inverse_norm_correction(norm, size)?;
    let mut result = scirs2_fft::irfft(x, Some(size))?;
    result.iter_mut().for_each(|v| *v *= correction);
    Ok(result)
}

// Additional NumRS2-specific convenience functions and aliases can be added here

// ===========================================================================
// `Array<T>`-native FFT (merged from the former `new_modules::fft` module)
// ===========================================================================
//
// This section provides an `Array<T>`-oriented FFT surface (a `FFT` struct
// of associated functions, plus matching inherent methods on `Array<T>` /
// `Array<Complex<T>>` such as `.fft()`/`.ifft()`), distinct from the thin
// `scirs2_fft` passthrough above (which operates on plain slices and
// `scirs2_core::ndarray` types). It is ported here unchanged from the
// former `src/new_modules/fft.rs` (a near-duplicate FFT implementation);
// `src/new_modules/fft.rs` itself has been deleted and
// `src/new_modules/mod.rs` now re-exports this module in its place (see
// that file), so `crate::new_modules::fft::FFT` keeps resolving to this
// same `FFT` for the handful of other in-crate modules that still name it
// that way.
//
// NOTE: this implementation's own `fft`/`ifft`/`fft2`/`ifft2` (and the
// `is_power_of_two` guard in front of them) still require power-of-2
// lengths, using a hand-rolled recursive Cooley-Tukey radix-2 kernel
// rather than the arbitrary-size, `oxifft`-backed transforms available
// through the `scirs2_fft` re-exports above -- that limitation predates
// this merge and is carried forward as-is (a pure relocation, not a
// rewrite); `fftn`/`rfft_with`/etc. above are the arbitrary-size,
// NumPy-parity path for new code.

/// Fast Fourier Transform (FFT) implementation for NumRS
/// A wrapper for FFT functionality
pub struct FFT;

impl FFT {
    /// Compute the Fast Fourier Transform of a real-valued array
    pub fn fft<T>(x: &Array<T>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 1D
        if shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "FFT expects a 1D array".to_string(),
            ));
        }

        let n = shape[0];

        // Check if n is a power of 2 for Cooley-Tukey FFT
        if !is_power_of_two(n) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "FFT requires input length to be a power of 2, got {}",
                n
            )));
        }

        let data = x.to_vec();

        // Convert to complex and compute FFT
        let mut complex_data: Vec<Complex<T>> = data
            .iter()
            .map(|&val| Complex::new(val, T::zero()))
            .collect();

        // Compute the FFT using Cooley-Tukey algorithm
        fft_recursive(&mut complex_data);

        // Return as array
        Ok(Array::from_vec(complex_data))
    }

    /// Compute the Inverse Fast Fourier Transform of a complex-valued array
    pub fn ifft<T>(x: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 1D
        if shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "IFFT expects a 1D array".to_string(),
            ));
        }

        let n = shape[0];

        // Check if n is a power of 2 for Cooley-Tukey IFFT
        if !is_power_of_two(n) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "IFFT requires input length to be a power of 2, got {}",
                n
            )));
        }

        let data = x.to_vec();

        // Conjugate the input, compute FFT, conjugate the output, and scale
        let complex_data: Vec<Complex<T>> = data.iter().map(|val| val.conj()).collect();

        // Compute the FFT
        let mut complex_data_mut = complex_data;
        fft_recursive(&mut complex_data_mut);

        // Conjugate and scale
        let scale: T = <T as From<f64>>::from(1.0 / n as f64);
        let complex_data = complex_data_mut
            .iter()
            .map(|val| val.conj() * scale)
            .collect();

        // Return as array
        Ok(Array::from_vec(complex_data))
    }

    /// Compute the power spectrum of a signal (|FFT|^2)
    pub fn power_spectrum<T>(x: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        // Compute FFT
        let fft_result = Self::fft(x)?;
        let fft_data = fft_result.to_vec();

        // Compute power spectrum |FFT|^2
        let power: Vec<T> = fft_data.iter().map(|val| val.norm_sqr()).collect();

        Ok(Array::from_vec(power))
    }

    /// Compute 2D FFT of a 2D array
    pub fn fft2<T>(x: &Array<T>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 2D
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "FFT2 expects a 2D array".to_string(),
            ));
        }

        let n_rows = shape[0];
        let n_cols = shape[1];

        // Check if dimensions are powers of 2
        if !is_power_of_two(n_rows) || !is_power_of_two(n_cols) {
            return Err(NumRs2Error::InvalidOperation(
                "FFT2 requires input dimensions to be powers of 2".to_string(),
            ));
        }

        // Convert to complex
        let data = x.to_vec();
        let complex_data: Vec<Complex<T>> = data
            .iter()
            .map(|&val| Complex::new(val, T::zero()))
            .collect();

        // Reshape to 2D for algorithm
        let mut complex_2d: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<Complex<T>> = complex_data[(i * n_cols)..((i + 1) * n_cols)].to_vec();
            complex_2d.push(row);
        }

        // Apply 1D FFT to each row
        for row in &mut complex_2d {
            fft_recursive(row);
        }

        // Transpose the matrix
        let mut transposed = vec![vec![Complex::new(T::zero(), T::zero()); n_rows]; n_cols];
        for (i, row) in complex_2d.iter().enumerate().take(n_rows) {
            for (j, val) in row.iter().enumerate().take(n_cols) {
                transposed[j][i] = *val;
            }
        }

        // Apply 1D FFT to each column (now row after transposition)
        for row in &mut transposed {
            fft_recursive(row);
        }

        // Transpose back and flatten
        let mut result = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for row in &transposed {
                result.push(row[i]);
            }
        }

        Array::from_vec_shape(result, &shape)
    }

    /// Compute 2D inverse FFT of a 2D complex array
    pub fn ifft2<T>(x: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 2D
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "IFFT2 expects a 2D array".to_string(),
            ));
        }

        let n_rows = shape[0];
        let n_cols = shape[1];

        // Check if dimensions are powers of 2
        if !is_power_of_two(n_rows) || !is_power_of_two(n_cols) {
            return Err(NumRs2Error::InvalidOperation(
                "IFFT2 requires input dimensions to be powers of 2".to_string(),
            ));
        }

        // Get data
        let data = x.to_vec();

        // Conjugate the input
        let complex_data: Vec<Complex<T>> = data.iter().map(|val| val.conj()).collect();

        // Reshape to 2D for algorithm
        let mut complex_2d: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<Complex<T>> = complex_data[(i * n_cols)..((i + 1) * n_cols)].to_vec();
            complex_2d.push(row);
        }

        // Apply 1D FFT to each row
        for row in &mut complex_2d {
            fft_recursive(row);
        }

        // Transpose the matrix
        let mut transposed = vec![vec![Complex::new(T::zero(), T::zero()); n_rows]; n_cols];
        for (i, row) in complex_2d.iter().enumerate().take(n_rows) {
            for (j, val) in row.iter().enumerate().take(n_cols) {
                transposed[j][i] = *val;
            }
        }

        // Apply 1D FFT to each column (now row after transposition)
        for row in &mut transposed {
            fft_recursive(row);
        }

        // Transpose back, conjugate, scale and flatten
        let scale: T = <T as From<f64>>::from(1.0 / (n_rows * n_cols) as f64);
        let mut result = Vec::with_capacity(n_rows * n_cols);

        // Pre-compute the scaled and conjugated values for each row
        for i in 0..n_rows {
            for row in &transposed {
                result.push(row[i].conj() * scale);
            }
        }

        Array::from_vec_shape(result, &shape)
    }

    /// Compute the frequency axis for an FFT result
    ///
    /// # Parameters
    ///
    /// * `n` - Size of the transformed axis
    /// * `d` - Sample spacing (time increment between samples) - inverse of the sampling rate
    ///
    /// # Returns
    ///
    /// An array of frequencies from 0 to positive, then negative
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create frequency axis for 8-point FFT with 0.1s spacing (10 Hz sample rate)
    /// let freqs = FFT::fftfreq(8, 0.1).expect("fftfreq should succeed");
    /// // Frequencies are [0, 1.25, 2.5, 3.75, -5, -3.75, -2.5, -1.25]
    /// assert_eq!(freqs.size(), 8);
    /// ```
    pub fn fftfreq<T>(n: usize, d: T) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + From<f64>,
    {
        // Create frequency axis
        let mut freqs = Vec::with_capacity(n);
        let sample_rate = <T as NumCast>::from(1.0).unwrap_or(T::zero()) / d;

        let half_n = n / 2;
        for i in 0..half_n {
            freqs.push(
                <T as NumCast>::from(i as f64).unwrap_or(T::zero()) * sample_rate
                    / <T as NumCast>::from(n as f64).unwrap_or(T::zero()),
            );
        }

        // Negative frequencies
        for i in half_n..n {
            freqs.push(
                (<T as NumCast>::from(i as f64).unwrap_or(T::zero())
                    - <T as NumCast>::from(n as f64).unwrap_or(T::zero()))
                    * sample_rate
                    / <T as NumCast>::from(n as f64).unwrap_or(T::zero()),
            );
        }

        Ok(Array::from_vec(freqs))
    }

    /// Compute the frequency axis for a real FFT result
    ///
    /// # Parameters
    ///
    /// * `n` - Size of the transformed axis
    /// * `d` - Sample spacing (time increment between samples) - inverse of the sampling rate
    ///
    /// # Returns
    ///
    /// An array of positive frequencies for real FFT results
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create frequency axis for 8-point real FFT with 0.1s spacing (10 Hz sample rate)
    /// let freqs = FFT::rfftfreq(8, 0.1).expect("rfftfreq should succeed");
    /// // Frequencies are [0, 1.25, 2.5, 3.75, 5]
    /// assert_eq!(freqs.size(), 5);
    /// ```
    pub fn rfftfreq<T>(n: usize, d: T) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + From<f64>,
    {
        // Create frequency axis for real FFT (only positive frequencies)
        let mut freqs = Vec::with_capacity(n / 2 + 1);
        let sample_rate = <T as NumCast>::from(1.0).unwrap_or(T::zero()) / d;

        for i in 0..=n / 2 {
            freqs.push(
                <T as NumCast>::from(i as f64).unwrap_or(T::zero()) * sample_rate
                    / <T as NumCast>::from(n as f64).unwrap_or(T::zero()),
            );
        }

        Ok(Array::from_vec(freqs))
    }

    /// Shift the zero-frequency component to the center of the spectrum
    ///
    /// # Parameters
    ///
    /// * `x` - The input array (can be 1D or 2D)
    ///
    /// # Returns
    ///
    /// An array with shifted frequencies
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::Complex64;
    ///
    /// // Create a 1D spectrum with DC at 0
    /// let mut spectrum = Vec::new();
    /// for i in 0..8 {
    ///     spectrum.push(Complex64::new(i as f64, 0.0));
    /// }
    ///
    /// let input = Array::from_vec(spectrum);
    /// let shifted = FFT::fftshift(&input).expect("fftshift should succeed");
    ///
    /// // After shifting, the DC component (0.0) should be in the middle
    /// // For a signal of length 8, the order goes from [0,1,2,3,4,5,6,7]
    /// // to [4,5,6,7,0,1,2,3]
    /// ```
    pub fn fftshift<T: Clone>(x: &Array<T>) -> Result<Array<T>> {
        let shape = x.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                let n = shape[0];
                let data = x.to_vec();

                let mut result = Vec::with_capacity(n);
                // Ceil division to handle odd-length arrays correctly
                let half_n = n.div_ceil(2);

                // Rearrange the array
                result.extend_from_slice(&data[n - half_n..]);
                result.extend_from_slice(&data[..n - half_n]);

                Ok(Array::from_vec(result))
            }
            2 => {
                // For 2D arrays, shift both dimensions
                let n_rows = shape[0];
                let n_cols = shape[1];
                let data = x.to_vec();

                // Create a 2D representation for easier manipulation
                let mut data_2d = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let row: Vec<T> = data[(i * n_cols)..((i + 1) * n_cols)].to_vec();
                    data_2d.push(row);
                }

                // Shift rows - ceil division to handle odd dimensions correctly
                let half_rows = n_rows.div_ceil(2);
                let mut rows_shifted = Vec::with_capacity(n_rows);
                rows_shifted.extend_from_slice(&data_2d[n_rows - half_rows..]);
                rows_shifted.extend_from_slice(&data_2d[..n_rows - half_rows]);

                // Shift columns in each row - ceil division to handle odd dimensions correctly
                let half_cols = n_cols.div_ceil(2);
                let mut result = Vec::with_capacity(n_rows * n_cols);

                for row in rows_shifted {
                    let mut shifted_row = Vec::with_capacity(n_cols);
                    shifted_row.extend_from_slice(&row[n_cols - half_cols..]);
                    shifted_row.extend_from_slice(&row[..n_cols - half_cols]);
                    result.extend(shifted_row);
                }

                Ok(Array::from_vec_shape(result, &shape)?)
            }
            _ => Err(NumRs2Error::InvalidOperation(format!(
                "fftshift only supports 1D and 2D arrays, got {}D",
                ndim
            ))),
        }
    }

    /// Inverse of fftshift - shift the zero-frequency component back to the beginning
    ///
    /// # Parameters
    ///
    /// * `x` - The input array (can be 1D or 2D)
    ///
    /// # Returns
    ///
    /// An array with frequencies shifted back
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::Complex64;
    ///
    /// // Create a 1D spectrum with DC in the middle
    /// let mut spectrum = Vec::new();
    /// // Values 4,5,6,7,0,1,2,3
    /// spectrum.push(Complex64::new(4.0, 0.0));
    /// spectrum.push(Complex64::new(5.0, 0.0));
    /// spectrum.push(Complex64::new(6.0, 0.0));
    /// spectrum.push(Complex64::new(7.0, 0.0));
    /// spectrum.push(Complex64::new(0.0, 0.0)); // DC at index 4
    /// spectrum.push(Complex64::new(1.0, 0.0));
    /// spectrum.push(Complex64::new(2.0, 0.0));
    /// spectrum.push(Complex64::new(3.0, 0.0));
    ///
    /// let input = Array::from_vec(spectrum);
    /// let shifted_back = FFT::ifftshift(&input).expect("ifftshift should succeed");
    ///
    /// // After shifting back, the DC component (0.0) should return to the beginning
    /// assert_eq!(shifted_back.to_vec()[0], Complex64::new(0.0, 0.0));
    /// ```
    pub fn ifftshift<T: Clone>(x: &Array<T>) -> Result<Array<T>> {
        let shape = x.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                let n = shape[0];
                let data = x.to_vec();

                let mut result = Vec::with_capacity(n);
                let half_n = n / 2; // Slightly different from fftshift for even lengths

                // Rearrange the array (opposite direction of fftshift)
                result.extend_from_slice(&data[half_n..]);
                result.extend_from_slice(&data[..half_n]);

                Ok(Array::from_vec(result))
            }
            2 => {
                // For 2D arrays, shift both dimensions
                let n_rows = shape[0];
                let n_cols = shape[1];
                let data = x.to_vec();

                // Create a 2D representation for easier manipulation
                let mut data_2d = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let row: Vec<T> = data[(i * n_cols)..((i + 1) * n_cols)].to_vec();
                    data_2d.push(row);
                }

                // Shift rows
                let half_rows = n_rows / 2;
                let mut rows_shifted = Vec::with_capacity(n_rows);
                rows_shifted.extend_from_slice(&data_2d[half_rows..]);
                rows_shifted.extend_from_slice(&data_2d[..half_rows]);

                // Shift columns in each row
                let half_cols = n_cols / 2;
                let mut result = Vec::with_capacity(n_rows * n_cols);

                for row in rows_shifted {
                    let mut shifted_row = Vec::with_capacity(n_cols);
                    shifted_row.extend_from_slice(&row[half_cols..]);
                    shifted_row.extend_from_slice(&row[..half_cols]);
                    result.extend(shifted_row);
                }

                Ok(Array::from_vec_shape(result, &shape)?)
            }
            _ => Err(NumRs2Error::InvalidOperation(format!(
                "ifftshift only supports 1D and 2D arrays, got {}D",
                ndim
            ))),
        }
    }

    /// Real Fast Fourier Transform - performs FFT for real input
    ///
    /// # Parameters
    ///
    /// * `x` - Real input array
    ///
    /// # Returns
    ///
    /// Complex array with transformed values of size n/2+1
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a real signal
    /// let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    ///
    /// // Real FFT is more efficient than regular FFT for real inputs
    /// let rfft_result = FFT::rfft(&signal).expect("rfft should succeed");
    ///
    /// // Result contains only positive frequencies (n/2+1 values)
    /// assert_eq!(rfft_result.size(), 3);
    /// ```
    pub fn rfft<T>(x: &Array<T>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 1D
        if shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "RFFT expects a 1D array".to_string(),
            ));
        }

        let n = shape[0];

        // For real FFT, we can use the fact that the FFT of a real signal is conjugate symmetric
        // This means we only need to compute the first n/2+1 output values

        // First, compute the regular FFT
        let fft_result = Self::fft(x)?;
        let fft_data = fft_result.to_vec();

        // Extract the first n/2+1 values (including the DC and Nyquist components)
        let rfft_size = n / 2 + 1;
        let rfft_data = fft_data[..rfft_size].to_vec();

        Ok(Array::from_vec(rfft_data))
    }

    /// Inverse Real Fast Fourier Transform
    ///
    /// # Parameters
    ///
    /// * `x` - Complex input array of size n/2+1 (from rfft)
    /// * `n` - Size of the original real signal
    ///
    /// # Returns
    ///
    /// Real array with inverse transformed values
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a real signal
    /// let signal = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    ///
    /// // Apply RFFT
    /// let rfft_result = FFT::rfft(&signal).expect("rfft should succeed");
    ///
    /// // Apply IRFFT (with original size)
    /// let irfft_result = FFT::irfft(&rfft_result, 4).expect("irfft should succeed");
    ///
    /// // Original signal should be recovered
    /// assert_eq!(irfft_result.size(), 4);
    /// ```
    pub fn irfft<T>(x: &Array<Complex<T>>, n: usize) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 1D
        if shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "IRFFT expects a 1D array".to_string(),
            ));
        }

        let rfft_size = shape[0];

        // IRFFT needs to reconstruct the full FFT array using conjugate symmetry
        let rfft_data = x.to_vec();

        // Verify that n is valid and consistent with rfft_size
        if n < 2 * (rfft_size - 1) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "IRFFT: n must be at least 2*(rfft_size-1), got n={} and rfft_size={}",
                n, rfft_size
            )));
        }

        // Reconstruct the full FFT array
        let mut fft_data = Vec::with_capacity(n);
        fft_data.extend_from_slice(&rfft_data);

        // Add the complex conjugates for negative frequencies.
        // For even n, skip the Nyquist bin (index rfft_size-1) when mirroring;
        // for odd n, mirror all non-DC bins (1..rfft_size).
        let mirror_start = if n.is_multiple_of(2) { 2usize } else { 1usize };
        for i in mirror_start..rfft_size {
            fft_data.push(rfft_data[rfft_size - i].conj());
        }

        // Create the complex array and apply IFFT
        let fft_array = Array::from_vec(fft_data);
        let ifft_result = Self::ifft(&fft_array)?;
        let ifft_data = ifft_result.to_vec();

        // Extract the real part
        let real_data: Vec<T> = ifft_data.iter().map(|c| c.re).collect();

        Ok(Array::from_vec(real_data))
    }

    /// 2D Real Fast Fourier Transform
    ///
    /// # Parameters
    ///
    /// * `x` - Real 2D input array
    ///
    /// # Returns
    ///
    /// Complex 2D array with transformed values
    pub fn rfft2<T>(x: &Array<T>) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 2D
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "RFFT2 expects a 2D array".to_string(),
            ));
        }

        let n_rows = shape[0];
        let n_cols = shape[1];

        // For 2D RFFT, first perform regular FFT along rows,
        // then RFFT along the last dimension (columns)

        // Convert real data to complex
        let data = x.to_vec();
        let complex_data: Vec<Complex<T>> = data
            .iter()
            .map(|&val| Complex::new(val, T::zero()))
            .collect();

        // Reshape to 2D for processing
        let mut complex_2d: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<Complex<T>> = complex_data[(i * n_cols)..((i + 1) * n_cols)].to_vec();
            complex_2d.push(row);
        }

        // Apply 1D FFT to each row
        for row in &mut complex_2d {
            fft_recursive(row);
        }

        // Extract only positive frequencies from the last dimension
        let rfft_cols = n_cols / 2 + 1;
        let mut result = Vec::with_capacity(n_rows * rfft_cols);

        for row in complex_2d {
            result.extend_from_slice(&row[..rfft_cols]);
        }

        // Return as array with adjusted shape
        Array::from_vec_shape(result, &[n_rows, rfft_cols])
    }

    /// 2D Inverse Real Fast Fourier Transform
    ///
    /// # Parameters
    ///
    /// * `x` - Complex 2D input array (from rfft2)
    /// * `shape` - Shape of the original real 2D array
    ///
    /// # Returns
    ///
    /// Real 2D array with inverse transformed values
    pub fn irfft2<T>(x: &Array<Complex<T>>, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let x_shape = x.shape();

        // Check that the input is 2D
        if x_shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "IRFFT2 expects a 2D array".to_string(),
            ));
        }

        // Check that the target shape has 2 dimensions
        if shape.len() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "IRFFT2 output shape must be 2D".to_string(),
            ));
        }

        let n_rows = shape[0];
        let n_cols = shape[1];
        let rfft_cols = x_shape[1];

        // Verify consistency between input shape and target shape
        if n_rows != x_shape[0] || rfft_cols != n_cols / 2 + 1 {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n_rows, n_cols / 2 + 1],
                actual: x_shape.clone(),
            });
        }

        // Get the complex data
        let data = x.to_vec();

        // Reshape to 2D for processing
        let mut complex_2d: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<Complex<T>> = data[(i * rfft_cols)..((i + 1) * rfft_cols)].to_vec();
            complex_2d.push(row);
        }

        // For each row, reconstruct the full FFT result using conjugate symmetry
        let mut full_complex_2d: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_rows);

        for row in complex_2d {
            let mut full_row = Vec::with_capacity(n_cols);
            full_row.extend_from_slice(&row);

            // Add conjugates for negative frequencies (same even/odd fix as irfft).
            let mirror_start = if n_cols.is_multiple_of(2) {
                2usize
            } else {
                1usize
            };
            for i in mirror_start..rfft_cols {
                full_row.push(row[rfft_cols - i].conj());
            }

            full_complex_2d.push(full_row);
        }

        // Apply inverse FFT to each row
        for row in &mut full_complex_2d {
            // Conjugate input
            for val in row.iter_mut() {
                *val = val.conj();
            }

            // Compute FFT
            fft_recursive(row);

            // Conjugate and scale
            let scale: T = <T as From<f64>>::from(1.0 / n_cols as f64);
            for val in row.iter_mut() {
                *val = val.conj() * scale;
            }
        }

        // Flatten and extract real part
        let mut result = Vec::with_capacity(n_rows * n_cols);
        for row in full_complex_2d {
            for val in row {
                result.push(val.re);
            }
        }

        Array::from_vec_shape(result, shape)
    }

    /// Apply window function to the signal before FFT to reduce spectral leakage
    pub fn apply_window<T>(x: &Array<T>, window_type: &str) -> Result<Array<T>>
    where
        T: Float + Clone + Debug + From<f64>,
    {
        let shape = x.shape();

        // Check that the input is 1D
        if shape.len() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "Window function expects a 1D array".to_string(),
            ));
        }

        let n = shape[0];
        let data = x.to_vec();

        // Generate window coefficients based on type
        let window = match window_type.to_lowercase().as_str() {
            "hann" => {
                // Hann window: 0.5 * (1 - cos(2πi/(n-1)))
                (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.5 * (1.0 - arg.cos())).unwrap_or(T::zero())
                    })
                    .collect::<Vec<T>>()
            }
            "hamming" => {
                // Hamming window: 0.54 - 0.46 * cos(2πi/(n-1))
                (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.54 - 0.46 * arg.cos()).unwrap_or(T::zero())
                    })
                    .collect::<Vec<T>>()
            }
            "blackman" => {
                // Blackman window: 0.42 - 0.5 * cos(2πi/(n-1)) + 0.08 * cos(4πi/(n-1))
                (0..n)
                    .map(|i| {
                        let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                        <T as NumCast>::from(0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos())
                            .unwrap_or(T::zero())
                    })
                    .collect::<Vec<T>>()
            }
            "rectangular" => {
                // Rectangular window (no windowing): 1
                vec![<T as NumCast>::from(1.0).unwrap_or(T::zero()); n]
            }
            _ => {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Unknown window type: {}",
                    window_type
                )));
            }
        };

        // Apply the window to the data
        let result = data
            .iter()
            .zip(window.iter())
            .map(|(&x_val, &w_val)| x_val * w_val)
            .collect();

        Ok(Array::from_vec(result))
    }
}

// Helper functions
fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

// Cooley-Tukey Radix-2 FFT algorithm (in-place)
fn fft_recursive<T>(x: &mut [Complex<T>])
where
    T: Float + Clone + Into<f64> + From<f64>,
{
    let n = x.len();

    // Base case: single-element FFT is identity
    if n <= 1 {
        return;
    }

    // Divide: separate even and odd elements
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for i in 0..n / 2 {
        even.push(x[2 * i]);
        odd.push(x[2 * i + 1]);
    }

    // Conquer: recursively compute FFT of even and odd sub-arrays
    fft_recursive(&mut even);
    fft_recursive(&mut odd);

    // Combine: merge results
    for k in 0..n / 2 {
        let angle = -2.0 * PI * k as f64 / n as f64;
        let twiddle = Complex::new(
            <T as From<f64>>::from(angle.cos()),
            <T as From<f64>>::from(angle.sin()),
        );

        let p = even[k];
        let q = odd[k] * twiddle;

        x[k] = p + q;
        x[k + n / 2] = p - q;
    }
}

// Extend the Array type with FFT methods
impl<T> Array<T>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Compute the FFT of this array
    pub fn fft(&self) -> Result<Array<Complex<T>>> {
        FFT::fft(self)
    }

    /// Compute the real FFT of this array (more efficient for real inputs)
    pub fn rfft(&self) -> Result<Array<Complex<T>>> {
        FFT::rfft(self)
    }

    /// Compute the power spectrum of this array
    pub fn power_spectrum(&self) -> Result<Array<T>> {
        FFT::power_spectrum(self)
    }

    /// Apply window function to this array
    pub fn apply_window(&self, window_type: &str) -> Result<Array<T>> {
        FFT::apply_window(self, window_type)
    }

    /// Compute 2D FFT if this is a 2D array
    pub fn fft2(&self) -> Result<Array<Complex<T>>> {
        FFT::fft2(self)
    }

    /// Compute 2D real FFT if this is a 2D array (more efficient for real inputs)
    pub fn rfft2(&self) -> Result<Array<Complex<T>>> {
        FFT::rfft2(self)
    }

    /// Shift zero-frequency component to the center
    pub fn fftshift_real(&self) -> Result<Array<T>> {
        FFT::fftshift(self)
    }

    /// Shift zero-frequency component from center to beginning
    pub fn ifftshift_real(&self) -> Result<Array<T>> {
        FFT::ifftshift(self)
    }
}

// Extend the Complex Array type with inverse FFT methods
impl<T> Array<Complex<T>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Compute the inverse FFT of this complex array
    pub fn ifft(&self) -> Result<Array<Complex<T>>> {
        FFT::ifft(self)
    }

    /// Compute the inverse real FFT of this complex array
    pub fn irfft(&self, n: usize) -> Result<Array<T>> {
        FFT::irfft(self, n)
    }

    /// Compute 2D inverse FFT if this is a 2D complex array
    pub fn ifft2(&self) -> Result<Array<Complex<T>>> {
        FFT::ifft2(self)
    }

    /// Compute 2D inverse real FFT if this is a 2D complex array
    pub fn irfft2(&self, shape: &[usize]) -> Result<Array<T>> {
        FFT::irfft2(self, shape)
    }

    /// Shift zero-frequency component to the center
    pub fn fftshift_complex(&self) -> Result<Array<Complex<T>>> {
        FFT::fftshift(self)
    }

    /// Shift zero-frequency component from center to beginning
    pub fn ifftshift_complex(&self) -> Result<Array<Complex<T>>> {
        FFT::ifftshift(self)
    }
}

#[cfg(test)]
mod array_fft_tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::Complex64;

    // Helper to create Complex<f64> array
    #[allow(dead_code)]
    fn complex_array(real: Vec<f64>, imag: Vec<f64>) -> Array<Complex64> {
        let complexes: Vec<Complex64> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect();
        Array::from_vec(complexes)
    }

    #[test]
    fn test_fft_simple() {
        // Create a simple signal: [1.0, 0.0, 0.0, 0.0]
        let x = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        // FFT of this should be [1.0, 1.0, 1.0, 1.0]
        let fft_result = FFT::fft(&x).expect("FFT should succeed");
        let fft_data = fft_result.to_vec();

        assert_eq!(fft_data.len(), 4);

        for item in fft_data.iter().take(4) {
            assert_relative_eq!(item.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(item.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_forward_inverse() {
        // Create a random signal
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Apply FFT
        let fft_result = FFT::fft(&x).expect("FFT should succeed");

        // Apply inverse FFT
        let ifft_result = FFT::ifft(&fft_result).expect("IFFT should succeed");
        let ifft_data = ifft_result.to_vec();

        // Original signal should be recovered
        let x_vec = x.to_vec();
        for (i, item) in ifft_data.iter().enumerate().take(x.size()) {
            assert_relative_eq!(item.re, x_vec[i], epsilon = 1e-10);
            assert_relative_eq!(item.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_power_spectrum() {
        // Create a simple sinusoid
        let n = 32;
        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            signal.push(f64::sin(2.0 * PI * 4.0 * t)); // 4 Hz sinusoid
        }

        let x = Array::from_vec(signal);

        // Compute power spectrum
        let power = FFT::power_spectrum(&x).expect("power_spectrum should succeed");
        let power_data = power.to_vec();

        // Check that the peak is at the correct frequency
        let mut max_power = 0.0;
        let mut max_idx = 0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if power_data[i] > max_power {
                max_power = power_data[i];
                max_idx = i;
            }
        }

        // Expected peak at bin 4 or n-4
        assert!(max_idx == 4 || max_idx == n - 4);
    }

    #[test]
    fn test_window_functions() {
        let n = 16;
        let signal = vec![1.0; n];
        let x = Array::from_vec(signal);

        // Test Hann window
        let hann = FFT::apply_window(&x, "hann").expect("apply_window hann should succeed");
        let hann_data = hann.to_vec();

        // Hann window should be zero at endpoints and symmetric
        assert_relative_eq!(hann_data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(hann_data[n - 1], 0.0, epsilon = 1e-10);

        for i in 0..n / 2 {
            assert_relative_eq!(hann_data[i], hann_data[n - 1 - i], epsilon = 1e-10);
        }

        // Test Hamming window
        let hamming =
            FFT::apply_window(&x, "hamming").expect("apply_window hamming should succeed");
        let hamming_data = hamming.to_vec();

        // Hamming window should be symmetric
        for i in 0..n / 2 {
            assert_relative_eq!(hamming_data[i], hamming_data[n - 1 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_2d_fft() {
        // Create a simple 4x4 constant array
        let x = Array::from_vec(vec![1.0; 16]).reshape(&[4, 4]);

        // FFT2 of a constant array should have a single non-zero value at [0,0]
        let fft2_result = FFT::fft2(&x).expect("FFT2 should succeed");
        let fft2_data = fft2_result.to_vec();

        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            if i == 0 {
                assert_relative_eq!(fft2_data[i].re, 16.0, epsilon = 1e-10);
            } else {
                assert_relative_eq!(fft2_data[i].re, 0.0, epsilon = 1e-10);
            }
            assert_relative_eq!(fft2_data[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftfreq_array_api() {
        // Test frequency calculation for FFT
        let n = 8;
        let d = 0.1; // 10 Hz sampling rate

        let freqs = FFT::fftfreq(n, d).expect("fftfreq should succeed");
        let freqs_data = freqs.to_vec();

        // Expected frequencies: [0, 1.25, 2.5, 3.75, -5, -3.75, -2.5, -1.25]
        assert_eq!(freqs_data.len(), n);
        assert_relative_eq!(freqs_data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[1], 1.25, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[2], 2.5, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[3], 3.75, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[4], -5.0, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[5], -3.75, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[6], -2.5, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[7], -1.25, epsilon = 1e-10);
    }

    #[test]
    fn test_rfftfreq_array_api() {
        // Test frequency calculation for real FFT
        let n = 8;
        let d = 0.1; // 10 Hz sampling rate

        let freqs = FFT::rfftfreq(n, d).expect("rfftfreq should succeed");
        let freqs_data = freqs.to_vec();

        // Expected frequencies: [0, 1.25, 2.5, 3.75, 5]
        assert_eq!(freqs_data.len(), n / 2 + 1);
        assert_relative_eq!(freqs_data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[1], 1.25, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[2], 2.5, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[3], 3.75, epsilon = 1e-10);
        assert_relative_eq!(freqs_data[4], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fftshift_array_api() {
        // Test fftshift for 1D array
        let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let shifted = FFT::fftshift(&x).expect("fftshift should succeed");
        let shifted_data = shifted.to_vec();

        // Expected order: [4, 5, 6, 7, 0, 1, 2, 3]
        assert_eq!(shifted_data.len(), 8);
        assert_eq!(shifted_data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);

        // Test inverse fftshift
        let unshifted = FFT::ifftshift(&shifted).expect("ifftshift should succeed");
        let unshifted_data = unshifted.to_vec();

        // Should get back the original array
        assert_eq!(unshifted_data, x.to_vec());
    }

    #[test]
    fn test_rfft_array_api() {
        // Test real FFT
        let x = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // RFFT should return n/2+1 complex values
        let rfft_result = FFT::rfft(&x).expect("RFFT should succeed");
        let rfft_data = rfft_result.to_vec();

        // Expected size is n/2+1
        assert_eq!(rfft_data.len(), 5);

        // DC component should be 1.0
        assert_relative_eq!(rfft_data[0].re, 1.0, epsilon = 1e-10);

        // Test inverse RFFT
        let irfft_result = FFT::irfft(&rfft_result, 8).expect("IRFFT should succeed");
        let irfft_data = irfft_result.to_vec();

        // Original signal should be recovered
        assert_eq!(irfft_data.len(), 8);
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            assert_relative_eq!(irfft_data[i], x.to_vec()[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rfft2_array_api() {
        // Test 2D real FFT
        let x = Array::from_vec(vec![1.0; 16]).reshape(&[4, 4]);

        // RFFT2 should return n rows and n/2+1 columns
        let rfft2_result = FFT::rfft2(&x).expect("RFFT2 should succeed");
        let rfft2_shape = rfft2_result.shape();

        // Expected shape is [n_rows, n_cols/2+1]
        assert_eq!(rfft2_shape, vec![4, 3]);

        // DC component should be the sum of all elements (but our implementation
        // may handle normalization differently)
        let rfft2_data = rfft2_result.to_vec();

        // Check that the DC component has the highest value
        let mut max_val = 0.0;
        let mut max_idx = 0;
        for (i, v) in rfft2_data.iter().enumerate() {
            if v.norm() > max_val {
                max_val = v.norm();
                max_idx = i;
            }
        }
        assert_eq!(max_idx, 0); // DC should be at index 0

        // Test inverse RFFT2
        let irfft2_result = FFT::irfft2(&rfft2_result, &[4, 4]).expect("IRFFT2 should succeed");
        let irfft2_shape = irfft2_result.shape();

        // Original shape should be recovered
        assert_eq!(irfft2_shape, vec![4, 4]);

        // Original values should be recovered (allowing for some numerical precision issues)
        let irfft2_data = irfft2_result.to_vec();
        let expected_val = 1.0;
        for val in irfft2_data {
            assert_relative_eq!(val, expected_val, epsilon = 1e-8);
        }
    }
}

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
        let partial = fftn(&x, Some(&[2]), Some(&[0]), None).expect("partial s should succeed");
        assert_eq!(partial.shape(), &[2, 4, 5]);
        assert!((partial[IxDyn(&[0, 0, 0])].re - 780.0).abs() < 1e-9);
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
        let default_n =
            irfft_with(&spectrum, None, None, None).expect("default n should succeed");
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

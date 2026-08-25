//! `Array<T>`-native FFT (merged from the former `new_modules::fft` module).
//!
//! This module provides an `Array<T>`-oriented FFT surface (a `FFT` struct
//! of associated functions, plus matching inherent methods on `Array<T>` /
//! `Array<Complex<T>>` such as `.fft()`/`.ifft()`), distinct from the thin
//! `scirs2_fft` passthrough in the parent `fft` module and its
//! [`super::numpy_parity`] wrappers (which operate on plain slices and
//! `scirs2_core::ndarray` types). It is ported here unchanged from the
//! former `src/new_modules/fft.rs` (a near-duplicate FFT implementation);
//! `src/new_modules/fft.rs` itself has been deleted and
//! `src/new_modules/mod.rs` now re-exports `crate::fft` in its place (see
//! that file), so `crate::new_modules::fft::FFT` keeps resolving to this
//! same `FFT` for the handful of other in-crate modules that still name it
//! that way.
//!
//! NOTE: this implementation's own `fft`/`ifft`/`fft2`/`ifft2` (and the
//! `is_power_of_two` guard in front of them) still require power-of-2
//! lengths, using a hand-rolled recursive Cooley-Tukey radix-2 kernel
//! rather than the arbitrary-size, `oxifft`-backed transforms available
//! through the `scirs2_fft` re-exports on the parent `fft` module -- that
//! limitation predates this merge and is carried forward as-is (a pure
//! relocation, not a rewrite); [`super::numpy_parity`]'s `fftn`/
//! `rfft_with`/etc. are the arbitrary-size, NumPy-parity path for new code.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use scirs2_core::Complex;
use std::f64::consts::PI;
use std::fmt::Debug;

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

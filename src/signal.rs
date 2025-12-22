//! Signal Processing Module
//!
//! This module provides signal processing functionality including FFT operations,
//! frequency domain analysis, and related mathematical transforms.
//!
//! ## Features
//!
//! - Fast Fourier Transform (FFT) and inverse FFT
//! - Enhanced FFT operations with frequency domain utilities
//! - Real and complex FFT variants
//! - Multi-dimensional FFT support

// Re-export functionality from the former new_modules
pub use crate::new_modules::fft::*;

// Additional enhanced FFT functionality
pub mod enhanced {
    //! Enhanced FFT operations with additional frequency domain utilities
    pub use crate::new_modules::fft_enhanced::*;
}

use crate::array::Array;
use crate::error::Result;
use num_traits::{Float, Zero};
use scirs2_core::parallel_ops::*;
use std::ops::{Add, Mul};

/// Compute the 1D convolution of two arrays
///
/// The discrete convolution operation is defined as:
/// `(a * b)[n] = sum(a[m] * b[n - m])` for m from 0 to M-1
/// where M is the length of array a.
///
/// # Parameters
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `mode` - Mode of convolution:
///   - "full": Output size is M + N - 1 (default)
///   - "valid": Output size is max(M, N) - min(M, N) + 1
///   - "same": Output size is max(M, N)
///
/// # Returns
///
/// The convolution of `a` and `b`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![0.0, 1.0, 0.5]);
/// let c = convolve(&a, &b, "full").unwrap();
/// assert_eq!(c.to_vec(), vec![0.0, 1.0, 2.5, 4.0, 1.5]);
///
/// let c_valid = convolve(&a, &b, "valid").unwrap();
/// assert_eq!(c_valid.to_vec(), vec![2.5]);
/// ```
pub fn convolve<T>(a: &Array<T>, b: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Float + Clone + Zero + Send + Sync,
{
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(crate::error::NumRs2Error::DimensionMismatch(
            "convolve requires 1D arrays".to_string(),
        ));
    }

    let a_len = a.len();
    let b_len = b.len();

    if a_len == 0 || b_len == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    let (output_size, start_idx) = match mode {
        "full" => (a_len + b_len - 1, 0),
        "valid" => {
            if a_len >= b_len {
                (a_len - b_len + 1, b_len - 1)
            } else {
                (b_len - a_len + 1, a_len - 1)
            }
        }
        "same" => {
            let size = a_len.max(b_len);
            let start = if a_len >= b_len {
                (b_len - 1) / 2
            } else {
                (a_len - 1) / 2
            };
            (size, start)
        }
        _ => {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Invalid mode '{}'. Use 'full', 'valid', or 'same'",
                mode
            )))
        }
    };

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    // Use parallel processing for large convolutions
    const PARALLEL_THRESHOLD: usize = 1000;
    let full_len = a_len + b_len - 1;

    let full_result: Vec<T> = if full_len >= PARALLEL_THRESHOLD {
        // Parallel computation of convolution
        (0..full_len)
            .into_par_iter()
            .map(|n| {
                let mut sum = T::zero();
                let m_start = n.saturating_sub(b_len - 1);
                let m_end = (n + 1).min(a_len);
                for m in m_start..m_end {
                    sum = sum + a_vec[m] * b_vec[n - m];
                }
                sum
            })
            .collect()
    } else {
        // Sequential computation for small arrays
        let mut result = Vec::with_capacity(full_len);
        for n in 0..full_len {
            let mut sum = T::zero();
            let m_start = n.saturating_sub(b_len - 1);
            let m_end = (n + 1).min(a_len);
            for m in m_start..m_end {
                sum = sum + a_vec[m] * b_vec[n - m];
            }
            result.push(sum);
        }
        result
    };

    // Extract the appropriate portion based on mode
    let result = match mode {
        "full" => full_result,
        _ => full_result[start_idx..start_idx + output_size].to_vec(),
    };

    Ok(Array::from_vec(result))
}

/// Compute the cross-correlation of two 1D arrays
///
/// The correlation is defined as:
/// `(a ⋆ b)[n] = sum(a[m] * conj(b[m - n]))` for all m
///
/// This is related to convolution by: correlate(a, b) = convolve(a, b[::-1])
///
/// # Parameters
///
/// * `a` - First input array
/// * `b` - Second input array  
/// * `mode` - Mode of correlation:
///   - "full": Output size is M + N - 1 (default)
///   - "valid": Output size is max(M, N) - min(M, N) + 1
///   - "same": Output size is max(M, N)
///
/// # Returns
///
/// The cross-correlation of `a` and `b`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Array::from_vec(vec![0.0, 1.0, 0.5]);
/// let c = correlate(&a, &b, "full").unwrap();
/// // Note: correlation is convolution with second array reversed
///
/// let c_valid = correlate(&a, &b, "valid").unwrap();
/// ```
pub fn correlate<T>(a: &Array<T>, b: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Float + Clone + Zero + Send + Sync,
{
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(crate::error::NumRs2Error::DimensionMismatch(
            "correlate requires 1D arrays".to_string(),
        ));
    }

    // Reverse array b
    let b_reversed = Array::from_vec(b.to_vec().into_iter().rev().collect());

    // Correlation is convolution with reversed second array
    convolve(a, &b_reversed, mode)
}

/// Compute the 2D convolution of two arrays
///
/// The 2D discrete convolution operation extends the 1D convolution to 2D arrays.
/// Each output element is computed as the sum of element-wise products of the
/// input array with a shifted version of the kernel.
///
/// # Parameters
///
/// * `input` - Input 2D array
/// * `kernel` - 2D kernel/filter array
/// * `mode` - Mode of convolution:
///   - "full": Output size is (M + P - 1, N + Q - 1) where input is MxN and kernel is PxQ
///   - "valid": Output contains only elements that do not rely on zero-padding
///   - "same": Output is the same size as input (centered)
///
/// # Returns
///
/// The 2D convolution of `input` and `kernel`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::convolve2d;
///
/// let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let kernel = Array::from_vec(vec![1.0, 0.0, 0.0, -1.0]).reshape(&[2, 2]);
/// let result = convolve2d(&input, &kernel, "full").expect("convolve2d failed");
/// // Applies 2D convolution with the kernel
/// ```
pub fn convolve2d<T>(input: &Array<T>, kernel: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Float + Clone + Zero + Add<Output = T> + Mul<Output = T> + Send + Sync,
{
    if input.ndim() != 2 || kernel.ndim() != 2 {
        return Err(crate::error::NumRs2Error::DimensionMismatch(
            "convolve2d requires 2D arrays".to_string(),
        ));
    }

    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    let (m, n) = (input_shape[0], input_shape[1]);
    let (p, q) = (kernel_shape[0], kernel_shape[1]);

    if m == 0 || n == 0 || p == 0 || q == 0 {
        return Ok(Array::zeros(&[0, 0]));
    }

    // Determine output size based on mode
    let (output_rows, output_cols, row_offset, col_offset) = match mode {
        "full" => (m + p - 1, n + q - 1, 0, 0),
        "valid" => {
            if m < p || n < q {
                return Ok(Array::zeros(&[0, 0]));
            }
            (m - p + 1, n - q + 1, p - 1, q - 1)
        }
        "same" => (m, n, (p - 1) / 2, (q - 1) / 2),
        _ => {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Invalid mode '{}'. Use 'full', 'valid', or 'same'",
                mode
            )))
        }
    };

    let input_data = input.to_vec();
    let kernel_data = kernel.to_vec();

    // Use parallel processing for larger outputs
    const PARALLEL_THRESHOLD: usize = 256;
    let total_output = output_rows * output_cols;

    let result = if total_output >= PARALLEL_THRESHOLD {
        use scirs2_core::parallel_ops::*;

        // Parallel 2D convolution
        (0..total_output)
            .into_par_iter()
            .map(|idx| {
                let out_i = idx / output_cols;
                let out_j = idx % output_cols;
                let mut sum = T::zero();

                // Convolve at this output position
                for k_i in 0..p {
                    for k_j in 0..q {
                        // Calculate input indices
                        let in_i = out_i + k_i;
                        let in_j = out_j + k_j;

                        // Adjust for mode offsets
                        let adj_in_i = in_i as isize - row_offset as isize;
                        let adj_in_j = in_j as isize - col_offset as isize;

                        // Check bounds
                        if adj_in_i >= 0
                            && adj_in_i < m as isize
                            && adj_in_j >= 0
                            && adj_in_j < n as isize
                        {
                            let in_idx = adj_in_i as usize * n + adj_in_j as usize;
                            let k_idx = k_i * q + k_j;
                            sum = sum + input_data[in_idx] * kernel_data[k_idx];
                        }
                    }
                }

                sum
            })
            .collect()
    } else {
        // Sequential 2D convolution for small outputs
        let mut result = vec![T::zero(); total_output];
        for out_i in 0..output_rows {
            for out_j in 0..output_cols {
                let mut sum = T::zero();

                // Convolve at this output position
                for k_i in 0..p {
                    for k_j in 0..q {
                        // Calculate input indices
                        let in_i = out_i + k_i;
                        let in_j = out_j + k_j;

                        // Adjust for mode offsets
                        let adj_in_i = in_i as isize - row_offset as isize;
                        let adj_in_j = in_j as isize - col_offset as isize;

                        // Check bounds
                        if adj_in_i >= 0
                            && adj_in_i < m as isize
                            && adj_in_j >= 0
                            && adj_in_j < n as isize
                        {
                            let in_idx = adj_in_i as usize * n + adj_in_j as usize;
                            let k_idx = k_i * q + k_j;
                            sum = sum + input_data[in_idx] * kernel_data[k_idx];
                        }
                    }
                }

                result[out_i * output_cols + out_j] = sum;
            }
        }
        result
    };

    Ok(Array::from_vec(result).reshape(&[output_rows, output_cols]))
}

/// Compute the 2D cross-correlation of two arrays
///
/// The 2D correlation is similar to 2D convolution but without flipping the kernel.
/// This is equivalent to convolve2d with a flipped kernel.
///
/// # Parameters
///
/// * `input` - Input 2D array
/// * `template` - 2D template array to correlate with
/// * `mode` - Mode of correlation:
///   - "full": Output size includes all positions where arrays overlap
///   - "valid": Output contains only elements that do not rely on zero-padding
///   - "same": Output is the same size as input
///
/// # Returns
///
/// The 2D cross-correlation of `input` and `template`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::correlate2d;
///
/// let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let template = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
/// let result = correlate2d(&input, &template, "valid").expect("correlate2d failed");
/// // Finds where template matches in the input
/// ```
pub fn correlate2d<T>(input: &Array<T>, template: &Array<T>, mode: &str) -> Result<Array<T>>
where
    T: Float + Clone + Zero + Add<Output = T> + Mul<Output = T> + Send + Sync,
{
    if input.ndim() != 2 || template.ndim() != 2 {
        return Err(crate::error::NumRs2Error::DimensionMismatch(
            "correlate2d requires 2D arrays".to_string(),
        ));
    }

    // Flip the template both horizontally and vertically
    let template_shape = template.shape();
    let (rows, cols) = (template_shape[0], template_shape[1]);
    let template_data = template.to_vec();

    let mut flipped_data = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            // Flip both dimensions
            let orig_idx = i * cols + j;
            let flip_idx = (rows - 1 - i) * cols + (cols - 1 - j);
            flipped_data[flip_idx] = template_data[orig_idx];
        }
    }

    let flipped_template = Array::from_vec(flipped_data).reshape(&[rows, cols]);

    // Correlation is convolution with flipped template
    convolve2d(input, &flipped_template, mode)
}

/// Unwrap phase angles by changing deltas between values to 2*pi complement
///
/// This function unwraps a phase array by adding or subtracting 2π when
/// the difference between consecutive phase values exceeds the discontinuity
/// threshold (default: π).
///
/// # Arguments
/// * `phase` - Array of phase values (typically in radians)
/// * `discont` - Maximum discontinuity between values (default: π)
/// * `axis` - Axis along which to unwrap (if None, unwrap flattened array)
/// * `period` - Size of the period (default: 2π)
///
/// # Returns
/// * `Result<Array<T>>` - Unwrapped phase array
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::unwrap;
/// use std::f64::consts::PI;
///
/// // Phase with discontinuity
/// let phase = Array::from_vec(vec![0.0, 0.5, 1.0, -2.5, -2.0, -1.5]);
/// let unwrapped = unwrap(&phase, None, None, None).unwrap();
/// // Discontinuity at index 3 is unwrapped by adding 2π
/// ```
pub fn unwrap<T>(
    phase: &Array<T>,
    discont: Option<T>,
    axis: Option<usize>,
    period: Option<T>,
) -> Result<Array<T>>
where
    T: Float + std::fmt::Debug,
{
    let pi = T::from(std::f64::consts::PI).unwrap();
    let two_pi = pi + pi;

    let discont = discont.unwrap_or(pi);
    let period = period.unwrap_or(two_pi);
    let half_period = period / T::from(2.0).unwrap();

    match axis {
        None => {
            // Unwrap flattened array using cumulative correction approach
            // This is O(n) instead of O(n^2) for the nested loop version
            let data = phase.to_vec();
            if data.is_empty() {
                return Ok(Array::from_vec(vec![]));
            }

            let mut result = Vec::with_capacity(data.len());
            result.push(data[0]);
            let mut cumulative_correction = T::zero();

            // Single pass: track cumulative correction instead of re-applying
            for i in 1..data.len() {
                let adjusted = data[i] + cumulative_correction;
                let diff = adjusted - result[i - 1];

                // Check for discontinuity
                if diff.abs() > discont {
                    // Calculate the number of periods to add/subtract
                    let correction = if diff > T::zero() {
                        -((diff + half_period) / period).floor() * period
                    } else {
                        -((diff - half_period) / period).floor() * period
                    };
                    cumulative_correction = cumulative_correction + correction;
                }

                result.push(data[i] + cumulative_correction);
            }

            Ok(Array::from_vec(result).reshape(&phase.shape()))
        }
        Some(ax) => {
            let shape = phase.shape();
            if ax >= shape.len() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax,
                    shape.len()
                )));
            }

            let ndim = shape.len();

            // Calculate strides for iteration
            let axis_len = shape[ax];
            if axis_len <= 1 {
                return Ok(phase.clone()); // Nothing to unwrap
            }

            // Number of 1D arrays to process along the axis
            let mut outer_shape = shape.clone();
            outer_shape.remove(ax);
            let n_arrays: usize = outer_shape.iter().product();

            if n_arrays == 0 {
                return Ok(phase.clone());
            }

            // Convert to flat Vec for efficient manipulation
            let phase_data = phase.to_vec();
            let mut result_data = phase_data.clone();

            // Compute strides for the array
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Precompute outer_strides for multi-index computation
            let mut outer_strides = vec![1usize; outer_shape.len()];
            if !outer_shape.is_empty() {
                for i in (0..outer_shape.len() - 1).rev() {
                    outer_strides[i] = outer_strides[i + 1] * outer_shape[i + 1];
                }
            }

            // Process each 1D array along the specified axis
            for array_idx in 0..n_arrays {
                // Compute base index in flat array for this 1D slice
                let mut base_idx = 0usize;
                let mut temp = array_idx;
                let mut outer_dim_idx = 0;

                for dim in 0..ndim {
                    if dim != ax {
                        let dim_size = if outer_dim_idx < outer_strides.len() {
                            (temp / outer_strides[outer_dim_idx]) % outer_shape[outer_dim_idx]
                        } else {
                            temp % outer_shape[outer_dim_idx]
                        };
                        base_idx += dim_size * strides[dim];
                        if outer_dim_idx < outer_strides.len() {
                            temp %= outer_strides[outer_dim_idx];
                        }
                        outer_dim_idx += 1;
                    }
                }

                // Extract 1D slice along axis using stride
                let axis_stride = strides[ax];

                // Unwrap using cumulative correction (O(axis_len) instead of O(axis_len^2))
                let mut cumulative_correction = T::zero();
                let mut prev_val = result_data[base_idx];

                for i in 1..axis_len {
                    let idx = base_idx + i * axis_stride;
                    let adjusted = result_data[idx] + cumulative_correction;
                    let diff = adjusted - prev_val;

                    if diff.abs() > discont {
                        let correction = if diff > T::zero() {
                            -((diff + half_period) / period).floor() * period
                        } else {
                            -((diff - half_period) / period).floor() * period
                        };
                        cumulative_correction = cumulative_correction + correction;
                    }

                    let new_val = result_data[idx] + cumulative_correction;
                    result_data[idx] = new_val;
                    prev_val = new_val;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&shape))
        }
    }
}

/// Return the Bartlett window
///
/// The Bartlett window is a triangular window; it is zero at the beginning and end,
/// with a linear ramp in between. Also known as the triangular window.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window, with the maximum value normalized to 1 (though the value 1
/// does not appear if `m` is even and the window is symmetric)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::bartlett;
///
/// let window = bartlett::<f64>(12).unwrap();
/// assert_eq!(window.len(), 12);
/// // Bartlett window is symmetric and triangular
/// assert_eq!(window.get(&[0]).unwrap(), 0.0);
/// assert_eq!(window.get(&[11]).unwrap(), 0.0);
/// ```
pub fn bartlett<T>(m: usize) -> Result<Array<T>>
where
    T: Float + Clone + std::fmt::Debug + num_traits::FromPrimitive,
{
    if m == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if m == 1 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let val = if i < m / 2 {
            2.0 * i as f64 / (m - 1) as f64
        } else {
            2.0 - 2.0 * i as f64 / (m - 1) as f64
        };
        window.push(T::from_f64(val).unwrap_or(T::zero()));
    }

    Ok(Array::from_vec(window))
}

/// Return a Blackman window
///
/// The Blackman window is a taper formed by using a raised cosine or sine-squared
/// with three terms. The Blackman window function is:
/// w(n) = 0.42 - 0.5*cos(2*pi*n/(M-1)) + 0.08*cos(4*pi*n/(M-1))
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window, with the maximum value normalized to 1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::blackman;
///
/// let window = blackman::<f64>(12).unwrap();
/// assert_eq!(window.len(), 12);
/// // Blackman window has minimum side lobe leakage
/// ```
pub fn blackman<T>(m: usize) -> Result<Array<T>>
where
    T: Float + Clone + std::fmt::Debug + num_traits::FromPrimitive,
{
    if m == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if m == 1 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let arg = 2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64;
        let val = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
        // Clamp to [0, 1] to handle floating point precision issues
        let clamped_val = val.clamp(0.0, 1.0);
        window.push(T::from_f64(clamped_val).unwrap_or(T::zero()));
    }

    Ok(Array::from_vec(window))
}

/// Return a Hanning window
///
/// The Hanning window is a taper formed by using a raised cosine or sine-squared
/// with two terms:
/// w(n) = 0.5 - 0.5*cos(2*pi*n/(M-1)) = 0.5*(1 - cos(2*pi*n/(M-1)))
///
/// The Hanning window is defined as:
/// w(n) = sin²(π*n/(M-1)) = 0.5*(1 - cos(2*π*n/(M-1)))
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window, with the maximum value normalized to 1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::hanning;
///
/// let window = hanning::<f64>(12).unwrap();
/// assert_eq!(window.len(), 12);
/// // Hanning window is smooth and tapered
/// assert_eq!(window.get(&[0]).unwrap(), 0.0);
/// assert_eq!(window.get(&[11]).unwrap(), 0.0);
/// ```
pub fn hanning<T>(m: usize) -> Result<Array<T>>
where
    T: Float + Clone + std::fmt::Debug + num_traits::FromPrimitive,
{
    if m == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if m == 1 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let arg = 2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64;
        let val = 0.5 * (1.0 - arg.cos());
        window.push(T::from_f64(val).unwrap_or(T::zero()));
    }

    Ok(Array::from_vec(window))
}

/// Return a Hamming window
///
/// The Hamming window is a taper formed by using a raised cosine with two terms:
/// w(n) = 0.54 - 0.46*cos(2*pi*n/(M-1))
///
/// The Hamming window was named for R. W. Hamming, an associate of J. W. Tukey and
/// is described as a modification of the Hanning window.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window, with the maximum value normalized to 1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::hamming;
///
/// let window = hamming::<f64>(12).unwrap();
/// assert_eq!(window.len(), 12);
/// // Hamming window has non-zero endpoints
/// assert!(window.get(&[0]).unwrap() > 0.0);
/// assert!(window.get(&[11]).unwrap() > 0.0);
/// ```
pub fn hamming<T>(m: usize) -> Result<Array<T>>
where
    T: Float + Clone + std::fmt::Debug + num_traits::FromPrimitive,
{
    if m == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if m == 1 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let arg = 2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64;
        let val = 0.54 - 0.46 * arg.cos();
        window.push(T::from_f64(val).unwrap_or(T::zero()));
    }

    Ok(Array::from_vec(window))
}

/// Return a Kaiser window
///
/// The Kaiser window is a taper formed by using a Bessel function.
/// It was designed to be optimal in the sense of the highest possible
/// spectral concentration in the main lobe compared to the side lobes.
///
/// The Kaiser window is defined as:
/// w(n) = I₀(β*sqrt(1-(2*n/(M-1)-1)²)) / I₀(β)
///
/// where I₀ is the modified zeroth-order Bessel function of the first kind.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
/// * `beta` - Shape parameter, determines the trade-off between main-lobe width
///   and side lobe level. As beta gets large, the window narrows.
///   Default is 8.6 if None provided.
///
/// # Returns
///
/// The window, with the maximum value normalized to 1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::signal::kaiser;
///
/// let window = kaiser::<f64>(12, Some(8.6)).unwrap();
/// assert_eq!(window.len(), 12);
///
/// // Kaiser window with different beta values
/// let window_narrow = kaiser::<f64>(12, Some(14.0)).unwrap();
/// let window_wide = kaiser::<f64>(12, Some(2.0)).unwrap();
/// ```
pub fn kaiser<T>(m: usize, beta: Option<T>) -> Result<Array<T>>
where
    T: Float + Clone + std::fmt::Debug + num_traits::FromPrimitive,
{
    if m == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if m == 1 {
        return Ok(Array::from_vec(vec![T::one()]));
    }

    let beta_val = beta.unwrap_or(T::from_f64(8.6).unwrap_or(T::zero()));
    let beta_f64 = beta_val.to_f64().unwrap_or(8.6);

    // Pre-compute I₀(β) for normalization
    let i0_beta = modified_bessel_i0(beta_f64);

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let x = 2.0 * i as f64 / (m - 1) as f64 - 1.0;
        let arg = beta_f64 * (1.0 - x * x).sqrt();
        let val = modified_bessel_i0(arg) / i0_beta;
        window.push(T::from_f64(val).unwrap_or(T::zero()));
    }

    Ok(Array::from_vec(window))
}

/// Modified Bessel function of the first kind, order 0
///
/// This is a helper function for the Kaiser window computation.
/// Uses rational approximation for computational efficiency.
fn modified_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        let y = x / 3.75;
        let y2 = y * y;
        1.0 + 3.5156229 * y2
            + 3.0899424 * y2 * y2
            + 1.2067492 * y2 * y2 * y2
            + 0.2659732 * y2 * y2 * y2 * y2
            + 0.0360768 * y2 * y2 * y2 * y2 * y2
            + 0.0045813 * y2 * y2 * y2 * y2 * y2 * y2
    } else {
        let y = 3.75 / ax;
        let result = 0.39894228 + 0.01328592 * y + 0.00225319 * y * y - 0.00157565 * y * y * y
            + 0.00916281 * y * y * y * y
            - 0.02057706 * y * y * y * y * y
            + 0.02635537 * y * y * y * y * y * y
            - 0.01647633 * y * y * y * y * y * y * y
            + 0.00392377 * y * y * y * y * y * y * y * y;

        (ax.exp() / ax.sqrt()) * result
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bartlett_window() {
        let window = bartlett::<f64>(10).unwrap();
        let data = window.to_vec();

        assert_eq!(data.len(), 10);

        // Bartlett window should be zero at endpoints
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data[9], 0.0, epsilon = 1e-10);

        // Should be symmetric
        for i in 0..5 {
            assert_relative_eq!(data[i], data[9 - i], epsilon = 1e-10);
        }

        // Peak should be at center (for odd lengths)
        let window_odd = bartlett::<f64>(9).unwrap();
        let data_odd = window_odd.to_vec();
        assert_relative_eq!(data_odd[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_blackman_window() {
        let window = blackman::<f64>(10).unwrap();
        let data = window.to_vec();

        assert_eq!(data.len(), 10);

        // Blackman window should be close to zero at endpoints
        assert!(data[0] < 0.01);
        assert!(data[9] < 0.01);

        // Should be symmetric
        for i in 0..5 {
            assert_relative_eq!(data[i], data[9 - i], epsilon = 1e-10);
        }

        // All values should be positive and <= 1
        for &val in &data {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_hanning_window() {
        let window = hanning::<f64>(10).unwrap();
        let data = window.to_vec();

        assert_eq!(data.len(), 10);

        // Hanning window should be zero at endpoints
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data[9], 0.0, epsilon = 1e-10);

        // Should be symmetric
        for i in 0..5 {
            assert_relative_eq!(data[i], data[9 - i], epsilon = 1e-10);
        }

        // All values should be in [0, 1]
        for &val in &data {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_hamming_window() {
        let window = hamming::<f64>(10).unwrap();
        let data = window.to_vec();

        assert_eq!(data.len(), 10);

        // Hamming window should be non-zero at endpoints (characteristic feature)
        assert!(data[0] > 0.0);
        assert!(data[9] > 0.0);

        // Should be symmetric
        for i in 0..5 {
            assert_relative_eq!(data[i], data[9 - i], epsilon = 1e-10);
        }

        // All values should be positive and <= 1
        for &val in &data {
            assert!(val > 0.0 && val <= 1.0);
        }

        // Endpoints should be approximately 0.08 (54% - 46%)
        assert_relative_eq!(data[0], 0.08, epsilon = 1e-10);
        assert_relative_eq!(data[9], 0.08, epsilon = 1e-10);
    }

    #[test]
    fn test_kaiser_window() {
        let window = kaiser::<f64>(10, Some(8.6)).unwrap();
        let data = window.to_vec();

        assert_eq!(data.len(), 10);

        // Should be symmetric
        for i in 0..5 {
            assert_relative_eq!(data[i], data[9 - i], epsilon = 1e-10);
        }

        // All values should be positive and <= 1
        for &val in &data {
            assert!(val > 0.0 && val <= 1.0);
        }

        // Center should be peak value (approximately 1.0)
        let max_val = data.iter().fold(0.0, |a, &b| a.max(b));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-1);

        // Test with different beta values
        let window_narrow = kaiser::<f64>(10, Some(14.0)).unwrap();
        let window_wide = kaiser::<f64>(10, Some(2.0)).unwrap();

        assert_eq!(window_narrow.len(), 10);
        assert_eq!(window_wide.len(), 10);
    }

    #[test]
    fn test_window_edge_cases() {
        // Test empty windows
        assert_eq!(bartlett::<f64>(0).unwrap().len(), 0);
        assert_eq!(blackman::<f64>(0).unwrap().len(), 0);
        assert_eq!(hanning::<f64>(0).unwrap().len(), 0);
        assert_eq!(hamming::<f64>(0).unwrap().len(), 0);
        assert_eq!(kaiser::<f64>(0, Some(8.6)).unwrap().len(), 0);

        // Test single-point windows
        assert_relative_eq!(
            bartlett::<f64>(1).unwrap().get(&[0]).unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            blackman::<f64>(1).unwrap().get(&[0]).unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            hanning::<f64>(1).unwrap().get(&[0]).unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            hamming::<f64>(1).unwrap().get(&[0]).unwrap(),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            kaiser::<f64>(1, Some(8.6)).unwrap().get(&[0]).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_modified_bessel_i0() {
        // Test some known values
        assert_relative_eq!(modified_bessel_i0(0.0), 1.0, epsilon = 1e-10);

        // Test positive and negative values (should be the same)
        let val = 2.5;
        assert_relative_eq!(
            modified_bessel_i0(val),
            modified_bessel_i0(-val),
            epsilon = 1e-10
        );

        // Test that function returns reasonable values
        assert!(modified_bessel_i0(1.0) > 1.0);
        assert!(modified_bessel_i0(5.0) > modified_bessel_i0(1.0));
    }
}

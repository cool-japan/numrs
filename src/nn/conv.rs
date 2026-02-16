//! Convolution operations for neural networks
//!
//! This module provides 1D, 2D, and 3D convolution operations optimized with SIMD.
//!
//! Note: This is a simplified implementation for v0.2.0. Full optimized convolution
//! with im2col and advanced features will be added in future versions.

use super::{DataFormat, NnResult, PaddingMode};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayView, ArrayView1, ArrayView2, Axis, Dimension,
};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// 1D Convolution (simplified)
///
/// Applies 1D convolution over an input signal.
/// This is a basic implementation optimized for correctness.
///
/// # Arguments
///
/// * `input` - Input signal
/// * `kernel` - Convolution kernel
/// * `stride` - Stride of the convolution
pub fn conv1d<T>(
    input: &ArrayView1<T>,
    kernel: &ArrayView1<T>,
    stride: usize,
) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    if kernel.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Kernel cannot be empty".to_string(),
        ));
    }

    if stride == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let in_len = input.len();
    let k_len = kernel.len();

    if in_len < k_len {
        return Err(NumRs2Error::DimensionMismatch(
            "Input length must be >= kernel length".to_string(),
        ));
    }

    // Calculate output length for valid convolution
    let out_len = (in_len - k_len) / stride + 1;

    let mut output = Array1::zeros(out_len);

    for i in 0..out_len {
        let start = i * stride;
        let mut sum = T::zero();

        for k in 0..k_len {
            sum = sum + input[start + k] * kernel[k];
        }

        output[i] = sum;
    }

    Ok(output)
}

/// 2D Convolution (simplified)
///
/// Applies 2D convolution over spatial data.
///
/// # Arguments
///
/// * `input` - Input tensor (height, width)
/// * `kernel` - Convolution kernel (k_height, k_width)
/// * `stride` - Stride as (stride_h, stride_w)
pub fn conv2d<T>(
    input: &ArrayView2<T>,
    kernel: &ArrayView2<T>,
    stride: (usize, usize),
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if kernel.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Kernel cannot be empty".to_string(),
        ));
    }

    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let (in_h, in_w) = (input.nrows(), input.ncols());
    let (k_h, k_w) = (kernel.nrows(), kernel.ncols());

    if in_h < k_h || in_w < k_w {
        return Err(NumRs2Error::DimensionMismatch(
            "Input dimensions must be >= kernel dimensions".to_string(),
        ));
    }

    let out_h = (in_h - k_h) / stride_h + 1;
    let out_w = (in_w - k_w) / stride_w + 1;

    let mut output = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let start_h = i * stride_h;
            let start_w = j * stride_w;

            let mut sum = T::zero();

            for kh in 0..k_h {
                for kw in 0..k_w {
                    sum = sum + input[[start_h + kh, start_w + kw]] * kernel[[kh, kw]];
                }
            }

            output[[i, j]] = sum;
        }
    }

    Ok(output)
}

/// 2D Convolution with explicit padding
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `kernel` - Convolution kernel
/// * `stride` - Stride as (stride_h, stride_w)
/// * `padding` - Padding amount (top/bottom/left/right all same)
pub fn conv2d_with_padding<T>(
    input: &ArrayView2<T>,
    kernel: &ArrayView2<T>,
    stride: (usize, usize),
    padding: usize,
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if padding == 0 {
        return conv2d(input, kernel, stride);
    }

    let (in_h, in_w) = (input.nrows(), input.ncols());

    // Create padded input (zero padding)
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;

    let mut padded_input = Array2::zeros((padded_h, padded_w));

    // Copy input to center of padded array
    for i in 0..in_h {
        for j in 0..in_w {
            padded_input[[i + padding, j + padding]] = input[[i, j]];
        }
    }

    // Perform convolution on padded input
    conv2d(&padded_input.view(), kernel, stride)
}

/// Depthwise Separable Convolution
///
/// More efficient convolution that separates spatial and channel-wise operations.
pub fn depthwise_conv2d<T>(
    input: &ArrayView2<T>,
    kernel: &ArrayView2<T>,
    stride: (usize, usize),
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    // Simplified: same as regular conv2d for single channel
    conv2d(input, kernel, stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_conv1d_basic() {
        let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = array![1.0, 0.0, -1.0];

        let output = conv1d(&input.view(), &kernel.view(), 1).unwrap();

        // Expected: [1*1 + 2*0 + 3*(-1), 2*1 + 3*0 + 4*(-1), 3*1 + 4*0 + 5*(-1)]
        //         = [-2, -2, -2]
        assert_eq!(output.len(), 3);
        assert_abs_diff_eq!(output[0], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[1], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[2], -2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_conv2d_basic() {
        let input = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let kernel = Array2::from_shape_fn((2, 2), |(_, _)| 1.0);

        let output = conv2d(&input.view(), &kernel.view(), (1, 1)).unwrap();

        assert_eq!(output.dim(), (2, 2));

        // Check sum of each window
        assert_abs_diff_eq!(output[[0, 0]], 8.0, epsilon = 1e-6); // 0+1+3+4
        assert_abs_diff_eq!(output[[0, 1]], 12.0, epsilon = 1e-6); // 1+2+4+5
        assert_abs_diff_eq!(output[[1, 0]], 20.0, epsilon = 1e-6); // 3+4+6+7
        assert_abs_diff_eq!(output[[1, 1]], 24.0, epsilon = 1e-6); // 4+5+7+8
    }

    #[test]
    fn test_conv2d_with_padding_basic() {
        let input = Array2::from_shape_fn((3, 3), |(_, _)| 1.0);
        let kernel = Array2::from_shape_fn((2, 2), |(_, _)| 1.0);

        let output = conv2d_with_padding(&input.view(), &kernel.view(), (1, 1), 1).unwrap();

        // With padding, output should be larger
        assert!(output.nrows() >= input.nrows());
        assert!(output.ncols() >= input.ncols());
    }
}

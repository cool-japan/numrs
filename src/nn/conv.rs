//! Convolution operations for neural networks
//!
//! This module provides 1D, 2D, and 3D convolution operations optimized with SIMD.
//!
//! Note: Full implementation using scirs2-linalg im2col algorithm.

use super::{DataFormat, NnResult, PaddingMode};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayView, ArrayView1, ArrayView2, ArrayView4, Axis,
    Dimension, ScalarOperand,
};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use scirs2_core::simd_ops::SimdUnifiedOps;
pub use scirs2_linalg::convolution::{col2im, im2col};
use scirs2_linalg::convolution::{conv2d_im2col, conv_transpose2d};
use scirs2_linalg::error::LinalgError;
use std::iter::Sum;

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

/// Depthwise 2D convolution (single-channel, backward compatible)
///
/// For a single channel, depthwise convolution is the same as regular conv2d.
/// For multi-channel depthwise convolution, use `depthwise_conv2d_batched`.
pub fn depthwise_conv2d<T>(
    input: &ArrayView2<T>,
    kernel: &ArrayView2<T>,
    stride: (usize, usize),
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    conv2d(input, kernel, stride)
}

/// Depthwise 2D convolution (NCHW format)
///
/// Each input channel is convolved independently with its own kernel.
/// Input: (batch, channels, height, width)
/// Kernel: (channels, 1, kernel_h, kernel_w) - one kernel per channel
/// Output: (batch, channels, out_h, out_w)
pub fn depthwise_conv2d_batched<T>(
    input: &ArrayView4<T>,
    kernel: &ArrayView4<T>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> NnResult<Array4<T>>
where
    T: Float + SimdUnifiedOps + NumAssign + Zero + ScalarOperand + Sum,
{
    let (batch, channels, in_h, in_w) = input.dim();
    let (k_out, k_depth, k_h, k_w) = kernel.dim();

    // Validate kernel shape: (channels, 1, kh, kw)
    if k_out != channels {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Kernel first dimension ({}) must match input channels ({})",
            k_out, channels
        )));
    }
    if k_depth != 1 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Kernel depth must be 1 for depthwise convolution, got {}",
            k_depth
        )));
    }

    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let (pad_h, pad_w) = padding;
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;

    if padded_h < k_h || padded_w < k_w {
        return Err(NumRs2Error::DimensionMismatch(
            "Padded input dimensions must be >= kernel dimensions".to_string(),
        ));
    }

    let out_h = (padded_h - k_h) / stride_h + 1;
    let out_w = (padded_w - k_w) / stride_w + 1;

    let mut output = Array4::<T>::zeros((batch, channels, out_h, out_w));

    for b in 0..batch {
        for c in 0..channels {
            let kernel_2d = kernel.slice(scirs2_core::ndarray::s![c, 0, .., ..]);

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = T::zero();
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            // Handle padding: check if within original input bounds
                            if ih >= pad_h && ih < pad_h + in_h && iw >= pad_w && iw < pad_w + in_w
                            {
                                sum += input[[b, c, ih - pad_h, iw - pad_w]] * kernel_2d[[kh, kw]];
                            }
                        }
                    }
                    output[[b, c, oh, ow]] = sum;
                }
            }
        }
    }

    Ok(output)
}

/// Grouped 2D convolution (NCHW format)
///
/// Splits input channels into groups, each group convolved independently.
/// Input: (batch, in_channels, H, W)
/// Kernel: (out_channels, in_channels/groups, kH, kW)
/// groups must divide both in_channels and out_channels evenly.
///
/// When groups == in_channels and out_channels == in_channels, this is
/// equivalent to depthwise convolution.
pub fn grouped_conv2d<T>(
    input: &ArrayView4<T>,
    kernel: &ArrayView4<T>,
    groups: usize,
    stride: (usize, usize),
    padding: (usize, usize),
) -> NnResult<Array4<T>>
where
    T: Float + SimdUnifiedOps + NumAssign + Zero + ScalarOperand + Sum,
{
    let (batch, in_channels, in_h, in_w) = input.dim();
    let (out_channels, k_in_per_group, k_h, k_w) = kernel.dim();

    if groups == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Groups must be positive".to_string(),
        ));
    }
    if in_channels % groups != 0 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "in_channels ({}) must be divisible by groups ({})",
            in_channels, groups
        )));
    }
    if out_channels % groups != 0 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "out_channels ({}) must be divisible by groups ({})",
            out_channels, groups
        )));
    }

    let in_per_group = in_channels / groups;
    if k_in_per_group != in_per_group {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Kernel in_channels/group ({}) must equal in_channels/groups ({})",
            k_in_per_group, in_per_group
        )));
    }

    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let (pad_h, pad_w) = padding;
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;

    if padded_h < k_h || padded_w < k_w {
        return Err(NumRs2Error::DimensionMismatch(
            "Padded input dimensions must be >= kernel dimensions".to_string(),
        ));
    }

    let out_h = (padded_h - k_h) / stride_h + 1;
    let out_w = (padded_w - k_w) / stride_w + 1;
    let out_per_group = out_channels / groups;

    let mut output = Array4::<T>::zeros((batch, out_channels, out_h, out_w));

    for b in 0..batch {
        for g in 0..groups {
            let in_start = g * in_per_group;
            let out_start = g * out_per_group;

            for oc in 0..out_per_group {
                let abs_oc = out_start + oc;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::zero();
                        for ic in 0..in_per_group {
                            let abs_ic = in_start + ic;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;

                                    if ih >= pad_h
                                        && ih < pad_h + in_h
                                        && iw >= pad_w
                                        && iw < pad_w + in_w
                                    {
                                        sum += input[[b, abs_ic, ih - pad_h, iw - pad_w]]
                                            * kernel[[abs_oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        output[[b, abs_oc, oh, ow]] = sum;
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Batched 2D convolution using im2col algorithm (NCHW format)
///
/// Significantly faster than naive loop-based convolution by reformulating
/// the convolution as a matrix multiplication via the im2col transformation.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch, in_channels, height, width)
/// * `kernel` - Convolution kernel of shape (out_channels, in_channels, kernel_h, kernel_w)
/// * `bias` - Optional bias of shape (out_channels,)
/// * `stride` - Stride as (stride_h, stride_w)
/// * `padding` - Padding as (pad_h, pad_w)
/// * `dilation` - Dilation as (dilation_h, dilation_w)
///
/// # Returns
///
/// Output tensor of shape (batch, out_channels, out_h, out_w)
pub fn conv2d_batched<T>(
    input: &Array4<T>,
    kernel: &Array4<T>,
    bias: Option<scirs2_core::ndarray::ArrayView1<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> NnResult<Array4<T>>
where
    T: Float + NumAssign + Sum + Zero + ScalarOperand + SimdUnifiedOps,
{
    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    conv2d_im2col(
        &input.view(),
        &kernel.view(),
        bias,
        stride,
        padding,
        dilation,
    )
    .map_err(|e| NumRs2Error::ComputationError(e.to_string()))
}

/// Transposed 2D convolution (deconvolution) using NCHW format
///
/// Computes the transposed (gradient) convolution, also known as deconvolution,
/// commonly used in upsampling layers of encoder-decoder networks.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch, in_channels, height, width)
/// * `kernel` - Kernel of shape (in_channels, out_channels, kernel_h, kernel_w)
/// * `bias` - Optional bias of shape (out_channels,)
/// * `stride` - Stride as (stride_h, stride_w)
/// * `padding` - Padding as (pad_h, pad_w)
/// * `output_padding` - Additional size added to one side of the output shape
/// * `dilation` - Dilation as (dilation_h, dilation_w)
///
/// # Returns
///
/// Output tensor of shape (batch, out_channels, out_h, out_w)
pub fn conv_transpose2d_batched<T>(
    input: &Array4<T>,
    kernel: &Array4<T>,
    bias: Option<scirs2_core::ndarray::ArrayView1<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
) -> NnResult<Array4<T>>
where
    T: Float + NumAssign + Sum + Zero + ScalarOperand + SimdUnifiedOps,
{
    let (stride_h, stride_w) = stride;
    if stride_h == 0 || stride_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    conv_transpose2d(
        &input.view(),
        &kernel.view(),
        bias,
        stride,
        padding,
        output_padding,
        dilation,
    )
    .map_err(|e| NumRs2Error::ComputationError(e.to_string()))
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

        let output = conv1d(&input.view(), &kernel.view(), 1).expect("test: valid conv1d params");

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

        let output =
            conv2d(&input.view(), &kernel.view(), (1, 1)).expect("test: valid conv2d params");

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

        let output = conv2d_with_padding(&input.view(), &kernel.view(), (1, 1), 1)
            .expect("test: valid conv2d params");

        // With padding, output should be larger
        assert!(output.nrows() >= input.nrows());
        assert!(output.ncols() >= input.ncols());
    }

    #[test]
    fn test_conv2d_batched_basic() {
        use scirs2_core::ndarray::Array4;
        // input: (1, 1, 4, 4), kernel: (1, 1, 2, 2)
        let input = Array4::<f64>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w) as f64);
        let kernel = Array4::<f64>::from_shape_fn((1, 1, 2, 2), |(_, _, _, _)| 1.0);

        let output = conv2d_batched(&input, &kernel, None, (1, 1), (0, 0), (1, 1))
            .expect("test: valid conv2d_batched params");

        // Output shape: (1, 1, 3, 3)
        assert_eq!(output.dim(), (1, 1, 3, 3));

        // Top-left patch sum: 0+1+4+5 = 10
        use approx::assert_abs_diff_eq;
        assert_abs_diff_eq!(output[[0, 0, 0, 0]], 10.0, epsilon = 1e-6);
        // Top-right patch sum: 2+3+6+7 = 18
        assert_abs_diff_eq!(output[[0, 0, 0, 2]], 18.0, epsilon = 1e-6);
    }

    #[test]
    fn test_depthwise_conv2d_batched_basic() {
        use scirs2_core::ndarray::Array4;
        // 1 batch, 2 channels, 4x4 spatial
        let input = Array4::<f64>::from_shape_fn((1, 2, 4, 4), |(_, c, h, w)| {
            ((c + 1) * (h * 4 + w + 1)) as f64
        });
        // 2 kernels (one per channel), depth=1, 2x2
        let kernel = Array4::<f64>::from_shape_fn(
            (2, 1, 2, 2),
            |(c, _, _, _)| {
                if c == 0 {
                    1.0
                } else {
                    -1.0
                }
            },
        );

        let output = depthwise_conv2d_batched(&input.view(), &kernel.view(), (1, 1), (0, 0))
            .expect("depthwise conv should succeed");

        // Output shape: (1, 2, 3, 3)
        assert_eq!(output.dim(), (1, 2, 3, 3));

        // Channel 0, kernel all 1s: sum of 2x2 patch at (0,0) = 1+2+5+6 = 14
        assert_abs_diff_eq!(output[[0, 0, 0, 0]], 14.0, epsilon = 1e-6);

        // Channel 1, kernel all -1s: sum of 2x2 patch at (0,0) for channel 1
        // channel 1 values: 2*(h*4+w+1), so (0,0)=2, (0,1)=4, (1,0)=10, (1,1)=12
        // with kernel=-1: -(2+4+10+12) = -28
        assert_abs_diff_eq!(output[[0, 1, 0, 0]], -28.0, epsilon = 1e-6);
    }

    #[test]
    fn test_depthwise_conv2d_batched_with_padding() {
        use scirs2_core::ndarray::Array4;
        let input = Array4::<f64>::ones((1, 1, 3, 3));
        let kernel = Array4::<f64>::ones((1, 1, 3, 3));

        let output = depthwise_conv2d_batched(&input.view(), &kernel.view(), (1, 1), (1, 1))
            .expect("padded depthwise conv should succeed");

        // With padding=1 on 3x3 input with 3x3 kernel, output is 3x3
        assert_eq!(output.dim(), (1, 1, 3, 3));

        // Center element: full overlap, 9 ones = 9
        assert_abs_diff_eq!(output[[0, 0, 1, 1]], 9.0, epsilon = 1e-6);

        // Corner element: only 1x1 overlap = 1 (top-left corner with padding)
        // Actually, (0,0) with pad=1: only positions (0..3)x(0..3) in padded coords
        // that fall in [1..4)x[1..4) in padded space => (1,1),(1,2),(2,1),(2,2) => 4 elements
        assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_depthwise_conv2d_batched_stride() {
        use scirs2_core::ndarray::Array4;
        let input = Array4::<f64>::ones((1, 1, 4, 4));
        let kernel = Array4::<f64>::ones((1, 1, 2, 2));

        let output = depthwise_conv2d_batched(&input.view(), &kernel.view(), (2, 2), (0, 0))
            .expect("strided depthwise conv should succeed");

        // (4-2)/2 + 1 = 2
        assert_eq!(output.dim(), (1, 1, 2, 2));
        // All ones, 2x2 kernel => each output = 4
        assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_depthwise_conv2d_batched_invalid_kernel() {
        use scirs2_core::ndarray::Array4;
        let input = Array4::<f64>::ones((1, 2, 4, 4));
        // Wrong: 3 kernels for 2 channels
        let kernel = Array4::<f64>::ones((3, 1, 2, 2));
        let result = depthwise_conv2d_batched(&input.view(), &kernel.view(), (1, 1), (0, 0));
        assert!(result.is_err());

        // Wrong: depth != 1
        let kernel2 = Array4::<f64>::ones((2, 2, 2, 2));
        let result2 = depthwise_conv2d_batched(&input.view(), &kernel2.view(), (1, 1), (0, 0));
        assert!(result2.is_err());
    }

    #[test]
    fn test_grouped_conv2d_groups_1() {
        use scirs2_core::ndarray::Array4;
        // groups=1 should be equivalent to standard convolution
        let input =
            Array4::<f64>::from_shape_fn((1, 2, 4, 4), |(_, c, h, w)| (c * 16 + h * 4 + w) as f64);
        let kernel = Array4::<f64>::ones((1, 2, 2, 2));

        let grouped = grouped_conv2d(&input.view(), &kernel.view(), 1, (1, 1), (0, 0))
            .expect("grouped conv groups=1 should succeed");

        // Compare with standard batched conv
        let standard = conv2d_batched(&input, &kernel, None, (1, 1), (0, 0), (1, 1))
            .expect("standard conv should succeed");

        assert_eq!(grouped.dim(), standard.dim());
        for ((idx, &g), &s) in grouped.indexed_iter().zip(standard.iter()) {
            assert_abs_diff_eq!(g, s, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_grouped_conv2d_depthwise_equivalence() {
        use scirs2_core::ndarray::Array4;
        // groups == in_channels => depthwise
        let channels = 3;
        let input = Array4::<f64>::from_shape_fn((1, channels, 4, 4), |(_, c, h, w)| {
            ((c + 1) * (h * 4 + w + 1)) as f64
        });
        // Depthwise: kernel (channels, 1, kh, kw)
        let dw_kernel =
            Array4::<f64>::from_shape_fn((channels, 1, 2, 2), |(c, _, _, _)| (c + 1) as f64);

        let dw_result = depthwise_conv2d_batched(&input.view(), &dw_kernel.view(), (1, 1), (0, 0))
            .expect("depthwise should succeed");

        // Grouped with groups=channels, same kernel
        let grouped_result =
            grouped_conv2d(&input.view(), &dw_kernel.view(), channels, (1, 1), (0, 0))
                .expect("grouped conv should succeed");

        assert_eq!(dw_result.dim(), grouped_result.dim());
        for (&d, &g) in dw_result.iter().zip(grouped_result.iter()) {
            assert_abs_diff_eq!(d, g, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_grouped_conv2d_two_groups() {
        use scirs2_core::ndarray::Array4;
        // 4 input channels, 2 groups => 2 channels per group
        // 4 output channels => 2 output channels per group
        let input = Array4::<f64>::ones((1, 4, 3, 3));
        // kernel: (4 out, 2 in_per_group, 2, 2)
        let kernel = Array4::<f64>::ones((4, 2, 2, 2));

        let output = grouped_conv2d(&input.view(), &kernel.view(), 2, (1, 1), (0, 0))
            .expect("grouped conv 2 groups should succeed");

        assert_eq!(output.dim(), (1, 4, 2, 2));
        // Each output: 2 input channels * 4 elements (2x2 kernel) = 8
        assert_abs_diff_eq!(output[[0, 0, 0, 0]], 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[0, 3, 1, 1]], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_grouped_conv2d_invalid_groups() {
        use scirs2_core::ndarray::Array4;
        let input = Array4::<f64>::ones((1, 3, 4, 4));
        let kernel = Array4::<f64>::ones((3, 1, 2, 2));

        // groups=2 doesn't divide in_channels=3
        let result = grouped_conv2d(&input.view(), &kernel.view(), 2, (1, 1), (0, 0));
        assert!(result.is_err());

        // groups=0
        let result = grouped_conv2d(&input.view(), &kernel.view(), 0, (1, 1), (0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_transpose2d_batched_basic() {
        use scirs2_core::ndarray::Array4;
        // input: (1, 1, 2, 2), kernel: (1, 1, 2, 2) — in_channels first for transpose
        let input =
            Array4::<f64>::from_shape_fn((1, 1, 2, 2), |(_, _, h, w)| (h * 2 + w + 1) as f64);
        let kernel = Array4::<f64>::from_shape_fn((1, 1, 2, 2), |(_, _, _, _)| 1.0);

        let output =
            conv_transpose2d_batched(&input, &kernel, None, (1, 1), (0, 0), (0, 0), (1, 1))
                .expect("test: valid conv_transpose2d_batched params");

        // Transposed conv with stride=1, no padding, 2x2 input, 2x2 kernel -> 3x3 output
        assert_eq!(output.dim(), (1, 1, 3, 3));
    }
}

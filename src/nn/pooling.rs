//! Pooling operations for neural networks
//!
//! This module provides max pooling, average pooling, and adaptive pooling operations.

use super::NnResult;
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Max Pooling 2D
///
/// Applies max pooling over 2D spatial data.
///
/// # Arguments
///
/// * `x` - Input tensor (height, width) or (batch, channels, height, width)
/// * `pool_size` - Size of the pooling window (e.g., (2, 2))
/// * `stride` - Stride of the pooling operation
pub fn max_pool2d<T>(
    x: &ArrayView2<T>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let (h, w) = (x.nrows(), x.ncols());
    let (ph, pw) = pool_size;
    let (sh, sw) = stride;

    if ph == 0 || pw == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Pool size must be positive".to_string(),
        ));
    }

    if sh == 0 || sw == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let out_h = (h - ph) / sh + 1;
    let out_w = (w - pw) / sw + 1;

    let mut result = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let start_h = i * sh;
            let start_w = j * sw;

            let mut max_val = T::neg_infinity();

            for dh in 0..ph {
                for dw in 0..pw {
                    let h_idx = start_h + dh;
                    let w_idx = start_w + dw;

                    if h_idx < h && w_idx < w {
                        let val = x[[h_idx, w_idx]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
            }

            result[[i, j]] = max_val;
        }
    }

    Ok(result)
}

/// Average Pooling 2D
///
/// Applies average pooling over 2D spatial data.
pub fn avg_pool2d<T>(
    x: &ArrayView2<T>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let (h, w) = (x.nrows(), x.ncols());
    let (ph, pw) = pool_size;
    let (sh, sw) = stride;

    if ph == 0 || pw == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Pool size must be positive".to_string(),
        ));
    }

    if sh == 0 || sw == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Stride must be positive".to_string(),
        ));
    }

    let out_h = (h - ph) / sh + 1;
    let out_w = (w - pw) / sw + 1;

    let mut result = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let start_h = i * sh;
            let start_w = j * sw;

            let mut sum = T::zero();
            let mut count = 0;

            for dh in 0..ph {
                for dw in 0..pw {
                    let h_idx = start_h + dh;
                    let w_idx = start_w + dw;

                    if h_idx < h && w_idx < w {
                        sum = sum + x[[h_idx, w_idx]];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let count_t = T::from(count).ok_or_else(|| {
                    NumRs2Error::ConversionError("Failed to convert count".to_string())
                })?;
                result[[i, j]] = sum / count_t;
            }
        }
    }

    Ok(result)
}

/// Adaptive Average Pooling 2D
///
/// Pools input to a fixed output size regardless of input size.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `output_size` - Desired output size (height, width)
pub fn adaptive_avg_pool2d<T>(x: &ArrayView2<T>, output_size: (usize, usize)) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let (in_h, in_w) = (x.nrows(), x.ncols());
    let (out_h, out_w) = output_size;

    if out_h == 0 || out_w == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Output size must be positive".to_string(),
        ));
    }

    let mut result = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            // Calculate input window for this output position
            let start_h = (i * in_h) / out_h;
            let end_h = ((i + 1) * in_h) / out_h;
            let start_w = (j * in_w) / out_w;
            let end_w = ((j + 1) * in_w) / out_w;

            let mut sum = T::zero();
            let mut count = 0;

            for h_idx in start_h..end_h {
                for w_idx in start_w..end_w {
                    if h_idx < in_h && w_idx < in_w {
                        sum = sum + x[[h_idx, w_idx]];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let count_t = T::from(count).ok_or_else(|| {
                    NumRs2Error::ConversionError("Failed to convert count".to_string())
                })?;
                result[[i, j]] = sum / count_t;
            }
        }
    }

    Ok(result)
}

/// Global Average Pooling
///
/// Reduces spatial dimensions to single value per channel by averaging.
pub fn global_avg_pool<T>(x: &ArrayView2<T>) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    let sum = x.sum();
    let count = T::from(x.len())
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert size".to_string()))?;

    Ok(sum / count)
}

/// Global Max Pooling
///
/// Reduces spatial dimensions to single value per channel by taking maximum.
pub fn global_max_pool<T>(x: &ArrayView2<T>) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    let max_val = x.fold(T::neg_infinity(), |acc, &v| if v > acc { v } else { acc });

    if !max_val.is_finite() {
        return Err(NumRs2Error::InvalidOperation(
            "No valid maximum found".to_string(),
        ));
    }

    Ok(max_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_max_pool2d() {
        let x = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let result = max_pool2d(&x.view(), (2, 2), (2, 2)).expect("test: valid pool params");

        assert_eq!(result.dim(), (2, 2));

        // Check that max values are extracted correctly
        assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-6); // max of 0,1,4,5
        assert_abs_diff_eq!(result[[0, 1]], 7.0, epsilon = 1e-6); // max of 2,3,6,7
        assert_abs_diff_eq!(result[[1, 0]], 13.0, epsilon = 1e-6); // max of 8,9,12,13
        assert_abs_diff_eq!(result[[1, 1]], 15.0, epsilon = 1e-6); // max of 10,11,14,15
    }

    #[test]
    fn test_avg_pool2d() {
        let x = Array2::from_shape_fn((4, 4), |(_, _)| 1.0);
        let result = avg_pool2d(&x.view(), (2, 2), (2, 2)).expect("test: valid pool params");

        assert_eq!(result.dim(), (2, 2));

        // Average of all ones should be 1
        for &val in result.iter() {
            assert_abs_diff_eq!(val, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_global_avg_pool() {
        let x = Array2::from_shape_fn((3, 3), |(_, _)| 2.0);
        let result = global_avg_pool(&x.view()).expect("test: valid global avg pool params");

        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }
}

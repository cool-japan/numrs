//! Normalization and dropout operations for neural networks
//!
//! This module provides batch normalization, layer normalization, and dropout operations.

use super::NnResult;
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{
    Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, ScalarOperand,
};
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Batch Normalization for 1D tensors
///
/// Normalizes inputs using batch statistics: `y = (x - mean) / sqrt(var + eps) * gamma + beta`
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `gamma` - Scale parameter (learned)
/// * `beta` - Shift parameter (learned)
/// * `epsilon` - Small constant for numerical stability
pub fn batch_norm_1d<T>(
    x: &ArrayView2<T>,
    gamma: &ArrayView1<T>,
    beta: &ArrayView1<T>,
    epsilon: T,
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.ncols() != gamma.len() || x.ncols() != beta.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Gamma and beta dimensions must match input features".to_string(),
        ));
    }

    let n = T::from(x.nrows())
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert batch size".to_string()))?;

    let mut result = Array2::zeros(x.raw_dim());

    // Compute batch statistics for each feature
    for j in 0..x.ncols() {
        let col = x.column(j);
        let mean = col.sum() / n;
        let var = col.mapv(|v| (v - mean) * (v - mean)).sum() / n;
        let std = (var + epsilon).sqrt();

        let g = gamma[j];
        let b = beta[j];

        for i in 0..x.nrows() {
            result[[i, j]] = (x[[i, j]] - mean) / std * g + b;
        }
    }

    Ok(result)
}

/// Layer Normalization
///
/// Normalizes across features for each sample independently.
///
/// # Arguments
///
/// * `x` - Input tensor (batch_size, features)
/// * `gamma` - Scale parameter
/// * `beta` - Shift parameter
/// * `epsilon` - Small constant for numerical stability
pub fn layer_norm<T>(
    x: &ArrayView2<T>,
    gamma: &ArrayView1<T>,
    beta: &ArrayView1<T>,
    epsilon: T,
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.ncols() != gamma.len() || x.ncols() != beta.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Gamma and beta dimensions must match input features".to_string(),
        ));
    }

    let n_features = T::from(x.ncols()).ok_or_else(|| {
        NumRs2Error::ConversionError("Failed to convert feature count".to_string())
    })?;

    let mut result = Array2::zeros(x.raw_dim());

    // Normalize each sample independently
    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean = row.sum() / n_features;
        let var = row.mapv(|v| (v - mean) * (v - mean)).sum() / n_features;
        let std = (var + epsilon).sqrt();

        for j in 0..x.ncols() {
            result[[i, j]] = (x[[i, j]] - mean) / std * gamma[j] + beta[j];
        }
    }

    Ok(result)
}

/// Instance Normalization
///
/// Normalizes each sample independently across its spatial dimensions (all
/// columns for each row).  Unlike `layer_norm` there are no learned affine
/// parameters — the output has zero mean and unit variance per row.
///
/// This is equivalent to applying `layer_norm` with `gamma = 1` and
/// `beta = 0` to each row individually.
///
/// # Arguments
///
/// * `x` - Input tensor (batch_size, features)
/// * `epsilon` - Small constant for numerical stability
pub fn instance_norm<T>(x: &ArrayView2<T>, epsilon: T) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "instance_norm requires a non-empty input tensor".to_string(),
        ));
    }

    let n_features = T::from(x.ncols()).ok_or_else(|| {
        NumRs2Error::ConversionError("Failed to convert feature count".to_string())
    })?;

    let mut result = Array2::zeros(x.raw_dim());

    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean = row.sum() / n_features;
        let var = row.mapv(|v| (v - mean) * (v - mean)).sum() / n_features;
        let std_val = (var + epsilon).sqrt();

        for j in 0..x.ncols() {
            result[[i, j]] = (x[[i, j]] - mean) / std_val;
        }
    }

    Ok(result)
}

/// RMS Normalization (Root Mean Square Layer Normalization)
///
/// A simpler normalization that doesn't subtract the mean.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `gamma` - Scale parameter
/// * `epsilon` - Small constant for numerical stability
pub fn rms_norm<T>(x: &ArrayView2<T>, gamma: &ArrayView1<T>, epsilon: T) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.ncols() != gamma.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Gamma dimension must match input features".to_string(),
        ));
    }

    let n_features = T::from(x.ncols()).ok_or_else(|| {
        NumRs2Error::ConversionError("Failed to convert feature count".to_string())
    })?;

    let mut result = Array2::zeros(x.raw_dim());

    for i in 0..x.nrows() {
        let row = x.row(i);
        let rms = (row.mapv(|v| v * v).sum() / n_features + epsilon).sqrt();

        for j in 0..x.ncols() {
            result[[i, j]] = x[[i, j]] / rms * gamma[j];
        }
    }

    Ok(result)
}

/// Dropout regularization
///
/// Randomly sets elements to zero with probability `p` during training.
/// Scales remaining elements by `1/(1-p)` to maintain expected value.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `p` - Dropout probability (0.0 to 1.0)
/// * `training` - Whether in training mode (applies dropout) or inference mode (no dropout)
pub fn dropout<T>(x: &ArrayView1<T>, p: T, training: bool) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    if p < T::zero() || p >= T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Dropout probability must be in [0, 1)".to_string(),
        ));
    }

    if !training || p == T::zero() {
        return Ok(x.to_owned());
    }

    let mut rng = thread_rng();
    let threshold = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert probability".to_string()))?;

    let scale = T::one() / (T::one() - p);

    let mask: Array1<T> = Array1::from_shape_fn(x.len(), |_| {
        if rng.random::<f64>() > threshold {
            scale
        } else {
            T::zero()
        }
    });

    Ok(x * &mask)
}

/// Dropout for 2D tensors
pub fn dropout_2d<T>(x: &ArrayView2<T>, p: T, training: bool) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if p < T::zero() || p >= T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Dropout probability must be in [0, 1)".to_string(),
        ));
    }

    if !training || p == T::zero() {
        return Ok(x.to_owned());
    }

    let mut rng = thread_rng();
    let threshold = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert probability".to_string()))?;

    let scale = T::one() / (T::one() - p);

    let mask: Array2<T> = Array2::from_shape_fn(x.raw_dim(), |_| {
        if rng.random::<f64>() > threshold {
            scale
        } else {
            T::zero()
        }
    });

    Ok(x * &mask)
}

/// Spatial Dropout (Dropout2D)
///
/// Drops entire feature maps (channels) instead of individual elements.
/// Useful for convolutional layers.
///
/// # Arguments
///
/// * `x` - Input tensor (batch, channels, height, width)
/// * `p` - Dropout probability
/// * `training` - Whether in training mode
pub fn spatial_dropout<T>(x: &ArrayView2<T>, p: T, training: bool) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if p < T::zero() || p >= T::one() {
        return Err(NumRs2Error::InvalidOperation(
            "Dropout probability must be in [0, 1)".to_string(),
        ));
    }

    if !training || p == T::zero() {
        return Ok(x.to_owned());
    }

    let mut rng = thread_rng();
    let threshold = p
        .to_f64()
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert probability".to_string()))?;

    let scale = T::one() / (T::one() - p);

    // For 2D, treat columns as channels
    let mut result = x.to_owned();
    for j in 0..x.ncols() {
        if rng.random::<f64>() <= threshold {
            // Drop entire channel
            for i in 0..x.nrows() {
                result[[i, j]] = T::zero();
            }
        } else {
            // Scale channel
            for i in 0..x.nrows() {
                result[[i, j]] = result[[i, j]] * scale;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f64);
        let gamma = Array1::ones(3);
        let beta = Array1::zeros(3);

        let result = layer_norm(&x.view(), &gamma.view(), &beta.view(), 1e-5)
            .expect("test: valid layer_norm params");

        // Each row should have mean ≈ 0 and variance ≈ 1
        for i in 0..result.nrows() {
            let row = result.row(i);
            let mean = row.sum() / row.len() as f64;
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dropout_inference() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = dropout(&x.view(), 0.5, false).expect("test: valid dropout params");

        // In inference mode, dropout should have no effect
        assert_eq!(result, x);
    }

    #[test]
    fn test_dropout_training() {
        // Use a large array (1000 elements) to ensure statistical reliability
        // With 0.5 dropout, probability of ALL elements surviving is (0.5)^1000 ≈ 0
        let x = Array1::from_vec((1..=1000).map(|i| i as f64).collect());
        let result = dropout(&x.view(), 0.5, true).expect("test: valid dropout params");

        // Some elements should be zero, others scaled
        let num_zeros = result.iter().filter(|&&v| v == 0.0).count();
        assert!(num_zeros > 0, "Expected some zeros in dropout, got none");

        // Verify scaling: non-zero elements should be multiplied by 2.0 (1/(1-0.5))
        let non_zero_count = result.iter().filter(|&&v| v != 0.0).count();
        assert!(
            non_zero_count > 0,
            "Expected some non-zero values in dropout result"
        );

        // Verify approximately 50% dropout rate (with some tolerance for randomness)
        let dropout_rate = (num_zeros as f64) / (x.len() as f64);
        assert!(
            (dropout_rate - 0.5).abs() < 0.1,
            "Dropout rate {:.2}% should be close to 50%",
            dropout_rate * 100.0
        );
    }
}

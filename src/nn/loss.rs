//! Loss functions for neural networks
//!
//! This module provides various loss functions optimized with SIMD operations.
//! All functions follow the SCIRS2 integration policy.
//!
//! # Mathematical Formulas
//!
//! - **MSE (Mean Squared Error)**: `L = (1/n) Σ (y_true - y_pred)²`
//! - **MAE (Mean Absolute Error)**: `L = (1/n) Σ |y_true - y_pred|`
//! - **Cross Entropy**: `L = -(1/n) Σ y_true * log(y_pred)`
//! - **Binary Cross Entropy**: `L = -(1/n) Σ [y*log(p) + (1-y)*log(1-p)]`
//! - **Huber Loss**: Smooth combination of MSE and MAE
//! - **Focal Loss**: `L = -α(1-p_t)^γ log(p_t)` for handling class imbalance

use super::{NnResult, ReductionMode};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{ArrayView1, ArrayView2, Axis, Zip};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Mean Squared Error (MSE) loss
///
/// Computes `L = (1/n) Σ (y_true - y_pred)²`
///
/// # Arguments
///
/// * `y_true` - Ground truth values
/// * `y_pred` - Predicted values
/// * `reduction` - How to reduce the loss (None, Mean, Sum)
///
/// # Returns
///
/// Scalar loss value or array of losses depending on reduction mode
pub fn mse_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let diff = y_pred - y_true;
    let squared = &diff * &diff;

    match reduction {
        ReductionMode::None => {
            // Return first element for scalar API compatibility
            Ok(squared[0])
        }
        ReductionMode::Mean => {
            let sum = squared.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(squared.sum()),
    }
}

/// MSE loss for 2D arrays (batch processing)
pub fn mse_loss_2d<T>(
    y_true: &ArrayView2<T>,
    y_pred: &ArrayView2<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.shape() != y_pred.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true shape {:?}, y_pred shape {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let diff = y_pred - y_true;
    let squared = &diff * &diff;

    match reduction {
        ReductionMode::None => Ok(squared[[0, 0]]),
        ReductionMode::Mean => {
            let sum = squared.sum();
            let n = T::from(squared.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(squared.sum()),
    }
}

/// Mean Absolute Error (MAE) loss
///
/// Computes `L = (1/n) Σ |y_true - y_pred|`
pub fn mae_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let diff = y_pred - y_true;
    let abs_diff = diff.mapv(|x| x.abs());

    match reduction {
        ReductionMode::None => Ok(abs_diff[0]),
        ReductionMode::Mean => {
            let sum = abs_diff.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(abs_diff.sum()),
    }
}

/// Huber loss (smooth L1 loss)
///
/// Combines MSE and MAE for robustness to outliers.
/// - For |x| ≤ δ: L = 0.5 * x²
/// - For |x| > δ: L = δ * (|x| - 0.5 * δ)
///
/// # Arguments
///
/// * `y_true` - Ground truth values
/// * `y_pred` - Predicted values
/// * `delta` - Threshold parameter (typically 1.0)
/// * `reduction` - How to reduce the loss
pub fn huber_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    delta: T,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if delta <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Huber loss delta must be positive".to_string(),
        ));
    }

    let half = T::from(0.5)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert 0.5".to_string()))?;

    let diff = y_pred - y_true;
    let abs_diff = diff.mapv(|x| x.abs());

    let loss = Zip::from(&abs_diff).and(&diff).map_collect(|&a, &d| {
        if a <= delta {
            half * d * d
        } else {
            delta * (a - half * delta)
        }
    });

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

/// Smooth L1 loss (same as Huber loss with delta=1.0)
pub fn smooth_l1_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    huber_loss(y_true, y_pred, T::one(), reduction)
}

/// Binary Cross-Entropy loss
///
/// Computes `L = -(1/n) Σ [y*log(p) + (1-y)*log(1-p)]`
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0 or 1)
/// * `y_pred` - Predicted probabilities (0 to 1)
/// * `reduction` - How to reduce the loss
pub fn binary_cross_entropy<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let eps = T::from(1e-7)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    let one = T::one();

    // Clip predictions to avoid log(0)
    let y_pred_clipped = y_pred.mapv(|p| p.max(eps).min(one - eps));

    let loss = Zip::from(y_true)
        .and(&y_pred_clipped)
        .map_collect(|&y, &p| -(y * p.ln() + (one - y) * (one - p).ln()));

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

/// Binary Cross-Entropy with logits
///
/// More numerically stable version that takes logits (unnormalized predictions)
/// instead of probabilities.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0 or 1)
/// * `logits` - Raw model outputs (before sigmoid)
/// * `reduction` - How to reduce the loss
pub fn bce_with_logits<T>(
    y_true: &ArrayView1<T>,
    logits: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != logits.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, logits has {}",
            y_true.len(),
            logits.len()
        )));
    }

    let one = T::one();

    // Use log-sum-exp trick for numerical stability
    // BCE = max(x, 0) - x * y + log(1 + exp(-|x|))
    let loss = Zip::from(y_true).and(logits).map_collect(|&y, &x| {
        let max_val = x.max(T::zero());
        max_val - x * y + (one + (-x.abs()).exp()).ln()
    });

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

/// Categorical Cross-Entropy loss
///
/// Computes `L = -(1/n) Σ Σ_c y_true_c * log(y_pred_c)`
///
/// # Arguments
///
/// * `y_true` - Ground truth probabilities (one-hot or soft labels)
/// * `y_pred` - Predicted probabilities (from softmax)
/// * `reduction` - How to reduce the loss
pub fn categorical_cross_entropy<T>(
    y_true: &ArrayView2<T>,
    y_pred: &ArrayView2<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.shape() != y_pred.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true shape {:?}, y_pred shape {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let eps = T::from(1e-7)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    // Clip predictions to avoid log(0)
    let y_pred_clipped = y_pred.mapv(|p| p.max(eps));

    let loss_per_element = y_true * &y_pred_clipped.mapv(|p| p.ln());
    let loss = -loss_per_element.sum_axis(Axis(1));

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.nrows()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert batch size".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

/// Sparse Categorical Cross-Entropy loss
///
/// Optimized version for integer class labels instead of one-hot vectors.
///
/// # Arguments
///
/// * `y_true` - Ground truth class indices (0 to num_classes-1)
/// * `y_pred` - Predicted probabilities for each class
/// * `reduction` - How to reduce the loss
pub fn sparse_categorical_cross_entropy<T>(
    y_true: &[usize],
    y_pred: &ArrayView2<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.nrows() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Batch size mismatch: y_true has {} samples, y_pred has {}",
            y_true.len(),
            y_pred.nrows()
        )));
    }

    let eps = T::from(1e-7)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    let num_classes = y_pred.ncols();

    let mut losses = Vec::with_capacity(y_true.len());

    for (i, &class_idx) in y_true.iter().enumerate() {
        if class_idx >= num_classes {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Class index {} out of bounds for {} classes",
                class_idx, num_classes
            )));
        }

        let pred_prob = y_pred[[i, class_idx]];
        let clipped_prob = pred_prob.max(eps);
        losses.push(-clipped_prob.ln());
    }

    match reduction {
        ReductionMode::None => Ok(losses[0]),
        ReductionMode::Mean => {
            let sum = losses.iter().fold(T::zero(), |acc, &x| acc + x);
            let n = T::from(losses.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(losses.iter().fold(T::zero(), |acc, &x| acc + x)),
    }
}

/// Negative Log-Likelihood loss
///
/// Used with log-softmax outputs. Equivalent to cross-entropy with log probabilities.
///
/// # Arguments
///
/// * `y_true` - Ground truth class indices
/// * `log_probs` - Log probabilities from log-softmax
/// * `reduction` - How to reduce the loss
pub fn nll_loss<T>(
    y_true: &[usize],
    log_probs: &ArrayView2<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != log_probs.nrows() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Batch size mismatch: y_true has {} samples, log_probs has {}",
            y_true.len(),
            log_probs.nrows()
        )));
    }

    let num_classes = log_probs.ncols();

    let mut losses = Vec::with_capacity(y_true.len());

    for (i, &class_idx) in y_true.iter().enumerate() {
        if class_idx >= num_classes {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Class index {} out of bounds for {} classes",
                class_idx, num_classes
            )));
        }

        losses.push(-log_probs[[i, class_idx]]);
    }

    match reduction {
        ReductionMode::None => Ok(losses[0]),
        ReductionMode::Mean => {
            let sum = losses.iter().fold(T::zero(), |acc, &x| acc + x);
            let n = T::from(losses.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(losses.iter().fold(T::zero(), |acc, &x| acc + x)),
    }
}

/// Kullback-Leibler Divergence loss
///
/// Measures how one probability distribution differs from another.
/// `KL(P||Q) = Σ P(x) * log(P(x) / Q(x))`
///
/// # Arguments
///
/// * `p` - True probability distribution
/// * `q` - Approximate probability distribution
/// * `reduction` - How to reduce the loss
pub fn kl_div_loss<T>(p: &ArrayView1<T>, q: &ArrayView1<T>, reduction: ReductionMode) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if p.len() != q.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: p has {} elements, q has {}",
            p.len(),
            q.len()
        )));
    }

    let eps = T::from(1e-10)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    let kl = Zip::from(p).and(q).map_collect(|&p_val, &q_val| {
        if p_val > eps {
            p_val * ((p_val + eps) / (q_val + eps)).ln()
        } else {
            T::zero()
        }
    });

    match reduction {
        ReductionMode::None => Ok(kl[0]),
        ReductionMode::Mean => {
            let sum = kl.sum();
            let n = T::from(p.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(kl.sum()),
    }
}

/// Hinge loss for SVM classification
///
/// `L = max(0, 1 - y_true * y_pred)`
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (-1 or +1)
/// * `y_pred` - Predicted scores (not probabilities)
/// * `reduction` - How to reduce the loss
pub fn hinge_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let one = T::one();
    let zero = T::zero();

    let loss = Zip::from(y_true)
        .and(y_pred)
        .map_collect(|&y, &pred| (one - y * pred).max(zero));

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

/// Cosine Embedding loss
///
/// Measures the cosine similarity between two embeddings.
/// - If y = 1: L = 1 - cos(x1, x2)
/// - If y = -1: L = max(0, cos(x1, x2) - margin)
///
/// # Arguments
///
/// * `x1` - First embedding vector
/// * `x2` - Second embedding vector
/// * `y` - Label (1 for similar, -1 for dissimilar)
/// * `margin` - Margin for dissimilar pairs
pub fn cosine_embedding_loss<T>(
    x1: &ArrayView1<T>,
    x2: &ArrayView1<T>,
    y: T,
    margin: T,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if x1.len() != x2.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Embedding dimension mismatch: x1 has {}, x2 has {}",
            x1.len(),
            x2.len()
        )));
    }

    let eps = T::from(1e-8)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    // Compute cosine similarity
    let dot_product = Zip::from(x1)
        .and(x2)
        .fold(T::zero(), |acc, &a, &b| acc + a * b);

    let norm1 = x1.mapv(|v| v * v).sum().sqrt();
    let norm2 = x2.mapv(|v| v * v).sum().sqrt();

    let cos_sim = dot_product / ((norm1 * norm2) + eps);

    let one = T::one();
    let zero = T::zero();

    if y == one {
        Ok(one - cos_sim)
    } else {
        Ok((cos_sim - margin).max(zero))
    }
}

/// Focal Loss for handling class imbalance
///
/// `FL(p_t) = -α(1 - p_t)^γ log(p_t)`
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0 or 1)
/// * `y_pred` - Predicted probabilities (0 to 1)
/// * `alpha` - Weighting factor for class imbalance (typically 0.25)
/// * `gamma` - Focusing parameter (typically 2.0)
/// * `reduction` - How to reduce the loss
pub fn focal_loss<T>(
    y_true: &ArrayView1<T>,
    y_pred: &ArrayView1<T>,
    alpha: T,
    gamma: T,
    reduction: ReductionMode,
) -> NnResult<T>
where
    T: Float + SimdUnifiedOps,
{
    if y_true.len() != y_pred.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let eps = T::from(1e-7)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

    let one = T::one();

    // Clip predictions
    let y_pred_clipped = y_pred.mapv(|p| p.max(eps).min(one - eps));

    let loss = Zip::from(y_true)
        .and(&y_pred_clipped)
        .map_collect(|&y, &p| {
            let p_t = if y == one { p } else { one - p };
            let alpha_t = if y == one { alpha } else { one - alpha };

            -alpha_t * (one - p_t).powf(gamma) * p_t.ln()
        });

    match reduction {
        ReductionMode::None => Ok(loss[0]),
        ReductionMode::Mean => {
            let sum = loss.sum();
            let n = T::from(y_true.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert length".to_string())
            })?;
            Ok(sum / n)
        }
        ReductionMode::Sum => Ok(loss.sum()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mse_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 3.1];

        let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
            .expect("test: valid mse loss params");

        // MSE = mean((0.1)^2 + (0.1)^2 + (0.1)^2) = 0.01
        assert_abs_diff_eq!(loss, 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 3.1];

        let loss = mae_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
            .expect("test: valid mae loss params");

        // MAE = mean(|0.1| + |0.1| + |0.1|) = 0.1
        assert_abs_diff_eq!(loss, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_loss() {
        let y_true = array![0.0, 0.0, 0.0];
        let y_pred = array![0.5, 1.0, 2.0];

        let loss = huber_loss(&y_true.view(), &y_pred.view(), 1.0, ReductionMode::Mean)
            .expect("test: valid huber loss params");

        // For delta=1: values <= 1 use squared, > 1 use linear
        // |0.5| <= 1: 0.5 * 0.5^2 = 0.125
        // |1.0| <= 1: 0.5 * 1.0^2 = 0.5
        // |2.0| > 1: 1.0 * (2.0 - 0.5) = 1.5
        // Mean = (0.125 + 0.5 + 1.5) / 3 = 0.708333...
        assert_abs_diff_eq!(loss, 0.708333, epsilon = 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let y_true = array![1.0, 0.0, 1.0];
        let y_pred = array![0.9, 0.1, 0.8];

        let loss = binary_cross_entropy(&y_true.view(), &y_pred.view(), ReductionMode::Mean)
            .expect("test: valid bce params");

        // Should be small since predictions are close to truth
        assert!(loss < 0.2);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_mse_shape_mismatch() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];

        let result = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean);
        assert!(result.is_err());
    }
}

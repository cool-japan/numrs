//! Activation functions for neural networks
//!
//! This module provides common activation functions optimized with SIMD operations.
//! All functions follow the SCIRS2 integration policy and use scirs2_core abstractions.
//!
//! # Mathematical Formulas
//!
//! - **ReLU**: `f(x) = max(0, x)`
//! - **GELU**: `f(x) = x * Φ(x)` where Φ is the cumulative distribution function of the standard normal
//! - **Swish/SiLU**: `f(x) = x * sigmoid(x)`
//! - **Mish**: `f(x) = x * tanh(softplus(x))`
//! - **ELU**: `f(x) = x if x > 0 else α(exp(x) - 1)`
//! - **SELU**: `f(x) = λ * (x if x > 0 else α(exp(x) - 1))`
//! - **Leaky ReLU**: `f(x) = x if x > 0 else α * x`
//! - **Softmax**: `f(x)_i = exp(x_i) / Σ exp(x_j)`

use super::NnResult;
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{
    s, Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, ScalarOperand, Zip,
};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// ReLU (Rectified Linear Unit) activation function
///
/// Computes `f(x) = max(0, x)` element-wise.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with ReLU applied element-wise
///
/// # Example
///
/// ```rust,ignore
/// use numrs2::nn::activation::relu;
/// use scirs2_core::ndarray::array;
///
/// let x = array![-1.0, 0.0, 1.0, 2.0];
/// let y = relu(&x.view()).expect("valid relu input");
/// // y = [0.0, 0.0, 1.0, 2.0]
/// ```
pub fn relu<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let zero = T::zero();
    Ok(x.mapv(|v| if v > zero { v } else { zero }))
}

/// ReLU activation for 2D arrays (in-place along last axis)
pub fn relu_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let zero = T::zero();
    Ok(x.mapv(|v| if v > zero { v } else { zero }))
}

/// In-place ReLU activation
pub fn relu_inplace<T>(x: &mut Array1<T>)
where
    T: Float + SimdUnifiedOps,
{
    let zero = T::zero();
    x.mapv_inplace(|v| if v > zero { v } else { zero });
}

/// Leaky ReLU activation function
///
/// Computes `f(x) = x if x > 0 else α * x` element-wise.
///
/// # Arguments
///
/// * `x` - Input array
/// * `alpha` - Negative slope coefficient (typically 0.01)
///
/// # Returns
///
/// Array with Leaky ReLU applied
pub fn leaky_relu<T>(x: &ArrayView1<T>, alpha: T) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    if alpha < T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Leaky ReLU alpha must be non-negative".to_string(),
        ));
    }

    let zero = T::zero();
    Ok(x.mapv(|v| if v > zero { v } else { alpha * v }))
}

/// Leaky ReLU for 2D arrays
pub fn leaky_relu_2d<T>(x: &ArrayView2<T>, alpha: T) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if alpha < T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "Leaky ReLU alpha must be non-negative".to_string(),
        ));
    }

    let zero = T::zero();
    Ok(x.mapv(|v| if v > zero { v } else { alpha * v }))
}

/// ELU (Exponential Linear Unit) activation function
///
/// Computes `f(x) = x if x > 0 else α(exp(x) - 1)` element-wise.
///
/// # Arguments
///
/// * `x` - Input array
/// * `alpha` - Scale for negative values (typically 1.0)
pub fn elu<T>(x: &ArrayView1<T>, alpha: T) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    if alpha <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "ELU alpha must be positive".to_string(),
        ));
    }

    let zero = T::zero();
    let one = T::one();
    Ok(x.mapv(|v| if v > zero { v } else { alpha * (v.exp() - one) }))
}

/// ELU for 2D arrays
pub fn elu_2d<T>(x: &ArrayView2<T>, alpha: T) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if alpha <= T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "ELU alpha must be positive".to_string(),
        ));
    }

    let zero = T::zero();
    let one = T::one();
    Ok(x.mapv(|v| if v > zero { v } else { alpha * (v.exp() - one) }))
}

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Computes `f(x) = λ * (x if x > 0 else α(exp(x) - 1))` with specific constants
/// that ensure self-normalizing properties.
///
/// # Arguments
///
/// * `x` - Input array
pub fn selu<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    // SELU constants for self-normalizing properties
    let lambda = T::from(1.0507009873554804934193349852946).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert SELU lambda constant".to_string())
    })?;
    let alpha = T::from(1.6732632423543772848170429916717).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert SELU alpha constant".to_string())
    })?;

    let zero = T::zero();
    let one = T::one();
    Ok(x.mapv(|v| {
        if v > zero {
            lambda * v
        } else {
            lambda * alpha * (v.exp() - one)
        }
    }))
}

/// SELU for 2D arrays
pub fn selu_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let lambda = T::from(1.0507009873554804934193349852946).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert SELU lambda constant".to_string())
    })?;
    let alpha = T::from(1.6732632423543772848170429916717).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert SELU alpha constant".to_string())
    })?;

    let zero = T::zero();
    let one = T::one();
    Ok(x.mapv(|v| {
        if v > zero {
            lambda * v
        } else {
            lambda * alpha * (v.exp() - one)
        }
    }))
}

/// Sigmoid activation function
///
/// Computes `f(x) = 1 / (1 + exp(-x))` element-wise.
///
/// # Arguments
///
/// * `x` - Input array
pub fn sigmoid<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| one / (one + (-v).exp())))
}

/// Sigmoid for 2D arrays
pub fn sigmoid_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| one / (one + (-v).exp())))
}

/// Hyperbolic tangent activation function
///
/// Computes `f(x) = tanh(x)` element-wise.
pub fn tanh<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    Ok(x.mapv(|v| v.tanh()))
}

/// Tanh for 2D arrays
pub fn tanh_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    Ok(x.mapv(|v| v.tanh()))
}

/// Swish/SiLU activation function
///
/// Computes `f(x) = x * sigmoid(x)` element-wise.
/// Also known as Sigmoid Linear Unit (SiLU).
///
/// # Arguments
///
/// * `x` - Input array
pub fn swish<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| v / (one + (-v).exp())))
}

/// Swish for 2D arrays
pub fn swish_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| v / (one + (-v).exp())))
}

/// SiLU (Sigmoid Linear Unit) - alias for Swish
pub fn silu<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    swish(x)
}

/// SiLU for 2D arrays
pub fn silu_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    swish_2d(x)
}

/// Mish activation function
///
/// Computes `f(x) = x * tanh(softplus(x))` where `softplus(x) = ln(1 + exp(x))`.
///
/// # Arguments
///
/// * `x` - Input array
pub fn mish<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| {
        let softplus = (one + v.exp()).ln();
        v * softplus.tanh()
    }))
}

/// Mish for 2D arrays
pub fn mish_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    Ok(x.mapv(|v| {
        let softplus = (one + v.exp()).ln();
        v * softplus.tanh()
    }))
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Computes an approximation of `f(x) = x * Φ(x)` where Φ is the cumulative
/// distribution function of the standard normal distribution.
///
/// Uses the approximation: `f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
///
/// # Arguments
///
/// * `x` - Input array
pub fn gelu<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let half = T::from(0.5)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert 0.5".to_string()))?;
    let one = T::one();
    let coeff = T::from(0.7978845608028654).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert GELU coefficient".to_string())
    })?; // sqrt(2/pi)
    let cubic_coeff = T::from(0.044715).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert GELU cubic coefficient".to_string())
    })?;

    Ok(x.mapv(|v| {
        let three = T::from(3.0).unwrap_or(one + one + one);
        let cubic = v.powi(3);
        let inner = coeff * (v + cubic_coeff * cubic);
        half * v * (one + inner.tanh())
    }))
}

/// GELU for 2D arrays
pub fn gelu_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let half = T::from(0.5)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert 0.5".to_string()))?;
    let one = T::one();
    let coeff = T::from(0.7978845608028654).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert GELU coefficient".to_string())
    })?;
    let cubic_coeff = T::from(0.044715).ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert GELU cubic coefficient".to_string())
    })?;

    Ok(x.mapv(|v| {
        let three = T::from(3.0).unwrap_or(one + one + one);
        let cubic = v.powi(3);
        let inner = coeff * (v + cubic_coeff * cubic);
        half * v * (one + inner.tanh())
    }))
}

/// Softmax activation function
///
/// Computes `f(x)_i = exp(x_i) / Σ exp(x_j)` with numerical stability.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Probability distribution over input elements
pub fn softmax<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.is_empty() {
        return Err(NumRs2Error::DimensionMismatch(
            "Softmax requires non-empty input".to_string(),
        ));
    }

    // Numerical stability: subtract max value
    let max_val = x.fold(T::neg_infinity(), |acc, &v| if v > acc { v } else { acc });

    if !max_val.is_finite() {
        return Err(NumRs2Error::InvalidOperation(
            "Softmax input contains non-finite values".to_string(),
        ));
    }

    let shifted = x.mapv(|v| (v - max_val).exp());
    let sum = shifted.sum();

    if sum == T::zero() || !sum.is_finite() {
        return Err(NumRs2Error::InvalidOperation(
            "Softmax normalization failed (sum is zero or non-finite)".to_string(),
        ));
    }

    Ok(shifted / sum)
}

/// Softmax for 2D arrays (along specified axis)
///
/// # Arguments
///
/// * `x` - Input array
/// * `axis` - Axis along which to apply softmax (typically last axis)
pub fn softmax_2d<T>(x: &ArrayView2<T>, axis: usize) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.is_empty() {
        return Err(NumRs2Error::DimensionMismatch(
            "Softmax requires non-empty input".to_string(),
        ));
    }

    if axis >= 2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Invalid axis {} for 2D array",
            axis
        )));
    }

    let mut result = Array2::zeros(x.raw_dim());

    if axis == 1 {
        // Apply softmax along rows
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let softmax_row = softmax(&row)?;
            result.row_mut(i).assign(&softmax_row);
        }
    } else {
        // Apply softmax along columns
        for (j, col) in x.axis_iter(Axis(1)).enumerate() {
            let softmax_col = softmax(&col)?;
            result.column_mut(j).assign(&softmax_col);
        }
    }

    Ok(result)
}

/// Log-softmax activation function
///
/// Computes `f(x)_i = log(exp(x_i) / Σ exp(x_j))` with numerical stability.
/// More numerically stable than computing log(softmax(x)).
///
/// # Arguments
///
/// * `x` - Input array
pub fn log_softmax<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(NumRs2Error::DimensionMismatch(
            "Log-softmax requires non-empty input".to_string(),
        ));
    }

    // Numerical stability: subtract max value
    let max_val = x.fold(T::neg_infinity(), |acc, &v| if v > acc { v } else { acc });

    if !max_val.is_finite() {
        return Err(NumRs2Error::InvalidOperation(
            "Log-softmax input contains non-finite values".to_string(),
        ));
    }

    let shifted = x.mapv(|v| v - max_val);
    let log_sum_exp = shifted.mapv(|v| v.exp()).sum().ln();

    if !log_sum_exp.is_finite() {
        return Err(NumRs2Error::InvalidOperation(
            "Log-softmax computation failed (non-finite log_sum_exp)".to_string(),
        ));
    }

    Ok(shifted.mapv(|v| v - log_sum_exp))
}

/// Log-softmax for 2D arrays (along specified axis)
pub fn log_softmax_2d<T>(x: &ArrayView2<T>, axis: usize) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.is_empty() {
        return Err(NumRs2Error::DimensionMismatch(
            "Log-softmax requires non-empty input".to_string(),
        ));
    }

    if axis >= 2 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Invalid axis {} for 2D array",
            axis
        )));
    }

    let mut result = Array2::zeros(x.raw_dim());

    if axis == 1 {
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let log_softmax_row = log_softmax(&row)?;
            result.row_mut(i).assign(&log_softmax_row);
        }
    } else {
        for (j, col) in x.axis_iter(Axis(1)).enumerate() {
            let log_softmax_col = log_softmax(&col)?;
            result.column_mut(j).assign(&log_softmax_col);
        }
    }

    Ok(result)
}

/// Softplus activation function
///
/// Computes `f(x) = ln(1 + exp(x))` element-wise with numerical stability.
///
/// # Arguments
///
/// * `x` - Input array
pub fn softplus<T>(x: &ArrayView1<T>) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    let threshold = T::from(20.0)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert threshold".to_string()))?;

    // For numerical stability, use x directly when x > threshold
    Ok(x.mapv(|v| {
        if v > threshold {
            v
        } else {
            (one + v.exp()).ln()
        }
    }))
}

/// Softplus for 2D arrays
pub fn softplus_2d<T>(x: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let one = T::one();
    let threshold = T::from(20.0)
        .ok_or_else(|| NumRs2Error::InvalidOperation("Failed to convert threshold".to_string()))?;

    Ok(x.mapv(|v| {
        if v > threshold {
            v
        } else {
            (one + v.exp()).ln()
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_relu() {
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = relu(&x.view()).expect("test: valid relu input");
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[3], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[4], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let x = array![0.0];
        let y = sigmoid(&x.view()).expect("test: valid sigmoid input");
        assert_abs_diff_eq!(y[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = array![1.0, 2.0, 3.0];
        let y = softmax(&x.view()).expect("test: valid softmax input");

        // Sum should be 1.0
        let sum: f64 = y.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Values should be positive
        assert!(y.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow exp
        let x = array![1000.0, 1001.0, 1002.0];
        let y = softmax(&x.view()).expect("test: valid softmax input");

        // Should not contain NaN or Inf
        assert!(y.iter().all(|&v| v.is_finite()));

        // Sum should still be 1.0
        let sum: f64 = y.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gelu() {
        let x = array![0.0];
        let y = gelu(&x.view()).expect("test: valid gelu input");
        // GELU(0) ≈ 0
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_swish() {
        let x = array![0.0];
        let y = swish(&x.view()).expect("test: valid swish input");
        // Swish(0) = 0
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-6);
    }
}

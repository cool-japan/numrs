//! Attention mechanisms and embedding operations for neural networks
//!
//! This module provides attention mechanisms including scaled dot-product attention
//! and multi-head attention, as well as embedding operations.

use super::activation::softmax_2d;
use super::NnResult;
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Scaled Dot-Product Attention
///
/// Core attention mechanism: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
///
/// # Arguments
///
/// * `query` - Query matrix (seq_len_q, d_k)
/// * `key` - Key matrix (seq_len_k, d_k)
/// * `value` - Value matrix (seq_len_k, d_v)
/// * `mask` - Optional attention mask (prevents attention to certain positions)
pub fn scaled_dot_product_attention<T>(
    query: &ArrayView2<T>,
    key: &ArrayView2<T>,
    value: &ArrayView2<T>,
    mask: Option<&ArrayView2<T>>,
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    let d_k = query.ncols();

    if key.ncols() != d_k {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Query and key dimension mismatch: {} vs {}",
            d_k,
            key.ncols()
        )));
    }

    if key.nrows() != value.nrows() {
        return Err(NumRs2Error::DimensionMismatch(
            "Key and value sequence length mismatch".to_string(),
        ));
    }

    // Scale factor: 1 / sqrt(d_k)
    let scale = T::from(1.0)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert scale".to_string()))?
        / T::from(d_k)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dimension".to_string()))?
            .sqrt();

    // Compute Q * K^T
    let scores = query.dot(&key.t()) * scale;

    // Apply mask if provided
    let masked_scores = if let Some(m) = mask {
        if m.shape() != scores.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Mask shape mismatch with attention scores".to_string(),
            ));
        }

        let neg_inf = T::neg_infinity();
        let zero = T::zero();

        Array2::from_shape_fn(scores.raw_dim(), |(i, j)| {
            if m[[i, j]] == zero {
                neg_inf
            } else {
                scores[[i, j]]
            }
        })
    } else {
        scores
    };

    // Apply softmax along last dimension
    let attention_weights = softmax_2d(&masked_scores.view(), 1)?;

    // Compute attention output: softmax(scores) * V
    let output = attention_weights.dot(value);

    Ok(output)
}

/// Self-Attention
///
/// Computes attention where query, key, and value come from the same source.
///
/// # Arguments
///
/// * `x` - Input matrix (seq_len, d_model)
/// * `w_q` - Query projection matrix (d_model, d_k)
/// * `w_k` - Key projection matrix (d_model, d_k)
/// * `w_v` - Value projection matrix (d_model, d_v)
pub fn self_attention<T>(
    x: &ArrayView2<T>,
    w_q: &ArrayView2<T>,
    w_k: &ArrayView2<T>,
    w_v: &ArrayView2<T>,
) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if x.ncols() != w_q.nrows() || x.ncols() != w_k.nrows() || x.ncols() != w_v.nrows() {
        return Err(NumRs2Error::DimensionMismatch(
            "Input dimension mismatch with projection matrices".to_string(),
        ));
    }

    // Project input to queries, keys, and values
    let query = x.dot(w_q);
    let key = x.dot(w_k);
    let value = x.dot(w_v);

    scaled_dot_product_attention(&query.view(), &key.view(), &value.view(), None)
}

/// Embedding lookup
///
/// Maps integer indices to dense vectors.
///
/// # Arguments
///
/// * `indices` - Array of indices to look up
/// * `embedding_matrix` - Embedding weight matrix (vocab_size, embedding_dim)
pub fn embedding<T>(indices: &[usize], embedding_matrix: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let vocab_size = embedding_matrix.nrows();
    let embedding_dim = embedding_matrix.ncols();

    let mut output = Array2::zeros((indices.len(), embedding_dim));

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= vocab_size {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index {} out of bounds for vocabulary size {}",
                idx, vocab_size
            )));
        }

        output.row_mut(i).assign(&embedding_matrix.row(idx));
    }

    Ok(output)
}

/// Positional Encoding (Sinusoidal)
///
/// Adds positional information to embeddings using sine and cosine functions.
///
/// # Arguments
///
/// * `seq_len` - Sequence length
/// * `d_model` - Model dimension (must be even)
pub fn positional_encoding<T>(seq_len: usize, d_model: usize) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if !d_model.is_multiple_of(2) {
        return Err(NumRs2Error::InvalidOperation(
            "Model dimension must be even for sinusoidal positional encoding".to_string(),
        ));
    }

    let mut pe = Array2::zeros((seq_len, d_model));

    let two = T::from(2.0)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert constant".to_string()))?;

    let ten_thousand = T::from(10000.0)
        .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert constant".to_string()))?;

    for pos in 0..seq_len {
        let pos_t = T::from(pos).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert position".to_string())
        })?;

        for i in 0..(d_model / 2) {
            let i_t = T::from(i).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert index".to_string())
            })?;

            let d_model_t = T::from(d_model).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert dimension".to_string())
            })?;

            let div_term = two * i_t / d_model_t;
            let angle = pos_t / ten_thousand.powf(div_term);

            pe[[pos, 2 * i]] = angle.sin();
            pe[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    Ok(pe)
}

/// Add positional encoding to embeddings
pub fn add_positional_encoding<T>(embeddings: &ArrayView2<T>) -> NnResult<Array2<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    let (seq_len, d_model) = embeddings.dim();

    let pe = positional_encoding(seq_len, d_model)?;

    Ok(embeddings + &pe)
}

/// Embedding Bag (aggregated embeddings)
///
/// Computes aggregated embeddings over a bag of indices.
///
/// # Arguments
///
/// * `indices` - Array of indices to aggregate
/// * `embedding_matrix` - Embedding weight matrix
/// * `mode` - Aggregation mode: "sum", "mean", or "max"
pub fn embedding_bag<T>(
    indices: &[usize],
    embedding_matrix: &ArrayView2<T>,
    mode: &str,
) -> NnResult<Array1<T>>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    if indices.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Indices cannot be empty".to_string(),
        ));
    }

    let vocab_size = embedding_matrix.nrows();
    let embedding_dim = embedding_matrix.ncols();

    let mut result = Array1::zeros(embedding_dim);

    match mode {
        "sum" => {
            for &idx in indices {
                if idx >= vocab_size {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} out of bounds",
                        idx
                    )));
                }
                result = result + embedding_matrix.row(idx);
            }
        }
        "mean" => {
            for &idx in indices {
                if idx >= vocab_size {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} out of bounds",
                        idx
                    )));
                }
                result = result + embedding_matrix.row(idx);
            }

            let count = T::from(indices.len()).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert count".to_string())
            })?;

            result = result / count;
        }
        "max" => {
            result = Array1::from_elem(embedding_dim, T::neg_infinity());

            for &idx in indices {
                if idx >= vocab_size {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} out of bounds",
                        idx
                    )));
                }

                let emb = embedding_matrix.row(idx);
                for j in 0..embedding_dim {
                    if emb[j] > result[j] {
                        result[j] = emb[j];
                    }
                }
            }
        }
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Unknown mode: {}. Must be 'sum', 'mean', or 'max'",
                mode
            )));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_embedding() {
        let embedding_matrix = Array2::from_shape_fn((10, 5), |(i, j)| (i * 10 + j) as f64);
        let indices = vec![0, 2, 5];

        let result =
            embedding(&indices, &embedding_matrix.view()).expect("test: valid embedding params");

        assert_eq!(result.dim(), (3, 5));

        // Check first embedding (index 0)
        for j in 0..5 {
            assert_abs_diff_eq!(result[[0, j]], j as f64, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding::<f64>(10, 8).expect("test: valid positional encoding params");

        assert_eq!(pe.dim(), (10, 8));

        // Values should be bounded between -1 and 1 (sine/cosine)
        for &val in pe.iter() {
            assert!((-1.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_embedding_bag_sum() {
        let embedding_matrix = Array2::from_shape_fn((5, 3), |(_, _)| 1.0);
        let indices = vec![0, 1, 2];

        let result = embedding_bag(&indices, &embedding_matrix.view(), "sum")
            .expect("test: valid embedding_bag params");

        // Sum of 3 vectors of ones should be [3, 3, 3]
        for &val in result.iter() {
            assert_abs_diff_eq!(val, 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_embedding_bag_mean() {
        let embedding_matrix = Array2::from_shape_fn((5, 3), |(_, _)| 2.0);
        let indices = vec![0, 1, 2, 3];

        let result = embedding_bag(&indices, &embedding_matrix.view(), "mean")
            .expect("test: valid embedding_bag params");

        // Mean of vectors of twos should be [2, 2, 2]
        for &val in result.iter() {
            assert_abs_diff_eq!(val, 2.0, epsilon = 1e-6);
        }
    }
}

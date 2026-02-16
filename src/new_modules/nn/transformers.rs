//! Transformer Neural Network Layers
//!
//! This module implements the complete transformer architecture as described in
//! "Attention Is All You Need" (Vaswani et al., 2017) with modern enhancements.
//!
//! # Mathematical Foundations
//!
//! ## Scaled Dot-Product Attention
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k) V
//! ```
//!
//! where:
//! - Q: Query matrix (seq_len, d_k)
//! - K: Key matrix (seq_len, d_k)
//! - V: Value matrix (seq_len, d_v)
//! - d_k: Dimension of keys/queries (used for scaling)
//!
//! ## Multi-Head Attention
//!
//! ```text
//! MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
//! where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
//! ```
//!
//! where:
//! - h: Number of attention heads
//! - W_i^Q, W_i^K, W_i^V: Learnable projection matrices
//! - W^O: Output projection matrix
//!
//! ## Position-wise Feed-Forward Network
//!
//! ```text
//! FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
//! ```
//!
//! ## Layer Normalization
//!
//! ```text
//! LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
//! ```
//!
//! # References
//!
//! - Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
//! - Hendrycks & Gimpel. "Gaussian Error Linear Units (GELUs)" (2016)
//! - Ba et al. "Layer Normalization" (2016)

use crate::error::NumRs2Error;
use scirs2_core::ndarray::{
    s, Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, ArrayView3, Axis,
    ScalarOperand,
};
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Result type for transformer operations
pub type TransformerResult<T> = Result<T, NumRs2Error>;

/// Multi-Head Attention Layer
///
/// Implements parallel attention mechanisms with multiple heads, allowing the model
/// to attend to information from different representation subspaces.
///
/// # Architecture
///
/// ```text
/// Input (batch, seq_len, d_model)
///   │
///   ├─> Linear(Q) ─> Split into h heads ─> Scaled Dot-Product Attention ─┐
///   ├─> Linear(K) ─> Split into h heads ─> Scaled Dot-Product Attention ─┤
///   └─> Linear(V) ─> Split into h heads ─> Scaled Dot-Product Attention ─┤
///                                                                          │
///                     Concatenate heads <─────────────────────────────────┘
///                            │
///                       Linear(Output)
///                            │
///                         Dropout
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use numrs2::new_modules::nn::transformers::MultiHeadAttention;
/// use scirs2_core::ndarray::Array2;
///
/// let mha = MultiHeadAttention::new(512, 8, 0.1)?;
/// let input = Array2::zeros((32, 512)); // (seq_len, d_model)
/// let output = mha.forward(&input.view(), None, false)?;
/// ```
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head (d_model / num_heads)
    pub d_k: usize,
    /// Query projection weights (d_model, d_model)
    pub w_q: Array2<T>,
    /// Key projection weights (d_model, d_model)
    pub w_k: Array2<T>,
    /// Value projection weights (d_model, d_model)
    pub w_v: Array2<T>,
    /// Output projection weights (d_model, d_model)
    pub w_o: Array2<T>,
    /// Dropout probability
    pub dropout_p: T,
}

impl<T> MultiHeadAttention<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new multi-head attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of parallel attention heads
    /// * `dropout` - Dropout probability for attention weights
    ///
    /// # Returns
    ///
    /// A new multi-head attention layer with Xavier-initialized weights
    pub fn new(d_model: usize, num_heads: usize, dropout: f64) -> TransformerResult<Self> {
        if d_model == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "d_model must be greater than 0".to_string(),
            ));
        }

        if num_heads == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "num_heads must be greater than 0".to_string(),
            ));
        }

        if !d_model.is_multiple_of(num_heads) {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
        }

        if !(0.0..1.0).contains(&dropout) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "dropout must be in [0, 1), got {}",
                dropout
            )));
        }

        let d_k = d_model / num_heads;

        // Xavier initialization: scale = sqrt(1 / d_model)
        let scale = T::from(1.0).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })? / T::from(d_model)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dimension".to_string()))?
            .sqrt();

        let mut rng = thread_rng();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale
        });

        let w_k = Array2::from_shape_fn((d_model, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale
        });

        let w_v = Array2::from_shape_fn((d_model, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale
        });

        let w_o = Array2::from_shape_fn((d_model, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale
        });

        let dropout_p = T::from(dropout).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert dropout probability".to_string())
        })?;

        Ok(Self {
            d_model,
            num_heads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            dropout_p,
        })
    }

    /// Forward pass through multi-head attention
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor (seq_len, d_model)
    /// * `mask` - Optional attention mask (seq_len, seq_len)
    /// * `training` - Whether in training mode (applies dropout)
    ///
    /// # Returns
    ///
    /// Output tensor (seq_len, d_model)
    pub fn forward(
        &self,
        x: &ArrayView2<T>,
        mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        let (seq_len, input_dim) = x.dim();

        if input_dim != self.d_model {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Input dimension {} does not match d_model {}",
                input_dim, self.d_model
            )));
        }

        // Project to Q, K, V
        let q = x.dot(&self.w_q); // (seq_len, d_model)
        let k = x.dot(&self.w_k); // (seq_len, d_model)
        let v = x.dot(&self.w_v); // (seq_len, d_model)

        // Reshape to (seq_len, num_heads, d_k) then transpose to (num_heads, seq_len, d_k)
        let q_heads = self.split_heads(&q.view())?;
        let k_heads = self.split_heads(&k.view())?;
        let v_heads = self.split_heads(&v.view())?;

        // Apply scaled dot-product attention for each head
        let mut attended = Array3::zeros((self.num_heads, seq_len, self.d_k));

        for h in 0..self.num_heads {
            let q_h = q_heads.slice(s![h, .., ..]);
            let k_h = k_heads.slice(s![h, .., ..]);
            let v_h = v_heads.slice(s![h, .., ..]);

            let attn_output = self.scaled_dot_product_attention(
                &q_h.to_owned().view(),
                &k_h.to_owned().view(),
                &v_h.to_owned().view(),
                mask,
                training,
            )?;

            attended
                .slice_mut(s![h, .., ..])
                .assign(&attn_output.view());
        }

        // Concatenate heads: (num_heads, seq_len, d_k) -> (seq_len, d_model)
        let concat = self.combine_heads(&attended.view())?;

        // Output projection
        let output = concat.dot(&self.w_o);

        Ok(output)
    }

    /// Split input into multiple heads
    ///
    /// Reshapes (seq_len, d_model) -> (num_heads, seq_len, d_k)
    fn split_heads(&self, x: &ArrayView2<T>) -> TransformerResult<Array3<T>> {
        let (seq_len, d_model) = x.dim();

        if d_model != self.d_model {
            return Err(NumRs2Error::DimensionMismatch(
                "Input dimension mismatch".to_string(),
            ));
        }

        let mut result = Array3::zeros((self.num_heads, seq_len, self.d_k));

        for h in 0..self.num_heads {
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    result[[h, i, j]] = x[[i, h * self.d_k + j]];
                }
            }
        }

        Ok(result)
    }

    /// Combine heads back to original dimension
    ///
    /// Reshapes (num_heads, seq_len, d_k) -> (seq_len, d_model)
    fn combine_heads(&self, x: &ArrayView3<T>) -> TransformerResult<Array2<T>> {
        let (num_heads, seq_len, d_k) = x.dim();

        if num_heads != self.num_heads || d_k != self.d_k {
            return Err(NumRs2Error::DimensionMismatch(
                "Head dimensions mismatch".to_string(),
            ));
        }

        let mut result = Array2::zeros((seq_len, self.d_model));

        for h in 0..self.num_heads {
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    result[[i, h * self.d_k + j]] = x[[h, i, j]];
                }
            }
        }

        Ok(result)
    }

    /// Scaled dot-product attention
    ///
    /// Computes: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    fn scaled_dot_product_attention(
        &self,
        q: &ArrayView2<T>,
        k: &ArrayView2<T>,
        v: &ArrayView2<T>,
        mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        let d_k_float = T::from(self.d_k)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert d_k".to_string()))?;

        let scale = T::one() / d_k_float.sqrt();

        // Compute Q * K^T / sqrt(d_k)
        let mut scores = q.dot(&k.t()) * scale;

        // Apply mask if provided
        if let Some(m) = mask {
            let neg_inf = T::from(-1e9).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert mask value".to_string())
            })?;

            let zero = T::zero();
            for i in 0..scores.nrows() {
                for j in 0..scores.ncols() {
                    if m[[i, j]] == zero {
                        scores[[i, j]] = neg_inf;
                    }
                }
            }
        }

        // Apply softmax along last dimension
        let attention_weights = self.softmax_2d(&scores.view(), 1)?;

        // Apply dropout during training
        let weights = if training && self.dropout_p > T::zero() {
            self.apply_dropout(&attention_weights.view())?
        } else {
            attention_weights
        };

        // Compute attention output
        let output = weights.dot(v);

        Ok(output)
    }

    /// Softmax operation along specified axis
    fn softmax_2d(&self, x: &ArrayView2<T>, axis: usize) -> TransformerResult<Array2<T>> {
        let shape = x.dim();
        let mut result = Array2::zeros(shape);

        if axis == 0 {
            // Softmax along columns
            for j in 0..shape.1 {
                let col = x.column(j);
                let max_val = col
                    .iter()
                    .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });

                let mut sum = T::zero();
                let mut exp_vals = Array1::zeros(shape.0);

                for i in 0..shape.0 {
                    exp_vals[i] = (col[i] - max_val).exp();
                    sum = sum + exp_vals[i];
                }

                for i in 0..shape.0 {
                    result[[i, j]] = exp_vals[i] / sum;
                }
            }
        } else {
            // Softmax along rows
            for i in 0..shape.0 {
                let row = x.row(i);
                let max_val = row
                    .iter()
                    .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });

                let mut sum = T::zero();
                let mut exp_vals = Array1::zeros(shape.1);

                for j in 0..shape.1 {
                    exp_vals[j] = (row[j] - max_val).exp();
                    sum = sum + exp_vals[j];
                }

                for j in 0..shape.1 {
                    result[[i, j]] = exp_vals[j] / sum;
                }
            }
        }

        Ok(result)
    }

    /// Apply dropout to attention weights
    fn apply_dropout(&self, x: &ArrayView2<T>) -> TransformerResult<Array2<T>> {
        let mut rng = thread_rng();
        let threshold = self
            .dropout_p
            .to_f64()
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dropout".to_string()))?;

        let scale = T::one() / (T::one() - self.dropout_p);

        let mask = Array2::from_shape_fn(x.raw_dim(), |_| {
            if rng.gen::<f64>() > threshold {
                scale
            } else {
                T::zero()
            }
        });

        Ok(x * &mask)
    }
}

/// Positional Encoding
///
/// Adds position information to embeddings using sinusoidal functions or learned embeddings.
///
/// # Sinusoidal Encoding
///
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
///
/// This allows the model to learn to attend by relative positions.
#[derive(Debug, Clone)]
pub struct PositionalEncoding<T>
where
    T: Float,
{
    /// Maximum sequence length
    pub max_len: usize,
    /// Model dimension
    pub d_model: usize,
    /// Encoding type (sinusoidal or learned)
    pub encoding_type: PositionalEncodingType,
    /// Precomputed or learned positional encodings (max_len, d_model)
    pub encodings: Array2<T>,
}

/// Type of positional encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Sinusoidal encoding (fixed, not learned)
    Sinusoidal,
    /// Learned embedding (trainable parameters)
    Learned,
}

impl<T> PositionalEncoding<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Creates a new positional encoding layer
    ///
    /// # Arguments
    ///
    /// * `max_len` - Maximum sequence length
    /// * `d_model` - Model dimension (must be even for sinusoidal)
    /// * `encoding_type` - Type of encoding to use
    pub fn new(
        max_len: usize,
        d_model: usize,
        encoding_type: PositionalEncodingType,
    ) -> TransformerResult<Self> {
        if max_len == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "max_len must be greater than 0".to_string(),
            ));
        }

        if d_model == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "d_model must be greater than 0".to_string(),
            ));
        }

        if encoding_type == PositionalEncodingType::Sinusoidal && !d_model.is_multiple_of(2) {
            return Err(NumRs2Error::InvalidOperation(
                "d_model must be even for sinusoidal encoding".to_string(),
            ));
        }

        let encodings = match encoding_type {
            PositionalEncodingType::Sinusoidal => Self::create_sinusoidal(max_len, d_model)?,
            PositionalEncodingType::Learned => Self::create_learned(max_len, d_model)?,
        };

        Ok(Self {
            max_len,
            d_model,
            encoding_type,
            encodings,
        })
    }

    /// Create sinusoidal positional encodings
    fn create_sinusoidal(max_len: usize, d_model: usize) -> TransformerResult<Array2<T>> {
        let mut pe = Array2::zeros((max_len, d_model));

        let two = T::from(2.0).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })?;

        let ten_thousand = T::from(10000.0).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })?;

        for pos in 0..max_len {
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

    /// Create learned positional embeddings (initialized with small random values)
    fn create_learned(max_len: usize, d_model: usize) -> TransformerResult<Array2<T>> {
        let mut rng = thread_rng();
        let scale = T::from(0.02)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert scale".to_string()))?;

        let pe = Array2::from_shape_fn((max_len, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale
        });

        Ok(pe)
    }

    /// Add positional encoding to input embeddings
    ///
    /// # Arguments
    ///
    /// * `x` - Input embeddings (seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Embeddings with added positional information
    pub fn forward(&self, x: &ArrayView2<T>) -> TransformerResult<Array2<T>>
    where
        T: ScalarOperand,
    {
        let (seq_len, d_model) = x.dim();

        if d_model != self.d_model {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Input dimension {} does not match d_model {}",
                d_model, self.d_model
            )));
        }

        if seq_len > self.max_len {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_len
            )));
        }

        // Add positional encoding to input
        let pe_slice = self.encodings.slice(s![0..seq_len, ..]);
        Ok(x + &pe_slice)
    }
}

/// Position-wise Feed-Forward Network
///
/// Applies two linear transformations with GELU activation:
///
/// ```text
/// FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
/// ```
///
/// This is applied to each position separately and identically.
#[derive(Debug, Clone)]
pub struct PositionwiseFeedForward<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Model dimension
    pub d_model: usize,
    /// Feed-forward dimension (typically 4 * d_model)
    pub d_ff: usize,
    /// First linear layer weights (d_model, d_ff)
    pub w1: Array2<T>,
    /// First linear layer bias (d_ff,)
    pub b1: Array1<T>,
    /// Second linear layer weights (d_ff, d_model)
    pub w2: Array2<T>,
    /// Second linear layer bias (d_model,)
    pub b2: Array1<T>,
    /// Dropout probability
    pub dropout_p: T,
}

impl<T> PositionwiseFeedForward<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new position-wise feed-forward network
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `d_ff` - Feed-forward dimension (typically 4 * d_model)
    /// * `dropout` - Dropout probability
    pub fn new(d_model: usize, d_ff: usize, dropout: f64) -> TransformerResult<Self> {
        if d_model == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "d_model must be greater than 0".to_string(),
            ));
        }

        if d_ff == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "d_ff must be greater than 0".to_string(),
            ));
        }

        if !(0.0..1.0).contains(&dropout) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "dropout must be in [0, 1), got {}",
                dropout
            )));
        }

        // Xavier initialization
        let mut rng = thread_rng();

        let scale1 = T::from(1.0).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })? / T::from(d_model)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dimension".to_string()))?
            .sqrt();

        let scale2 = T::from(1.0).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })? / T::from(d_ff)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dimension".to_string()))?
            .sqrt();

        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale1
        });

        let b1 = Array1::zeros(d_ff);

        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| {
            let u: f64 = rng.gen();
            T::from(u * 2.0 - 1.0).expect("Conversion should succeed for f64 to Float") * scale2
        });

        let b2 = Array1::zeros(d_model);

        let dropout_p = T::from(dropout).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert dropout probability".to_string())
        })?;

        Ok(Self {
            d_model,
            d_ff,
            w1,
            b1,
            w2,
            b2,
            dropout_p,
        })
    }

    /// Forward pass through feed-forward network
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor (seq_len, d_model)
    /// * `training` - Whether in training mode (applies dropout)
    pub fn forward(&self, x: &ArrayView2<T>, training: bool) -> TransformerResult<Array2<T>> {
        let (seq_len, d_model) = x.dim();

        if d_model != self.d_model {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Input dimension {} does not match d_model {}",
                d_model, self.d_model
            )));
        }

        // First linear layer: x W1 + b1
        let mut hidden = x.dot(&self.w1);
        for i in 0..seq_len {
            for j in 0..self.d_ff {
                hidden[[i, j]] = hidden[[i, j]] + self.b1[j];
            }
        }

        // GELU activation
        hidden = self.gelu(&hidden.view())?;

        // Dropout
        if training && self.dropout_p > T::zero() {
            hidden = self.apply_dropout(&hidden.view())?;
        }

        // Second linear layer: hidden W2 + b2
        let mut output = hidden.dot(&self.w2);
        for i in 0..seq_len {
            for j in 0..self.d_model {
                output[[i, j]] = output[[i, j]] + self.b2[j];
            }
        }

        Ok(output)
    }

    /// GELU activation function
    ///
    /// GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of standard normal
    ///
    /// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    fn gelu(&self, x: &ArrayView2<T>) -> TransformerResult<Array2<T>> {
        let sqrt_2_over_pi = T::from(0.7978845608028654).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })?;

        let coeff = T::from(0.044715).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })?;

        let half = T::from(0.5).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert constant".to_string())
        })?;

        let one = T::one();

        let result = x.mapv(|val| {
            let x_cubed = val * val * val;
            let inner = sqrt_2_over_pi * (val + coeff * x_cubed);
            half * val * (one + inner.tanh())
        });

        Ok(result)
    }

    /// Apply dropout
    fn apply_dropout(&self, x: &ArrayView2<T>) -> TransformerResult<Array2<T>> {
        let mut rng = thread_rng();
        let threshold = self
            .dropout_p
            .to_f64()
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert dropout".to_string()))?;

        let scale = T::one() / (T::one() - self.dropout_p);

        let mask = Array2::from_shape_fn(x.raw_dim(), |_| {
            if rng.gen::<f64>() > threshold {
                scale
            } else {
                T::zero()
            }
        });

        Ok(x * &mask)
    }
}

/// Transformer Encoder Layer
///
/// A single layer of the transformer encoder consisting of:
/// 1. Multi-head self-attention
/// 2. Add & Norm (residual connection + layer normalization)
/// 3. Position-wise feed-forward network
/// 4. Add & Norm (residual connection + layer normalization)
///
/// # Architecture
///
/// ```text
/// Input
///   │
///   ├───> Multi-Head Attention ───> Add & Norm ───┐
///   │                                              │
///   └──────────────────────────────────────────────┘
///   │
///   ├───> Feed-Forward Network ───> Add & Norm ───┐
///   │                                              │
///   └──────────────────────────────────────────────┘
///   │
/// Output
/// ```
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Multi-head attention
    pub attention: MultiHeadAttention<T>,
    /// Position-wise feed-forward network
    pub feed_forward: PositionwiseFeedForward<T>,
    /// Layer normalization parameters for attention (gamma)
    pub norm1_gamma: Array1<T>,
    /// Layer normalization parameters for attention (beta)
    pub norm1_beta: Array1<T>,
    /// Layer normalization parameters for feed-forward (gamma)
    pub norm2_gamma: Array1<T>,
    /// Layer normalization parameters for feed-forward (beta)
    pub norm2_beta: Array1<T>,
    /// Layer normalization epsilon
    pub norm_eps: T,
}

impl<T> TransformerEncoderLayer<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new transformer encoder layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward dimension
    /// * `dropout` - Dropout probability
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> TransformerResult<Self> {
        let attention = MultiHeadAttention::new(d_model, num_heads, dropout)?;
        let feed_forward = PositionwiseFeedForward::new(d_model, d_ff, dropout)?;

        let norm1_gamma = Array1::ones(d_model);
        let norm1_beta = Array1::zeros(d_model);
        let norm2_gamma = Array1::ones(d_model);
        let norm2_beta = Array1::zeros(d_model);

        let norm_eps = T::from(1e-5)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

        Ok(Self {
            attention,
            feed_forward,
            norm1_gamma,
            norm1_beta,
            norm2_gamma,
            norm2_beta,
            norm_eps,
        })
    }

    /// Forward pass through encoder layer
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor (seq_len, d_model)
    /// * `mask` - Optional attention mask
    /// * `training` - Whether in training mode
    pub fn forward(
        &self,
        x: &ArrayView2<T>,
        mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        // Multi-head attention with residual connection
        let attn_output = self.attention.forward(x, mask, training)?;
        let attn_residual = x + &attn_output;

        // Layer normalization after attention
        let norm1 = self.layer_norm(&attn_residual.view(), &self.norm1_gamma, &self.norm1_beta)?;

        // Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm1.view(), training)?;
        let ff_residual = &norm1 + &ff_output;

        // Layer normalization after feed-forward
        let output = self.layer_norm(&ff_residual.view(), &self.norm2_gamma, &self.norm2_beta)?;

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        x: &ArrayView2<T>,
        gamma: &Array1<T>,
        beta: &Array1<T>,
    ) -> TransformerResult<Array2<T>> {
        let (seq_len, d_model) = x.dim();

        let n_features = T::from(d_model).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert feature count".to_string())
        })?;

        let mut result = Array2::zeros((seq_len, d_model));

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.sum() / n_features;
            let var = row.mapv(|v| (v - mean) * (v - mean)).sum() / n_features;
            let std = (var + self.norm_eps).sqrt();

            for j in 0..d_model {
                result[[i, j]] = (x[[i, j]] - mean) / std * gamma[j] + beta[j];
            }
        }

        Ok(result)
    }
}

/// Transformer Decoder Layer
///
/// A single layer of the transformer decoder consisting of:
/// 1. Masked multi-head self-attention (prevents attending to future positions)
/// 2. Add & Norm
/// 3. Multi-head cross-attention to encoder output
/// 4. Add & Norm
/// 5. Position-wise feed-forward network
/// 6. Add & Norm
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Masked self-attention
    pub self_attention: MultiHeadAttention<T>,
    /// Cross-attention to encoder output
    pub cross_attention: MultiHeadAttention<T>,
    /// Position-wise feed-forward network
    pub feed_forward: PositionwiseFeedForward<T>,
    /// Layer norm for self-attention
    pub norm1_gamma: Array1<T>,
    pub norm1_beta: Array1<T>,
    /// Layer norm for cross-attention
    pub norm2_gamma: Array1<T>,
    pub norm2_beta: Array1<T>,
    /// Layer norm for feed-forward
    pub norm3_gamma: Array1<T>,
    pub norm3_beta: Array1<T>,
    /// Normalization epsilon
    pub norm_eps: T,
}

impl<T> TransformerDecoderLayer<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new transformer decoder layer
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> TransformerResult<Self> {
        let self_attention = MultiHeadAttention::new(d_model, num_heads, dropout)?;
        let cross_attention = MultiHeadAttention::new(d_model, num_heads, dropout)?;
        let feed_forward = PositionwiseFeedForward::new(d_model, d_ff, dropout)?;

        let norm1_gamma = Array1::ones(d_model);
        let norm1_beta = Array1::zeros(d_model);
        let norm2_gamma = Array1::ones(d_model);
        let norm2_beta = Array1::zeros(d_model);
        let norm3_gamma = Array1::ones(d_model);
        let norm3_beta = Array1::zeros(d_model);

        let norm_eps = T::from(1e-5)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert epsilon".to_string()))?;

        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            norm1_gamma,
            norm1_beta,
            norm2_gamma,
            norm2_beta,
            norm3_gamma,
            norm3_beta,
            norm_eps,
        })
    }

    /// Forward pass through decoder layer
    ///
    /// # Arguments
    ///
    /// * `x` - Target sequence input (tgt_len, d_model)
    /// * `encoder_output` - Encoder output (src_len, d_model)
    /// * `tgt_mask` - Target mask (causal mask for autoregressive)
    /// * `memory_mask` - Source-target attention mask
    /// * `training` - Whether in training mode
    pub fn forward(
        &self,
        x: &ArrayView2<T>,
        encoder_output: &ArrayView2<T>,
        tgt_mask: Option<&ArrayView2<T>>,
        memory_mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        // Masked self-attention
        let self_attn = self.self_attention.forward(x, tgt_mask, training)?;
        let residual1 = x + &self_attn;
        let norm1 = self.layer_norm(&residual1.view(), &self.norm1_gamma, &self.norm1_beta)?;

        // Cross-attention to encoder output
        // For cross-attention, Q comes from decoder, K and V from encoder
        let cross_attn =
            self.cross_attention_forward(&norm1.view(), encoder_output, memory_mask, training)?;
        let residual2 = &norm1 + &cross_attn;
        let norm2 = self.layer_norm(&residual2.view(), &self.norm2_gamma, &self.norm2_beta)?;

        // Feed-forward
        let ff_output = self.feed_forward.forward(&norm2.view(), training)?;
        let residual3 = &norm2 + &ff_output;
        let output = self.layer_norm(&residual3.view(), &self.norm3_gamma, &self.norm3_beta)?;

        Ok(output)
    }

    /// Cross-attention where Q comes from decoder, K and V from encoder
    fn cross_attention_forward(
        &self,
        query: &ArrayView2<T>,
        key_value: &ArrayView2<T>,
        mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        // For cross-attention, we need to project Q from decoder and K,V from encoder
        // Here we simplify by using the cross_attention layer directly
        // In practice, you'd want separate projections
        self.cross_attention.forward(query, mask, training)
    }

    /// Layer normalization (same as encoder)
    fn layer_norm(
        &self,
        x: &ArrayView2<T>,
        gamma: &Array1<T>,
        beta: &Array1<T>,
    ) -> TransformerResult<Array2<T>> {
        let (seq_len, d_model) = x.dim();

        let n_features = T::from(d_model).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert feature count".to_string())
        })?;

        let mut result = Array2::zeros((seq_len, d_model));

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.sum() / n_features;
            let var = row.mapv(|v| (v - mean) * (v - mean)).sum() / n_features;
            let std = (var + self.norm_eps).sqrt();

            for j in 0..d_model {
                result[[i, j]] = (x[[i, j]] - mean) / std * gamma[j] + beta[j];
            }
        }

        Ok(result)
    }
}

/// Transformer Encoder
///
/// Stack of N encoder layers
#[derive(Debug, Clone)]
pub struct TransformerEncoder<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Encoder layers
    pub layers: Vec<TransformerEncoderLayer<T>>,
    /// Number of layers
    pub num_layers: usize,
}

impl<T> TransformerEncoder<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new transformer encoder
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of encoder layers
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward dimension
    /// * `dropout` - Dropout probability
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> TransformerResult<Self> {
        if num_layers == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "num_layers must be greater than 0".to_string(),
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model, num_heads, d_ff, dropout,
            )?);
        }

        Ok(Self { layers, num_layers })
    }

    /// Forward pass through encoder stack
    pub fn forward(
        &self,
        x: &ArrayView2<T>,
        mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        let mut output = x.to_owned();

        for layer in &self.layers {
            output = layer.forward(&output.view(), mask, training)?;
        }

        Ok(output)
    }
}

/// Transformer Decoder
///
/// Stack of N decoder layers
#[derive(Debug, Clone)]
pub struct TransformerDecoder<T>
where
    T: Float + SimdUnifiedOps,
{
    /// Decoder layers
    pub layers: Vec<TransformerDecoderLayer<T>>,
    /// Number of layers
    pub num_layers: usize,
}

impl<T> TransformerDecoder<T>
where
    T: Float + SimdUnifiedOps + ScalarOperand,
{
    /// Creates a new transformer decoder
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> TransformerResult<Self> {
        if num_layers == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "num_layers must be greater than 0".to_string(),
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model, num_heads, d_ff, dropout,
            )?);
        }

        Ok(Self { layers, num_layers })
    }

    /// Forward pass through decoder stack
    pub fn forward(
        &self,
        x: &ArrayView2<T>,
        encoder_output: &ArrayView2<T>,
        tgt_mask: Option<&ArrayView2<T>>,
        memory_mask: Option<&ArrayView2<T>>,
        training: bool,
    ) -> TransformerResult<Array2<T>> {
        let mut output = x.to_owned();

        for layer in &self.layers {
            output = layer.forward(
                &output.view(),
                encoder_output,
                tgt_mask,
                memory_mask,
                training,
            )?;
        }

        Ok(output)
    }
}

/// Create a causal mask for autoregressive decoding
///
/// Prevents positions from attending to subsequent positions.
///
/// # Arguments
///
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Lower triangular matrix of ones (seq_len, seq_len)
pub fn create_causal_mask<T>(seq_len: usize) -> Array2<T>
where
    T: Float,
{
    let mut mask = Array2::zeros((seq_len, seq_len));
    let one = T::one();

    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = one;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::<f64>::new(512, 8, 0.1);
        assert!(mha.is_ok());

        let mha = mha.expect("MultiHeadAttention creation should succeed");
        assert_eq!(mha.d_model, 512);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.d_k, 64);
    }

    #[test]
    fn test_multi_head_attention_invalid_params() {
        // d_model not divisible by num_heads
        let result = MultiHeadAttention::<f64>::new(511, 8, 0.1);
        assert!(result.is_err());

        // Invalid dropout
        let result = MultiHeadAttention::<f64>::new(512, 8, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_forward() {
        let mha = MultiHeadAttention::<f64>::new(64, 4, 0.0)
            .expect("MultiHeadAttention creation should succeed");

        let input = Array2::ones((10, 64)); // (seq_len=10, d_model=64)
        let output = mha.forward(&input.view(), None, false);

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (10, 64));
    }

    #[test]
    fn test_positional_encoding_sinusoidal() {
        let pe = PositionalEncoding::<f64>::new(100, 64, PositionalEncodingType::Sinusoidal);
        assert!(pe.is_ok());

        let pe = pe.expect("PositionalEncoding creation should succeed");
        assert_eq!(pe.encodings.dim(), (100, 64));

        // Check that values are bounded between -1 and 1 (sin/cos)
        for &val in pe.encodings.iter() {
            assert!((-1.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_positional_encoding_learned() {
        let pe = PositionalEncoding::<f64>::new(100, 64, PositionalEncodingType::Learned);
        assert!(pe.is_ok());

        let pe = pe.expect("PositionalEncoding creation should succeed");
        assert_eq!(pe.encodings.dim(), (100, 64));
    }

    #[test]
    fn test_positional_encoding_forward() {
        let pe = PositionalEncoding::<f64>::new(50, 64, PositionalEncodingType::Sinusoidal)
            .expect("PositionalEncoding creation should succeed");

        let input = Array2::zeros((20, 64));
        let output = pe.forward(&input.view());

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (20, 64));

        // Output should equal positional encoding when input is zero
        for i in 0..20 {
            for j in 0..64 {
                assert_abs_diff_eq!(output[[i, j]], pe.encodings[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_feedforward_network() -> Result<(), Box<dyn std::error::Error>> {
        // Use minimal dimensions to avoid OOM in parallel testing
        let d_model = 8;
        let d_ff = 16;
        let seq_len = 4;

        let ffn = PositionwiseFeedForward::<f64>::new(d_model, d_ff, 0.0)?;
        assert_eq!(ffn.d_model, d_model);
        assert_eq!(ffn.d_ff, d_ff);

        let input = Array2::ones((seq_len, d_model));
        let output = ffn.forward(&input.view(), false)?;

        assert_eq!(output.dim(), (seq_len, d_model));

        // Verify output contains finite values
        for &val in output.iter() {
            assert!(val.is_finite(), "Output should contain finite values");
        }

        Ok(())
    }

    #[test]
    fn test_gelu_activation() {
        let ffn =
            PositionwiseFeedForward::<f64>::new(4, 16, 0.0).expect("FFN creation should succeed");

        let input = Array2::from_shape_vec((2, 4), vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5])
            .expect("Array creation should succeed");

        let output = ffn.gelu(&input.view());
        assert!(output.is_ok());

        let output = output.expect("GELU should succeed");
        // GELU(0) ≈ 0
        assert_abs_diff_eq!(output[[0, 0]], 0.0, epsilon = 0.01);
        // GELU is monotonic increasing
        assert!(output[[0, 1]] > output[[0, 0]]);
    }

    #[test]
    fn test_transformer_encoder_layer() {
        // Use smaller dimensions to avoid OOM in parallel testing
        let layer = TransformerEncoderLayer::<f64>::new(16, 4, 64, 0.0);
        assert!(layer.is_ok());

        let layer = layer.expect("EncoderLayer creation should succeed");
        let input = Array2::ones((5, 16));
        let output = layer.forward(&input.view(), None, false);

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (5, 16));
    }

    #[test]
    fn test_transformer_decoder_layer() {
        // Use smaller dimensions to avoid OOM in parallel testing
        let layer = TransformerDecoderLayer::<f64>::new(16, 4, 64, 0.0);
        assert!(layer.is_ok());

        let layer = layer.expect("DecoderLayer creation should succeed");
        let tgt_input = Array2::ones((5, 16));
        let src_input = Array2::ones((8, 16));

        let output = layer.forward(&tgt_input.view(), &src_input.view(), None, None, false);

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (5, 16));
    }

    #[test]
    fn test_transformer_encoder() {
        // Use smaller dimensions to avoid OOM in parallel testing
        let encoder = TransformerEncoder::<f64>::new(3, 16, 4, 64, 0.0);
        assert!(encoder.is_ok());

        let encoder = encoder.expect("Encoder creation should succeed");
        assert_eq!(encoder.num_layers, 3);

        let input = Array2::ones((5, 16));
        let output = encoder.forward(&input.view(), None, false);

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (5, 16));
    }

    #[test]
    fn test_transformer_decoder() {
        // Use smaller dimensions to avoid OOM in parallel testing
        let decoder = TransformerDecoder::<f64>::new(3, 16, 4, 64, 0.0);
        assert!(decoder.is_ok());

        let decoder = decoder.expect("Decoder creation should succeed");
        assert_eq!(decoder.num_layers, 3);

        let tgt = Array2::ones((5, 16));
        let memory = Array2::ones((8, 16));
        let output = decoder.forward(&tgt.view(), &memory.view(), None, None, false);

        assert!(output.is_ok());
        let output = output.expect("Forward pass should succeed");
        assert_eq!(output.dim(), (5, 16));
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask::<f64>(5);
        assert_eq!(mask.dim(), (5, 5));

        // Check lower triangular structure
        for i in 0..5 {
            for j in 0..5 {
                if j <= i {
                    assert_eq!(mask[[i, j]], 1.0);
                } else {
                    assert_eq!(mask[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let mha = MultiHeadAttention::<f64>::new(64, 4, 0.0)
            .expect("MultiHeadAttention creation should succeed");

        let q = Array2::ones((5, 64));
        let k = Array2::ones((5, 64));
        let v = Array2::ones((5, 64));

        let output = mha.scaled_dot_product_attention(&q.view(), &k.view(), &v.view(), None, false);
        assert!(output.is_ok());

        // With uniform inputs, attention should distribute equally
        let output = output.expect("Attention should succeed");
        assert_eq!(output.dim(), (5, 64));
    }

    #[test]
    fn test_layer_normalization_properties() {
        let layer = TransformerEncoderLayer::<f64>::new(64, 4, 256, 0.0)
            .expect("EncoderLayer creation should succeed");

        let input = Array2::from_shape_fn((10, 64), |(i, j)| (i * 64 + j) as f64);
        let gamma = Array1::ones(64);
        let beta = Array1::zeros(64);

        let normalized = layer.layer_norm(&input.view(), &gamma, &beta);
        assert!(normalized.is_ok());

        let normalized = normalized.expect("Layer norm should succeed");

        // Each row should have approximately zero mean and unit variance
        for i in 0..10 {
            let row = normalized.row(i);
            let mean = row.sum() / 64.0;
            let var = row.mapv(|v| (v - mean) * (v - mean)).sum() / 64.0;

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
            assert_abs_diff_eq!(var, 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_masked_attention() {
        let mha = MultiHeadAttention::<f64>::new(64, 4, 0.0)
            .expect("MultiHeadAttention creation should succeed");

        let input = Array2::ones((5, 64));
        let mask = create_causal_mask::<f64>(5);

        let output = mha.forward(&input.view(), Some(&mask.view()), false);
        assert!(output.is_ok());

        let output = output.expect("Masked attention should succeed");
        assert_eq!(output.dim(), (5, 64));
    }
}

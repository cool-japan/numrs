# Neural Network Module Guide

**NumRS2 v0.2.0 Neural Network Primitives**

This guide provides comprehensive documentation for NumRS2's neural network (`nn`) module, including architecture overview, usage patterns, performance optimization, and best practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Activation Functions](#activation-functions)
5. [Convolution Operations](#convolution-operations)
6. [Pooling Layers](#pooling-layers)
7. [Normalization](#normalization)
8. [Attention Mechanisms](#attention-mechanisms)
9. [Loss Functions](#loss-functions)
10. [SIMD Optimization](#simd-optimization)
11. [Performance Tips](#performance-tips)
12. [Training Patterns](#training-patterns)
13. [Error Handling](#error-handling)
14. [Examples](#examples)
15. [API Reference](#api-reference)

---

## Overview

The NumRS2 neural network module provides high-performance building blocks for deep learning applications. All operations are:

- **SIMD-optimized**: Leveraging AVX2/AVX512 (x86) and NEON (ARM)
- **Pure Rust**: No C/C++ dependencies via OxiBLAS
- **Numerically stable**: Careful handling of edge cases and overflow
- **Well-tested**: Comprehensive test coverage with property-based tests
- **SciRS2-integrated**: Following NumRS2's ecosystem policies

### Key Design Principles

1. **Performance First**: SIMD acceleration for all hot paths
2. **Numerical Stability**: Robust handling of edge cases
3. **Ergonomic API**: Clean, intuitive function signatures
4. **Zero-Cost Abstractions**: Compile-time optimization
5. **Pure Rust**: No foreign function interface overhead

---

## Architecture

### Module Structure

```
src/nn/
├── mod.rs              # Module exports and common types
├── activation.rs       # Activation functions
├── attention.rs        # Attention mechanisms
├── conv.rs            # Convolution operations
├── loss.rs            # Loss functions
├── normalization.rs   # Normalization and dropout
├── pooling.rs         # Pooling operations
└── simd_ops.rs        # SIMD-optimized kernels
```

### SCIRS2 Integration

The `nn` module strictly follows NumRS2's SCIRS2 integration policy:

```rust
// ✅ CORRECT: Use SciRS2 abstractions
use scirs2_core::ndarray::*;          // Array operations
use scirs2_core::simd_ops::*;         // SIMD operations
use scirs2_core::random::*;           // RNG for dropout
use scirs2_linalg::*;                 // Linear algebra (OxiBLAS)

// ❌ FORBIDDEN: Direct external dependencies
// use ndarray::*;                    // Use scirs2_core::ndarray
// use rand::*;                       // Use scirs2_core::random
```

### Type System

```rust
// Result type for all operations
pub type NnResult<T> = Result<T, NumRs2Error>;

// Reduction modes for loss functions
pub enum ReductionMode {
    None,   // No reduction (return per-element loss)
    Mean,   // Average over all elements
    Sum,    // Sum over all elements
}

// Padding modes for convolution
pub enum PaddingMode {
    Valid,             // No padding
    Same,              // Preserve input size
    Full,              // Maximum padding
    Explicit(usize),   // Custom padding
}

// Data format for tensors
pub enum DataFormat {
    NCHW,  // Channels first (N, C, H, W)
    NHWC,  // Channels last (N, H, W, C)
}
```

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.2.0"
scirs2-core = "0.1.1"
```

### Basic Example

```rust
use numrs2::nn::*;
use scirs2_core::ndarray::{Array1, Array2, array};

fn main() -> NnResult<()> {
    // Simple feedforward pass
    let input = array![-1.0, 0.0, 1.0, 2.0];

    // Apply activation
    let hidden = relu(&input.view())?;

    // Apply softmax
    let output = softmax(&hidden.view())?;

    println!("Output: {:?}", output);
    Ok(())
}
```

### Building a Network

```rust
use numrs2::nn::*;
use scirs2_core::ndarray::{Array1, Array2};

fn feedforward(
    input: &Array2<f64>,
    weights1: &Array2<f64>,
    weights2: &Array2<f64>,
) -> NnResult<Array2<f64>> {
    // Hidden layer: Linear + ReLU + BatchNorm
    let hidden = simd_matmul_f64(&input.view(), &weights1.view())?;
    let hidden = relu_2d(&hidden.view())?;

    let gamma = Array1::ones(hidden.ncols());
    let beta = Array1::zeros(hidden.ncols());
    let hidden = batch_norm_1d(&hidden.view(), &gamma.view(), &beta.view(), 1e-5)?;

    // Output layer: Linear + Softmax
    let output = simd_matmul_f64(&hidden.view(), &weights2.view())?;
    let output = softmax_2d(&output.view(), 1)?;

    Ok(output)
}
```

---

## Activation Functions

### ReLU and Variants

#### ReLU (Rectified Linear Unit)

**Formula**: `f(x) = max(0, x)`

```rust
use numrs2::nn::activation::*;
use scirs2_core::ndarray::array;

let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = relu(&x.view())?;
// y = [0.0, 0.0, 0.0, 1.0, 2.0]
```

**Properties**:
- Fast computation (simple comparison)
- Non-saturating for positive values
- Can cause "dying ReLU" problem (neurons output 0)
- SIMD-optimized: ~4-8x speedup

**When to use**: Default choice for hidden layers, especially in CNNs.

#### Leaky ReLU

**Formula**: `f(x) = x if x > 0 else α * x` (typically α = 0.01)

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = leaky_relu(&x.view(), 0.01)?;
// y = [-0.02, -0.01, 0.0, 1.0, 2.0]
```

**When to use**: When experiencing dying ReLU problem.

#### ELU (Exponential Linear Unit)

**Formula**: `f(x) = x if x > 0 else α(exp(x) - 1)`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = elu(&x.view(), 1.0)?;
```

**Properties**:
- Smooth negative part
- Mean activation closer to zero
- More computationally expensive

**When to use**: When you need smoother gradients than ReLU.

#### SELU (Scaled ELU)

**Formula**: `f(x) = λ * (x if x > 0 else α(exp(x) - 1))`

Constants chosen for self-normalizing properties.

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = selu(&x.view())?;
```

**Properties**:
- Self-normalizing (maintains mean and variance)
- Requires special initialization (LeCun normal)

**When to use**: Deep fully-connected networks (>4 layers).

### Smooth Activations

#### Sigmoid

**Formula**: `f(x) = 1 / (1 + exp(-x))`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = sigmoid(&x.view())?;
// y ≈ [0.119, 0.269, 0.5, 0.731, 0.881]
```

**Properties**:
- Output range: (0, 1)
- Can saturate (gradients → 0)

**When to use**: Binary classification output, gates in RNNs.

#### Tanh

**Formula**: `f(x) = tanh(x)`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = tanh(&x.view())?;
// y ≈ [-0.964, -0.762, 0.0, 0.762, 0.964]
```

**Properties**:
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)

**When to use**: RNN hidden states, when zero-centered output needed.

### Modern Activations

#### GELU (Gaussian Error Linear Unit)

**Formula**: `f(x) = x * Φ(x)` where Φ is the CDF of standard normal

**Approximation**: `f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = gelu(&x.view())?;
```

**Properties**:
- Smooth, non-monotonic
- Stochastic regularization interpretation
- Used in BERT, GPT models

**When to use**: Transformer models, modern architectures.

#### Swish / SiLU (Sigmoid Linear Unit)

**Formula**: `f(x) = x * sigmoid(x)`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = swish(&x.view())?;
// or: let y = silu(&x.view())?;  // same function
```

**Properties**:
- Self-gating mechanism
- Smooth, non-monotonic
- Better than ReLU in many cases

**When to use**: When you need better performance than ReLU.

#### Mish

**Formula**: `f(x) = x * tanh(softplus(x))`

```rust
let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
let y = mish(&x.view())?;
```

**Properties**:
- Smooth throughout
- Better than Swish in some cases
- More expensive to compute

**When to use**: When training time is not critical and you want best accuracy.

### Probability Distributions

#### Softmax

**Formula**: `f(x)_i = exp(x_i) / Σ exp(x_j)`

```rust
let logits = array![1.0, 2.0, 3.0];
let probs = softmax(&logits.view())?;
// Sum of probs = 1.0
```

**Properties**:
- Converts logits to probabilities
- Temperature parameter available (via scaling input)
- Numerically stable implementation

**When to use**: Multi-class classification output layer.

#### Log-Softmax

**Formula**: `f(x)_i = log(exp(x_i) / Σ exp(x_j))`

```rust
let logits = array![1.0, 2.0, 3.0];
let log_probs = log_softmax(&logits.view())?;
```

**Properties**:
- More numerically stable than log(softmax(x))
- Used with negative log-likelihood loss

**When to use**: When computing log probabilities for NLL loss.

### SIMD-Optimized Variants

All activation functions have SIMD-optimized versions for f32:

```rust
use numrs2::nn::simd_ops::*;

let x = Array1::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0]);

let y1 = simd_relu_f32(&x.view());        // 4-8x faster
let y2 = simd_sigmoid_f32(&x.view());     // 3-5x faster
let y3 = simd_gelu_f32(&x.view());        // 3-4x faster
let y4 = simd_swish_f32(&x.view());       // 3-5x faster
```

**Performance tip**: Use f32 for activations when precision allows (2x more data per SIMD instruction).

---

## Convolution Operations

### 1D Convolution

For sequence/time-series data.

```rust
use numrs2::nn::conv::*;
use scirs2_core::ndarray::array;

let signal = array![1.0, 2.0, 3.0, 4.0, 5.0];
let kernel = array![1.0, 0.0, -1.0];  // Edge detection

let output = conv1d(&signal.view(), &kernel.view(), 1)?;
// Applies convolution with stride 1
```

**Applications**:
- Text processing (character/word level)
- Time series analysis
- Audio signal processing

### 2D Convolution

For image/spatial data.

```rust
use numrs2::nn::conv::*;
use scirs2_core::ndarray::Array2;

let image = Array2::ones((5, 5));
let kernel = Array2::from_shape_vec(
    (3, 3),
    vec![1.0, 1.0, 1.0,
         1.0, -8.0, 1.0,
         1.0, 1.0, 1.0]
)?;  // Laplacian edge detection

let output = conv2d(&image.view(), &kernel.view(), (1, 1))?;
```

**Parameters**:
- `stride`: `(stride_h, stride_w)` - how many pixels to move
- Higher stride = smaller output, faster computation

### Padding Modes

```rust
// Valid convolution (no padding)
let output = conv2d(&image.view(), &kernel.view(), (1, 1))?;

// Explicit padding
let output = conv2d_with_padding(
    &image.view(),
    &kernel.view(),
    (1, 1),
    1  // pad by 1 on all sides
)?;
```

**Padding types**:
- **Valid**: No padding, output smaller than input
- **Same**: Pad to preserve input size (for stride=1)
- **Explicit**: Custom padding amount

### Depthwise Separable Convolution

Efficient convolution for mobile models:

```rust
let output = depthwise_conv2d(&input.view(), &kernel.view(), (1, 1))?;
```

**Benefits**:
- Fewer parameters than regular convolution
- Faster computation
- Used in MobileNet, EfficientNet

---

## Pooling Layers

Reduce spatial dimensions while preserving important features.

### Max Pooling

Extracts maximum value from each window.

```rust
use numrs2::nn::pooling::*;
use scirs2_core::ndarray::Array2;

let feature_map = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);

// Pool with 2x2 window, stride 2
let output = max_pool2d(&feature_map.view(), (2, 2), (2, 2))?;
// Output: (2, 2) - downsampled by 2x
```

**Properties**:
- Preserves strongest activations
- Translation invariant
- Non-differentiable (but sub-differentiable)

**When to use**: CNNs for classification (preserves edge features).

### Average Pooling

Computes mean over each window.

```rust
let output = avg_pool2d(&feature_map.view(), (2, 2), (2, 2))?;
```

**Properties**:
- Smoother than max pooling
- All values contribute
- Differentiable

**When to use**: When you want smoother downsampling, before final classification layer.

### Adaptive Pooling

Pools to a fixed output size regardless of input size.

```rust
let input = Array2::ones((8, 8));

// Always outputs 4x4, regardless of input size
let output = adaptive_avg_pool2d(&input.view(), (4, 4))?;
```

**When to use**: When input sizes vary, or for multi-scale feature extraction.

### Global Pooling

Reduces entire spatial dimensions to single values.

```rust
// Global average: entire feature map → single value
let avg = global_avg_pool(&feature_map.view())?;

// Global max: entire feature map → single value
let max = global_max_pool(&feature_map.view())?;
```

**When to use**:
- Classifier head (replace fully-connected layers)
- Reduce parameters in CNNs
- Feature extraction

---

## Normalization

Techniques to stabilize training and improve convergence.

### Batch Normalization

Normalizes across the batch dimension.

**Formula**: `y = (x - μ_batch) / √(σ²_batch + ε) * γ + β`

```rust
use numrs2::nn::normalization::*;
use scirs2_core::ndarray::{Array1, Array2};

let x = Array2::from_shape_fn((4, 10), |(i, j)| (i + j) as f64);

// Learnable parameters (typically learned during training)
let gamma = Array1::ones(10);   // Scale
let beta = Array1::zeros(10);   // Shift
let epsilon = 1e-5;

let normalized = batch_norm_1d(&x.view(), &gamma.view(), &beta.view(), epsilon)?;
```

**Properties**:
- Stabilizes training
- Allows higher learning rates
- Acts as regularizer
- Different behavior in training vs inference

**When to use**: After convolutional or fully-connected layers in CNNs.

### Layer Normalization

Normalizes across features for each sample independently.

**Formula**: `y = (x - μ_features) / √(σ²_features + ε) * γ + β`

```rust
let normalized = layer_norm(&x.view(), &gamma.view(), &beta.view(), epsilon)?;
```

**Properties**:
- Independent of batch size
- Same behavior in training and inference
- Used in transformers (BERT, GPT)

**When to use**: RNNs, transformers, or small batch sizes.

### RMS Normalization

Simplified normalization without mean subtraction.

**Formula**: `y = x / √(mean(x²) + ε) * γ`

```rust
let gamma = Array1::ones(10);
let normalized = rms_norm(&x.view(), &gamma.view(), epsilon)?;
```

**Properties**:
- Faster than layer norm (no mean calculation)
- Used in modern LLMs (LLaMA, PaLM)

**When to use**: Large language models where speed matters.

### Dropout

Randomly zeros elements during training for regularization.

```rust
use numrs2::nn::normalization::*;

// Training mode: apply dropout
let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let output = dropout(&x.view(), 0.5, true)?;
// ~50% of elements are zero, others scaled by 2.0

// Inference mode: no dropout
let output = dropout(&x.view(), 0.5, false)?;
// output == x (no change)
```

**Parameters**:
- `p`: Probability of dropping (0.0 to 1.0)
- `training`: Enable dropout (true) or pass-through (false)

**When to use**: Prevent overfitting, especially in large networks.

### Spatial Dropout

Drops entire feature maps/channels instead of individual elements.

```rust
let x = Array2::ones((4, 10));  // batch=4, channels=10
let output = spatial_dropout(&x.view(), 0.2, true)?;
// Entire channels are dropped
```

**When to use**: Convolutional layers (preserves spatial correlation).

---

## Attention Mechanisms

Core component of transformer models.

### Scaled Dot-Product Attention

**Formula**: `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`

```rust
use numrs2::nn::attention::*;
use scirs2_core::ndarray::Array2;

let seq_len = 10;
let d_k = 64;

let query = Array2::ones((seq_len, d_k));
let key = Array2::ones((seq_len, d_k));
let value = Array2::ones((seq_len, d_k));

// Basic attention
let output = scaled_dot_product_attention(
    &query.view(),
    &key.view(),
    &value.view(),
    None,  // no mask
)?;

// With attention mask (prevent attending to certain positions)
let mask = Array2::ones((seq_len, seq_len));  // 1 = attend, 0 = mask
let output = scaled_dot_product_attention(
    &query.view(),
    &key.view(),
    &value.view(),
    Some(&mask.view()),
)?;
```

**Components**:
- **Query (Q)**: What we're looking for
- **Key (K)**: What each position has
- **Value (V)**: Actual content to retrieve
- **Scaling (√d_k)**: Prevents softmax saturation

### Self-Attention

Query, key, and value all come from the same input.

```rust
let x = Array2::ones((10, 512));  // seq_len=10, d_model=512

// Projection matrices (typically learned)
let w_q = Array2::from_shape_fn((512, 64), |(i, j)| 0.1);
let w_k = Array2::from_shape_fn((512, 64), |(i, j)| 0.1);
let w_v = Array2::from_shape_fn((512, 64), |(i, j)| 0.1);

let output = self_attention(
    &x.view(),
    &w_q.view(),
    &w_k.view(),
    &w_v.view(),
)?;
```

### Embeddings

Convert discrete tokens to dense vectors.

```rust
use numrs2::nn::attention::*;

let vocab_size = 10000;
let embedding_dim = 512;
let embedding_matrix = Array2::ones((vocab_size, embedding_dim));

let indices = vec![1, 42, 100, 256];  // Token IDs
let embeddings = embedding(&indices, &embedding_matrix.view())?;
// Output: (4, 512) - one embedding per token
```

### Positional Encoding

Add position information to embeddings.

```rust
// Sinusoidal positional encoding
let seq_len = 100;
let d_model = 512;
let pos_encoding = positional_encoding::<f64>(seq_len, d_model)?;

// Add to embeddings
let embeddings_with_pos = add_positional_encoding(&embeddings.view())?;
```

**Why needed**: Transformers have no inherent notion of position.

### Embedding Bag

Aggregate multiple embeddings.

```rust
let indices = vec![1, 2, 3, 4, 5];

// Sum aggregation
let sum_emb = embedding_bag(&indices, &embedding_matrix.view(), "sum")?;

// Mean aggregation
let mean_emb = embedding_bag(&indices, &embedding_matrix.view(), "mean")?;

// Max aggregation
let max_emb = embedding_bag(&indices, &embedding_matrix.view(), "max")?;
```

**When to use**: Bag-of-words models, document embeddings.

---

## Loss Functions

### Regression Losses

#### Mean Squared Error (MSE)

**Formula**: `L = (1/n) Σ (y_true - y_pred)²`

```rust
use numrs2::nn::loss::*;
use scirs2_core::ndarray::array;

let y_true = array![1.0, 2.0, 3.0];
let y_pred = array![1.1, 2.1, 2.9];

let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
// loss ≈ 0.0067
```

**Properties**:
- Sensitive to outliers (squared error)
- Smooth gradients
- Assumes Gaussian noise

**When to use**: Regression when data is Gaussian, no outliers.

#### Mean Absolute Error (MAE)

**Formula**: `L = (1/n) Σ |y_true - y_pred|`

```rust
let loss = mae_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
```

**Properties**:
- Robust to outliers
- Constant gradient magnitude
- Assumes Laplacian noise

**When to use**: Regression with outliers.

#### Huber Loss

**Formula**:
- For |x| ≤ δ: `L = 0.5 * x²`
- For |x| > δ: `L = δ * (|x| - 0.5 * δ)`

```rust
let delta = 1.0;
let loss = huber_loss(&y_true.view(), &y_pred.view(), delta, ReductionMode::Mean)?;
```

**Properties**:
- Combines MSE and MAE
- Smooth near zero, linear for outliers
- Robust to outliers while maintaining smoothness

**When to use**: Regression with some outliers, when you want smooth gradients.

### Classification Losses

#### Binary Cross-Entropy

**Formula**: `L = -(1/n) Σ [y*log(p) + (1-y)*log(1-p)]`

```rust
let y_true = array![1.0, 0.0, 1.0];  // Binary labels
let y_pred = array![0.9, 0.1, 0.8];  // Probabilities

let loss = binary_cross_entropy(
    &y_true.view(),
    &y_pred.view(),
    ReductionMode::Mean
)?;
```

**When to use**: Binary classification with probabilities.

#### BCE with Logits

More numerically stable variant that takes logits (pre-sigmoid).

```rust
let logits = array![2.0, -2.0, 1.5];  // Raw outputs
let loss = bce_with_logits(&y_true.view(), &logits.view(), ReductionMode::Mean)?;
```

**When to use**: Binary classification (preferred over BCE).

#### Categorical Cross-Entropy

**Formula**: `L = -(1/n) Σ Σ_c y_true_c * log(y_pred_c)`

```rust
use scirs2_core::ndarray::Array2;

// One-hot encoded labels
let y_true = Array2::from_shape_vec((2, 3), vec![
    1.0, 0.0, 0.0,  // Sample 0: class 0
    0.0, 1.0, 0.0,  // Sample 1: class 1
])?;

// Predicted probabilities (after softmax)
let y_pred = Array2::from_shape_vec((2, 3), vec![
    0.7, 0.2, 0.1,
    0.1, 0.8, 0.1,
])?;

let loss = categorical_cross_entropy(
    &y_true.view(),
    &y_pred.view(),
    ReductionMode::Mean
)?;
```

**When to use**: Multi-class classification with one-hot labels.

#### Sparse Categorical Cross-Entropy

Optimized for integer labels instead of one-hot vectors.

```rust
let y_true = vec![0, 1, 2];  // Class indices
let y_pred = Array2::from_shape_vec((3, 3), vec![
    0.7, 0.2, 0.1,
    0.1, 0.8, 0.1,
    0.1, 0.1, 0.8,
])?;

let loss = sparse_categorical_cross_entropy(
    &y_true,
    &y_pred.view(),
    ReductionMode::Mean
)?;
```

**When to use**: Multi-class classification (more efficient than categorical).

#### Negative Log-Likelihood (NLL)

For use with log-softmax outputs.

```rust
let log_probs = log_softmax_2d(&logits.view(), 1)?;
let loss = nll_loss(&y_true_indices, &log_probs.view(), ReductionMode::Mean)?;
```

**When to use**: With log-softmax activation (common in PyTorch-style code).

#### Focal Loss

**Formula**: `FL(p_t) = -α(1 - p_t)^γ log(p_t)`

Addresses class imbalance by down-weighting easy examples.

```rust
let alpha = 0.25;  // Class weighting
let gamma = 2.0;   // Focusing parameter

let loss = focal_loss(
    &y_true.view(),
    &y_pred.view(),
    alpha,
    gamma,
    ReductionMode::Mean
)?;
```

**Parameters**:
- **α**: Balances positive/negative examples
- **γ**: Focuses on hard examples (γ=0 → standard CE)

**When to use**: Object detection (RetinaNet), heavily imbalanced datasets.

### Distance Losses

#### KL Divergence

**Formula**: `KL(P||Q) = Σ P(x) * log(P(x) / Q(x))`

Measures how one probability distribution differs from another.

```rust
let p = array![0.4, 0.3, 0.3];  // True distribution
let q = array![0.5, 0.3, 0.2];  // Approximate distribution

let kl = kl_div_loss(&p.view(), &q.view(), ReductionMode::Mean)?;
```

**When to use**: Distillation, VAE regularization.

#### Hinge Loss

**Formula**: `L = max(0, 1 - y_true * y_pred)`

```rust
let y_true = array![1.0, -1.0, 1.0];   // Labels: +1 or -1
let y_pred = array![0.8, -0.9, 0.7];   // Scores (not probabilities)

let loss = hinge_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
```

**When to use**: SVM classification, margin-based learning.

#### Cosine Embedding Loss

Measures cosine similarity between embeddings.

```rust
let x1 = array![1.0, 2.0, 3.0];
let x2 = array![1.1, 2.1, 3.1];
let y = 1.0;      // 1 = similar, -1 = dissimilar
let margin = 0.5;

let loss = cosine_embedding_loss(&x1.view(), &x2.view(), y, margin)?;
```

**When to use**: Siamese networks, metric learning.

### Reduction Modes

All loss functions support three reduction modes:

```rust
// No reduction: return per-element loss
let losses = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::None)?;

// Mean: average over all elements
let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;

// Sum: sum over all elements
let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Sum)?;
```

---

## SIMD Optimization

### Platform Detection

Check SIMD capabilities at runtime:

```rust
use numrs2::nn::simd_ops::*;

// Get detailed SIMD information
println!("{}", get_simd_info());

// Check if SIMD is available
if is_simd_available() {
    println!("SIMD acceleration enabled");
}

// Get recommended batch size for optimal SIMD usage
let batch_size = recommended_batch_size();
```

### SIMD Operations

All major operations have SIMD-optimized variants:

```rust
use numrs2::nn::simd_ops::*;
use scirs2_core::ndarray::Array1;

let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

// Activations (4-8x faster)
let relu = simd_relu_f32(&x.view());
let sigmoid = simd_sigmoid_f32(&x.view());
let tanh = simd_tanh_f32(&x.view());
let gelu = simd_gelu_f32(&x.view());
let swish = simd_swish_f32(&x.view());

// Element-wise operations (8-16x faster)
let y = Array1::from_vec(vec![0.5f32, 1.0, 1.5, 2.0]);
let sum = simd_add_f32(&x.view(), &y.view());
let product = simd_mul_f32(&x.view(), &y.view());

// Reductions (4-8x faster)
let total = simd_sum_f32(&x.view());
let average = simd_mean_f32(&x.view());
let maximum = simd_max_f32(&x.view());

// Matrix operations (10-30x faster for small matrices)
let a = Array2::ones((4, 8));
let b = Array2::ones((8, 4));
let c = simd_matmul_f32(&a.view(), &b.view())?;
```

### Performance Comparison

Typical speedups on modern CPUs:

| Operation        | Scalar | SIMD (AVX2) | SIMD (AVX512) | SIMD (NEON) |
|------------------|--------|-------------|---------------|-------------|
| ReLU             | 1x     | 6x          | 12x           | 4x          |
| Sigmoid          | 1x     | 4x          | 8x            | 3x          |
| GELU             | 1x     | 3x          | 6x            | 2.5x        |
| Element-wise add | 1x     | 8x          | 16x           | 4x          |
| Matrix multiply  | 1x     | 15x         | 30x           | 8x          |

### f32 vs f64

Use f32 when possible for 2x better SIMD performance:

```rust
// f64: 4 elements per AVX2 instruction
let x_f64 = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
let y_f64 = simd_relu_f64(&x_f64.view());

// f32: 8 elements per AVX2 instruction (2x throughput)
let x_f32 = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
let y_f32 = simd_relu_f32(&x_f32.view());
```

**Recommendation**: Use f32 for forward pass, f64 for numerical stability in sensitive operations.

---

## Performance Tips

### 1. Use Appropriate Data Types

```rust
// ✅ GOOD: f32 for better SIMD performance
let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
let y = simd_relu_f32(&x.view());

// ⚠️ OK: f64 when precision is critical
let x = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
let y = relu(&x.view())?;
```

### 2. Batch Operations

Process multiple samples at once:

```rust
// ❌ BAD: Process one at a time
for i in 0..1000 {
    let sample = samples.row(i);
    let output = relu(&sample)?;
    // ...
}

// ✅ GOOD: Process entire batch
let output = relu_2d(&samples.view())?;
```

### 3. Minimize Allocations

```rust
// ❌ BAD: Allocate every iteration
for epoch in 0..100 {
    let output = Array2::zeros((batch_size, hidden_size));
    // ...
}

// ✅ GOOD: Reuse allocation
let mut output = Array2::zeros((batch_size, hidden_size));
for epoch in 0..100 {
    // Reuse output
    // ...
}
```

### 4. Use In-Place Operations

```rust
// ❌ BAD: Allocate new array
let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
let y = relu(&x.view())?;

// ✅ GOOD: Modify in-place
let mut x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
relu_inplace(&mut x);
```

### 5. Choose Right Loss Function

```rust
// ❌ SLOWER: Categorical with one-hot encoding
let y_onehot = Array2::from_shape_vec((n, num_classes), one_hot_data)?;
let loss = categorical_cross_entropy(&y_onehot.view(), &probs.view(), ReductionMode::Mean)?;

// ✅ FASTER: Sparse categorical with indices
let y_indices = vec![0, 1, 2, 0, 1];
let loss = sparse_categorical_cross_entropy(&y_indices, &probs.view(), ReductionMode::Mean)?;
```

### 6. Profile Your Code

Use NumRS2's benchmarking tools:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_activation(c: &mut Criterion) {
    let x = Array1::from_vec(vec![1.0f32; 1024]);

    c.bench_function("relu_scalar", |b| {
        b.iter(|| relu(&black_box(x.view())))
    });

    c.bench_function("relu_simd", |b| {
        b.iter(|| simd_relu_f32(&black_box(x.view())))
    });
}
```

### 7. Optimize Memory Layout

```rust
// ✅ GOOD: Contiguous memory layout
let x = Array2::from_shape_vec((n, m), data)?;

// ⚠️ SLOWER: Non-contiguous (after transpose)
let x_t = x.t();
```

---

## Training Patterns

### Basic Training Loop

```rust
use numrs2::nn::*;
use scirs2_core::ndarray::{Array1, Array2};

fn train_epoch(
    data: &Array2<f64>,
    labels: &[usize],
    weights: &mut Array2<f64>,
    learning_rate: f64,
) -> NnResult<f64> {
    let batch_size = data.nrows();

    // Forward pass
    let logits = simd_matmul_f64(&data.view(), &weights.view())?;
    let probs = softmax_2d(&logits.view(), 1)?;

    // Compute loss
    let loss = sparse_categorical_cross_entropy(
        labels,
        &probs.view(),
        ReductionMode::Mean,
    )?;

    // Backward pass (gradient computation)
    // ... compute gradients ...

    // Update weights
    // weights -= learning_rate * gradients

    Ok(loss)
}
```

### Multi-Layer Network

```rust
struct Network {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    gamma1: Array1<f64>,
    beta1: Array1<f64>,
}

impl Network {
    fn forward(&self, x: &Array2<f64>, training: bool) -> NnResult<Array2<f64>> {
        // Layer 1: Linear + ReLU + BatchNorm + Dropout
        let h1 = simd_matmul_f64(&x.view(), &self.w1.view())?;
        let h1 = relu_2d(&h1.view())?;
        let h1 = batch_norm_1d(
            &h1.view(),
            &self.gamma1.view(),
            &self.beta1.view(),
            1e-5
        )?;
        let h1 = dropout_2d(&h1.view(), 0.5, training)?;

        // Layer 2: Linear + Softmax
        let h2 = simd_matmul_f64(&h1.view(), &self.w2.view())?;
        let output = softmax_2d(&h2.view(), 1)?;

        Ok(output)
    }
}
```

### Validation Loop

```rust
fn evaluate(
    network: &Network,
    val_data: &Array2<f64>,
    val_labels: &[usize],
) -> NnResult<(f64, f64)> {
    // Forward pass in inference mode
    let probs = network.forward(val_data, false)?;

    // Compute loss
    let loss = sparse_categorical_cross_entropy(
        val_labels,
        &probs.view(),
        ReductionMode::Mean,
    )?;

    // Compute accuracy
    let mut correct = 0;
    for (i, &label) in val_labels.iter().enumerate() {
        let pred = probs.row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if pred == label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / val_labels.len() as f64;

    Ok((loss, accuracy))
}
```

---

## Error Handling

All operations return `NnResult<T>` for comprehensive error handling:

```rust
use numrs2::nn::*;
use numrs2::error::NumRs2Error;

fn process_data(x: &Array1<f64>) -> NnResult<Array1<f64>> {
    // Dimension validation
    if x.is_empty() {
        return Err(NumRs2Error::DimensionMismatch(
            "Input cannot be empty".to_string()
        ));
    }

    // Operation with error propagation
    let y = relu(&x.view())?;
    let z = softmax(&y.view())?;

    Ok(z)
}
```

### Common Error Types

```rust
match result {
    Err(NumRs2Error::DimensionMismatch(msg)) => {
        // Shape incompatibility
        eprintln!("Dimension error: {}", msg);
    }
    Err(NumRs2Error::InvalidOperation(msg)) => {
        // Invalid parameters (e.g., negative probability)
        eprintln!("Invalid operation: {}", msg);
    }
    Err(NumRs2Error::ConversionError(msg)) => {
        // Type conversion failure
        eprintln!("Conversion error: {}", msg);
    }
    Err(NumRs2Error::IndexOutOfBounds(msg)) => {
        // Index out of range
        eprintln!("Index error: {}", msg);
    }
    Ok(value) => {
        // Success
    }
}
```

---

## Examples

### Complete Neural Network Example

See `examples/neural_network_basics.rs` for a comprehensive demonstration including:

- Activation functions showcase
- Normalization and dropout
- Convolution and pooling
- Loss function comparisons
- Complete feedforward network
- Mini training loop structure

Run the example:

```bash
cargo run --example neural_network_basics
```

### Simple Classification Network

```rust
use numrs2::nn::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::*;

fn main() -> NnResult<()> {
    // Network architecture: 4 inputs -> 8 hidden -> 3 outputs

    // Generate synthetic data
    let mut rng = thread_rng();
    let input = Array2::from_shape_fn((100, 4), |_| rng.gen::<f64>());
    let labels = (0..100).map(|i| i % 3).collect::<Vec<_>>();

    // Initialize weights
    let w1 = Array2::from_shape_fn((4, 8), |_| rng.gen::<f64>() * 0.1);
    let w2 = Array2::from_shape_fn((8, 3), |_| rng.gen::<f64>() * 0.1);

    // Forward pass
    let h1 = simd_matmul_f64(&input.view(), &w1.view())?;
    let h1 = relu_2d(&h1.view())?;

    let logits = simd_matmul_f64(&h1.view(), &w2.view())?;
    let probs = softmax_2d(&logits.view(), 1)?;

    // Compute loss
    let loss = sparse_categorical_cross_entropy(
        &labels,
        &probs.view(),
        ReductionMode::Mean,
    )?;

    println!("Loss: {:.4}", loss);

    Ok(())
}
```

---

## API Reference

### Module Structure

- **`activation`**: Activation functions
  - `relu`, `leaky_relu`, `elu`, `selu`
  - `sigmoid`, `tanh`, `swish`, `mish`, `gelu`
  - `softmax`, `log_softmax`, `softplus`
  - 2D variants: `relu_2d`, `softmax_2d`, etc.
  - In-place: `relu_inplace`

- **`attention`**: Attention and embeddings
  - `scaled_dot_product_attention`
  - `self_attention`
  - `embedding`, `embedding_bag`
  - `positional_encoding`, `add_positional_encoding`

- **`conv`**: Convolution operations
  - `conv1d`, `conv2d`
  - `conv2d_with_padding`
  - `depthwise_conv2d`

- **`loss`**: Loss functions
  - Regression: `mse_loss`, `mae_loss`, `huber_loss`
  - Classification: `binary_cross_entropy`, `categorical_cross_entropy`, `sparse_categorical_cross_entropy`
  - Advanced: `focal_loss`, `kl_div_loss`, `hinge_loss`, `cosine_embedding_loss`

- **`normalization`**: Normalization and regularization
  - `batch_norm_1d`, `layer_norm`, `rms_norm`
  - `dropout`, `dropout_2d`, `spatial_dropout`

- **`pooling`**: Pooling operations
  - `max_pool2d`, `avg_pool2d`
  - `adaptive_avg_pool2d`
  - `global_avg_pool`, `global_max_pool`

- **`simd_ops`**: SIMD-optimized operations
  - Detection: `detect_simd_capabilities`, `get_simd_info`, `is_simd_available`
  - Activations: `simd_relu_f32`, `simd_sigmoid_f32`, `simd_gelu_f32`, etc.
  - Operations: `simd_add_f32`, `simd_mul_f32`, `simd_matmul_f32`, etc.
  - Reductions: `simd_sum_f32`, `simd_mean_f32`, `simd_max_f32`

### Common Types

```rust
// Result type
pub type NnResult<T> = Result<T, NumRs2Error>;

// Enumerations
pub enum ReductionMode { None, Mean, Sum }
pub enum PaddingMode { Valid, Same, Full, Explicit(usize) }
pub enum DataFormat { NCHW, NHWC }
```

---

## Further Reading

- **Examples**: `examples/neural_network_basics.rs`
- **Benchmarks**: `bench/nn_benchmarks.rs`
- **Tests**: `tests/nn/`
- **SciRS2 Policy**: `SCIRS2_INTEGRATION_POLICY.md`
- **Architecture**: `docs/ARCHITECTURE.md`

---

## Version History

- **v0.2.0**: Initial neural network module release
  - Complete activation functions
  - Convolution and pooling operations
  - Attention mechanisms
  - Comprehensive loss functions
  - SIMD optimization
  - Pure Rust implementation via OxiBLAS

---

*For questions, issues, or contributions, please visit the NumRS2 repository.*

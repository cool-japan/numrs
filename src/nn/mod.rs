//! Neural Network Primitives for NumRS2
//!
//! This module provides deep learning primitives optimized for numerical computing.
//! All operations are SIMD-optimized using SciRS2-Core abstractions.
//!
//! # Overview
//!
//! The `nn` module provides a comprehensive set of neural network building blocks that can be
//! used to construct and train deep learning models. All operations are designed for performance
//! with SIMD acceleration while maintaining numerical stability and ease of use.
//!
//! # Features
//!
//! ## Activation Functions
//!
//! - **ReLU**: Rectified Linear Unit and variants (Leaky ReLU, ELU, SELU)
//! - **Sigmoid**: Logistic sigmoid function
//! - **Tanh**: Hyperbolic tangent
//! - **GELU**: Gaussian Error Linear Unit (used in transformers)
//! - **Swish/SiLU**: Self-gated activation function
//! - **Mish**: Smooth non-monotonic activation
//! - **Softmax**: Normalized exponential for probability distributions
//! - **Log-Softmax**: Numerically stable log(softmax(x))
//!
//! ## Convolution Operations
//!
//! - **1D Convolution**: For sequence/time-series data
//! - **2D Convolution**: For image/spatial data
//! - **Padding Support**: Valid, same, and explicit padding modes
//! - **Depthwise Separable**: Efficient mobile convolutions
//!
//! ## Pooling Layers
//!
//! - **Max Pooling**: Extract maximum values from windows
//! - **Average Pooling**: Compute mean over windows
//! - **Adaptive Pooling**: Output fixed size regardless of input
//! - **Global Pooling**: Reduce spatial dimensions to single values
//!
//! ## Normalization
//!
//! - **Batch Normalization**: Normalize across batch dimension
//! - **Layer Normalization**: Normalize across features (used in transformers)
//! - **RMS Normalization**: Simplified normalization without mean subtraction
//! - **Dropout**: Regularization through random zeroing (standard and spatial)
//!
//! ## Attention Mechanisms
//!
//! - **Scaled Dot-Product Attention**: Core attention operation
//! - **Self-Attention**: Query, key, value from same source
//! - **Positional Encoding**: Sinusoidal position embeddings
//! - **Embedding Operations**: Token and positional embeddings
//!
//! ## Loss Functions
//!
//! ### Regression Losses
//! - **MSE**: Mean Squared Error
//! - **MAE**: Mean Absolute Error
//! - **Huber Loss**: Smooth L1 loss for robustness to outliers
//!
//! ### Classification Losses
//! - **Cross-Entropy**: Categorical and sparse versions
//! - **Binary Cross-Entropy**: With numerically stable logits variant
//! - **Negative Log-Likelihood**: For use with log-softmax
//! - **Focal Loss**: Address class imbalance in object detection
//! - **Hinge Loss**: SVM-style margin loss
//!
//! ### Advanced Losses
//! - **KL Divergence**: Distribution distance measurement
//! - **Cosine Embedding Loss**: Similarity-based loss
//!
//! ## SIMD Optimization
//!
//! All performance-critical operations leverage SIMD instructions:
//! - **Platform Detection**: Automatic CPU capability detection
//! - **AVX2/AVX512**: x86_64 acceleration
//! - **NEON**: ARM acceleration
//! - **Automatic Fallback**: Scalar implementation when SIMD unavailable
//! - **4-8x Speedup**: Typical performance improvement for activations
//! - **8-16x Speedup**: Element-wise operations with advanced SIMD
//!
//! # Architecture and Design
//!
//! ## SCIRS2 Integration Policy
//!
//! This module follows NumRS2's SCIRS2 integration policy:
//! - Uses `scirs2_core::ndarray` for all array operations
//! - Uses `scirs2_core::simd_ops::SimdUnifiedOps` for SIMD acceleration
//! - Uses `scirs2_core::random` for randomness (dropout, initialization)
//! - Uses `scirs2_linalg` for linear algebra via OxiBLAS (pure Rust)
//! - **Pure Rust**: No C/C++ dependencies, full cross-platform compatibility
//!
//! ## Error Handling
//!
//! All operations return `NnResult<T>` which is `Result<T, NumRs2Error>`:
//! - Dimension mismatches are caught at runtime
//! - Invalid parameters (negative probabilities, etc.) are validated
//! - Numerical issues (NaN, Inf) are detected and reported
//!
//! ## Memory Efficiency
//!
//! - In-place operations available where possible (e.g., `relu_inplace`)
//! - Memory pooling recommended for training loops
//! - Batch operations avoid temporary allocations
//!
//! # Quick Start
//!
//! ## Basic Activation Functions
//!
//! ```rust,ignore
//! use numrs2::nn::*;
//! use scirs2_core::ndarray::array;
//!
//! // Apply ReLU to a 1D array
//! let x = array![-1.0, 0.0, 1.0, 2.0];
//! let y = relu(&x.view())?;
//! // y = [0.0, 0.0, 1.0, 2.0]
//!
//! // Softmax for classification
//! let logits = array![1.0, 2.0, 3.0];
//! let probs = softmax(&logits.view())?;
//! // probs sums to 1.0
//! ```
//!
//! ## Building a Simple Network
//!
//! ```rust,ignore
//! use numrs2::nn::*;
//! use scirs2_core::ndarray::{Array2, Array1};
//!
//! // Network: Input(10) -> Hidden(20) -> Output(3)
//! let input = Array2::ones((4, 10)); // batch_size=4, features=10
//!
//! // Hidden layer
//! let weights1 = Array2::from_shape_fn((10, 20), |(i, j)| 0.1);
//! let hidden = simd_matmul_f64(&input.view(), &weights1.view())?;
//! let hidden = relu_2d(&hidden.view())?;
//!
//! // Batch normalization
//! let gamma = Array1::ones(20);
//! let beta = Array1::zeros(20);
//! let hidden = batch_norm_1d(&hidden.view(), &gamma.view(), &beta.view(), 1e-5)?;
//!
//! // Output layer
//! let weights2 = Array2::from_shape_fn((20, 3), |(i, j)| 0.1);
//! let output = simd_matmul_f64(&hidden.view(), &weights2.view())?;
//! let probs = softmax_2d(&output.view(), 1)?;
//! ```
//!
//! ## Computing Loss
//!
//! ```rust,ignore
//! use numrs2::nn::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Regression with MSE
//! let y_true = array![1.0, 2.0, 3.0];
//! let y_pred = array![1.1, 2.1, 2.9];
//! let loss = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
//!
//! // Classification with cross-entropy
//! let y_true = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
//! let y_pred = Array2::from_shape_vec((2, 3), vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1])?;
//! let loss = categorical_cross_entropy(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
//! ```
//!
//! # Performance Tips
//!
//! ## SIMD Acceleration
//!
//! ```rust,ignore
//! // Check SIMD capabilities
//! println!("{}", get_simd_info());
//!
//! // Use f32 for better SIMD performance (2x data per instruction vs f64)
//! let x_f32 = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
//! let y = simd_relu_f32(&x_f32.view()); // 4-8x faster than scalar
//! ```
//!
//! ## Batch Processing
//!
//! ```rust,ignore
//! // Process multiple samples at once for better cache utilization
//! let batch = Array2::ones((32, 784)); // batch_size=32
//! let output = relu_2d(&batch.view())?;
//! ```
//!
//! ## Memory Management
//!
//! ```rust,ignore
//! // Use in-place operations when possible
//! let mut x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
//! relu_inplace(&mut x);
//!
//! // Reuse allocations in training loops
//! let mut output = Array2::zeros((batch_size, hidden_size));
//! for epoch in 0..num_epochs {
//!     // Reuse output allocation
//!     // ...
//! }
//! ```
//!
//! # Advanced Usage
//!
//! ## Attention Mechanisms
//!
//! ```rust,ignore
//! use numrs2::nn::*;
//!
//! // Scaled dot-product attention
//! let seq_len = 10;
//! let d_k = 64;
//! let query = Array2::ones((seq_len, d_k));
//! let key = Array2::ones((seq_len, d_k));
//! let value = Array2::ones((seq_len, d_k));
//!
//! let output = scaled_dot_product_attention(
//!     &query.view(),
//!     &key.view(),
//!     &value.view(),
//!     None, // no mask
//! )?;
//! ```
//!
//! ## Custom Training Loops
//!
//! See `examples/neural_network_basics.rs` for a complete example of:
//! - Building feedforward networks
//! - Forward pass implementation
//! - Loss computation
//! - Mini-batch training structure
//!
//! # Module Organization
//!
//! - [`activation`]: Activation functions (ReLU, GELU, Sigmoid, etc.)
//! - [`attention`]: Attention mechanisms and embeddings
//! - [`conv`]: Convolution operations (1D, 2D, 3D)
//! - [`loss`]: Loss functions for training
//! - [`normalization`]: Batch/layer normalization and dropout
//! - [`pooling`]: Pooling operations (max, average, adaptive)
//! - [`simd_ops`]: SIMD-optimized kernels
//!
//! # See Also
//!
//! - **Examples**: `examples/neural_network_basics.rs`
//! - **Benchmarks**: `bench/nn_benchmarks.rs`
//! - **Documentation**: `docs/NN_GUIDE.md`
//! - **SciRS2 Policy**: `SCIRS2_INTEGRATION_POLICY.md`

pub mod activation;
pub mod attention;
pub mod conv;
pub mod loss;
pub mod normalization;
pub mod pooling;
pub mod simd_ops;

// Re-export commonly used items
pub use activation::*;
pub use attention::*;
pub use conv::*;
pub use loss::*;
pub use normalization::*;
pub use pooling::*;
pub use simd_ops::{
    detect_simd_capabilities, get_simd_info, is_simd_available, recommended_batch_size,
    simd_add_f32, simd_div_f32, simd_dot_f32, simd_elu_f32, simd_gelu_f32, simd_gelu_f64,
    simd_leaky_relu_f32, simd_matmul_f32, simd_matmul_f64, simd_max_f32, simd_mean_f32,
    simd_min_f32, simd_mish_f32, simd_mul_f32, simd_norm_f32, simd_relu_2d_f32, simd_relu_2d_f64,
    simd_relu_f32, simd_relu_f64, simd_selu_f32, simd_sigmoid_f32, simd_sigmoid_f64, simd_sub_f32,
    simd_sum_f32, simd_swish_f32, simd_swish_f64, simd_tanh_f32, simd_tanh_f64,
};

use crate::error::NumRs2Error;

/// Result type for neural network operations
pub type NnResult<T> = Result<T, NumRs2Error>;

/// Padding mode for convolution and pooling operations
///
/// Controls how padding is applied to convolution operations.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::nn::PaddingMode;
///
/// // No padding - output smaller than input
/// let mode = PaddingMode::Valid;
///
/// // Same padding - preserves input size (for stride=1)
/// let mode = PaddingMode::Same;
///
/// // Custom padding amount
/// let mode = PaddingMode::Explicit(2);  // Pad by 2 on all sides
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingMode {
    /// No padding (valid convolution)
    ///
    /// Output size = (input_size - kernel_size) / stride + 1
    #[default]
    Valid,
    /// Same padding (output size = input size for stride 1)
    ///
    /// Automatically calculates padding to preserve input dimensions
    Same,
    /// Full padding (output size = input size + kernel size - 1)
    ///
    /// Maximum padding, ensures all input elements are covered
    Full,
    /// Explicit padding (specify padding amount)
    ///
    /// Custom padding value applied to all sides
    Explicit(usize),
}

/// Reduction mode for loss functions
///
/// Determines how individual losses are combined into a single value.
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::nn::{mse_loss, ReductionMode};
/// use scirs2_core::ndarray::array;
///
/// let y_true = array![1.0, 2.0, 3.0];
/// let y_pred = array![1.1, 2.1, 3.1];
///
/// // Mean: average of losses (typical for training)
/// let loss_mean = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Mean)?;
///
/// // Sum: sum of losses (useful for some algorithms)
/// let loss_sum = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::Sum)?;
///
/// // None: per-element losses (for custom aggregation)
/// let loss_none = mse_loss(&y_true.view(), &y_pred.view(), ReductionMode::None)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReductionMode {
    /// No reduction, return loss for each element
    ///
    /// Returns the first element's loss when scalar is expected.
    /// Use this when you need per-sample losses for custom processing.
    None,
    /// Mean reduction over all elements
    ///
    /// Returns the average loss: `sum(losses) / count`
    /// This is the most common mode for training.
    #[default]
    Mean,
    /// Sum reduction over all elements
    ///
    /// Returns the total loss: `sum(losses)`
    /// Useful when losses need to be normalized externally.
    Sum,
}

/// Data format for tensors
///
/// Specifies the memory layout of multi-dimensional tensors.
///
/// # Channel Ordering
///
/// - **NCHW**: Channels first (N, C, H, W) - PyTorch, Caffe style
/// - **NHWC**: Channels last (N, H, W, C) - TensorFlow style
///
/// where:
/// - N = batch size
/// - C = number of channels
/// - H = height
/// - W = width
///
/// # Performance Considerations
///
/// - NCHW is typically faster on GPUs and with SIMD operations
/// - NHWC can be more cache-friendly for some CPU operations
/// - Choose based on your framework/backend requirements
///
/// # Examples
///
/// ```rust,ignore
/// use numrs2::nn::DataFormat;
///
/// // PyTorch-style (channels first)
/// let format = DataFormat::NCHW;
/// // Tensor shape: (batch=32, channels=3, height=224, width=224)
///
/// // TensorFlow-style (channels last)
/// let format = DataFormat::NHWC;
/// // Tensor shape: (batch=32, height=224, width=224, channels=3)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFormat {
    /// Channels first (N, C, H, W)
    ///
    /// Memory layout: \[batch\]\[channel\]\[height\]\[width\]
    /// Common in PyTorch, Caffe. Better for SIMD operations.
    #[default]
    NCHW,
    /// Channels last (N, H, W, C)
    ///
    /// Memory layout: \[batch\]\[height\]\[width\]\[channel\]
    /// Common in TensorFlow. Can be more cache-friendly.
    NHWC,
}

//! NumRS2 Native Model Format
//!
//! Defines the core model format structure for NumRS2 neural networks.
//! The format supports comprehensive metadata, layer serialization, and
//! optimization state preservation.

use crate::error::NumRs2Error;
use oxicode::{decode_from_slice, encode_to_vec, Decode, Encode};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;

/// Result type for model format operations
pub type FormatResult<T> = Result<T, NumRs2Error>;

/// NumRS2 model format version
pub const MODEL_FORMAT_VERSION: &str = "0.4.0";

/// NumRS2 model file extension
pub const MODEL_EXTENSION: &str = ".numrs2";

/// Model format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFormat {
    /// Format version string
    pub version: String,
    /// Magic number for format identification
    pub magic: [u8; 8],
    /// Compression type
    pub compression: CompressionType,
    /// Endianness (little or big)
    pub endian: String,
}

impl Default for ModelFormat {
    fn default() -> Self {
        Self {
            version: MODEL_FORMAT_VERSION.to_string(),
            magic: *b"NUMRS2\x00\x00",
            compression: CompressionType::Oxicode,
            endian: "little".to_string(),
        }
    }
}

/// Compression type for model data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// Oxicode SIMD-optimized serialization
    Oxicode,
    /// OxiARC ZIP compression
    Zip,
}

/// Complete NumRS2 model structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumRS2Model {
    /// Model format specification
    pub format: ModelFormat,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Layer data
    pub layers: Vec<LayerData>,
    /// Optional optimizer state
    pub optimizer_state: Option<OptimizerState>,
}

impl NumRS2Model {
    /// Creates a new NumRS2 model
    pub fn new(metadata: ModelMetadata, layers: Vec<LayerData>) -> Self {
        Self {
            format: ModelFormat::default(),
            metadata,
            layers,
            optimizer_state: None,
        }
    }

    /// Creates a new model with optimizer state
    pub fn new_with_optimizer(
        metadata: ModelMetadata,
        layers: Vec<LayerData>,
        optimizer_state: OptimizerState,
    ) -> Self {
        Self {
            format: ModelFormat::default(),
            metadata,
            layers,
            optimizer_state: Some(optimizer_state),
        }
    }

    /// Gets the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Gets the total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Gets a layer by index
    pub fn get_layer(&self, index: usize) -> FormatResult<&LayerData> {
        self.layers.get(index).ok_or_else(|| {
            NumRs2Error::IndexOutOfBounds(format!("Layer index {} out of bounds", index))
        })
    }

    /// Gets a layer by name
    pub fn get_layer_by_name(&self, name: &str) -> FormatResult<&LayerData> {
        self.layers
            .iter()
            .find(|layer| layer.name == name)
            .ok_or_else(|| NumRs2Error::ValueError(format!("Layer '{}' not found", name)))
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture type
    pub architecture: String,
    /// Model description
    pub description: Option<String>,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Training information
    pub training_info: Option<TrainingInfo>,
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Author/creator
    pub author: Option<String>,
    /// Custom metadata fields
    pub custom: HashMap<String, String>,
}

impl ModelMetadata {
    /// Creates a new metadata builder
    pub fn builder() -> ModelMetadataBuilder {
        ModelMetadataBuilder::default()
    }
}

/// Builder for ModelMetadata
#[derive(Debug, Default)]
pub struct ModelMetadataBuilder {
    name: Option<String>,
    version: Option<String>,
    architecture: Option<String>,
    description: Option<String>,
    hyperparameters: HashMap<String, String>,
    training_info: Option<TrainingInfo>,
    author: Option<String>,
    custom: HashMap<String, String>,
}

impl ModelMetadataBuilder {
    /// Sets the model name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the model version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Sets the architecture type
    pub fn architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = Some(architecture.into());
        self
    }

    /// Sets the description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a hyperparameter
    pub fn hyperparameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Sets hyperparameters
    pub fn hyperparameters(mut self, hyperparameters: HashMap<String, String>) -> Self {
        self.hyperparameters = hyperparameters;
        self
    }

    /// Sets training information
    pub fn training_info(mut self, training_info: TrainingInfo) -> Self {
        self.training_info = Some(training_info);
        self
    }

    /// Sets the author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Adds a custom field
    pub fn custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Builds the metadata
    pub fn build(self) -> FormatResult<ModelMetadata> {
        let now = chrono::Utc::now().to_rfc3339();

        Ok(ModelMetadata {
            name: self
                .name
                .ok_or_else(|| NumRs2Error::ValueError("Model name is required".to_string()))?,
            version: self.version.unwrap_or_else(|| "1.0.0".to_string()),
            architecture: self.architecture.unwrap_or_else(|| "Unknown".to_string()),
            description: self.description,
            hyperparameters: self.hyperparameters,
            training_info: self.training_info,
            created_at: now.clone(),
            modified_at: now,
            author: self.author,
            custom: self.custom,
        })
    }
}

/// Training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Number of training epochs
    pub epochs: usize,
    /// Training loss history
    pub train_loss: Vec<f64>,
    /// Validation loss history
    pub val_loss: Option<Vec<f64>>,
    /// Training accuracy history
    pub train_acc: Option<Vec<f64>>,
    /// Validation accuracy history
    pub val_acc: Option<Vec<f64>>,
    /// Best validation loss
    pub best_val_loss: Option<f64>,
    /// Best epoch
    pub best_epoch: Option<usize>,
    /// Learning rate schedule
    pub learning_rate_schedule: Option<Vec<f64>>,
    /// Total training time (seconds)
    pub training_time_secs: Option<f64>,
}

/// Layer type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Dense/Fully connected layer
    Dense,
    /// Convolutional layer (1D, 2D, 3D)
    Conv,
    /// Pooling layer
    Pooling,
    /// Normalization layer
    Normalization,
    /// Activation layer
    Activation,
    /// Dropout layer
    Dropout,
    /// Embedding layer
    Embedding,
    /// Attention layer
    Attention,
    /// Transformer encoder layer
    TransformerEncoder,
    /// Transformer decoder layer
    TransformerDecoder,
    /// Recurrent layer (RNN, LSTM, GRU)
    Recurrent,
    /// Custom layer
    Custom,
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    LeakyReLU,
    ELU,
    SELU,
    GELU,
    Swish,
    Mish,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    None,
}

/// Layer data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerData {
    /// Layer name/identifier
    pub name: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Weights (serialized as bytes)
    pub weights: Vec<u8>,
    /// Bias (serialized as bytes, optional)
    pub bias: Option<Vec<u8>>,
    /// Activation function
    pub activation: Option<ActivationType>,
    /// Layer-specific parameters
    pub parameters: HashMap<String, String>,
}

impl LayerData {
    /// Creates a dense layer
    pub fn dense(name: impl Into<String>, weights: Array2<f64>, bias: Option<Array1<f64>>) -> Self {
        let input_shape = vec![weights.shape()[0]];
        let output_shape = vec![weights.shape()[1]];

        let weights_bytes = serde_json::to_vec(&weights).unwrap_or_else(|_| Vec::new());
        let bias_bytes = bias.map(|b| serde_json::to_vec(&b).unwrap_or_else(|_| Vec::new()));

        Self {
            name: name.into(),
            layer_type: LayerType::Dense,
            input_shape,
            output_shape,
            weights: weights_bytes,
            bias: bias_bytes,
            activation: None,
            parameters: HashMap::new(),
        }
    }

    /// Creates a convolutional layer
    pub fn conv(
        name: impl Into<String>,
        weights: Array4<f64>,
        bias: Option<Array1<f64>>,
        stride: usize,
        padding: usize,
    ) -> Self {
        let shape = weights.shape();
        let input_shape = vec![shape[1], shape[2], shape[3]];
        let output_shape = vec![shape[0]];

        let weights_bytes = serde_json::to_vec(&weights).unwrap_or_else(|_| Vec::new());
        let bias_bytes = bias.map(|b| serde_json::to_vec(&b).unwrap_or_else(|_| Vec::new()));

        let mut parameters = HashMap::new();
        parameters.insert("stride".to_string(), stride.to_string());
        parameters.insert("padding".to_string(), padding.to_string());

        Self {
            name: name.into(),
            layer_type: LayerType::Conv,
            input_shape,
            output_shape,
            weights: weights_bytes,
            bias: bias_bytes,
            activation: None,
            parameters,
        }
    }

    /// Creates an attention layer
    pub fn attention(
        name: impl Into<String>,
        w_q: Array2<f64>,
        w_k: Array2<f64>,
        w_v: Array2<f64>,
        w_o: Array2<f64>,
        num_heads: usize,
    ) -> Self {
        let d_model = w_q.shape()[0];
        let input_shape = vec![d_model];
        let output_shape = vec![d_model];

        // Concatenate all weight matrices
        let all_weights = vec![w_q, w_k, w_v, w_o];
        let weights_bytes = serde_json::to_vec(&all_weights).unwrap_or_else(|_| Vec::new());

        let mut parameters = HashMap::new();
        parameters.insert("num_heads".to_string(), num_heads.to_string());
        parameters.insert("d_model".to_string(), d_model.to_string());

        Self {
            name: name.into(),
            layer_type: LayerType::Attention,
            input_shape,
            output_shape,
            weights: weights_bytes,
            bias: None,
            activation: None,
            parameters,
        }
    }

    /// Gets the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        // Estimate based on weight bytes (approximate)
        let weight_params = self.weights.len() / 8; // Assuming f64
        let bias_params = self.bias.as_ref().map(|b| b.len() / 8).unwrap_or(0);
        weight_params + bias_params
    }

    /// Deserializes weights as `Array2<f64>`
    pub fn weights_as_array2(&self) -> FormatResult<Array2<f64>> {
        serde_json::from_slice(&self.weights).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to deserialize weights: {}", e))
        })
    }

    /// Deserializes bias as `Array1<f64>`
    pub fn bias_as_array1(&self) -> FormatResult<Option<Array1<f64>>> {
        match &self.bias {
            Some(b) => serde_json::from_slice(b).map(Some).map_err(|e| {
                NumRs2Error::DeserializationError(format!("Failed to deserialize bias: {}", e))
            }),
            None => Ok(None),
        }
    }
}

/// Optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer name (Adam, SGD, RMSprop, etc.)
    pub optimizer_name: String,
    /// Learning rate
    pub learning_rate: f64,
    /// Current step/iteration
    pub step: usize,
    /// Optimizer parameters (beta1, beta2, etc.)
    pub parameters: HashMap<String, f64>,
    /// First moment estimates (for Adam, etc.)
    pub first_moments: Option<Vec<Vec<u8>>>,
    /// Second moment estimates (for Adam, etc.)
    pub second_moments: Option<Vec<Vec<u8>>>,
    /// Velocity (for SGD with momentum)
    pub velocity: Option<Vec<Vec<u8>>>,
}

impl OptimizerState {
    /// Creates a new optimizer state
    pub fn new(optimizer_name: impl Into<String>, learning_rate: f64) -> Self {
        Self {
            optimizer_name: optimizer_name.into(),
            learning_rate,
            step: 0,
            parameters: HashMap::new(),
            first_moments: None,
            second_moments: None,
            velocity: None,
        }
    }

    /// Creates Adam optimizer state
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("beta1".to_string(), beta1);
        parameters.insert("beta2".to_string(), beta2);
        parameters.insert("epsilon".to_string(), epsilon);

        Self {
            optimizer_name: "Adam".to_string(),
            learning_rate,
            step: 0,
            parameters,
            first_moments: Some(Vec::new()),
            second_moments: Some(Vec::new()),
            velocity: None,
        }
    }

    /// Creates SGD optimizer state with momentum
    pub fn sgd(learning_rate: f64, momentum: f64) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("momentum".to_string(), momentum);

        Self {
            optimizer_name: "SGD".to_string(),
            learning_rate,
            step: 0,
            parameters,
            first_moments: None,
            second_moments: None,
            velocity: Some(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_model_format_default() {
        let format = ModelFormat::default();
        assert_eq!(format.version, MODEL_FORMAT_VERSION);
        assert_eq!(format.magic, *b"NUMRS2\x00\x00");
        assert_eq!(format.compression, CompressionType::Oxicode);
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("Transformer")
            .description("A test model")
            .hyperparameter("hidden_size", "512")
            .hyperparameter("num_layers", "6")
            .author("NumRS2")
            .build();

        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.architecture, "Transformer");
        assert_eq!(metadata.hyperparameters.get("hidden_size").unwrap(), "512");
    }

    #[test]
    fn test_metadata_builder_missing_name() {
        let result = ModelMetadata::builder()
            .version("1.0.0")
            .architecture("Transformer")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dense_layer_creation() {
        let weights = Array2::ones((10, 5));
        let bias = Some(Array1::zeros(5));

        let layer = LayerData::dense("dense1", weights, bias);

        assert_eq!(layer.name, "dense1");
        assert_eq!(layer.layer_type, LayerType::Dense);
        assert_eq!(layer.input_shape, vec![10]);
        assert_eq!(layer.output_shape, vec![5]);
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_layer_num_parameters() {
        let weights = Array2::ones((10, 5));
        let bias = Some(Array1::zeros(5));

        let layer = LayerData::dense("dense1", weights, bias);

        // Should have approximately 10*5 + 5 = 55 parameters
        let num_params = layer.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_numrs2_model_creation() {
        let metadata = ModelMetadata::builder()
            .name("test_model")
            .version("1.0.0")
            .architecture("MLP")
            .build()
            .unwrap();

        let layer1 = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let layer2 = LayerData::dense("layer2", Array2::ones((5, 2)), None);

        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        assert_eq!(model.num_layers(), 2);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_get_layer_by_index() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer1 = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let layer2 = LayerData::dense("layer2", Array2::ones((5, 2)), None);

        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        let layer = model.get_layer(0);
        assert!(layer.is_ok());
        assert_eq!(layer.unwrap().name, "layer1");

        let layer_invalid = model.get_layer(10);
        assert!(layer_invalid.is_err());
    }

    #[test]
    fn test_get_layer_by_name() {
        let metadata = ModelMetadata::builder().name("test_model").build().unwrap();

        let layer1 = LayerData::dense("layer1", Array2::ones((10, 5)), None);
        let layer2 = LayerData::dense("layer2", Array2::ones((5, 2)), None);

        let model = NumRS2Model::new(metadata, vec![layer1, layer2]);

        let layer = model.get_layer_by_name("layer2");
        assert!(layer.is_ok());
        assert_eq!(layer.unwrap().name, "layer2");

        let layer_invalid = model.get_layer_by_name("nonexistent");
        assert!(layer_invalid.is_err());
    }

    #[test]
    fn test_optimizer_state_adam() {
        let opt = OptimizerState::adam(0.001, 0.9, 0.999, 1e-8);

        assert_eq!(opt.optimizer_name, "Adam");
        assert_eq!(opt.learning_rate, 0.001);
        assert_eq!(opt.parameters.get("beta1").unwrap(), &0.9);
        assert_eq!(opt.parameters.get("beta2").unwrap(), &0.999);
        assert!(opt.first_moments.is_some());
        assert!(opt.second_moments.is_some());
    }

    #[test]
    fn test_optimizer_state_sgd() {
        let opt = OptimizerState::sgd(0.01, 0.9);

        assert_eq!(opt.optimizer_name, "SGD");
        assert_eq!(opt.learning_rate, 0.01);
        assert_eq!(opt.parameters.get("momentum").unwrap(), &0.9);
        assert!(opt.velocity.is_some());
    }

    #[test]
    fn test_layer_weights_deserialization() {
        let weights = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let layer = LayerData::dense("test", weights.clone(), None);

        let deserialized = layer.weights_as_array2();
        assert!(deserialized.is_ok());

        let recovered = deserialized.unwrap();
        assert_eq!(recovered.shape(), weights.shape());
    }

    #[test]
    fn test_compression_type_serialization() {
        let comp = CompressionType::Oxicode;
        let serialized = serde_json::to_string(&comp).unwrap();
        let deserialized: CompressionType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(comp, deserialized);
    }

    #[test]
    fn test_layer_type_variants() {
        assert_eq!(LayerType::Dense as u32, 0);
        assert_ne!(LayerType::Conv, LayerType::Dense);
        assert_ne!(LayerType::Attention, LayerType::Dense);
    }

    #[test]
    fn test_activation_type_variants() {
        assert_ne!(ActivationType::ReLU, ActivationType::GELU);
        assert_ne!(ActivationType::Sigmoid, ActivationType::Tanh);
    }
}

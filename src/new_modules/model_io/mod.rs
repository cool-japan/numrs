//! Model Serialization and Deployment Formats
//!
//! This module provides comprehensive model serialization, loading, and export functionality
//! for neural network models in NumRS2. It supports multiple formats and provides utilities
//! for model versioning, compression, and validation.
//!
//! # Features
//!
//! ## Core Format
//! - **NumRS2 Native Format** (.numrs2): Optimized binary format with compression
//! - **Model Metadata**: Architecture, version, hyperparameters, training info
//! - **Layer Serialization**: Support for all NN layer types (Dense, Conv, Transformer, etc.)
//! - **Weight Compression**: SIMD-optimized compression using Oxicode
//!
//! ## Serialization Operations
//! - **Save/Load**: Complete model serialization with metadata
//! - **Checkpoint Management**: Versioning and best model tracking
//! - **Incremental Saves**: Streaming for large models
//! - **Optimizer State**: Save/load optimizer parameters
//!
//! ## Export Formats
//! - **JSON**: Human-readable model description
//! - **MessagePack**: Compact binary format
//! - **NPY/NPZ**: Weight matrices for NumPy/SciPy interoperability
//! - **Architecture Description**: Standalone architecture export
//!
//! ## Compatibility and Validation
//! - **Version Migration**: Automatic migration from older versions
//! - **Backward Compatibility**: Support for legacy formats
//! - **Model Validation**: Integrity checks after loading
//! - **Schema Evolution**: Handle format changes gracefully
//!
//! ## Utilities
//! - **Model Compression**: Quantization metadata, pruning info
//! - **Fingerprinting**: Cryptographic hashes for verification
//! - **Format Detection**: Automatic format identification
//! - **Streaming**: Memory-efficient large model handling
//!
//! # SCIRS2 Integration Policy
//!
//! This module follows NumRS2's SCIRS2 integration policy:
//! - **Serialization**: Use Oxicode (NEVER bincode - COOLJAPAN Policy)
//! - **Compression**: Use OxiARC-Archive for ZIP/archive operations
//! - **Array Operations**: Use scirs2_core::ndarray
//! - **Error Handling**: Base on NumRs2Error
//! - **Pure Rust**: 100% Pure Rust implementation
//!
//! # Quick Start
//!
//! ## Save a Model
//!
//! ```rust,ignore
//! use numrs2::new_modules::model_io::*;
//! use scirs2_core::ndarray::Array2;
//!
//! // Create model metadata
//! let metadata = ModelMetadata::builder()
//!     .name("my_transformer")
//!     .version("1.0.0")
//!     .architecture("Transformer")
//!     .build()?;
//!
//! // Create layer data
//! let layers = vec![
//!     LayerData::dense("layer1", Array2::ones((512, 256)), None),
//!     LayerData::dense("layer2", Array2::ones((256, 128)), None),
//! ];
//!
//! // Save model
//! let model = NumRS2Model::new(metadata, layers);
//! model.save("model.numrs2")?;
//! ```
//!
//! ## Load a Model
//!
//! ```rust,ignore
//! use numrs2::new_modules::model_io::*;
//!
//! // Load model
//! let model = NumRS2Model::load("model.numrs2")?;
//!
//! // Validate model
//! model.validate()?;
//!
//! // Access metadata
//! println!("Model: {} v{}", model.metadata.name, model.metadata.version);
//! ```
//!
//! ## Export to Different Formats
//!
//! ```rust,ignore
//! use numrs2::new_modules::model_io::*;
//!
//! let model = NumRS2Model::load("model.numrs2")?;
//!
//! // Export to JSON (human-readable)
//! model.export_json("model.json")?;
//!
//! // Export to MessagePack (compact)
//! model.export_messagepack("model.msgpack")?;
//!
//! // Export weights to NPZ
//! model.export_weights_npz("weights.npz")?;
//! ```
//!
//! # Module Organization
//!
//! - [`mod@format`]: NumRS2 native model format definitions
//! - [`serialize`]: Save/load and checkpoint management
//! - [`export`]: Export to various formats (JSON, MessagePack, NPY/NPZ)
//! - [`compat`]: Version migration and compatibility utilities
//! - [`utils`]: Compression, fingerprinting, and helper functions

pub mod compat;
pub mod export;
pub mod format;
pub mod serialize;
pub mod utils;

// Re-export commonly used items
pub use compat::{
    migrate_model, validate_model_version, ModelMigration, ModelValidator, VersionInfo,
};
pub use export::{
    export_architecture, export_to_json, export_to_messagepack, export_weights_npy,
    export_weights_npz, ExportFormat, ModelExporter,
};
pub use format::{
    ActivationType, CompressionType, LayerData, LayerType, ModelFormat, ModelMetadata,
    ModelMetadataBuilder, NumRS2Model, OptimizerState, TrainingInfo,
};
pub use serialize::{
    load_checkpoint, load_model, save_checkpoint, save_model, CheckpointManager, ModelCheckpoint,
    ModelSerializer,
};
pub use utils::{
    compress_weights, compute_model_hash, decompress_weights, detect_format, quantize_weights,
    ModelCompression, ModelFingerprint, StreamingSerializer,
};

//! Extended Neural Network Modules
//!
//! This module provides advanced neural network architectures and components
//! that extend beyond the basic primitives in `src/nn/`.
//!
//! # Modules
//!
//! - **graph**: Graph Neural Network primitives (GCN, GAT, GraphSAGE, MPNN, GIN)
//! - **transformers**: Transformer architecture components (Multi-head attention, Encoder/Decoder layers)
//!
//! # SCIRS2 Integration Policy
//!
//! All modules follow NumRS2's SCIRS2 integration policy:
//! - Use `scirs2_core::ndarray` for array operations
//! - Use `scirs2_core::simd_ops` for SIMD acceleration
//! - Use `scirs2_linalg` for linear algebra
//! - Use `scirs2_core::parallel_ops` for parallel operations
//! - Pure Rust implementation via OxiBLAS

pub mod graph;
pub mod transformers;

// Re-export commonly used graph types
pub use graph::*;

// Re-export commonly used transformer types
pub use transformers::{
    create_causal_mask, MultiHeadAttention, PositionalEncoding, PositionalEncodingType,
    PositionwiseFeedForward, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer,
};

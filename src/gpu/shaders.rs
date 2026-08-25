//! GPU Shader Modules
//!
//! This module provides access to the WGSL sources used for GPU operations.
//!
//! Sources ending in `_TEMPLATE` are not valid WGSL on their own: their
//! `SCALAR`, `NEG_LIMIT` and `POS_LIMIT` placeholders are substituted per
//! element type when [`crate::gpu::GpuContext`] compiles them, which keeps the
//! f32 and f64 kernels from drifting apart. The word-based kernels need no
//! substitution because they move raw 32-bit words rather than typed values.

// Re-export all shader files as constants for inclusion
#[allow(dead_code)]
pub const ELEMENT_WISE_F32: &str = include_str!("shaders/element_wise_f32.wgsl");
#[allow(dead_code)]
pub const ELEMENT_WISE_F64: &str = include_str!("shaders/element_wise_f64.wgsl");
#[allow(dead_code)]
pub const MATMUL_F32: &str = include_str!("shaders/matmul_f32.wgsl");
#[allow(dead_code)]
pub const MATMUL_F64: &str = include_str!("shaders/matmul_f64.wgsl");
#[allow(dead_code)]
pub const REDUCTION_TEMPLATE: &str = include_str!("shaders/reduction_template.wgsl");
#[allow(dead_code)]
pub const BROADCAST_TEMPLATE: &str = include_str!("shaders/broadcast_template.wgsl");
#[allow(dead_code)]
pub const GATHER_WORDS: &str = include_str!("shaders/gather_words.wgsl");
#[allow(dead_code)]
pub const IM2COL_WORDS: &str = include_str!("shaders/im2col_words.wgsl");

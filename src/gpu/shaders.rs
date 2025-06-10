//! GPU Shader Modules
//!
//! This module provides access to the compiled WGSL shaders used for GPU operations.

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
pub const REDUCTION_F32: &str = include_str!("shaders/reduction_f32.wgsl");
#[allow(dead_code)]
pub const REDUCTION_F64: &str = include_str!("shaders/reduction_f64.wgsl");

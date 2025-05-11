//! Placeholder types for non-GPU builds
//!
//! This module provides placeholder types and functions when the GPU feature is not enabled.
//! This allows code to still compile with appropriate error messages when GPU functionality 
//! is attempted to be used without the feature enabled.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::fmt;
use std::marker::PhantomData;

/// Placeholder for GPU context when GPU support is not enabled
pub struct GpuContext;

impl GpuContext {
    /// Creates a new GPU context
    pub fn new() -> Result<Self> {
        Err(NumRs2Error::NotImplemented(
            "GPU support is not enabled. Recompile with --features gpu".to_string()
        ))
    }
}

/// Placeholder for GPU array when GPU support is not enabled
pub struct GpuArray<T> {
    _phantom: PhantomData<T>
}

impl<T> GpuArray<T> {
    /// Creates a new GPU array from a CPU array
    pub fn from_array(_array: &Array<T>) -> Result<Self> {
        Err(NumRs2Error::NotImplemented(
            "GPU support is not enabled. Recompile with --features gpu".to_string()
        ))
    }

    /// Converts the GPU array back to a CPU array
    pub fn to_array(&self) -> Result<Array<T>> {
        Err(NumRs2Error::NotImplemented(
            "GPU support is not enabled. Recompile with --features gpu".to_string()
        ))
    }
}

impl<T> fmt::Debug for GpuArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuArray<_> (GPU support not enabled)")
    }
}

/// Placeholder for GPU add operation
pub fn add<T>(_a: &GpuArray<T>, _b: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}

/// Placeholder for GPU subtract operation
pub fn subtract<T>(_a: &GpuArray<T>, _b: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}

/// Placeholder for GPU multiply operation
pub fn multiply<T>(_a: &GpuArray<T>, _b: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}

/// Placeholder for GPU divide operation
pub fn divide<T>(_a: &GpuArray<T>, _b: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}

/// Placeholder for GPU matrix multiplication
pub fn matmul<T>(_a: &GpuArray<T>, _b: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}

/// Placeholder for GPU transpose operation
pub fn transpose<T>(_a: &GpuArray<T>) -> Result<GpuArray<T>> {
    Err(NumRs2Error::NotImplemented(
        "GPU support is not enabled. Recompile with --features gpu".to_string()
    ))
}
//! GPU Utility Functions
//!
//! This module provides utility functions for working with GPU arrays and contexts.

use crate::error::Result;
use crate::gpu::context::{GpuContext, GpuContextRef};
use std::sync::{Mutex, OnceLock};

// Global context for GPU operations
static DEFAULT_CONTEXT: OnceLock<Mutex<Option<GpuContextRef>>> = OnceLock::new();

/// Gets the default GPU context, creating it if it doesn't exist
pub fn get_default_context() -> Result<GpuContextRef> {
    let context_mutex = DEFAULT_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut context_guard = context_mutex
        .lock()
        .expect("GPU context mutex poisoned - another thread panicked while holding the lock");

    if context_guard.is_none() {
        *context_guard = Some(crate::gpu::context::new_context()?);
    }

    Ok(context_guard
        .as_ref()
        .expect("GPU context should be initialized at this point")
        .clone())
}

/// Checks if the hardware supports GPU acceleration
#[allow(dead_code)]
pub fn is_gpu_available() -> bool {
    // Try to create a context and return true if it succeeds
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();

    let rt = match rt {
        Ok(runtime) => runtime,
        Err(_) => return false,
    };
    let context_result = rt.block_on(GpuContext::new());

    context_result.is_ok()
}

/// Returns the name of the GPU if available
#[allow(dead_code)]
pub fn get_gpu_info() -> Option<String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    // Create a new instance
    let instance = wgpu::Instance::default();

    // Request an adapter
    let adapter_future = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    });

    let adapter = rt.block_on(adapter_future).ok()?;
    let info = adapter.get_info();

    Some(format!("{} ({:?})", info.name, info.backend))
}

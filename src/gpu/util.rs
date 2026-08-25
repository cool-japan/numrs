//! GPU Utility Functions
//!
//! This module provides utility functions for working with GPU arrays and contexts.

use crate::error::{NumRs2Error, Result};
use crate::gpu::context::{adapter_options, fallback_adapter_requested, GpuContext, GpuContextRef};
use std::sync::{Mutex, OnceLock};

// Global context for GPU operations
static DEFAULT_CONTEXT: OnceLock<Mutex<Option<GpuContextRef>>> = OnceLock::new();

/// Gets the default GPU context, creating it if it doesn't exist
///
/// The context is created once and shared by every GPU array that does not
/// name a context explicitly, so a program only ever opens one device.
pub fn get_default_context() -> Result<GpuContextRef> {
    let context_mutex = DEFAULT_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut context_guard = context_mutex.lock().map_err(|_| {
        NumRs2Error::RuntimeError(
            "GPU context mutex poisoned - another thread panicked while holding the lock"
                .to_string(),
        )
    })?;

    // Populate the slot on first use, then hand out clones of the stored
    // reference. Matching on the slot avoids re-reading it (and the
    // unreachable "must be initialized" assumption that came with it).
    match context_guard.as_ref() {
        Some(context) => Ok(context.clone()),
        None => {
            let context = crate::gpu::context::new_context()?;
            *context_guard = Some(context.clone());
            Ok(context)
        }
    }
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

    // Request an adapter, honouring the software-adapter opt-in so that the
    // reported device matches the one operations would actually run on.
    let adapter_future = instance.request_adapter(&adapter_options(fallback_adapter_requested()));

    let adapter = rt.block_on(adapter_future).ok()?;
    let info = adapter.get_info();

    Some(format!("{} ({:?})", info.name, info.backend))
}

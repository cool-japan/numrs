//! GPU Utility Functions
//!
//! This module provides utility functions for working with GPU arrays and contexts.

use crate::error::{NumRs2Error, Result};
use crate::gpu::context::{
    adapter_options, fallback_adapter_requested, runtime_access, GpuContextRef, RuntimeAccess,
};
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
///
/// Delegates to [`crate::gpu::new_context`], so this inherits the same
/// runtime-nesting guard: called from within a single-threaded Tokio
/// runtime - where probing would otherwise have to nest a second runtime
/// inside the caller's and panic - it conservatively reports `false`
/// (`new_context` returns an error there) rather than attempting an unsafe
/// block. Async code that needs an accurate answer from within such a
/// runtime should probe with [`crate::gpu::new_context_async`] directly
/// instead of calling this function.
pub fn is_gpu_available() -> bool {
    crate::gpu::context::new_context().is_ok()
}

/// Returns the name of the GPU if available
///
/// Like [`is_gpu_available`], this refuses to nest a Tokio runtime: called
/// from within a single-threaded runtime it returns `None` instead of
/// panicking. See `RuntimeAccess` (in `crate::gpu::context`) for the full
/// set of cases.
pub fn get_gpu_info() -> Option<String> {
    // Request an adapter, honouring the software-adapter opt-in so that the
    // reported device matches the one operations would actually run on.
    let instance = wgpu::Instance::default();
    let adapter_future = instance.request_adapter(&adapter_options(fallback_adapter_requested()));

    let adapter = match runtime_access() {
        RuntimeAccess::MultiThread(handle) => {
            tokio::task::block_in_place(|| handle.block_on(adapter_future)).ok()?
        }
        RuntimeAccess::CurrentThread => return None,
        RuntimeAccess::None => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()?;
            rt.block_on(adapter_future).ok()?
        }
    };
    let info = adapter.get_info();

    Some(format!("{} ({:?})", info.name, info.backend))
}

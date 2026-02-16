//! GPU Module Tests
//!
//! This module contains comprehensive tests for GPU functionality including
//! compute shader management, memory management, linear algebra operations,
//! and general GPU operations.

#[cfg(feature = "gpu")]
mod test_compute;

#[cfg(feature = "gpu")]
mod test_gpu_memory;

#[cfg(feature = "gpu")]
mod test_gpu_linalg;

#[cfg(feature = "gpu")]
mod test_gpu_ops;

#[cfg(feature = "gpu")]
mod test_batching;

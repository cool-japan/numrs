//! Signal Processing Module
//!
//! This module provides signal processing functionality including FFT operations,
//! frequency domain analysis, and related mathematical transforms.
//!
//! ## Features
//!
//! - Fast Fourier Transform (FFT) and inverse FFT
//! - Enhanced FFT operations with frequency domain utilities
//! - Real and complex FFT variants
//! - Multi-dimensional FFT support

// Re-export functionality from the former new_modules
pub use crate::new_modules::fft::*;

// Additional enhanced FFT functionality  
pub mod enhanced {
    //! Enhanced FFT operations with additional frequency domain utilities
    pub use crate::new_modules::fft_enhanced::*;
}
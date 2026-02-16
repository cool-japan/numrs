//! # Wavelets Module
//!
//! This module provides a comprehensive implementation of wavelet transforms and analysis
//! for signal and image processing applications.
//!
//! ## Overview
//!
//! Wavelets are mathematical functions that decompose signals into different frequency
//! components at multiple scales, enabling efficient time-frequency analysis. This module
//! implements:
//!
//! - **Wavelet Families**: Haar, Daubechies, Symlets, Coiflets
//! - **Discrete Wavelet Transform (DWT)**: Fast decomposition and reconstruction
//! - **Continuous Wavelet Transform (CWT)**: Time-frequency analysis
//! - **Wavelet Packets**: Full binary tree decomposition with best basis selection
//! - **Multiresolution Analysis (MRA)**: Signal denoising and reconstruction
//!
//! ## Mathematical Background
//!
//! A wavelet function ψ(t) satisfies:
//!
//! ```text
//! ∫_{-∞}^{∞} ψ(t) dt = 0  (zero mean)
//! ∫_{-∞}^{∞} |ψ(t)|² dt = 1  (unit energy)
//! ```
//!
//! The DWT uses a pair of filters (h, g) for decomposition and reconstruction:
//! - h: low-pass filter (approximation)
//! - g: high-pass filter (detail)
//!
//! ## Examples
//!
//! ```rust,ignore
//! use numrs::new_modules::wavelets::{Wavelet, WaveletType, dwt_1d, idwt_1d};
//!
//! // Create a signal
//! let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//!
//! // Perform DWT with Haar wavelet
//! let wavelet = WaveletType::Haar.create();
//! let (approx, detail) = dwt_1d(&signal, &wavelet, ExtensionMode::Periodic)?;
//!
//! // Reconstruct signal
//! let reconstructed = idwt_1d(&approx, &detail, &wavelet, signal.len())?;
//! ```
//!
//! ## Features
//!
//! - Pure Rust implementation with no external dependencies beyond SciRS2
//! - SIMD-optimized operations where applicable
//! - Comprehensive error handling
//! - Multiple extension modes for boundary handling
//! - Entropy-based best basis selection for wavelet packets

use crate::error::NumRs2Error;
use std::fmt;

/// Result type for wavelet operations
pub type WaveletResult<T> = Result<T, WaveletError>;

/// Error types for wavelet operations
#[derive(Debug, Clone)]
pub enum WaveletError {
    /// Invalid wavelet type or parameters
    InvalidWavelet(String),
    /// Signal length incompatible with operation
    InvalidLength(String),
    /// Invalid decomposition level
    InvalidLevel(String),
    /// Invalid scale parameter
    InvalidScale(String),
    /// Filter length mismatch
    FilterMismatch(String),
    /// Insufficient data for operation
    InsufficientData(String),
    /// Numerical computation error
    ComputationError(String),
    /// Conversion error from NumRs2Error
    NumRs2Error(String),
}

impl fmt::Display for WaveletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveletError::InvalidWavelet(msg) => write!(f, "Invalid wavelet: {}", msg),
            WaveletError::InvalidLength(msg) => write!(f, "Invalid length: {}", msg),
            WaveletError::InvalidLevel(msg) => write!(f, "Invalid level: {}", msg),
            WaveletError::InvalidScale(msg) => write!(f, "Invalid scale: {}", msg),
            WaveletError::FilterMismatch(msg) => write!(f, "Filter mismatch: {}", msg),
            WaveletError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            WaveletError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            WaveletError::NumRs2Error(msg) => write!(f, "NumRs2 error: {}", msg),
        }
    }
}

impl std::error::Error for WaveletError {}

impl From<NumRs2Error> for WaveletError {
    fn from(err: NumRs2Error) -> Self {
        WaveletError::NumRs2Error(err.to_string())
    }
}

/// Signal extension mode for boundary handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionMode {
    /// Periodic extension (wrap-around)
    Periodic,
    /// Symmetric extension (mirror)
    Symmetric,
    /// Zero-padding extension
    ZeroPad,
    /// Smooth extension (replicate edge values)
    Smooth,
}

impl ExtensionMode {
    /// Get the description of the extension mode
    pub fn description(&self) -> &str {
        match self {
            ExtensionMode::Periodic => "Periodic wrap-around extension",
            ExtensionMode::Symmetric => "Symmetric mirror extension",
            ExtensionMode::ZeroPad => "Zero-padding extension",
            ExtensionMode::Smooth => "Smooth edge replication extension",
        }
    }
}

// Submodules
pub mod cwt;
pub mod dwt;
pub mod families;
pub mod mra;
pub mod packets;

// Re-exports
pub use cwt::{cwt, CWTResult, ContinuousWavelet};
pub use dwt::{dwt_1d, idwt_1d, idwt_1d_mode, wavedec, waverec, waverec_mode};
pub use families::{Wavelet, WaveletType};
pub use mra::{denoise_signal, MultiresolutionAnalysis, ThresholdType};
pub use packets::{packet_decompose, BestBasisCriterion, WaveletPacket};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_mode_description() {
        assert!(!ExtensionMode::Periodic.description().is_empty());
        assert!(!ExtensionMode::Symmetric.description().is_empty());
        assert!(!ExtensionMode::ZeroPad.description().is_empty());
        assert!(!ExtensionMode::Smooth.description().is_empty());
    }

    #[test]
    fn test_wavelet_error_display() {
        let err = WaveletError::InvalidWavelet("test".to_string());
        assert!(err.to_string().contains("Invalid wavelet"));

        let err = WaveletError::InvalidLength("test".to_string());
        assert!(err.to_string().contains("Invalid length"));
    }

    #[test]
    fn test_wavelet_error_from_numrs2() {
        let numrs2_err = NumRs2Error::ShapeMismatch {
            expected: vec![2, 2],
            actual: vec![3, 3],
        };
        let wavelet_err: WaveletError = numrs2_err.into();
        assert!(matches!(wavelet_err, WaveletError::NumRs2Error(_)));
    }
}

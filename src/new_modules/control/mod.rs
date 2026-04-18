//! # Control Systems Module
//!
//! This module provides comprehensive control systems analysis and design tools for NumRS2.
//!
//! ## Overview
//!
//! The control systems module implements classical and modern control theory tools including:
//! - Transfer function representation and analysis
//! - State-space representation and transformations
//! - Stability analysis (Routh-Hurwitz, Nyquist, Bode, Root Locus)
//! - PID controller design and tuning
//! - Frequency response analysis
//! - Time-domain simulation
//!
//! ## Control Theory Background
//!
//! Control systems can be represented in two main forms:
//!
//! ### Transfer Function
//! ```text
//! H(s) = N(s)/D(s) = (b_n*s^n + ... + b_1*s + b_0)/(a_m*s^m + ... + a_1*s + a_0)
//! ```
//! where s is the complex frequency variable (Laplace domain).
//!
//! ### State-Space
//! ```text
//! dx/dt = Ax + Bu  (state equation)
//! y = Cx + Du      (output equation)
//! ```
//! where x is the state vector, u is the input, and y is the output.
//!
//! ## Examples
//!
//! ### Transfer Function Analysis
//! ```rust
//! use numrs2::new_modules::control::{TransferFunction, ControlResult};
//!
//! # fn main() -> ControlResult<()> {
//! // Create a transfer function: H(s) = (s + 2)/(s^2 + 3s + 2)
//! let num = vec![1.0, 2.0];  // s + 2
//! let den = vec![1.0, 3.0, 2.0];  // s^2 + 3s + 2
//! let tf = TransferFunction::new(num, den)?;
//!
//! // Compute poles and zeros
//! let poles = tf.poles()?;
//! let zeros = tf.zeros()?;
//!
//! // Evaluate frequency response
//! let frequencies = vec![0.1, 1.0, 10.0];
//! let response = tf.frequency_response(&frequencies)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### State-Space Representation
//! ```rust
//! use numrs2::new_modules::control::{StateSpace, ControlResult};
//! use scirs2_core::ndarray::array;
//!
//! # fn main() -> ControlResult<()> {
//! // Create a state-space system
//! let a = array![[0.0, 1.0], [-2.0, -3.0]];
//! let b = array![[0.0], [1.0]];
//! let c = array![[1.0, 0.0]];
//! let d = array![[0.0]];
//! let ss = StateSpace::new(a, b, c, d)?;
//!
//! // Check controllability and observability
//! let is_controllable = ss.is_controllable()?;
//! let is_observable = ss.is_observable()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### PID Controller Design
//! ```rust
//! use numrs2::new_modules::control::{PIDController, TuningMethod, ControlResult};
//!
//! # fn main() -> ControlResult<()> {
//! // Create a PID controller
//! let mut pid = PIDController::new(1.0, 0.5, 0.1, 0.01);
//!
//! // Auto-tune using Ziegler-Nichols
//! let ultimate_gain = 4.0;
//! let ultimate_period = 2.0;
//! pid.tune_ziegler_nichols(ultimate_gain, ultimate_period)?;
//!
//! // Compute control output
//! let error = 1.0;
//! let output = pid.update(error);
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;
use std::fmt;

// Submodules
pub mod pid;
pub mod stability;
pub mod state_space;
pub mod transfer_function;

// Re-exports
pub use pid::{PIDController, TuningMethod};
pub use stability::{BodePlotData, NyquistPlotData, RouthHurwitz, StabilityMargins};
pub use state_space::StateSpace;
pub use transfer_function::TransferFunction;

/// Result type for control systems operations
pub type ControlResult<T> = Result<T, ControlError>;

/// Error types for control systems operations
#[derive(Debug, Clone)]
pub enum ControlError {
    /// Invalid dimensions for matrices or vectors
    DimensionMismatch { expected: String, actual: String },
    /// Invalid polynomial coefficients (e.g., all zeros)
    InvalidPolynomial(String),
    /// System is not controllable
    NotControllable,
    /// System is not observable
    NotObservable,
    /// System is unstable
    Unstable(String),
    /// Invalid parameters for controller design
    InvalidParameters(String),
    /// Numerical computation failed
    NumericalError(String),
    /// Conversion between representations failed
    ConversionError(String),
    /// Invalid system configuration
    InvalidSystem(String),
    /// Linear algebra operation failed
    LinAlgError(String),
    /// General control systems error
    General(String),
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ControlError::InvalidPolynomial(msg) => {
                write!(f, "Invalid polynomial: {}", msg)
            }
            ControlError::NotControllable => {
                write!(f, "System is not controllable")
            }
            ControlError::NotObservable => {
                write!(f, "System is not observable")
            }
            ControlError::Unstable(msg) => {
                write!(f, "System is unstable: {}", msg)
            }
            ControlError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            ControlError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
            ControlError::ConversionError(msg) => {
                write!(f, "Conversion error: {}", msg)
            }
            ControlError::InvalidSystem(msg) => {
                write!(f, "Invalid system: {}", msg)
            }
            ControlError::LinAlgError(msg) => {
                write!(f, "Linear algebra error: {}", msg)
            }
            ControlError::General(msg) => {
                write!(f, "Control error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ControlError {}

impl From<ControlError> for crate::error::NumRs2Error {
    fn from(err: ControlError) -> Self {
        crate::error::NumRs2Error::ControlError(err.to_string())
    }
}

/// Frequency response data containing magnitude and phase
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Frequencies at which response was computed (rad/s)
    pub frequencies: Array1<f64>,
    /// Magnitude response (linear scale)
    pub magnitude: Array1<f64>,
    /// Phase response (radians)
    pub phase: Array1<f64>,
}

impl FrequencyResponse {
    /// Create a new frequency response
    pub fn new(
        frequencies: Array1<f64>,
        magnitude: Array1<f64>,
        phase: Array1<f64>,
    ) -> ControlResult<Self> {
        if frequencies.len() != magnitude.len() || frequencies.len() != phase.len() {
            return Err(ControlError::DimensionMismatch {
                expected: format!(
                    "all arrays same length as frequencies ({})",
                    frequencies.len()
                ),
                actual: format!("magnitude: {}, phase: {}", magnitude.len(), phase.len()),
            });
        }
        Ok(Self {
            frequencies,
            magnitude,
            phase,
        })
    }

    /// Get magnitude in decibels
    pub fn magnitude_db(&self) -> Array1<f64> {
        self.magnitude.mapv(|m| 20.0 * m.log10())
    }

    /// Get phase in degrees
    pub fn phase_deg(&self) -> Array1<f64> {
        self.phase.mapv(|p| p.to_degrees())
    }
}

/// System type (continuous or discrete)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemType {
    /// Continuous-time system (Laplace domain, s)
    Continuous,
    /// Discrete-time system (Z domain, z)
    Discrete { sample_time: u64 }, // sample_time in microseconds to avoid float
}

impl SystemType {
    /// Check if system is continuous
    pub fn is_continuous(&self) -> bool {
        matches!(self, SystemType::Continuous)
    }

    /// Check if system is discrete
    pub fn is_discrete(&self) -> bool {
        matches!(self, SystemType::Discrete { .. })
    }

    /// Get sample time for discrete systems (in seconds)
    pub fn sample_time(&self) -> Option<f64> {
        match self {
            SystemType::Discrete { sample_time } => Some(*sample_time as f64 / 1_000_000.0),
            SystemType::Continuous => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_response_creation() {
        let freqs = Array1::from_vec(vec![1.0, 10.0, 100.0]);
        let mags = Array1::from_vec(vec![1.0, 0.5, 0.1]);
        let phases = Array1::from_vec(vec![0.0, -0.5, -1.0]);

        let fr = FrequencyResponse::new(freqs.clone(), mags.clone(), phases.clone());
        assert!(fr.is_ok());

        let fr = fr.expect("test: valid frequency response creation");
        assert_eq!(fr.frequencies, freqs);
        assert_eq!(fr.magnitude, mags);
        assert_eq!(fr.phase, phases);
    }

    #[test]
    fn test_frequency_response_dimension_mismatch() {
        let freqs = Array1::from_vec(vec![1.0, 10.0, 100.0]);
        let mags = Array1::from_vec(vec![1.0, 0.5]);
        let phases = Array1::from_vec(vec![0.0, -0.5, -1.0]);

        let fr = FrequencyResponse::new(freqs, mags, phases);
        assert!(fr.is_err());
    }

    #[test]
    fn test_magnitude_db_conversion() {
        let freqs = Array1::from_vec(vec![1.0]);
        let mags = Array1::from_vec(vec![10.0]);
        let phases = Array1::from_vec(vec![0.0]);

        let fr =
            FrequencyResponse::new(freqs, mags, phases).expect("test: valid frequency response");
        let mag_db = fr.magnitude_db();

        assert!((mag_db[0] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_deg_conversion() {
        let freqs = Array1::from_vec(vec![1.0]);
        let mags = Array1::from_vec(vec![1.0]);
        let phases = Array1::from_vec(vec![std::f64::consts::PI]);

        let fr =
            FrequencyResponse::new(freqs, mags, phases).expect("test: valid frequency response");
        let phase_deg = fr.phase_deg();

        assert!((phase_deg[0] - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_system_type() {
        let cont = SystemType::Continuous;
        assert!(cont.is_continuous());
        assert!(!cont.is_discrete());
        assert_eq!(cont.sample_time(), None);

        let disc = SystemType::Discrete {
            sample_time: 100_000,
        }; // 0.1 seconds
        assert!(!disc.is_continuous());
        assert!(disc.is_discrete());
        assert!(
            (disc
                .sample_time()
                .expect("test: discrete system has sample time")
                - 0.1)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_error_conversion() {
        let err = ControlError::InvalidPolynomial("test".to_string());
        let numrs_err: crate::error::NumRs2Error = err.into();

        match numrs_err {
            crate::error::NumRs2Error::ControlError(msg) => {
                assert!(msg.contains("Invalid polynomial"));
            }
            _ => panic!("Expected ControlError"),
        }
    }
}

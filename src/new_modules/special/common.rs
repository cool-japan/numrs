//! Common types and traits for special functions
//!
//! This module provides common imports and types used across all special function implementations.

#![allow(unused_imports)]

use num_traits::Float;
use std::fmt::Debug;

// Re-export commonly used types
pub use crate::array::Array as SpecialArray;
pub use crate::error::{NumRs2Error as SpecialError, Result as SpecialResult};

/// Trait for special function compatible floating point types
pub trait SpecialFloat: Float + Debug + Clone {}

impl<T: Float + Debug + Clone> SpecialFloat for T {}

/// Helper function to convert a float constant, returning a descriptive error
#[inline]
#[allow(dead_code)]
pub fn float_const<T: Float>(value: f64, name: &str) -> T {
    T::from(value).unwrap_or_else(|| panic!("{} should convert to float type", name))
}

/// Euler's constant (gamma)
pub const EULER_GAMMA: f64 = 0.577_215_664_901_533;

/// Omega constant (W(1))
pub const OMEGA_CONSTANT: f64 = 0.567_143_290_409_783_9;

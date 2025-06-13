//! Random Number Generation
//!
//! This module provides a NumPy-like interface for generating random numbers from various
//! probability distributions. It features both a modern, object-oriented approach and a
//! legacy functional approach for backward compatibility.
//!
//! ## Quick Start
//!
//! ```
//! use numrs2::random::{default_rng, pcg64_seed_rng};
//!
//! // Create a Generator with the default bit generator
//! let rng = default_rng();
//!
//! // Generate uniform random numbers between 0 and 1
//! let uniform_array = rng.random::<f64>(&[3, 3]).unwrap();
//!
//! // Generate random numbers from a normal distribution
//! let normal_array = rng.normal(0.0, 1.0, &[5]).unwrap();
//!
//! // Create a seeded generator for reproducible results
//! let seeded_rng = pcg64_seed_rng(42);
//! let random_array = seeded_rng.random::<f64>(&[3]).unwrap();
//! ```
//!
//! ## Module Structure
//!
//! - `generator.rs`: Modern Generator class and BitGenerator implementations
//! - `state.rs`: RandomState class for legacy interface
//! - `distributions.rs`: Common probability distributions
//! - `advanced_distributions.rs`: Specialized distributions not in standard libraries
//!
//! ## Available Interfaces
//!
//! ### 1. Modern Interface (Recommended)
//!
//! Uses the Generator class with BitGenerator implementations. This is similar to
//! NumPy's Generator class introduced in NumPy 1.17.
//!
//! ```
//! use numrs2::random::default_rng;
//!
//! // Create a Generator with the default bit generator
//! let rng = default_rng();
//!
//! // Generate random numbers
//! let arr = rng.random::<f64>(&[3, 3]).unwrap();
//! ```
//!
//! ### 2. Legacy Interface
//!
//! Uses the RandomState class and global functions. This is provided for
//! backward compatibility with earlier NumPy versions.
//!
//! ```
//! use numrs2::random::{RandomState, normal, set_seed};
//!
//! // Create a RandomState with a specific seed
//! let rng = RandomState::with_seed(42);
//!
//! // Generate random numbers with the instance
//! let arr = rng.random::<f64>(&[3, 3]).unwrap();
//!
//! // Or use global functions
//! set_seed(42);
//! let normal_array = normal(0.0, 1.0, &[5]).unwrap();
//! ```
//!
//! ## Bit Generators
//!
//! The module provides two bit generator implementations:
//!
//! - `StdBitGenerator`: Based on the rand crate's StdRng, which uses ChaCha algorithm
//! - `PCG64BitGenerator`: Based on the PCG64 algorithm, providing high-quality randomness
//!
//! ## Advanced Usage
//!
//! See the examples directory for comprehensive usage demonstrations:
//! - `random_distributions_example.rs`: Shows all distribution functions
//! - `random_simple_test.rs`: A simplified example for quick verification

// Import the modules
pub mod advanced_distributions;
pub mod distributions;
pub mod distributions_enhanced;
pub mod generator;
pub mod legacy;
pub mod state;

// Re-export essential items from the modules
#[cfg(not(feature = "scirs"))]
pub use advanced_distributions::{maxwell, noncentral_chisquare, noncentral_f, vonmises, wald};
pub use distributions::*;
pub use generator::{default_rng, BitGenerator, Generator, PCG64BitGenerator, StdBitGenerator};
pub use generator::{pcg64_rng, pcg64_seed_rng};
pub use state::RandomState;

// Re-export enhanced distributions
#[cfg(not(feature = "scirs"))]
pub use distributions_enhanced::{
    copula, latin_hypercube, mixture_of_normals, multivariate_normal_cholesky, power,
    random_correlation_matrix, sobol_sequence, truncated_normal,
};

#[cfg(feature = "scirs")]
pub use distributions_enhanced::{
    copula, latin_hypercube, mixture_of_normals, multivariate_normal_cholesky, power,
    random_correlation_matrix, sobol_sequence,
};

// Use SciRS2 integration when the feature is enabled
#[cfg(feature = "scirs")]
pub use crate::interop::scirs_compat::{
    maxwell, multivariate_normal_with_rotation, noncentral_chisquare, noncentral_f,
    truncated_normal, vonmises,
};

/// Create a new RandomState with the given seed.
///
/// This is a convenience function for creating a RandomState with a specific seed.
///
/// # Arguments
///
/// * `seed` - The seed to use for the random number generator.
///
/// # Returns
///
/// A new RandomState.
///
/// # Example
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create a RandomState with a specific seed
/// let rng = numrs2::random::seed_rng(42);
///
/// // Generate random numbers
/// let arr = rng.random::<f64>(&[3, 3]).unwrap();
/// ```
pub fn seed_rng(seed: u64) -> RandomState {
    RandomState::with_seed(seed)
}

/// Legacy backward compatibility module (re-exports from random_base.rs)
pub mod random_base {
    //! Legacy random number generation interface
    //!
    //! This module provides backward compatibility for the original random_base.rs functionality.
    //! All functionality has been moved to the unified random module structure.
    
    pub use super::legacy::*;
}

// Direct re-exports for legacy compatibility
pub use legacy::{Generator as LegacyGenerator, choice, rand, randn, seed, shuffle, uniform};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_state() {
        let rng1 = RandomState::with_seed(42);
        let rng2 = RandomState::with_seed(42);

        let arr1 = rng1.random::<f64>(&[5, 5]).unwrap();
        let arr2 = rng2.random::<f64>(&[5, 5]).unwrap();

        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }

    #[test]
    fn test_seed_rng() {
        let rng = seed_rng(123);
        let arr = rng.random::<f64>(&[3, 3]).unwrap();

        assert_eq!(arr.shape(), vec![3, 3]);
    }
}

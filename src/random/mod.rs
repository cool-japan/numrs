//! Random number generation
//!
//! This module provides functionality for generating random numbers from various
//! probability distributions, similar to NumPy's random module.

// We don't re-export the base module, just have it available
// for backward compatibility if needed

// Import the new modules
pub mod state;
pub mod distributions;

// Re-export essential items from the new modules
pub use state::RandomState;
pub use distributions::*;

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
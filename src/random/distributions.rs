//! Advanced random distributions 
//!
//! This module provides functions for generating random arrays from various
//! probability distributions, similar to NumPy's random.distributions module.

use crate::array::Array;
use crate::error::Result;
use crate::random::state::RandomState;

// Global generator for convenience
lazy_static::lazy_static! {
    static ref GLOBAL_RANDOM_STATE: std::sync::Mutex<RandomState> = std::sync::Mutex::new(RandomState::new());
}

/// Set the random seed for the global generator
pub fn set_seed(seed: u64) {
    if let Ok(mut guard) = GLOBAL_RANDOM_STATE.lock() {
        *guard = RandomState::with_seed(seed);
    }
}

/// Get a reference to the global random state
fn get_global_random_state() -> Result<std::sync::MutexGuard<'static, RandomState>> {
    GLOBAL_RANDOM_STATE.lock().map_err(|e| {
        crate::error::NumRs2Error::InvalidOperation(
            format!("Failed to acquire global random state lock: {}", e)
        )
    })
}

/// Generate random values from a beta distribution using the global generator
///
/// # Arguments
///
/// * `a` - Alpha parameter of the beta distribution
/// * `b` - Beta parameter of the beta distribution
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the beta distribution
pub fn beta<T>(a: T, b: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.beta(a, b, shape)
}

/// Generate random values from a binomial distribution using the global generator
///
/// # Arguments
///
/// * `n` - Number of trials
/// * `p` - Probability of success in each trial
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the binomial distribution
pub fn binomial<T>(n: u64, p: f64, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::NumCast + Clone + std::fmt::Debug
{
    let rng = get_global_random_state()?;
    rng.binomial(n, p, shape)
}

/// Generate random values from a chi-square distribution using the global generator
///
/// # Arguments
///
/// * `df` - Degrees of freedom
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the chi-square distribution
pub fn chisquare<T>(df: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.chisquare(df, shape)
}

/// Generate random values from a Dirichlet distribution using the global generator
///
/// # Arguments
///
/// * `alpha` - Concentration parameters
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Dirichlet distribution
pub fn dirichlet<T>(alpha: &[T], shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.dirichlet(alpha, shape)
}

/// Generate random values from a gamma distribution using the global generator
///
/// # Arguments
///
/// * `shape` - Shape parameter of the gamma distribution
/// * `scale` - Scale parameter of the gamma distribution
/// * `output_shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the gamma distribution
pub fn gamma<T>(shape_param: T, scale: T, output_shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.gamma(shape_param, scale, output_shape)
}

/// Generate random values from a normal distribution using the global generator
///
/// # Arguments
///
/// * `mean` - Mean of the normal distribution
/// * `std` - Standard deviation of the normal distribution
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the normal distribution
pub fn normal<T>(mean: T, std: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.normal(mean, std, shape)
}

/// Generate random values from a standard normal distribution using the global generator
///
/// # Arguments
///
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the standard normal distribution
pub fn standard_normal<T>(shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.standard_normal(shape)
}

/// Generate random values from a Poisson distribution using the global generator
///
/// # Arguments
///
/// * `lam` - Mean of the Poisson distribution
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Poisson distribution
pub fn poisson<T>(lam: f64, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::NumCast + Clone + std::fmt::Debug
{
    let rng = get_global_random_state()?;
    rng.poisson(lam, shape)
}

/// Generate random values from a uniform distribution using the global generator
///
/// # Arguments
///
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (inclusive)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the uniform distribution
pub fn uniform<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform
{
    let rng = get_global_random_state()?;
    rng.uniform(low, high, shape)
}

/// Generate random integers in the range [low, high) using the global generator
///
/// # Arguments
///
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random integers
pub fn integers<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform + Into<i64> + TryFrom<i64>,
    <T as TryFrom<i64>>::Error: std::fmt::Debug
{
    let rng = get_global_random_state()?;
    rng.integers(low, high, shape)
}

/// Generate random values from a log-normal distribution using the global generator
///
/// # Arguments
///
/// * `mean` - Mean of the log-normal distribution
/// * `sigma` - Standard deviation of the log-normal distribution
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the log-normal distribution
pub fn lognormal<T>(mean: T, sigma: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.lognormal(mean, sigma, shape)
}

/// Generate random values from a Cauchy distribution using the global generator
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Cauchy distribution
pub fn cauchy<T>(loc: T, scale: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.cauchy(loc, scale, shape)
}

/// Generate random values from a Student's t-distribution using the global generator
///
/// # Arguments
///
/// * `df` - Degrees of freedom
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Student's t-distribution
pub fn student_t<T>(df: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.student_t(df, shape)
}

/// Generate random values from an exponential distribution using the global generator
///
/// # Arguments
///
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the exponential distribution
pub fn exponential<T>(scale: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.exponential(scale, shape)
}

/// Generate random values from a Weibull distribution using the global generator
///
/// # Arguments
///
/// * `shape_param` - Shape parameter of the Weibull distribution
/// * `scale` - Scale parameter of the Weibull distribution
/// * `output_shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Weibull distribution
pub fn weibull<T>(shape_param: T, scale: T, output_shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.weibull(shape_param, scale, output_shape)
}

/// Generate random binary values with given probability of success
///
/// # Arguments
///
/// * `p` - Probability of success
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random binary values
pub fn bernoulli<T>(p: T, shape: &[usize]) -> Result<Array<T>> 
where 
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display
{
    let rng = get_global_random_state()?;
    rng.bernoulli(p, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_beta_distribution() {
        let arr = beta(2.0, 5.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);
    }
    
    #[test]
    fn test_normal_distribution() {
        let arr = normal(0.0, 1.0, &[5, 5]).unwrap();
        assert_eq!(arr.shape(), vec![5, 5]);
    }
    
    #[test]
    fn test_standard_normal_distribution() {
        let arr = standard_normal::<f64>(&[3, 3]).unwrap();
        assert_eq!(arr.shape(), vec![3, 3]);
    }
    
    #[test]
    fn test_binomial_distribution() {
        let arr = binomial::<u64>(10, 0.5, &[5]).unwrap();
        assert_eq!(arr.shape(), vec![5]);
        
        // Values should be in the range [0, 10]
        for val in arr.to_vec() {
            assert!(val <= 10);
        }
    }
    
    #[test]
    fn test_gamma_distribution() {
        let arr = gamma(2.0, 2.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);
        
        // Gamma values should be positive
        for val in arr.to_vec() {
            assert!(val > 0.0);
        }
    }
    
    #[test]
    fn test_set_seed() {
        set_seed(12345);
        let arr1 = normal(0.0, 1.0, &[5]).unwrap();
        
        set_seed(12345);
        let arr2 = normal(0.0, 1.0, &[5]).unwrap();
        
        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }
}
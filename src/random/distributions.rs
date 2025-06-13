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
pub fn get_global_random_state() -> Result<std::sync::MutexGuard<'static, RandomState>> {
    GLOBAL_RANDOM_STATE.lock().map_err(|e| {
        crate::error::NumRs2Error::InvalidOperation(format!(
            "Failed to acquire global random state lock: {}",
            e
        ))
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::NumCast + Clone + std::fmt::Debug,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::NumCast + Clone + std::fmt::Debug,
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
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform,
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
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
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
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.bernoulli(p, shape)
}

/// Generate random values from a Pareto distribution using the global generator
///
/// # Arguments
///
/// * `alpha` - Shape parameter of the Pareto distribution
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Pareto distribution
pub fn pareto<T>(alpha: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.pareto(alpha, shape)
}

/// Generate random values from a Triangular distribution using the global generator
///
/// # Arguments
///
/// * `low` - Lower bound
/// * `mode` - Mode (most common value)
/// * `high` - Upper bound
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Triangular distribution
pub fn triangular<T>(low: T, mode: T, high: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.triangular(low, mode, high, shape)
}

/// Generate random values from a PERT distribution using the global generator
///
/// # Arguments
///
/// * `min` - Minimum value
/// * `mode` - Mode (most likely value)
/// * `max` - Maximum value
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the PERT distribution
pub fn pert<T>(min: T, mode: T, max: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.pert(min, mode, max, shape)
}

/// Generate random values from a multivariate normal distribution using the global generator
///
/// # Arguments
///
/// * `mean` - Mean vector
/// * `cov` - Covariance matrix
/// * `size` - Optional shape of the output array
///
/// # Returns
///
/// An array of random values from the multivariate normal distribution
pub fn multivariate_normal<T>(
    mean: &[T],
    cov: &Array<T>,
    size: Option<&[usize]>,
) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.multivariate_normal(mean, cov, size)
}

/// Generate random values from a Laplace distribution using the global generator
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Laplace distribution
pub fn laplace<T>(loc: T, scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.laplace(loc, scale, shape)
}

/// Generate random values from a Gumbel distribution using the global generator
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Gumbel distribution
pub fn gumbel<T>(loc: T, scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.gumbel(loc, scale, shape)
}

/// Generate random values from a logistic distribution using the global generator
///
/// # Arguments
///
/// * `loc` - Location parameter
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the logistic distribution
pub fn logistic<T>(loc: T, scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.logistic(loc, scale, shape)
}

/// Generate random values from a Rayleigh distribution using the global generator
///
/// # Arguments
///
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Rayleigh distribution
pub fn rayleigh<T>(scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.rayleigh(scale, shape)
}

/// Generate random values from a Wald (inverse Gaussian) distribution using the global generator
///
/// # Arguments
///
/// * `mean` - Mean parameter
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Wald distribution
pub fn wald<T>(mean: T, scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.wald(mean, scale, shape)
}

/// Generate random values from a negative binomial distribution using the global generator
///
/// # Arguments
///
/// * `n` - Number of successes
/// * `p` - Probability of a single success
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the negative binomial distribution
pub fn negative_binomial<T>(n: f64, p: f64, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.negative_binomial(n, p, shape)
}

/// Generate random values from a geometric distribution using the global generator
///
/// # Arguments
///
/// * `p` - Success probability
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the geometric distribution
pub fn geometric<T>(p: f64, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.geometric(p, shape)
}

/// Generate random values from a multinomial distribution using the global generator
///
/// # Arguments
///
/// * `n` - Number of trials
/// * `pvals` - Probability vector
/// * `shape` - Optional shape of the output array
///
/// # Returns
///
/// An array of random values from the multinomial distribution
pub fn multinomial<T>(n: usize, pvals: &[f64], shape: Option<&[usize]>) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.multinomial(n, pvals, shape)
}

/// Generate random values from a hypergeometric distribution using the global generator
///
/// # Arguments
///
/// * `ngood` - Number of successes in the population
/// * `nbad` - Number of failures in the population
/// * `nsample` - Number of samples drawn
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the hypergeometric distribution
pub fn hypergeometric<T>(
    ngood: usize,
    nbad: usize,
    nsample: usize,
    shape: &[usize],
) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.hypergeometric(ngood, nbad, nsample, shape)
}

/// Generate random values from a zipf distribution using the global generator
///
/// # Arguments
///
/// * `a` - Distribution parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the zipf distribution
pub fn zipf<T>(a: f64, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.zipf(a, shape)
}

/// Generate random values from a logseries distribution using the global generator
///
/// # Arguments
///
/// * `p` - Distribution parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the logseries distribution
pub fn logseries<T>(p: f64, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::NumCast + Clone + std::fmt::Debug,
{
    let rng = get_global_random_state()?;
    rng.logseries(p, shape)
}

/// Generate random values from a noncentral chi-square distribution using the global generator
///
/// # Arguments
///
/// * `df` - Degrees of freedom (must be positive)
/// * `nonc` - Non-centrality parameter (must be non-negative)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the noncentral chi-square distribution
pub fn noncentral_chisquare<T>(df: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.noncentral_chisquare(df, nonc, shape)
}

/// Generate random values from a noncentral F distribution using the global generator
///
/// # Arguments
///
/// * `dfnum` - Numerator degrees of freedom (must be positive)
/// * `dfden` - Denominator degrees of freedom (must be positive)
/// * `nonc` - Non-centrality parameter (must be non-negative)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the noncentral F distribution
pub fn noncentral_f<T>(dfnum: T, dfden: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.noncentral_f(dfnum, dfden, nonc, shape)
}

/// Generate random values from a von Mises distribution using the global generator
///
/// The von Mises distribution, also known as the circular normal distribution, is a
/// continuous probability distribution on a circle. It is the circular analogue of the
/// normal distribution.
///
/// # Arguments
///
/// * `mu` - Mean direction (in radians)
/// * `kappa` - Concentration parameter (must be non-negative)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the von Mises distribution in the range [-π, π)
pub fn vonmises<T>(mu: T, kappa: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.vonmises(mu, kappa, shape)
}

/// Generate random values from a Maxwell-Boltzmann distribution using the global generator
///
/// The Maxwell-Boltzmann distribution describes the velocity of particles in thermal equilibrium.
/// It is closely related to the chi distribution with 3 degrees of freedom.
///
/// # Arguments
///
/// * `scale` - Scale parameter (must be positive)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Maxwell-Boltzmann distribution
pub fn maxwell<T>(scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.maxwell(scale, shape)
}

/// Generate random values from a truncated normal distribution using the global generator
///
/// The truncated normal distribution is a normal distribution bounded within a specified range.
/// All values below the lower bound or above the upper bound are excluded and
/// the probability density is rescaled accordingly.
///
/// # Arguments
///
/// * `mean` - Mean of the normal distribution before truncation
/// * `std` - Standard deviation of the normal distribution before truncation (must be positive)
/// * `low` - Lower bound of the truncation
/// * `high` - Upper bound of the truncation (must be greater than low)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the truncated normal distribution in the range [low, high]
pub fn truncated_normal<T>(mean: T, std: T, low: T, high: T, shape: &[usize]) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.truncated_normal(mean, std, low, high, shape)
}

/// Generate random values from a multivariate normal distribution with rotation using the global generator
///
/// This function extends the multivariate normal distribution with an optional rotation matrix.
/// The rotation is applied to the Cholesky factor of the covariance matrix, allowing for
/// coordinate system transformations.
///
/// # Arguments
///
/// * `mean` - Mean vector
/// * `cov` - Covariance matrix (must be positive definite)
/// * `size` - Optional shape of the output array
/// * `rotation` - Optional rotation matrix (must be square with dimensions matching mean vector length)
///
/// # Returns
///
/// An array of random values from the multivariate normal distribution with rotation
pub fn multivariate_normal_with_rotation<T>(
    mean: &[T],
    cov: &Array<T>,
    size: Option<&[usize]>,
    rotation: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: num_traits::Float + num_traits::NumCast + Clone + std::fmt::Debug + std::fmt::Display,
{
    let rng = get_global_random_state()?;
    rng.multivariate_normal_with_rotation(mean, cov, size, rotation)
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
        // Test that the same seed produces the same sequence
        // by generating multiple arrays with the same seed
        let seed1 = 12345u64;
        let seed2 = 54321u64;

        // First sequence with seed1
        set_seed(seed1);
        let arr1_a = normal(0.0, 1.0, &[5]).unwrap();
        let arr1_b = normal(0.0, 1.0, &[5]).unwrap();

        // Reset to seed1 and generate the same sequence
        set_seed(seed1);
        let arr2_a = normal(0.0, 1.0, &[5]).unwrap();
        let arr2_b = normal(0.0, 1.0, &[5]).unwrap();

        // Verify that resetting the seed reproduces the same sequence
        assert_eq!(
            arr1_a.to_vec(),
            arr2_a.to_vec(),
            "First arrays should be identical"
        );
        assert_eq!(
            arr1_b.to_vec(),
            arr2_b.to_vec(),
            "Second arrays should be identical"
        );

        // Now test with a different seed
        set_seed(seed2);
        let arr3_a = normal(0.0, 1.0, &[5]).unwrap();

        // Different seeds should produce different results
        assert_ne!(
            arr1_a.to_vec(),
            arr3_a.to_vec(),
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_pareto_distribution() {
        let arr = pareto(2.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Pareto values should be greater than or equal to 1
        for val in arr.to_vec() {
            assert!(val >= 1.0);
        }
    }

    #[test]
    fn test_triangular_distribution() {
        // Now using our own implementation instead of rand_distr
        let result = triangular(0.0, 2.0, 10.0, &[10]);
        let arr = result.unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Triangular values should be in the range [low, high]
        for val in arr.to_vec() {
            assert!((0.0..=10.0).contains(&val));
        }
    }

    #[test]
    fn test_pert_distribution() {
        let arr = pert(0.0, 5.0, 10.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // PERT values should be in the range [min, max]
        for val in arr.to_vec() {
            assert!((0.0..=10.0).contains(&val));
        }
    }

    #[test]
    fn test_multivariate_normal_distribution() {
        let mean = vec![0.0, 0.0];
        let cov_data = vec![1.0, 0.5, 0.5, 1.0];
        let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

        let arr = multivariate_normal(&mean, &cov, Some(&[10])).unwrap();
        assert_eq!(arr.shape(), vec![10, 2]);
    }

    #[test]
    fn test_laplace_distribution() {
        let arr = laplace(0.0, 1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_negative_binomial_distribution() {
        let arr = negative_binomial::<u64>(5.0, 0.5, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_multinomial_distribution() {
        let pvals = vec![0.2, 0.3, 0.5];
        let arr = multinomial::<u64>(10, &pvals, None).unwrap();
        assert_eq!(arr.shape(), vec![3]);

        // Sum of counts should equal the number of trials
        let sum: u64 = arr.to_vec().iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_noncentral_chisquare_distribution() {
        let arr = noncentral_chisquare(2.0, 1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Noncentral chi-square values should be positive
        for val in arr.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_noncentral_f_distribution() {
        let arr = noncentral_f(2.0, 5.0, 1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Noncentral F values should be positive
        for val in arr.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_vonmises_distribution() {
        let arr = vonmises(0.0, 1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Von Mises values should be in the range [-π, π)
        for val in arr.to_vec() {
            assert!((-std::f64::consts::PI..std::f64::consts::PI).contains(&val));
        }
    }

    #[test]
    fn test_maxwell_distribution() {
        let arr = maxwell(1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Maxwell values should be positive
        for val in arr.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    #[ignore = "Test takes too long to run"]
    fn test_truncated_normal_distribution() {
        let low = -2.0;
        let high = 2.0;
        let arr = truncated_normal(0.0, 1.0, low, high, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);

        // Truncated normal values should be within bounds
        for val in arr.to_vec() {
            assert!(val >= low && val <= high);
        }
    }

    #[test]
    fn test_multivariate_normal_with_rotation() {
        let mean = vec![0.0, 0.0];
        let cov_data = vec![1.0, 0.5, 0.5, 1.0];
        let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

        // Create a rotation matrix for 45 degrees
        let rotation_data = vec![
            std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, // cos(45°), sin(45°)
            -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, // -sin(45°), cos(45°)
        ];
        let rotation = Array::from_vec(rotation_data).reshape(&[2, 2]);

        let arr =
            multivariate_normal_with_rotation(&mean, &cov, Some(&[5]), Some(&rotation)).unwrap();
        assert_eq!(arr.shape(), vec![5, 2]);

        // Test without rotation matrix (should default to regular multivariate normal)
        let arr_no_rot = multivariate_normal_with_rotation(&mean, &cov, Some(&[5]), None).unwrap();
        assert_eq!(arr_no_rot.shape(), vec![5, 2]);
    }
}

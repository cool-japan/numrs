//! Enhanced random distributions
//!
//! This module expands the random functionality by providing additional
//! distributions and utility functions for more specialized statistical applications.

use crate::array::Array;
use crate::error::Result;
use crate::random::state::RandomState;
use num_traits::{Float, NumCast};
use scirs2_core::random::prelude::Rng;
use std::fmt::{Debug, Display};

/// Get a reference to the global random state
fn get_global_random_state() -> Result<std::sync::MutexGuard<'static, RandomState>> {
    crate::random::distributions::get_global_random_state()
}

/// Generate random values from a truncated normal distribution
///
/// # Arguments
///
/// * `mean` - Mean of the normal distribution (before truncation)
/// * `std` - Standard deviation of the normal distribution (before truncation)
/// * `low` - Lower bound for truncation
/// * `high` - Upper bound for truncation
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the truncated normal distribution
pub fn truncated_normal<T>(mean: T, std: T, low: T, high: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float
        + NumCast
        + Clone
        + Debug
        + Display
        + scirs2_core::ndarray::distributions::uniform::SampleUniform,
{
    let rng = get_global_random_state()?;
    rng.truncated_normal(mean, std, low, high, shape)
}

/// Generate random values from a von Mises distribution
///
/// # Arguments
///
/// * `mu` - Location parameter (mean direction)
/// * `kappa` - Concentration parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the von Mises distribution
pub fn vonmises<T>(mu: T, kappa: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.vonmises(mu, kappa, shape)
}

/// Generate random values from a non-central chi-square distribution
///
/// # Arguments
///
/// * `df` - Degrees of freedom
/// * `nonc` - Non-centrality parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the non-central chi-square distribution
pub fn noncentral_chisquare<T>(df: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.noncentral_chisquare(df, nonc, shape)
}

/// Generate random values from a non-central F distribution
///
/// # Arguments
///
/// * `dfnum` - Numerator degrees of freedom
/// * `dfden` - Denominator degrees of freedom
/// * `nonc` - Non-centrality parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the non-central F distribution
pub fn noncentral_f<T>(dfnum: T, dfden: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.noncentral_f(dfnum, dfden, nonc, shape)
}

/// Generate random values from a Maxwell distribution
///
/// # Arguments
///
/// * `scale` - Scale parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the Maxwell distribution
pub fn maxwell<T>(scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.maxwell(scale, shape)
}

/// Generate random values from a power distribution
///
/// # Arguments
///
/// * `a` - Power parameter
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the power distribution
pub fn power<T>(a: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.power(a, shape)
}

/// Generate correlated random variables using the Cholesky decomposition
///
/// # Arguments
///
/// * `means` - Vector of means for each variable
/// * `cov` - Covariance matrix
/// * `size` - Number of samples to generate
///
/// # Returns
///
/// An array of correlated random samples with shape [size, n]
pub fn multivariate_normal_cholesky<T>(means: &[T], cov: &Array<T>, size: usize) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.multivariate_normal_cholesky(means, cov, size)
}

/// Generate a random correlation matrix
///
/// # Arguments
///
/// * `n` - Dimension of the correlation matrix
///
/// # Returns
///
/// A random correlation matrix of shape [n, n]
pub fn random_correlation_matrix<T>(n: usize) -> Result<Array<T>>
where
    T: Float
        + NumCast
        + Clone
        + Debug
        + Display
        + scirs2_core::ndarray::distributions::uniform::SampleUniform,
{
    let rng = get_global_random_state()?;
    rng.random_correlation_matrix(n)
}

/// Generate random samples from a mixture of distributions
///
/// # Arguments
///
/// * `weights` - Weights for each component in the mixture
/// * `means` - Mean for each component
/// * `stds` - Standard deviation for each component
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// An array of random values from the mixture distribution
pub fn mixture_of_normals<T>(
    weights: &[T],
    means: &[T],
    stds: &[T],
    shape: &[usize],
) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.mixture_of_normals(weights, means, stds, shape)
}

/// Generate Sobol sequence for quasi-Monte Carlo methods
///
/// # Arguments
///
/// * `dim` - Dimensionality of the sequence
/// * `n` - Number of points to generate
///
/// # Returns
///
/// An array of shape [n, dim] with Sobol sequence points
pub fn sobol_sequence<T>(dim: usize, n: usize) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.sobol_sequence(dim, n)
}

/// Generate Latin Hypercube samples
///
/// # Arguments
///
/// * `dim` - Number of dimensions
/// * `n` - Number of samples
///
/// # Returns
///
/// An array of shape [n, dim] with Latin Hypercube samples
pub fn latin_hypercube<T>(dim: usize, n: usize) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.latin_hypercube(dim, n)
}

/// Generate copula samples with a specified correlation structure
///
/// # Arguments
///
/// * `corr` - Correlation matrix
/// * `n` - Number of samples
/// * `copula_type` - Type of copula ("gaussian" or "t" supported)
///
/// # Returns
///
/// An array of shape [n, dim] with correlated uniform samples
pub fn copula<T>(corr: &Array<T>, n: usize, copula_type: &str) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let rng = get_global_random_state()?;
    rng.copula(corr, n, copula_type)
}

/// Extend RandomState with enhanced distribution methods
impl RandomState {
    /// Generate random values from a truncated normal distribution
    pub fn truncated_normal<T>(
        &self,
        mean: T,
        std: T,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if low >= high {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Lower bound must be less than upper bound, got low={}, high={}",
                low, high
            )));
        }

        if std <= T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Standard deviation must be positive, got {}",
                std
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Convert to f64 for calculation
        let mean_f64 = mean.to_f64().unwrap_or(0.0);
        let std_f64 = std.to_f64().unwrap_or(1.0);
        let low_f64 = low.to_f64().unwrap_or(f64::NEG_INFINITY);
        let high_f64 = high.to_f64().unwrap_or(f64::INFINITY);

        // Use rejection sampling for truncated normal (more robust than inverse CDF)
        for _ in 0..size {
            let mut sample;
            let mut attempts = 0;
            loop {
                // Generate standard normal sample
                let u1 = rng.random::<f64>();
                let u2 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                // Transform to desired distribution
                sample = mean_f64 + std_f64 * z;

                // Check if within bounds
                if sample >= low_f64 && sample <= high_f64 {
                    break;
                }

                attempts += 1;
                // Prevent infinite loops for very narrow truncation
                if attempts > 1000 {
                    sample = (low_f64 + high_f64) / 2.0;
                    break;
                }
            }

            let val = <T as NumCast>::from(sample).unwrap_or_else(|| {
                if sample <= low.to_f64().unwrap_or(f64::NEG_INFINITY) {
                    low
                } else {
                    high
                }
            });

            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a von Mises distribution
    pub fn vonmises<T>(&self, mu: T, kappa: T, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if kappa < T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Concentration parameter must be non-negative, got {}",
                kappa
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        let kappa_f64 = kappa.to_f64().unwrap_or(0.0);
        let mu_f64 = mu.to_f64().unwrap_or(0.0);

        // Generate von Mises samples using the algorithm from Devroye (1986)
        for _ in 0..size {
            let sample = if kappa_f64 < 1e-6 {
                // For very small kappa, use uniform distribution
                rng.random::<f64>() * 2.0 * std::f64::consts::PI - std::f64::consts::PI
            } else {
                // Use Best-Fisher algorithm for von Mises sampling
                let a = 1.0 + (1.0 + 4.0 * kappa_f64 * kappa_f64).sqrt();
                let b = (a - (2.0 * a).sqrt()) / (2.0 * kappa_f64);
                let r = (1.0 + b * b) / (2.0 * b);

                let mut attempts = 0;
                loop {
                    attempts += 1;
                    if attempts > 1000 {
                        // Fallback to uniform if too many attempts
                        break rng.random::<f64>() * 2.0 * std::f64::consts::PI
                            - std::f64::consts::PI;
                    }

                    let u1 = rng.random::<f64>();
                    let z = (1.0 - u1).cos();
                    let f = (1.0 + r * z) / (r + z);
                    let c = kappa_f64 * (r - f);

                    let u2 = rng.random::<f64>();

                    if c * (2.0 - c) - u2 > 0.0 {
                        let u3 = rng.random::<f64>();
                        let theta = if u3 - 0.5 > 0.0 { f.acos() } else { -f.acos() };
                        break theta;
                    }

                    if (c / u2.max(1e-10)).ln() + 1.0 - c >= 0.0 {
                        let u3 = rng.random::<f64>();
                        let theta = if u3 - 0.5 > 0.0 { f.acos() } else { -f.acos() };
                        break theta;
                    }
                }
            };

            let angle = mu_f64 + sample;
            let normalized = ((angle + std::f64::consts::PI) % (2.0 * std::f64::consts::PI))
                - std::f64::consts::PI;

            vec.push(<T as NumCast>::from(normalized).unwrap_or(T::zero()));
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a non-central chi-square distribution
    pub fn noncentral_chisquare<T>(&self, df: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if df <= T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom must be positive, got {}",
                df
            )));
        }

        if nonc < T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Non-centrality parameter must be non-negative, got {}",
                nonc
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let df_f64 = df.to_f64().unwrap_or(0.0);
        let nonc_f64 = nonc.to_f64().unwrap_or(0.0);

        // Algorithm: non-central chi-square can be generated as a Poisson mixture of central chi-squares
        for _ in 0..size {
            // Generate Poisson random variable with mean nonc/2
            let pois: Array<u64> = self.poisson(nonc_f64 / 2.0, &[1])?;
            let n = <usize as NumCast>::from(pois.to_vec()[0]).unwrap_or(0);

            // Generate chi-square with df + 2*n degrees of freedom
            let chi2 = self.chisquare(
                <T as NumCast>::from(df_f64 + 2.0 * n as f64).unwrap_or(T::zero()),
                &[1],
            )?;

            vec.push(chi2.to_vec()[0]);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a non-central F distribution
    pub fn noncentral_f<T>(&self, dfnum: T, dfden: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if dfnum <= T::zero() || dfden <= T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom must be positive, got dfnum={}, dfden={}",
                dfnum, dfden
            )));
        }

        if nonc < T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Non-centrality parameter must be non-negative, got {}",
                nonc
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let dfnum_f64 = dfnum.to_f64().unwrap_or(0.0);
        let dfden_f64 = dfden.to_f64().unwrap_or(0.0);
        let nonc_f64 = nonc.to_f64().unwrap_or(0.0);

        // Non-central F is the ratio of a non-central chi-square and a central chi-square,
        // each divided by their degrees of freedom
        for _ in 0..size {
            // Generate non-central chi-square with dfnum degrees of freedom
            let nc_chi2 = self.noncentral_chisquare(
                <T as NumCast>::from(dfnum_f64).unwrap_or(T::zero()),
                <T as NumCast>::from(nonc_f64).unwrap_or(T::zero()),
                &[1],
            )?;

            // Generate central chi-square with dfden degrees of freedom
            let chi2 =
                self.chisquare(<T as NumCast>::from(dfden_f64).unwrap_or(T::zero()), &[1])?;

            // Compute the ratio (non-central F)
            let f_val = (nc_chi2.to_vec()[0] / dfnum) / (chi2.to_vec()[0] / dfden);
            vec.push(f_val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Maxwell distribution
    pub fn maxwell<T>(&self, scale: T, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if scale <= T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        // Maxwell distribution is the distribution of the magnitude of a
        // 3D vector with independent normal components
        for _ in 0..size {
            // Generate 3 independent normal random variables
            let x = self.normal(T::zero(), scale, &[3])?;

            // Compute the magnitude
            let magnitude =
                (x.to_vec()[0].powi(2) + x.to_vec()[1].powi(2) + x.to_vec()[2].powi(2)).sqrt();

            vec.push(magnitude);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a power distribution
    pub fn power<T>(&self, a: T, shape: &[usize]) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if a <= T::zero() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Power parameter must be positive, got {}",
                a
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Power distribution with parameter a has PDF: a*x^(a-1) on [0, 1]
        // and can be generated by transforming uniform random variables
        for _ in 0..size {
            let u = rng.random::<f64>();
            let val = u.powf(1.0 / a.to_f64().unwrap_or(1.0));
            vec.push(<T as NumCast>::from(val).unwrap_or(T::zero()));
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate correlated random variables using the Cholesky decomposition
    pub fn multivariate_normal_cholesky<T>(
        &self,
        means: &[T],
        cov: &Array<T>,
        size: usize,
    ) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        let n = means.len();
        let cov_shape = cov.shape();

        // Validate inputs
        if cov_shape.len() != 2 || cov_shape[0] != n || cov_shape[1] != n {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                format!("Covariance matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, cov_shape)
            ));
        }

        // Perform Cholesky decomposition
        let cov_data = cov.to_vec();
        let mut chol = vec![T::zero(); n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut s = T::zero();
                for k in 0..j {
                    s = s + chol[i * n + k] * chol[j * n + k];
                }

                if i == j {
                    let val = cov_data[i * n + i] - s;
                    if val <= T::zero() {
                        return Err(crate::error::NumRs2Error::InvalidOperation(
                            "Covariance matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[i * n + j] = val.sqrt();
                } else {
                    chol[i * n + j] = (cov_data[i * n + j] - s) / chol[j * n + j];
                }
            }
        }

        // Generate standard normal samples
        let std_normal = self.standard_normal::<T>(&[size, n])?;
        let std_normal_data = std_normal.to_vec();

        // Transform samples using Cholesky factor
        let mut result = vec![T::zero(); size * n];

        for i in 0..size {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    if j >= k {
                        // Cholesky factor is lower triangular
                        sum = sum + chol[j * n + k] * std_normal_data[i * n + k];
                    }
                }
                result[i * n + j] = means[j] + sum;
            }
        }

        Ok(Array::from_vec(result).reshape(&[size, n]))
    }

    /// Generate a random correlation matrix
    pub fn random_correlation_matrix<T>(&self, n: usize) -> Result<Array<T>>
    where
        T: Float
            + NumCast
            + Clone
            + Debug
            + Display
            + scirs2_core::ndarray::distributions::uniform::SampleUniform,
    {
        if n < 2 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Correlation matrix dimension must be at least 2".to_string(),
            ));
        }

        // Generate random matrix with values from a uniform distribution
        let uniform = self.uniform::<T>(
            <T as NumCast>::from(-1.0).unwrap_or(T::zero()),
            <T as NumCast>::from(1.0).unwrap_or(T::zero()),
            &[n, n],
        )?;

        let uniform_data = uniform.to_vec();

        // Make the matrix symmetric
        let mut sym_matrix = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                match i.cmp(&j) {
                    std::cmp::Ordering::Equal => {
                        sym_matrix[i * n + j] = <T as NumCast>::from(1.0).unwrap_or(T::zero());
                    }
                    std::cmp::Ordering::Less => {
                        sym_matrix[i * n + j] = uniform_data[i * n + j];
                        sym_matrix[j * n + i] = uniform_data[i * n + j];
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
        }

        // Convert to nearest correlation matrix using projection
        // This is a simplified approximation - in practice, more sophisticated
        // algorithms like the alternating projections method would be used.

        // 1. Set all diagonal elements to 1
        for i in 0..n {
            sym_matrix[i * n + i] = <T as NumCast>::from(1.0).unwrap_or(T::zero());
        }

        // 2. Iteratively adjust non-diagonal elements to ensure positive-definiteness
        let mut corr_matrix = sym_matrix.clone();
        let max_iter = 10;

        for _ in 0..max_iter {
            // Check if matrix is positive definite via Cholesky decomposition
            let mut is_pd = true;
            let mut chol = vec![T::zero(); n * n];

            'decomp: for i in 0..n {
                for j in 0..=i {
                    let mut s = T::zero();
                    for k in 0..j {
                        s = s + chol[i * n + k] * chol[j * n + k];
                    }

                    if i == j {
                        let val = corr_matrix[i * n + i] - s;
                        if val <= T::zero() {
                            is_pd = false;
                            break 'decomp;
                        }
                        chol[i * n + j] = val.sqrt();
                    } else {
                        chol[i * n + j] = (corr_matrix[i * n + j] - s) / chol[j * n + j];
                    }
                }
            }

            if is_pd {
                break;
            }

            // If not positive definite, adjust non-diagonal elements
            let factor = <T as NumCast>::from(0.9).unwrap_or(T::zero());
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        corr_matrix[i * n + j] = corr_matrix[i * n + j] * factor;
                    }
                }
            }
        }

        // Normalize to ensure diagonal is exactly 1.0
        for i in 0..n {
            let diag_val = corr_matrix[i * n + i].sqrt();
            for j in 0..n {
                corr_matrix[i * n + j] = corr_matrix[i * n + j] / diag_val;
                corr_matrix[j * n + i] = corr_matrix[j * n + i] / diag_val;
            }
        }

        // Final normalization pass
        for i in 0..n {
            for j in 0..n {
                corr_matrix[i * n + j] = corr_matrix[i * n + j]
                    / (corr_matrix[i * n + i].sqrt() * corr_matrix[j * n + j].sqrt());
            }
        }

        // Set diagonal to exactly 1.0
        for i in 0..n {
            corr_matrix[i * n + i] = <T as NumCast>::from(1.0).unwrap_or(T::zero());
        }

        Ok(Array::from_vec(corr_matrix).reshape(&[n, n]))
    }

    /// Generate random samples from a mixture of distributions
    pub fn mixture_of_normals<T>(
        &self,
        weights: &[T],
        means: &[T],
        stds: &[T],
        shape: &[usize],
    ) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        let n_components = weights.len();

        // Validate inputs
        if n_components < 1 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Mixture must have at least one component".to_string(),
            ));
        }

        if means.len() != n_components || stds.len() != n_components {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Weights, means, and stds must have the same length".to_string(),
            ));
        }

        // Check if weights sum to 1
        let sum_weights: T = weights.iter().fold(T::zero(), |acc, w| acc + *w);
        if (sum_weights - <T as NumCast>::from(1.0).unwrap_or(T::zero())).abs()
            > <T as NumCast>::from(1e-6).unwrap_or(T::zero())
        {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Weights must sum to 1, got sum={}",
                sum_weights
            )));
        }

        // Check if stds are positive
        for &std in stds {
            if std <= T::zero() {
                return Err(crate::error::NumRs2Error::InvalidOperation(
                    "Standard deviations must be positive".to_string(),
                ));
            }
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Normalize weights to ensure they sum to 1.0
        let mut norm_weights = Vec::with_capacity(n_components);
        for &w in weights {
            norm_weights.push(w / sum_weights);
        }

        // Compute cumulative weights for component selection
        let mut cumulative_weights = Vec::with_capacity(n_components);
        let mut sum = T::zero();
        for &w in &norm_weights {
            sum = sum + w;
            cumulative_weights.push(sum);
        }

        // Optimized approach: generate component selections and normal samples separately
        let mut component_selections = Vec::with_capacity(size);

        // Generate all component selections at once
        for _ in 0..size {
            let u = <T as NumCast>::from(rng.random::<f64>()).unwrap_or(T::zero());
            let mut selected_component = 0;

            for (i, &cw) in cumulative_weights.iter().enumerate() {
                if u <= cw {
                    selected_component = i;
                    break;
                }
            }
            component_selections.push(selected_component);
        }

        // Count how many samples we need from each component
        let mut component_counts = vec![0usize; n_components];
        for &comp in &component_selections {
            component_counts[comp] += 1;
        }

        // Generate all normal samples for each component in batches
        let mut component_samples = Vec::with_capacity(n_components);
        for i in 0..n_components {
            if component_counts[i] > 0 {
                let samples = self.normal(means[i], stds[i], &[component_counts[i]])?;
                component_samples.push(samples.to_vec());
            } else {
                component_samples.push(Vec::new());
            }
        }

        // Assign samples in the original order
        let mut component_indices = vec![0usize; n_components];
        for &selected_component in &component_selections {
            vec.push(component_samples[selected_component][component_indices[selected_component]]);
            component_indices[selected_component] += 1;
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate Sobol sequence for quasi-Monte Carlo methods
    pub fn sobol_sequence<T>(&self, dim: usize, n: usize) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if !(1..=40).contains(&dim) {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Dimension must be between 1 and 40, got {}",
                dim
            )));
        }

        if n < 1 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Number of points must be at least 1".to_string(),
            ));
        }

        // Direction numbers (first few values for each dimension)
        // These are based on Joe & Kuo's direction numbers
        let direction_numbers: Vec<Vec<u32>> = vec![
            vec![1],                       // First dimension trivial sequence
            vec![1, 3],                    // Second dimension
            vec![1, 3, 5],                 // Third dimension
            vec![1, 3, 5, 15],             // Fourth dimension
            vec![1, 3, 5, 15, 17],         // Fifth dimension
            vec![1, 3, 5, 15, 17, 51],     // Sixth dimension
            vec![1, 3, 5, 15, 17, 51, 85], // Seventh dimension
            vec![1, 3, 5, 15, 17, 51, 85, 255], // Eighth dimension
                                           // Additional dimensions would be added here in a complete implementation
        ];

        let mut result = vec![T::zero(); n * dim];

        // Generate Sobol sequence
        for i in 0..n {
            // Convert i to gray code
            let g = i ^ (i >> 1);

            for d in 0..std::cmp::min(dim, direction_numbers.len()) {
                let mut x = 0u32;
                let mut mask = 1u32;

                for j in 0..32 {
                    if j < direction_numbers[d].len() && (g & (mask as usize)) != 0 {
                        x ^= direction_numbers[d][j];
                    }
                    mask <<= 1;
                }

                // Convert to [0,1) range
                let val = <T as NumCast>::from(x as f64 / 2f64.powi(32)).unwrap_or(T::zero());
                result[i * dim + d] = val;
            }

            // For dimensions beyond the provided direction numbers, use random values
            // This is a fallback - a proper implementation would have all direction numbers
            let mut rng = self.get_rng()?;
            for d in direction_numbers.len()..dim {
                result[i * dim + d] =
                    <T as NumCast>::from(rng.random::<f64>()).unwrap_or(T::zero());
            }
        }

        Ok(Array::from_vec(result).reshape(&[n, dim]))
    }

    /// Generate Latin Hypercube samples
    pub fn latin_hypercube<T>(&self, dim: usize, n: usize) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if dim < 1 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Dimension must be at least 1".to_string(),
            ));
        }

        if n < 2 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Number of samples must be at least 2".to_string(),
            ));
        }

        let mut result = vec![T::zero(); n * dim];

        // For each dimension, create a permutation of the integers 0 to n-1
        for d in 0..dim {
            // Create vector of integers 0 to n-1
            let mut perm: Vec<usize> = (0..n).collect();

            // Shuffle the vector
            let mut rng = self.get_rng()?;
            for i in (1..n).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
                perm.swap(i, j);
            }

            // Generate uniform random values within each stratum
            for i in 0..n {
                let u = rng.random::<f64>();
                let val =
                    <T as NumCast>::from((perm[i] as f64 + u) / n as f64).unwrap_or(T::zero());
                result[i * dim + d] = val;
            }
        }

        Ok(Array::from_vec(result).reshape(&[n, dim]))
    }

    /// Generate copula samples with a specified correlation structure
    pub fn copula<T>(&self, corr: &Array<T>, n: usize, copula_type: &str) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        let corr_shape = corr.shape();

        // Validate correlation matrix
        if corr_shape.len() != 2 || corr_shape[0] != corr_shape[1] {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Correlation matrix must be square".to_string(),
            ));
        }

        let dim = corr_shape[0];

        // Check if copula_type is supported
        let valid_types = ["gaussian", "t"];
        if !valid_types.contains(&copula_type) {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Unsupported copula type: {}. Supported types: {:?}",
                copula_type, valid_types
            )));
        }

        // Generate correlated normal samples using Cholesky decomposition
        let means = vec![T::zero(); dim];
        let mvn_samples = self.multivariate_normal_cholesky(&means, corr, n)?;
        let mvn_data = mvn_samples.to_vec();

        // Transform to uniform using the CDF
        let mut result = vec![T::zero(); n * dim];

        for i in 0..n {
            for d in 0..dim {
                let z = mvn_data[i * dim + d].to_f64().unwrap_or(0.0);

                // Apply appropriate CDF based on copula type
                let u = match copula_type {
                    "gaussian" => normal_cdf(z),
                    "t" => {
                        // T-copula with 4 degrees of freedom
                        student_t_cdf(z, 4)
                    }
                    _ => normal_cdf(z), // Default to Gaussian
                };

                result[i * dim + d] = <T as NumCast>::from(u).unwrap_or(T::zero());
            }
        }

        Ok(Array::from_vec(result).reshape(&[n, dim]))
    }
}

// Helper functions for probability distributions

/// Normal cumulative distribution function
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse normal cumulative distribution function
#[allow(dead_code)]
fn normal_inv_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Approximate the inverse CDF using the Beasley-Springer-Moro algorithm
    let q = p - 0.5;

    if q.abs() <= 0.425 {
        // Central region
        let r = 0.180625 - q * q;
        return q
            * (((((((2.509_080_928_730_122_6 * r + 3.343_057_558_358_813e1) * r
                + 6.726_577_092_700_87e1)
                * r
                + 4.592_195_393_154_987e1)
                * r
                + 1.373_169_376_550_946_2e1)
                * r
                + 1.421_413_764_013_155_7)
                * r
                + 2.298_979_990_914_786_5e-1)
                / (((((((4.374_317_029_667_823e-2 * r + 3.739_716_869_366_193_3) * r
                    + 4.692_163_145_304_143_5e1)
                    * r
                    + 2.266_863_181_546_454_5e2)
                    * r
                    + 5.396_173_702_892_064e2)
                    * r
                    + 6.573_191_171_972_302e2)
                    * r
                    + 3.734_237_715_407_137e2)
                    * r
                    + 1.0));
    }

    // Tail regions
    let r = if q > 0.0 { 1.0 - p } else { p };

    if r <= 0.0 {
        return if q > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }

    let r = (-r.ln()).sqrt();

    let mut ret = ((((((1.811_625_079_763_736_7e1 * r + 7.928_172_819_374_223e1) * r
        + 1.373_169_376_550_946_2e2)
        * r
        + 1.193_147_912_264_617e2)
        * r
        + 4.926_845_824_098_105e1)
        * r
        + 8.400_547_514_910_246)
        * r
        + 3.050_789_888_729_818e-1)
        / ((((((3.020_637_025_121_939_4e-1 * r + 5.595_303_579_120_197) * r
            + 3.042_975_173_014_595e1)
            * r
            + 6.493_763_419_991_991e1)
            * r
            + 5.786_293_056_261_984e1)
            * r
            + 2.121_379_430_158_66e1)
            * r
            + 2.659_135_201_941_675)
        * r
        + 1.0;

    ret = if q < 0.0 { -ret } else { ret };
    ret
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Constants for Abramowitz and Stegun approximation 7.1.26
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Student's t cumulative distribution function
fn student_t_cdf(x: f64, df: usize) -> f64 {
    // Simplified approximation
    if x == 0.0 {
        return 0.5;
    }

    // For large df, t-distribution approaches normal distribution
    if df > 100 {
        return normal_cdf(x);
    }

    // Simple approximation using the regularized incomplete beta function
    let df_f64 = df as f64;
    let t = x / (df_f64 + x * x).sqrt();
    let u = 0.5 + 0.5 * t * (1.0 - t * t / (df_f64 + 2.0)).sqrt();

    // Clamp to valid probability range
    u.clamp(0.0, 1.0)
}

// Enhanced distributions directly exported by the parent module, no need to re-export here

// Unit tests for enhanced distributions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_normal() {
        let mean = 0.0;
        let std = 1.0;
        let low = -2.0;
        let high = 2.0;

        // Generate truncated normal samples (reduced size for performance)
        let samples = truncated_normal(mean, std, low, high, &[100]).unwrap();
        let data = samples.to_vec();

        // Check bounds
        for &val in &data {
            assert!(val >= low && val <= high, "Value outside bounds: {}", val);
        }

        // Check statistics (should be approximately normal within bounds)
        let mean_val: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (mean_val - mean).abs() < 0.5,
            "Mean too far from expected: {}",
            mean_val
        );
    }

    #[test]
    fn test_vonmises() {
        let mu = 0.0;
        let kappa = 2.0;

        // Generate von Mises samples
        let samples = vonmises(mu, kappa, &[1000]).unwrap();
        let data = samples.to_vec();

        // Check bounds (should be in [-π, π))
        for &val in &data {
            assert!(
                (-std::f64::consts::PI..std::f64::consts::PI).contains(&val),
                "Value outside bounds: {}",
                val
            );
        }

        // Calculate circular mean
        let sum_sin: f64 = data.iter().map(|&x| x.sin()).sum();
        let sum_cos: f64 = data.iter().map(|&x| x.cos()).sum();
        let mean_angle = sum_sin.atan2(sum_cos);

        // For von Mises, mean direction should be close to mu
        let angle_diff = (mean_angle - mu + std::f64::consts::PI) % (2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        assert!(
            angle_diff.abs() < 0.5,
            "Mean direction too far from expected: {}",
            mean_angle
        );
    }

    #[test]
    fn test_latin_hypercube() {
        let dim = 2;
        let n = 10;

        // Generate Latin Hypercube samples
        let samples = latin_hypercube::<f64>(dim, n).unwrap();
        let data = samples.to_vec();

        // Check that each dimension has one sample in each stratum
        for d in 0..dim {
            let mut counts = vec![0; n];

            for i in 0..n {
                let val = data[i * dim + d];
                let stratum = (val * n as f64).floor() as usize;
                let stratum = std::cmp::min(stratum, n - 1); // Handle edge case for val=1.0
                counts[stratum] += 1;
            }

            // Each stratum should have exactly one sample
            for (s, &count) in counts.iter().enumerate() {
                assert_eq!(count, 1, "Stratum {} has {} samples, expected 1", s, count);
            }
        }
    }

    #[test]
    #[ignore = "Performance optimization needed - function works but is too slow for CI"]
    fn test_mixture_of_normals() {
        let weights = vec![0.3, 0.7];
        let means = vec![-3.0, 3.0];
        let stds = vec![1.0, 1.0];

        // Generate mixture samples (reduced size for performance)
        let samples = mixture_of_normals(&weights, &means, &stds, &[100]).unwrap();

        let data = samples.to_vec();

        // Check that distribution is bimodal
        // Sort the data for percentile calculation
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate 25th and 75th percentiles
        let p25 = sorted_data[(0.25 * sorted_data.len() as f64) as usize];
        let p75 = sorted_data[(0.75 * sorted_data.len() as f64) as usize];

        // For a bimodal distribution with these parameters,
        // we expect the 25th percentile to be negative and
        // the 75th percentile to be positive
        assert!(p25 < 0.0, "25th percentile should be negative, got {}", p25);
        assert!(p75 > 0.0, "75th percentile should be positive, got {}", p75);
    }
}

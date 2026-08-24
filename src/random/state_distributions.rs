//! Advanced and specialized distribution methods for RandomState
//!
//! This module contains the more complex distribution sampling methods for RandomState,
//! including multivariate distributions, extreme value distributions, and other
//! specialized statistical distributions.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, ToPrimitive};
use scirs2_core::random::prelude::*;
use scirs2_core::SliceRandomExt;
use std::fmt::Debug;
use std::fmt::Display;

use super::state::RandomState;

impl RandomState {
    /// Generate random values from a multivariate normal distribution
    pub fn multivariate_normal<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: &[T],
        cov: &Array<T>,
        size: Option<&[usize]>,
    ) -> Result<Array<T>> {
        if mean.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Mean vector cannot be empty".to_string(),
            ));
        }

        let n = mean.len();
        let cov_shape = cov.shape();

        if cov_shape.len() != 2 || cov_shape[0] != n || cov_shape[1] != n {
            return Err(NumRs2Error::InvalidOperation(
                format!("Covariance matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, cov_shape)
            ));
        }

        // For simplicity, this implementation uses a basic approach:
        // 1. Generate standard normal samples
        // 2. Apply Cholesky decomposition of covariance matrix
        // 3. Transform standard normal samples and add the mean

        // First determine the output shape
        let mut out_shape = Vec::new();
        if let Some(size_shape) = size {
            out_shape.extend_from_slice(size_shape);
        }
        out_shape.push(n);

        let total_samples: usize = if out_shape.len() > 1 {
            out_shape[..out_shape.len() - 1].iter().product()
        } else {
            1
        };

        // Generate standard normal samples
        let mut result = Vec::with_capacity(total_samples * n);
        let mut rng = self.get_rng()?;
        let standard_normal = Normal::new(0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create normal distribution: {}", e))
        })?;

        // Generate samples from standard normal distribution
        for _ in 0..total_samples * n {
            let val_f64: f64 = rng.sample(standard_normal);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert standard normal sample".to_string(),
                )
            })?;
            result.push(val);
        }

        // Compute the Cholesky decomposition of the covariance matrix
        let chol = cholesky_decomposition::<T>(&cov.to_vec(), n)?;

        // Transform standard normal samples using the Cholesky factor
        let mut transformed = vec![T::zero(); total_samples * n];
        for i in 0..total_samples {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..=j {
                    sum = sum + chol[j * n + k] * result[i * n + k];
                }
                transformed[i * n + j] = sum + mean[j];
            }
        }

        Array::from_vec_shape(transformed, &out_shape)
    }

    /// Generate random values from a multivariate normal distribution with rotation
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean vector
    /// * `cov` - Covariance matrix
    /// * `size` - Optional shape of the output array
    /// * `rotation` - Optional rotation matrix
    ///
    /// # Returns
    ///
    /// An array of random values from the multivariate normal distribution with rotation
    pub fn multivariate_normal_with_rotation<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: &[T],
        cov: &Array<T>,
        size: Option<&[usize]>,
        rotation: Option<&Array<T>>,
    ) -> Result<Array<T>> {
        if mean.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Mean vector cannot be empty".to_string(),
            ));
        }

        let n = mean.len();
        let cov_shape = cov.shape();

        if cov_shape.len() != 2 || cov_shape[0] != n || cov_shape[1] != n {
            return Err(NumRs2Error::InvalidOperation(
                format!("Covariance matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, cov_shape)
            ));
        }

        // Check rotation matrix if provided
        if let Some(rot) = rotation {
            let rot_shape = rot.shape();
            if rot_shape.len() != 2 || rot_shape[0] != n || rot_shape[1] != n {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Rotation matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, rot_shape)
                ));
            }
        }

        // First determine the output shape
        let mut out_shape = Vec::new();
        if let Some(size_shape) = size {
            out_shape.extend_from_slice(size_shape);
        }
        out_shape.push(n);

        let total_samples: usize = if out_shape.len() > 1 {
            out_shape[..out_shape.len() - 1].iter().product()
        } else {
            1
        };

        // Generate standard normal samples
        let mut result = Vec::with_capacity(total_samples * n);
        let mut rng = self.get_rng()?;
        let standard_normal = Normal::new(0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create normal distribution: {}", e))
        })?;

        // Generate samples from standard normal distribution
        for _ in 0..total_samples * n {
            let val_f64: f64 = rng.sample(standard_normal);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert standard normal sample".to_string(),
                )
            })?;
            result.push(val);
        }

        // Compute the Cholesky decomposition of the covariance matrix
        let chol = cholesky_decomposition::<T>(&cov.to_vec(), n)?;

        // Apply rotation if provided
        let transform = if let Some(rot) = rotation {
            // Multiply rotation by Cholesky factor
            let rot_data = rot.to_vec();
            let mut rotated_chol = vec![T::zero(); n * n];

            // Matrix multiplication: rot * chol
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        rotated_chol[i * n + j] =
                            rotated_chol[i * n + j] + rot_data[i * n + k] * chol[k * n + j];
                    }
                }
            }

            rotated_chol
        } else {
            chol
        };

        // Transform standard normal samples using the Cholesky factor or rotated factor
        let mut transformed = vec![T::zero(); total_samples * n];
        for i in 0..total_samples {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..=j {
                    sum = sum + transform[j * n + k] * result[i * n + k];
                }
                transformed[i * n + j] = sum + mean[j];
            }
        }

        Array::from_vec_shape(transformed, &out_shape)
    }

    /// Generate random values from a Laplace (double exponential) distribution
    pub fn laplace<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        loc: T,
        scale: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let loc_f64 = loc.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert location parameter to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale parameter to f64".to_string())
        })?;

        // Use the inverse CDF method for generating Laplace random variables
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Generate uniform random variable in (0, 1)
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            // Transform using inverse CDF
            let val_f64 = if u < 0.5 {
                loc_f64 + scale_f64 * (u * 2.0).ln()
            } else {
                loc_f64 - scale_f64 * ((1.0 - u) * 2.0).ln()
            };

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Laplace sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a Gumbel distribution
    pub fn gumbel<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        loc: T,
        scale: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let loc_f64 = loc.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert location parameter to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale parameter to f64".to_string())
        })?;

        // Use the inverse CDF method for generating Gumbel random variables
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Generate uniform random variable in (0, 1)
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            // Transform using inverse CDF: X = loc - scale * ln(-ln(U))
            let val_f64 = loc_f64 - scale_f64 * (-u.ln()).ln();

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Gumbel sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a logistic distribution
    pub fn logistic<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        loc: T,
        scale: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let loc_f64 = loc.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert location parameter to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale parameter to f64".to_string())
        })?;

        // Use the inverse CDF method for generating logistic random variables
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Generate uniform random variable in (0, 1)
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            // Transform using inverse CDF: X = loc + scale * ln(u / (1 - u))
            let val_f64 = loc_f64 + scale_f64 * (u / (1.0 - u)).ln();

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert logistic sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a rayleigh distribution
    pub fn rayleigh<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        scale: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale parameter to f64".to_string())
        })?;

        // Use the inverse CDF method for generating Rayleigh random variables
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Generate uniform random variable in (0, 1)
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            // Transform using inverse CDF: X = scale * sqrt(-2 * ln(1 - u))
            let val_f64 = scale_f64 * (-2.0 * (1.0 - u).ln()).sqrt();

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Rayleigh sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a Wald (inverse Gaussian) distribution
    pub fn wald<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: T,
        scale: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if mean <= T::zero() || scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Mean and scale parameters must be positive, got mean={}, scale={}",
                mean, scale
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mean_f64 = mean.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mean parameter to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale parameter to f64".to_string())
        })?;

        // Implementation uses the algorithm from:
        // https://www.r-project.org/conferences/DSC-2003/Proceedings/MichaelEJ.pdf
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Generate a standard normal random variable
            let standard_normal = Normal::new(0.0, 1.0).map_err(|e| {
                NumRs2Error::InvalidOperation(format!(
                    "Failed to create normal distribution: {}",
                    e
                ))
            })?;
            let z: f64 = rng.sample(standard_normal);

            // Calculate intermediate values
            let y = z * z;
            let x1 = mean_f64 + (mean_f64 * mean_f64 * y) / (2.0 * scale_f64)
                - (mean_f64 / (2.0 * scale_f64))
                    * ((4.0 * mean_f64 * scale_f64 * y) + (mean_f64 * mean_f64 * y * y)).sqrt();

            // Generate a uniform random variable
            let u = rng.random::<f64>();

            // Based on acceptance criteria, either use x1 or its reciprocal transformation
            let val_f64 = if u <= mean_f64 / (mean_f64 + x1) {
                x1
            } else {
                mean_f64 * mean_f64 / x1
            };

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Wald sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a negative binomial distribution
    pub fn negative_binomial<T: NumCast + Clone + Debug>(
        &self,
        n: f64,
        p: f64,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if n <= 0.0 || p <= 0.0 || p >= 1.0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Parameters must satisfy n > 0 and 0 < p < 1, got n={}, p={}",
                n, p
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Generate gamma random variable with shape=n and scale=(1-p)/p
        let gamma_dist = Gamma::new(n, (1.0 - p) / p).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
        })?;

        // Generate negative binomial using gamma-poisson mixture
        for _ in 0..size {
            let lambda: f64 = rng.sample(gamma_dist);

            // 2. Generate Poisson random variable with mean=lambda
            let poisson_dist = Poisson::new(lambda).map_err(|e| {
                NumRs2Error::InvalidOperation(format!(
                    "Failed to create poisson distribution: {}",
                    e
                ))
            })?;

            let val_f64: f64 = rng.sample(poisson_dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert negative binomial sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a geometric distribution
    pub fn geometric<T: NumCast + Clone + Debug>(
        &self,
        p: f64,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if p <= 0.0 || p > 1.0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Probability must be in (0, 1], got {}",
                p
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Generate geometric random variables using inverse transform method
        for _ in 0..size {
            // Generate uniform random variable in (0, 1)
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            // Geometric(p) = floor(log(U) / log(1-p)) + 1
            let val_u64 = (u.ln() / (1.0 - p).ln()).floor() as u64 + 1;

            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert geometric sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a multinomial distribution
    pub fn multinomial<T: NumCast + Clone + Debug>(
        &self,
        n: usize,
        pvals: &[f64],
        shape: Option<&[usize]>,
    ) -> Result<Array<T>> {
        if pvals.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Probability array cannot be empty".to_string(),
            ));
        }

        // Validate probabilities sum to approximately 1
        let p_sum: f64 = pvals.iter().sum();
        if (p_sum - 1.0).abs() > 1e-10 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Probabilities must sum to 1, got sum={}",
                p_sum
            )));
        }

        // Validate all probabilities are non-negative
        for &p in pvals {
            if p < 0.0 {
                return Err(NumRs2Error::InvalidOperation(
                    "All probabilities must be non-negative".to_string(),
                ));
            }
        }

        let k = pvals.len(); // number of categories

        // First determine the output shape
        let mut out_shape = Vec::new();
        if let Some(size_shape) = shape {
            out_shape.extend_from_slice(size_shape);
        }
        out_shape.push(k);

        let total_samples: usize = if out_shape.len() > 1 {
            out_shape[..out_shape.len() - 1].iter().product()
        } else {
            1
        };

        let mut result = Vec::with_capacity(total_samples * k);
        let mut rng = self.get_rng()?;

        // Generate multinomial samples
        for _ in 0..total_samples {
            let mut sample = vec![0u64; k];

            // Generate n samples, where each sample falls into a category based on pvals
            for _ in 0..n {
                let u = rng.random::<f64>();
                let mut cumsum = 0.0;

                for i in 0..k {
                    cumsum += pvals[i];
                    if u <= cumsum {
                        sample[i] += 1;
                        break;
                    }
                }
            }

            // Convert u64 to target type T
            for count in sample {
                let val = T::from(count).ok_or_else(|| {
                    NumRs2Error::InvalidOperation(
                        "Failed to convert multinomial sample to target type".to_string(),
                    )
                })?;
                result.push(val);
            }
        }

        Array::from_vec_shape(result, &out_shape)
    }

    /// Generate random values from a hypergeometric distribution
    pub fn hypergeometric<T: NumCast + Clone + Debug>(
        &self,
        ngood: usize,
        nbad: usize,
        nsample: usize,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if nsample > ngood + nbad {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Cannot sample {} from population of size {}",
                nsample,
                ngood + nbad
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Naive implementation using direct sampling
        for _ in 0..size {
            // Create a population with ngood ones and nbad zeros
            let mut population = vec![false; nbad];
            population.extend(vec![true; ngood]);

            // Shuffle the population
            population.shuffle(&mut *rng);

            // Count number of ones in the sample
            let count = population[..nsample].iter().filter(|&&x| x).count() as u64;

            let val = T::from(count).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert hypergeometric sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a zipf distribution
    pub fn zipf<T: NumCast + Clone + Debug>(&self, a: f64, shape: &[usize]) -> Result<Array<T>> {
        if a <= 1.0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Parameter a must be > 1.0, got {}",
                a
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Implementation of Zipf distribution using rejection sampling
        // Based on algorithm from Luc Devroye's book "Non-Uniform Random Variate Generation"
        let b = 2.0f64.powf(a - 1.0);

        for _ in 0..size {
            // Initialize x variable
            let mut x: u64;
            let mut t: f64;

            loop {
                // Generate uniform variables
                let u = rng.random::<f64>();
                let v = rng.random::<f64>();

                // Initial candidate
                x = (u.powf(-1.0 / (a - 1.0))) as u64;
                if x < 1 {
                    x = 1;
                }

                // Acceptance-rejection test
                t = (1.0 + 1.0 / x as f64).powf(a - 1.0);
                if v * x as f64 * (t - 1.0) / (b - 1.0) <= t / b {
                    break;
                }
            }

            let val = T::from(x).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert zipf sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a logseries distribution
    pub fn logseries<T: NumCast + Clone + Debug>(
        &self,
        p: f64,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if p <= 0.0 || p >= 1.0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Parameter p must be in (0, 1), got {}",
                p
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        // Implementation using rejection method
        let r = (-p / (1.0 - p)).ln();

        for _ in 0..size {
            let mut x: u64;

            loop {
                // Generate uniform random variable
                let u = rng.random::<f64>();

                // Generate geometric random variable
                let v = rng.random::<f64>();
                x = (1.0 + (v.ln() / r).floor()) as u64;

                // Accept with probability x/(x+1)
                if u <= x as f64 / (x as f64 + 1.0) {
                    break;
                }
            }

            let val = T::from(x).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert logseries sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
    }

    /// Generate random values from a multivariate t-distribution
    ///
    /// The multivariate t-distribution is a generalization of the Student's t-distribution
    /// to multiple dimensions. It's characterized by a mean vector, covariance matrix,
    /// and degrees of freedom parameter.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean vector (length n)
    /// * `cov` - Covariance matrix (n x n, must be positive definite)
    /// * `df` - Degrees of freedom (must be positive)
    /// * `size` - Optional shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random values from the multivariate t-distribution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Mean vector is empty
    /// - Covariance matrix is not square or doesn't match mean vector dimensions
    /// - Covariance matrix is not positive definite
    /// - Degrees of freedom is not positive
    pub fn multivariate_t<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: &[T],
        cov: &Array<T>,
        df: T,
        size: Option<&[usize]>,
    ) -> Result<Array<T>> {
        if mean.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Mean vector cannot be empty".to_string(),
            ));
        }

        if df <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom must be positive, got {}",
                df
            )));
        }

        let n = mean.len();
        let cov_shape = cov.shape();

        if cov_shape.len() != 2 || cov_shape[0] != n || cov_shape[1] != n {
            return Err(NumRs2Error::InvalidOperation(
                format!("Covariance matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, cov_shape)
            ));
        }

        // Multivariate t-distribution is generated as:
        // X = mu + Y / sqrt(U/nu)
        // where Y ~ N(0, Sigma) and U ~ chi-squared(nu) are independent

        // First determine the output shape
        let mut out_shape = Vec::new();
        if let Some(size_shape) = size {
            out_shape.extend_from_slice(size_shape);
        }
        out_shape.push(n);

        let total_samples: usize = if out_shape.len() > 1 {
            out_shape[..out_shape.len() - 1].iter().product()
        } else {
            1
        };

        // Generate samples from standard normal distribution
        let mut normal_samples = Vec::with_capacity(total_samples * n);
        let df_f64 = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string())
        })?;

        let normal_dist = Normal::new(0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create normal distribution: {}", e))
        })?;

        let chi_square_dist = ChiSquared::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create chi-square distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..total_samples * n {
            let val_f64: f64 = rng.sample(normal_dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert normal sample to target type".to_string(),
                )
            })?;
            normal_samples.push(val);
        }

        // Compute Cholesky decomposition of covariance matrix
        let chol = cholesky_decomposition::<T>(&cov.to_vec(), n)?;

        // Generate chi-square samples and transform
        let mut result = vec![T::zero(); total_samples * n];
        for i in 0..total_samples {
            // Generate chi-square sample
            let chi_val: f64 = rng.sample(chi_square_dist);
            let scale_factor = T::from((df_f64 / chi_val).sqrt()).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert scale factor to target type".to_string(),
                )
            })?;

            // Transform normal samples using Cholesky factor and scale
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..=j {
                    sum = sum + chol[j * n + k] * normal_samples[i * n + k];
                }
                result[i * n + j] = mean[j] + sum * scale_factor;
            }
        }

        Array::from_vec_shape(result, &out_shape)
    }

    /// Generate random values from a Wishart distribution
    ///
    /// The Wishart distribution is a generalization of the chi-squared distribution to
    /// positive-definite matrices. It's commonly used as the conjugate prior for the
    /// precision matrix (inverse covariance) in multivariate normal distributions.
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom (must be >= dimension of scale matrix)
    /// * `scale` - Scale matrix (positive definite, p x p)
    /// * `size` - Optional shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random positive-definite matrices from the Wishart distribution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Scale matrix is not square
    /// - Scale matrix is not positive definite
    /// - Degrees of freedom is less than the dimension
    pub fn wishart<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        df: T,
        scale: &Array<T>,
        size: Option<&[usize]>,
    ) -> Result<Array<T>> {
        let scale_shape = scale.shape();

        if scale_shape.len() != 2 || scale_shape[0] != scale_shape[1] {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale matrix must be square, got shape {:?}",
                scale_shape
            )));
        }

        let p = scale_shape[0];
        let df_val = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string())
        })?;

        if df_val < p as f64 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom ({}) must be >= dimension of scale matrix ({})",
                df_val, p
            )));
        }

        // Compute Cholesky decomposition of scale matrix
        let chol = cholesky_decomposition::<T>(&scale.to_vec(), p)?;

        // Determine output shape
        let mut out_shape = Vec::new();
        if let Some(size_shape) = size {
            out_shape.extend_from_slice(size_shape);
        }
        out_shape.push(p);
        out_shape.push(p);

        let total_samples: usize = if out_shape.len() > 2 {
            out_shape[..out_shape.len() - 2].iter().product()
        } else {
            1
        };

        let mut result = Vec::with_capacity(total_samples * p * p);

        // Use Bartlett decomposition method
        // Generate each Wishart matrix independently
        let mut rng = self.get_rng()?;
        for _ in 0..total_samples {
            // Create Bartlett matrix A (lower triangular)
            let mut a = vec![T::zero(); p * p];

            // Fill diagonal with chi-distributed values
            for i in 0..p {
                let chi_df = df_val - i as f64;
                let chi_dist = ChiSquared::new(chi_df).map_err(|e| {
                    NumRs2Error::InvalidOperation(format!(
                        "Failed to create chi-square distribution: {}",
                        e
                    ))
                })?;
                let chi_val: f64 = rng.sample(chi_dist);
                a[i * p + i] = T::from(chi_val.sqrt()).ok_or_else(|| {
                    NumRs2Error::InvalidOperation("Failed to convert chi-square sample".to_string())
                })?;
            }

            // Fill lower triangle with standard normal values
            let normal_dist = Normal::new(0.0, 1.0).map_err(|e| {
                NumRs2Error::InvalidOperation(format!(
                    "Failed to create normal distribution: {}",
                    e
                ))
            })?;

            for i in 1..p {
                for j in 0..i {
                    let norm_val: f64 = rng.sample(normal_dist);
                    a[i * p + j] = T::from(norm_val).ok_or_else(|| {
                        NumRs2Error::InvalidOperation("Failed to convert normal sample".to_string())
                    })?;
                }
            }

            // Compute L * A where L is Cholesky factor of scale matrix
            let mut la = vec![T::zero(); p * p];
            for i in 0..p {
                for j in 0..=i {
                    let mut sum = T::zero();
                    for k in 0..=j.min(i) {
                        sum = sum + chol[i * p + k] * a[j * p + k];
                    }
                    la[i * p + j] = sum;
                }
            }

            // Compute result = L * A * A^T * L^T = (L * A) * (L * A)^T
            for i in 0..p {
                for j in 0..=i {
                    let mut sum = T::zero();
                    for k in 0..p {
                        sum = sum + la[i * p + k] * la[j * p + k];
                    }
                    result.push(sum);
                    if i != j {
                        result.push(sum); // Symmetric matrix
                    }
                }
            }
        }

        Array::from_vec_shape(result, &out_shape)
    }

    /// Generate random values from a Frechet distribution
    ///
    /// The Frechet distribution (Type II extreme value distribution) is used in extreme
    /// value theory to model the maximum of a large sample of random variables.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (alpha, must be positive)
    /// * `loc` - Location parameter
    /// * `scale` - Scale parameter (must be positive)
    /// * `output_shape` - Shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random values from the Frechet distribution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Shape parameter is not positive
    /// - Scale parameter is not positive
    pub fn frechet<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape: T,
        loc: T,
        scale: T,
        output_shape: &[usize],
    ) -> Result<Array<T>> {
        if shape <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Shape parameter must be positive, got {}",
                shape
            )));
        }

        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = output_shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        let shape_f64 = shape.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
        })?;
        let loc_f64 = loc.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert location to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;

        // Use inverse CDF method: F^{-1}(u) = loc + scale * (-ln(u))^{-1/alpha}
        for _ in 0..size {
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            let val_f64 = loc_f64 + scale_f64 * (-u.ln()).powf(-1.0 / shape_f64);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Frechet sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, output_shape)
    }

    /// Generate random values from a Generalized Extreme Value (GEV) distribution
    ///
    /// The GEV distribution combines three types of extreme value distributions:
    /// - Type I (Gumbel): xi = 0
    /// - Type II (Frechet): xi > 0
    /// - Type III (Weibull): xi < 0
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (xi)
    /// * `loc` - Location parameter (mu)
    /// * `scale` - Scale parameter (sigma, must be positive)
    /// * `output_shape` - Shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random values from the GEV distribution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Scale parameter is not positive
    pub fn gev<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape: T,
        loc: T,
        scale: T,
        output_shape: &[usize],
    ) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Scale parameter must be positive, got {}",
                scale
            )));
        }

        let size: usize = output_shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        let xi = shape.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
        })?;
        let mu = loc.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert location to f64".to_string())
        })?;
        let sigma = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;

        let xi_threshold = 1e-10; // Threshold for considering xi approximately 0

        // Use inverse CDF method
        for _ in 0..size {
            let u = loop {
                let v = rng.random::<f64>();
                if v > 0.0 && v < 1.0 {
                    break v;
                }
            };

            let val_f64 = if xi.abs() < xi_threshold {
                // Type I (Gumbel): x = mu - sigma * ln(-ln(u))
                mu - sigma * (-u.ln()).ln()
            } else {
                // Type II (Frechet, xi > 0) or Type III (Weibull, xi < 0):
                // x = mu + sigma * ((-ln(u))^{-xi} - 1) / xi
                mu + sigma * ((-u.ln()).powf(-xi) - 1.0) / xi
            };

            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert GEV sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, output_shape)
    }
}

/// Compute Cholesky decomposition of a symmetric positive-definite matrix
///
/// Given a flat row-major matrix of size n x n, computes the lower triangular
/// matrix L such that A = L * L^T.
///
/// # Arguments
///
/// * `matrix_data` - Flat row-major matrix data
/// * `n` - Dimension of the matrix
///
/// # Returns
///
/// The lower triangular Cholesky factor as a flat row-major vector
fn cholesky_decomposition<T: Float + NumCast + Clone + Debug + Display>(
    matrix_data: &[T],
    n: usize,
) -> Result<Vec<T>> {
    let mut chol = vec![T::zero(); n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::zero();
            for k in 0..j {
                sum = sum + chol[i * n + k] * chol[j * n + k];
            }

            if i == j {
                let val = matrix_data[i * n + i] - sum;
                if val <= T::zero() {
                    return Err(NumRs2Error::InvalidOperation(
                        "Covariance matrix is not positive definite".to_string(),
                    ));
                }
                chol[i * n + j] = val.sqrt();
            } else {
                chol[i * n + j] = T::one() / chol[j * n + j] * (matrix_data[i * n + j] - sum);
            }
        }
    }

    Ok(chol)
}

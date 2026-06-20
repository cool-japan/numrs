//! Random state for compatibility with different random number generator states
//!
//! This module provides the RandomState struct, which is a wrapper around
//! different types of random number generators. This is similar to the RandomState
//! in NumPy's random module.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, ToPrimitive};
// SCIRS2 POLICY COMPLIANT imports - always use SciRS2
use scirs2_core::ndarray::distributions::{
    uniform::SampleUniform, Distribution as NdArrayDistribution,
};
use scirs2_core::random::prelude::*;
use scirs2_core::SliceRandomExt;
use scirs2_core::{Distribution as CoreDistribution, Pert};
use scirs2_stats::distributions::{
    f::F as FDistribution, lognormal::Lognormal, Bernoulli, Beta, Binomial, Cauchy, ChiSquare,
    Exponential, Gamma, Pareto, Poisson, StudentT, Weibull,
};
use scirs2_stats::Distribution;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Sample from a triangular distribution using inverse CDF
fn sample_triangular(low: f64, mode: f64, high: f64, u: f64) -> f64 {
    let fc = (mode - low) / (high - low);

    if u < fc {
        low + ((high - low) * (mode - low) * u).sqrt()
    } else {
        high - ((high - low) * (high - mode) * (1.0 - u)).sqrt()
    }
}

/// RandomState for managing the state of random number generators
///
/// This struct is a wrapper around different types of random number generators.
/// In the current implementation, it uses StdRng, but can be extended to support
/// other RNG types in the future.
pub struct RandomState {
    rng: Arc<Mutex<StdRng>>,
}

impl RandomState {
    /// Create a new RandomState with a random seed
    pub fn new() -> Self {
        // Use current time as seed if none provided
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(1));

        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(now.as_secs()))),
        }
    }

    /// Create a new RandomState with the given seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }

    /// Get a locked reference to the RNG
    pub fn get_rng(&self) -> Result<std::sync::MutexGuard<'_, StdRng>> {
        match self.rng.lock() {
            Ok(guard) => Ok(guard),
            Err(poisoned) => {
                // If the lock is poisoned, we can still recover by getting the guard
                // This allows the random number generation to continue working even after a panic
                Ok(poisoned.into_inner())
            }
        }
    }

    /// Generate uniform random values in [0, 1)
    pub fn random<T>(&self, shape: &[usize]) -> Result<Array<T>>
    where
        T: Clone
            + scirs2_core::ndarray::distributions::uniform::SampleUniform
            + num_traits::NumCast,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let rng = self.get_rng()?;

        for _ in 0..size {
            // Use SciRS2 uniform distribution for [0, 1) and convert to T
            let uniform_dist = scirs2_stats::distributions::Uniform::new(0.0f64, 1.0f64)
                .expect("random: uniform distribution [0, 1) should always be valid");
            let val_f64 = uniform_dist.rvs(1).expect("uniform sampling failed")[0];
            let val = num_traits::NumCast::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert uniform sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate integers in the range [low, high)
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive)
    /// * `shape` - Shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random integers.
    pub fn integers<
        T: Clone + PartialOrd + SampleUniform + Into<i64> + TryFrom<i64> + ToPrimitive,
    >(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>>
    where
        <T as TryFrom<i64>>::Error: Debug,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        for _ in 0..size {
            // Convert bounds to f64, use SciRS2 uniform distribution, then convert back
            let low_f64 = low
                .clone()
                .into()
                .to_f64()
                .expect("integers: low bound should be convertible to f64");
            let high_f64 = high
                .clone()
                .into()
                .to_f64()
                .expect("integers: high bound should be convertible to f64");

            let uniform_dist = scirs2_stats::distributions::Uniform::new(low_f64, high_f64)
                .expect("integers: uniform distribution should be valid for given bounds");
            let val_f64 = uniform_dist.rvs(1).expect("uniform sampling failed")[0];
            let val_i64 = val_f64.floor() as i64;
            let val = T::try_from(val_i64).map_err(|_| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert integer sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a normal (Gaussian) distribution
    pub fn normal<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: T,
        std: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if std <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Standard deviation must be positive, got {}",
                std
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mean_f64 = mean.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mean to f64".to_string())
        })?;
        let std_f64 = std.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert std to f64".to_string())
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            // Box-Muller transform: drive sampling through the seeded RNG for reproducibility.
            // Guard against u1 == 0.0 which would produce ln(0) = -∞.
            let u1 = {
                let mut u = rng.random::<f64>();
                while u == 0.0 {
                    u = rng.random::<f64>();
                }
                u
            };
            let u2 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let val_f64 = mean_f64 + std_f64 * z;
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert normal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a log-normal distribution
    pub fn lognormal<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        mean: T,
        sigma: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if sigma <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Sigma must be positive, got {}",
                sigma
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mean_f64 = mean.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mean to f64".to_string())
        })?;
        let sigma_f64 = sigma.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert sigma to f64".to_string())
        })?;

        let dist = Lognormal::new(mean_f64, sigma_f64, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create log-normal distribution: {}",
                e
            ))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert lognormal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Beta distribution
    pub fn beta<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        a: T,
        b: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if a <= T::zero() || b <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Alpha and Beta parameters must be positive, got alpha={}, beta={}",
                a, b
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let a_f64 = a.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert alpha parameter to f64".to_string())
        })?;
        let b_f64 = b.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert beta parameter to f64".to_string())
        })?;

        let dist = Beta::new(a_f64, b_f64, 0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create beta distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert beta sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Chi-Square distribution
    pub fn chisquare<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        df: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if df <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom must be positive, got {}",
                df
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let df_f64 = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert degrees of freedom to f64".to_string())
        })?;

        let dist = ChiSquare::new(df_f64, 0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create chi-square distribution: {}",
                e
            ))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert chi-square sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Dirichlet distribution
    pub fn dirichlet<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        alpha: &[T],
        shape: &[usize],
    ) -> Result<Array<T>> {
        if alpha.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Alpha parameter must have at least one value".to_string(),
            ));
        }

        for &a in alpha {
            if a <= T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "All alpha parameters must be positive".to_string(),
                ));
            }
        }

        let size: usize = shape.iter().product();
        let k = alpha.len();
        let mut result = Vec::with_capacity(size * k);

        let alpha_f64: Vec<f64> = alpha
            .iter()
            .map(|&a| {
                a.to_f64().ok_or_else(|| {
                    NumRs2Error::InvalidOperation(
                        "Failed to convert alpha parameter to f64".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        let rng = self.get_rng()?;

        // Implement Dirichlet using gamma distribution sampling
        // A Dirichlet sample is generated by:
        // 1. Sample X_i ~ Gamma(alpha_i, 1) for each i
        // 2. Return [X_1/S, X_2/S, ..., X_k/S] where S = X_1 + X_2 + ... + X_k
        for _ in 0..size {
            let mut sample = Vec::with_capacity(k);
            let mut sum = 0.0;

            // Generate gamma samples for each component
            for &a in &alpha_f64 {
                let gamma = Gamma::new(a, 1.0, 0.0).map_err(|e| {
                    NumRs2Error::InvalidOperation(format!(
                        "Failed to create gamma distribution: {}",
                        e
                    ))
                })?;

                let gamma_sample = gamma.rvs(1).expect("gamma sampling failed")[0];
                sum += gamma_sample;
                sample.push(gamma_sample);
            }

            // Normalize to get a Dirichlet sample
            for val_f64 in sample {
                let normalized = val_f64 / sum;
                let val = T::from(normalized).ok_or_else(|| {
                    NumRs2Error::InvalidOperation(
                        "Failed to convert Dirichlet sample to target type".to_string(),
                    )
                })?;
                result.push(val);
            }
        }

        // Reshape to include the k dimension
        let mut out_shape = shape.to_vec();
        out_shape.push(k);

        Ok(Array::from_vec(result).reshape(&out_shape))
    }

    /// Generate random values from a Student's t-distribution
    pub fn student_t<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        df: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if df <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Degrees of freedom must be positive, got {}",
                df
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let df_f64 = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert degrees of freedom to f64".to_string())
        })?;

        let dist = StudentT::new(df_f64, 0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create Student's t-distribution: {}",
                e
            ))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Student's t sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Poisson distribution
    pub fn poisson<T: NumCast + Clone + Debug>(
        &self,
        lam: f64,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if lam <= 0.0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Lambda must be positive, got {}",
                lam
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let dist = Poisson::new(lam, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Poisson distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_u64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Poisson sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Binomial distribution
    pub fn binomial<T: NumCast + Clone + Debug>(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if !(0.0..=1.0).contains(&p) {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Probability must be in [0, 1], got {}",
                p
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let dist = Binomial::new(n as usize, p).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Binomial distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_u64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Binomial sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Cauchy (Lorentz) distribution
    pub fn cauchy<T: Float + NumCast + Clone + Debug + Display>(
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

        let dist = Cauchy::new(loc_f64, scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Cauchy distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Cauchy sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a uniform distribution
    pub fn uniform<T: Clone + PartialOrd + SampleUniform + ToPrimitive + NumCast>(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        use scirs2_core::random::Rng;

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        // Use the RandomState's RNG for reproducibility
        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to lock RNG".to_string()))?;

        // Convert to f64 for scirs2_core compatibility
        let low_f64 = low.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert low bound to f64".to_string())
        })?;
        let high_f64 = high.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert high bound to f64".to_string())
        })?;

        for _ in 0..size {
            let val_f64 = rng.gen_range(low_f64..high_f64);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert uniform sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate binary random values with given probability of success
    pub fn bernoulli<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        p: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if p < T::zero() || p > T::one() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Probability must be in [0, 1], got {}",
                p
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let p_f64 = p.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert probability to f64".to_string())
        })?;

        let dist = Bernoulli::new(p_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Bernoulli distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val_bool = val_f64 > 0.5; // Convert SciRS2 f64 result to bool
            let val = if val_bool { T::one() } else { T::zero() };
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a gamma distribution
    pub fn gamma<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape_param: T,
        scale: T,
        size_shape: &[usize],
    ) -> Result<Array<T>> {
        if shape_param <= T::zero() || scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Shape and scale parameters must be positive, got shape={}, scale={}",
                shape_param, scale
            )));
        }

        let arr_size: usize = size_shape.iter().product();
        let mut vec = Vec::with_capacity(arr_size);
        let shape_f64 = shape_param.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;

        // Gamma distribution parameters: (shape, scale, location)
        let dist = Gamma::new(shape_f64, scale_f64, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..arr_size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert gamma sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(size_shape))
    }

    /// Generate random values from an exponential distribution
    pub fn exponential<T: Float + NumCast + Clone + Debug + Display>(
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
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;

        // CORRECTED: SciRS2 Exponential::new(rate, location) expects rate = 1/scale
        // For exponential distribution with scale s: rate = 1/s, mean = s, variance = s²
        let rate = 1.0 / scale_f64;
        let dist = Exponential::new(rate, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create exponential distribution: {}",
                e
            ))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert exponential sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Weibull distribution
    pub fn weibull<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape_param: T,
        scale: T,
        size_shape: &[usize],
    ) -> Result<Array<T>> {
        if shape_param <= T::zero() || scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Shape and scale parameters must be positive, got shape={}, scale={}",
                shape_param, scale
            )));
        }

        let arr_size: usize = size_shape.iter().product();
        let mut vec = Vec::with_capacity(arr_size);
        let shape_f64 = shape_param.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert shape to f64".to_string())
        })?;
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;

        let dist = Weibull::new(shape_f64, scale_f64, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Weibull distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..arr_size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Weibull sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(size_shape))
    }

    /// Shuffle an array in-place
    pub fn shuffle<T: Clone>(&self, array: &mut Array<T>) -> Result<()> {
        let rng = self.get_rng()?;

        let mut data = array.to_vec();
        data.shuffle(&mut thread_rng());

        // Update the array with shuffled data
        let shape = array.shape();
        *array = Array::from_vec(data).reshape(&shape);

        Ok(())
    }

    /// Random choice from elements in an array
    pub fn choice<T: Clone>(
        &self,
        array: &Array<T>,
        size: Option<usize>,
        replace: Option<bool>,
    ) -> Result<Array<T>> {
        let data = array.to_vec();
        if data.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot choose from an empty array".to_string(),
            ));
        }

        let choose_size = size.unwrap_or(1);
        let with_replacement = replace.unwrap_or(true);

        if !with_replacement && choose_size > data.len() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Cannot choose {} items without replacement from array of size {}",
                choose_size,
                data.len()
            )));
        }

        let mut rng = self.get_rng()?;

        let mut result = Vec::with_capacity(choose_size);

        if with_replacement {
            // Sample with replacement
            for _ in 0..choose_size {
                let idx = rng.random_range(0..data.len());
                result.push(data[idx].clone());
            }
        } else {
            // Sample without replacement
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.shuffle(&mut thread_rng());

            for i in 0..choose_size {
                result.push(data[indices[i]].clone());
            }
        }

        if size.is_none() {
            // Return a single element, not an array
            Ok(Array::from_vec(result))
        } else {
            // Return an array of chosen elements
            Ok(Array::from_vec(result))
        }
    }

    /// Generate a permutation of integers from 0 to n-1
    pub fn permutation<T: NumCast + Clone>(&self, n: usize) -> Result<Array<T>> {
        let rng = self.get_rng()?;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut thread_rng());

        let mut result = Vec::with_capacity(n);
        for idx in indices {
            let val = T::from(idx).ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert index to target type".to_string())
            })?;
            result.push(val);
        }

        Ok(Array::from_vec(result))
    }

    /// Generate a standard normal distribution
    pub fn standard_normal<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape: &[usize],
    ) -> Result<Array<T>> {
        self.normal(T::zero(), T::one(), shape)
    }

    /// Generate random values from a Pareto distribution
    pub fn pareto<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        alpha: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if alpha <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Alpha parameter must be positive, got {}",
                alpha
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let alpha_f64 = alpha.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert alpha parameter to f64".to_string())
        })?;

        // Pareto::new(shape, scale, loc) - alpha is the shape parameter
        // NumPy uses scale=1.0 by default to match standard Pareto Type I
        let dist = Pareto::new(alpha_f64, 1.0, 0.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Pareto distribution: {}", e))
        })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.rvs(1).expect("distribution sampling failed")[0];
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Pareto sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a Triangular distribution
    pub fn triangular<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        low: T,
        mode: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if low > mode || mode > high || low > high {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Parameters must satisfy low <= mode <= high, got low={}, mode={}, high={}",
                low, mode, high
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let low_f64 = low.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert low parameter to f64".to_string())
        })?;
        let mode_f64 = mode.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mode parameter to f64".to_string())
        })?;
        let high_f64 = high.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert high parameter to f64".to_string())
        })?;

        // Implement triangular distribution using inverse CDF method
        // This avoids issues with rand_distr::Triangular
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let u = rng.random::<f64>();
            let val_f64 = sample_triangular(low_f64, mode_f64, high_f64, u);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert triangular sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a PERT distribution
    pub fn pert<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        min: T,
        mode: T,
        max: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if min > mode || mode > max || min > max {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Parameters must satisfy min <= mode <= max, got min={}, mode={}, max={}",
                min, mode, max
            )));
        }

        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let min_f64 = min.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert min parameter to f64".to_string())
        })?;
        let mode_f64 = mode.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mode parameter to f64".to_string())
        })?;
        let max_f64 = max.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert max parameter to f64".to_string())
        })?;

        let dist = Pert::new(min_f64, max_f64)
            .with_mode(mode_f64)
            .map_err(|e| {
                NumRs2Error::InvalidOperation(format!("Failed to create PERT distribution: {}", e))
            })?;

        let rng = self.get_rng()?;

        for _ in 0..size {
            let mut temp_rng = thread_rng();
            let val_f64 = dist.sample(&mut temp_rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert PERT sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random samples from an F (Fisher-Snedecor) distribution.
    ///
    /// The F distribution arises as the ratio of two independent chi-squared
    /// random variables divided by their respective degrees of freedom.  It is
    /// widely used in ANOVA and regression analysis.
    ///
    /// # Arguments
    ///
    /// * `dfnum` - Numerator degrees of freedom (must be > 0)
    /// * `dfden` - Denominator degrees of freedom (must be > 0)
    /// * `shape` - Shape of the output array
    pub fn f_dist<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        dfnum: T,
        dfden: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        if dfnum <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Numerator degrees of freedom must be positive, got {}",
                dfnum
            )));
        }
        if dfden <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Denominator degrees of freedom must be positive, got {}",
                dfden
            )));
        }

        let dfnum_f64 = dfnum.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation(
                "Failed to convert numerator degrees of freedom to f64".to_string(),
            )
        })?;
        let dfden_f64 = dfden.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation(
                "Failed to convert denominator degrees of freedom to f64".to_string(),
            )
        })?;

        let dist = FDistribution::<f64>::new(dfnum_f64, dfden_f64, 0.0, 1.0).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create F distribution: {}", e))
        })?;

        let _rng = self.get_rng()?;
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        for _ in 0..size {
            let samples = dist.rvs(1).map_err(|e| {
                NumRs2Error::InvalidOperation(format!(
                    "Failed to sample from F distribution: {}",
                    e
                ))
            })?;
            let val = T::from(samples[0]).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert F distribution sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }
}

impl Default for RandomState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_state_random() {
        let rng = RandomState::with_seed(42);
        let arr = rng
            .random::<f64>(&[3, 3])
            .expect("test: random should succeed");

        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_random_state_normal() {
        let rng = RandomState::new();
        let arr = rng
            .normal(0.0, 1.0, &[10])
            .expect("test: normal should succeed");

        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_random_state_beta() {
        let rng = RandomState::new();
        let arr = rng.beta(2.0, 5.0, &[5]).expect("test: beta should succeed");

        assert_eq!(arr.shape(), vec![5]);

        // Beta values should be between 0 and 1
        for val in arr.to_vec() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_random_state_dirichlet() {
        let rng = RandomState::new();
        let alpha = vec![1.0, 1.0, 1.0];
        let arr = rng
            .dirichlet::<f64>(&alpha, &[2])
            .expect("test: dirichlet should succeed");

        // Shape should be [2, 3] as each sample has 3 values
        assert_eq!(arr.shape(), vec![2, 3]);

        // Each row should sum to approximately 1.0
        let data = arr.to_vec();
        assert!((data[0] + data[1] + data[2] - 1.0).abs() < 1e-10);
        assert!((data[3] + data[4] + data[5] - 1.0).abs() < 1e-10);
    }
}

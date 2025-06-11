//! Random state for compatibility with different random number generator states
//!
//! This module provides the RandomState struct, which is a wrapper around
//! different types of random number generators. This is similar to the RandomState
//! in NumPy's random module.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Bernoulli, Distribution, Exp as Exponential, Gamma, LogNormal, Normal, Uniform};
use rand_distr::{Beta, Binomial, Cauchy, ChiSquared as ChiSquare, Poisson, StudentT, Weibull};
use rand_distr::{Pareto, Pert, StandardNormal, Triangular};
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
            .expect("Time went backwards");

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
        self.rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))
    }

    /// Generate uniform random values in [0, 1)
    pub fn random<T>(&self, shape: &[usize]) -> Result<Array<T>>
    where
        T: Clone,
        rand_distr::StandardUniform: rand_distr::Distribution<T>,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self.get_rng()?;

        for _ in 0..size {
            vec.push(rng.random::<T>());
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
    pub fn integers<T: Clone + PartialOrd + SampleUniform + Into<i64> + TryFrom<i64>>(
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

        let dist = Uniform::new_inclusive(low, high).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create uniform integer distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            vec.push(dist.sample(&mut *rng));
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

        let dist = Normal::new(mean_f64, std_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create normal distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = LogNormal::new(mean_f64, sigma_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create log-normal distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = Beta::new(a_f64, b_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create beta distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = ChiSquare::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create chi-square distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let mut rng = self.get_rng()?;

        // Implement Dirichlet using gamma distribution sampling
        // A Dirichlet sample is generated by:
        // 1. Sample X_i ~ Gamma(alpha_i, 1) for each i
        // 2. Return [X_1/S, X_2/S, ..., X_k/S] where S = X_1 + X_2 + ... + X_k
        for _ in 0..size {
            let mut sample = Vec::with_capacity(k);
            let mut sum = 0.0;

            // Generate gamma samples for each component
            for &a in &alpha_f64 {
                let gamma = rand_distr::Gamma::new(a, 1.0).map_err(|e| {
                    NumRs2Error::InvalidOperation(format!(
                        "Failed to create gamma distribution: {}",
                        e
                    ))
                })?;

                let gamma_sample = gamma.sample(&mut *rng);
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

        let dist = StudentT::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create Student's t-distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = Poisson::new(lam).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Poisson distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_u64 = dist.sample(&mut *rng);
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

        let dist = Binomial::new(n, p).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Binomial distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_u64 = dist.sample(&mut *rng);
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

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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
    pub fn uniform<T: Clone + PartialOrd + SampleUniform>(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let dist = Uniform::new_inclusive(low, high).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create uniform distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            vec.push(dist.sample(&mut *rng));
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

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_bool = dist.sample(&mut *rng);
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

        let dist = Gamma::new(shape_f64, scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..arr_size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = Exponential::new(1.0 / scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create exponential distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = Weibull::new(shape_f64, scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Weibull distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..arr_size {
            let val_f64 = dist.sample(&mut *rng);
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
        let mut rng = self.get_rng()?;

        let mut data = array.to_vec();
        data.shuffle(&mut *rng);

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
            indices.shuffle(&mut *rng);

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
        let mut rng = self.get_rng()?;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut *rng);

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

        let dist = Pareto::new(1.0, alpha_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Pareto distribution: {}", e))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let dist = Triangular::new(low_f64, mode_f64, high_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create triangular distribution: {}",
                e
            ))
        })?;

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
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

        let mut rng = self.get_rng()?;

        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert PERT sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

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

        // Generate samples from standard normal distribution
        for _ in 0..total_samples * n {
            let val_f64: f64 = StandardNormal.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert standard normal sample".to_string(),
                )
            })?;
            result.push(val);
        }

        // We need to compute the Cholesky decomposition of the covariance matrix
        // This is a simplified implementation that assumes the covariance matrix is positive definite
        // In a real implementation, we would use a proper linear algebra library for this
        let mut chol = vec![T::zero(); n * n];
        let cov_data = cov.to_vec();

        // Compute Cholesky decomposition (L such that Σ = L·L^T)
        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + chol[i * n + k] * chol[j * n + k];
                }

                if i == j {
                    let val = cov_data[i * n + i] - sum;
                    if val <= T::zero() {
                        return Err(NumRs2Error::InvalidOperation(
                            "Covariance matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[i * n + j] = val.sqrt();
                } else {
                    chol[i * n + j] = T::one() / chol[j * n + j] * (cov_data[i * n + j] - sum);
                }
            }
        }

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

        Ok(Array::from_vec(transformed).reshape(&out_shape))
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

        // Generate samples from standard normal distribution
        for _ in 0..total_samples * n {
            let val_f64: f64 = StandardNormal.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert standard normal sample".to_string(),
                )
            })?;
            result.push(val);
        }

        // We need to compute the Cholesky decomposition of the covariance matrix
        // This is a simplified implementation that assumes the covariance matrix is positive definite
        // In a real implementation, we would use a proper linear algebra library for this
        let mut chol = vec![T::zero(); n * n];
        let cov_data = cov.to_vec();

        // Compute Cholesky decomposition (L such that Σ = L·L^T)
        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + chol[i * n + k] * chol[j * n + k];
                }

                if i == j {
                    let val = cov_data[i * n + i] - sum;
                    if val <= T::zero() {
                        return Err(NumRs2Error::InvalidOperation(
                            "Covariance matrix is not positive definite".to_string(),
                        ));
                    }
                    chol[i * n + j] = val.sqrt();
                } else {
                    chol[i * n + j] = T::one() / chol[j * n + j] * (cov_data[i * n + j] - sum);
                }
            }
        }

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

        Ok(Array::from_vec(transformed).reshape(&out_shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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
            let z: f64 = StandardNormal.sample(&mut *rng);

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

        Ok(Array::from_vec(vec).reshape(shape))
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

        // Generate negative binomial using gamma-poisson mixture
        for _ in 0..size {
            // 1. Generate gamma random variable with shape=n and scale=(1-p)/p
            let gamma_dist = rand_distr::Gamma::new(n, (1.0 - p) / p).map_err(|e| {
                NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
            })?;

            let lambda = gamma_dist.sample(&mut *rng);

            // 2. Generate Poisson random variable with mean=lambda
            let poisson_dist = rand_distr::Poisson::new(lambda).map_err(|e| {
                NumRs2Error::InvalidOperation(format!(
                    "Failed to create poisson distribution: {}",
                    e
                ))
            })?;

            let val_u64 = poisson_dist.sample(&mut *rng);
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert negative binomial sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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
            // No need to track remaining samples in this implementation
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

            // Alternative approach using binomial distribution (more efficient for large n)
            // if n is large, we can use sequential binomial sampling
            /*
            let mut prob_remaining = 1.0;
            for i in 0..k-1 {
                if prob_remaining <= 0.0 {
                    break;
                }

                let p_adj = pvals[i] / prob_remaining;
                let dist = Binomial::new(remaining_samples as u64, p_adj).map_err(|e| {
                    NumRs2Error::InvalidOperation(format!("Failed to create binomial distribution: {}", e))
                })?;

                let count = dist.sample(&mut *rng);
                sample[i] = count;

                remaining_samples -= count as usize;
                prob_remaining -= pvals[i];
            }
            sample[k-1] = remaining_samples as u64;
            */

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

        Ok(Array::from_vec(result).reshape(&out_shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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

        Ok(Array::from_vec(vec).reshape(shape))
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
        let arr = rng.random::<f64>(&[3, 3]).unwrap();

        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_random_state_normal() {
        let rng = RandomState::new();
        let arr = rng.normal(0.0, 1.0, &[10]).unwrap();

        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_random_state_beta() {
        let rng = RandomState::new();
        let arr = rng.beta(2.0, 5.0, &[5]).unwrap();

        assert_eq!(arr.shape(), vec![5]);

        // Beta values should be between 0 and 1
        for val in arr.to_vec() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_random_state_dirichlet() {
        let rng = RandomState::new();
        let alpha = vec![1.0, 1.0, 1.0];
        let arr = rng.dirichlet::<f64>(&alpha, &[2]).unwrap();

        // Shape should be [2, 3] as each sample has 3 values
        assert_eq!(arr.shape(), vec![2, 3]);

        // Each row should sum to approximately 1.0
        let data = arr.to_vec();
        assert!((data[0] + data[1] + data[2] - 1.0).abs() < 1e-10);
        assert!((data[3] + data[4] + data[5] - 1.0).abs() < 1e-10);
    }
}

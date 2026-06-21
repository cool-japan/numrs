//! Legacy random number generation interface
//!
//! This module provides backward compatibility for the original random_base.rs functionality.
//! It maintains the same API while being integrated into the unified random module structure.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
// Note: Legacy module - may need fallback to rand for compatibility
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::prelude::*;
use scirs2_core::random::Exp;
use scirs2_core::SliceRandomExt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A Generator for random number streams (legacy interface)
pub struct Generator {
    rng: Arc<Mutex<StdRng>>,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new()
    }
}

impl Generator {
    /// Create a new generator with a thread-local RNG
    pub fn new() -> Self {
        // Use current time as seed if none provided
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(1));

        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(now.as_secs()))),
        }
    }

    /// Create a new generator with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }

    /// Generate uniform random values in [0, 1)
    pub fn random<T>(&self, shape: &[usize]) -> Result<Array<T>>
    where
        T: Clone + SampleUniform + NumCast,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            let val_f64: f64 = rng.random::<f64>();
            let val = NumCast::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert uniform sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate normal (Gaussian) random values
    pub fn normal<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
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

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert normal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate log-normal random values
    pub fn lognormal<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
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

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert lognormal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a uniform distribution
    pub fn uniform<T: Clone + PartialOrd + SampleUniform + Float + NumCast + std::fmt::Display>(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            vec.push(rng.random_range(low..high));
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate binary random values with given probability of success
    pub fn bernoulli<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
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

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            let val_bool: bool = rng.sample(dist);
            let val = if val_bool { T::one() } else { T::zero() };
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a gamma distribution
    pub fn gamma<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
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

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..arr_size {
            let val_f64: f64 = rng.sample(dist);
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
    pub fn exponential<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
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
        let dist = Exp::new(rate).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create exponential distribution: {}",
                e
            ))
        })?;

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert exponential sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Shuffle an array in-place
    pub fn shuffle<T: Clone>(&self, array: &mut Array<T>) -> Result<()> {
        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

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

        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

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
        let mut rng = self
            .rng
            .lock()
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string()))?;

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
}

// Global generator for convenience
lazy_static::lazy_static! {
    static ref GLOBAL_GENERATOR: Mutex<Generator> = Mutex::new(Generator::new());
}

/// Set the random seed for the global generator
pub fn seed(seed: u64) {
    if let Ok(mut generator) = GLOBAL_GENERATOR.lock() {
        *generator = Generator::with_seed(seed);
    }
}

/// Generate uniform random values in [0, 1) using the global generator
pub fn rand<T>(shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + SampleUniform + NumCast,
{
    let generator = GLOBAL_GENERATOR.lock().map_err(|_| {
        NumRs2Error::InvalidOperation("Failed to acquire global generator lock".to_string())
    })?;
    generator.random(shape)
}

/// Generate normal (Gaussian) random values using the global generator
pub fn randn<T: Float + NumCast + Clone + std::fmt::Debug + std::fmt::Display>(
    shape: &[usize],
) -> Result<Array<T>> {
    let generator = GLOBAL_GENERATOR.lock().map_err(|_| {
        NumRs2Error::InvalidOperation("Failed to acquire global generator lock".to_string())
    })?;
    generator.normal(T::zero(), T::one(), shape)
}

/// Generate uniform random values using the global generator
pub fn uniform<T: Clone + PartialOrd + SampleUniform + Float + NumCast + std::fmt::Display>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>> {
    let generator = GLOBAL_GENERATOR.lock().map_err(|_| {
        NumRs2Error::InvalidOperation("Failed to acquire global generator lock".to_string())
    })?;
    generator.uniform(low, high, shape)
}

/// Shuffle an array using the global generator
pub fn shuffle<T: Clone>(array: &mut Array<T>) -> Result<()> {
    let generator = GLOBAL_GENERATOR.lock().map_err(|_| {
        NumRs2Error::InvalidOperation("Failed to acquire global generator lock".to_string())
    })?;
    generator.shuffle(array)
}

/// Random choice from elements in an array using the global generator
pub fn choice<T: Clone>(
    array: &Array<T>,
    size: Option<usize>,
    replace: Option<bool>,
) -> Result<Array<T>> {
    let generator = GLOBAL_GENERATOR.lock().map_err(|_| {
        NumRs2Error::InvalidOperation("Failed to acquire global generator lock".to_string())
    })?;
    generator.choice(array, size, replace)
}

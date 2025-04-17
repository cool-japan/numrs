//! Random state for compatibility with different random number generator states
//!
//! This module provides the RandomState struct, which is a wrapper around
//! different types of random number generators. This is similar to the RandomState
//! in NumPy's random module.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform, Normal, LogNormal, Bernoulli, Gamma, Exp as Exponential};
use rand_distr::{Beta, ChiSquared as ChiSquare, StudentT, Poisson, Binomial, Cauchy, Weibull};
use rand_distr::uniform::SampleUniform;
use num_traits::{Float, NumCast};
use std::fmt::Display;
use std::fmt::Debug;
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
    fn get_rng(&self) -> Result<std::sync::MutexGuard<'_, StdRng>> {
        self.rng.lock().map_err(|_| {
            NumRs2Error::InvalidOperation("Failed to acquire RNG lock".to_string())
        })
    }
    
    /// Generate uniform random values in [0, 1)
    pub fn random<T>(&self, shape: &[usize]) -> Result<Array<T>> 
    where 
        T: Clone,
        rand_distr::StandardUniform: rand_distr::Distribution<T>
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
    pub fn integers<T: Clone + PartialOrd + SampleUniform + Into<i64> + TryFrom<i64>>(&self, low: T, high: T, shape: &[usize]) -> Result<Array<T>> 
    where 
        <T as TryFrom<i64>>::Error: Debug
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        
        let dist = Uniform::new_inclusive(low, high).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create uniform integer distribution: {}", e))
        })?;
        
        let mut rng = self.get_rng()?;
        
        for _ in 0..size {
            vec.push(dist.sample(&mut *rng));
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a normal (Gaussian) distribution
    pub fn normal<T: Float + NumCast + Clone + Debug + Display>(&self, mean: T, std: T, shape: &[usize]) -> Result<Array<T>> {
        if std <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Standard deviation must be positive, got {}", std)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert normal sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a log-normal distribution
    pub fn lognormal<T: Float + NumCast + Clone + Debug + Display>(&self, mean: T, sigma: T, shape: &[usize]) -> Result<Array<T>> {
        if sigma <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Sigma must be positive, got {}", sigma)
            ));
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
            NumRs2Error::InvalidOperation(format!("Failed to create log-normal distribution: {}", e))
        })?;
        
        let mut rng = self.get_rng()?;
        
        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert lognormal sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Beta distribution
    pub fn beta<T: Float + NumCast + Clone + Debug + Display>(&self, a: T, b: T, shape: &[usize]) -> Result<Array<T>> {
        if a <= T::zero() || b <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Alpha and Beta parameters must be positive, got alpha={}, beta={}", a, b)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert beta sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Chi-Square distribution
    pub fn chisquare<T: Float + NumCast + Clone + Debug + Display>(&self, df: T, shape: &[usize]) -> Result<Array<T>> {
        if df <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Degrees of freedom must be positive, got {}", df)
            ));
        }
        
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let df_f64 = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert degrees of freedom to f64".to_string())
        })?;
        
        let dist = ChiSquare::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create chi-square distribution: {}", e))
        })?;
        
        let mut rng = self.get_rng()?;
        
        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert chi-square sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Dirichlet distribution
    pub fn dirichlet<T: Float + NumCast + Clone + Debug + Display>(&self, alpha: &[T], shape: &[usize]) -> Result<Array<T>> {
        if alpha.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Alpha parameter must have at least one value".to_string()
            ));
        }
        
        for &a in alpha {
            if a <= T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "All alpha parameters must be positive".to_string()
                ));
            }
        }
        
        let size: usize = shape.iter().product();
        let k = alpha.len();
        let mut result = Vec::with_capacity(size * k);
        
        let alpha_f64: Vec<f64> = alpha.iter().map(|&a| {
            a.to_f64().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert alpha parameter to f64".to_string())
            })
        }).collect::<Result<Vec<f64>>>()?;
        
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
                    NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
                })?;
                
                let gamma_sample = gamma.sample(&mut *rng);
                sum += gamma_sample;
                sample.push(gamma_sample);
            }
            
            // Normalize to get a Dirichlet sample
            for val_f64 in sample {
                let normalized = val_f64 / sum;
                let val = T::from(normalized).ok_or_else(|| {
                    NumRs2Error::InvalidOperation("Failed to convert Dirichlet sample to target type".to_string())
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
    pub fn student_t<T: Float + NumCast + Clone + Debug + Display>(&self, df: T, shape: &[usize]) -> Result<Array<T>> {
        if df <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Degrees of freedom must be positive, got {}", df)
            ));
        }
        
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let df_f64 = df.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert degrees of freedom to f64".to_string())
        })?;
        
        let dist = StudentT::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Student's t-distribution: {}", e))
        })?;
        
        let mut rng = self.get_rng()?;
        
        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert Student's t sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Poisson distribution
    pub fn poisson<T: NumCast + Clone + Debug>(&self, lam: f64, shape: &[usize]) -> Result<Array<T>> {
        if lam <= 0.0 {
            return Err(NumRs2Error::InvalidOperation(
                format!("Lambda must be positive, got {}", lam)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert Poisson sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Binomial distribution
    pub fn binomial<T: NumCast + Clone + Debug>(&self, n: u64, p: f64, shape: &[usize]) -> Result<Array<T>> {
        if p < 0.0 || p > 1.0 {
            return Err(NumRs2Error::InvalidOperation(
                format!("Probability must be in [0, 1], got {}", p)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert Binomial sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Cauchy (Lorentz) distribution
    pub fn cauchy<T: Float + NumCast + Clone + Debug + Display>(&self, loc: T, scale: T, shape: &[usize]) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Scale parameter must be positive, got {}", scale)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert Cauchy sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a uniform distribution
    pub fn uniform<T: Clone + PartialOrd + SampleUniform>(&self, low: T, high: T, shape: &[usize]) -> Result<Array<T>> {
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
    pub fn bernoulli<T: Float + NumCast + Clone + Debug + Display>(&self, p: T, shape: &[usize]) -> Result<Array<T>> {
        if p < T::zero() || p > T::one() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Probability must be in [0, 1], got {}", p)
            ));
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
    pub fn gamma<T: Float + NumCast + Clone + Debug + Display>(&self, shape_param: T, scale: T, size_shape: &[usize]) -> Result<Array<T>> {
        if shape_param <= T::zero() || scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Shape and scale parameters must be positive, got shape={}, scale={}", shape_param, scale)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert gamma sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(size_shape))
    }
    
    /// Generate random values from an exponential distribution
    pub fn exponential<T: Float + NumCast + Clone + Debug + Display>(&self, scale: T, shape: &[usize]) -> Result<Array<T>> {
        if scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Scale parameter must be positive, got {}", scale)
            ));
        }
        
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let scale_f64 = scale.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
        })?;
        
        let dist = Exponential::new(1.0 / scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create exponential distribution: {}", e))
        })?;
        
        let mut rng = self.get_rng()?;
        
        for _ in 0..size {
            let val_f64 = dist.sample(&mut *rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert exponential sample to target type".to_string())
            })?;
            vec.push(val);
        }
        
        Ok(Array::from_vec(vec).reshape(shape))
    }
    
    /// Generate random values from a Weibull distribution
    pub fn weibull<T: Float + NumCast + Clone + Debug + Display>(&self, shape_param: T, scale: T, size_shape: &[usize]) -> Result<Array<T>> {
        if shape_param <= T::zero() || scale <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Shape and scale parameters must be positive, got shape={}, scale={}", shape_param, scale)
            ));
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
                NumRs2Error::InvalidOperation("Failed to convert Weibull sample to target type".to_string())
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
    pub fn choice<T: Clone>(&self, array: &Array<T>, size: Option<usize>, replace: Option<bool>) -> Result<Array<T>> {
        let data = array.to_vec();
        if data.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot choose from an empty array".to_string()
            ));
        }
        
        let choose_size = size.unwrap_or(1);
        let with_replacement = replace.unwrap_or(true);
        
        if !with_replacement && choose_size > data.len() {
            return Err(NumRs2Error::InvalidOperation(
                format!("Cannot choose {} items without replacement from array of size {}", choose_size, data.len())
            ));
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
    pub fn standard_normal<T: Float + NumCast + Clone + Debug + Display>(&self, shape: &[usize]) -> Result<Array<T>> {
        self.normal(T::zero(), T::one(), shape)
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
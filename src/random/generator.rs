//! Generator for modern random number generation
//!
//! This module provides the Generator struct for advanced random number generation.
//! It's modeled after NumPy's Generator class, which is the modern interface for
//! random number generation in NumPy.
//!
//! ## Overview
//!
//! The Generator class provides a modern, object-oriented interface for generating random
//! numbers from various probability distributions. It's designed to be:
//!
//! - Thread-safe: Generators can be safely shared between threads
//! - Extensible: New bit generators can be implemented by implementing the BitGenerator trait
//! - Reproducible: Seeds can be set for repeatable random sequences
//!
//! ## Bit Generators
//!
//! This module provides two bit generator implementations:
//!
//! - `StdBitGenerator`: Based on the rand crate's StdRng, which uses ChaCha algorithm
//! - `PCG64BitGenerator`: Based on the PCG64 algorithm, providing high-quality randomness
//!
//! ## Available Distributions
//!
//! The Generator class provides methods for generating random numbers from various distributions:
//!
//! - `random()`: Uniform values in [0, 1)
//! - `uniform()`: Uniform values in [low, high)
//! - `normal()`: Normal (Gaussian) distribution with given mean and standard deviation
//! - `standard_normal()`: Normal distribution with mean 0 and std 1
//! - `beta()`: Beta distribution with parameters a and b
//! - `gamma()`: Gamma distribution with shape and scale parameters
//! - `exponential()`: Exponential distribution with given scale
//! - `weibull()`: Weibull distribution with shape and scale
//! - `poisson()`: Poisson distribution with given mean
//! - `binomial()`: Binomial distribution with n trials and p probability
//! - `bernoulli()`: Bernoulli distribution with success probability p
//! - `chisquare()`: Chi-square distribution with degrees of freedom
//!
//! More distributions are available through the module-level functions in advanced_distributions.rs.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Bernoulli, Distribution, Exp as Exponential, Gamma, LogNormal, Normal, Uniform};
use rand_distr::{Beta, Binomial, ChiSquared as ChiSquare, Poisson, Weibull};
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

/// Bit generator trait for implementing different random number bit generators
pub trait BitGenerator {
    /// Return next u64 random value
    fn next_u64(&mut self) -> u64;

    /// Return next u32 random value
    fn next_u32(&mut self) -> u32;

    /// Return random values between 0 and 1
    fn next_f64(&mut self) -> f64;

    /// Seed the bit generator
    fn seed(&mut self, seed: u64);
}

/// A standard RNG bit generator based on StdRng from rand crate
pub struct StdBitGenerator {
    rng: StdRng,
}

impl StdBitGenerator {
    /// Create a new bit generator with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a new bit generator with a random seed
    pub fn new_random() -> Self {
        let seed = rand::random::<u64>();
        Self::new(seed)
    }
}

impl BitGenerator for StdBitGenerator {
    fn next_u64(&mut self) -> u64 {
        self.rng.random()
    }

    fn next_u32(&mut self) -> u32 {
        self.rng.random()
    }

    fn next_f64(&mut self) -> f64 {
        self.rng.random()
    }

    fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// PCG64 bit generator (Permuted Congruential Generator)
///
/// This is a high-quality generator that's widely used in scientific computing.
/// It's equivalent to the PCG64 generator in NumPy's random module.
pub struct PCG64BitGenerator {
    state: u128,
    inc: u128,
    multiplier: u128,
}

impl PCG64BitGenerator {
    /// Create a new PCG64 bit generator with the given seed
    pub fn new(seed: u64) -> Self {
        // Use the same initialization as in NumPy
        let state = (seed as u128) << 64 | seed as u128;
        let inc = ((seed.wrapping_add(1) as u128) << 64) | 1;
        let multiplier = 0x2360ED051FC65DA44385DF649FCCF645;

        let mut gen = Self {
            state,
            inc,
            multiplier,
        };
        // Warm up the generator
        for _ in 0..10 {
            gen.next_u64();
        }
        gen
    }

    /// Create a new PCG64 bit generator with a random seed
    pub fn new_random() -> Self {
        let seed = rand::random::<u64>();
        Self::new(seed)
    }

    /// Create a new PCG64 bit generator with specific state and increment values
    pub fn with_state_and_inc(state: u128, inc: u128) -> Self {
        let multiplier = 0x2360ED051FC65DA44385DF649FCCF645;
        Self {
            state,
            inc,
            multiplier,
        }
    }

    /// Get the current state of the generator
    pub fn get_state(&self) -> u128 {
        self.state
    }

    /// Get the increment value of the generator
    pub fn get_inc(&self) -> u128 {
        self.inc
    }
}

impl BitGenerator for PCG64BitGenerator {
    fn next_u64(&mut self) -> u64 {
        // PCG update step
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(self.multiplier)
            .wrapping_add(self.inc);

        // Output function (XSH RR: xorshift high (bits), random rotation)
        let xorshifted = (((old_state >> 64) ^ old_state) >> 64) as u64;
        let rot = (old_state >> 122) as u32;

        // Rotate right - need to handle the case where rot is 0
        if rot == 0 {
            xorshifted
        } else {
            (xorshifted >> rot) | (xorshifted << (64 - rot))
        }
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        // Convert to float in [0, 1) range
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn seed(&mut self, seed: u64) {
        *self = PCG64BitGenerator::new(seed);
    }
}

/// Generator for random number streams with modern interface
///
/// This class is modeled after NumPy's Generator class, which is the modern
/// interface for random number generation in NumPy.
pub struct Generator<B: BitGenerator> {
    bit_generator: Arc<Mutex<B>>,
}

impl<B: BitGenerator> Generator<B> {
    /// Create a new generator with the given bit generator
    pub fn new(bit_generator: B) -> Self {
        Self {
            bit_generator: Arc::new(Mutex::new(bit_generator)),
        }
    }

    /// Get a locked reference to the bit generator
    fn get_bit_generator(&self) -> Result<std::sync::MutexGuard<'_, B>> {
        self.bit_generator.lock().map_err(|_| {
            NumRs2Error::InvalidOperation("Failed to acquire bit generator lock".to_string())
        })
    }

    /// Generate uniform random values in [0, 1)
    pub fn random<T>(&self, shape: &[usize]) -> Result<Array<T>>
    where
        T: Clone,
        rand_distr::StandardUniform: rand_distr::Distribution<T>,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);
        let mut bit_gen = self.get_bit_generator()?;

        // Create a standard uniform distribution
        let dist = rand_distr::StandardUniform;

        for _ in 0..size {
            // Create a temp RNG for the distribution
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            vec.push(dist.sample(&mut temp_rng));
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            vec.push(dist.sample(&mut temp_rng));
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert normal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate a standard normal distribution
    pub fn standard_normal<T: Float + NumCast + Clone + Debug + Display>(
        &self,
        shape: &[usize],
    ) -> Result<Array<T>> {
        self.normal(T::zero(), T::one(), shape)
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert chi-square sample to target type".to_string(),
                )
            })?;
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..arr_size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..arr_size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_f64 = dist.sample(&mut temp_rng);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Weibull sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(size_shape))
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            vec.push(dist.sample(&mut temp_rng));
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_bool = dist.sample(&mut temp_rng);
            let val = if val_bool { T::one() } else { T::zero() };
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_u64 = dist.sample(&mut temp_rng);
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Poisson sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate random values from a binomial distribution
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

        let mut bit_gen = self.get_bit_generator()?;

        for _ in 0..size {
            // Create a temp RNG for the distribution using a random seed from our bit generator
            let mut temp_rng = StdRng::seed_from_u64(bit_gen.next_u64());
            let val_u64 = dist.sample(&mut temp_rng);
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Binomial sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Ok(Array::from_vec(vec).reshape(shape))
    }

    /// Generate integers in a given range
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive)
    /// * `shape` - Shape of the output array
    ///
    /// # Returns
    ///
    /// An array of random integers in the specified range.
    pub fn integers_simple<T: Clone + PartialOrd + SampleUniform>(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        self.uniform(low, high, shape)
    }

    /// Access the underlying bit generator
    pub fn bit_generator(&self) -> Result<std::sync::MutexGuard<'_, B>> {
        self.get_bit_generator()
    }
}

/// Create a default generator
///
/// Returns a Generator with the default bit generator.
///
/// # Examples
///
/// ```
/// use numrs2::random::default_rng;
///
/// let rng = default_rng();
/// let random_array = rng.random::<f64>(&[3, 3]).unwrap();
/// ```
pub fn default_rng() -> Generator<StdBitGenerator> {
    Generator::new(StdBitGenerator::new_random())
}

/// Create a generator with a specific seed
///
/// Returns a Generator with the default bit generator seeded with the given seed.
///
/// # Examples
///
/// ```
/// use numrs2::random::seed_rng;
///
/// let rng = seed_rng(42);
/// let random_array = rng.random::<f64>(&[3, 3]).unwrap();
/// ```
pub fn seed_rng(seed: u64) -> Generator<StdBitGenerator> {
    Generator::new(StdBitGenerator::new(seed))
}

/// Create a PCG64 generator
///
/// Returns a Generator with the PCG64 bit generator, which is a high-quality
/// generator used in scientific computing. This is equivalent to the PCG64
/// generator in NumPy's random module.
///
/// # Examples
///
/// ```
/// use numrs2::random::pcg64_rng;
///
/// let rng = pcg64_rng();
/// let random_array = rng.random::<f64>(&[3, 3]).unwrap();
/// ```
pub fn pcg64_rng() -> Generator<PCG64BitGenerator> {
    Generator::new(PCG64BitGenerator::new_random())
}

/// Create a PCG64 generator with a specific seed
///
/// Returns a Generator with the PCG64 bit generator seeded with the given seed.
///
/// # Examples
///
/// ```
/// use numrs2::random::pcg64_seed_rng;
///
/// let rng = pcg64_seed_rng(42);
/// let random_array = rng.random::<f64>(&[3, 3]).unwrap();
/// ```
pub fn pcg64_seed_rng(seed: u64) -> Generator<PCG64BitGenerator> {
    Generator::new(PCG64BitGenerator::new(seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rng() {
        let rng = default_rng();
        let arr = rng.random::<f64>(&[3, 3]).unwrap();
        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_seed_rng() {
        let rng1 = seed_rng(42);
        let arr1 = rng1.random::<f64>(&[3, 3]).unwrap();

        let rng2 = seed_rng(42);
        let arr2 = rng2.random::<f64>(&[3, 3]).unwrap();

        // Same seed should produce the same random numbers
        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }

    #[test]
    fn test_generator_normal() {
        let rng = default_rng();
        let arr = rng.normal(0.0, 1.0, &[10]).unwrap();
        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_pcg64_generator() {
        let rng = pcg64_rng();
        let arr = rng.random::<f64>(&[3, 3]).unwrap();
        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_pcg64_seed_produces_same_output() {
        let rng1 = pcg64_seed_rng(42);
        let arr1 = rng1.random::<f64>(&[5]).unwrap();

        let rng2 = pcg64_seed_rng(42);
        let arr2 = rng2.random::<f64>(&[5]).unwrap();

        // Same seed should produce the same random numbers
        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }

    #[test]
    fn test_generator_distributions() {
        let rng = default_rng();

        // Test various distributions
        let beta_arr = rng.beta(2.0, 5.0, &[10]).unwrap();
        assert_eq!(beta_arr.shape(), vec![10]);

        let gamma_arr = rng.gamma(2.0, 2.0, &[10]).unwrap();
        assert_eq!(gamma_arr.shape(), vec![10]);

        let uniform_arr = rng.uniform(0.0, 1.0, &[10]).unwrap();
        assert_eq!(uniform_arr.shape(), vec![10]);

        let binomial_arr = rng.binomial::<u32>(10, 0.5, &[10]).unwrap();
        assert_eq!(binomial_arr.shape(), vec![10]);

        let poisson_arr = rng.poisson::<u32>(5.0, &[10]).unwrap();
        assert_eq!(poisson_arr.shape(), vec![10]);
    }

    #[test]
    fn test_pcg64_state() {
        let mut rng = PCG64BitGenerator::new(42);
        let initial_state = rng.get_state();

        // Generate some random numbers
        for _ in 0..10 {
            rng.next_u64();
        }

        // State should have changed
        assert_ne!(initial_state, rng.get_state());

        // Reset the state
        rng.seed(42);

        // Should get a new state after reseeding (because of the warm-up steps)
        let _state_after_reset = rng.get_state();

        // Create a new generator with the same seed
        let rng2 = PCG64BitGenerator::new(42);

        // Both generators should have the same state
        assert_eq!(rng.get_state(), rng2.get_state());
    }

    #[test]
    fn test_bit_generator_methods() {
        let mut std_rng = StdBitGenerator::new(42);
        let mut pcg_rng = PCG64BitGenerator::new(42);

        // Each bit generator should produce different values
        let std_u64 = std_rng.next_u64();
        let pcg_u64 = pcg_rng.next_u64();

        // Values should be different since they use different algorithms
        assert_ne!(std_u64, pcg_u64);

        // But each algorithm should be consistent
        std_rng.seed(42);
        assert_eq!(std_rng.next_u64(), std_u64);

        pcg_rng.seed(42);
        assert_eq!(pcg_rng.next_u64(), pcg_u64);
    }
}

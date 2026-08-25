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

use super::philox::Philox4x64BitGenerator;
use super::seed_sequence::SeedSequence;
use super::sfc64::SFC64BitGenerator;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, ToPrimitive};
// SCIRS2 POLICY COMPLIANT imports - always use SciRS2
use scirs2_core::ndarray::Axis;
use scirs2_core::random::prelude::*;
use scirs2_core::random::uniform::SampleUniform;
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

/// A [`BitGenerator`] that can be constructed from a [`SeedSequence`].
///
/// `SeedSequence`-based construction is what makes [`Generator::spawn`]
/// possible: spawning derives independent child `SeedSequence`s, and each
/// child needs a way to turn itself into a fresh bit generator. This is
/// implemented by the entropy-pool-seeded generators added for NumPy
/// parity ([`Philox4x64BitGenerator`], [`SFC64BitGenerator`]); it is
/// intentionally not implemented for [`StdBitGenerator`] or
/// [`PCG64BitGenerator`], whose `new(seed: u64)` constructors predate
/// `SeedSequence` and are kept exactly as-is for backward compatibility.
pub trait SeedableBitGenerator: BitGenerator + Sized {
    /// Build a new instance of this bit generator from a [`SeedSequence`].
    fn from_seed_sequence(seed_seq: &SeedSequence) -> Self;
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
        let mut rng = thread_rng();
        let seed = rng.random::<u64>();
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
        let mut rng = thread_rng();
        let seed = rng.random::<u64>();
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

        // Rotate right
        xorshifted.rotate_right(rot)
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
    /// The [`SeedSequence`] this generator was constructed from, if any.
    ///
    /// Populated by [`Generator::from_seed_sequence`] and consumed by
    /// [`Generator::spawn`]; `None` for every other constructor (including
    /// plain [`Generator::new`]), since a directly-supplied bit generator
    /// has no seed sequence to derive independent children from.
    seed_sequence: Option<Arc<Mutex<SeedSequence>>>,
}

impl<B: BitGenerator> Generator<B> {
    /// Create a new generator with the given bit generator
    pub fn new(bit_generator: B) -> Self {
        Self {
            bit_generator: Arc::new(Mutex::new(bit_generator)),
            seed_sequence: None,
        }
    }

    /// Create a new generator from any [`BitGenerator`] implementation.
    ///
    /// An alias for [`Generator::new`], spelled to match the "plug a bit
    /// generator into the existing `Generator`/distribution machinery"
    /// pattern: e.g. `Generator::from_bit_generator(Philox4x64BitGenerator::new(seed))`
    /// immediately gains every distribution method (`random`, `normal`,
    /// `uniform`, `permuted`, ...) defined on `Generator<B>`.
    pub fn from_bit_generator(bit_generator: B) -> Self {
        Self::new(bit_generator)
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
        T: Clone + Float + NumCast,
    {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            // Uniform values in [0, 1)
            let val_f64: f64 = rng.random::<f64>();
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert uniform sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Convert bounds to f64 once.
        let low_f64 = low.clone().into().to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert low bound to f64".to_string())
        })?;
        let high_f64 = high.clone().into().to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert high bound to f64".to_string())
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            // Uniform value in [low, high), floored to an integer.
            let val_f64: f64 = rng.random_range(low_f64..high_f64);
            let val_i64 = val_f64.floor() as i64;
            let val = T::try_from(val_i64).map_err(|_| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert integer sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert normal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert lognormal sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        let dist = BetaDist::new(a_f64, b_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create beta distribution: {}", e))
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert beta sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        let dist = ChiSquared::new(df_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create chi-square distribution: {}",
                e
            ))
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert chi-square sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // rand_distr::Gamma::new(shape, scale) takes the plain scale parameter.
        let dist = Gamma::new(shape_f64, scale_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create gamma distribution: {}", e))
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..arr_size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert gamma sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, size_shape)
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

        // rand_distr::Exp::new(rate) expects rate = 1/scale.
        // For an exponential distribution with scale s: rate = 1/s, mean = s, variance = s².
        let rate = 1.0 / scale_f64;
        let dist = scirs2_core::random::rand_distributions::Exp::new(rate).map_err(|e| {
            NumRs2Error::InvalidOperation(format!(
                "Failed to create exponential distribution: {}",
                e
            ))
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert exponential sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // rand_distr::Weibull::new(scale, shape) takes scale first, then shape.
        let dist = Weibull::new(scale_f64, shape_f64).map_err(|e| {
            NumRs2Error::InvalidOperation(format!("Failed to create Weibull distribution: {}", e))
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..arr_size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Weibull sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, size_shape)
    }

    /// Generate random values from a uniform distribution
    pub fn uniform<T: Clone + PartialOrd + SampleUniform + ToPrimitive + NumCast>(
        &self,
        low: T,
        high: T,
        shape: &[usize],
    ) -> Result<Array<T>> {
        let size: usize = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        // Convert bounds to f64 once.
        let low_f64 = low.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert low bound to f64".to_string())
        })?;
        let high_f64 = high.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert high bound to f64".to_string())
        })?;

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            // Uniform value in [low, high).
            let val_f64: f64 = rng.random_range(low_f64..high_f64);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert uniform sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_bool: bool = rng.sample(dist);
            let val = if val_bool { T::one() } else { T::zero() };
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_f64: f64 = rng.sample(dist);
            let val = T::from(val_f64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Poisson sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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

        // Seed a per-call RNG from the bit generator stream so output is
        // reproducible from the global seed.
        let mut bit_gen = self.get_bit_generator()?;
        let mut rng = StdRng::seed_from_u64(bit_gen.next_u64());

        for _ in 0..size {
            let val_u64: u64 = rng.sample(dist);
            let val = T::from(val_u64).ok_or_else(|| {
                NumRs2Error::InvalidOperation(
                    "Failed to convert Binomial sample to target type".to_string(),
                )
            })?;
            vec.push(val);
        }

        Array::from_vec_shape(vec, shape)
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
    pub fn integers_simple<T: Clone + PartialOrd + SampleUniform + num_traits::NumCast>(
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

    /// Randomly permute an array, either by fully flattening it
    /// (`axis = None`) or independently along one axis (`axis = Some(k)`).
    ///
    /// # NumPy semantics
    ///
    /// - `axis = None`: `a` is flattened (row-major/C order), the flat copy
    ///   is shuffled as a whole, and the result is reshaped back to `a`'s
    ///   original shape.
    /// - `axis = Some(k)`: every 1-D "lane" along axis `k` (i.e. every
    ///   slice obtained by fixing all other axis indices) is shuffled
    ///   **independently of every other lane**. This is the key difference
    ///   from a `shuffle`-style operation (as in NumPy's
    ///   `Generator.shuffle`, or this crate's legacy
    ///   `random::legacy::Generator::shuffle`), which instead moves whole
    ///   slices as units using a *single* permutation shared by every lane.
    ///   `permuted` never mixes elements between lanes; only the order
    ///   *within* each lane changes, and each lane gets its own independent
    ///   random permutation.
    ///
    /// Matches `numpy.random.Generator.permuted(a, axis=axis)` with
    /// `out=None`, i.e. this always returns a new array rather than
    /// permuting in place.
    ///
    /// # Exactness
    ///
    /// This crate's `StdBitGenerator`/`PCG64BitGenerator` do not reproduce
    /// NumPy's actual default `PCG64` bit generator byte-for-byte (verified
    /// during the implementation of this method: NumPy's `PCG64` derives
    /// its 128-bit state and increment from a `SeedSequence`, while
    /// `PCG64BitGenerator::new` here derives them directly from the `u64`
    /// seed with a different formula entirely). Because of that, a
    /// `numrs2` `Generator`'s `permuted` output is only reproducible
    /// against another `numrs2` `Generator` seeded the same way, not
    /// against `np.random.default_rng(seed).permuted(...)`. `Generator`s
    /// built on [`Philox4x64BitGenerator`] or [`SFC64BitGenerator`] (whose
    /// *raw* output streams **do** match NumPy exactly, see those types'
    /// docs) still go through this same seed-a-fresh-`StdRng`-per-call
    /// sampling strategy for the shuffle itself, so even they are not
    /// NumPy-`permuted`-bit-identical. What *is* guaranteed to match NumPy
    /// is the semantic contract above (independent per-lane shuffling vs.
    /// whole-slice `shuffle`), which this module's tests verify
    /// statistically and by construction (shape/permutation validity).
    ///
    /// # Errors
    ///
    /// Returns an error if `axis` is `Some(k)` with `k >= a.ndim()`.
    pub fn permuted<T: Clone>(&self, a: &Array<T>, axis: Option<usize>) -> Result<Array<T>> {
        // Seed a per-call RNG from the bit generator stream, exactly like
        // every other sampling method on this type, so the whole shuffle
        // (all lanes, for the `axis = Some(k)` case) is reproducible from
        // the generator's seed.
        let mut rng = {
            let mut bit_gen = self.get_bit_generator()?;
            StdRng::seed_from_u64(bit_gen.next_u64())
        };

        match axis {
            None => {
                let mut flat = a.to_vec();
                fisher_yates_shuffle(flat.len(), &mut rng, |i, j| flat.swap(i, j));
                Array::from_vec_shape(flat, &a.shape())
            }
            Some(axis) => {
                if axis >= a.ndim() {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "axis {} out of bounds for array with {} dimension(s)",
                        axis,
                        a.ndim()
                    )));
                }
                let mut result = a.clone();
                for mut lane in result.array_mut().lanes_mut(Axis(axis)) {
                    let len = lane.len();
                    fisher_yates_shuffle(len, &mut rng, |i, j| lane.swap(i, j));
                }
                Ok(result)
            }
        }
    }
}

/// In-place Fisher-Yates shuffle driven by `rng`, expressed generically
/// over how to swap two positions. This lets the same shuffle logic drive
/// both a flat `Vec<T>` (the `axis = None` case of [`Generator::permuted`])
/// and a single ndarray lane view (the `axis = Some(k)` case) without an
/// intermediate copy for the lane case, since `ndarray`'s view types expose
/// the same `swap(i, j)` signature as `[T]::swap`.
fn fisher_yates_shuffle(len: usize, rng: &mut StdRng, mut swap: impl FnMut(usize, usize)) {
    if len < 2 {
        return;
    }
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        swap(i, j);
    }
}

impl<B: BitGenerator + SeedableBitGenerator> Generator<B> {
    /// Create a generator whose bit generator is seeded from a
    /// [`SeedSequence`], enabling [`Generator::spawn`].
    ///
    /// Mirrors the NumPy pattern `Generator(BitGeneratorClass(seed_sequence))`
    /// used throughout NumPy's documentation for seeding parallel streams.
    pub fn from_seed_sequence(seed_sequence: SeedSequence) -> Self {
        let bit_gen = B::from_seed_sequence(&seed_sequence);
        Self {
            bit_generator: Arc::new(Mutex::new(bit_gen)),
            seed_sequence: Some(Arc::new(Mutex::new(seed_sequence))),
        }
    }

    /// Spawn `n` new, statistically independent generators.
    ///
    /// Only available for a `Generator` that was itself built (directly, or
    /// transitively via a previous `spawn` call) from a [`SeedSequence`]
    /// via [`Generator::from_seed_sequence`] -- otherwise there is no
    /// `SeedSequence` to derive children from, and this returns an error
    /// rather than fabricating one from the live bit generator state (which
    /// would risk silently overlapping streams).
    ///
    /// Mirrors `numpy.random.SeedSequence.spawn`, used to seed `n`
    /// independent `Generator`s for parallel work, e.g.:
    /// `[Generator(Philox(s)) for s in SeedSequence(seed).spawn(n)]`.
    ///
    /// # Errors
    ///
    /// Returns an error if this `Generator` was not constructed via
    /// [`Generator::from_seed_sequence`].
    pub fn spawn(&self, n: usize) -> Result<Vec<Self>> {
        let seed_sequence = self.seed_sequence.as_ref().ok_or_else(|| {
            NumRs2Error::InvalidOperation(
                "Generator was not constructed via Generator::from_seed_sequence; \
                 there is no SeedSequence to spawn children from"
                    .to_string(),
            )
        })?;
        let mut seq = seed_sequence.lock().map_err(|_| {
            NumRs2Error::InvalidOperation("Failed to acquire seed sequence lock".to_string())
        })?;
        let children = seq.spawn(n);
        drop(seq);
        Ok(children.into_iter().map(Self::from_seed_sequence).collect())
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
/// let random_array = rng.random::<f64>(&[3, 3]).expect("random should succeed");
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
/// let random_array = rng.random::<f64>(&[3, 3]).expect("seeded random should succeed");
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
/// let random_array = rng.random::<f64>(&[3, 3]).expect("pcg64 random should succeed");
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
/// let random_array = rng.random::<f64>(&[3, 3]).expect("seeded pcg64 random should succeed");
/// ```
pub fn pcg64_seed_rng(seed: u64) -> Generator<PCG64BitGenerator> {
    Generator::new(PCG64BitGenerator::new(seed))
}

/// Create a Philox4x64-10 generator with fresh OS-backed entropy.
///
/// Returns a `Generator` built via [`Generator::from_seed_sequence`], so it
/// can be split into independent streams with [`Generator::spawn`].
///
/// # Examples
///
/// ```
/// use numrs2::random::philox_rng;
///
/// let rng = philox_rng();
/// let random_array = rng.random::<f64>(&[3, 3]).expect("philox random should succeed");
/// ```
pub fn philox_rng() -> Generator<Philox4x64BitGenerator> {
    Generator::from_seed_sequence(SeedSequence::from_os_entropy())
}

/// Create a Philox4x64-10 generator with a specific seed.
///
/// [`Philox4x64BitGenerator`] reproduces `np.random.Philox(seed=seed)`'s
/// raw output exactly; see its documentation for the compatibility
/// guarantee and why it is a "4x64", not "4x32", generator despite being
/// commonly just called "Philox".
///
/// # Examples
///
/// ```
/// use numrs2::random::philox_seed_rng;
///
/// let rng = philox_seed_rng(42);
/// let random_array = rng.random::<f64>(&[3, 3]).expect("seeded philox random should succeed");
/// ```
pub fn philox_seed_rng(seed: u64) -> Generator<Philox4x64BitGenerator> {
    Generator::new(Philox4x64BitGenerator::new(seed))
}

/// Create a Philox4x64-10 generator from an explicit [`SeedSequence`],
/// enabling [`Generator::spawn`] for parallel, independent streams.
pub fn philox_from_seed_sequence(seed_sequence: SeedSequence) -> Generator<Philox4x64BitGenerator> {
    Generator::from_seed_sequence(seed_sequence)
}

/// Create an SFC64 generator with fresh OS-backed entropy.
///
/// Returns a `Generator` built via [`Generator::from_seed_sequence`], so it
/// can be split into independent streams with [`Generator::spawn`].
///
/// # Examples
///
/// ```
/// use numrs2::random::sfc64_rng;
///
/// let rng = sfc64_rng();
/// let random_array = rng.random::<f64>(&[3, 3]).expect("sfc64 random should succeed");
/// ```
pub fn sfc64_rng() -> Generator<SFC64BitGenerator> {
    Generator::from_seed_sequence(SeedSequence::from_os_entropy())
}

/// Create an SFC64 generator with a specific seed.
///
/// [`SFC64BitGenerator`] reproduces `np.random.SFC64(seed)`'s raw output
/// exactly; see its documentation for the compatibility guarantee.
///
/// # Examples
///
/// ```
/// use numrs2::random::sfc64_seed_rng;
///
/// let rng = sfc64_seed_rng(42);
/// let random_array = rng.random::<f64>(&[3, 3]).expect("seeded sfc64 random should succeed");
/// ```
pub fn sfc64_seed_rng(seed: u64) -> Generator<SFC64BitGenerator> {
    Generator::new(SFC64BitGenerator::new(seed))
}

/// Create an SFC64 generator from an explicit [`SeedSequence`], enabling
/// [`Generator::spawn`] for parallel, independent streams.
pub fn sfc64_from_seed_sequence(seed_sequence: SeedSequence) -> Generator<SFC64BitGenerator> {
    Generator::from_seed_sequence(seed_sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rng() {
        let rng = default_rng();
        let arr = rng
            .random::<f64>(&[3, 3])
            .expect("test: random should succeed");
        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_seed_rng() {
        let rng1 = seed_rng(42);
        let arr1 = rng1
            .random::<f64>(&[3, 3])
            .expect("test: random should succeed");

        let rng2 = seed_rng(42);
        let arr2 = rng2
            .random::<f64>(&[3, 3])
            .expect("test: random should succeed");

        // Same seed should produce the same random numbers
        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }

    #[test]
    fn test_generator_normal() {
        let rng = default_rng();
        let arr = rng
            .normal(0.0, 1.0, &[10])
            .expect("test: normal should succeed");
        assert_eq!(arr.shape(), vec![10]);
    }

    #[test]
    fn test_pcg64_generator() {
        let rng = pcg64_rng();
        let arr = rng
            .random::<f64>(&[3, 3])
            .expect("test: random should succeed");
        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn test_pcg64_seed_produces_same_output() {
        let rng1 = pcg64_seed_rng(42);
        let arr1 = rng1
            .random::<f64>(&[5])
            .expect("test: random should succeed");

        let rng2 = pcg64_seed_rng(42);
        let arr2 = rng2
            .random::<f64>(&[5])
            .expect("test: random should succeed");

        // Same seed should produce the same random numbers
        assert_eq!(arr1.to_vec(), arr2.to_vec());
    }

    #[test]
    fn test_generator_distributions() {
        let rng = default_rng();

        // Test various distributions
        let beta_arr = rng
            .beta(2.0, 5.0, &[10])
            .expect("test: beta should succeed");
        assert_eq!(beta_arr.shape(), vec![10]);

        let gamma_arr = rng
            .gamma(2.0, 2.0, &[10])
            .expect("test: gamma should succeed");
        assert_eq!(gamma_arr.shape(), vec![10]);

        let uniform_arr = rng
            .uniform(0.0, 1.0, &[10])
            .expect("test: uniform should succeed");
        assert_eq!(uniform_arr.shape(), vec![10]);

        let binomial_arr = rng
            .binomial::<u32>(10, 0.5, &[10])
            .expect("test: binomial should succeed");
        assert_eq!(binomial_arr.shape(), vec![10]);

        let poisson_arr = rng
            .poisson::<u32>(5.0, &[10])
            .expect("test: poisson should succeed");
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

    // ---- W4-G: Philox4x64 / SFC64 wiring, SeedSequence::spawn, permuted ----

    #[test]
    fn from_bit_generator_works_with_existing_distributions() {
        // Literal requirement: "Generator::from_bit_generator(philox) works
        // with existing distributions."
        let rng = Generator::from_bit_generator(Philox4x64BitGenerator::new(42));
        let normal_arr = rng
            .normal(0.0, 1.0, &[10])
            .expect("normal should succeed with a Philox bit generator");
        assert_eq!(normal_arr.shape(), vec![10]);

        let uniform_arr = rng
            .random::<f64>(&[5])
            .expect("random should succeed with a Philox bit generator");
        assert_eq!(uniform_arr.shape(), vec![5]);
    }

    #[test]
    fn philox_seed_rng_raw_stream_matches_bare_bit_generator() {
        // Wiring a Philox4x64BitGenerator through `Generator` must not
        // disturb its raw output stream (which is pinned against NumPy in
        // `philox.rs`'s own tests).
        let rng = philox_seed_rng(42);
        let mut expected = Philox4x64BitGenerator::new(42);
        let mut bit_gen = rng.bit_generator().expect("bit generator lock");
        for _ in 0..8 {
            assert_eq!(bit_gen.next_u64(), expected.next_u64());
        }
    }

    #[test]
    fn sfc64_seed_rng_raw_stream_matches_bare_bit_generator() {
        let rng = sfc64_seed_rng(42);
        let mut expected = SFC64BitGenerator::new(42);
        let mut bit_gen = rng.bit_generator().expect("bit generator lock");
        for _ in 0..8 {
            assert_eq!(bit_gen.next_u64(), expected.next_u64());
        }
    }

    #[test]
    fn philox_and_sfc64_rng_produce_arrays() {
        let philox = philox_rng();
        let arr = philox
            .random::<f64>(&[3, 3])
            .expect("philox random should succeed");
        assert_eq!(arr.shape(), vec![3, 3]);

        let sfc64 = sfc64_rng();
        let arr = sfc64
            .random::<f64>(&[3, 3])
            .expect("sfc64 random should succeed");
        assert_eq!(arr.shape(), vec![3, 3]);
    }

    #[test]
    fn spawn_requires_a_seed_sequence() {
        // A plain `Generator::new` has no SeedSequence to spawn from.
        // (`Generator<B>` isn't `Debug`, so match directly rather than
        // using `expect_err`, which requires the `Ok` type to be `Debug`.)
        let rng = philox_seed_rng(42);
        match rng.spawn(2) {
            Err(NumRs2Error::InvalidOperation(_)) => {}
            Err(other) => panic!("expected InvalidOperation, got {other:?}"),
            Ok(_) => panic!("spawn without a SeedSequence must error"),
        }
    }

    #[test]
    fn spawn_matches_numpy_end_to_end() {
        // The single strongest evidence for "Generator gains .spawn(n) via
        // SeedSequence" is that the *composition* -- SeedSequence::spawn
        // feeding Philox4x64BitGenerator::from_seed_sequence, both wired
        // through Generator::from_seed_sequence/spawn -- matches NumPy end
        // to end, not just each piece in isolation. Computed with numpy
        // 2.4.2:
        //     sg = np.random.SeedSequence(42)
        //     for c in sg.spawn(3):
        //         np.random.Philox(c).random_raw(4)
        let rng: Generator<Philox4x64BitGenerator> =
            Generator::from_seed_sequence(SeedSequence::new(42));
        let children = rng.spawn(3).expect("spawn should succeed");

        let expected: [[u64; 4]; 3] = [
            [
                4053553053279984544,
                8415284293751313562,
                3393911196759700467,
                8231072555151242699,
            ],
            [
                14262185550820668636,
                7822093348542869227,
                263861048173042464,
                7266916232282267429,
            ],
            [
                1717407965986836309,
                6786652523942058177,
                9512072801243946500,
                3885077021913483620,
            ],
        ];

        for (child, expected_block) in children.iter().zip(expected.iter()) {
            let mut bit_gen = child.bit_generator().expect("bit generator lock");
            let actual: Vec<u64> = (0..4).map(|_| bit_gen.next_u64()).collect();
            assert_eq!(&actual, expected_block);
        }
    }

    #[test]
    fn spawn_produces_independent_reproducible_children() {
        // Re-spawning a fresh generator built the same way reproduces the
        // same children (determinism). Checked with an untouched `children`
        // set so neither side's internal state has been advanced yet --
        // drawing from a child first would make it diverge from a freshly
        // constructed twin, by design (that's what "independent streams"
        // means), so that check uses its own separate spawn below.
        let rng: Generator<Philox4x64BitGenerator> =
            Generator::from_seed_sequence(SeedSequence::new(42));
        let children = rng.spawn(3).expect("spawn should succeed");
        assert_eq!(children.len(), 3);

        let rng2: Generator<Philox4x64BitGenerator> =
            Generator::from_seed_sequence(SeedSequence::new(42));
        let children2 = rng2.spawn(3).expect("spawn should succeed");
        for (a, b) in children.iter().zip(children2.iter()) {
            let arr_a = a.random::<f64>(&[4]).expect("random should succeed");
            let arr_b = b.random::<f64>(&[4]).expect("random should succeed");
            assert_eq!(arr_a.to_vec(), arr_b.to_vec());
        }

        // Siblings draw different streams from each other (fresh spawn, so
        // this doesn't interact with the determinism check above).
        let rng3: Generator<Philox4x64BitGenerator> =
            Generator::from_seed_sequence(SeedSequence::new(42));
        let children3 = rng3.spawn(3).expect("spawn should succeed");
        let draws: Vec<f64> = children3
            .iter()
            .map(|c| {
                c.random::<f64>(&[1])
                    .expect("random should succeed")
                    .to_vec()[0]
            })
            .collect();
        assert_ne!(draws[0], draws[1]);
        assert_ne!(draws[1], draws[2]);
    }

    #[test]
    fn spawn_is_transitive_through_children() {
        // A spawned child was itself built via `from_seed_sequence`, so it
        // can spawn grandchildren too.
        let rng: Generator<Philox4x64BitGenerator> =
            Generator::from_seed_sequence(SeedSequence::new(7));
        let children = rng.spawn(2).expect("spawn should succeed");
        let grandchildren = children[0]
            .spawn(2)
            .expect("spawn of a child should succeed");
        assert_eq!(grandchildren.len(), 2);
    }

    #[test]
    fn permuted_axis_none_is_a_valid_permutation_with_same_shape() {
        let rng = pcg64_seed_rng(123);
        let original = Array::from_vec_shape((0..12).collect::<Vec<i32>>(), &[3, 4])
            .expect("array construction should succeed");

        let permuted = rng
            .permuted(&original, None)
            .expect("permuted should succeed");

        assert_eq!(permuted.shape(), original.shape());
        let mut sorted_original = original.to_vec();
        let mut sorted_permuted = permuted.to_vec();
        sorted_original.sort_unstable();
        sorted_permuted.sort_unstable();
        assert_eq!(sorted_original, sorted_permuted);
    }

    #[test]
    fn permuted_is_deterministic_for_a_given_seed() {
        // Our default/PCG64 bit generators do not match NumPy's, so
        // `permuted` is pinned against our own seeded output rather than
        // `np.random.default_rng(seed).permuted(...)` (documented on
        // `Generator::permuted`). What we *can* and do assert exactly is
        // reproducibility: the same seed must always permute identically.
        let original = Array::from_vec_shape((0..20).collect::<Vec<i32>>(), &[20])
            .expect("array construction should succeed");

        let rng1 = pcg64_seed_rng(2024);
        let out1 = rng1
            .permuted(&original, None)
            .expect("permuted should succeed");

        let rng2 = pcg64_seed_rng(2024);
        let out2 = rng2
            .permuted(&original, None)
            .expect("permuted should succeed");

        assert_eq!(out1.to_vec(), out2.to_vec());
        // And it must actually reorder a 20-element array (astronomically
        // unlikely to round-trip to the identity by chance).
        assert_ne!(out1.to_vec(), original.to_vec());
    }

    #[test]
    fn permuted_axis_preserves_shape_and_per_lane_contents() {
        let rng = pcg64_seed_rng(7);
        // 4 rows, each row = [0, 1, 2, 3, 4] before permuting.
        let data: Vec<i32> = (0..4).flat_map(|_| 0..5).collect();
        let original =
            Array::from_vec_shape(data, &[4, 5]).expect("array construction should succeed");

        let permuted = rng
            .permuted(&original, Some(1))
            .expect("permuted should succeed");

        assert_eq!(permuted.shape(), vec![4, 5]);
        let flat = permuted.to_vec();
        for row in 0..4 {
            let mut row_vals = flat[row * 5..(row + 1) * 5].to_vec();
            row_vals.sort_unstable();
            assert_eq!(row_vals, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn permuted_axis_shuffles_lanes_independently_not_as_whole_slices() {
        // This is the statistical independence test: unlike `shuffle`
        // (which would apply one permutation to every row), `permuted`
        // with `axis = Some(k)` must give *different* rows *different*
        // permutations. With 64 rows of length 5 (120 possible orderings),
        // seeing every row collapse onto the same permutation by chance is
        // astronomically unlikely (~(1/120)^63); observing it would
        // indicate a whole-slice-shuffle bug rather than bad luck.
        let rng = pcg64_seed_rng(99);
        let rows = 64;
        let cols = 5;
        let data: Vec<i32> = (0..rows).flat_map(|_| 0..cols as i32).collect();
        let original =
            Array::from_vec_shape(data, &[rows, cols]).expect("array construction should succeed");

        let permuted = rng
            .permuted(&original, Some(1))
            .expect("permuted should succeed");
        let flat = permuted.to_vec();

        let row_perms: std::collections::HashSet<Vec<i32>> = (0..rows)
            .map(|row| flat[row * cols..(row + 1) * cols].to_vec())
            .collect();
        assert!(
            row_perms.len() > 1,
            "expected independent per-row permutations, got the same permutation on every row: {row_perms:?}"
        );
    }

    #[test]
    fn permuted_rejects_out_of_bounds_axis() {
        let rng = pcg64_seed_rng(1);
        let original =
            Array::from_vec_shape(vec![1, 2, 3, 4], &[2, 2]).expect("array construction");
        let err = rng
            .permuted(&original, Some(2))
            .expect_err("axis 2 is out of bounds for a 2-D array");
        match err {
            NumRs2Error::IndexOutOfBounds(_) => {}
            other => panic!("expected IndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn permuted_handles_trivial_lengths() {
        let rng = pcg64_seed_rng(1);
        // Length-0 and length-1 flat arrays must not panic and must return
        // the (only possible) input back unchanged.
        let empty: Array<i32> = Array::from_vec_shape(vec![], &[0]).expect("array construction");
        let out = rng.permuted(&empty, None).expect("permuted should succeed");
        assert_eq!(out.to_vec(), Vec::<i32>::new());

        let single = Array::from_vec_shape(vec![42], &[1]).expect("array construction");
        let out = rng
            .permuted(&single, None)
            .expect("permuted should succeed");
        assert_eq!(out.to_vec(), vec![42]);
    }
}

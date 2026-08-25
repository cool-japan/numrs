//! Enhanced random distributions
//!
//! This module expands the random functionality by providing additional
//! distributions and utility functions for more specialized statistical applications.

use crate::array::Array;
use crate::error::Result;
use crate::linalg_stable::StableDecompositions;
use crate::new_modules::special::error_functions::erf_scalar;
use crate::random::state::RandomState;
use num_traits::{Float, NumCast};
use scirs2_core::random::prelude::Rng;
use scirs2_special::betainc_regularized;
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

/// Number of bits used to represent each Sobol direction number. Matches the
/// `2f64.powi(32)` scaling used when converting a generated point to `[0,1)`.
const SOBOL_BITS: usize = 32;

/// Maximum Sobol dimensionality supported by [`SOBOL_DIRECTION_DATA`].
const SOBOL_MAX_DIM: usize = 40;

/// Joe & Kuo (2008) "new-joe-kuo-6.21201" primitive polynomial coefficients and
/// initial direction numbers for Sobol dimensions 2 through 40 (search
/// criterion 6 — the default used by `scipy.stats.qmc.Sobol`).
///
/// Row `k` (0-indexed) describes dimension `k + 2`: `.0` is the primitive
/// polynomial's middle coefficients `a_1..a_{s-1}` packed into a single
/// integer (`a_1` in the most-significant bit of the `s - 1`-bit value), and
/// `.1` is the degree-`s` slice of initial direction numbers `m_1..m_s`.
/// Dimension 1 is the trivial base-2 (van der Corput) sequence and is
/// generated directly by [`sobol_direction_numbers`] without this table.
///
/// These values were extracted from `scipy`'s own
/// `_sobol_direction_numbers.npz` data file and the resulting sequences were
/// verified bit-for-bit against unscrambled `scipy.stats.qmc.Sobol` output
/// for dimensions 1, 2, 3, 5, 9, 17, 33 and 40.
#[rustfmt::skip]
const SOBOL_DIRECTION_DATA: [(u32, &[u32]); 39] = [
    (0,  &[1]),                              // dim=2,  s=1
    (1,  &[1, 3]),                           // dim=3,  s=2
    (1,  &[1, 3, 1]),                        // dim=4,  s=3
    (2,  &[1, 1, 1]),                        // dim=5,  s=3
    (1,  &[1, 1, 3, 3]),                     // dim=6,  s=4
    (4,  &[1, 3, 5, 13]),                    // dim=7,  s=4
    (2,  &[1, 1, 5, 5, 17]),                 // dim=8,  s=5
    (4,  &[1, 1, 5, 5, 5]),                  // dim=9,  s=5
    (7,  &[1, 1, 7, 11, 19]),                // dim=10, s=5
    (11, &[1, 1, 5, 1, 1]),                  // dim=11, s=5
    (13, &[1, 1, 1, 3, 11]),                 // dim=12, s=5
    (14, &[1, 3, 5, 5, 31]),                 // dim=13, s=5
    (1,  &[1, 3, 3, 9, 7, 49]),              // dim=14, s=6
    (13, &[1, 1, 1, 15, 21, 21]),            // dim=15, s=6
    (16, &[1, 3, 1, 13, 27, 49]),            // dim=16, s=6
    (19, &[1, 1, 1, 15, 7, 5]),              // dim=17, s=6
    (22, &[1, 3, 1, 15, 13, 25]),            // dim=18, s=6
    (25, &[1, 1, 5, 5, 19, 61]),             // dim=19, s=6
    (1,  &[1, 3, 7, 11, 23, 15, 103]),       // dim=20, s=7
    (4,  &[1, 3, 7, 13, 13, 15, 69]),        // dim=21, s=7
    (7,  &[1, 1, 3, 13, 7, 35, 63]),         // dim=22, s=7
    (8,  &[1, 3, 5, 9, 1, 25, 53]),          // dim=23, s=7
    (14, &[1, 3, 1, 13, 9, 35, 107]),        // dim=24, s=7
    (19, &[1, 3, 1, 5, 27, 61, 31]),         // dim=25, s=7
    (21, &[1, 1, 5, 11, 19, 41, 61]),        // dim=26, s=7
    (28, &[1, 3, 5, 3, 3, 13, 69]),          // dim=27, s=7
    (31, &[1, 1, 7, 13, 1, 19, 1]),          // dim=28, s=7
    (32, &[1, 3, 7, 5, 13, 19, 59]),         // dim=29, s=7
    (37, &[1, 1, 3, 9, 25, 29, 41]),         // dim=30, s=7
    (41, &[1, 3, 5, 13, 23, 1, 55]),         // dim=31, s=7
    (42, &[1, 3, 7, 3, 13, 59, 17]),         // dim=32, s=7
    (50, &[1, 3, 1, 3, 5, 53, 69]),          // dim=33, s=7
    (55, &[1, 1, 5, 5, 23, 33, 13]),         // dim=34, s=7
    (56, &[1, 1, 7, 7, 1, 61, 123]),         // dim=35, s=7
    (59, &[1, 1, 7, 9, 13, 61, 49]),         // dim=36, s=7
    (62, &[1, 3, 3, 5, 3, 55, 33]),          // dim=37, s=7
    (14, &[1, 3, 1, 15, 31, 13, 49, 245]),   // dim=38, s=8
    (21, &[1, 3, 5, 15, 31, 59, 63, 97]),    // dim=39, s=8
    (22, &[1, 3, 1, 11, 11, 11, 77, 249]),   // dim=40, s=8
];

/// Build the `SOBOL_BITS`-bit direction numbers `V[1..=SOBOL_BITS]` for one
/// Sobol dimension (1-indexed, `1..=SOBOL_MAX_DIM`).
///
/// `dim == 1` is the trivial base-2 sequence: `V[j] = 1 << (SOBOL_BITS - j)`
/// directly. For `dim >= 2`, the standard Bratley-Fox / Joe-Kuo recurrence is
/// used with the degree-`s` primitive polynomial and initial direction
/// numbers `m_1..m_s` looked up from [`SOBOL_DIRECTION_DATA`]:
///
/// ```text
/// V[j] = m_j << (SOBOL_BITS - j)                                    for j <= s
/// V[j] = V[j-s] ^ (V[j-s] >> s) ^ XOR_{k=1}^{s-1} (a_k * V[j-k])     for j >  s
/// ```
///
/// where `a_k` is bit `s - 1 - k` of the packed polynomial coefficient.
/// Returned as a plain array indexed `0..SOBOL_BITS`, where index `j - 1`
/// holds `V[j]`.
fn sobol_direction_numbers(dim: usize) -> [u32; SOBOL_BITS] {
    let mut v = [0u32; SOBOL_BITS];

    if dim == 1 {
        for j in 1..=SOBOL_BITS {
            v[j - 1] = 1u32 << (SOBOL_BITS - j);
        }
        return v;
    }

    let (a, m) = SOBOL_DIRECTION_DATA[dim - 2];
    let s = m.len();

    for (j, &m_j) in (1..=s).zip(m.iter()) {
        v[j - 1] = m_j << (SOBOL_BITS - j);
    }

    for j in (s + 1)..=SOBOL_BITS {
        let mut val = v[j - 1 - s] ^ (v[j - 1 - s] >> s);
        for k in 1..s {
            let bit = (a >> (s - 1 - k)) & 1;
            if bit != 0 {
                val ^= v[j - 1 - k];
            }
        }
        v[j - 1] = val;
    }

    v
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(vec, shape)
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

        Array::from_vec_shape(result, &[size, n])
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

        // Project the random symmetric matrix onto the nearest valid
        // correlation matrix (symmetric, PSD, unit diagonal) using Higham's
        // alternating projections algorithm.
        let sym_array = Array::from_vec_shape(sym_matrix, &[n, n])?;
        nearest_correlation_matrix(&sym_array)
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

        Array::from_vec_shape(vec, shape)
    }

    /// Generate Sobol sequence for quasi-Monte Carlo methods
    ///
    /// Uses the Joe & Kuo (2008) "new-joe-kuo-6.21201" primitive polynomials and
    /// initial direction numbers (search criterion 6) for dimensions 1 through 40,
    /// matching the unscrambled output of `scipy.stats.qmc.Sobol` bit-for-bit.
    /// This is a fully deterministic, low-discrepancy sequence: no part of it
    /// falls back to pseudorandom values, and no random state is consumed.
    pub fn sobol_sequence<T>(&self, dim: usize, n: usize) -> Result<Array<T>>
    where
        T: Float + NumCast + Clone + Debug + Display,
    {
        if !(1..=SOBOL_MAX_DIM).contains(&dim) {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "Dimension must be between 1 and {}, got {}",
                SOBOL_MAX_DIM, dim
            )));
        }

        if n < 1 {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "Number of points must be at least 1".to_string(),
            ));
        }

        // Build the full L-bit direction-number table for each requested dimension.
        let direction_numbers: Vec<[u32; SOBOL_BITS]> =
            (1..=dim).map(sobol_direction_numbers).collect();

        let scale = 2f64.powi(SOBOL_BITS as i32);
        let mut result = vec![T::zero(); n * dim];

        // Generate Sobol sequence points via the Gray-code construction:
        // X_i = XOR of V[j] over every bit (j-1) set in Gray(i) = i ^ (i >> 1).
        for i in 0..n {
            let g = i ^ (i >> 1);

            for (d, v) in direction_numbers.iter().enumerate() {
                let mut x = 0u32;
                for j in 1..=SOBOL_BITS {
                    if (g >> (j - 1)) & 1 != 0 {
                        x ^= v[j - 1];
                    }
                }

                // Convert to [0,1) range
                let val = <T as NumCast>::from(x as f64 / scale).unwrap_or(T::zero());
                result[i * dim + d] = val;
            }
        }

        Array::from_vec_shape(result, &[n, dim])
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

        Array::from_vec_shape(result, &[n, dim])
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

        Array::from_vec_shape(result, &[n, dim])
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

/// Error function.
///
/// Delegates to the crate's shared high-precision implementation
/// (`new_modules::special::error_functions::erf_scalar`) so the crate has a
/// single erf accuracy instead of a second, lower-precision copy living here.
fn erf(x: f64) -> f64 {
    erf_scalar(x)
}

/// Student's t-distribution cumulative distribution function.
///
/// Computed via the regularized incomplete beta function `I_x(a, b)`:
///
/// ```text
/// x_beta = df / (df + x^2)
/// cdf    = 1 - 0.5 * I_{x_beta}(df/2, 1/2)   for x > 0
/// cdf    =     0.5 * I_{x_beta}(df/2, 1/2)   for x < 0
/// cdf    = 0.5                               for x == 0
/// ```
///
/// which is exact for all `df > 0` (no large-`df` normal approximation is
/// needed, unlike the previous implementation here).
fn student_t_cdf(x: f64, df: usize) -> f64 {
    if x == 0.0 {
        return 0.5;
    }

    let df_f64 = df as f64;
    if df_f64 <= 0.0 {
        return 0.5;
    }

    let x_beta = df_f64 / (df_f64 + x * x);
    let beta_val = match betainc_regularized(x_beta, df_f64 / 2.0, 0.5) {
        Ok(v) => v,
        Err(_) => return 0.5,
    };

    let cdf = if x > 0.0 {
        1.0 - 0.5 * beta_val
    } else {
        0.5 * beta_val
    };

    cdf.clamp(0.0, 1.0)
}

/// Convergence tolerance for [`nearest_correlation_matrix`]'s alternating
/// projections: iteration stops once the Frobenius norm of the change
/// between successive iterates drops below this value.
const NEAREST_CORRELATION_TOL: f64 = 1e-10;

/// Maximum number of alternating-projection iterations for
/// [`nearest_correlation_matrix`] before giving up and returning an error.
const NEAREST_CORRELATION_MAX_ITER: usize = 200;

/// Project an arbitrary symmetric matrix onto the nearest valid correlation
/// matrix (symmetric, positive semi-definite, unit diagonal) in Frobenius
/// norm.
///
/// Implements Higham's (2002) alternating projections algorithm with
/// Dykstra's correction: at each iteration the current iterate `Y_k` (minus
/// the accumulated correction `ΔS_k`) is projected onto the cone of
/// positive-semidefinite matrices (via eigenvalue clipping), the correction
/// is updated, and the result is projected onto the affine set of
/// unit-diagonal matrices (by resetting the diagonal to 1). Iteration stops
/// once `‖Y_k - Y_{k-1}‖_F < 1e-10`, or fails with an error after 200
/// iterations.
///
/// # Errors
///
/// Returns an error if `matrix` is not square or empty, or if the iteration
/// fails to converge within 200 iterations.
pub fn nearest_correlation_matrix<T>(matrix: &Array<T>) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "nearest_correlation_matrix requires a square matrix".to_string(),
        ));
    }
    let n = shape[0];
    if n == 0 {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "nearest_correlation_matrix requires a non-empty matrix".to_string(),
        ));
    }

    // Work in f64 for numerically robust eigendecomposition, converting back
    // to T only for the final result.
    let a_data: Vec<f64> = matrix
        .to_vec()
        .into_iter()
        .map(|v| v.to_f64().unwrap_or(0.0))
        .collect();

    let mut y = a_data;
    let mut delta_s = vec![0.0f64; n * n];
    let mut converged = false;

    for _ in 0..NEAREST_CORRELATION_MAX_ITER {
        // R_k = Y_k - ΔS_k
        let r: Vec<f64> = y
            .iter()
            .zip(delta_s.iter())
            .map(|(&yv, &dv)| yv - dv)
            .collect();

        // PSD projection of R_k via eigenvalue clipping: R = V diag(λ) V^T,
        // X = V diag(max(λ, 0)) V^T.
        let r_array = Array::from_vec_shape(r.clone(), &[n, n])?;
        let (eigenvalues, eigenvectors) =
            StableDecompositions::symmetric_eigendecomposition(&r_array)?;
        let evec: Vec<f64> = eigenvectors.to_vec();
        let clipped: Vec<f64> = eigenvalues.iter().map(|&e: &f64| e.max(0.0)).collect();

        // Reconstruct X = V * diag(clipped) * V^T (eigenvectors are the
        // columns of `eigenvectors`, i.e. evec[i * n + k] == V[i, k]).
        let mut x = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0f64;
                for (k, &lambda) in clipped.iter().enumerate() {
                    s += evec[i * n + k] * lambda * evec[j * n + k];
                }
                x[i * n + j] = s;
                x[j * n + i] = s;
            }
        }

        // ΔS_{k+1} = X_k - R_k
        for idx in 0..(n * n) {
            delta_s[idx] = x[idx] - r[idx];
        }

        // Y_{k+1} = projection of X_k onto the unit-diagonal affine set.
        let mut y_new = x;
        for i in 0..n {
            y_new[i * n + i] = 1.0;
        }

        // Convergence check: Frobenius norm of the iterate-to-iterate change.
        let diff: f64 = y_new
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        y = y_new;

        if diff < NEAREST_CORRELATION_TOL {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(crate::error::NumRs2Error::NumericalError(format!(
            "nearest_correlation_matrix did not converge within {} iterations",
            NEAREST_CORRELATION_MAX_ITER
        )));
    }

    let result: Vec<T> = y
        .into_iter()
        .map(|v| <T as NumCast>::from(v).unwrap_or(T::zero()))
        .collect();

    Array::from_vec_shape(result, &[n, n])
}

// Enhanced distributions directly exported by the parent module, no need to re-export here

// Unit tests for enhanced distributions
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_truncated_normal() {
        let mean = 0.0;
        let std = 1.0;
        let low = -2.0;
        let high = 2.0;

        // Generate truncated normal samples (reduced size for performance)
        let samples = truncated_normal(mean, std, low, high, &[100])
            .expect("test: truncated_normal should succeed");
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
    #[serial]
    fn test_vonmises() {
        let mu = 0.0;
        let kappa = 2.0;

        // Generate von Mises samples
        let samples = vonmises(mu, kappa, &[1000]).expect("test: vonmises should succeed");
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
    #[serial]
    fn test_latin_hypercube() {
        let dim = 2;
        let n = 10;

        // Generate Latin Hypercube samples
        let samples = latin_hypercube::<f64>(dim, n).expect("test: latin_hypercube should succeed");
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
    #[serial]
    fn test_mixture_of_normals() {
        // Was `#[ignore]`d as "too slow for CI", which was never true: the
        // impl above (`RandomState::mixture_of_normals`) is O(size *
        // n_components) -- generate `size` component selections, batch-draw
        // normals per component, reassign in order -- microseconds even at
        // the original size=100. The real problem was statistical, not
        // performance: with weights=[0.3, 0.7] and no seed, the number of
        // draws from the -3 component is ~Binomial(100, 0.3), and
        // `p25 = sorted[25]` needs at least ~26 of those to land negative
        // for the assertion to hold. P(Binomial(100, 0.3) <= 25) ~= 16% --
        // an honest ~1-in-6 flake with an *unseeded* global RNG, misdiagnosed
        // as a perf problem at some point and `#[ignore]`d instead of fixed.
        //
        // Fixed by (a) seeding the global RNG explicitly -- deterministic
        // per the CI policy on stochastic tests -- and (b) raising size to
        // 1000, which pushes the analogous tail probability to ~3e-4
        // (mean 300, std ~14.5 for the same binomial, needing <=250) so the
        // seed has comfortable margin rather than merely happening to pass.
        // Order-of-magnitude estimate: the sorted index (250 of ~300 draws
        // from the -3 component) sits near that component's own 83rd
        // percentile, i.e. roughly -3 + 0.97*std ~= -2.0 (symmetric +2.0 for
        // p75) -- tightened the assertions accordingly, from the original
        // "merely negative" (a razor-thin margin at the 0.0 line) to
        // comfortably inside each component's mass.
        crate::random::distributions::set_seed(20260825);

        let weights = vec![0.3, 0.7];
        let means = vec![-3.0, 3.0];
        let stds = vec![1.0, 1.0];

        let samples = mixture_of_normals(&weights, &means, &stds, &[1000])
            .expect("test: mixture_of_normals should succeed");

        let data = samples.to_vec();

        // Check that distribution is bimodal
        // Sort the data for percentile calculation
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("test: f64 comparison should succeed for non-NaN values")
        });

        // Calculate 25th and 75th percentiles
        let p25 = sorted_data[(0.25 * sorted_data.len() as f64) as usize];
        let p75 = sorted_data[(0.75 * sorted_data.len() as f64) as usize];

        // For a bimodal distribution with these parameters, the 25th
        // percentile should sit well inside the -3 component's mass and the
        // 75th percentile well inside the +3 component's mass -- not merely
        // on the correct side of zero (see the margin note above).
        assert!(
            p25 < -0.5,
            "25th percentile should be clearly negative, got {}",
            p25
        );
        assert!(
            p75 > 0.5,
            "75th percentile should be clearly positive, got {}",
            p75
        );
    }

    // ------------------------------------------------------------------
    // erf: crate-wide single accuracy (delegates to
    // new_modules::special::error_functions::erf_scalar).
    //
    // erf_scalar uses a full-precision Taylor series for |x| <= 0.5 and a
    // Cody rational minimax approximation for |x| > 4 (both ~1e-15), but
    // its `erfc_positive` branch for 0.5 < |x| <= 4 still uses the A&S
    // 7.1.26 rational approximation (~1e-7 absolute error) — see
    // src/new_modules/special/error_functions.rs:167-180. erf(1) and
    // erf(2) below fall in that mid-range branch, so their tolerance
    // reflects what is actually achievable today rather than the crate's
    // best-case precision (asserted tightly at erf(0.5) and erf(5), which
    // exercise the high-precision branches).
    // ------------------------------------------------------------------

    #[test]
    fn test_erf_reference_values() {
        // scipy.special.erf / math.erf reference values.
        let cases_high_precision: [(f64, f64, f64); 3] = [
            (0.0, 0.0, 1e-15),
            (0.5, 0.5204998778130465, 1e-12),
            (5.0, 0.9999999999984626, 1e-12),
        ];
        for (x, expected, tol) in cases_high_precision {
            let got = erf(x);
            assert!(
                (got - expected).abs() < tol,
                "erf({x}) = {got}, expected {expected} within {tol}"
            );
        }

        // Mid-range (0.5 < |x| <= 4): erfc_positive's A&S 7.1.26 branch,
        // ~1e-7 absolute error. See comment above.
        let cases_mid_range: [(f64, f64, f64); 2] = [
            (1.0, 0.8427007929497148, 2e-7),
            (2.0, 0.9953222650189527, 2e-7),
        ];
        for (x, expected, tol) in cases_mid_range {
            let got = erf(x);
            assert!(
                (got - expected).abs() < tol,
                "erf({x}) = {got}, expected {expected} within {tol}"
            );
        }
    }

    #[test]
    fn test_erf_odd_symmetry() {
        for &x in &[0.3, 1.0, 2.0, 3.7] {
            assert!(
                (erf(-x) + erf(x)).abs() < 1e-12,
                "erf should be odd at x={x}"
            );
        }
    }

    // ------------------------------------------------------------------
    // student_t_cdf: regularized incomplete beta reimplementation.
    // Reference values from scipy.stats.t.cdf.
    // ------------------------------------------------------------------

    #[test]
    fn test_student_t_cdf_reference_values() {
        // (df, x, expected, tolerance)
        let cases: [(usize, f64, f64, f64); 6] = [
            (1, 1.0, 0.75, 1e-9),  // Cauchy: exact
            (1, -1.0, 0.25, 1e-9), // Cauchy: exact
            (2, 0.0, 0.5, 1e-12),
            (10, 1.812, 0.9499623689670764, 1e-9),
            (5, 2.5, 0.9727549503288119, 1e-9),
            (3, -2.0, 0.06966298427942152, 1e-9),
        ];
        for (df, x, expected, tol) in cases {
            let got = student_t_cdf(x, df);
            assert!(
                (got - expected).abs() < tol,
                "student_t_cdf(x={x}, df={df}) = {got}, expected {expected} within {tol}"
            );
        }
    }

    #[test]
    fn test_student_t_cdf_large_df_close_to_normal() {
        // As df -> infinity, the t distribution approaches the standard
        // normal. scipy: t.cdf(1.96, 200) = 0.9743075795770934,
        // norm.cdf(1.96) = 0.9750021048517795 (diff ~6.9e-4).
        let got = student_t_cdf(1.96, 200);
        let expected = 0.9743075795770934;
        assert!(
            (got - expected).abs() < 1e-6,
            "student_t_cdf(1.96, 200) = {got}, expected {expected}"
        );
        let normal = normal_cdf(1.96);
        assert!(
            (got - normal).abs() < 1e-2,
            "df=200 t CDF should be close to normal CDF: t={got}, normal={normal}"
        );
    }

    #[test]
    fn test_student_t_cdf_large_x_saturates() {
        // Large |x| should saturate towards 1 / 0 without producing NaN or
        // an out-of-range value (regression for copula()'s use of
        // student_t_cdf with large z-scores).
        let hi = student_t_cdf(100.0, 4);
        let lo = student_t_cdf(-100.0, 4);
        assert!((0.0..=1.0).contains(&hi), "hi={hi} out of [0,1]");
        assert!((0.0..=1.0).contains(&lo), "lo={lo} out of [0,1]");
        assert!(hi > 0.999, "hi={hi} should be close to 1");
        assert!(lo < 0.001, "lo={lo} should be close to 0");
        assert!(!hi.is_nan() && !lo.is_nan());
    }

    // ------------------------------------------------------------------
    // Sobol direction numbers: internal sanity checks on the embedded
    // Joe-Kuo table (the full scipy cross-validation and stratification
    // property tests live in tests/test_random_quality.rs, since they only
    // need the public `sobol_sequence` API).
    // ------------------------------------------------------------------

    #[test]
    fn test_sobol_direction_data_table_shape() {
        assert_eq!(SOBOL_DIRECTION_DATA.len(), SOBOL_MAX_DIM - 1);
        for (idx, (_, m)) in SOBOL_DIRECTION_DATA.iter().enumerate() {
            let dim = idx + 2;
            assert!(
                !m.is_empty() && m.len() <= 9,
                "dim={dim} has unexpected degree {}",
                m.len()
            );
            // Initial direction numbers m_i must be odd (Joe-Kuo requirement).
            for (i, &mi) in m.iter().enumerate() {
                assert_eq!(mi % 2, 1, "dim={dim} m_{} = {mi} must be odd", i + 1);
            }
        }
    }

    #[test]
    fn test_sobol_direction_numbers_dim1_is_trivial() {
        let v = sobol_direction_numbers(1);
        for (j, &vj) in v.iter().enumerate() {
            let expected = 1u32 << (SOBOL_BITS - (j + 1));
            assert_eq!(vj, expected, "V[{}] mismatch for trivial dim 1", j + 1);
        }
    }

    // ------------------------------------------------------------------
    // nearest_correlation_matrix: internal access is not required (the
    // function is pub), but a quick smoke test lives here too; the
    // detailed literature-matching test is in tests/test_random_quality.rs.
    // ------------------------------------------------------------------

    #[test]
    fn test_nearest_correlation_matrix_identity_is_fixed_point() {
        let identity =
            Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(&[3, 3]);
        let result = nearest_correlation_matrix(&identity)
            .expect("test: nearest_correlation_matrix should succeed on the identity");
        let data = result.to_vec();
        let expected = identity.to_vec();
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got={got}, want={want}");
        }
    }
}

//! SciRS2 compatibility module for advanced statistical distributions
//!
//! This module provides compatibility layers and adapters for using SciRS2's
//! advanced statistical distributions within NumRS2. It enables seamless
//! integration with SciRS2 while maintaining the NumRS2 API.
//!
//! Note: This module is currently being updated for SciRS2 v0.1.0-alpha.4 compatibility

#![allow(unexpected_cfgs)]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use std::fmt::{Debug, Display};

// When the scirs feature is enabled, import the necessary modules from SciRS2
// Note: Currently disabled while updating for SciRS2 v0.1.0-alpha.4 API changes
#[cfg(feature = "__never")]
use {
    // Import distributions from SciRS2
    scirs2_stats::distributions::continuous::*,

    // Import array and generator types
    scirs2_core::ndarray_ext::Array as SciArray,
    scirs2_core::random::Generator as SciGenerator,

    // Import SCIRS linalg for linear equation solving
    scirs2_linalg::decomposition::lu_solve,

    // Import random number generation
    rand::Rng,
    rand::SeedableRng,
    rand::rngs::StdRng,
};

// Conversion helpers between NumRS2 and SciRS2 arrays
#[cfg(feature = "__never")]
fn convert_to_numrs_array<T>(sci_array: SciArray<f64>, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    let data = sci_array.to_vec();
    let mut result = Vec::with_capacity(data.len());

    for &val in &data {
        let converted = T::from(val).ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert SciRS2 array to NumRS2 array type".to_string())
        })?;
        result.push(converted);
    }

    Ok(Array::from_vec(result).reshape(shape))
}

/// Adapter function for SciRS2's noncentral chi-square distribution
///
/// This function creates a noncentral chi-square distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `df` - Degrees of freedom (> 0)
/// * `nonc` - Non-centrality parameter (≥ 0)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the noncentral chi-square distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn noncentral_chisquare<T>(df: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Convert parameters to f64 for SciRS2
    let df_f64 = df.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert df to f64".to_string())
    })?;

    let nonc_f64 = nonc.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert nonc to f64".to_string())
    })?;

    // Check parameter validity
    if df_f64 <= 0.0 || nonc_f64 < 0.0 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Parameters must satisfy df > 0 and nonc >= 0, got df = {}, nonc = {}", df, nonc)
        ));
    }

    // Create the noncentral chi-square distribution using SciRS2
    let dist = match NoncentralChiSquared::new(df_f64, nonc_f64) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create noncentral chi-square distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Calculate the total size of the output array
    let size: usize = shape.iter().product();

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, size) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating noncentral chi-square samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn noncentral_chisquare<T>(_df: T, _nonc: T, _shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Noncentral chi-square distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's noncentral F distribution
///
/// This function creates a noncentral F distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `dfnum` - Numerator degrees of freedom (> 0)
/// * `dfden` - Denominator degrees of freedom (> 0)
/// * `nonc` - Non-centrality parameter (≥ 0)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the noncentral F distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn noncentral_f<T>(dfnum: T, dfden: T, nonc: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Convert parameters to f64 for SciRS2
    let dfnum_f64 = dfnum.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert dfnum to f64".to_string())
    })?;

    let dfden_f64 = dfden.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert dfden to f64".to_string())
    })?;

    let nonc_f64 = nonc.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert nonc to f64".to_string())
    })?;

    // Check parameter validity
    if dfnum_f64 <= 0.0 || dfden_f64 <= 0.0 || nonc_f64 < 0.0 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Parameters must satisfy dfnum > 0, dfden > 0, and nonc >= 0, got dfnum = {}, dfden = {}, nonc = {}",
                    dfnum, dfden, nonc)
        ));
    }

    // Create the noncentral F distribution using SciRS2
    let dist = match FNoncentral::new(dfnum_f64, dfden_f64, nonc_f64) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create noncentral F distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Calculate the total size of the output array
    let size: usize = shape.iter().product();

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, size) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating noncentral F samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn noncentral_f<T>(_dfnum: T, _dfden: T, _nonc: T, _shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Noncentral F distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's von Mises distribution
///
/// This function creates a von Mises distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `mu` - Location parameter (circular mean direction)
/// * `kappa` - Concentration parameter (≥ 0)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the von Mises distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn vonmises<T>(mu: T, kappa: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Convert parameters to f64 for SciRS2
    let mu_f64 = mu.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert mu to f64".to_string())
    })?;

    let kappa_f64 = kappa.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert kappa to f64".to_string())
    })?;

    // Check parameter validity
    if kappa_f64 < 0.0 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Concentration parameter kappa must be >= 0, got kappa = {}", kappa)
        ));
    }

    // Create the von Mises distribution using SciRS2
    let dist = match VonMises::new(mu_f64, kappa_f64) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create von Mises distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Calculate the total size of the output array
    let size: usize = shape.iter().product();

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, size) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating von Mises samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn vonmises<T>(_mu: T, _kappa: T, _shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Von Mises distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's Maxwell distribution
///
/// This function creates a Maxwell distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `scale` - Scale parameter (> 0)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the Maxwell distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn maxwell<T>(scale: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Convert parameters to f64 for SciRS2
    let scale_f64 = scale.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert scale to f64".to_string())
    })?;

    // Check parameter validity
    if scale_f64 <= 0.0 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Scale parameter must be > 0, got scale = {}", scale)
        ));
    }

    // Create the Maxwell distribution using SciRS2
    let dist = match MaxwellBoltzmann::new(scale_f64) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create Maxwell distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Calculate the total size of the output array
    let size: usize = shape.iter().product();

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, size) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating Maxwell samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn maxwell<T>(_scale: T, _shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Maxwell distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's multivariate normal distribution with factor rotation
///
/// This enhanced version of multivariate normal adds factor rotation capabilities,
/// useful for generating correlated random variables with specific properties.
///
/// # Arguments
///
/// * `means` - Mean vector
/// * `cov` - Covariance matrix
/// * `size` - Optional shape of the output array
/// * `rotation` - Optional rotation matrix for factor loading
///
/// # Returns
///
/// * A NumRS2 array with samples from the rotated multivariate normal distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn multivariate_normal_with_rotation<T>(
    means: &[T],
    cov: &Array<T>,
    size: Option<&[usize]>,
    rotation: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    if means.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Mean vector cannot be empty".to_string()
        ));
    }

    let n = means.len();
    let cov_shape = cov.shape();

    if cov_shape.len() != 2 || cov_shape[0] != n || cov_shape[1] != n {
        return Err(NumRs2Error::InvalidOperation(
            format!("Covariance matrix must be square with dimensions matching mean vector length ({}), got shape {:?}", n, cov_shape)
        ));
    }

    // Convert parameters to f64 for SciRS2
    let means_f64: Vec<f64> = means.iter().map(|&m| {
        m.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert mean to f64".to_string())
        })
    }).collect::<Result<Vec<f64>>>()?;

    let cov_data = cov.to_vec();
    let cov_data_f64: Vec<f64> = cov_data.iter().map(|&c| {
        c.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert covariance to f64".to_string())
        })
    }).collect::<Result<Vec<f64>>>()?;

    // Build SciRS2 arrays
    let means_sci = SciArray::from_vec(means_f64);
    let cov_sci = SciArray::from_vec(cov_data_f64).reshape(&[n, n]);

    let rotation_sci = if let Some(rot) = rotation {
        let rot_shape = rot.shape();
        let rot_data = rot.to_vec();

        let rot_data_f64: Vec<f64> = rot_data.iter().map(|&r| {
            r.to_f64().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to convert rotation matrix to f64".to_string())
            })
        }).collect::<Result<Vec<f64>>>()?;

        Some(SciArray::from_vec(rot_data_f64).reshape(&rot_shape))
    } else {
        None
    };

    // Create the multivariate normal distribution using SciRS2
    let dist = match MultivariateNormal::new(&means_sci, &cov_sci, rotation_sci.as_ref()) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create multivariate normal distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Determine number of samples and output shape
    let (num_samples, out_shape) = if let Some(shape_spec) = size {
        let samples = shape_spec.iter().product();
        let mut shape_with_dim = shape_spec.to_vec();
        shape_with_dim.push(n);
        (samples, shape_with_dim)
    } else {
        (1, vec![n])
    };

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, num_samples) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating multivariate normal samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, &out_shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn multivariate_normal_with_rotation<T>(
    _means: &[T],
    _cov: &Array<T>,
    _size: Option<&[usize]>,
    _rotation: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Multivariate normal with rotation requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's truncated normal distribution
///
/// This function creates a truncated normal distribution using SciRS2
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `mean` - Mean of the underlying normal distribution
/// * `std` - Standard deviation of the underlying normal distribution
/// * `low` - Lower bound for truncation (can be -inf)
/// * `high` - Upper bound for truncation (can be +inf)
/// * `shape` - Shape of the output array
///
/// # Returns
///
/// * A NumRS2 array with samples from the truncated normal distribution
///
/// # Errors
///
/// Returns an error if the distribution parameters are invalid or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn truncated_normal<T>(mean: T, std: T, low: T, high: T, shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Convert parameters to f64 for SciRS2
    let mean_f64 = mean.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert mean to f64".to_string())
    })?;

    let std_f64 = std.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert std to f64".to_string())
    })?;

    let low_f64 = low.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert low to f64".to_string())
    })?;

    let high_f64 = high.to_f64().ok_or_else(|| {
        NumRs2Error::InvalidOperation("Failed to convert high to f64".to_string())
    })?;

    // Check parameter validity
    if std_f64 <= 0.0 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Standard deviation must be > 0, got std = {}", std)
        ));
    }

    if low_f64 >= high_f64 {
        return Err(NumRs2Error::InvalidOperation(
            format!("Lower bound must be < upper bound, got low = {}, high = {}", low, high)
        ));
    }

    // Create the truncated normal distribution using SciRS2
    let dist = match TruncatedNormal::new(mean_f64, std_f64, low_f64, high_f64) {
        Ok(d) => d,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to create truncated normal distribution: {}", e)
        )),
    };

    // Create a generator with SciRS2 (using a random seed)
    let rng = StdRng::seed_from_u64(rand::random::<u64>());
    let gen = SciGenerator::new(rng);

    // Calculate the total size of the output array
    let size: usize = shape.iter().product();

    // Generate samples using SciRS2
    let samples = match dist.sample_array(&gen, size) {
        Ok(arr) => arr,
        Err(e) => return Err(NumRs2Error::RuntimeError(
            format!("Error generating truncated normal samples: {}", e)
        )),
    };

    // Convert SciRS2 array to NumRS2 array
    convert_to_numrs_array(samples, shape)
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn truncated_normal<T>(_mean: T, _std: T, _low: T, _high: T, _shape: &[usize]) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Truncated normal distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

/// Adapter function for SciRS2's linear equation solver
///
/// This function solves a linear system Ax = b using SciRS2's LU decomposition
/// and converts the results to NumRS2 arrays.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
///
/// # Returns
///
/// * A NumRS2 array representing the solution vector x
///
/// # Errors
///
/// Returns an error if the matrix is singular or if the
/// SciRS2 integration fails.
#[cfg(feature = "__never")]
pub fn solve_linear_system<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    // Check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "solve_linear_system requires a square coefficient matrix".to_string()
        ));
    }

    if b_shape.len() != 1 || b_shape[0] != a_shape[0] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a_shape[0]],
            actual: b_shape.to_vec(),
        });
    }

    // Convert a to SCIRS array
    let a_data = a.to_vec();
    let a_data_f64: Vec<f64> = a_data.iter().map(|&c| {
        c.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert matrix elements to f64".to_string())
        })
    }).collect::<Result<Vec<f64>>>()?;

    // Convert b to SCIRS array
    let b_data = b.to_vec();
    let b_data_f64: Vec<f64> = b_data.iter().map(|&c| {
        c.to_f64().ok_or_else(|| {
            NumRs2Error::InvalidOperation("Failed to convert vector elements to f64".to_string())
        })
    }).collect::<Result<Vec<f64>>>()?;

    // Create SCIRS arrays
    let a_sci = SciArray::from_vec(a_data_f64).reshape(&a_shape);
    let b_sci = SciArray::from_vec(b_data_f64).reshape(&b_shape);

    // Solve the system using SCIRS
    let x_sci = match lu_solve(&a_sci, &b_sci) {
        Ok(x) => x,
        Err(e) => return Err(NumRs2Error::InvalidOperation(
            format!("Failed to solve linear system with SCIRS: {}", e)
        )),
    };

    // Convert SCIRS result back to NumRS array
    convert_to_numrs_array(x_sci, &[a_shape[0]])
}

/// Placeholder for non-SciRS2 build
#[cfg(not(feature = "__never"))]
pub fn solve_linear_system<T>(_a: &Array<T>, _b: &Array<T>) -> Result<Array<T>>
where
    T: Float + NumCast + Clone + Debug + Display,
{
    Err(NumRs2Error::NotImplemented(
        "Linear equation system solver requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.".to_string()
    ))
}

#[cfg(test)]
#[cfg(feature = "__never")]
mod tests {
    use super::*;

    #[test]
    fn test_noncentral_chisquare() {
        let samples = noncentral_chisquare(2.0f64, 1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // All samples should be positive
        for val in samples.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_noncentral_f() {
        let samples = noncentral_f(2.0f64, 3.0f64, 1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // All samples should be positive
        for val in samples.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_vonmises() {
        let samples = vonmises(0.0f64, 1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // Von Mises samples should be in the range [-π, π]
        for val in samples.to_vec() {
            assert!(val >= -std::f64::consts::PI && val <= std::f64::consts::PI);
        }
    }

    #[test]
    fn test_maxwell() {
        let samples = maxwell(1.0f64, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // Maxwell samples should be positive
        for val in samples.to_vec() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_truncated_normal() {
        let low = -2.0f64;
        let high = 2.0f64;
        let samples = truncated_normal(0.0f64, 1.0f64, low, high, &[10]).unwrap();
        assert_eq!(samples.shape(), vec![10]);

        // All samples should be within the truncation bounds
        for val in samples.to_vec() {
            assert!(val >= low && val <= high);
        }
    }

    #[test]
    fn test_multivariate_normal_with_rotation() {
        let mean = vec![0.0f64, 0.0f64];
        let cov_data = vec![1.0f64, 0.5f64, 0.5f64, 1.0f64];
        let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

        // Create a rotation matrix for 45 degrees
        let rotation_data = vec![
            0.7071f64, 0.7071f64,  // cos(45°), sin(45°)
            -0.7071f64, 0.7071f64  // -sin(45°), cos(45°)
        ];
        let rotation = Array::from_vec(rotation_data).reshape(&[2, 2]);

        let samples = multivariate_normal_with_rotation(&mean, &cov, Some(&[3]), Some(&rotation)).unwrap();
        assert_eq!(samples.shape(), vec![3, 2]);
    }

    #[test]
    fn test_solve_linear_system() {
        // Test a simple 2x2 system with known solution
        // System:
        //   2x + y = 5
        //   x + 3y = 7
        // Solution: x = 2, y = 1

        let a_data = vec![2.0f64, 1.0f64, 1.0f64, 3.0f64];
        let a = Array::from_vec(a_data).reshape(&[2, 2]);

        let b_data = vec![5.0f64, 7.0f64];
        let b = Array::from_vec(b_data);

        let x = solve_linear_system(&a, &b).unwrap();

        // Check shape
        assert_eq!(x.shape(), vec![2]);

        // Check solution values
        let x_data = x.to_vec();
        assert_relative_eq!(x_data[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x_data[1], 1.0, epsilon = 1e-10);

        // Test a 3x3 system
        // System:
        //   3x + 2y + z = 10
        //   2x + 5y + 3z = 15
        //   x + y + 4z = 12
        // Solution: x = 1, y = 1, z = 2

        let a_data = vec![
            3.0f64, 2.0f64, 1.0f64,
            2.0f64, 5.0f64, 3.0f64,
            1.0f64, 1.0f64, 4.0f64
        ];
        let a = Array::from_vec(a_data).reshape(&[3, 3]);

        let b_data = vec![10.0f64, 15.0f64, 12.0f64];
        let b = Array::from_vec(b_data);

        let x = solve_linear_system(&a, &b).unwrap();

        // Check shape
        assert_eq!(x.shape(), vec![3]);

        // Check solution values
        let x_data = x.to_vec();
        assert_relative_eq!(x_data[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x_data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x_data[2], 2.0, epsilon = 1e-10);

        // Test error handling for singular matrix
        let a_data = vec![1.0f64, 2.0f64, 2.0f64, 4.0f64]; // Singular matrix (second row is 2x first row)
        let a = Array::from_vec(a_data).reshape(&[2, 2]);

        let b_data = vec![3.0f64, 6.0f64];
        let b = Array::from_vec(b_data);

        // Should return an error for singular matrix
        assert!(solve_linear_system(&a, &b).is_err());

        // Test error handling for dimension mismatch
        let a_data = vec![1.0f64, 2.0f64, 3.0f64, 4.0f64];
        let a = Array::from_vec(a_data).reshape(&[2, 2]);

        let b_data = vec![5.0f64, 6.0f64, 7.0f64]; // Wrong size for b
        let b = Array::from_vec(b_data);

        // Should return an error for dimension mismatch
        assert!(solve_linear_system(&a, &b).is_err());
    }
}
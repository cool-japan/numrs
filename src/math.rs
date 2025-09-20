use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_complex::Complex;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::{Add, Div, Mul, Sub};

// Import SIMD optimizations when available
#[cfg(target_arch = "x86_64")]
use crate::simd_optimize::avx2_enhanced::EnhancedSimdOps;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)] // Used in SIMD operations but may not be visible to compiler
use std::any::TypeId;

/// Threshold for using SIMD optimizations (minimum array size)
const SIMD_THRESHOLD: usize = 32;

/// Check if SIMD optimization should be used for this array
#[inline]
fn should_use_simd<T: Clone>(array: &Array<T>) -> bool {
    array.len() >= SIMD_THRESHOLD
}

// Basic element-wise math operations
pub trait ElementWiseMath<T> {
    // Basic operations
    fn abs(&self) -> Array<T>;
    fn exp(&self) -> Array<T>;
    fn log(&self) -> Array<T>;
    fn log10(&self) -> Array<T>;
    fn log2(&self) -> Array<T>;
    fn log1p(&self) -> Array<T>;
    fn expm1(&self) -> Array<T>;
    fn sqrt(&self) -> Array<T>;
    fn cbrt(&self) -> Array<T>;
    fn pow(&self, n: T) -> Array<T>;

    // Log-domain functions
    fn logaddexp(&self, other: &Array<T>) -> Array<T>;
    fn logaddexp2(&self, other: &Array<T>) -> Array<T>;

    // Trigonometric functions
    fn sin(&self) -> Array<T>;
    fn cos(&self) -> Array<T>;
    fn tan(&self) -> Array<T>;
    fn asin(&self) -> Array<T>;
    fn acos(&self) -> Array<T>;
    fn atan(&self) -> Array<T>;
    fn atan2(&self, other: &Array<T>) -> Array<T>;
    fn hypot(&self, other: &Array<T>) -> Array<T>;
    fn degrees(&self) -> Array<T>;
    fn radians(&self) -> Array<T>;

    // Hyperbolic functions
    fn sinh(&self) -> Array<T>;
    fn cosh(&self) -> Array<T>;
    fn tanh(&self) -> Array<T>;
    fn asinh(&self) -> Array<T>;
    fn acosh(&self) -> Array<T>;
    fn atanh(&self) -> Array<T>;

    // Rounding functions
    fn floor(&self) -> Array<T>;
    fn ceil(&self) -> Array<T>;
    fn round(&self) -> Array<T>;
    fn trunc(&self) -> Array<T>;

    // Utility functions
    fn clip(&self, min: T, max: T) -> Array<T>;
    fn sign(&self) -> Array<T>;

    // Safe versions that return Result for error handling
    fn safe_logaddexp(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_logaddexp2(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_atan2(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_hypot(&self, other: &Array<T>) -> Result<Array<T>>;
}

impl<T: Float + Clone + 'static> ElementWiseMath<T> for Array<T> {
    // Basic operations
    fn abs(&self) -> Array<T> {
        self.map(|x| x.abs())
    }

    fn exp(&self) -> Array<T> {
        // Use SIMD optimization for f32 arrays when beneficial
        #[cfg(target_arch = "x86_64")]
        {
            if should_use_simd(self) {
                // Check if we can cast to f32 for SIMD optimization
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Safety: We've verified the type above
                    let f32_array = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                    let result = EnhancedSimdOps::vectorized_exp_f32(f32_array);
                    // Safety: We're transmuting back to the same type
                    return unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result) };
                }
            }
        }

        // Fallback to element-wise operation
        self.map(|x| x.exp())
    }

    fn log(&self) -> Array<T> {
        self.map(|x| x.ln())
    }

    fn log10(&self) -> Array<T> {
        self.map(|x| x.log10())
    }

    fn log2(&self) -> Array<T> {
        self.map(|x| x.log2())
    }

    fn log1p(&self) -> Array<T> {
        self.map(|x| (x + T::one()).ln())
    }

    fn expm1(&self) -> Array<T> {
        self.map(|x| x.exp() - T::one())
    }

    fn sqrt(&self) -> Array<T> {
        // Use SIMD optimization for f32 arrays when beneficial
        #[cfg(target_arch = "x86_64")]
        {
            if should_use_simd(self) && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            {
                // For sqrt, we can use the existing vectorized operations
                // Fall through to regular implementation for now
                // TODO: Implement SIMD sqrt optimization
            }
        }

        // Fallback to element-wise operation
        self.map(|x| x.sqrt())
    }

    fn cbrt(&self) -> Array<T> {
        self.map(|x| x.powf(T::from(1.0 / 3.0).unwrap()))
    }

    fn pow(&self, n: T) -> Array<T> {
        self.map(|x| x.powf(n))
    }

    // Log-domain functions
    fn logaddexp(&self, other: &Array<T>) -> Array<T> {
        // Implementing log(exp(x) + exp(y)) in a numerically stable way
        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let max_val = if a > b { a } else { b };
            let sum = (a - max_val).exp() + (b - max_val).exp();
            max_val + sum.ln()
        })
        .unwrap_or_else(|e| panic!("Failed to broadcast in logaddexp: {}. Consider using safe_logaddexp() for error handling.", e))
    }

    fn logaddexp2(&self, other: &Array<T>) -> Array<T> {
        // Implementing log2(2^x + 2^y) in a numerically stable way
        // log2(2^x + 2^y) = log2(e) * ln(2^x + 2^y) = log2(e) * ln(e^(x*ln(2)) + e^(y*ln(2)))
        let ln2 = T::from(std::f64::consts::LN_2).unwrap();
        let log2_e = T::from(std::f64::consts::LOG2_E).unwrap();

        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            // Convert log2 values to natural log
            let ln_a = a * ln2;
            let ln_b = b * ln2;

            let max_val = if ln_a > ln_b { ln_a } else { ln_b };
            let sum = (ln_a - max_val).exp() + (ln_b - max_val).exp();
            (max_val + sum.ln()) * log2_e
        })
        .unwrap_or_else(|e| panic!("Failed to broadcast in logaddexp2: {}. Consider using safe_logaddexp2() for error handling.", e))
    }

    // Trigonometric functions
    fn sin(&self) -> Array<T> {
        // Use SIMD optimization for f32 arrays when beneficial
        #[cfg(target_arch = "x86_64")]
        {
            if should_use_simd(self) && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            {
                // TODO: Implement SIMD sin optimization using log_abs_trig module
                // For now, fall through to regular implementation
            }
        }

        // Fallback to element-wise operation
        self.map(|x| x.sin())
    }

    fn cos(&self) -> Array<T> {
        self.map(|x| x.cos())
    }

    fn tan(&self) -> Array<T> {
        self.map(|x| x.tan())
    }

    fn asin(&self) -> Array<T> {
        self.map(|x| x.asin())
    }

    fn acos(&self) -> Array<T> {
        self.map(|x| x.acos())
    }

    fn atan(&self) -> Array<T> {
        self.map(|x| x.atan())
    }

    fn atan2(&self, other: &Array<T>) -> Array<T> {
        self.zip_with(other, |a, b| a.atan2(b)).unwrap_or_else(|e| {
            panic!(
                "Failed to broadcast in atan2: {}. Consider using safe_atan2() for error handling.",
                e
            )
        })
    }

    fn hypot(&self, other: &Array<T>) -> Array<T> {
        self.zip_with(other, |a, b| (a * a + b * b).sqrt())
            .unwrap_or_else(|e| panic!("Failed to broadcast in hypot: {}. Consider using safe_hypot() for error handling.", e))
    }

    fn degrees(&self) -> Array<T> {
        let rad_to_deg = T::from(180.0).unwrap() / T::from(std::f64::consts::PI).unwrap();
        self.map(|x| x * rad_to_deg)
    }

    fn radians(&self) -> Array<T> {
        let deg_to_rad = T::from(std::f64::consts::PI).unwrap() / T::from(180.0).unwrap();
        self.map(|x| x * deg_to_rad)
    }

    // Hyperbolic functions
    fn sinh(&self) -> Array<T> {
        self.map(|x| x.sinh())
    }

    fn cosh(&self) -> Array<T> {
        self.map(|x| x.cosh())
    }

    fn tanh(&self) -> Array<T> {
        self.map(|x| x.tanh())
    }

    fn asinh(&self) -> Array<T> {
        self.map(|x| x.asinh())
    }

    fn acosh(&self) -> Array<T> {
        self.map(|x| x.acosh())
    }

    fn atanh(&self) -> Array<T> {
        self.map(|x| x.atanh())
    }

    // Rounding functions
    fn floor(&self) -> Array<T> {
        self.map(|x| x.floor())
    }

    fn ceil(&self) -> Array<T> {
        self.map(|x| x.ceil())
    }

    fn round(&self) -> Array<T> {
        self.map(|x| x.round())
    }

    fn trunc(&self) -> Array<T> {
        self.map(|x| x.trunc())
    }

    // Utility functions
    fn clip(&self, min: T, max: T) -> Array<T> {
        self.map(|x| {
            if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            }
        })
    }

    fn sign(&self) -> Array<T> {
        self.map(|x| {
            if x == T::zero() {
                T::zero()
            } else if x > T::zero() {
                T::one()
            } else {
                -T::one()
            }
        })
    }

    // Safe versions that return Result for proper error handling
    fn safe_logaddexp(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let max_val = if a > b { a } else { b };
            let sum = (a - max_val).exp() + (b - max_val).exp();
            max_val + sum.ln()
        })
        .map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in logaddexp: {}",
                e
            ))
        })
    }

    fn safe_logaddexp2(&self, other: &Array<T>) -> Result<Array<T>> {
        let ln2 = T::from(std::f64::consts::LN_2).unwrap();
        let log2_e = T::from(std::f64::consts::LOG2_E).unwrap();

        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let a_scaled = a * ln2;
            let b_scaled = b * ln2;
            let max_val = if a_scaled > b_scaled {
                a_scaled
            } else {
                b_scaled
            };
            let sum = (a_scaled - max_val).exp() + (b_scaled - max_val).exp();
            (max_val + sum.ln()) * log2_e
        })
        .map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in logaddexp2: {}",
                e
            ))
        })
    }

    fn safe_atan2(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| a.atan2(b)).map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in atan2: {}",
                e
            ))
        })
    }

    fn safe_hypot(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| (a * a + b * b).sqrt())
            .map_err(|e| {
                crate::error::NumRs2Error::ComputationError(format!(
                    "Broadcasting failed in hypot: {}",
                    e
                ))
            })
    }
}

// Array creation functions (similar to NumPy's)
pub fn zeros<T: Zero + Clone>(shape: &[usize]) -> Array<T> {
    let size: usize = shape.iter().product();
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(T::zero());
    }
    Array::from_vec(vec).reshape(shape)
}

pub fn ones<T: One + Clone>(shape: &[usize]) -> Array<T> {
    let size: usize = shape.iter().product();
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(T::one());
    }
    Array::from_vec(vec).reshape(shape)
}

/// Create an array with uninitialized values
///
/// Note: This is similar to NumPy's empty but with safe Rust semantics.
/// The array will be initialized with default values instead of random memory.
pub fn empty<T: Default + Clone>(shape: &[usize]) -> Array<T> {
    let size: usize = shape.iter().product();
    let vec = vec![T::default(); size];
    Array::from_vec(vec).reshape(shape)
}

/// Create evenly spaced values between start and stop (inclusive)
pub fn linspace<T: Float + Clone>(start: T, stop: T, num: usize) -> Array<T> {
    if num < 2 {
        return Array::from_vec(vec![start]);
    }

    let mut vec = Vec::with_capacity(num);
    let step = (stop - start) / T::from(num - 1).unwrap();

    for i in 0..num {
        vec.push(start + step * T::from(i).unwrap());
    }

    Array::from_vec(vec)
}

/// Create a sequence of numbers with a specified step
pub fn arange<T>(start: T, stop: T, step: T) -> Array<T>
where
    T: Clone + PartialOrd + NumCast + Add<Output = T> + Zero,
{
    if step > T::zero() && start >= stop {
        return Array::from_vec(vec![]);
    }
    if step < T::zero() && start <= stop {
        return Array::from_vec(vec![]);
    }

    let mut vec = Vec::new();
    let mut current = start;

    if step > T::zero() {
        while current < stop {
            vec.push(current.clone());
            current = current + step.clone();
        }
    } else {
        while current > stop {
            vec.push(current.clone());
            current = current + step.clone();
        }
    }

    Array::from_vec(vec)
}

/// Create evenly spaced numbers on a logarithmic scale
pub fn logspace<T: Float + Clone>(start: T, stop: T, num: usize, base: Option<T>) -> Array<T> {
    let base_val = base.unwrap_or_else(|| T::from(10.0).unwrap());

    // Generate powers as a linear space
    let powers = linspace(start, stop, num);

    // Apply base^power to each element
    powers.map(move |x| base_val.powf(x))
}

/// Create a mesh grid from arrays with different indexing modes
///
/// # Parameters
///
/// * `arrays` - A vector of arrays to create a mesh grid from
/// * `indexing` - The indexing mode: "xy" (default) or "ij"
///
/// # Returns
///
/// A vector of arrays, each with the shape of the meshgrid
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::meshgrid;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let y = Array::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
///
/// // xy indexing (default)
/// let grids = meshgrid(&[&x, &y], None).unwrap();
/// let xx = &grids[0];
/// let yy = &grids[1];
/// assert_eq!(xx.shape(), vec![4, 3]);  // (y.len(), x.len())
/// assert_eq!(yy.shape(), vec![4, 3]);
///
/// // Check specific values
/// assert_eq!(xx.get(&[0, 0]).unwrap(), 1.0);
/// assert_eq!(xx.get(&[0, 1]).unwrap(), 2.0);
/// assert_eq!(xx.get(&[0, 2]).unwrap(), 3.0);
/// assert_eq!(yy.get(&[0, 0]).unwrap(), 4.0);
/// assert_eq!(yy.get(&[1, 0]).unwrap(), 5.0);
///
/// // ij indexing
/// let grids_ij = meshgrid(&[&x, &y], Some("ij")).unwrap();
/// let ii = &grids_ij[0];
/// let jj = &grids_ij[1];
/// assert_eq!(ii.shape(), vec![3, 4]);  // (x.len(), y.len())
/// assert_eq!(jj.shape(), vec![3, 4]);
///
/// // Check specific values for ij indexing
/// assert_eq!(ii.get(&[0, 0]).unwrap(), 1.0);
/// assert_eq!(ii.get(&[1, 0]).unwrap(), 2.0);
/// assert_eq!(ii.get(&[2, 0]).unwrap(), 3.0);
/// assert_eq!(jj.get(&[0, 0]).unwrap(), 4.0);
/// assert_eq!(jj.get(&[0, 1]).unwrap(), 5.0);
/// ```
/// Create a mesh grid from arrays with different indexing modes
///
/// Creates n-dimensional coordinate arrays from 1-dimensional coordinate arrays.
/// Supports both "xy" (default) and "ij" indexing modes to match NumPy's behavior.
///
/// # Parameters
///
/// * `arrays` - A slice of arrays, each representing a coordinate vector
/// * `indexing` - The indexing mode: "xy" (default) or "ij"
///
/// # Returns
///
/// A vector of arrays, each with a shape determined by the input arrays
///
/// # Indexing modes
///
/// - "xy" (default): Cartesian indexing, consistent with plotting coordinates
///   where x is the second axis (columns) and y is the first axis (rows)
/// - "ij": Matrix indexing, where i is the first axis (rows) and j is the second axis (columns)
///
/// For n > 2 dimensions, the remaining dimensions follow normal matrix indexing.
pub fn meshgrid<T: Clone>(arrays: &[&Array<T>], indexing: Option<&str>) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    let indexing_mode = indexing.unwrap_or("xy");

    if indexing_mode != "xy" && indexing_mode != "ij" {
        return Err(crate::error::NumRs2Error::InvalidOperation(format!(
            "Indexing mode '{}' not supported, must be 'xy' or 'ij'",
            indexing_mode
        )));
    }

    let n = arrays.len();
    let mut shape = vec![0; n];

    // Determine the shape of the output arrays
    for (i, arr) in arrays.iter().enumerate() {
        shape[i] = arr.size();
    }

    // Prepare output arrays
    let mut output = Vec::with_capacity(n);

    for i in 0..n {
        // Create a shape with all 1s
        let mut out_shape = vec![1; n];

        // For each output array, we insert the size of the source array
        // in the dimension corresponding to the coordinate
        if indexing_mode == "xy" && n >= 2 && (i == 0 || i == 1) {
            // Special case for xy indexing: swap the first two dimensions
            out_shape[0] = if i == 1 { arrays[i].size() } else { 1 };
            out_shape[1] = if i == 0 { arrays[i].size() } else { 1 };
        } else {
            // For ij indexing or dimensions beyond the first two
            out_shape[i] = arrays[i].size();
        }

        // Reshape the source array
        let reshaped = Array::from_vec(arrays[i].to_vec()).reshape(&out_shape);

        // Determine the target broadcast shape
        let target_shape = if indexing_mode == "xy" && n >= 2 {
            // For xy indexing, the first two dimensions are swapped
            let mut broadcast_shape = Vec::with_capacity(n);
            for j in 0..n {
                if j == 0 && n >= 1 {
                    broadcast_shape.push(shape[1]); // y dimension
                } else if j == 1 && n >= 2 {
                    broadcast_shape.push(shape[0]); // x dimension
                } else {
                    broadcast_shape.push(shape[j]); // other dimensions
                }
            }
            broadcast_shape
        } else {
            // For ij indexing, use the shape directly
            shape.clone()
        };

        // Broadcast to the target shape
        let broadcast_result = reshaped.broadcast_to(&target_shape)?;
        output.push(broadcast_result);
    }

    Ok(output)
}

/// Legacy function for 2D meshgrid with xy indexing
pub fn meshgrid2d<T: Clone>(x: &Array<T>, y: &Array<T>) -> Result<(Array<T>, Array<T>)> {
    let result = meshgrid(&[x, y], Some("xy"))?;
    Ok((result[0].clone(), result[1].clone()))
}

/// Create a dense multi-dimensional meshgrid
///
/// # Parameters
///
/// * `ranges` - A slice of arrays, each representing a coordinate vector
///
/// # Returns
///
/// A vector of arrays, each with a shape determined by the input ranges
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::{mgrid, linspace};
///
/// // Create a 2D meshgrid
/// let grids = mgrid(&[
///     &linspace(0.0, 2.0, 3),
///     &linspace(0.0, 2.0, 3)
/// ]).unwrap();
/// let xx = &grids[0];
/// let yy = &grids[1];
///
/// assert_eq!(xx.shape(), vec![3, 3]);
/// assert_eq!(yy.shape(), vec![3, 3]);
///
/// // Check values
/// assert_eq!(xx.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(xx.get(&[0, 1]).unwrap(), 0.0);
/// assert_eq!(xx.get(&[0, 2]).unwrap(), 0.0);
/// assert_eq!(xx.get(&[1, 0]).unwrap(), 1.0);
/// assert_eq!(xx.get(&[1, 1]).unwrap(), 1.0);
/// assert_eq!(xx.get(&[1, 2]).unwrap(), 1.0);
/// assert_eq!(xx.get(&[2, 0]).unwrap(), 2.0);
///
/// assert_eq!(yy.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(yy.get(&[1, 0]).unwrap(), 0.0);
/// assert_eq!(yy.get(&[2, 0]).unwrap(), 0.0);
/// assert_eq!(yy.get(&[0, 1]).unwrap(), 1.0);
/// assert_eq!(yy.get(&[1, 1]).unwrap(), 1.0);
/// assert_eq!(yy.get(&[2, 1]).unwrap(), 1.0);
/// assert_eq!(yy.get(&[0, 2]).unwrap(), 2.0);
/// ```
pub fn mgrid<T: Clone + NumCast + Zero>(ranges: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if ranges.is_empty() {
        return Ok(vec![]);
    }

    // Calculate the output shape
    let mut shape = Vec::with_capacity(ranges.len());
    for range in ranges {
        shape.push(range.size());
    }

    // Create the output arrays
    let mut output = Vec::with_capacity(ranges.len());
    for _ in 0..ranges.len() {
        output.push(Array::zeros(&shape));
    }

    // Fill the output arrays
    // This is a simplified implementation, a more efficient one would use
    // broadcasting and reshaping operations
    let total_size: usize = shape.iter().product();

    for i in 0..total_size {
        let mut indices = Vec::with_capacity(shape.len());
        let mut temp = i;

        for j in (1..shape.len()).rev() {
            let prod: usize = shape[j..].iter().product();
            indices.insert(0, temp / prod);
            temp %= prod;
        }
        indices.insert(0, temp);

        for dim in 0..ranges.len() {
            // Get the flat index for the current output array
            let mut flat_idx = 0;
            let mut stride = 1;

            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Set the value from the corresponding range
            let range_val = ranges[dim].to_vec()[indices[dim]].clone();
            let output_array = &mut output[dim];
            let mut_data = output_array.array_mut();

            // This is a simplification; in a real implementation we'd use
            // ndarray's mutable indexing
            let flat_data = mut_data.as_slice_mut().unwrap();
            flat_data[flat_idx] = range_val;
        }
    }

    Ok(output)
}

/// Create an open multi-dimensional meshgrid
///
/// Creates a sequence of n-dimensional coordinate arrays where each array has
/// all dimensions of size 1 except for the specific one for that coordinate.
/// This is memory-efficient compared to mgrid, which creates full n-dimensional arrays.
///
/// # Parameters
///
/// * `ranges` - A slice of arrays, each representing a coordinate vector
///
/// # Returns
///
/// A vector of arrays, each with shape having 1's except in the dimension corresponding to the coordinate
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::{ogrid, linspace};
///
/// // Create a 2D open meshgrid
/// let grids = ogrid(&[
///     &linspace(0.0, 2.0, 3),
///     &linspace(0.0, 2.0, 3)
/// ]).unwrap();
/// let xx = &grids[0];
/// let yy = &grids[1];
///
/// assert_eq!(xx.shape(), vec![3, 1]);
/// assert_eq!(yy.shape(), vec![1, 3]);
///
/// // Check values
/// assert_eq!(xx.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(xx.get(&[1, 0]).unwrap(), 1.0);
/// assert_eq!(xx.get(&[2, 0]).unwrap(), 2.0);
///
/// assert_eq!(yy.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(yy.get(&[0, 1]).unwrap(), 1.0);
/// assert_eq!(yy.get(&[0, 2]).unwrap(), 2.0);
///
/// // Unlike mgrid which creates full meshgrids, ogrid creates 1D arrays arranged for broadcasting
/// // This means they can be used efficiently in operations
/// // For example, to create a grid of x^2 + y^2:
/// let x_squared = xx.map(|x| x * x);
/// let y_squared = yy.map(|y| y * y);
/// // Using element-wise addition to create r_squared
/// let x_squared_vec = x_squared.to_vec();
/// let y_squared_vec = y_squared.to_vec();
/// let mut r_squared_vec = Vec::new();
///
/// // Manual broadcasting - for each x value, add all y values
/// for i in 0..3 {
///     for j in 0..3 {
///         r_squared_vec.push(x_squared_vec[i] + y_squared_vec[j]);
///     }
/// }
/// let r_squared = Array::from_vec(r_squared_vec).reshape(&[3, 3]);
///
/// assert_eq!(r_squared.shape(), vec![3, 3]);
/// assert_eq!(r_squared.get(&[0, 0]).unwrap(), 0.0);
/// assert_eq!(r_squared.get(&[1, 1]).unwrap(), 2.0);
/// assert_eq!(r_squared.get(&[2, 2]).unwrap(), 8.0);
/// ```
pub fn ogrid<T: Clone + NumCast + Zero>(ranges: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if ranges.is_empty() {
        return Ok(vec![]);
    }

    let n = ranges.len();
    let mut output = Vec::with_capacity(n);

    for (i, range) in ranges.iter().enumerate() {
        // Create a shape with 1s except at the ith position
        let mut shape = vec![1; n];
        shape[i] = range.size();

        // Reshape the range to this shape
        let reshaped = Array::from_vec(range.to_vec()).reshape(&shape);
        output.push(reshaped);
    }

    Ok(output)
}

/// Create an array with values evenly spaced on a log scale
pub fn geomspace<T: Float + Clone>(start: T, stop: T, num: usize) -> Array<T> {
    if start <= T::zero() || stop <= T::zero() {
        panic!("geomspace requires positive start and stop values");
    }

    let log_start = start.ln();
    let log_stop = stop.ln();

    linspace(log_start, log_stop, num).map(|x| x.exp())
}

// Complex number functions
/// Extract the real part of complex numbers
pub fn real<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.re)
}

/// Extract the imaginary part of complex numbers
pub fn imag<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.im)
}

/// Calculate the complex conjugate of complex numbers
pub fn conj<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<Complex<T>> {
    complex_array.map(|c| c.conj())
}

/// Calculate the absolute value (magnitude) of complex numbers
pub fn complex_abs<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.norm())
}

/// Calculate the phase angle (argument) of complex numbers
pub fn angle<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.arg())
}

/// Unwrap phase angles by changing deltas between values to 2π complements
pub fn unwrap<T: Float + Clone>(phase_array: &Array<T>) -> Array<T> {
    // If array is empty or has a single element, there's nothing to unwrap
    if phase_array.size() <= 1 {
        return phase_array.clone();
    }

    let data = phase_array.to_vec();
    let mut result = Vec::with_capacity(data.len());

    // Start with the first phase value
    result.push(data[0]);

    // The 2π value
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

    // Process the rest of the array
    for i in 1..data.len() {
        let mut delta = data[i] - data[i - 1];

        // Adjust for jumps larger than π
        while delta > T::from(std::f64::consts::PI).unwrap() {
            delta = delta - two_pi;
        }
        while delta < -T::from(std::f64::consts::PI).unwrap() {
            delta = delta + two_pi;
        }

        result.push(result[i - 1] + delta);
    }

    Array::from_vec(result)
}

/// Return coordinate matrices from coordinate vectors
// This function is similar to mgrid but takes owned references
#[allow(dead_code)]
fn mgrid_owned<T: Clone + NumCast + Zero>(ranges: &[Array<T>]) -> Result<Vec<Array<T>>> {
    if ranges.is_empty() {
        return Ok(vec![]);
    }

    // Calculate the output shape
    let mut shape = Vec::with_capacity(ranges.len());
    for range in ranges {
        shape.push(range.size());
    }

    // Create the output arrays
    let mut output = Vec::with_capacity(ranges.len());
    for _ in 0..ranges.len() {
        output.push(Array::zeros(&shape));
    }

    // Fill the output arrays
    // This is a simplified implementation, a more efficient one would use
    // broadcasting and reshaping operations
    let total_size: usize = shape.iter().product();

    for i in 0..total_size {
        let mut indices = Vec::with_capacity(shape.len());
        let mut temp = i;

        for j in (1..shape.len()).rev() {
            let prod: usize = shape[j..].iter().product();
            indices.insert(0, temp / prod);
            temp %= prod;
        }
        indices.insert(0, temp);

        for dim in 0..ranges.len() {
            // Get the flat index for the current output array
            let mut flat_idx = 0;
            let mut stride = 1;

            for j in (0..shape.len()).rev() {
                flat_idx += indices[j] * stride;
                stride *= shape[j];
            }

            // Set the value from the corresponding range
            let range_val = ranges[dim].to_vec()[indices[dim]].clone();
            let output_array = &mut output[dim];
            let mut_data = output_array.array_mut();

            // This is a simplification; in a real implementation we'd use
            // ndarray's mutable indexing
            let flat_data = mut_data.as_slice_mut().unwrap();
            flat_data[flat_idx] = range_val;
        }
    }

    Ok(output)
}

/// Calculate the discrete difference along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `n` - The number of times values are differenced. Default is 1.
/// * `axis` - The axis along which the difference is taken. Default is the last axis.
///
/// # Returns
///
/// The n-th differences. The shape of the output is the same as input except along axis where the dimension is smaller by n.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
/// let d = diff(&a, 1, None).unwrap();
/// assert_eq!(d.to_vec(), vec![1.0, 2.0, 3.0, -7.0]);
/// ```
pub fn diff<T>(array: &Array<T>, n: usize, axis: Option<usize>) -> Result<Array<T>>
where
    T: Clone + Zero + std::ops::Sub<Output = T>,
{
    if n == 0 {
        return Ok(array.clone());
    }

    let axis = axis.unwrap_or(array.ndim().saturating_sub(1));

    if axis >= array.ndim() {
        return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let axis_size = array.shape()[axis];
    if axis_size <= n {
        // Create empty array with appropriate shape
        let mut new_shape = array.shape();
        new_shape[axis] = 0;
        return Ok(Array::zeros(&new_shape));
    }

    let mut result = array.clone();

    for _ in 0..n {
        let axis_size = result.shape()[axis];
        if axis_size <= 1 {
            let mut new_shape = result.shape();
            new_shape[axis] = 0;
            return Ok(Array::zeros(&new_shape));
        }

        // Create new shape with axis dimension reduced by 1
        let mut new_shape = result.shape();
        new_shape[axis] -= 1;

        let mut new_data = Vec::with_capacity(new_shape.iter().product());

        // Calculate strides
        let mut strides = vec![1; result.ndim()];
        for i in (0..result.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * result.shape()[i + 1];
        }

        // Iterate through all indices in the new array
        let total_size: usize = new_shape.iter().product();
        for i in 0..total_size {
            // Convert flat index to multi-dimensional indices
            let mut indices = vec![0; new_shape.len()];
            let mut temp = i;
            for j in 0..new_shape.len() {
                indices[j] = temp / strides[j];
                temp %= strides[j];
            }

            // Get values at current position and next position along axis
            let mut indices_next = indices.clone();
            indices_next[axis] += 1;

            let val1 = result.get(&indices)?;
            let val2 = result.get(&indices_next)?;

            new_data.push(val2 - val1);
        }

        result = Array::from_vec(new_data).reshape(&new_shape);
    }

    Ok(result)
}

/// The differences between consecutive elements of an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `to_end` - Optional values to append at the end of the returned differences
/// * `to_begin` - Optional values to prepend at the beginning of the returned differences
///
/// # Returns
///
/// The differences. Loosely, this is ``a.flatten()[1:] - a.flatten()[:-1]``.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 4, 7, 0]);
/// let d = ediff1d(&a, None, None).unwrap();
/// assert_eq!(d.to_vec(), vec![1, 2, 3, -7]);
/// ```
pub fn ediff1d<T>(
    array: &Array<T>,
    to_end: Option<&Array<T>>,
    to_begin: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    let flat = array.to_vec();

    if flat.len() <= 1 {
        // Create result with just to_begin and to_end
        let mut result = Vec::new();
        if let Some(begin) = to_begin {
            result.extend(begin.to_vec());
        }
        if let Some(end) = to_end {
            result.extend(end.to_vec());
        }
        return Ok(Array::from_vec(result));
    }

    let mut result = Vec::new();

    // Add to_begin values
    if let Some(begin) = to_begin {
        result.extend(begin.to_vec());
    }

    // Calculate differences
    for i in 1..flat.len() {
        result.push(flat[i].clone() - flat[i - 1].clone());
    }

    // Add to_end values
    if let Some(end) = to_end {
        result.extend(end.to_vec());
    }

    Ok(Array::from_vec(result))
}

/// Return the gradient of an N-dimensional array
///
/// # Parameters
///
/// * `f` - An N-dimensional array containing samples of a scalar function
/// * `varargs` - Spacing between f values. Default unitary spacing for all dimensions.
/// * `axis` - Gradient is calculated only along the given axis or axes
/// * `edge_order` - Gradient is calculated using N-th order accurate differences at the boundaries (1 or 2)
///
/// # Returns
///
/// A vector of ndarrays (or a single ndarray if there is only one dimension) corresponding to the derivatives of f with respect to each dimension.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let f = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
/// let grad = gradient(&f, None, None, 1).unwrap();
/// assert_eq!(grad.len(), 1);
/// // First order differences at edges: [1, 1.5, 2.5, 3.5, 4]
/// ```
/// Integrate along the given axis using the composite trapezoidal rule
///
/// # Parameters
///
/// * `y` - Input array to integrate
/// * `x` - Optional array of sample points corresponding to the y values
/// * `dx` - Spacing between sample points when x is None. Default is 1.
/// * `axis` - The axis along which to integrate. Default is the last axis.
///
/// # Returns
///
/// Definite integral as approximated by trapezoidal rule
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let integral = trapz(&y, None, 1.0, None).unwrap();
/// assert_eq!(integral.to_vec()[0], 4.0); // (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4.0
/// ```
pub fn trapz<T>(y: &Array<T>, x: Option<&Array<T>>, dx: T, axis: Option<usize>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let axis = axis.unwrap_or(y.ndim().saturating_sub(1));

    if axis >= y.ndim() {
        return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            y.ndim()
        )));
    }

    let axis_size = y.shape()[axis];
    if axis_size < 2 {
        // Create result array with axis dimension removed
        let mut new_shape = y.shape();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        return Ok(Array::zeros(&new_shape));
    }

    // If x is provided, check compatibility
    if let Some(x_arr) = x {
        if x_arr.ndim() != 1 || x_arr.size() != axis_size {
            return Err(crate::error::NumRs2Error::ShapeMismatch {
                expected: vec![axis_size],
                actual: x_arr.shape(),
            });
        }
    }

    // Create output shape (remove the integration axis)
    let mut out_shape = y.shape();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_size: usize = out_shape.iter().product();
    let mut result_data = vec![T::zero(); out_size];

    // Calculate strides for input array
    let mut in_strides = vec![1; y.ndim()];
    for i in (0..y.ndim() - 1).rev() {
        in_strides[i] = in_strides[i + 1] * y.shape()[i + 1];
    }

    // Calculate strides for output array
    let mut out_strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Iterate through output positions
    for out_idx in 0..out_size {
        // Convert flat output index to multi-dimensional indices
        let mut out_indices = vec![0; out_shape.len()];
        let mut temp = out_idx;
        for i in 0..out_shape.len() {
            out_indices[i] = temp / out_strides[i];
            temp %= out_strides[i];
        }

        // Build input indices by inserting axis dimension
        let mut sum = T::zero();

        for i in 0..axis_size - 1 {
            // Build indices for current and next position
            let mut indices_curr = Vec::with_capacity(y.ndim());
            let mut indices_next = Vec::with_capacity(y.ndim());

            let mut out_idx_ptr = 0;
            for j in 0..y.ndim() {
                if j == axis {
                    indices_curr.push(i);
                    indices_next.push(i + 1);
                } else {
                    indices_curr.push(out_indices[out_idx_ptr]);
                    indices_next.push(out_indices[out_idx_ptr]);
                    out_idx_ptr += 1;
                }
            }

            let y_curr = y.get(&indices_curr)?;
            let y_next = y.get(&indices_next)?;

            let width = if let Some(x_arr) = x {
                let x_vec = x_arr.to_vec();
                x_vec[i + 1] - x_vec[i]
            } else {
                dx
            };

            sum = sum + (y_curr + y_next) * width / T::from(2.0).unwrap();
        }

        result_data[out_idx] = sum;
    }

    Ok(Array::from_vec(result_data).reshape(&out_shape))
}

/// Find the indices of the maximum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the maximum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the maximum values along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0]).reshape(&[2, 3]);
///
/// // Find argmax along axis 1
/// let indices = argmax(&a, Some(1), false).unwrap();
/// assert_eq!(indices.to_vec(), vec![1, 1]); // max indices in each row
///
/// // Find argmax of flattened array
/// let index = argmax(&a, None, false).unwrap();
/// assert_eq!(index.to_vec(), vec![4]); // index 4 has value 5.0
/// ```
pub fn argmax<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find argmax of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find argmax of flattened array
            let data = array.to_vec();
            let mut max_idx = 0;
            let mut max_val = &data[0];

            for (i, val) in data.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            Ok(Array::from_vec(vec![max_idx]))
        }
        Some(ax) => {
            if ax >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[ax] = 1;
            } else {
                out_shape.remove(ax);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![0_usize; out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > ax || (i == ax && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find max along the axis
                let mut max_idx = 0;
                let mut max_val = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if max_val.is_none() || &val > max_val.as_ref().unwrap() {
                        max_val = Some(val);
                        max_idx = j;
                    }
                }

                result_data[out_idx] = max_idx;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Find the indices of the minimum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - The axis along which to find the minimum indices. If None, the array is flattened
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array of indices of the minimum values along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![5.0, 3.0, 2.0, 4.0, 1.0, 6.0]).reshape(&[2, 3]);
///
/// // Find argmin along axis 1
/// let indices = argmin(&a, Some(1), false).unwrap();
/// assert_eq!(indices.to_vec(), vec![2, 1]); // min indices in each row
///
/// // Find argmin of flattened array
/// let index = argmin(&a, None, false).unwrap();
/// assert_eq!(index.to_vec(), vec![4]); // index 4 has value 1.0
/// ```
pub fn argmin<T>(array: &Array<T>, axis: Option<usize>, keepdims: bool) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find argmin of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find argmin of flattened array
            let data = array.to_vec();
            let mut min_idx = 0;
            let mut min_val = &data[0];

            for (i, val) in data.iter().enumerate().skip(1) {
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
            }

            Ok(Array::from_vec(vec![min_idx]))
        }
        Some(ax) => {
            if ax >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[ax];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[ax] = 1;
            } else {
                out_shape.remove(ax);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![0_usize; out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < ax {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > ax || (i == ax && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find min along the axis
                let mut min_idx = 0;
                let mut min_val = None;

                for j in 0..axis_size {
                    indices[ax] = j;
                    let val = array.get(&indices)?;

                    if min_val.is_none() || &val < min_val.as_ref().unwrap() {
                        min_val = Some(val);
                        min_idx = j;
                    }
                }

                result_data[out_idx] = min_idx;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Returns the indices that would sort an array
///
/// # Parameters
///
/// * `array` - Input array to sort
/// * `axis` - Axis along which to sort. If None, the array is flattened before sorting
/// * `kind` - Sorting algorithm (currently only supports default stable sort)
/// * `order` - Not used (for NumPy compatibility)
///
/// # Returns
///
/// Array of indices that sort the array along the specified axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 1.0, 2.0]);
/// let indices = argsort(&a, None, None, None).unwrap();
/// assert_eq!(indices.to_vec(), vec![1, 2, 0]); // sorted order: [1.0, 2.0, 3.0]
/// ```
pub fn argsort<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _kind: Option<&str>,
    _order: Option<&[&str]>,
) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    let axis = if let Some(ax) = axis {
        if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        }
    } else {
        // If axis is None, flatten the array
        let data = array.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return Ok(Array::from_vec(indices));
    };

    if axis >= array.ndim() {
        return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let axis_size = shape[axis];
    let result_shape = shape.clone();

    // The output has the same shape as input
    let total_size: usize = shape.iter().product();
    let mut result_data = vec![0_usize; total_size];

    // Calculate strides
    let mut strides = vec![1; array.ndim()];
    for i in (0..array.ndim() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Number of sorts to perform
    let n_sorts = total_size / axis_size;

    for sort_idx in 0..n_sorts {
        // Collect values along the axis for this position
        let mut values_with_indices: Vec<(T, usize)> = Vec::with_capacity(axis_size);

        // Determine the base indices for this sort
        let mut base_indices = vec![0; array.ndim()];
        let mut temp = sort_idx;
        let _dim_idx = 0;

        for i in 0..array.ndim() {
            if i != axis {
                let size = shape[i];
                base_indices[i] = temp % size;
                temp /= size;
            }
        }

        // Collect values along the axis
        for j in 0..axis_size {
            base_indices[axis] = j;
            let val = array.get(&base_indices)?;
            values_with_indices.push((val, j));
        }

        // Sort by value
        values_with_indices
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Place sorted indices in result
        for (k, (_, idx)) in values_with_indices.into_iter().enumerate() {
            base_indices[axis] = k;
            let flat_idx = base_indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| idx * strides[i])
                .sum::<usize>();
            result_data[flat_idx] = idx;
        }
    }

    Ok(Array::from_vec(result_data).reshape(&result_shape))
}

/// Round array elements to the given number of decimals
///
/// # Parameters
///
/// * `array` - Input array
/// * `decimals` - Number of decimal places to round to (default: 0)
/// * `out` - Optional output array
///
/// # Returns
///
/// Array with rounded values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.234, 2.567, 3.891]);
/// let rounded = around(&a, Some(2), None).unwrap();
/// assert_eq!(rounded.to_vec(), vec![1.23, 2.57, 3.89]);
/// ```
pub fn around<T>(
    array: &Array<T>,
    decimals: Option<i32>,
    out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let decimals = decimals.unwrap_or(0);
    let multiplier = T::from(10.0_f64.powi(decimals)).unwrap();

    let result = array.map(|x| {
        if decimals >= 0 {
            (x * multiplier).round() / multiplier
        } else {
            let divisor = T::from(10.0_f64.powi(-decimals)).unwrap();
            (x / divisor).round() * divisor
        }
    });

    if let Some(out_array) = out {
        if out_array.shape() != result.shape() {
            return Err(crate::error::NumRs2Error::ShapeMismatch {
                expected: result.shape(),
                actual: out_array.shape(),
            });
        }
        *out_array = result;
        Ok(out_array.clone())
    } else {
        Ok(result)
    }
}

/// Return the cumulative sum of array elements along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the cumulative sum is computed. If None, the array is flattened
/// * `dtype` - Data type of the returned array (uses input type if None)
/// * `out` - Alternative output array to place the result
///
/// # Returns
///
/// Array with cumulative sum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let cs = cumsum(&a, None, None).unwrap();
/// assert_eq!(cs.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
/// ```
pub fn cumsum<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T>,
{
    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Flatten and compute cumsum
            let data = array.to_vec();
            let mut result = Vec::with_capacity(data.len());
            let mut sum = T::zero();

            for val in data {
                sum = sum + val;
                result.push(sum);
            }

            Ok(Array::from_vec(result))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let mut result_data = vec![T::zero(); array.size()];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions perpendicular to the axis
            let axis_size = shape[axis];
            let n_iterations = array.size() / axis_size;

            for iter_idx in 0..n_iterations {
                // Determine the base indices for this iteration
                let mut indices = vec![0; array.ndim()];
                let mut temp = iter_idx;

                for i in (0..array.ndim()).rev() {
                    if i != axis {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }
                }

                // Compute cumulative sum along the axis
                let mut sum = T::zero();
                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    sum = sum + val;

                    let flat_idx = indices
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| idx * strides[i])
                        .sum::<usize>();
                    result_data[flat_idx] = sum;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&shape))
        }
    }
}

/// Return the cumulative product of array elements along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the cumulative product is computed. If None, the array is flattened
/// * `dtype` - Data type of the returned array (uses input type if None)
/// * `out` - Alternative output array to place the result
///
/// # Returns
///
/// Array with cumulative product values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let cp = cumprod(&a, None, None).unwrap();
/// assert_eq!(cp.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
/// ```
pub fn cumprod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T>,
{
    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Flatten and compute cumprod
            let data = array.to_vec();
            let mut result = Vec::with_capacity(data.len());
            let mut prod = T::one();

            for val in data {
                prod = prod * val;
                result.push(prod);
            }

            Ok(Array::from_vec(result))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let mut result_data = vec![T::zero(); array.size()];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through all positions perpendicular to the axis
            let axis_size = shape[axis];
            let n_iterations = array.size() / axis_size;

            for iter_idx in 0..n_iterations {
                // Determine the base indices for this iteration
                let mut indices = vec![0; array.ndim()];
                let mut temp = iter_idx;

                for i in (0..array.ndim()).rev() {
                    if i != axis {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }
                }

                // Compute cumulative product along the axis
                let mut prod = T::one();
                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    prod = prod * val;

                    let flat_idx = indices
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| idx * strides[i])
                        .sum::<usize>();
                    result_data[flat_idx] = prod;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&shape))
        }
    }
}

/// Compute the arithmetic mean along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the mean is computed. If None, compute mean of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the mean values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let m = mean(&a, Some(1), false).unwrap();
/// assert_eq!(m.to_vec(), vec![2.0, 5.0]); // mean of each row
/// ```
pub fn mean<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Div<Output = T> + NumCast,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute mean of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Compute mean of flattened array
            let data = array.to_vec();
            let sum = data.iter().fold(T::zero(), |acc, x| acc + *x);
            let count = T::from(data.len()).unwrap();
            let mean_val = sum / count;

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![mean_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![mean_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];
            let axis_size_t = T::from(axis_size).unwrap();

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Compute sum along the axis
                let mut sum = T::zero();
                for j in 0..axis_size {
                    indices[axis] = j;
                    sum = sum + array.get(&indices)?;
                }

                result_data[out_idx] = sum / axis_size_t;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return the product of array elements over a given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the product is computed. If None, compute product of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
/// * `initial` - Starting value for the product
///
/// # Returns
///
/// Array containing the product values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let p = prod(&a, None, false, None).unwrap();
/// assert_eq!(p.to_vec(), vec![24.0]); // 1 * 2 * 3 * 4
/// ```
pub fn prod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    keepdims: bool,
    initial: Option<T>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T>,
{
    if array.is_empty() && initial.is_none() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute product of empty array without initial value".to_string(),
        ));
    }

    let init_val = initial.unwrap_or_else(|| T::one());

    match axis {
        None => {
            // Compute product of flattened array
            let data = array.to_vec();
            let product = data.iter().fold(init_val, |acc, x| acc * *x);

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![product]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![product]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Compute product along the axis
                let mut product = init_val;
                for j in 0..axis_size {
                    indices[axis] = j;
                    product = product * array.get(&indices)?;
                }

                result_data[out_idx] = product;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return a new array with the specified shape
///
/// # Parameters
///
/// * `array` - Input array
/// * `new_shape` - Shape of resized array
///
/// # Returns
///
/// New array with the specified shape. If the new array is larger, it is filled
/// with repeated copies of the input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let resized = resize(&a, &[5]).unwrap();
/// assert_eq!(resized.to_vec(), vec![1, 2, 3, 1, 2]); // repeated to fill
/// ```
pub fn resize<T>(array: &Array<T>, new_shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Zero,
{
    let new_size: usize = new_shape.iter().product();

    if new_size == 0 || array.is_empty() {
        return Ok(Array::zeros(new_shape));
    }

    let old_data = array.to_vec();
    let old_size = old_data.len();

    let mut new_data = Vec::with_capacity(new_size);

    // Fill new array by cycling through old data
    for i in 0..new_size {
        new_data.push(old_data[i % old_size].clone());
    }

    Ok(Array::from_vec(new_data).reshape(new_shape))
}

/// Compute the standard deviation along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the standard deviation is computed. If None, compute over flattened array
/// * `ddof` - Delta degrees of freedom. The divisor is N - ddof (default: 0)
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the standard deviation values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::std;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let s = std(&a, None, 0, false).expect("std failed");
/// assert!((s.to_vec()[0] - 1.414f64).abs() < 0.01); // approximately sqrt(2)
/// ```
pub fn std<T>(
    array: &Array<T>,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast,
{
    // First compute variance, then take square root
    let variance = var(array, axis, ddof, keepdims)?;
    Ok(variance.map(|x| x.sqrt()))
}

/// Compute the variance along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the variance is computed. If None, compute over flattened array
/// * `ddof` - Delta degrees of freedom. The divisor is N - ddof (default: 0)
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the variance values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let v = var(&a, None, 0, false).unwrap();
/// assert_eq!(v.to_vec(), vec![2.0]); // variance is 2.0
/// ```
pub fn var<T>(
    array: &Array<T>,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> Result<Array<T>>
where
    T: Float
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute variance of empty array".to_string(),
        ));
    }

    // First compute the mean
    let mean_arr = mean(array, axis, keepdims)?;

    match axis {
        None => {
            // Compute variance of flattened array
            let data = array.to_vec();
            let mean_val = mean_arr.to_vec()[0];

            let sum_sq = data
                .iter()
                .map(|x| {
                    let diff = *x - mean_val;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            let n = data.len();
            if n <= ddof {
                return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                    "Degrees of freedom {} >= number of elements {}",
                    ddof, n
                )));
            }

            let divisor = T::from(n - ddof).unwrap();
            let var_val = sum_sq / divisor;

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![var_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![var_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            if axis_size <= ddof {
                return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                    "Degrees of freedom {} >= axis size {}",
                    ddof, axis_size
                )));
            }

            let divisor = T::from(axis_size - ddof).unwrap();

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Get mean values as vector
            let mean_vec = mean_arr.to_vec();

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Compute sum of squared differences along the axis
                let mean_val = mean_vec[out_idx];
                let mut sum_sq = T::zero();

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    let diff = val - mean_val;
                    sum_sq = sum_sq + (diff * diff);
                }

                result_data[out_idx] = sum_sq / divisor;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Clip (limit) the values in an array
///
/// # Parameters
///
/// * `array` - Input array
/// * `min` - Minimum value. If None, clipping is not performed on lower interval edge
/// * `max` - Maximum value. If None, clipping is not performed on upper interval edge
///
/// # Returns
///
/// Array with values clipped to [min, max]
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let clipped = clip(&a, Some(2.0), Some(4.0)).unwrap();
/// assert_eq!(clipped.to_vec(), vec![2.0, 2.0, 3.0, 4.0, 4.0]);
/// ```
pub fn clip<T>(array: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: PartialOrd + Clone,
{
    Ok(array.map(|x| {
        let mut val = x;
        if let Some(ref min_val) = min {
            if &val < min_val {
                val = min_val.clone();
            }
        }
        if let Some(ref max_val) = max {
            if &val > max_val {
                val = max_val.clone();
            }
        }
        val
    }))
}

/// Array maximum along a given axis (alias for max)
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find maximum values. If None, the maximum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing maximum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0]).reshape(&[2, 3]);
/// let maxs = amax(&a, Some(1), false).unwrap();
/// assert_eq!(maxs.to_vec(), vec![3.0, 5.0]); // max of each row
/// ```
pub fn amax<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    max(array, axis, keepdims)
}

/// Array minimum along a given axis (alias for min)
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find minimum values. If None, the minimum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing minimum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![5.0, 3.0, 2.0, 4.0, 1.0, 6.0]).reshape(&[2, 3]);
/// let mins = amin(&a, Some(1), false).unwrap();
/// assert_eq!(mins.to_vec(), vec![2.0, 1.0]); // min of each row
/// ```
pub fn amin<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    min(array, axis, keepdims)
}

/// Find the maximum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find maximum values. If None, the maximum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing maximum values
pub fn max<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find max of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find max of flattened array
            let data = array.to_vec();
            let max_val =
                data.iter().skip(1).fold(
                    data[0].clone(),
                    |max, x| {
                        if x > &max {
                            x.clone()
                        } else {
                            max
                        }
                    },
                );

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![max_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![max_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find max along the axis
                let mut max_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if max_val.is_none() || &val > max_val.as_ref().unwrap() {
                        max_val = Some(val);
                    }
                }

                result_data[out_idx] = max_val.unwrap();
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Find the minimum values along an axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to find minimum values. If None, the minimum of the flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing minimum values
pub fn min<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot find min of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Find min of flattened array
            let data = array.to_vec();
            let min_val =
                data.iter().skip(1).fold(
                    data[0].clone(),
                    |min, x| {
                        if x < &min {
                            x.clone()
                        } else {
                            min
                        }
                    },
                );

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![min_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![min_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Find min along the axis
                let mut min_val = None;

                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;

                    if min_val.is_none() || &val < min_val.as_ref().unwrap() {
                        min_val = Some(val);
                    }
                }

                result_data[out_idx] = min_val.unwrap();
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Sum of array elements over a given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to sum. If None, sum over flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing sum values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
/// let sums = sum(&a, Some(1), false).unwrap();
/// assert_eq!(sums.to_vec(), vec![6.0, 15.0]); // sum of each row
/// ```
pub fn sum<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
{
    if array.is_empty() {
        return Ok(if keepdims {
            let shape = if axis.is_none() {
                vec![1; array.ndim()]
            } else {
                let mut shape = array.shape();
                let ax = if let Some(a) = axis {
                    if a < 0 {
                        (array.ndim() as isize + a) as usize
                    } else {
                        a as usize
                    }
                } else {
                    0
                };
                if ax < shape.len() {
                    shape[ax] = 1;
                }
                shape
            };
            Array::zeros(&shape)
        } else {
            Array::zeros(&[1])
        });
    }

    match axis {
        None => {
            // Sum of flattened array
            let data = array.to_vec();
            let sum_val = data.iter().fold(T::zero(), |acc, x| acc + *x);

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![sum_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![sum_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Compute sum along the axis
                let mut sum = T::zero();
                for j in 0..axis_size {
                    indices[axis] = j;
                    sum = sum + array.get(&indices)?;
                }

                result_data[out_idx] = sum;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Sort an array along the given axis
///
/// # Parameters
///
/// * `array` - Array to be sorted
/// * `axis` - Axis along which to sort. If None, the array is flattened before sorting
/// * `kind` - Sorting algorithm (currently only supports default stable sort)
/// * `order` - Not used (for NumPy compatibility)
///
/// # Returns
///
/// Sorted array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]).reshape(&[2, 3]);
/// let sorted = sort(&a, Some(1), None, None).unwrap();
/// assert_eq!(sorted.get(&[0, 0]).unwrap(), 1.0);
/// assert_eq!(sorted.get(&[0, 1]).unwrap(), 3.0);
/// assert_eq!(sorted.get(&[0, 2]).unwrap(), 4.0);
/// ```
pub fn sort<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _kind: Option<&str>,
    _order: Option<&[&str]>,
) -> Result<Array<T>>
where
    T: PartialOrd + Clone + Zero,
{
    if array.is_empty() {
        return Ok(array.clone());
    }

    match axis {
        None => {
            // Sort flattened array
            let mut data = array.to_vec();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Array::from_vec(data))
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];
            let total_size: usize = shape.iter().product();
            let mut result_data = vec![T::zero(); total_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Number of sorts to perform
            let n_sorts = total_size / axis_size;

            for sort_idx in 0..n_sorts {
                // Collect values along the axis for this position
                let mut values: Vec<T> = Vec::with_capacity(axis_size);

                // Determine the base indices for this sort
                let mut base_indices = vec![0; array.ndim()];
                let mut temp = sort_idx;

                for i in 0..array.ndim() {
                    if i != axis {
                        let size = shape[i];
                        base_indices[i] = temp % size;
                        temp /= size;
                    }
                }

                // Collect values along the axis
                for j in 0..axis_size {
                    base_indices[axis] = j;
                    values.push(array.get(&base_indices)?);
                }

                // Sort values
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Place sorted values in result
                for (k, val) in values.into_iter().enumerate() {
                    base_indices[axis] = k;
                    let flat_idx = base_indices
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| idx * strides[i])
                        .sum::<usize>();
                    result_data[flat_idx] = val;
                }
            }

            Ok(Array::from_vec(result_data).reshape(&shape))
        }
    }
}

/// Perform an indirect partition along the given axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `kth` - Element index to partition by. The element at this index will be in its final sorted position
/// * `axis` - Axis along which to sort. If None, the array is flattened
/// * `kind` - Selection algorithm (currently only supports default)
/// * `order` - Not used (for NumPy compatibility)
///
/// # Returns
///
/// Array of indices that partition the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
/// let indices = argpartition(&a, 2, None, None, None).unwrap();
/// // After partitioning: values at indices[0] and indices[1] are <= value at indices[2]
/// // and value at indices[3] >= value at indices[2]
/// ```
pub fn argpartition<T>(
    array: &Array<T>,
    kth: usize,
    axis: Option<isize>,
    _kind: Option<&str>,
    _order: Option<&[&str]>,
) -> Result<Array<usize>>
where
    T: PartialOrd + Clone + Zero,
{
    let axis = if let Some(ax) = axis {
        if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        }
    } else {
        // If axis is None, flatten the array
        let data = array.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();

        if kth >= data.len() {
            return Err(crate::error::NumRs2Error::InvalidOperation(format!(
                "kth ({}) out of bounds for array of size {}",
                kth,
                data.len()
            )));
        }

        // Partition the indices
        indices.select_nth_unstable_by(kth, |&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        return Ok(Array::from_vec(indices));
    };

    if axis >= array.ndim() {
        return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis,
            array.ndim()
        )));
    }

    let shape = array.shape();
    let axis_size = shape[axis];

    if kth >= axis_size {
        return Err(crate::error::NumRs2Error::InvalidOperation(format!(
            "kth ({}) out of bounds for axis {} of size {}",
            kth, axis, axis_size
        )));
    }

    // The output has the same shape as input
    let total_size: usize = shape.iter().product();
    let mut result_data = vec![0_usize; total_size];

    // Calculate strides
    let mut strides = vec![1; array.ndim()];
    for i in (0..array.ndim() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Number of partitions to perform
    let n_partitions = total_size / axis_size;

    for part_idx in 0..n_partitions {
        // Collect values along the axis for this position
        let mut values_with_indices: Vec<(T, usize)> = Vec::with_capacity(axis_size);

        // Determine the base indices for this partition
        let mut base_indices = vec![0; array.ndim()];
        let mut temp = part_idx;

        for i in 0..array.ndim() {
            if i != axis {
                let size = shape[i];
                base_indices[i] = temp % size;
                temp /= size;
            }
        }

        // Collect values along the axis
        for j in 0..axis_size {
            base_indices[axis] = j;
            let val = array.get(&base_indices)?;
            values_with_indices.push((val, j));
        }

        // Create indices array
        let mut indices: Vec<usize> = (0..axis_size).collect();

        // Partition by kth element
        indices.select_nth_unstable_by(kth, |&a, &b| {
            values_with_indices[a]
                .0
                .partial_cmp(&values_with_indices[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Place partitioned indices in result
        for (k, &idx) in indices.iter().enumerate() {
            base_indices[axis] = k;
            let flat_idx = base_indices
                .iter()
                .enumerate()
                .map(|(i, &idx)| idx * strides[i])
                .sum::<usize>();
            result_data[flat_idx] = values_with_indices[idx].1;
        }
    }

    Ok(Array::from_vec(result_data).reshape(&shape))
}

/// Round array elements to the nearest integer
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with rounded values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::round;
///
/// let a = Array::from_vec(vec![1.5, 2.3, 3.7, 4.5]);
/// let rounded = round(&a).expect("round failed");
/// assert_eq!(rounded.to_vec(), vec![2.0, 2.0, 4.0, 5.0]);
/// ```
pub fn round<T>(array: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    Ok(array.map(|x| x.round()))
}

/// Alias for cumsum - Return the cumulative sum of array elements
pub fn cumulative_sum<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T>,
{
    cumsum(array, axis, _out)
}

/// Alias for cumprod - Return the cumulative product of array elements
pub fn cumulative_prod<T>(
    array: &Array<T>,
    axis: Option<isize>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T>,
{
    cumprod(array, axis, _out)
}

/// Compute sum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute sum (None for flattened array)
///
/// # Returns
///
/// Sum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let sum = nansum(&a, None).unwrap();
/// assert_eq!(sum.to_vec(), vec![7.0]);
/// ```
pub fn nansum<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Sum along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::zeros(&new_shape);

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut sum = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Sum along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    sum = sum + value;
                }
            }

            result.set(&res_indices, sum)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let sum = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::zero(), |acc, x| acc + *x);
        Ok(Array::from_vec(vec![sum]))
    }
}

/// Compute mean ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute mean (None for flattened array)
///
/// # Returns
///
/// Mean of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanmean;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let mean = nanmean(&a, None).expect("nanmean failed");
/// assert!((mean.to_vec()[0] - 2.333333333333333).abs() < 1e-10);
/// ```
pub fn nanmean<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Div<Output = T> + Zero,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        let sums = nansum(array, Some(ax as isize))?;

        // Count non-NaN values along axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut counts = Array::zeros(&new_shape);

        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut count = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Count along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    count = count + T::one();
                }
            }

            counts.set(&res_indices, count)?;
        }

        // Divide sums by counts
        Ok(sums.zip_with(
            &counts,
            |s, c| {
                if c == T::zero() {
                    T::nan()
                } else {
                    s / c
                }
            },
        )?)
    } else {
        let mut sum = T::zero();
        let mut count = 0;

        let array_vec = array.to_vec();
        for value in array_vec.iter() {
            if !value.is_nan() {
                sum = sum + *value;
                count += 1;
            }
        }

        if count == 0 {
            Ok(Array::from_vec(vec![T::nan()]))
        } else {
            Ok(Array::from_vec(vec![sum / T::from(count).unwrap()]))
        }
    }
}

/// Compute standard deviation ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute std (None for flattened array)
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Standard deviation of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let std = nanstd(&a, None, 0).unwrap();
/// ```
pub fn nanstd<T>(array: &Array<T>, axis: Option<isize>, ddof: usize) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero,
{
    let variance = nanvar(array, axis, ddof)?;
    Ok(variance.map(|x| x.sqrt()))
}

/// Compute variance ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute variance (None for flattened array)
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Variance of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let var = nanvar(&a, None, 0).unwrap();
/// ```
pub fn nanvar<T>(array: &Array<T>, axis: Option<isize>, ddof: usize) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Zero,
{
    let mean = nanmean(array, axis)?;

    if let Some(axis_val) = axis {
        // Compute variance along axis
        let ax = if axis_val < 0 {
            (array.ndim() as isize + axis_val) as usize
        } else {
            axis_val as usize
        };

        // Expand mean for broadcasting
        let mut mean_shape = vec![1; array.ndim()];
        mean_shape[ax] = array.shape()[ax];
        let mean_expanded = mean.reshape(&mean_shape);

        // Compute squared differences
        let diff = array.zip_with(&mean_expanded, |a, m| a - m)?;
        let diff_sq = diff.map(|x| if x.is_nan() { T::zero() } else { x * x });

        // Count non-NaN values along axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut counts = Array::zeros(&new_shape);

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut count = T::zero();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    count = count + T::one();
                }
            }

            counts.set(&res_indices, count)?;
        }

        // Sum squared differences along axis
        let sum_sq = nansum(&diff_sq, Some(ax as isize))?;

        // Compute variance
        Ok(sum_sq.zip_with(&counts, |s, c| {
            let adjusted_count = c - T::from(ddof).unwrap();
            if adjusted_count <= T::zero() {
                T::nan()
            } else {
                s / adjusted_count
            }
        })?)
    } else {
        // Compute variance for flattened array
        let mean_val = mean.to_vec()[0];
        let mut sum_sq = T::zero();
        let mut count = 0;

        let array_vec = array.to_vec();
        for value in array_vec.iter() {
            if !value.is_nan() {
                let diff = *value - mean_val;
                sum_sq = sum_sq + diff * diff;
                count += 1;
            }
        }

        if count <= ddof {
            Ok(Array::from_vec(vec![T::nan()]))
        } else {
            Ok(Array::from_vec(vec![
                sum_sq / T::from(count - ddof).unwrap(),
            ]))
        }
    }
}

/// Compute minimum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute min (None for flattened array)
///
/// # Returns
///
/// Minimum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let min = nanmin(&a, None).unwrap();
/// assert_eq!(min.to_vec(), vec![1.0]);
/// ```
pub fn nanmin<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find min along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::nan());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut min_val = T::nan();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find min along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() && (min_val.is_nan() || value < min_val) {
                    min_val = value;
                }
            }

            result.set(&res_indices, min_val)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let min = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(T::nan());
        Ok(Array::from_vec(vec![min]))
    }
}

/// Compute maximum ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute max (None for flattened array)
///
/// # Returns
///
/// Maximum of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
/// let max = nanmax(&a, None).unwrap();
/// assert_eq!(max.to_vec(), vec![4.0]);
/// ```
pub fn nanmax<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find max along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::nan());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut max_val = T::nan();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find max along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() && (max_val.is_nan() || value > max_val) {
                    max_val = value;
                }
            }

            result.set(&res_indices, max_val)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let max = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(T::nan());
        Ok(Array::from_vec(vec![max]))
    }
}

/// Return cumulative sum of elements along axis, ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute cumulative sum. If None, flattened array
///
/// # Returns
///
/// Cumulative sum array with same shape as input
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nancumsum;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
/// let cumsum = nancumsum(&a, None).expect("nancumsum failed");
/// assert_eq!(cumsum.to_vec(), vec![1.0, 1.0, 4.0, 8.0]);
/// ```
pub fn nancumsum<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Add<Output = T> + Zero,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        if ax >= array.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax,
                array.ndim()
            )));
        }

        // Compute cumulative sum along specified axis
        let shape = array.shape();
        let mut result = Array::zeros(&shape);
        let axis_len = shape[ax];

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let total_elems: usize = shape.iter().product();
        let axis_stride = strides[ax];
        let group_size = axis_stride * axis_len;

        // Process each group independently
        for group_start in (0..total_elems).step_by(group_size) {
            for offset in 0..axis_stride {
                let mut cumsum = T::zero();

                for i in 0..axis_len {
                    let idx = group_start + i * axis_stride + offset;
                    let flat_idx = idx;

                    // Convert flat index to multi-dimensional
                    let mut indices = vec![0; shape.len()];
                    let mut temp = flat_idx;
                    for j in 0..shape.len() {
                        indices[j] = temp / strides[j];
                        temp %= strides[j];
                    }

                    let value = array.get(&indices)?;
                    if !value.is_nan() {
                        cumsum = cumsum + value;
                    }
                    result.set(&indices, cumsum)?;
                }
            }
        }

        Ok(result)
    } else {
        // Flatten array and compute cumulative sum
        let flat = array.to_vec();
        let mut result = Vec::with_capacity(flat.len());
        let mut cumsum = T::zero();

        for value in flat {
            if !value.is_nan() {
                cumsum = cumsum + value;
            }
            result.push(cumsum);
        }

        Ok(Array::from_vec(result).reshape(&array.shape()))
    }
}

/// Return product of elements along axis, ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to compute product. If None, compute over flattened array
///
/// # Returns
///
/// Product of non-NaN elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanprod;
///
/// let a = Array::from_vec(vec![2.0, f64::NAN, 3.0, 4.0]);
/// let prod = nanprod(&a, None).expect("nanprod failed");
/// assert_eq!(prod.to_vec(), vec![24.0]);
/// ```
pub fn nanprod<T>(array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Float + Clone + Mul<Output = T> + One,
{
    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        // Find product along axis ignoring NaN
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let mut result = Array::full(&new_shape, T::one());

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut prod = T::one();
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Find product along the axis
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    prod = prod * value;
                }
            }

            result.set(&res_indices, prod)?;
        }

        Ok(result)
    } else {
        let array_vec = array.to_vec();
        let prod = array_vec
            .iter()
            .filter(|x| !x.is_nan())
            .fold(T::one(), |acc, &x| acc * x);
        Ok(Array::from_vec(vec![prod]))
    }
}

/// Compute percentile of array ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `q` - Percentile to compute (0-100)
/// * `axis` - Axis along which to compute percentile. If None, compute over flattened array
/// * `method` - Method to use for percentile computation (same as percentile)
///
/// # Returns
///
/// Percentile value(s)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanpercentile;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
/// let p50 = nanpercentile(&a, 50.0, None, None).expect("nanpercentile failed");
/// assert_eq!(p50.to_vec(), vec![3.0]);
/// ```
pub fn nanpercentile<T>(
    array: &Array<T>,
    q: T,
    axis: Option<isize>,
    method: Option<&str>,
) -> Result<Array<T>>
where
    T: Float + Clone + NumCast + std::fmt::Display,
{
    // Convert percentile to array and call nanquantile
    let q_arr = Array::from_vec(vec![q / T::from(100.0).unwrap()]);
    nanquantile(array, &q_arr, axis, method)
}

/// Compute quantile of array ignoring NaN values
///
/// # Parameters
///
/// * `array` - Input array
/// * `q` - Quantile(s) to compute (0-1)
/// * `axis` - Axis along which to compute quantile. If None, compute over flattened array
/// * `method` - Method to use for quantile computation
///
/// # Returns
///
/// Quantile value(s)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nanquantile;
///
/// let a = Array::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
/// let q = Array::from_vec(vec![0.5]);
/// let median = nanquantile(&a, &q, None, None).expect("nanquantile failed");
/// assert_eq!(median.to_vec(), vec![3.0]);
/// ```
pub fn nanquantile<T>(
    array: &Array<T>,
    q: &Array<T>,
    axis: Option<isize>,
    method: Option<&str>,
) -> Result<Array<T>>
where
    T: Float + Clone + NumCast + std::fmt::Display,
{
    let method_str = method.unwrap_or("linear");

    if let Some(ax) = axis {
        let ax = if ax < 0 {
            (array.ndim() as isize + ax) as usize
        } else {
            ax as usize
        };

        if ax >= array.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "axis {} is out of bounds for array of dimension {}",
                ax,
                array.ndim()
            )));
        }

        // Compute quantile along specified axis
        let shape = array.shape();
        let mut new_shape = shape.clone();
        new_shape.remove(ax);

        if new_shape.is_empty() {
            new_shape = vec![1];
        }

        let axis_len = shape[ax];
        let q_vec = q.to_vec();
        let n_quantiles = q_vec.len();

        // Result shape is new_shape + [n_quantiles]
        let mut result_shape = new_shape.clone();
        result_shape.push(n_quantiles);
        let mut result = Array::zeros(&result_shape);

        let result_size: usize = new_shape.iter().product();

        for res_idx in 0..result_size {
            let mut res_indices = vec![0; new_shape.len()];
            let mut temp = res_idx;

            // Convert flat index to multi-dimensional
            for i in (0..new_shape.len()).rev() {
                res_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            // Collect non-NaN values along the axis
            let mut values = Vec::new();
            for ax_idx in 0..axis_len {
                let mut full_indices = vec![0; shape.len()];
                let mut res_idx_ptr = 0;

                for i in 0..shape.len() {
                    if i == ax {
                        full_indices[i] = ax_idx;
                    } else {
                        full_indices[i] = res_indices[res_idx_ptr];
                        res_idx_ptr += 1;
                    }
                }

                let value = array.get(&full_indices)?;
                if !value.is_nan() {
                    values.push(value);
                }
            }

            // Sort values and compute quantiles
            if values.is_empty() {
                // All values were NaN
                for (q_idx, _) in q_vec.iter().enumerate() {
                    let mut result_indices = res_indices.clone();
                    result_indices.push(q_idx);
                    result.set(&result_indices, T::nan())?;
                }
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = values.len();

                for (q_idx, &q_val) in q_vec.iter().enumerate() {
                    if q_val < T::zero() || q_val > T::one() {
                        return Err(NumRs2Error::InvalidOperation(format!(
                            "Quantile value {} out of bounds [0, 1]",
                            q_val
                        )));
                    }

                    let idx_float = q_val * T::from(n - 1).unwrap();
                    let idx_lower = idx_float.floor();
                    let idx_upper = idx_float.ceil();
                    let idx_lower_usize = idx_lower.to_usize().unwrap();
                    let idx_upper_usize = idx_upper.to_usize().unwrap();

                    let quantile = match method_str {
                        "linear" => {
                            if idx_lower == idx_upper {
                                values[idx_lower_usize]
                            } else {
                                let fraction = idx_float - idx_lower;
                                values[idx_lower_usize] + fraction * (values[idx_upper_usize] - values[idx_lower_usize])
                            }
                        },
                        "lower" => values[idx_lower_usize],
                        "higher" => values[idx_upper_usize],
                        "nearest" => {
                            if idx_float - idx_lower < idx_upper - idx_float {
                                values[idx_lower_usize]
                            } else {
                                values[idx_upper_usize]
                            }
                        },
                        "midpoint" => {
                            if idx_lower == idx_upper {
                                values[idx_lower_usize]
                            } else {
                                (values[idx_lower_usize] + values[idx_upper_usize]) / T::from(2.0).unwrap()
                            }
                        },
                        _ => return Err(NumRs2Error::InvalidOperation(
                            format!("Invalid method '{}'. Must be one of 'linear', 'lower', 'higher', 'nearest', 'midpoint'", method_str)
                        ))
                    };

                    let mut result_indices = res_indices.clone();
                    result_indices.push(q_idx);
                    result.set(&result_indices, quantile)?;
                }
            }
        }

        Ok(result)
    } else {
        // Flatten array and compute quantiles
        let array_vec = array.to_vec();
        let mut values: Vec<T> = array_vec.into_iter().filter(|x| !x.is_nan()).collect();

        if values.is_empty() {
            // All values were NaN
            let q_vec = q.to_vec();
            let result = vec![T::nan(); q_vec.len()];
            return Ok(Array::from_vec(result));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = values.len();
        let q_vec = q.to_vec();
        let mut result = Vec::with_capacity(q_vec.len());

        for &q_val in &q_vec {
            if q_val < T::zero() || q_val > T::one() {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Quantile value {} out of bounds [0, 1]",
                    q_val
                )));
            }

            let idx_float = q_val * T::from(n - 1).unwrap();
            let idx_lower = idx_float.floor();
            let idx_upper = idx_float.ceil();
            let idx_lower_usize = idx_lower.to_usize().unwrap();
            let idx_upper_usize = idx_upper.to_usize().unwrap();

            let quantile = match method_str {
                "linear" => {
                    if idx_lower == idx_upper {
                        values[idx_lower_usize]
                    } else {
                        let fraction = idx_float - idx_lower;
                        values[idx_lower_usize] + fraction * (values[idx_upper_usize] - values[idx_lower_usize])
                    }
                },
                "lower" => values[idx_lower_usize],
                "higher" => values[idx_upper_usize],
                "nearest" => {
                    if idx_float - idx_lower < idx_upper - idx_float {
                        values[idx_lower_usize]
                    } else {
                        values[idx_upper_usize]
                    }
                },
                "midpoint" => {
                    if idx_lower == idx_upper {
                        values[idx_lower_usize]
                    } else {
                        (values[idx_lower_usize] + values[idx_upper_usize]) / T::from(2.0).unwrap()
                    }
                },
                _ => return Err(NumRs2Error::InvalidOperation(
                    format!("Invalid method '{}'. Must be one of 'linear', 'lower', 'higher', 'nearest', 'midpoint'", method_str)
                ))
            };

            result.push(quantile);
        }

        Ok(Array::from_vec(result))
    }
}

/// Count number of occurrences of each value in array of non-negative integers
///
/// # Parameters
///
/// * `array` - Input array of non-negative integers
/// * `minlength` - Minimum number of bins (output size will be at least minlength)
///
/// # Returns
///
/// Array where the value at index i is the count of occurrences of i in the input
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![0, 1, 1, 3, 2, 1, 7]);
/// let counts = bincount(&a, None).unwrap();
/// assert_eq!(counts.to_vec(), vec![1, 3, 1, 1, 0, 0, 0, 1]);
/// ```
pub fn bincount(array: &Array<usize>, minlength: Option<usize>) -> Result<Array<usize>> {
    let array_vec = array.to_vec();
    let max_val = array_vec.iter().max().cloned().unwrap_or(0);
    let size = if let Some(min) = minlength {
        min.max(max_val + 1)
    } else {
        max_val + 1
    };

    let mut counts = vec![0; size];
    for &val in array_vec.iter() {
        counts[val] += 1;
    }

    Ok(Array::from_vec(counts))
}

/// Return indices of bins to which each value belongs
///
/// # Parameters
///
/// * `array` - Input array
/// * `bins` - Array of bin edges (must be monotonically increasing)
/// * `right` - If true, intervals include right edge; otherwise left edge
///
/// # Returns
///
/// Array of indices indicating which bin each value belongs to
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]);
/// let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
/// let indices = digitize(&x, &bins, true).unwrap();
/// assert_eq!(indices.to_vec(), vec![1, 4, 3, 2]);
/// ```
pub fn digitize<T>(array: &Array<T>, bins: &Array<T>, right: bool) -> Result<Array<usize>>
where
    T: Float + Clone + PartialOrd,
{
    // Verify bins are monotonic
    let bins_vec = bins.to_vec();
    for i in 1..bins_vec.len() {
        if bins_vec[i] <= bins_vec[i - 1] {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "bins must be monotonically increasing".to_string(),
            ));
        }
    }

    let mut result = Vec::with_capacity(array.len());

    let array_vec = array.to_vec();
    for value in array_vec.iter() {
        let mut idx = 0;
        if right {
            // Find rightmost bin where value <= bin_edge
            for (i, &bin) in bins_vec.iter().enumerate() {
                if value <= &bin {
                    idx = i;
                    break;
                }
                idx = i + 1;
            }
        } else {
            // Find rightmost bin where value < bin_edge
            for (i, &bin) in bins_vec.iter().enumerate() {
                if value < &bin {
                    idx = i;
                    break;
                }
                idx = i + 1;
            }
        }
        result.push(idx);
    }

    Ok(Array::from_vec(result).reshape(&array.shape()))
}

/// Find indices where elements should be inserted to maintain order
///
/// # Parameters
///
/// * `sorted_array` - Array that is already sorted
/// * `values` - Values to insert
/// * `side` - If 'left', gives leftmost position; if 'right', gives rightmost
///
/// # Returns
///
/// Array of insertion indices
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let v = Array::from_vec(vec![0.5, 2.0, 3.5, 6.0]);
/// let indices = searchsorted(&a, &v, "left").unwrap();
/// assert_eq!(indices.to_vec(), vec![0, 1, 3, 5]);
/// ```
pub fn searchsorted<T>(
    sorted_array: &Array<T>,
    values: &Array<T>,
    side: &str,
) -> Result<Array<usize>>
where
    T: Float + Clone + PartialOrd,
{
    let arr_vec = sorted_array.to_vec();
    let mut result = Vec::with_capacity(values.len());

    let values_vec = values.to_vec();
    for value in values_vec.iter() {
        let idx = match side {
            "left" => arr_vec
                .iter()
                .position(|x| x >= value)
                .unwrap_or(arr_vec.len()),
            "right" => arr_vec
                .iter()
                .position(|x| x > value)
                .unwrap_or(arr_vec.len()),
            _ => {
                return Err(crate::error::NumRs2Error::InvalidOperation(
                    "side must be 'left' or 'right'".to_string(),
                ))
            }
        };
        result.push(idx);
    }

    Ok(Array::from_vec(result).reshape(&values.shape()))
}

/// Partially sort array so that kth element is in its final sorted position
///
/// # Parameters
///
/// * `array` - Input array
/// * `kth` - Index of element to partition by
/// * `axis` - Axis along which to sort
/// * `kind` - Selection algorithm (currently ignored)
/// * `order` - Not used, for compatibility
///
/// # Returns
///
/// Partially sorted array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let mut a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0]);
/// let result = partition(&a, 2, None, None, None).unwrap();
/// // Elements at indices 0,1 are <= element at index 2 <= elements at index 3
/// ```
pub fn partition<T>(
    array: &Array<T>,
    kth: usize,
    axis: Option<isize>,
    _kind: Option<&str>,
    _order: Option<&[&str]>,
) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    let mut result = array.clone();

    if let Some(axis_val) = axis {
        // Partition along axis
        let ax = if axis_val < 0 {
            (array.ndim() as isize + axis_val) as usize
        } else {
            axis_val as usize
        };
        let shape = array.shape();
        let axis_len = shape[ax];

        if kth >= axis_len {
            return Err(crate::error::NumRs2Error::IndexOutOfBounds(format!(
                "kth ({}) out of bounds for axis {} of size {}",
                kth, ax, axis_len
            )));
        }

        // Calculate strides for iteration
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let total_size: usize = shape.iter().product();
        let n_slices = total_size / axis_len;

        for slice_idx in 0..n_slices {
            // Get indices for this slice along the axis
            let _indices: Vec<usize> = Vec::with_capacity(axis_len);
            let mut base_indices = vec![0; shape.len()];

            // Convert slice index to multi-dimensional indices
            let mut temp = slice_idx;
            for i in (0..shape.len()).rev() {
                if i != ax {
                    let stride = if i < ax {
                        strides[i] / strides[ax]
                    } else {
                        strides[i]
                    };
                    base_indices[i] = temp / stride;
                    temp %= stride;
                }
            }

            // Collect values along the axis
            let mut values = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                base_indices[ax] = i;
                let _flat_idx = base_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| idx * strides[i])
                    .sum::<usize>();
                let value = array.get(&base_indices)?;
                values.push(value);
            }

            // Partition the values
            values.select_nth_unstable_by(kth, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Write back partitioned values
            for i in 0..axis_len {
                base_indices[ax] = i;
                result.set(&base_indices, values[i])?;
            }
        }

        Ok(result)
    } else {
        // Partition flattened array
        let mut data_vec = result.to_vec();
        if kth >= data_vec.len() {
            return Err(crate::error::NumRs2Error::IndexOutOfBounds(format!(
                "kth ({}) out of bounds for array of size {}",
                kth,
                data_vec.len()
            )));
        }

        data_vec.select_nth_unstable_by(kth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Array::from_vec(data_vec).reshape(&array.shape()))
    }
}

/// One-dimensional linear interpolation
///
/// # Parameters
///
/// * `x` - The x-coordinates at which to evaluate the interpolated values
/// * `xp` - The x-coordinates of the data points, must be increasing
/// * `fp` - The y-coordinates of the data points
/// * `left` - Value to return for x < xp[0], default is fp[0]
/// * `right` - Value to return for x > xp[-1], default is fp[-1]
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let xp = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let fp = Array::from_vec(vec![3.0, 2.0, 0.0]);
/// let x = Array::from_vec(vec![0.0, 1.5, 2.72, 3.14]);
/// let y = interp(&x, &xp, &fp, None, None).unwrap();
/// ```
pub fn interp<T>(
    x: &Array<T>,
    xp: &Array<T>,
    fp: &Array<T>,
    left: Option<T>,
    right: Option<T>,
) -> Result<Array<T>>
where
    T: Float + Clone,
{
    let xp_vec = xp.to_vec();
    let fp_vec = fp.to_vec();

    if xp_vec.len() != fp_vec.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![xp_vec.len()],
            actual: vec![fp_vec.len()],
        });
    }

    if xp_vec.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "xp must have at least 1 point".to_string(),
        ));
    }

    // Verify xp is sorted
    for i in 1..xp_vec.len() {
        if xp_vec[i] <= xp_vec[i - 1] {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "xp must be monotonically increasing".to_string(),
            ));
        }
    }

    let left_val = left.unwrap_or_else(|| fp_vec[0]);
    let right_val = right.unwrap_or_else(|| fp_vec[fp_vec.len() - 1]);

    let mut result = Vec::with_capacity(x.len());

    let x_vec = x.to_vec();
    for xi in x_vec.iter() {
        if xi < &xp_vec[0] {
            result.push(left_val);
        } else if xi > &xp_vec[xp_vec.len() - 1] {
            result.push(right_val);
        } else {
            // Binary search for the interval containing xi
            let mut lo = 0;
            let mut hi = xp_vec.len() - 1;

            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if xi < &xp_vec[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            // Linear interpolation
            let x0 = xp_vec[lo];
            let x1 = xp_vec[hi];
            let y0 = fp_vec[lo];
            let y1 = fp_vec[hi];

            let t = (*xi - x0) / (x1 - x0);
            let yi = y0 + t * (y1 - y0);
            result.push(yi);
        }
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

/// Compute the median along the specified axis
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which the median is computed. If None, compute median of flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Array containing the median values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let m = median(&a, None, false).unwrap();
/// assert_eq!(m.to_vec(), vec![3.0]); // median is 3.0
/// ```
pub fn median<T>(array: &Array<T>, axis: Option<isize>, keepdims: bool) -> Result<Array<T>>
where
    T: Float + Clone + PartialOrd,
{
    if array.is_empty() {
        return Err(crate::error::NumRs2Error::InvalidOperation(
            "Cannot compute median of empty array".to_string(),
        ));
    }

    match axis {
        None => {
            // Compute median of flattened array
            let mut data = array.to_vec();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = data.len();
            let median_val = if n % 2 == 0 {
                // Even number of elements - average of two middle values
                (data[n / 2 - 1] + data[n / 2]) / T::from(2.0).unwrap()
            } else {
                // Odd number of elements - middle value
                data[n / 2]
            };

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![median_val]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![median_val]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![T::zero(); out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Collect values along the axis
                let mut values = Vec::with_capacity(axis_size);
                for j in 0..axis_size {
                    indices[axis] = j;
                    values.push(array.get(&indices)?);
                }

                // Sort and find median
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_val = if axis_size % 2 == 0 {
                    (values[axis_size / 2 - 1] + values[axis_size / 2]) / T::from(2.0).unwrap()
                } else {
                    values[axis_size / 2]
                };

                result_data[out_idx] = median_val;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Test element-wise for NaN
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates NaN
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0, f64::INFINITY]);
/// let nan_mask = isnan(&a);
/// assert_eq!(nan_mask.to_vec(), vec![false, true, false, false]);
/// ```
pub fn isnan<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_nan())
}

/// Test element-wise for positive or negative infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let inf_mask = isinf(&a);
/// assert_eq!(inf_mask.to_vec(), vec![false, true, true, false]);
/// ```
pub fn isinf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite())
}

/// Test element-wise for finiteness (not infinity and not NaN)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of boolean values where True indicates finite values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NAN, 2.0]);
/// let finite_mask = isfinite(&a);
/// assert_eq!(finite_mask.to_vec(), vec![true, false, false, true]);
/// ```
pub fn isfinite<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_finite())
}

/// Test element-wise for positive infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates positive infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isposinf;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let posinf_mask = isposinf(&a);
/// assert_eq!(posinf_mask.to_vec(), vec![false, true, false, false]);
/// ```
pub fn isposinf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite() && x.is_sign_positive())
}

/// Test element-wise for negative infinity
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates negative infinity
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isneginf;
///
/// let a = Array::from_vec(vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
/// let neginf_mask = isneginf(&a);
/// assert_eq!(neginf_mask.to_vec(), vec![false, false, true, false]);
/// ```
pub fn isneginf<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_infinite() && x.is_sign_negative())
}

/// Test element-wise for normal numbers (not zero, subnormal, infinite or NaN)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates normal numbers
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isnormal;
///
/// let a = Array::from_vec(vec![1.0, 0.0, f64::INFINITY, f64::NAN]);
/// let normal_mask = isnormal(&a);
/// assert_eq!(normal_mask.to_vec(), vec![true, false, false, false]);
/// ```
pub fn isnormal<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|x| x.is_normal())
}

/// Test element-wise for real numbers (opposite of complex)
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates real numbers (always True for real arrays)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::isreal;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let real_mask = isreal(&a);
/// assert_eq!(real_mask.to_vec(), vec![true, true, true]);
/// ```
pub fn isreal<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|_x| true) // Real arrays are always real
}

/// Test element-wise for complex numbers
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Boolean array where True indicates complex numbers (always False for real arrays)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::iscomplex;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let complex_mask = iscomplex(&a);
/// assert_eq!(complex_mask.to_vec(), vec![false, false, false]);
/// ```
pub fn iscomplex<T>(array: &Array<T>) -> Array<bool>
where
    T: Float,
{
    array.map(|_x| false) // Real arrays are never complex
}

/// Replace NaN with zero and infinity with large finite numbers
///
/// # Parameters
///
/// * `array` - Input array
/// * `nan` - Value to replace NaN (default: 0.0)
/// * `posinf` - Value to replace positive infinity (default: very large positive number)
/// * `neginf` - Value to replace negative infinity (default: very large negative number)
///
/// # Returns
///
/// Array with replaced values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
/// let clean = nan_to_num(&a, None, None, None).unwrap();
/// // NaN -> 0.0, inf -> f64::MAX, -inf -> f64::MIN
/// ```
pub fn nan_to_num<T>(
    array: &Array<T>,
    nan: Option<T>,
    posinf: Option<T>,
    neginf: Option<T>,
) -> Result<Array<T>>
where
    T: Float + std::fmt::Debug,
{
    let nan_val = nan.unwrap_or_else(T::zero);
    let posinf_val = posinf.unwrap_or_else(T::max_value);
    let neginf_val = neginf.unwrap_or_else(T::min_value);

    Ok(array.map(|x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                posinf_val
            } else {
                neginf_val
            }
        } else {
            x
        }
    }))
}

/// Count the number of non-zero values in the array
///
/// # Parameters
///
/// * `array` - Input array
/// * `axis` - Axis along which to count non-zeros. If None, count over flattened array
/// * `keepdims` - If true, the axes which are reduced are left in the result as dimensions with size one
///
/// # Returns
///
/// Number of non-zero values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::count_nonzero;
///
/// let a = Array::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
/// let count = count_nonzero(&a, None, false).expect("count_nonzero failed");
/// assert_eq!(count.to_vec(), vec![3]); // 3 non-zero elements
/// ```
pub fn count_nonzero<T>(
    array: &Array<T>,
    axis: Option<isize>,
    keepdims: bool,
) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    match axis {
        None => {
            // Count non-zeros in flattened array
            let data = array.to_vec();
            let count = data.iter().filter(|&x| x != &T::zero()).count();

            if keepdims {
                let shape = vec![1; array.ndim()];
                Ok(Array::from_vec(vec![count]).reshape(&shape))
            } else {
                Ok(Array::from_vec(vec![count]))
            }
        }
        Some(ax) => {
            let axis = if ax < 0 {
                (array.ndim() as isize + ax) as usize
            } else {
                ax as usize
            };

            if axis >= array.ndim() {
                return Err(crate::error::NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis,
                    array.ndim()
                )));
            }

            let shape = array.shape();
            let axis_size = shape[axis];

            // Create output shape
            let mut out_shape = shape.clone();
            if keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let out_size: usize = out_shape.iter().product();
            let mut result_data = vec![0usize; out_size];

            // Calculate strides
            let mut strides = vec![1; array.ndim()];
            for i in (0..array.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Iterate through output positions
            for out_idx in 0..out_size {
                // Convert flat index to multi-dimensional indices
                let mut indices = vec![0; array.ndim()];
                let mut temp = out_idx;

                for i in 0..array.ndim() {
                    if i < axis {
                        let dim_size = shape[i];
                        indices[i] = temp % dim_size;
                        temp /= dim_size;
                    } else if i > axis || (i == axis && keepdims) {
                        let dim_idx = if keepdims { i } else { i - 1 };
                        if dim_idx < out_shape.len() {
                            let dim_size = out_shape[dim_idx];
                            indices[i] = temp % dim_size;
                            temp /= dim_size;
                        }
                    }
                }

                // Count non-zeros along the axis
                let mut count = 0;
                for j in 0..axis_size {
                    indices[axis] = j;
                    let val = array.get(&indices)?;
                    if val != T::zero() {
                        count += 1;
                    }
                }

                result_data[out_idx] = count;
            }

            Ok(Array::from_vec(result_data).reshape(&out_shape))
        }
    }
}

/// Return the indices of the elements that are non-zero
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Tuple of arrays, one for each dimension, containing indices of non-zero elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::nonzero;
///
/// let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]).reshape(&[2, 3]);
/// let indices = nonzero(&a).expect("nonzero failed");
/// assert_eq!(indices.len(), 2); // 2D array has 2 index arrays
/// assert_eq!(indices[0].to_vec(), vec![0, 1, 1]);
/// assert_eq!(indices[1].to_vec(), vec![1, 0, 2]);
/// ```
pub fn nonzero<T>(array: &Array<T>) -> Result<Vec<Array<usize>>>
where
    T: Clone + Zero + PartialEq,
{
    let shape = array.shape();
    let ndim = array.ndim();

    // Find all non-zero positions
    let mut nonzero_positions = Vec::new();
    let data = array.to_vec();

    for (idx, value) in data.iter().enumerate() {
        if value != &T::zero() {
            // Convert flat index to multi-dimensional indices
            let mut indices = vec![0; ndim];
            let mut temp = idx;

            for i in (0..ndim).rev() {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            nonzero_positions.push(indices);
        }
    }

    // Transpose the positions to get separate arrays for each dimension
    let mut result = Vec::with_capacity(ndim);
    for dim in 0..ndim {
        let dim_indices: Vec<usize> = nonzero_positions.iter().map(|pos| pos[dim]).collect();
        result.push(Array::from_vec(dim_indices));
    }

    Ok(result)
}

/// Return indices that are non-zero in the flattened version of the input array
///
/// # Parameters
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of indices in the flattened array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![0, 1, 0, 3, 0, 5]);
/// let indices = flatnonzero(&a).unwrap();
/// assert_eq!(indices.to_vec(), vec![1, 3, 5]);
/// ```
pub fn flatnonzero<T>(array: &Array<T>) -> Result<Array<usize>>
where
    T: Clone + Zero + PartialEq,
{
    let data = array.to_vec();
    let nonzero_indices: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, value)| *value != &T::zero())
        .map(|(idx, _)| idx)
        .collect();

    Ok(Array::from_vec(nonzero_indices))
}

/// Compute the greatest common divisor of two arrays element-wise
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing the greatest common divisor of corresponding elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![12, 15, 21]);
/// let b = Array::from_vec(vec![8, 10, 14]);
/// let result = gcd(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![4, 5, 7]);
/// ```
pub fn gcd<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Zero + PartialEq + std::ops::Rem<Output = T> + Copy,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| gcd_scalar(a, b))
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Helper function to compute GCD of two scalar values
fn gcd_scalar<T>(mut a: T, mut b: T) -> T
where
    T: Clone + Zero + PartialEq + std::ops::Rem<Output = T> + Copy,
{
    while b != T::zero() {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compute the least common multiple of two arrays element-wise
///
/// # Parameters
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing the least common multiple of corresponding elements
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![12, 15, 21]);
/// let b = Array::from_vec(vec![8, 10, 14]);
/// let result = lcm(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![24, 30, 42]);
/// ```
pub fn lcm<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Clone
        + Zero
        + PartialEq
        + std::ops::Rem<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + Copy,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| {
            if a == T::zero() || b == T::zero() {
                T::zero()
            } else {
                let gcd_val = gcd_scalar(a, b);
                a * b / gcd_val
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Copy the sign of one array to another element-wise
///
/// # Parameters
///
/// * `x1` - Array whose values to use
/// * `x2` - Array whose signs to use
///
/// # Returns
///
/// Array with magnitudes from x1 and signs from x2
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, -2.0, 3.0]);
/// let b = Array::from_vec(vec![-1.0, 1.0, -1.0]);
/// let result = copysign(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![-1.0, 2.0, -3.0]);
/// ```
pub fn copysign<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&val, &sign_val)| {
            if sign_val < T::zero() {
                -val.abs()
            } else {
                val.abs()
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Return the next floating-point value after x1 towards x2, element-wise
///
/// # Parameters
///
/// * `x1` - Starting values
/// * `x2` - Values to move towards
///
/// # Returns
///
/// Array with next representable floating-point values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1.0, 2.0]);
/// let b = Array::from_vec(vec![2.0, 1.0]);
/// let result = nextafter(&a, &b).unwrap();
/// // result[0] will be slightly larger than 1.0
/// // result[1] will be slightly smaller than 2.0
/// assert!(result.get(&[0]).unwrap() > 1.0);
/// assert!(result.get(&[1]).unwrap() < 2.0);
/// ```
pub fn nextafter<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    // Check shapes are compatible
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    // Use epsilon for approximation since we can't access raw bits in generic Float trait
    let epsilon = T::epsilon();

    let result_data: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&from, &to)| {
            if from == to {
                from
            } else if from.is_nan() || to.is_nan() {
                T::nan()
            } else if from < to {
                // Move towards positive infinity
                if from == T::infinity() {
                    from
                } else {
                    from + from.abs() * epsilon
                }
            } else {
                // Move towards negative infinity
                if from == T::neg_infinity() {
                    from
                } else {
                    from - from.abs() * epsilon
                }
            }
        })
        .collect();

    Ok(Array::from_vec(result_data).reshape(&x1.shape()))
}

/// Normalized sinc function
///
/// Return the normalized sinc function sin(π*x)/(π*x). The sinc function is 1
/// for x=0, and sin(π*x)/(π*x) for all other points.
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Array of sinc values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let y = sinc(&x);
/// // sinc(0) = 1, sinc(1) = 0, sinc(2) = 0, sinc(3) = 0
/// ```
pub fn sinc<T>(x: &Array<T>) -> Array<T>
where
    T: Float + Clone,
{
    x.map(|val| {
        if val == T::zero() {
            T::one()
        } else {
            let pi = T::from(std::f64::consts::PI).unwrap();
            let pix = pi * val;
            pix.sin() / pix
        }
    })
}

/// Heaviside step function
///
/// The Heaviside step function is defined as:
/// - 0 if x < 0
/// - h0 if x == 0
/// - 1 if x > 0
///
/// # Parameters
///
/// * `x` - Input array
/// * `h0` - The value of the function at x=0 (default: 0.5)
///
/// # Returns
///
/// Array of Heaviside function values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let y = heaviside(&x, Some(0.5));
/// assert_eq!(y.to_vec(), vec![0.0, 0.5, 1.0]);
/// ```
pub fn heaviside<T>(x: &Array<T>, h0: Option<T>) -> Array<T>
where
    T: Float + Clone,
{
    let h0_val = h0.unwrap_or_else(|| T::from(0.5).unwrap());

    x.map(|val| {
        if val < T::zero() {
            T::zero()
        } else if val == T::zero() {
            h0_val
        } else {
            T::one()
        }
    })
}

/// If input is complex with all imaginary parts close to zero, return real parts
///
/// "Close to zero" is defined as tol * (machine epsilon of the type).
///
/// # Parameters
///
/// * `a` - Input array (must be complex)
/// * `tol` - Tolerance in multiples of machine epsilon (default: 100)
///
/// # Returns
///
/// If the imaginary part of all elements is smaller than the tolerance,
/// return only the real part. Otherwise, return the original array.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use num_complex::Complex;
///
/// let a = Array::from_vec(vec![
///     Complex::new(1.0, 1e-15),
///     Complex::new(2.0, 1e-16),
///     Complex::new(3.0, 1e-14)
/// ]);
/// // If all imaginary parts are small enough, returns real array
/// let result = real_if_close(&a, Some(1000.0));
/// ```
pub fn real_if_close<T>(a: &Array<Complex<T>>, tol: Option<T>) -> Array<T>
where
    T: Float + Clone,
{
    let tolerance = tol.unwrap_or_else(|| T::from(100.0).unwrap());
    let epsilon = T::epsilon();
    let threshold = tolerance * epsilon;

    let data = a.to_vec();

    // Check if all imaginary parts are close to zero
    let all_close = data.iter().all(|c| c.im.abs() <= threshold);

    if all_close {
        // Return only real parts
        let real_data: Vec<T> = data.iter().map(|c| c.re).collect();
        Array::from_vec(real_data).reshape(&a.shape())
    } else {
        // Convert back to complex array (keeping as real for simplicity in this context)
        // In a real implementation, this would return Result<Either<Array<T>, Array<Complex<T>>>>
        let real_data: Vec<T> = data.iter().map(|c| c.re).collect();
        Array::from_vec(real_data).reshape(&a.shape())
    }
}

/// Load exponent: returns x * 2^exp element-wise
///
/// # Parameters
///
/// * `x` - Input array
/// * `exp` - Array of integer exponents
///
/// # Returns
///
/// Array with the same shape as the input where each element is x[i] * 2^exp[i]
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let exp = Array::from_vec(vec![1, 2, 3]);
/// let result = ldexp(&x, &exp);
/// // result = [2.0, 8.0, 24.0]
/// ```
pub fn ldexp<T, I>(x: &Array<T>, exp: &Array<I>) -> Result<Array<T>>
where
    T: Float + Clone,
    I: Clone + NumCast,
{
    if x.shape() != exp.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, exp has shape {:?}",
            x.shape(),
            exp.shape()
        )));
    }

    let x_vec = x.to_vec();
    let exp_vec = exp.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, exp_val) in x_vec.iter().zip(exp_vec.iter()) {
        let exp_f64 = I::to_f64(exp_val).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert exponent to f64".to_string())
        })?;
        let two = T::from(2.0).unwrap();
        result.push(*x_val * two.powf(T::from(exp_f64).unwrap()));
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

/// Extract mantissa and exponent from array elements
///
/// Decomposes floating-point values into mantissa and exponent such that
/// x = mantissa * 2^exponent, where 0.5 <= |mantissa| < 1.0
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Tuple of (mantissa_array, exponent_array)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![4.0, 0.5, -2.0]);
/// let (mantissa, exponent) = frexp(&x);
/// // mantissa = [0.5, 0.5, -0.5]
/// // exponent = [3, 0, 2]
/// ```
pub fn frexp<T>(x: &Array<T>) -> (Array<T>, Array<i32>)
where
    T: Float + Clone,
{
    let x_vec = x.to_vec();
    let mut mantissa_vec = Vec::with_capacity(x_vec.len());
    let mut exponent_vec = Vec::with_capacity(x_vec.len());

    for val in x_vec.iter() {
        if val.is_zero() {
            mantissa_vec.push(T::zero());
            exponent_vec.push(0);
        } else if val.is_infinite() || val.is_nan() {
            mantissa_vec.push(*val);
            exponent_vec.push(0);
        } else {
            // Get the binary exponent
            let log2_val = val.abs().log2();
            let exp = log2_val.floor();
            let exp_i32 = T::to_i32(&exp).unwrap() + 1;

            // Calculate mantissa
            let two = T::from(2.0).unwrap();
            let mantissa = *val / two.powf(T::from(exp_i32).unwrap());

            mantissa_vec.push(mantissa);
            exponent_vec.push(exp_i32);
        }
    }

    (
        Array::from_vec(mantissa_vec).reshape(&x.shape()),
        Array::from_vec(exponent_vec).reshape(&x.shape()),
    )
}

/// Return fractional and integral parts of array values
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Tuple of (fractional_part, integral_part)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![1.5, -2.3, 3.7]);
/// let (frac, integ) = modf(&x);
/// // frac = [0.5, -0.3, 0.7]
/// // integ = [1.0, -2.0, 3.0]
/// ```
pub fn modf<T>(x: &Array<T>) -> (Array<T>, Array<T>)
where
    T: Float + Clone,
{
    let x_vec = x.to_vec();
    let mut frac_vec = Vec::with_capacity(x_vec.len());
    let mut int_vec = Vec::with_capacity(x_vec.len());

    for val in x_vec.iter() {
        let int_part = val.trunc();
        let frac_part = *val - int_part;

        frac_vec.push(frac_part);
        int_vec.push(int_part);
    }

    (
        Array::from_vec(frac_vec).reshape(&x.shape()),
        Array::from_vec(int_vec).reshape(&x.shape()),
    )
}

/// Element-wise quotient and remainder
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Tuple of (quotient, remainder) arrays
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::divmod;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let (quot, rem) = divmod(&x, &y)?;
/// // quot = [2.0, -2.0, 2.0]
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn divmod<T>(x: &Array<T>, y: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut quot_vec = Vec::with_capacity(x_vec.len());
    let mut rem_vec = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        let quotient = (*x_val / *y_val).floor();
        let remainder = *x_val - quotient * *y_val;

        quot_vec.push(quotient);
        rem_vec.push(remainder);
    }

    Ok((
        Array::from_vec(quot_vec).reshape(&x.shape()),
        Array::from_vec(rem_vec).reshape(&x.shape()),
    ))
}

/// Return element-wise remainder of division
///
/// This differs from modulo operation for negative values.
/// The result has the same sign as the dividend.
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Array with remainder values
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::remainder;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let rem = remainder(&x, &y)?;
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn remainder<T>(x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        // IEEE remainder: r = x - n*y where n is the integer nearest x/y
        let n = (*x_val / *y_val).round();
        let rem = *x_val - n * *y_val;
        result.push(rem);
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

/// Return the element-wise remainder of division (C-style)
///
/// This is the C library fmod function. The result has the same sign as the dividend.
///
/// # Parameters
///
/// * `x` - Dividend array
/// * `y` - Divisor array
///
/// # Returns
///
/// Array with remainder values
///
/// # Examples
///
/// ```
/// # use numrs2::prelude::*;
/// # use numrs2::math::fmod;
/// # use numrs2::Result;
/// # fn main() -> Result<()> {
/// let x = Array::from_vec(vec![7.0, -7.0, 8.0]);
/// let y = Array::from_vec(vec![3.0, 3.0, 3.0]);
/// let rem = fmod(&x, &y)?;
/// // rem = [1.0, -1.0, 2.0]
/// # Ok(())
/// # }
/// ```
pub fn fmod<T>(x: &Array<T>, y: &Array<T>) -> Result<Array<T>>
where
    T: Float + Clone,
{
    if x.shape() != y.shape() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Shape mismatch: x has shape {:?}, y has shape {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut result = Vec::with_capacity(x_vec.len());

    for (x_val, y_val) in x_vec.iter().zip(y_vec.iter()) {
        // fmod: r = x - n*y where n = trunc(x/y)
        let n = (*x_val / *y_val).trunc();
        let rem = *x_val - n * *y_val;
        result.push(rem);
    }

    Ok(Array::from_vec(result).reshape(&x.shape()))
}

// Window functions for signal processing

/// Return the Hanning window
///
/// The Hanning window is a taper formed by using a weighted cosine.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window as a 1-D array of size M
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let window = hanning(10);
/// // window[0] and window[9] are close to 0
/// // window[5] is close to 1
/// ```
pub fn hanning(m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(vec![]);
    }
    if m == 1 {
        return Array::from_vec(vec![1.0]);
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let val = 0.5 - 0.5 * ((2.0 * std::f64::consts::PI * i as f64) / (m - 1) as f64).cos();
        window.push(val);
    }

    Array::from_vec(window)
}

/// Return the Hamming window
///
/// The Hamming window is a taper formed by using a weighted cosine.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window as a 1-D array of size M
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let window = hamming(10);
/// // window[0] and window[9] are small but not zero
/// // window[4] and window[5] are close to 1
/// ```
pub fn hamming(m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(vec![]);
    }
    if m == 1 {
        return Array::from_vec(vec![1.0]);
    }

    let mut window = Vec::with_capacity(m);
    for i in 0..m {
        let val = 0.54 - 0.46 * ((2.0 * std::f64::consts::PI * i as f64) / (m - 1) as f64).cos();
        window.push(val);
    }

    Array::from_vec(window)
}

/// Return the Blackman window
///
/// The Blackman window is a taper formed by using a weighted cosine.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window as a 1-D array of size M
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let window = blackman(10);
/// // window[0] and window[9] are very close to 0
/// // window[5] is close to 1
/// ```
pub fn blackman(m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(vec![]);
    }
    if m == 1 {
        return Array::from_vec(vec![1.0]);
    }

    let mut window = Vec::with_capacity(m);
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;

    for i in 0..m {
        let arg = 2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64;
        let val = a0 - a1 * arg.cos() + a2 * (2.0 * arg).cos();
        window.push(val);
    }

    Array::from_vec(window)
}

/// Return the Bartlett window (triangular window with zero endpoints)
///
/// The Bartlett window is very similar to a triangular window, except
/// that the end points are at zero.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
///
/// # Returns
///
/// The window as a 1-D array of size M
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let window = bartlett(10);
/// // window[0] and window[9] are 0
/// // window[4] and window[5] are close to 1
/// ```
pub fn bartlett(m: usize) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(vec![]);
    }
    if m == 1 {
        return Array::from_vec(vec![1.0]);
    }

    let mut window = Vec::with_capacity(m);
    let m_minus_1 = (m - 1) as f64;

    for i in 0..m {
        let val = if i as f64 <= m_minus_1 / 2.0 {
            2.0 * i as f64 / m_minus_1
        } else {
            2.0 - 2.0 * i as f64 / m_minus_1
        };
        window.push(val);
    }

    Array::from_vec(window)
}

/// Return the Kaiser window
///
/// The Kaiser window is a taper formed by using a Bessel function.
///
/// # Parameters
///
/// * `m` - Number of points in the output window
/// * `beta` - Shape parameter for window. As beta increases, the window
///   gets narrower (default = 8.6, which gives similar sidelobe
///   levels as a Blackman window)
///
/// # Returns
///
/// The window as a 1-D array of size M
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let window = kaiser(10, 8.6);
/// // window is a Kaiser window with beta=8.6
/// ```
pub fn kaiser(m: usize, beta: f64) -> Array<f64> {
    if m == 0 {
        return Array::from_vec(vec![]);
    }
    if m == 1 {
        return Array::from_vec(vec![1.0]);
    }

    let mut window = Vec::with_capacity(m);
    let m_minus_1 = (m - 1) as f64;
    let i0_beta = modified_bessel_i0(beta);

    for i in 0..m {
        let x = 2.0 * i as f64 / m_minus_1 - 1.0;
        let arg = beta * (1.0 - x * x).sqrt();
        let val = modified_bessel_i0(arg) / i0_beta;
        window.push(val);
    }

    Array::from_vec(window)
}

/// Modified Bessel function of the first kind of order 0
///
/// This is a helper function for the Kaiser window.
fn modified_bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_squared_over_4 = (x * x) / 4.0;

    // Series expansion: I_0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
    for k in 1..50 {
        term *= x_squared_over_4 / (k as f64 * k as f64);
        sum += term;

        // Convergence check
        if term < 1e-15 * sum {
            break;
        }
    }

    sum
}

/// Modified Bessel function of the first kind of order 0
///
/// Computes I₀(x) for each element in the input array.
/// This function is commonly used in signal processing and physics.
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Array of I₀(x) values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let result = i0(&x);
/// // Returns [1.0, 1.2660658777520084, 2.2795853023360672, 4.880792585865024]
/// ```
pub fn i0<T>(x: &Array<T>) -> Array<T>
where
    T: Float + Clone,
{
    x.map(|val| {
        let x_f64 = val.to_f64().unwrap();
        let result = modified_bessel_i0(x_f64.abs());
        T::from(result).unwrap()
    })
}

/// Calculate the gradient of N-dimensional array
///
/// The gradient is computed using second-order accurate central differences
/// in the interior points and either first or second-order accurate one-sided
/// differences at the boundaries.
///
/// # Arguments
/// * `f` - An N-dimensional array
/// * `varargs` - Spacing between f values. Can be:
///   - None: Default spacing of 1
///   - Single value: uniform spacing for all dimensions
///   - Array of values: spacing for each dimension
/// * `axis` - Calculate the gradient for selected dimensions. None means all axes.
/// * `edge_order` - Either 1 or 2. Gradient is calculated using N-th order
///   accurate differences at the boundaries. Default is 1.
///
/// # Returns
/// * A list of N arrays (or a single array if axis is specified), each with
///   the same shape as f, giving the derivative of f with respect to each dimension.
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::gradient;
///
/// // 1D gradient
/// let x = Array::from_vec(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
/// let grad = gradient(&x, None, None, 1).unwrap();
/// // Returns [1.0, 1.5, 2.5, 3.5, 4.0]
///
/// // 2D gradient with uniform spacing
/// let f = Array::from_vec(vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0]).reshape(&[2, 3]);
/// let grads = gradient(&f, None, None, 1).unwrap();
/// // grads[0] is gradient along axis 0, grads[1] is gradient along axis 1
/// ```
pub fn gradient<T>(
    f: &Array<T>,
    varargs: Option<GradientSpacing<T>>,
    axis: Option<Vec<usize>>,
    edge_order: usize,
) -> Result<Vec<Array<T>>>
where
    T: Float + Clone,
{
    let ndim = f.ndim();
    let shape = f.shape();

    // Validate edge_order
    if edge_order != 1 && edge_order != 2 {
        return Err(NumRs2Error::ValueError(
            "edge_order must be 1 or 2".to_string(),
        ));
    }

    // Determine axes to compute gradient for
    let axes = match axis {
        Some(a) => {
            // Validate axes
            for &ax in &a {
                if ax >= ndim {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "axis {} is out of bounds for array of dimension {}",
                        ax, ndim
                    )));
                }
            }
            a
        }
        None => (0..ndim).collect(),
    };

    // Parse spacing
    let spacings = match varargs {
        None => vec![T::one(); ndim],
        Some(GradientSpacing::Uniform(h)) => vec![h; ndim],
        Some(GradientSpacing::PerAxis(spacings)) => {
            if spacings.len() != ndim {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "spacing array length {} doesn't match array dimensions {}",
                    spacings.len(),
                    ndim
                )));
            }
            spacings
        }
    };

    let mut results = Vec::new();

    // Compute gradient for each axis
    for &ax in &axes {
        let mut grad = Array::zeros(&shape);
        let h = spacings[ax];
        let n = shape[ax];

        if n == 1 {
            // Gradient of constant is zero
            results.push(grad);
            continue;
        }

        // Helper to get/set values along an axis
        let mut indices = vec![0; ndim];

        // Iterate over all positions perpendicular to the axis
        let total_perp: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax)
            .map(|(_, &s)| s)
            .product();

        for perp_idx in 0..total_perp {
            // Convert linear index to multi-dimensional indices for perpendicular dimensions
            let mut temp = perp_idx;
            let mut _dim_idx = 0;
            for i in 0..ndim {
                if i != ax {
                    let stride: usize = shape
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j > i && *j != ax)
                        .map(|(_, &s)| s)
                        .product();
                    indices[i] = temp / stride;
                    temp %= stride;
                    _dim_idx += 1;
                }
            }

            // Compute gradient along the axis
            for i in 0..n {
                indices[ax] = i;

                let derivative = if i == 0 {
                    // Forward difference at the start
                    if edge_order == 1 || n < 3 {
                        indices[ax] = 1;
                        let f1 = f.get(&indices)?;
                        indices[ax] = 0;
                        let f0 = f.get(&indices)?;
                        (f1 - f0) / h
                    } else {
                        // Second-order forward difference
                        indices[ax] = 0;
                        let f0 = f.get(&indices)?;
                        indices[ax] = 1;
                        let f1 = f.get(&indices)?;
                        indices[ax] = 2;
                        let f2 = f.get(&indices)?;
                        (-f2 * T::from(0.5).unwrap() + f1 * T::from(2.0).unwrap()
                            - f0 * T::from(1.5).unwrap())
                            / h
                    }
                } else if i == n - 1 {
                    // Backward difference at the end
                    if edge_order == 1 || n < 3 {
                        indices[ax] = n - 1;
                        let fn1 = f.get(&indices)?;
                        indices[ax] = n - 2;
                        let fn2 = f.get(&indices)?;
                        (fn1 - fn2) / h
                    } else {
                        // Second-order backward difference
                        indices[ax] = n - 1;
                        let fn1 = f.get(&indices)?;
                        indices[ax] = n - 2;
                        let fn2 = f.get(&indices)?;
                        indices[ax] = n - 3;
                        let fn3 = f.get(&indices)?;
                        (fn3 * T::from(0.5).unwrap() - fn2 * T::from(2.0).unwrap()
                            + fn1 * T::from(1.5).unwrap())
                            / h
                    }
                } else {
                    // Central difference in the interior
                    indices[ax] = i + 1;
                    let fplus = f.get(&indices)?;
                    indices[ax] = i - 1;
                    let fminus = f.get(&indices)?;
                    (fplus - fminus) / (h * T::from(2.0).unwrap())
                };

                indices[ax] = i;
                grad.set(&indices, derivative)?;
            }
        }

        results.push(grad);
    }

    Ok(results)
}

/// Spacing specification for gradient calculation
pub enum GradientSpacing<T> {
    /// Uniform spacing for all dimensions
    Uniform(T),
    /// Per-axis spacing
    PerAxis(Vec<T>),
}

/// Test element-wise for signbit (whether the sign bit is set)
///
/// This function returns true where signbit is set (negative, including -0.0),
/// false otherwise. This is equivalent to NumPy's `np.signbit`.
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array of booleans indicating where signbit is set
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::signbit;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0, -0.0]);
/// let result = signbit(&a);
/// assert_eq!(result.to_vec(), vec![true, false, false, true]);
/// ```
pub fn signbit<T: Float + Clone>(array: &Array<T>) -> Array<bool> {
    array.map(|x| x.is_sign_negative())
}

/// Return the reciprocal of the argument, element-wise
///
/// Calculates `1/x` for each element in the array.
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with reciprocal values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::reciprocal;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 0.5]);
/// let result = reciprocal(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 0.5, 0.25, 2.0]);
/// ```
pub fn reciprocal<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| T::one() / x)
}

/// Return the numerical positive of each element (a no-op for real numbers)
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Copy of the input array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::positive;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let result = positive(&a);
/// assert_eq!(result.to_vec(), vec![-1.0, 0.0, 1.0]);
/// ```
pub fn positive<T: Clone>(array: &Array<T>) -> Array<T> {
    array.clone()
}

/// Return the numerical negative of each element
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with negated values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::negative;
///
/// let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
/// let result = negative(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 0.0, -1.0]);
/// ```
pub fn negative<T: Clone + std::ops::Neg<Output = T>>(array: &Array<T>) -> Array<T> {
    array.map(|x| -x)
}

/// Round elements to the nearest integer
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with values rounded to nearest integer
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::rint;
///
/// let a = Array::from_vec(vec![1.1, 1.5, 1.9, 2.5]);
/// let result = rint(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 2.0, 2.0, 3.0]);
/// ```
pub fn rint<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| x.round())
}

/// Round towards zero (truncate the fractional part)
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with values rounded towards zero
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fix;
///
/// let a = Array::from_vec(vec![1.1, 1.9, -1.1, -1.9]);
/// let result = fix(&a);
/// assert_eq!(result.to_vec(), vec![1.0, 1.0, -1.0, -1.0]);
/// ```
pub fn fix<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| x.trunc())
}

/// Element-wise maximum of array elements, ignoring NaN
///
/// # Arguments
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing element-wise maximum
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fmax;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let b = Array::from_vec(vec![2.0, 2.0, f64::NAN]);
/// let result = fmax(&a, &b).unwrap();
/// assert_eq!(result.to_vec()[0], 2.0);
/// assert_eq!(result.to_vec()[1], 2.0);
/// assert_eq!(result.to_vec()[2], 3.0);
/// ```
pub fn fmax<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.max(b)
            }
        })
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Element-wise minimum of array elements, ignoring NaN
///
/// # Arguments
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing element-wise minimum
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::fmin;
///
/// let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let b = Array::from_vec(vec![2.0, 2.0, f64::NAN]);
/// let result = fmin(&a, &b).unwrap();
/// assert_eq!(result.to_vec()[0], 1.0);
/// assert_eq!(result.to_vec()[1], 2.0);
/// assert_eq!(result.to_vec()[2], 3.0);
/// ```
pub fn fmin<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.min(b)
            }
        })
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Return the distance between x and the nearest adjacent number
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array with spacing values
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::spacing;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 4.0]);
/// let result = spacing(&a);
/// // Returns machine epsilon scaled by magnitude
/// ```
pub fn spacing<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| {
        if x.is_infinite() || x.is_nan() {
            T::nan()
        } else {
            // For IEEE 754 doubles: spacing = 2^(exponent - 52)
            // This is a simplified implementation
            let abs_x = x.abs();
            if abs_x == T::zero() {
                T::from(2.2250738585072014e-308).unwrap() // smallest positive f64
            } else {
                let exponent = abs_x.log2().floor();
                T::from(2.0).unwrap().powf(exponent) * T::epsilon()
            }
        }
    })
}

/// True division of the inputs, element-wise
///
/// # Arguments
///
/// * `x1` - Dividend array
/// * `x2` - Divisor array
///
/// # Returns
///
/// Array containing element-wise true division
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::true_divide;
///
/// let a = Array::from_vec(vec![3.0, 4.0, 5.0]);
/// let b = Array::from_vec(vec![2.0, 2.0, 2.0]);
/// let result = true_divide(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![1.5, 2.0, 2.5]);
/// ```
pub fn true_divide<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| a / b)
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Return the floor division of the inputs, element-wise
///
/// # Arguments
///
/// * `x1` - Dividend array
/// * `x2` - Divisor array
///
/// # Returns
///
/// Array containing element-wise floor division
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::floor_divide;
///
/// let a = Array::from_vec(vec![7.0, 9.0, 11.0]);
/// let b = Array::from_vec(vec![3.0, 4.0, 5.0]);
/// let result = floor_divide(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![2.0, 2.0, 2.0]);
/// ```
pub fn floor_divide<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x1.shape(),
            actual: x2.shape(),
        });
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();

    let result: Vec<T> = x1_data
        .into_iter()
        .zip(x2_data)
        .map(|(a, b)| (a / b).floor())
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Calculate 2**x for all x in the input array
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array containing 2**x for each element
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::exp2;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = exp2(&a);
/// assert_eq!(result.to_vec(), vec![2.0, 4.0, 8.0]);
/// ```
pub fn exp2<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| T::from(2.0).unwrap().powf(x))
}

/// Convert angles from degrees to radians
///
/// # Arguments
///
/// * `array` - Input array of angles in degrees
///
/// # Returns
///
/// Array containing angles in radians
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::deg2rad;
///
/// let degrees = Array::from_vec(vec![0.0, 90.0, 180.0, 270.0, 360.0]);
/// let radians = deg2rad(&degrees);
/// let expected = vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI,
///                     3.0 * std::f64::consts::PI / 2.0, 2.0 * std::f64::consts::PI];
/// assert!((radians.to_vec()[1] - expected[1]).abs() < 1e-10);
/// ```
pub fn deg2rad<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    let pi = T::from(std::f64::consts::PI).unwrap();
    let factor = pi / T::from(180.0).unwrap();
    array.map(|x| x * factor)
}

/// Convert angles from radians to degrees
///
/// # Arguments
///
/// * `array` - Input array of angles in radians
///
/// # Returns
///
/// Array containing angles in degrees
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::rad2deg;
///
/// let radians = Array::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
/// let degrees = rad2deg(&radians);
/// let expected = vec![0.0, 90.0, 180.0];
/// assert!((degrees.to_vec()[1] - expected[1]).abs() < 1e-10);
/// ```
pub fn rad2deg<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    let factor = T::from(180.0).unwrap() / T::from(std::f64::consts::PI).unwrap();
    array.map(|x| x * factor)
}

/// Calculate the Euclidean norm, sqrt(x1**2 + x2**2), element-wise
///
/// # Arguments
///
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
///
/// Array containing the element-wise Euclidean norm
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::hypot;
///
/// let x1 = Array::from_vec(vec![3.0, 0.0, 4.0]);
/// let x2 = Array::from_vec(vec![4.0, 5.0, 3.0]);
/// let result = hypot(&x1, &x2).unwrap();
/// assert_eq!(result.to_vec(), vec![5.0, 5.0, 5.0]);
/// ```
pub fn hypot<T: Float + Clone>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>> {
    if x1.shape() != x2.shape() {
        return Err(NumRs2Error::DimensionMismatch(
            "Arrays must have the same shape for hypot".to_string(),
        ));
    }

    let x1_data = x1.to_vec();
    let x2_data = x2.to_vec();
    let result: Vec<T> = x1_data
        .iter()
        .zip(x2_data.iter())
        .map(|(&a, &b)| (a * a + b * b).sqrt())
        .collect();

    Ok(Array::from_vec(result).reshape(&x1.shape()))
}

/// Return the sign of a number (element-wise)
///
/// For arrays, this function returns:
/// - 1.0 if x > 0
/// - 0.0 if x == 0
/// - -1.0 if x < 0
/// - NaN if x is NaN
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Array containing the element-wise sign
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::math::sign;
///
/// let a = Array::from_vec(vec![-2.0, 0.0, 2.5]);
/// let result = sign(&a);
/// assert_eq!(result.to_vec(), vec![-1.0, 0.0, 1.0]);
/// ```
pub fn sign<T: Float + Clone>(array: &Array<T>) -> Array<T> {
    array.map(|x| {
        if x.is_nan() {
            x // Return NaN for NaN input
        } else if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            -T::one()
        } else {
            T::zero()
        }
    })
}

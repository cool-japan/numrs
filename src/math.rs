use crate::array::Array;
use crate::error::Result;
use num_complex::Complex;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::Add;

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

impl<T: Float + Clone> ElementWiseMath<T> for Array<T> {
    // Basic operations
    fn abs(&self) -> Array<T> {
        self.map(|x| x.abs())
    }

    fn exp(&self) -> Array<T> {
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

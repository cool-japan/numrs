//! Array creation functions for NumRS2
//!
//! This module provides functions for creating arrays with specific values or patterns:
//!
//! ## Basic creation functions
//! - [`zeros`] - Create an array filled with zeros
//! - [`ones`] - Create an array filled with ones
//! - [`empty`] - Create an array with default values (safe Rust equivalent of uninitialized)
//!
//! ## Range and sequence functions
//! - [`linspace`] - Create evenly spaced values between start and stop (inclusive)
//! - [`arange`] - Create a sequence of numbers with a specified step
//! - [`logspace`] - Create evenly spaced numbers on a logarithmic scale
//! - [`geomspace`] - Create values evenly spaced on a geometric (log) scale
//!
//! ## Meshgrid functions
//! - [`meshgrid`] - Create coordinate matrices from coordinate vectors (supports xy/ij indexing)
//! - [`meshgrid2d`] - Legacy 2D meshgrid with xy indexing
//! - [`mgrid`] - Create a dense multi-dimensional meshgrid
//! - [`ogrid`] - Create an open (memory-efficient) multi-dimensional meshgrid
//!
//! ## Complex number functions
//! - [`real`] - Extract the real part of complex numbers
//! - [`imag`] - Extract the imaginary part of complex numbers
//! - [`conj`] - Calculate the complex conjugate
//! - [`complex_abs`] - Calculate the magnitude of complex numbers
//! - [`angle`] - Calculate the phase angle (argument) of complex numbers
//! - [`unwrap`] - Unwrap phase angles by changing deltas to 2pi complements

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::Complex;
use std::ops::Add;

/// Create an array filled with zeros
///
/// # Parameters
///
/// * `shape` - The shape of the array to create
///
/// # Returns
///
/// An array filled with zeros with the specified shape
///
/// # Examples
///
/// ```
/// use numrs2::math::zeros;
///
/// let arr: numrs2::array::Array<f64> = zeros(&[3, 4]);
/// assert_eq!(arr.shape(), vec![3, 4]);
/// assert_eq!(arr.get(&[0, 0]).expect("valid index"), 0.0);
/// ```
pub fn zeros<T: Zero + Clone>(shape: &[usize]) -> Array<T> {
    let size: usize = shape.iter().product();
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(T::zero());
    }
    Array::from_vec(vec).reshape(shape)
}

/// Create an array filled with ones
///
/// # Parameters
///
/// * `shape` - The shape of the array to create
///
/// # Returns
///
/// An array filled with ones with the specified shape
///
/// # Examples
///
/// ```
/// use numrs2::math::ones;
///
/// let arr: numrs2::array::Array<f64> = ones(&[2, 3]);
/// assert_eq!(arr.shape(), vec![2, 3]);
/// assert_eq!(arr.get(&[0, 0]).expect("valid index"), 1.0);
/// ```
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
///
/// # Parameters
///
/// * `shape` - The shape of the array to create
///
/// # Returns
///
/// An array with default values with the specified shape
///
/// # Examples
///
/// ```
/// use numrs2::math::empty;
///
/// let arr: numrs2::array::Array<f64> = empty(&[2, 2]);
/// assert_eq!(arr.shape(), vec![2, 2]);
/// ```
pub fn empty<T: Default + Clone>(shape: &[usize]) -> Array<T> {
    let size: usize = shape.iter().product();
    let vec = vec![T::default(); size];
    Array::from_vec(vec).reshape(shape)
}

/// Create evenly spaced values between start and stop (inclusive)
///
/// # Parameters
///
/// * `start` - The starting value of the sequence
/// * `stop` - The end value of the sequence (inclusive)
/// * `num` - Number of samples to generate
///
/// # Returns
///
/// An array with `num` evenly spaced values from `start` to `stop`
///
/// # Examples
///
/// ```
/// use numrs2::math::linspace;
///
/// let arr = linspace(0.0, 1.0, 5);
/// assert_eq!(arr.size(), 5);
/// // Values: 0.0, 0.25, 0.5, 0.75, 1.0
/// ```
pub fn linspace<T: Float + Clone + 'static>(start: T, stop: T, num: usize) -> Array<T> {
    if num < 2 {
        return Array::from_vec(vec![start]);
    }

    let mut vec = Vec::with_capacity(num);
    let step = (stop - start) / T::from(num - 1).expect("num-1 should be representable");

    for i in 0..num {
        vec.push(start + step * T::from(i).expect("index should be representable"));
    }

    Array::from_vec(vec)
}

/// Create a sequence of numbers with a specified step
///
/// # Parameters
///
/// * `start` - The starting value of the sequence
/// * `stop` - The end value (exclusive)
/// * `step` - The step between values
///
/// # Returns
///
/// An array containing values from `start` up to (but not including) `stop`,
/// incrementing by `step`
///
/// # Examples
///
/// ```
/// use numrs2::math::arange;
///
/// let arr = arange(0.0, 5.0, 1.0);
/// assert_eq!(arr.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// let arr_neg = arange(5.0, 0.0, -1.0);
/// assert_eq!(arr_neg.to_vec(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
/// ```
pub fn arange<T>(start: T, stop: T, step: T) -> Array<T>
where
    T: Clone + PartialOrd + NumCast + Add<Output = T> + Zero + 'static,
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
///
/// # Parameters
///
/// * `start` - The starting power (base^start is the first value)
/// * `stop` - The ending power (base^stop is the last value)
/// * `num` - Number of samples to generate
/// * `base` - The base of the log scale (default: 10.0)
///
/// # Returns
///
/// An array with `num` values logarithmically spaced from base^start to base^stop
///
/// # Examples
///
/// ```
/// use numrs2::math::logspace;
///
/// // Create 5 values from 10^0 to 10^4
/// let arr = logspace(0.0, 4.0, 5, None);
/// // Values: 1, 10, 100, 1000, 10000
/// ```
pub fn logspace<T: Float + Clone + 'static>(
    start: T,
    stop: T,
    num: usize,
    base: Option<T>,
) -> Array<T> {
    let base_val = base.unwrap_or_else(|| T::from(10.0).expect("10.0 should be representable"));

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
/// let grids = meshgrid(&[&x, &y], None).expect("meshgrid creation should succeed");
/// let xx = &grids[0];
/// let yy = &grids[1];
/// assert_eq!(xx.shape(), vec![4, 3]);  // (y.len(), x.len())
/// assert_eq!(yy.shape(), vec![4, 3]);
///
/// // Check specific values
/// assert_eq!(xx.get(&[0, 0]).expect("valid index"), 1.0);
/// assert_eq!(xx.get(&[0, 1]).expect("valid index"), 2.0);
/// assert_eq!(xx.get(&[0, 2]).expect("valid index"), 3.0);
/// assert_eq!(yy.get(&[0, 0]).expect("valid index"), 4.0);
/// assert_eq!(yy.get(&[1, 0]).expect("valid index"), 5.0);
///
/// // ij indexing
/// let grids_ij = meshgrid(&[&x, &y], Some("ij")).expect("meshgrid ij creation should succeed");
/// let ii = &grids_ij[0];
/// let jj = &grids_ij[1];
/// assert_eq!(ii.shape(), vec![3, 4]);  // (x.len(), y.len())
/// assert_eq!(jj.shape(), vec![3, 4]);
///
/// // Check specific values for ij indexing
/// assert_eq!(ii.get(&[0, 0]).expect("valid index"), 1.0);
/// assert_eq!(ii.get(&[1, 0]).expect("valid index"), 2.0);
/// assert_eq!(ii.get(&[2, 0]).expect("valid index"), 3.0);
/// assert_eq!(jj.get(&[0, 0]).expect("valid index"), 4.0);
/// assert_eq!(jj.get(&[0, 1]).expect("valid index"), 5.0);
/// ```
///
/// Creates n-dimensional coordinate arrays from 1-dimensional coordinate arrays.
/// Supports both "xy" (default) and "ij" indexing modes to match NumPy's behavior.
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
        return Err(NumRs2Error::InvalidOperation(format!(
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
///
/// # Parameters
///
/// * `x` - The x-coordinate array
/// * `y` - The y-coordinate array
///
/// # Returns
///
/// A tuple of (xx, yy) arrays representing the meshgrid
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
/// ]).expect("mgrid creation should succeed");
/// let xx = &grids[0];
/// let yy = &grids[1];
///
/// assert_eq!(xx.shape(), vec![3, 3]);
/// assert_eq!(yy.shape(), vec![3, 3]);
///
/// // Check values
/// assert_eq!(xx.get(&[0, 0]).expect("valid index"), 0.0);
/// assert_eq!(xx.get(&[0, 1]).expect("valid index"), 0.0);
/// assert_eq!(xx.get(&[0, 2]).expect("valid index"), 0.0);
/// assert_eq!(xx.get(&[1, 0]).expect("valid index"), 1.0);
/// assert_eq!(xx.get(&[1, 1]).expect("valid index"), 1.0);
/// assert_eq!(xx.get(&[1, 2]).expect("valid index"), 1.0);
/// assert_eq!(xx.get(&[2, 0]).expect("valid index"), 2.0);
///
/// assert_eq!(yy.get(&[0, 0]).expect("valid index"), 0.0);
/// assert_eq!(yy.get(&[1, 0]).expect("valid index"), 0.0);
/// assert_eq!(yy.get(&[2, 0]).expect("valid index"), 0.0);
/// assert_eq!(yy.get(&[0, 1]).expect("valid index"), 1.0);
/// assert_eq!(yy.get(&[1, 1]).expect("valid index"), 1.0);
/// assert_eq!(yy.get(&[2, 1]).expect("valid index"), 1.0);
/// assert_eq!(yy.get(&[0, 2]).expect("valid index"), 2.0);
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
            let flat_data = mut_data.as_slice_mut().expect("array should be contiguous");
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
/// ]).expect("ogrid creation should succeed");
/// let xx = &grids[0];
/// let yy = &grids[1];
///
/// assert_eq!(xx.shape(), vec![3, 1]);
/// assert_eq!(yy.shape(), vec![1, 3]);
///
/// // Check values
/// assert_eq!(xx.get(&[0, 0]).expect("valid index"), 0.0);
/// assert_eq!(xx.get(&[1, 0]).expect("valid index"), 1.0);
/// assert_eq!(xx.get(&[2, 0]).expect("valid index"), 2.0);
///
/// assert_eq!(yy.get(&[0, 0]).expect("valid index"), 0.0);
/// assert_eq!(yy.get(&[0, 1]).expect("valid index"), 1.0);
/// assert_eq!(yy.get(&[0, 2]).expect("valid index"), 2.0);
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
/// assert_eq!(r_squared.get(&[0, 0]).expect("valid index"), 0.0);
/// assert_eq!(r_squared.get(&[1, 1]).expect("valid index"), 2.0);
/// assert_eq!(r_squared.get(&[2, 2]).expect("valid index"), 8.0);
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

/// Create an array with values evenly spaced on a log scale (geometric sequence)
///
/// # Parameters
///
/// * `start` - The starting value (must be positive)
/// * `stop` - The end value (must be positive)
/// * `num` - Number of samples to generate
///
/// # Returns
///
/// An array with `num` values geometrically spaced from `start` to `stop`
///
/// # Panics
///
/// Panics if `start` or `stop` are not positive
///
/// # Examples
///
/// ```
/// use numrs2::math::geomspace;
///
/// let arr = geomspace(1.0, 1000.0, 4);
/// // Values: 1, 10, 100, 1000 (approximately)
/// ```
pub fn geomspace<T: Float + Clone + 'static>(start: T, stop: T, num: usize) -> Array<T> {
    if start <= T::zero() || stop <= T::zero() {
        panic!("geomspace requires positive start and stop values");
    }

    let log_start = start.ln();
    let log_stop = stop.ln();

    linspace(log_start, log_stop, num).map(|x| x.exp())
}

// Complex number functions

/// Extract the real part of complex numbers
///
/// # Parameters
///
/// * `complex_array` - An array of complex numbers
///
/// # Returns
///
/// An array containing the real parts of the input complex numbers
///
/// # Examples
///
/// ```
/// use numrs2::math::real;
/// use numrs2::array::Array;
/// use scirs2_core::Complex;
///
/// let c = Array::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
/// let r = real(&c);
/// assert_eq!(r.to_vec(), vec![1.0, 3.0]);
/// ```
pub fn real<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.re)
}

/// Extract the imaginary part of complex numbers
///
/// # Parameters
///
/// * `complex_array` - An array of complex numbers
///
/// # Returns
///
/// An array containing the imaginary parts of the input complex numbers
///
/// # Examples
///
/// ```
/// use numrs2::math::imag;
/// use numrs2::array::Array;
/// use scirs2_core::Complex;
///
/// let c = Array::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
/// let i = imag(&c);
/// assert_eq!(i.to_vec(), vec![2.0, 4.0]);
/// ```
pub fn imag<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.im)
}

/// Calculate the complex conjugate of complex numbers
///
/// # Parameters
///
/// * `complex_array` - An array of complex numbers
///
/// # Returns
///
/// An array containing the complex conjugates of the input numbers
///
/// # Examples
///
/// ```
/// use numrs2::math::conj;
/// use numrs2::array::Array;
/// use scirs2_core::Complex;
///
/// let c = Array::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)]);
/// let conjugate = conj(&c);
/// assert_eq!(conjugate.to_vec(), vec![Complex::new(1.0, -2.0), Complex::new(3.0, 4.0)]);
/// ```
pub fn conj<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<Complex<T>> {
    complex_array.map(|c| c.conj())
}

/// Calculate the absolute value (magnitude) of complex numbers
///
/// # Parameters
///
/// * `complex_array` - An array of complex numbers
///
/// # Returns
///
/// An array containing the magnitudes (|z| = sqrt(re^2 + im^2)) of the input numbers
///
/// # Examples
///
/// ```
/// use numrs2::math::complex_abs;
/// use numrs2::array::Array;
/// use scirs2_core::Complex;
///
/// let c = Array::from_vec(vec![Complex::new(3.0, 4.0)]);
/// let magnitude = complex_abs(&c);
/// assert_eq!(magnitude.to_vec(), vec![5.0]);
/// ```
pub fn complex_abs<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.norm())
}

/// Calculate the phase angle (argument) of complex numbers
///
/// # Parameters
///
/// * `complex_array` - An array of complex numbers
///
/// # Returns
///
/// An array containing the phase angles (in radians) of the input numbers
///
/// # Examples
///
/// ```
/// use numrs2::math::angle;
/// use numrs2::array::Array;
/// use scirs2_core::Complex;
/// use std::f64::consts::PI;
///
/// let c = Array::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]);
/// let phase = angle(&c);
/// assert!((phase.get(&[0]).expect("valid index") - 0.0_f64).abs() < 1e-10);
/// assert!((phase.get(&[1]).expect("valid index") - PI / 2.0_f64).abs() < 1e-10);
/// ```
pub fn angle<T: Float + Clone>(complex_array: &Array<Complex<T>>) -> Array<T> {
    complex_array.map(|c| c.arg())
}

/// Unwrap phase angles by changing deltas between values to 2*pi complements
///
/// This function unwraps a sequence of phase values by adjusting jumps greater
/// than pi to their 2*pi complement. This is useful for continuous phase signals
/// that have been wrapped to the [-pi, pi] range.
///
/// # Parameters
///
/// * `phase_array` - An array of phase values (in radians)
///
/// # Returns
///
/// An array with unwrapped phase values
///
/// # Examples
///
/// ```
/// use numrs2::math::unwrap;
/// use numrs2::array::Array;
/// use std::f64::consts::PI;
///
/// // Phase values that jump from near pi to near -pi
/// let wrapped = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0]);
/// let unwrapped = unwrap(&wrapped);
/// // The jump from 3.0 to -3.0 should be unwrapped
/// ```
pub fn unwrap<T: Float + Clone>(phase_array: &Array<T>) -> Array<T> {
    // If array is empty or has a single element, there's nothing to unwrap
    if phase_array.size() <= 1 {
        return phase_array.clone();
    }

    let data = phase_array.to_vec();
    let mut result = Vec::with_capacity(data.len());

    // Start with the first phase value
    result.push(data[0]);

    // The 2*pi value
    let two_pi = T::from(2.0 * std::f64::consts::PI).expect("2*PI should be representable");
    let pi = T::from(std::f64::consts::PI).expect("PI should be representable");

    // Process the rest of the array
    for i in 1..data.len() {
        let mut delta = data[i] - data[i - 1];

        // Adjust for jumps larger than pi
        while delta > pi {
            delta = delta - two_pi;
        }
        while delta < -pi {
            delta = delta + two_pi;
        }

        result.push(result[i - 1] + delta);
    }

    Array::from_vec(result)
}

/// Return coordinate matrices from coordinate vectors (internal helper)
///
/// This function is similar to mgrid but takes owned references.
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
            let flat_data = mut_data.as_slice_mut().expect("array should be contiguous");
            flat_data[flat_idx] = range_val;
        }
    }

    Ok(output)
}

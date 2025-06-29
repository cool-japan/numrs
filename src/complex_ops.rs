//! Complex number operations for Arrays
//!
//! This module provides functions for working with complex numbers in arrays,
//! including extracting real/imaginary parts, computing phase angles, and conjugates.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_complex::Complex;
use num_traits::Float;

/// Extract the real part of complex numbers
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<T>` - Array containing only the real parts
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::real;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 2.0),
///     Complex::new(3.0, -4.0),
///     Complex::new(-5.0, 6.0),
/// ]);
/// let real_parts = real(&complex_array);
/// // Returns [1.0, 3.0, -5.0]
/// ```
pub fn real<T: Float>(array: &Array<Complex<T>>) -> Array<T> {
    array.map(|c| c.re)
}

/// Extract the imaginary part of complex numbers
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<T>` - Array containing only the imaginary parts
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::imag;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 2.0),
///     Complex::new(3.0, -4.0),
///     Complex::new(-5.0, 6.0),
/// ]);
/// let imag_parts = imag(&complex_array);
/// // Returns [2.0, -4.0, 6.0]
/// ```
pub fn imag<T: Float>(array: &Array<Complex<T>>) -> Array<T> {
    array.map(|c| c.im)
}

/// Return the angle of the complex argument (phase angle)
///
/// # Arguments
/// * `array` - Array of complex numbers
/// * `deg` - If true, return angle in degrees; otherwise in radians (default: false)
///
/// # Returns
/// * `Array<T>` - Array containing the phase angles
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::angle;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 1.0),   // 45 degrees
///     Complex::new(1.0, 0.0),   // 0 degrees
///     Complex::new(0.0, 1.0),   // 90 degrees
///     Complex::new(-1.0, 0.0),  // 180 degrees
/// ]);
/// let angles_rad = angle(&complex_array, false);
/// let angles_deg = angle(&complex_array, true);
/// ```
pub fn angle<T: Float>(array: &Array<Complex<T>>, deg: bool) -> Array<T> {
    array.map(|c| {
        let phase = c.arg(); // Returns angle in radians
        if deg {
            phase * T::from(180.0).unwrap() / T::from(std::f64::consts::PI).unwrap()
        } else {
            phase
        }
    })
}

/// Return the complex conjugate, element-wise
///
/// The complex conjugate of a complex number is obtained by changing
/// the sign of its imaginary part.
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<Complex<T>>` - Array containing the complex conjugates
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::conj;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 2.0),
///     Complex::new(3.0, -4.0),
///     Complex::new(-5.0, 6.0),
/// ]);
/// let conjugates = conj(&complex_array);
/// // Returns [1-2i, 3+4i, -5-6i]
/// ```
pub fn conj<T: Float>(array: &Array<Complex<T>>) -> Array<Complex<T>> {
    array.map(|c| c.conj())
}

/// Convert real numbers to complex numbers with zero imaginary part
///
/// # Arguments
/// * `array` - Array of real numbers
///
/// # Returns
/// * `Array<Complex<T>>` - Array of complex numbers with zero imaginary parts
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::to_complex;
///
/// let real_array = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let complex_array = to_complex(&real_array);
/// // Returns [1+0i, 2+0i, 3+0i]
/// ```
pub fn to_complex<T: Float>(array: &Array<T>) -> Array<Complex<T>> {
    array.map(|x| Complex::new(x, T::zero()))
}

/// Compute absolute value (magnitude) of complex numbers
///
/// For complex numbers, this computes sqrt(re^2 + im^2).
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<T>` - Array containing the magnitudes
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::absolute;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(3.0, 4.0),   // magnitude = 5.0
///     Complex::new(5.0, 12.0),  // magnitude = 13.0
/// ]);
/// let magnitudes = absolute(&complex_array);
/// // Returns [5.0, 13.0]
/// ```
pub fn absolute<T: Float>(array: &Array<Complex<T>>) -> Array<T> {
    array.map(|c| c.norm())
}

/// Create a complex number from magnitude and phase angle
///
/// # Arguments
/// * `magnitude` - Array of magnitudes
/// * `angle` - Array of phase angles
/// * `deg` - If true, angles are in degrees; otherwise in radians
///
/// # Returns
/// * `Result<Array<Complex<T>>>` - Array of complex numbers
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::from_polar;
///
/// let magnitudes = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let angles = Array::from_vec(vec![0.0, 90.0, 180.0]);
/// let complex_array = from_polar(&magnitudes, &angles, true).unwrap();
/// ```
pub fn from_polar<T: Float>(
    magnitude: &Array<T>,
    angle: &Array<T>,
    deg: bool,
) -> Result<Array<Complex<T>>> {
    if magnitude.shape() != angle.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: magnitude.shape(),
            actual: angle.shape(),
        });
    }

    let mag_data = magnitude.to_vec();
    let angle_data = angle.to_vec();

    let result: Vec<Complex<T>> = mag_data
        .into_iter()
        .zip(angle_data)
        .map(|(r, theta)| {
            let theta_rad = if deg {
                theta * T::from(std::f64::consts::PI).unwrap() / T::from(180.0).unwrap()
            } else {
                theta
            };
            Complex::from_polar(r, theta_rad)
        })
        .collect();

    Ok(Array::from_vec(result).reshape(&magnitude.shape()))
}

/// Check if values are complex (have non-zero imaginary part)
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<bool>` - Boolean array where true indicates non-zero imaginary part
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::iscomplex;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 0.0),    // false (real)
///     Complex::new(1.0, 2.0),    // true (complex)
///     Complex::new(0.0, 3.0),    // true (complex)
/// ]);
/// let is_complex = iscomplex(&complex_array);
/// // Returns [false, true, true]
/// ```
pub fn iscomplex<T: Float>(array: &Array<Complex<T>>) -> Array<bool> {
    array.map(|c| c.im != T::zero())
}

/// Check if values are real (have zero imaginary part)
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `Array<bool>` - Boolean array where true indicates zero imaginary part
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::isreal;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 0.0),    // true (real)
///     Complex::new(1.0, 2.0),    // false (complex)
///     Complex::new(0.0, 0.0),    // true (real)
/// ]);
/// let is_real = isreal(&complex_array);
/// // Returns [true, false, true]
/// ```
pub fn isreal<T: Float>(array: &Array<Complex<T>>) -> Array<bool> {
    array.map(|c| c.im == T::zero())
}

/// Check if an array has a complex data type
///
/// Returns `true` if the array's data type is complex, regardless of the values.
/// This is different from `iscomplex()` which checks if individual values have
/// non-zero imaginary parts.
///
/// # Arguments
/// * `array` - Array of complex numbers
///
/// # Returns
/// * `bool` - True if the array data type is complex
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::iscomplexobj;
/// use num_complex::Complex;
///
/// let complex_array = Array::from_vec(vec![
///     Complex::new(1.0, 0.0),    // Even though imaginary part is zero
///     Complex::new(2.0, 0.0),    // the array type is still complex
/// ]);
/// assert_eq!(iscomplexobj(&complex_array), true);
/// ```
pub fn iscomplexobj<T: Float>(_array: &Array<Complex<T>>) -> bool {
    // If we have an Array<Complex<T>>, the data type is always complex
    true
}

/// Check if an array has a real data type
///
/// Returns `true` if the array's data type is real (not complex).
/// This is the opposite of `iscomplexobj()`.
///
/// # Arguments
/// * `array` - Array of real numbers
///
/// # Returns
/// * `bool` - True if the array data type is real
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::complex_ops::isrealobj;
///
/// let real_array = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(isrealobj(&real_array), true);
/// ```
pub fn isrealobj<T: Float>(_array: &Array<T>) -> bool {
    // If we have an Array<T> where T is not Complex, the data type is real
    true
}

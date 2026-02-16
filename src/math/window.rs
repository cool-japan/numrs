//! Window functions for signal processing
//!
//! This module provides various window functions commonly used in signal processing
//! and spectral analysis. Window functions are used to reduce spectral leakage
//! when performing Fourier transforms on finite-length signals.
//!
//! # Available Window Functions
//!
//! - [`hanning`] - Hanning (Hann) window with cosine taper
//! - [`hamming`] - Hamming window with weighted cosine
//! - [`blackman`] - Blackman window for better sidelobe suppression
//! - [`bartlett`] - Bartlett (triangular) window with zero endpoints
//! - [`kaiser`] - Kaiser window with adjustable shape parameter
//! - [`i0`] - Modified Bessel function of the first kind, order 0
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//!
//! // Create a 64-point Hanning window
//! let window = hanning(64);
//!
//! // Create a Kaiser window with beta=8.6 (similar to Blackman)
//! let kaiser_win = kaiser(64, 8.6);
//! ```

use crate::array::Array;
use num_traits::Float;

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
/// Computes I_0(x) for each element in the input array.
/// This function is commonly used in signal processing and physics.
///
/// # Parameters
///
/// * `x` - Input array
///
/// # Returns
///
/// Array of I_0(x) values
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
        let x_f64 = val.to_f64().expect("value should be convertible to f64");
        let result = modified_bessel_i0(x_f64.abs());
        T::from(result).expect("bessel result should be representable")
    })
}

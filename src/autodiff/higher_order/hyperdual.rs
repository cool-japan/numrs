//! HyperDual number type and arithmetic for exact second-order derivatives.

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Convert an f64 constant to a generic Float type with proper error handling
pub(super) fn float_const<T: Float>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| {
        NumRs2Error::NumericalError(format!("Cannot represent {} in target float type", val))
    })
}

/// Hyper-dual number for exact computation of second-order derivatives.
///
/// A hyper-dual number extends the dual number concept to capture both first
/// and second derivatives exactly (no numerical approximation). It represents:
///
/// `f = a + b*e1 + c*e2 + d*e1*e2`
///
/// where `e1^2 = e2^2 = 0` but `e1*e2 != 0`.
///
/// When evaluating `f(x)` with `x_k = x_k + delta(k,i)*e1 + delta(k,j)*e2`:
/// - `real` = f(x)
/// - `eps1` = df/dx_i
/// - `eps2` = df/dx_j
/// - `eps1eps2` = d^2f/(dx_i dx_j)  (exact second derivative!)
///
/// # Type Parameters
///
/// * `T` - The numeric type (typically `f32` or `f64`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual<T> {
    /// Function value (real part)
    real: T,
    /// First derivative w.r.t. first perturbation direction (e1 coefficient)
    eps1: T,
    /// First derivative w.r.t. second perturbation direction (e2 coefficient)
    eps2: T,
    /// Second cross derivative (e1*e2 coefficient)
    eps1eps2: T,
}

impl<T: Float> HyperDual<T> {
    /// Create a new hyper-dual number with all four components
    ///
    /// # Arguments
    ///
    /// * `real` - The function value
    /// * `eps1` - First derivative component (e1 direction)
    /// * `eps2` - First derivative component (e2 direction)
    /// * `eps1eps2` - Second cross derivative component
    #[inline]
    pub fn new(real: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        Self {
            real,
            eps1,
            eps2,
            eps1eps2,
        }
    }

    /// Create a constant hyper-dual number (all derivatives zero)
    #[inline]
    pub fn constant(value: T) -> Self {
        Self::new(value, T::zero(), T::zero(), T::zero())
    }

    /// Create a hyper-dual variable for computing d^2f/(dx_i dx_j)
    ///
    /// For input component x_k, set:
    /// - eps1 = 1 if k == i (variable index for first derivative direction)
    /// - eps2 = 1 if k == j (variable index for second derivative direction)
    /// - eps1eps2 = 0 always
    #[inline]
    pub fn make_variable(value: T, is_dir_i: bool, is_dir_j: bool) -> Self {
        Self::new(
            value,
            if is_dir_i { T::one() } else { T::zero() },
            if is_dir_j { T::one() } else { T::zero() },
            T::zero(),
        )
    }

    /// Get the function value (real part)
    #[inline]
    pub fn real(&self) -> T {
        self.real
    }

    /// Get the first derivative w.r.t. direction 1 (e1 coefficient)
    #[inline]
    pub fn eps1(&self) -> T {
        self.eps1
    }

    /// Get the first derivative w.r.t. direction 2 (e2 coefficient)
    #[inline]
    pub fn eps2(&self) -> T {
        self.eps2
    }

    /// Get the second cross derivative (e1*e2 coefficient)
    ///
    /// This is the key output: the exact mixed partial derivative d^2f/(dx_i dx_j).
    #[inline]
    pub fn eps1eps2(&self) -> T {
        self.eps1eps2
    }

    /// Scale all components by a scalar value
    #[inline]
    pub fn scale(self, s: T) -> Self {
        Self::new(
            self.real * s,
            self.eps1 * s,
            self.eps2 * s,
            self.eps1eps2 * s,
        )
    }

    // ========================================================================
    // Mathematical Functions
    // ========================================================================
    //
    // For a smooth function g applied to hyper-dual x = a + b*e1 + c*e2 + d*e1*e2:
    // g(x) = g(a) + g'(a)*b*e1 + g'(a)*c*e2 + (g''(a)*b*c + g'(a)*d)*e1*e2

    /// Compute power with float exponent: x^n
    ///
    /// Note: Requires `self.real() > 0` for non-integer exponents.
    /// For integer powers with possibly negative base, use `powi`.
    pub fn powf(self, n: T) -> Self {
        let a = self.real;
        let gp = n * a.powf(n - T::one());
        let gpp = n * (n - T::one()) * a.powf(n - T::one() - T::one());
        Self::new(
            a.powf(n),
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute integer power: x^n (safe for negative base)
    ///
    /// Uses exponentiation by squaring via operator overloading,
    /// which correctly propagates derivatives through multiplication.
    pub fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::constant(T::one()),
            1 => self,
            _ if n < 0 => Self::constant(T::one()) / self.powi(-n),
            _ => {
                let half = self.powi(n / 2);
                if n % 2 == 0 {
                    half * half
                } else {
                    half * half * self
                }
            }
        }
    }

    /// Compute exponential: e^x
    pub fn exp(self) -> Self {
        let ea = self.real.exp();
        // g'(a) = e^a, g''(a) = e^a
        Self::new(
            ea,
            ea * self.eps1,
            ea * self.eps2,
            ea * (self.eps1 * self.eps2 + self.eps1eps2),
        )
    }

    /// Compute natural logarithm: ln(x)
    pub fn ln(self) -> Self {
        let a = self.real;
        // g'(a) = 1/a, g''(a) = -1/a^2
        let gp = T::one() / a;
        let gpp = -T::one() / (a * a);
        Self::new(
            a.ln(),
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute sine: sin(x)
    pub fn sin(self) -> Self {
        let sin_a = self.real.sin();
        let cos_a = self.real.cos();
        // g'(a) = cos(a), g''(a) = -sin(a)
        Self::new(
            sin_a,
            cos_a * self.eps1,
            cos_a * self.eps2,
            -sin_a * self.eps1 * self.eps2 + cos_a * self.eps1eps2,
        )
    }

    /// Compute cosine: cos(x)
    pub fn cos(self) -> Self {
        let sin_a = self.real.sin();
        let cos_a = self.real.cos();
        // g'(a) = -sin(a), g''(a) = -cos(a)
        Self::new(
            cos_a,
            -sin_a * self.eps1,
            -sin_a * self.eps2,
            -cos_a * self.eps1 * self.eps2 - sin_a * self.eps1eps2,
        )
    }

    /// Compute tangent: tan(x)
    pub fn tan(self) -> Self {
        let tan_a = self.real.tan();
        let cos_a = self.real.cos();
        let sec2 = T::one() / (cos_a * cos_a);
        let two = T::one() + T::one();
        // g'(a) = sec^2(a), g''(a) = 2*tan(a)*sec^2(a)
        let gp = sec2;
        let gpp = two * tan_a * sec2;
        Self::new(
            tan_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute hyperbolic sine: sinh(x)
    pub fn sinh(self) -> Self {
        let sinh_a = self.real.sinh();
        let cosh_a = self.real.cosh();
        // g'(a) = cosh(a), g''(a) = sinh(a)
        Self::new(
            sinh_a,
            cosh_a * self.eps1,
            cosh_a * self.eps2,
            sinh_a * self.eps1 * self.eps2 + cosh_a * self.eps1eps2,
        )
    }

    /// Compute hyperbolic cosine: cosh(x)
    pub fn cosh(self) -> Self {
        let sinh_a = self.real.sinh();
        let cosh_a = self.real.cosh();
        // g'(a) = sinh(a), g''(a) = cosh(a)
        Self::new(
            cosh_a,
            sinh_a * self.eps1,
            sinh_a * self.eps2,
            cosh_a * self.eps1 * self.eps2 + sinh_a * self.eps1eps2,
        )
    }

    /// Compute hyperbolic tangent: tanh(x)
    pub fn tanh(self) -> Self {
        let tanh_a = self.real.tanh();
        let sech2 = T::one() - tanh_a * tanh_a;
        let two = T::one() + T::one();
        // g'(a) = sech^2(a), g''(a) = -2*tanh(a)*sech^2(a)
        let gp = sech2;
        let gpp = -two * tanh_a * sech2;
        Self::new(
            tanh_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute square root: sqrt(x)
    pub fn sqrt(self) -> Self {
        let a = self.real;
        let sqrt_a = a.sqrt();
        let two = T::one() + T::one();
        let four = two * two;
        // g'(a) = 1/(2*sqrt(a)), g''(a) = -1/(4*a^(3/2))
        let gp = T::one() / (two * sqrt_a);
        let gpp = -T::one() / (four * a * sqrt_a);
        Self::new(
            sqrt_a,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute absolute value: |x|
    ///
    /// Note: Derivative at x=0 is set to 0 by convention.
    pub fn abs(self) -> Self {
        if self.real >= T::zero() {
            self
        } else {
            -self
        }
    }

    /// Compute sigmoid function: 1 / (1 + e^(-x))
    pub fn sigmoid(self) -> Self {
        let a = self.real;
        let s = T::one() / (T::one() + (-a).exp());
        let two = T::one() + T::one();
        // g'(a) = s*(1-s), g''(a) = s*(1-s)*(1-2s)
        let gp = s * (T::one() - s);
        let gpp = gp * (T::one() - two * s);
        Self::new(
            s,
            gp * self.eps1,
            gp * self.eps2,
            gpp * self.eps1 * self.eps2 + gp * self.eps1eps2,
        )
    }

    /// Compute ReLU: max(0, x)
    ///
    /// Note: Derivative at x=0 is set to 0 by convention.
    pub fn relu(self) -> Self {
        if self.real > T::zero() {
            self
        } else {
            Self::constant(T::zero())
        }
    }
}

// ============================================================================
// Arithmetic Operators for HyperDual Numbers
// ============================================================================

impl<T: Float> Add for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real + rhs.real,
            self.eps1 + rhs.eps1,
            self.eps2 + rhs.eps2,
            self.eps1eps2 + rhs.eps1eps2,
        )
    }
}

impl<T: Float> Sub for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real - rhs.real,
            self.eps1 - rhs.eps1,
            self.eps2 - rhs.eps2,
            self.eps1eps2 - rhs.eps1eps2,
        )
    }
}

impl<T: Float> Mul for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + b*e1 + c*e2 + d*e1e2) * (e + f*e1 + g*e2 + h*e1e2)
        // real: a*e
        // e1:   a*f + b*e
        // e2:   a*g + c*e
        // e1e2: a*h + b*g + c*f + d*e
        Self::new(
            self.real * rhs.real,
            self.real * rhs.eps1 + self.eps1 * rhs.real,
            self.real * rhs.eps2 + self.eps2 * rhs.real,
            self.real * rhs.eps1eps2
                + self.eps1 * rhs.eps2
                + self.eps2 * rhs.eps1
                + self.eps1eps2 * rhs.real,
        )
    }
}

impl<T: Float> Div for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        // u/v derivation (see module-level documentation)
        let (a, b, c, d) = (self.real, self.eps1, self.eps2, self.eps1eps2);
        let (e, f, g, h) = (rhs.real, rhs.eps1, rhs.eps2, rhs.eps1eps2);
        let e2 = e * e;
        let e3 = e2 * e;
        let two = T::one() + T::one();

        Self::new(
            a / e,
            (b * e - a * f) / e2,
            (c * e - a * g) / e2,
            (d * e2 - a * h * e - b * g * e - c * f * e + two * a * f * g) / e3,
        )
    }
}

impl<T: Float> Neg for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.real, -self.eps1, -self.eps2, -self.eps1eps2)
    }
}

impl<T: Float + fmt::Display> fmt::Display for HyperDual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + {}*e1 + {}*e2 + {}*e1e2",
            self.real, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

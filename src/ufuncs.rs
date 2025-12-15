use crate::array::Array;
use crate::error::Result;
#[cfg(target_arch = "x86_64")]
use crate::simd_optimize::avx2_enhanced::EnhancedSimdOps;
#[cfg(target_arch = "aarch64")]
use crate::simd_optimize::neon_enhanced::NeonEnhancedOps;
use std::fmt;

/// Threshold for using SIMD-optimized implementations.
/// Arrays larger than this will use vectorized operations (AVX2 on x86_64, NEON on aarch64).
const SIMD_THRESHOLD: usize = 32;

/// Universal Function (ufunc) for element-wise binary operations with broadcasting
pub struct BinaryUfunc<F>
where
    F: Fn(f64, f64) -> f64,
{
    func: F,
    name: &'static str,
}

/// Universal Function (ufunc) for element-wise unary operations with broadcasting
pub struct UnaryUfunc<F>
where
    F: Fn(f64) -> f64,
{
    func: F,
    name: &'static str,
}

impl<F> fmt::Debug for BinaryUfunc<F>
where
    F: Fn(f64, f64) -> f64,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryUfunc({})", self.name)
    }
}

impl<F> fmt::Debug for UnaryUfunc<F>
where
    F: Fn(f64) -> f64,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnaryUfunc({})", self.name)
    }
}

impl<F> BinaryUfunc<F>
where
    F: Fn(f64, f64) -> f64,
{
    /// Create a new binary ufunc
    pub fn new(func: F, name: &'static str) -> Self {
        Self { func, name }
    }

    /// Apply the function to two arrays with broadcasting
    pub fn call(&self, a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
        // Use the broadcasting operation we implemented earlier
        a.zip_with(b, |x, y| (self.func)(x, y))
    }

    /// Apply the function to an array and a scalar with broadcasting
    pub fn call_scalar_right(&self, a: &Array<f64>, b: f64) -> Array<f64> {
        a.map(|x| (self.func)(x, b))
    }

    /// Apply the function to a scalar and an array with broadcasting
    pub fn call_scalar_left(&self, a: f64, b: &Array<f64>) -> Array<f64> {
        b.map(|x| (self.func)(a, x))
    }
}

impl<F> UnaryUfunc<F>
where
    F: Fn(f64) -> f64,
{
    /// Create a new unary ufunc
    pub fn new(func: F, name: &'static str) -> Self {
        Self { func, name }
    }

    /// Apply the function to an array with broadcasting
    pub fn call(&self, a: &Array<f64>) -> Array<f64> {
        a.map(|x| (self.func)(x))
    }
}

// Helper function to get commonly used binary ufuncs
fn get_add_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| a + b, "add")
}

fn get_subtract_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| a - b, "subtract")
}

fn get_multiply_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| a * b, "multiply")
}

fn get_divide_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| a / b, "divide")
}

fn get_power_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| a.powf(b), "power")
}

fn get_maximum_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| if a > b { a } else { b }, "maximum")
}

fn get_minimum_ufunc() -> BinaryUfunc<fn(f64, f64) -> f64> {
    BinaryUfunc::new(|a, b| if a < b { a } else { b }, "minimum")
}

// Helper functions for unary ufuncs
fn get_negative_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| -a, "negative")
}

fn get_absolute_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.abs(), "absolute")
}

fn get_square_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a * a, "square")
}

fn get_sqrt_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.sqrt(), "sqrt")
}

fn get_exp_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.exp(), "exp")
}

fn get_log_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.ln(), "log")
}

fn get_sin_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.sin(), "sin")
}

fn get_cos_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.cos(), "cos")
}

fn get_tan_ufunc() -> UnaryUfunc<fn(f64) -> f64> {
    UnaryUfunc::new(|a| a.tan(), "tan")
}

// Convenience functions for the ufuncs with SIMD optimization

/// Element-wise addition with SIMD optimization for large arrays
pub fn add(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_add_arrays_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_add_arrays_f64(a, b));
    }
    get_add_ufunc().call(a, b)
}

/// Element-wise subtraction with SIMD optimization for large arrays
pub fn subtract(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_sub_arrays_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_sub_arrays_f64(a, b));
    }
    get_subtract_ufunc().call(a, b)
}

/// Element-wise multiplication with SIMD optimization for large arrays
pub fn multiply(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_mul_arrays_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_mul_arrays_f64(a, b));
    }
    get_multiply_ufunc().call(a, b)
}

/// Element-wise division with SIMD optimization for large arrays
pub fn divide(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_div_arrays_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_div_arrays_f64(a, b));
    }
    get_divide_ufunc().call(a, b)
}

/// Element-wise power with SIMD optimization for large arrays
pub fn power(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_pow_f64(a, b));
    }
    get_power_ufunc().call(a, b)
}

/// Element-wise maximum with SIMD optimization for large arrays
pub fn maximum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_maximum_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_maximum_f64(a, b));
    }
    get_maximum_ufunc().call(a, b)
}

/// Element-wise minimum with SIMD optimization for large arrays
pub fn minimum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(EnhancedSimdOps::vectorized_minimum_f64(a, b));
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && a.len() == b.len() {
        return Ok(NeonEnhancedOps::vectorized_minimum_f64(a, b));
    }
    get_minimum_ufunc().call(a, b)
}

/// Scalar addition with SIMD optimization for large arrays
pub fn add_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_add_scalar_f64(a, b);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_add_scalar_f64(a, b);
    }
    get_add_ufunc().call_scalar_right(a, b)
}

/// Scalar subtraction with SIMD optimization for large arrays
pub fn subtract_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_sub_scalar_f64(a, b);
    }
    get_subtract_ufunc().call_scalar_right(a, b)
}

/// Scalar multiplication with SIMD optimization for large arrays
pub fn multiply_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_mul_scalar_f64(a, b);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_mul_scalar_f64(a, b);
    }
    get_multiply_ufunc().call_scalar_right(a, b)
}

/// Scalar division with SIMD optimization for large arrays
pub fn divide_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_div_scalar_f64(a, b);
    }
    get_divide_ufunc().call_scalar_right(a, b)
}

/// Scalar power with SIMD optimization for large arrays
pub fn power_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_pow_scalar_f64(a, b);
    }
    get_power_ufunc().call_scalar_right(a, b)
}

/// Element-wise negation with SIMD optimization for large arrays
pub fn negative(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_negative_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_negative_f64(a);
    }
    get_negative_ufunc().call(a)
}

/// Absolute value with SIMD optimization for large arrays
pub fn absolute(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_abs_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_abs_f64(a);
    }
    get_absolute_ufunc().call(a)
}

/// Square each element with SIMD optimization for large arrays
pub fn square(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_square_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_square_f64(a);
    }
    get_square_ufunc().call(a)
}

/// Square root with SIMD optimization for large arrays
pub fn sqrt(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_sqrt_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_sqrt_f64(a);
    }
    get_sqrt_ufunc().call(a)
}

/// Exponential function with SIMD optimization for large arrays
pub fn exp(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_exp_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_exp_f64(a);
    }
    get_exp_ufunc().call(a)
}

/// Natural logarithm with SIMD optimization for large arrays
pub fn log(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_log_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_log_f64(a);
    }
    get_log_ufunc().call(a)
}

/// Sine function with SIMD optimization for large arrays
pub fn sin(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_sin_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_sin_f64(a);
    }
    get_sin_ufunc().call(a)
}

/// Cosine function with SIMD optimization for large arrays
pub fn cos(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_cos_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_cos_f64(a);
    }
    get_cos_ufunc().call(a)
}

/// Tangent function with SIMD optimization for large arrays
pub fn tan(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_tan_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_tan_f64(a);
    }
    get_tan_ufunc().call(a)
}

/// Inverse sine with SIMD optimization for large arrays
pub fn arcsin(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_asin_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_asin_f64(a);
    }
    a.map(|x| x.asin())
}

/// Inverse cosine with SIMD optimization for large arrays
pub fn arccos(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_acos_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_acos_f64(a);
    }
    a.map(|x| x.acos())
}

/// Inverse tangent with SIMD optimization for large arrays
pub fn arctan(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_atan_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_atan_f64(a);
    }
    a.map(|x| x.atan())
}

/// Hyperbolic sine with SIMD optimization for large arrays
pub fn sinh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_sinh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_sinh_f64(a);
    }
    a.map(|x| x.sinh())
}

/// Hyperbolic cosine with SIMD optimization for large arrays
pub fn cosh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_cosh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_cosh_f64(a);
    }
    a.map(|x| x.cosh())
}

/// Hyperbolic tangent with SIMD optimization for large arrays
pub fn tanh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_tanh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_tanh_f64(a);
    }
    a.map(|x| x.tanh())
}

/// Inverse hyperbolic sine with SIMD optimization for large arrays
pub fn arcsinh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_asinh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_asinh_f64(a);
    }
    a.map(|x| x.asinh())
}

/// Inverse hyperbolic cosine with SIMD optimization for large arrays
pub fn arccosh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_acosh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_acosh_f64(a);
    }
    a.map(|x| x.acosh())
}

/// Inverse hyperbolic tangent with SIMD optimization for large arrays
pub fn arctanh(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_atanh_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_atanh_f64(a);
    }
    a.map(|x| x.atanh())
}

/// Floor function with SIMD optimization for large arrays
pub fn floor(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_floor_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_floor_f64(a);
    }
    a.map(|x| x.floor())
}

/// Ceiling function with SIMD optimization for large arrays
pub fn ceil(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_ceil_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_ceil_f64(a);
    }
    a.map(|x| x.ceil())
}

/// Round to nearest integer with SIMD optimization for large arrays
pub fn round(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_round_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_round_f64(a);
    }
    a.map(|x| x.round())
}

/// Sign function with SIMD optimization for large arrays
pub fn sign(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD {
        return EnhancedSimdOps::vectorized_sign_f64(a);
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_sign_f64(a);
    }
    a.map(|x| {
        if x == 0.0 {
            0.0
        } else if x > 0.0 {
            1.0
        } else {
            -1.0
        }
    })
}

// Add function to wrap the Result to handle the unwrap in the test
pub fn abs(a: &Array<f64>) -> Array<f64> {
    absolute(a)
}

// =============================================================================
// ADDITIONAL SIMD-OPTIMIZED MATHEMATICAL FUNCTIONS
// =============================================================================

/// Base-2 logarithm with SIMD optimization for large arrays
pub fn log2(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_log2_f64(a);
    }
    a.map(|x| x.log2())
}

/// Base-10 logarithm with SIMD optimization for large arrays
pub fn log10(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_log10_f64(a);
    }
    a.map(|x| x.log10())
}

/// Base-2 exponential with SIMD optimization for large arrays
pub fn exp2(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_exp2_f64(a);
    }
    a.map(|x| x.exp2())
}

/// exp(x) - 1 with improved precision for small x, SIMD optimized
pub fn expm1(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_expm1_f64(a);
    }
    a.map(|x| x.exp_m1())
}

/// log(1 + x) with improved precision for small x, SIMD optimized
pub fn log1p(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_log1p_f64(a);
    }
    a.map(|x| x.ln_1p())
}

/// Cube root with SIMD optimization for large arrays
pub fn cbrt(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_cbrt_f64(a);
    }
    a.map(|x| x.cbrt())
}

/// Reciprocal (1/x) with SIMD optimization for large arrays
pub fn reciprocal(a: &Array<f64>) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_reciprocal_f64(a);
    }
    a.map(|x| 1.0 / x)
}

/// Clamp values between min and max with SIMD optimization
pub fn clip(a: &Array<f64>, min: f64, max: f64) -> Array<f64> {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_clamp_f64(a, min, max);
    }
    a.map(|x| x.clamp(min, max))
}

/// Two-argument arctangent (atan2) with SIMD optimization
pub fn arctan2(y: &Array<f64>, x: &Array<f64>) -> Result<Array<f64>> {
    if y.shape() != x.shape() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if y.len() >= SIMD_THRESHOLD {
        return Ok(NeonEnhancedOps::vectorized_atan2_f64(y, x));
    }
    let y_data = y.to_vec();
    let x_data = x.to_vec();
    let result: Vec<f64> = y_data
        .iter()
        .zip(x_data.iter())
        .map(|(yi, xi)| yi.atan2(*xi))
        .collect();
    Ok(Array::from_vec(result).reshape(&y.shape()))
}

/// Hypotenuse calculation with SIMD optimization
pub fn hypot(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return Ok(NeonEnhancedOps::vectorized_hypot_f64(a, b));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai.hypot(*bi))
        .collect();
    Ok(Array::from_vec(result).reshape(&a.shape()))
}

/// Copy sign from second array to first with SIMD optimization
pub fn copysign(mag: &Array<f64>, sign: &Array<f64>) -> Result<Array<f64>> {
    if mag.shape() != sign.shape() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: mag.shape(),
            actual: sign.shape(),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if mag.len() >= SIMD_THRESHOLD {
        return Ok(NeonEnhancedOps::vectorized_copysign_f64(mag, sign));
    }
    let mag_data = mag.to_vec();
    let sign_data = sign.to_vec();
    let result: Vec<f64> = mag_data
        .iter()
        .zip(sign_data.iter())
        .map(|(m, s)| m.copysign(*s))
        .collect();
    Ok(Array::from_vec(result).reshape(&mag.shape()))
}

/// Fused multiply-add (a * b + c) with SIMD optimization
pub fn fma(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: if a.shape() != b.shape() {
                b.shape()
            } else {
                c.shape()
            },
        });
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return Ok(NeonEnhancedOps::vectorized_fma_f64(a, b, c));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let c_data = c.to_vec();
    let result: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .zip(c_data.iter())
        .map(|((ai, bi), ci)| ai.mul_add(*bi, *ci))
        .collect();
    Ok(Array::from_vec(result).reshape(&a.shape()))
}

/// Dot product with SIMD optimization
pub fn dot(a: &Array<f64>, b: &Array<f64>) -> Result<f64> {
    if a.len() != b.len() {
        return Err(crate::error::NumRs2Error::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return Ok(NeonEnhancedOps::vectorized_dot_f64(a, b));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result: f64 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai * bi)
        .sum();
    Ok(result)
}

/// L2 norm (Euclidean norm) with SIMD optimization
pub fn norm_l2(a: &Array<f64>) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_norm_l2_f64(a);
    }
    let data = a.to_vec();
    data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L1 norm (Manhattan norm) with SIMD optimization
pub fn norm_l1(a: &Array<f64>) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_norm_l1_f64(a);
    }
    let data = a.to_vec();
    data.iter().map(|x| x.abs()).sum::<f64>()
}

/// Variance with SIMD optimization
pub fn var(a: &Array<f64>) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_variance_f64(a);
    }
    let data = a.to_vec();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Standard deviation with SIMD optimization
pub fn std(a: &Array<f64>) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD {
        return NeonEnhancedOps::vectorized_std_f64(a);
    }
    var(a).sqrt()
}

//! Universal Functions (ufuncs) with SIMD-accelerated transcendental operations
//!
//! This module provides element-wise mathematical operations using scirs2-core's
//! SimdUnifiedOps trait for automatic platform-specific SIMD optimization.
//!
//! # Performance
//!
//! Operations automatically use the best available SIMD instruction set:
//! - AVX-512 (x86_64 with 512-bit vectors)
//! - AVX2 (x86_64 with 256-bit vectors)
//! - NEON (aarch64 with 128-bit vectors)
//! - Scalar fallback for other platforms
//!
//! # SCIRS2 POLICY Compliance
//!
//! Per SCIRS2 POLICY, all SIMD operations are routed through scirs2-core's
//! SimdUnifiedOps trait rather than using direct platform intrinsics.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::{self, Debug};

/// Threshold for using SIMD-optimized implementations.
/// Arrays smaller than this use scalar operations.
///
/// **Performance Rationale:**
/// SIMD operations through scirs2-core require data conversion (to_array_view/from_array1)
/// which involves 2-3 allocations per operation. For small arrays (< 64 elements), this
/// allocation overhead exceeds the SIMD computational benefits.
///
/// **Threshold Analysis:**
/// - Arrays < 64: Scalar is faster (avoids ~150ns allocation overhead)
/// - Arrays >= 64: SIMD is 2-4x faster (allocation cost amortized over computation)
/// - Break-even point: ~64-128 elements depending on operation
///
/// **Tuning:** For workloads with mostly small arrays, increase to 128.
/// For workloads with mostly large arrays, decrease to 32 (if zero-copy views are implemented).
const SIMD_THRESHOLD: usize = 64; // v0.2.0: Increased from 8 to 64 for better performance

// =============================================================================
// CONVERSION HELPERS
// =============================================================================

/// Convert NumRS2 Array to ndarray ArrayView1
fn to_array_view<T: Clone>(arr: &Array<T>) -> Array1<T> {
    Array1::from_vec(arr.to_vec())
}

/// Convert ndarray Array1 to NumRS2 Array with shape preserved
fn from_array1<T: Clone + Debug + NumCast>(arr: Array1<T>, shape: &[usize]) -> Array<T> {
    let data: Vec<T> = arr.into_iter().collect();
    Array::from_vec(data).reshape(shape)
}

/// Check if array is large enough for SIMD optimization
fn should_use_simd(len: usize) -> bool {
    len >= SIMD_THRESHOLD
}

// =============================================================================
// UNIVERSAL FUNCTION STRUCTS
// =============================================================================

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

    /// Apply the function to an array
    pub fn call(&self, a: &Array<f64>) -> Array<f64> {
        a.map(|x| (self.func)(x))
    }
}

// =============================================================================
// HELPER UFUNCS (Internal)
// =============================================================================

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

// =============================================================================
// BASIC ARITHMETIC OPERATIONS
// =============================================================================

/// Element-wise addition using SIMD optimization
pub fn add(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_add(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_add_ufunc().call(a, b)
}

/// Element-wise subtraction using SIMD optimization
pub fn subtract(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_sub(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_subtract_ufunc().call(a, b)
}

/// Element-wise multiplication using SIMD optimization
pub fn multiply(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_mul(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_multiply_ufunc().call(a, b)
}

/// Element-wise division using SIMD optimization
pub fn divide(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_div(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_divide_ufunc().call(a, b)
}

/// Element-wise power using SIMD optimization
pub fn power(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_pow(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_power_ufunc().call(a, b)
}

/// Element-wise maximum using SIMD optimization
pub fn maximum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_max(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_maximum_ufunc().call(a, b)
}

/// Element-wise minimum using SIMD optimization
pub fn minimum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.len() == b.len() && should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_min(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    get_minimum_ufunc().call(a, b)
}

// =============================================================================
// SCALAR OPERATIONS
// =============================================================================

/// Scalar addition using SIMD optimization
pub fn add_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = Array1::from_elem(a.len(), b);
        let result = f64::simd_add(&a_nd.view(), &b_nd.view());
        return from_array1(result, &a.shape());
    }
    get_add_ufunc().call_scalar_right(a, b)
}

/// Scalar subtraction using SIMD optimization
pub fn subtract_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = Array1::from_elem(a.len(), b);
        let result = f64::simd_sub(&a_nd.view(), &b_nd.view());
        return from_array1(result, &a.shape());
    }
    get_subtract_ufunc().call_scalar_right(a, b)
}

/// Scalar multiplication using SIMD optimization
pub fn multiply_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_scalar_mul(&a_nd.view(), b);
        return from_array1(result, &a.shape());
    }
    get_multiply_ufunc().call_scalar_right(a, b)
}

/// Scalar division using SIMD optimization
pub fn divide_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_scalar_mul(&a_nd.view(), 1.0 / b);
        return from_array1(result, &a.shape());
    }
    get_divide_ufunc().call_scalar_right(a, b)
}

/// Scalar power using SIMD optimization
pub fn power_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_powf(&a_nd.view(), b);
        return from_array1(result, &a.shape());
    }
    get_power_ufunc().call_scalar_right(a, b)
}

// =============================================================================
// UNARY OPERATIONS
// =============================================================================

/// Element-wise negation using SIMD optimization
pub fn negative(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_scalar_mul(&a_nd.view(), -1.0);
        return from_array1(result, &a.shape());
    }
    a.map(|x| -x)
}

/// Absolute value using SIMD optimization
pub fn absolute(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_abs(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.abs())
}

/// Alias for absolute
pub fn abs(a: &Array<f64>) -> Array<f64> {
    absolute(a)
}

/// Square each element using SIMD optimization
pub fn square(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_square(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x * x)
}

/// Square root using SIMD optimization
pub fn sqrt(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sqrt(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.sqrt())
}

/// Cube root using SIMD optimization
pub fn cbrt(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_cbrt(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.cbrt())
}

/// Reciprocal (1/x) using SIMD optimization
pub fn reciprocal(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_recip(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| 1.0 / x)
}

/// Inverse square root (1/sqrt(x)) using SIMD optimization
pub fn rsqrt(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_rsqrt(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| 1.0 / x.sqrt())
}

// =============================================================================
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
// =============================================================================

/// Exponential function (e^x) using SIMD optimization
pub fn exp(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_exp(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.exp())
}

/// Base-2 exponential (2^x) using SIMD optimization
pub fn exp2(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_exp2(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.exp2())
}

/// exp(x) - 1 with improved precision for small x
pub fn expm1(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_exp_m1(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.exp_m1())
}

/// Natural logarithm (ln) using SIMD optimization
pub fn log(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_ln(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.ln())
}

/// Alias for natural logarithm
pub fn ln(a: &Array<f64>) -> Array<f64> {
    log(a)
}

/// Base-2 logarithm using SIMD optimization
pub fn log2(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_log2(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.log2())
}

/// Base-10 logarithm using SIMD optimization
pub fn log10(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_log10(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.log10())
}

/// log(1 + x) with improved precision for small x
pub fn log1p(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_ln_1p(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.ln_1p())
}

// =============================================================================
// TRIGONOMETRIC FUNCTIONS
// =============================================================================

/// Sine function using SIMD optimization
pub fn sin(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sin(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.sin())
}

/// Cosine function using SIMD optimization
pub fn cos(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_cos(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.cos())
}

/// Tangent function using SIMD optimization
pub fn tan(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_tan(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.tan())
}

/// Simultaneous sine and cosine (more efficient than separate calls)
pub fn sincos(a: &Array<f64>) -> (Array<f64>, Array<f64>) {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let (sin_result, cos_result) = f64::simd_sincos(&a_nd.view());
        return (
            from_array1(sin_result, &a.shape()),
            from_array1(cos_result, &a.shape()),
        );
    }
    (a.map(|x| x.sin()), a.map(|x| x.cos()))
}

/// Normalized sinc function: sinc(x) = sin(pi*x) / (pi*x)
pub fn sinc(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sinc(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    use std::f64::consts::PI;
    a.map(|x| {
        if x == 0.0 {
            1.0
        } else {
            let px = PI * x;
            px.sin() / px
        }
    })
}

// =============================================================================
// INVERSE TRIGONOMETRIC FUNCTIONS
// =============================================================================

/// Arcsine using SIMD optimization
pub fn arcsin(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_asin(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.asin())
}

/// Alias for arcsin
pub fn asin(a: &Array<f64>) -> Array<f64> {
    arcsin(a)
}

/// Arccosine using SIMD optimization
pub fn arccos(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_acos(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.acos())
}

/// Alias for arccos
pub fn acos(a: &Array<f64>) -> Array<f64> {
    arccos(a)
}

/// Arctangent using SIMD optimization
pub fn arctan(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_atan(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.atan())
}

/// Alias for arctan
pub fn atan(a: &Array<f64>) -> Array<f64> {
    arctan(a)
}

/// Two-argument arctangent using SIMD optimization
pub fn arctan2(y: &Array<f64>, x: &Array<f64>) -> Result<Array<f64>> {
    if y.shape() != x.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: x.shape(),
            actual: y.shape(),
        });
    }
    if should_use_simd(y.len()) {
        let y_nd = to_array_view(y);
        let x_nd = to_array_view(x);
        let result = f64::simd_atan2(&y_nd.view(), &x_nd.view());
        return Ok(from_array1(result, &y.shape()));
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

/// Alias for arctan2
pub fn atan2(y: &Array<f64>, x: &Array<f64>) -> Result<Array<f64>> {
    arctan2(y, x)
}

// =============================================================================
// HYPERBOLIC FUNCTIONS
// =============================================================================

/// Hyperbolic sine using SIMD optimization
pub fn sinh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sinh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.sinh())
}

/// Hyperbolic cosine using SIMD optimization
pub fn cosh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_cosh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.cosh())
}

/// Hyperbolic tangent using SIMD optimization
pub fn tanh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_tanh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.tanh())
}

// =============================================================================
// INVERSE HYPERBOLIC FUNCTIONS
// =============================================================================

/// Inverse hyperbolic sine using SIMD optimization
pub fn arcsinh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_asinh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.asinh())
}

/// Alias for arcsinh
pub fn asinh(a: &Array<f64>) -> Array<f64> {
    arcsinh(a)
}

/// Inverse hyperbolic cosine using SIMD optimization
pub fn arccosh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_acosh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.acosh())
}

/// Alias for arccosh
pub fn acosh(a: &Array<f64>) -> Array<f64> {
    arccosh(a)
}

/// Inverse hyperbolic tangent using SIMD optimization
pub fn arctanh(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_atanh(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.atanh())
}

/// Alias for arctanh
pub fn atanh(a: &Array<f64>) -> Array<f64> {
    arctanh(a)
}

// =============================================================================
// ROUNDING FUNCTIONS
// =============================================================================

/// Floor function using SIMD optimization
pub fn floor(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_floor(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.floor())
}

/// Ceiling function using SIMD optimization
pub fn ceil(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_ceil(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.ceil())
}

/// Round to nearest integer using SIMD optimization
pub fn round(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_round(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.round())
}

/// Truncate toward zero using SIMD optimization
pub fn trunc(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_trunc(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.trunc())
}

/// Fractional part using SIMD optimization
pub fn fract(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_fract(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.fract())
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Sign function using SIMD optimization
pub fn sign(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sign(&a_nd.view());
        return from_array1(result, &a.shape());
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

/// Clamp values between min and max using SIMD optimization
pub fn clip(a: &Array<f64>, min: f64, max: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_clip(&a_nd.view(), min, max);
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.clamp(min, max))
}

/// Copy sign from second array to first using SIMD optimization
pub fn copysign(mag: &Array<f64>, sign: &Array<f64>) -> Result<Array<f64>> {
    if mag.shape() != sign.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: mag.shape(),
            actual: sign.shape(),
        });
    }
    if should_use_simd(mag.len()) {
        let mag_nd = to_array_view(mag);
        let sign_nd = to_array_view(sign);
        let result = f64::simd_copysign(&mag_nd.view(), &sign_nd.view());
        return Ok(from_array1(result, &mag.shape()));
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

/// Convert degrees to radians using SIMD optimization
pub fn deg2rad(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_to_radians(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.to_radians())
}

/// Alias for deg2rad
pub fn radians(a: &Array<f64>) -> Array<f64> {
    deg2rad(a)
}

/// Convert radians to degrees using SIMD optimization
pub fn rad2deg(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_to_degrees(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.to_degrees())
}

/// Alias for rad2deg
pub fn degrees(a: &Array<f64>) -> Array<f64> {
    rad2deg(a)
}

// =============================================================================
// BINARY UTILITY FUNCTIONS
// =============================================================================

/// Hypotenuse calculation using SIMD optimization
pub fn hypot(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_hypot(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
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

/// Fused multiply-add (a * b + c) using SIMD optimization
pub fn fma(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: if a.shape() != b.shape() {
                b.shape()
            } else {
                c.shape()
            },
        });
    }
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let c_nd = to_array_view(c);
        let result = f64::simd_fma(&a_nd.view(), &b_nd.view(), &c_nd.view());
        return Ok(from_array1(result, &a.shape()));
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

/// Linear interpolation using SIMD optimization
pub fn lerp(a: &Array<f64>, b: &Array<f64>, t: f64) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_lerp(&a_nd.view(), &b_nd.view(), t);
        return Ok(from_array1(result, &a.shape()));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai + t * (bi - ai))
        .collect();
    Ok(Array::from_vec(result).reshape(&a.shape()))
}

/// Stable log(e^a + e^b) computation using SIMD optimization
pub fn logaddexp(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != b.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: b.shape(),
        });
    }
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        let result = f64::simd_logaddexp(&a_nd.view(), &b_nd.view());
        return Ok(from_array1(result, &a.shape()));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result: Vec<f64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| {
            let max = ai.max(*bi);
            max + ((-(*ai - max).abs()).exp() + (-(*bi - max).abs()).exp()).ln()
        })
        .collect();
    Ok(Array::from_vec(result).reshape(&a.shape()))
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

/// Dot product using SIMD optimization
pub fn dot(a: &Array<f64>, b: &Array<f64>) -> Result<f64> {
    if a.len() != b.len() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let b_nd = to_array_view(b);
        return Ok(f64::simd_dot(&a_nd.view(), &b_nd.view()));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai * bi)
        .sum())
}

/// L2 norm (Euclidean norm) using SIMD optimization
pub fn norm_l2(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_norm(&a_nd.view());
    }
    let data = a.to_vec();
    data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L1 norm (Manhattan norm) using SIMD optimization
pub fn norm_l1(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_norm_l1(&a_nd.view());
    }
    let data = a.to_vec();
    data.iter().map(|x| x.abs()).sum::<f64>()
}

/// Sum of all elements using SIMD optimization
pub fn sum(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_sum(&a_nd.view());
    }
    a.to_vec().iter().sum()
}

/// Mean of all elements using SIMD optimization
pub fn mean(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_mean(&a_nd.view());
    }
    let data = a.to_vec();
    data.iter().sum::<f64>() / data.len() as f64
}

/// Variance using SIMD optimization
pub fn var(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_variance(&a_nd.view());
    }
    let data = a.to_vec();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Standard deviation using SIMD optimization
pub fn std(a: &Array<f64>) -> f64 {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        return f64::simd_std(&a_nd.view());
    }
    var(a).sqrt()
}

// =============================================================================
// SPECIAL FUNCTIONS
// =============================================================================

/// Error function using SIMD optimization
pub fn erf(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_erf(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Scalar fallback using approximation
    a.map(|x| {
        let t = 1.0 / (1.0 + 0.5 * x.abs());
        let tau = t
            * (-x * x - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587
                                            + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();
        if x >= 0.0 {
            1.0 - tau
        } else {
            tau - 1.0
        }
    })
}

/// Complementary error function using SIMD optimization
pub fn erfc(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_erfc(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| 1.0 - erf(&Array::from_vec(vec![x])).to_vec()[0])
}

/// Gamma function using SIMD optimization
pub fn gamma(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_gamma(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Lanczos approximation for scalar fallback
    a.map(|x| {
        if x <= 0.0 && x.floor() == x {
            f64::INFINITY
        } else {
            let g = 7;
            let c = [
                0.99999999999980993,
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ];
            if x < 0.5 {
                std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_single(1.0 - x))
            } else {
                let x = x - 1.0;
                let mut y = c[0];
                for i in 1..=g + 1 {
                    y += c[i] / (x + i as f64);
                }
                let t = x + g as f64 + 0.5;
                (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * y
            }
        }
    })
}

fn gamma_single(x: f64) -> f64 {
    let g = 7;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut y = c[0];
    for i in 1..=g + 1 {
        y += c[i] / (x + i as f64);
    }
    let t = x + g as f64 + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * y
}

/// Log-gamma function using SIMD optimization
pub fn lgamma(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_ln_gamma(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| gamma_single(x).abs().ln())
}

/// Alias for lgamma
pub fn gammaln(a: &Array<f64>) -> Array<f64> {
    lgamma(a)
}

/// Digamma function (psi) using SIMD optimization
pub fn digamma(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_digamma(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Asymptotic expansion for scalar fallback
    a.map(|x| {
        if x <= 0.0 {
            f64::NAN
        } else if x < 6.0 {
            // Recurrence relation
            digamma_scalar(x + 1.0) - 1.0 / x
        } else {
            // Asymptotic expansion
            let inv_x = 1.0 / x;
            let inv_x2 = inv_x * inv_x;
            x.ln() - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 / 252.0))
        }
    })
}

fn digamma_scalar(x: f64) -> f64 {
    if x < 6.0 {
        digamma_scalar(x + 1.0) - 1.0 / x
    } else {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        x.ln() - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 / 252.0))
    }
}

// =============================================================================
// NEURAL NETWORK ACTIVATION FUNCTIONS
// =============================================================================

/// Sigmoid activation: 1 / (1 + exp(-x)) using SIMD optimization
pub fn sigmoid(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_sigmoid(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Logit function (inverse of sigmoid): log(p / (1-p)) using SIMD optimization
pub fn logit(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_logit(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| (x / (1.0 - x)).ln())
}

/// GELU activation: x * Phi(x) using SIMD optimization
pub fn gelu(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_gelu(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    a.map(|x| {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    })
}

/// Swish/SiLU activation: x * sigmoid(x) using SIMD optimization
pub fn swish(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_swish(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x / (1.0 + (-x).exp()))
}

/// Alias for swish
pub fn silu(a: &Array<f64>) -> Array<f64> {
    swish(a)
}

/// Softplus activation: log(1 + exp(x)) using SIMD optimization
pub fn softplus(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_softplus(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Numerically stable: x + log1p(exp(-x)) for x > 0, log1p(exp(x)) for x <= 0
    a.map(|x| {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0 + x.exp()).ln()
        }
    })
}

/// Mish activation: x * tanh(softplus(x)) using SIMD optimization
pub fn mish(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_mish(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| {
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        x * sp.tanh()
    })
}

/// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
pub fn elu(a: &Array<f64>, alpha: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_elu(&a_nd.view(), alpha);
        return from_array1(result, &a.shape());
    }
    a.map(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
}

/// SELU activation (self-normalizing) using SIMD optimization
pub fn selu(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_selu(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    let lambda = 1.0507009873554804934193349852946;
    let alpha = 1.6732632423543772848170429916717;
    a.map(|x| lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
}

/// Hard sigmoid activation (piecewise linear) using SIMD optimization
pub fn hardsigmoid(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_hardsigmoid(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| {
        if x <= -3.0 {
            0.0
        } else if x >= 3.0 {
            1.0
        } else {
            x / 6.0 + 0.5
        }
    })
}

/// Hard swish activation (MobileNetV3) using SIMD optimization
pub fn hardswish(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_hardswish(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| {
        if x <= -3.0 {
            0.0
        } else if x >= 3.0 {
            x
        } else {
            x * (x + 3.0) / 6.0
        }
    })
}

/// ReLU activation: max(0, x) using SIMD optimization
pub fn relu(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_relu(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    a.map(|x| x.max(0.0))
}

/// Leaky ReLU activation: x if x > 0, alpha * x otherwise using SIMD optimization
pub fn leaky_relu(a: &Array<f64>, alpha: f64) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_leaky_relu(&a_nd.view(), alpha);
        return from_array1(result, &a.shape());
    }
    a.map(|x| if x > 0.0 { x } else { alpha * x })
}

/// PReLU with learned alpha per element
pub fn prelu(a: &Array<f64>, alpha: &Array<f64>) -> Result<Array<f64>> {
    if a.shape() != alpha.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: a.shape(),
            actual: alpha.shape(),
        });
    }
    let a_data = a.to_vec();
    let alpha_data = alpha.to_vec();
    let result: Vec<f64> = a_data
        .iter()
        .zip(alpha_data.iter())
        .map(|(x, alpha)| if *x > 0.0 { *x } else { alpha * x })
        .collect();
    Ok(Array::from_vec(result).reshape(&a.shape()))
}

/// Log-softmax (numerically stable) using SIMD optimization
pub fn log_softmax(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_log_softmax(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    // Numerically stable: x - max(x) - log(sum(exp(x - max(x))))
    let data = a.to_vec();
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = data.iter().map(|x| x - max_val).collect();
    let log_sum_exp = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
    let result: Vec<f64> = shifted.iter().map(|x| x - log_sum_exp).collect();
    Array::from_vec(result).reshape(&a.shape())
}

/// Softmax (numerically stable) using SIMD optimization
pub fn softmax(a: &Array<f64>) -> Array<f64> {
    if should_use_simd(a.len()) {
        let a_nd = to_array_view(a);
        let result = f64::simd_softmax(&a_nd.view());
        return from_array1(result, &a.shape());
    }
    let data = a.to_vec();
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = data.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let result: Vec<f64> = exps.iter().map(|x| x / sum).collect();
    Array::from_vec(result).reshape(&a.shape())
}

// =============================================================================
// SMOOTHSTEP FUNCTIONS
// =============================================================================

/// Smoothstep interpolation (Hermite)
pub fn smoothstep(edge0: f64, edge1: f64, x: &Array<f64>) -> Array<f64> {
    if should_use_simd(x.len()) {
        let x_nd = to_array_view(x);
        let result = f64::simd_smoothstep(edge0, edge1, &x_nd.view());
        return from_array1(result, &x.shape());
    }
    x.map(|v| {
        let t = ((v - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    })
}

/// Smootherstep interpolation (Ken Perlin's improved version)
pub fn smootherstep(edge0: f64, edge1: f64, x: &Array<f64>) -> Array<f64> {
    if should_use_simd(x.len()) {
        let x_nd = to_array_view(x);
        let result = f64::simd_smootherstep(edge0, edge1, &x_nd.view());
        return from_array1(result, &x.shape());
    }
    x.map(|v| {
        let t = ((v - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    })
}

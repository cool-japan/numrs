use crate::array::Array;
use crate::error::Result;
use std::fmt;

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

// Convenience functions for the ufuncs
pub fn add(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_add_ufunc().call(a, b)
}

pub fn subtract(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_subtract_ufunc().call(a, b)
}

pub fn multiply(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_multiply_ufunc().call(a, b)
}

pub fn divide(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_divide_ufunc().call(a, b)
}

pub fn power(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_power_ufunc().call(a, b)
}

pub fn maximum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_maximum_ufunc().call(a, b)
}

pub fn minimum(a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
    get_minimum_ufunc().call(a, b)
}

pub fn add_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    get_add_ufunc().call_scalar_right(a, b)
}

pub fn subtract_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    get_subtract_ufunc().call_scalar_right(a, b)
}

pub fn multiply_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    get_multiply_ufunc().call_scalar_right(a, b)
}

pub fn divide_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    get_divide_ufunc().call_scalar_right(a, b)
}

pub fn power_scalar(a: &Array<f64>, b: f64) -> Array<f64> {
    get_power_ufunc().call_scalar_right(a, b)
}

pub fn negative(a: &Array<f64>) -> Array<f64> {
    get_negative_ufunc().call(a)
}

pub fn absolute(a: &Array<f64>) -> Array<f64> {
    get_absolute_ufunc().call(a)
}

pub fn square(a: &Array<f64>) -> Array<f64> {
    get_square_ufunc().call(a)
}

pub fn sqrt(a: &Array<f64>) -> Array<f64> {
    get_sqrt_ufunc().call(a)
}

pub fn exp(a: &Array<f64>) -> Array<f64> {
    get_exp_ufunc().call(a)
}

pub fn log(a: &Array<f64>) -> Array<f64> {
    get_log_ufunc().call(a)
}

pub fn sin(a: &Array<f64>) -> Array<f64> {
    get_sin_ufunc().call(a)
}

pub fn cos(a: &Array<f64>) -> Array<f64> {
    get_cos_ufunc().call(a)
}

pub fn tan(a: &Array<f64>) -> Array<f64> {
    get_tan_ufunc().call(a)
}

// We'll limit the implementation for simplicity
pub fn arcsin(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.asin())
}

pub fn arccos(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.acos())
}

pub fn arctan(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.atan())
}

pub fn sinh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.sinh())
}

pub fn cosh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.cosh())
}

pub fn tanh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.tanh())
}

pub fn arcsinh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.asinh())
}

pub fn arccosh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.acosh())
}

pub fn arctanh(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.atanh())
}

pub fn floor(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.floor())
}

pub fn ceil(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.ceil())
}

pub fn round(a: &Array<f64>) -> Array<f64> {
    a.map(|x| x.round())
}

pub fn sign(a: &Array<f64>) -> Array<f64> {
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

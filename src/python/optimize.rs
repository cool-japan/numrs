//! Optimization operations for Python bindings

use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyTuple};
use std::cell::RefCell;

/// Minimize a scalar function of one or more variables.
///
/// Bridges a Python callable to the Rust optimizer.
/// Supported methods: `"nelder-mead"` (default) and `"bfgs"` (uses numerical gradient).
/// Returns a Python dict with keys `x`, `fun`, `success`, `nit`, `nfev`, `message`.
///
/// PyO3 0.29 does not infer a default of `None` for a trailing `Option<T>`
/// parameter without an explicit `#[pyo3(signature = ...)]` (confirmed
/// empirically -- see `eye`'s doc comment in `src/python/array.rs`), so
/// this and `root_scalar` below both need one.
#[pyfunction]
#[pyo3(signature = (fun, x0, method=None, tol=None))]
fn minimize(
    py: Python<'_>,
    fun: Py<PyAny>,
    x0: &PyArray,
    method: Option<String>,
    tol: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let x0_vec = x0.tolist();
    if x0_vec.is_empty() {
        return Err(PyValueError::new_err("x0 must not be empty"));
    }

    let method_str = method.as_deref().unwrap_or("nelder-mead").to_lowercase();

    // Capture Python exceptions that occur inside the closure.
    let captured_err: RefCell<Option<PyErr>> = RefCell::new(None);

    let closure_result = match method_str.as_str() {
        "nelder-mead" | "nm" | "neldermead" => {
            // Build closure that calls the Python function
            let closure = |x: &[f64]| -> f64 {
                Python::attach(|inner_py| {
                    let py_list: Vec<f64> = x.to_vec();
                    let args = match PyTuple::new(inner_py, [py_list.into_pyobject(inner_py).ok()])
                    {
                        Ok(t) => t,
                        Err(e) => {
                            *captured_err.borrow_mut() = Some(e);
                            return f64::NAN;
                        }
                    };
                    match fun.call(inner_py, args, None) {
                        Ok(result) => match result.extract::<f64>(inner_py) {
                            Ok(v) => v,
                            Err(e) => {
                                *captured_err.borrow_mut() = Some(e);
                                f64::NAN
                            }
                        },
                        Err(e) => {
                            *captured_err.borrow_mut() = Some(e);
                            f64::NAN
                        }
                    }
                })
            };

            let mut config = crate::optimize::OptimizeConfig::<f64>::default();
            if let Some(t) = tol {
                config.ftol = t;
                config.xtol = t;
            }
            crate::optimize::nelder_mead(closure, &x0_vec, Some(config))
        }
        "bfgs" => {
            // BFGS with numerical gradient (central differences)
            let eps = 1e-7_f64;

            let closure = |x: &[f64]| -> f64 {
                Python::attach(|inner_py| {
                    let py_list: Vec<f64> = x.to_vec();
                    let args = match PyTuple::new(inner_py, [py_list.into_pyobject(inner_py).ok()])
                    {
                        Ok(t) => t,
                        Err(e) => {
                            *captured_err.borrow_mut() = Some(e);
                            return f64::NAN;
                        }
                    };
                    match fun.call(inner_py, args, None) {
                        Ok(result) => match result.extract::<f64>(inner_py) {
                            Ok(v) => v,
                            Err(e) => {
                                *captured_err.borrow_mut() = Some(e);
                                f64::NAN
                            }
                        },
                        Err(e) => {
                            *captured_err.borrow_mut() = Some(e);
                            f64::NAN
                        }
                    }
                })
            };

            let grad_closure = |x: &[f64]| -> Vec<f64> {
                Python::attach(|inner_py| {
                    let n = x.len();
                    let mut g = vec![0.0_f64; n];
                    for i in 0..n {
                        let mut xph = x.to_vec();
                        let mut xmh = x.to_vec();
                        xph[i] += eps;
                        xmh[i] -= eps;

                        let call_val = |xv: Vec<f64>| -> f64 {
                            let args =
                                match PyTuple::new(inner_py, [xv.into_pyobject(inner_py).ok()]) {
                                    Ok(t) => t,
                                    Err(e) => {
                                        *captured_err.borrow_mut() = Some(e);
                                        return f64::NAN;
                                    }
                                };
                            match fun.call(inner_py, args, None) {
                                Ok(result) => result.extract::<f64>(inner_py).unwrap_or(f64::NAN),
                                Err(e) => {
                                    *captured_err.borrow_mut() = Some(e);
                                    f64::NAN
                                }
                            }
                        };
                        g[i] = (call_val(xph) - call_val(xmh)) / (2.0 * eps);
                    }
                    g
                })
            };

            let mut config = crate::optimize::OptimizeConfig::<f64>::default();
            if let Some(t) = tol {
                config.ftol = t;
                config.gtol = t;
            }
            crate::optimize::bfgs(closure, grad_closure, &x0_vec, Some(config))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown method '{}'. Supported: 'nelder-mead', 'bfgs'",
                other
            )));
        }
    };

    // Propagate any stored Python error from inside the closure
    if let Some(err) = captured_err.into_inner() {
        return Err(err);
    }

    let opt_result = closure_result.map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Return as Python dict mirroring scipy.optimize.OptimizeResult
    let dict = PyDict::new(py);
    dict.set_item("x", opt_result.x.into_pyobject(py)?.into_any().unbind())?;
    dict.set_item("fun", opt_result.fun)?;
    dict.set_item("success", opt_result.success)?;
    dict.set_item("nit", opt_result.nit)?;
    dict.set_item("nfev", opt_result.nfev)?;
    dict.set_item("message", opt_result.message)?;

    Ok(dict.into_any().unbind())
}

/// Find a root of a scalar function within a bracketing interval.
///
/// Bridges a Python callable to the Rust root-finder.
/// Supported methods: `"brentq"` (default), `"bisect"`.
/// Returns the root as f64.
#[pyfunction]
#[pyo3(signature = (fun, bracket, method=None))]
fn root_scalar(fun: Py<PyAny>, bracket: (f64, f64), method: Option<String>) -> PyResult<f64> {
    let (a, b) = bracket;

    let method_str = method.as_deref().unwrap_or("brentq").to_lowercase();

    let captured_err: RefCell<Option<PyErr>> = RefCell::new(None);

    let closure = |x: f64| -> f64 {
        Python::attach(|py| {
            let args = match PyTuple::new(py, [x]) {
                Ok(t) => t,
                Err(e) => {
                    *captured_err.borrow_mut() = Some(e);
                    return f64::NAN;
                }
            };
            match fun.call(py, args, None) {
                Ok(result) => match result.extract::<f64>(py) {
                    Ok(v) => v,
                    Err(e) => {
                        *captured_err.borrow_mut() = Some(e);
                        f64::NAN
                    }
                },
                Err(e) => {
                    *captured_err.borrow_mut() = Some(e);
                    f64::NAN
                }
            }
        })
    };

    let root_result = match method_str.as_str() {
        "brentq" | "brent" => crate::roots::brentq(closure, a, b, 1e-10, 1000)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        "bisect" | "bisection" => crate::roots::bisect(closure, a, b, 1e-10, 1000)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown method '{}'. Supported: 'brentq', 'bisect'",
                other
            )));
        }
    };

    // Propagate stored Python errors
    if let Some(err) = captured_err.into_inner() {
        return Err(err);
    }

    Ok(root_result.root)
}

/// Register optimization functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create optimize submodule
    let optimize_module = PyModule::new(m.py(), "optimize")?;

    // Add functions
    optimize_module.add_function(wrap_pyfunction!(minimize, m)?)?;
    optimize_module.add_function(wrap_pyfunction!(root_scalar, m)?)?;

    m.add_submodule(&optimize_module)?;

    Ok(())
}

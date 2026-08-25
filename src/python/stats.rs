//! Statistics operations for Python bindings

use crate::array::Array;
use crate::axis_ops::AxisOps;
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;

/// Compute the mean of array elements
///
/// When `axis` is None, returns a scalar f64.
/// When `axis` is Some(ax), reduces along that axis and returns a PyArray.
///
/// PyO3 0.29 does not infer a default of `None` for a trailing `Option<T>`
/// parameter without an explicit `#[pyo3(signature = ...)]` -- confirmed
/// empirically (see `eye`'s doc comment in `src/python/array.rs`), so
/// every `axis: Option<...>`-taking function in this file needed one added
/// for `nr.stats.mean(a)` (omitting `axis`) to actually work from Python.
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn mean(py: Python<'_>, a: &PyArray, axis: Option<usize>) -> PyResult<Py<PyAny>> {
    match axis {
        None => {
            let v = a.mean();
            Ok(v.into_pyobject(py)?.into_any().unbind())
        }
        Some(ax) => {
            let result = a
                .inner
                .mean_axis(Some(ax))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = PyArray { inner: result };
            py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
        }
    }
}

/// Compute the median of array elements
///
/// When `axis` is None, returns a scalar f64.
/// When `axis` is Some(ax), sorts each 1-D slice along that axis and returns the median
/// for each slice as a PyArray.
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn median(py: Python<'_>, a: &PyArray, axis: Option<usize>) -> PyResult<Py<PyAny>> {
    match axis {
        None => {
            let mut data = a.tolist();
            if data.is_empty() {
                return Err(PyValueError::new_err(
                    "Cannot compute median of empty array",
                ));
            }
            data.sort_by(|x: &f64, y: &f64| x.partial_cmp(y).unwrap_or(Ordering::Equal));
            let len = data.len();
            let v = if len.is_multiple_of(2) {
                (data[len / 2 - 1] + data[len / 2]) / 2.0
            } else {
                data[len / 2]
            };
            Ok(v.into_pyobject(py)?.into_any().unbind())
        }
        Some(ax) => {
            let shape = a.inner.shape();
            let ndim = a.inner.ndim();
            if ax >= ndim {
                return Err(PyValueError::new_err(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }
            let axis_len = shape[ax];
            if axis_len == 0 {
                return Err(PyValueError::new_err("Cannot compute median of empty axis"));
            }

            // Build output shape (input shape with axis `ax` removed)
            let mut out_shape = shape.clone();
            out_shape.remove(ax);
            let out_size: usize = out_shape.iter().product::<usize>().max(1);

            let data = a.inner.to_vec();
            let mut result = vec![0.0_f64; out_size];

            for out_i in 0..out_size {
                // Reconstruct multi-dim output index
                let mut out_idx = vec![0usize; out_shape.len()];
                let mut tmp = out_i;
                for d in (0..out_shape.len()).rev() {
                    out_idx[d] = tmp % out_shape[d];
                    tmp /= out_shape[d];
                }

                // Collect values along `ax`
                let mut slice = Vec::with_capacity(axis_len);
                for j in 0..axis_len {
                    // Build full index
                    let mut full_idx = out_idx.clone();
                    full_idx.insert(ax, j);

                    // Compute linear index
                    let mut linear = 0usize;
                    let mut stride = 1usize;
                    for d in (0..shape.len()).rev() {
                        linear += full_idx[d] * stride;
                        stride *= shape[d];
                    }
                    slice.push(data[linear]);
                }

                slice.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
                let n = slice.len();
                result[out_i] = if n.is_multiple_of(2) {
                    (slice[n / 2 - 1] + slice[n / 2]) / 2.0
                } else {
                    slice[n / 2]
                };
            }

            let arr = Array::from_vec_shape(result, &out_shape)?;
            let py_arr = PyArray { inner: arr };
            py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
        }
    }
}

/// Compute the standard deviation
///
/// When `axis` is None, returns a scalar f64.
/// When `axis` is Some(ax), reduces along that axis returning a PyArray.
/// `ddof` controls the degrees-of-freedom correction (default 0).
#[pyfunction]
#[pyo3(name = "std")]
#[pyo3(signature = (a, axis=None, ddof=None))]
fn stddev(
    py: Python<'_>,
    a: &PyArray,
    axis: Option<usize>,
    ddof: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let ddof_val = ddof.unwrap_or(0);
    match axis {
        None => {
            let data = a.tolist();
            let n = data.len();
            if n <= ddof_val {
                return Err(PyValueError::new_err("Sample size is too small"));
            }
            let mean = data.iter().sum::<f64>() / n as f64;
            let variance =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof_val) as f64;
            Ok(variance.sqrt().into_pyobject(py)?.into_any().unbind())
        }
        Some(ax) => {
            // Use var_axis (population variance = ddof 0) then scale for ddof
            let var_arr = a
                .inner
                .var_axis(Some(ax))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            // var_axis uses ddof=0 (divides by n). Scale to requested ddof.
            let axis_n = a.inner.shape()[ax];
            if axis_n <= ddof_val {
                return Err(PyValueError::new_err("Sample size is too small"));
            }
            let scale = axis_n as f64 / (axis_n - ddof_val) as f64;
            let scaled = var_arr.map(|v| (v * scale).sqrt());
            let py_arr = PyArray { inner: scaled };
            py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
        }
    }
}

/// Compute the variance
///
/// When `axis` is None, returns a scalar f64.
/// When `axis` is Some(ax), reduces along that axis returning a PyArray.
/// `ddof` controls the degrees-of-freedom correction (default 0).
#[pyfunction]
#[pyo3(signature = (a, axis=None, ddof=None))]
fn var(
    py: Python<'_>,
    a: &PyArray,
    axis: Option<usize>,
    ddof: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let ddof_val = ddof.unwrap_or(0);
    match axis {
        None => {
            let data = a.tolist();
            let n = data.len();
            if n <= ddof_val {
                return Err(PyValueError::new_err("Sample size is too small"));
            }
            let mean = data.iter().sum::<f64>() / n as f64;
            let variance =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof_val) as f64;
            Ok(variance.into_pyobject(py)?.into_any().unbind())
        }
        Some(ax) => {
            // var_axis uses ddof=0; scale to requested ddof
            let var_arr = a
                .inner
                .var_axis(Some(ax))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let axis_n = a.inner.shape()[ax];
            if axis_n <= ddof_val {
                return Err(PyValueError::new_err("Sample size is too small"));
            }
            let scale = axis_n as f64 / (axis_n - ddof_val) as f64;
            let scaled = var_arr.map(|v| v * scale);
            let py_arr = PyArray { inner: scaled };
            py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
        }
    }
}

/// Compute the Pearson correlation coefficient matrix.
///
/// Delegates to `crate::stats::correlation::corrcoef`, which implements
/// NumPy's full `corrcoef(x, y=None, rowvar=True)` semantics: `x`/`y` may
/// each be 1-D (a single variable) or 2-D (`rowvar` variables x
/// observations by default, or the transpose when `rowvar=False`); `y`, if
/// given, is treated as an additional variable appended to `x` (this
/// replaces the old restricted implementation, which only accepted two
/// 1-D arrays and rejected a single-array `x` outright).
///
/// Mirrors NumPy's own return-type quirk: a bare 1-D `x` with no `y` (one
/// variable "correlated with itself") collapses to a scalar `float`, e.g.
/// `np.corrcoef([1, 2, 3])` returns `np.float64(1.0)` rather than
/// `array([[1.]])`. Every other input shape returns the full correlation
/// matrix as an `Array`, matching `mean`/`median`/`std`/`var` above in
/// using the caller's argument shape to pick a scalar vs. `Array` result.
///
/// Note: unlike NumPy, a constant (zero-variance) input's diagonal entry is
/// `1.0` here rather than `NaN` -- this is the underlying
/// `crate::stats::correlation::cov`/`corrcoef` core's own convention
/// (avoiding `0/0` propagation), applied uniformly rather than special
/// enough to redo the shared core's math computed in the low-level call.
#[pyfunction]
#[pyo3(signature = (x, y=None, rowvar=None))]
fn corrcoef(
    py: Python<'_>,
    x: &PyArray,
    y: Option<&PyArray>,
    rowvar: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let y_inner = y.map(|a| &a.inner);
    let result = crate::stats::correlation::corrcoef::<f64>(&x.inner, y_inner, rowvar)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if y.is_none() && x.inner.ndim() == 1 {
        // NumPy's scalar-collapse case: see doc comment above.
        let v = *result
            .to_vec()
            .first()
            .ok_or_else(|| PyValueError::new_err("corrcoef produced an empty result"))?;
        return Ok(v.into_pyobject(py)?.into_any().unbind());
    }

    let py_arr = PyArray { inner: result };
    py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
}

/// Compute covariance matrix
///
/// Delegates to `crate::stats::correlation::cov` which is fully implemented.
/// `rowvar=true` (default): each row represents a variable.
#[pyfunction]
#[pyo3(signature = (m, rowvar=None))]
fn cov(m: &PyArray, rowvar: Option<bool>) -> PyResult<PyArray> {
    let cov_matrix = crate::stats::correlation::cov::<f64>(&m.inner, None, rowvar, None, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray { inner: cov_matrix })
}

/// Compute histogram
#[pyfunction]
#[pyo3(signature = (a, bins=None, range=None))]
fn histogram(
    a: &PyArray,
    bins: Option<usize>,
    range: Option<(f64, f64)>,
) -> PyResult<(PyArray, PyArray)> {
    let bins = bins.unwrap_or(10);
    let data = a.tolist();

    if data.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot compute histogram of empty array",
        ));
    }

    let (min_val, max_val) = if let Some((min, max)) = range {
        (min, max)
    } else {
        let min = data
            .iter()
            .copied()
            .min_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Cannot find minimum"))?;
        let max = data
            .iter()
            .copied()
            .max_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Cannot find maximum"))?;
        (min, max)
    };

    let bin_width = (max_val - min_val) / bins as f64;
    let mut counts = vec![0usize; bins];

    for &value in &data {
        if value >= min_val && value <= max_val {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1;
        }
    }

    let edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bin_width).collect();

    let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();

    Ok((
        PyArray {
            inner: crate::array::Array::from_vec(counts_f64),
        },
        PyArray {
            inner: crate::array::Array::from_vec(edges),
        },
    ))
}

/// Compute percentile
#[pyfunction]
fn percentile(a: &PyArray, q: f64) -> PyResult<f64> {
    if !(0.0..=100.0).contains(&q) {
        return Err(PyValueError::new_err(
            "Percentile must be between 0 and 100",
        ));
    }

    let mut data = a.tolist();
    if data.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot compute percentile of empty array",
        ));
    }

    data.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let index = (q / 100.0) * (data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    Ok(data[lower] + fraction * (data[upper] - data[lower]))
}

/// Register statistics functions
///
/// Random-number generation (`randn`/`rand` and the fuller `Generator` API)
/// used to live here as a token two-function `random` submodule; it has
/// moved to `crate::python::random`, which registers the real `random`
/// submodule itself (see `crate::python::mod::register`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create stats submodule
    let stats_module = PyModule::new(m.py(), "stats")?;

    // Add functions
    stats_module.add_function(wrap_pyfunction!(mean, m)?)?;
    stats_module.add_function(wrap_pyfunction!(median, m)?)?;
    stats_module.add_function(wrap_pyfunction!(stddev, m)?)?;
    stats_module.add_function(wrap_pyfunction!(var, m)?)?;
    stats_module.add_function(wrap_pyfunction!(corrcoef, m)?)?;
    stats_module.add_function(wrap_pyfunction!(cov, m)?)?;
    stats_module.add_function(wrap_pyfunction!(histogram, m)?)?;
    stats_module.add_function(wrap_pyfunction!(percentile, m)?)?;

    m.add_submodule(&stats_module)?;

    Ok(())
}

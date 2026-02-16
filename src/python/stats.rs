//! Statistics operations for Python bindings

use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;

/// Compute the mean of array elements
#[pyfunction]
fn mean(a: &PyArray, axis: Option<usize>) -> PyResult<f64> {
    if axis.is_some() {
        return Err(PyValueError::new_err("axis parameter not yet supported"));
    }
    Ok(a.mean())
}

/// Compute the median of array elements
#[pyfunction]
fn median(a: &PyArray, axis: Option<usize>) -> PyResult<f64> {
    if axis.is_some() {
        return Err(PyValueError::new_err("axis parameter not yet supported"));
    }

    let mut data = a.tolist();
    if data.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot compute median of empty array",
        ));
    }

    data.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let len = data.len();

    if len.is_multiple_of(2) {
        Ok((data[len / 2 - 1] + data[len / 2]) / 2.0)
    } else {
        Ok(data[len / 2])
    }
}

/// Compute the standard deviation
#[pyfunction]
#[pyo3(name = "std")]
fn stddev(a: &PyArray, axis: Option<usize>, ddof: Option<usize>) -> PyResult<f64> {
    if axis.is_some() {
        return Err(PyValueError::new_err("axis parameter not yet supported"));
    }

    let ddof = ddof.unwrap_or(0);
    let data = a.tolist();
    let n = data.len();

    if n <= ddof {
        return Err(PyValueError::new_err("Sample size is too small"));
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;

    Ok(variance.sqrt())
}

/// Compute the variance
#[pyfunction]
fn var(a: &PyArray, axis: Option<usize>, ddof: Option<usize>) -> PyResult<f64> {
    if axis.is_some() {
        return Err(PyValueError::new_err("axis parameter not yet supported"));
    }

    let ddof = ddof.unwrap_or(0);
    let data = a.tolist();
    let n = data.len();

    if n <= ddof {
        return Err(PyValueError::new_err("Sample size is too small"));
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;

    Ok(variance)
}

/// Compute the correlation coefficient
#[pyfunction]
fn corrcoef(x: &PyArray, y: Option<&PyArray>) -> PyResult<PyArray> {
    if let Some(y_arr) = y {
        let x_data = x.inner.to_vec();
        let y_data = y_arr.inner.to_vec();

        if x_data.len() != y_data.len() {
            return Err(PyValueError::new_err("Arrays must have the same length"));
        }

        if x_data.is_empty() {
            return Err(PyValueError::new_err("Arrays must not be empty"));
        }

        // Compute means
        let mean_x = x_data.iter().sum::<f64>() / x_data.len() as f64;
        let mean_y = y_data.iter().sum::<f64>() / y_data.len() as f64;

        // Compute covariance and standard deviations
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x_data.len() {
            let dx = x_data[i] - mean_x;
            let dy = y_data[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x == 0.0 || std_y == 0.0 {
            return Err(PyValueError::new_err("Standard deviation is zero"));
        }

        let corr = cov / (std_x * std_y);

        // Return 2x2 correlation matrix
        let data = vec![1.0, corr, corr, 1.0];
        Ok(PyArray {
            inner: crate::array::Array::from_vec(data).reshape(&[2, 2]),
        })
    } else {
        Err(PyValueError::new_err(
            "Single array correlation not yet supported",
        ))
    }
}

/// Compute covariance matrix
#[pyfunction]
fn cov(m: &PyArray, rowvar: Option<bool>) -> PyResult<PyArray> {
    let _rowvar = rowvar.unwrap_or(true);

    Err(PyValueError::new_err(
        "Covariance matrix calculation not yet implemented",
    ))
}

/// Compute histogram
#[pyfunction]
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

/// Generate random samples from normal distribution
#[pyfunction]
fn randn(size: Vec<usize>) -> PyResult<PyArray> {
    use scirs2_core::random::*;

    let total_size: usize = size.iter().product();
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, 1.0).map_err(|e| {
        PyValueError::new_err(format!("Failed to create normal distribution: {}", e))
    })?;

    let data: Vec<f64> = (0..total_size).map(|_| dist.sample(&mut rng)).collect();

    Ok(PyArray {
        inner: crate::array::Array::from_vec(data).reshape(&size),
    })
}

/// Generate random samples from uniform distribution
#[pyfunction]
fn rand(size: Vec<usize>) -> PyResult<PyArray> {
    use scirs2_core::random::*;

    let total_size: usize = size.iter().product();
    let mut rng = thread_rng();

    let data: Vec<f64> = (0..total_size).map(|_| rng.gen::<f64>()).collect();

    Ok(PyArray {
        inner: crate::array::Array::from_vec(data).reshape(&size),
    })
}

/// Register statistics functions
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

    // Add random module
    let random_module = PyModule::new(m.py(), "random")?;
    random_module.add_function(wrap_pyfunction!(randn, m)?)?;
    random_module.add_function(wrap_pyfunction!(rand, m)?)?;

    m.add_submodule(&random_module)?;

    Ok(())
}

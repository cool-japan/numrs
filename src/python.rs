//! Python bindings for NumRS2 via PyO3
//!
//! This module provides Python bindings for NumRS2's core functionality,
//! enabling seamless integration with Python and NumPy.

use crate::array::Array;
use crate::math;
use crate::NumRs2Error;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Convert NumRS2 error to Python exception
impl From<NumRs2Error> for PyErr {
    fn from(err: NumRs2Error) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}

/// Python wrapper for NumRS2 Array
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    inner: Array<f64>,
}

#[pymethods]
impl PyArray {
    /// Create a new array from a Python sequence
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try to extract as NumPy array first
        if let Ok(np_arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
            let shape = np_arr.shape().to_vec();
            let data_vec: Vec<f64> = np_arr
                .as_slice()
                .map_err(|_| {
                    PyValueError::new_err("Cannot convert NumPy array to contiguous slice")
                })?
                .to_vec();
            let array = Array::from_vec(data_vec).reshape(&shape);
            return Ok(PyArray { inner: array });
        }

        // Try as list
        if let Ok(list) = data.downcast::<PyList>() {
            let vec: Vec<f64> = list.extract()?;
            return Ok(PyArray {
                inner: Array::from_vec(vec),
            });
        }

        // Try as tuple
        if let Ok(tuple) = data.downcast::<PyTuple>() {
            let vec: Vec<f64> = tuple.extract()?;
            return Ok(PyArray {
                inner: Array::from_vec(vec),
            });
        }

        Err(PyTypeError::new_err("Expected list, tuple, or NumPy array"))
    }

    /// Get the shape of the array
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Get the number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get the total number of elements
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Reshape the array to a new shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyArray {
            inner: self.inner.clone().reshape(&shape),
        })
    }

    /// Transpose the array
    fn transpose(&self) -> PyResult<Self> {
        Ok(PyArray {
            inner: self.inner.transpose(),
        })
    }

    /// Convert to a flat Python list
    fn tolist(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    /// Convert to NumPy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        let vec = self.inner.to_vec();
        let shape: Vec<usize> = self.inner.shape();

        // Convert to NumPy array with proper shape
        vec.to_pyarray(py).reshape(shape).unwrap()
    }

    /// Element-wise addition
    fn __add__(&self, other: &PyArray) -> PyResult<Self> {
        Ok(PyArray {
            inner: &self.inner + &other.inner,
        })
    }

    /// Element-wise subtraction
    fn __sub__(&self, other: &PyArray) -> PyResult<Self> {
        Ok(PyArray {
            inner: &self.inner - &other.inner,
        })
    }

    /// Element-wise multiplication
    fn __mul__(&self, other: &PyArray) -> PyResult<Self> {
        Ok(PyArray {
            inner: &self.inner * &other.inner,
        })
    }

    /// Element-wise division
    fn __truediv__(&self, other: &PyArray) -> PyResult<Self> {
        Ok(PyArray {
            inner: &self.inner / &other.inner,
        })
    }

    /// Negation
    fn __neg__(&self) -> PyResult<Self> {
        Ok(PyArray {
            inner: -self.inner.clone(),
        })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, size={})", self.shape(), self.size())
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Create an array from a Python sequence
#[pyfunction]
fn array(data: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    PyArray::new(data)
}

/// Create an array of zeros
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyArray {
    PyArray {
        inner: Array::zeros(&shape),
    }
}

/// Create an array of ones
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyArray {
    PyArray {
        inner: Array::ones(&shape),
    }
}

/// Create an identity matrix
#[pyfunction]
fn eye(n: usize) -> PyArray {
    PyArray {
        inner: Array::eye(n, n, 0),
    }
}

/// Create an array with evenly spaced values
#[pyfunction]
fn linspace(start: f64, stop: f64, num: usize) -> PyArray {
    PyArray {
        inner: math::linspace(start, stop, num),
    }
}

/// Create an array with values in a range
#[pyfunction]
fn arange(start: f64, stop: f64, step: Option<f64>) -> PyArray {
    let step = step.unwrap_or(1.0);
    PyArray {
        inner: math::arange(start, stop, step),
    }
}

/// Matrix multiplication
#[pyfunction]
fn matmul(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = a.inner.matmul(&b.inner)?;
    Ok(PyArray { inner: result })
}

/// Compute the dot product
#[pyfunction]
fn dot(a: &PyArray, b: &PyArray) -> PyResult<f64> {
    let result = a.inner.dot(&b.inner)?;
    Ok(result)
}

/// NumRS2 Python module
#[pymodule]
fn _numrs2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add classes
    m.add_class::<PyArray>()?;

    // Add creation functions
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;

    // Add operations
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_module_creation() {
        // This test just ensures the module compiles
        // Actual Python tests are in tests/python/ directory
    }
}

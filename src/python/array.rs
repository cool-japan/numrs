//! Array operations for Python bindings
//!
//! Provides comprehensive array creation, manipulation, and operations.

use crate::array::Array;
use crate::math;
use crate::NumRs2Error;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use scirs2_numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};

/// Python wrapper for NumRS2 Array
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    pub(crate) inner: Array<f64>,
}

#[pymethods]
impl PyArray {
    /// Create a new array from a Python sequence
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try to extract as NumPy array first. `.as_array()` is a safe,
        // stride-aware view (unlike `.as_slice()`, which only succeeds for
        // a C-contiguous buffer and errors out otherwise), so this also
        // accepts non-contiguous NumPy input (e.g. a transposed or
        // strided view) instead of rejecting it; `.to_vec()` walks that
        // view in the array's logical (row-major) element order,
        // materializing exactly one copy -- the same single copy
        // `.as_slice()` + `.to_vec()` paid in the contiguous-only case,
        // just handling the general case too.
        if let Ok(np_arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
            let shape = np_arr.shape().to_vec();
            // `ArrayBase::to_vec()` only exists for 1-D arrays (see
            // ndarray's `impl_1d.rs`); `.iter().cloned().collect()` is the
            // ND-generic equivalent, walking the view in logical
            // (row-major) order regardless of its strides.
            let data_vec: Vec<f64> = np_arr.as_array().iter().cloned().collect();
            let array = Array::from_vec_shape(data_vec, &shape)?;
            return Ok(PyArray { inner: array });
        }

        // `Array` (this binding's `PyArray`) is f64-only storage -- there is
        // no zero-copy view onto a `float32` NumPy buffer to offer, so an
        // `np.float32` input still costs a copy same as the `f64` path
        // above, just with an added widening cast per element instead of a
        // plain clone. Without this branch, a `float32` array fails the
        // `PyReadonlyArrayDyn<f64>` extraction above (exact-dtype match,
        // no implicit cast) and then *also* fails the list/tuple checks
        // below (a NumPy array is neither), surfacing the misleading
        // "Expected list, tuple, or NumPy array" error for input that
        // plainly was a NumPy array.
        if let Ok(np_arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
            let shape = np_arr.shape().to_vec();
            let data_vec: Vec<f64> = np_arr.as_array().iter().map(|&v| v as f64).collect();
            let array = Array::from_vec_shape(data_vec, &shape)?;
            return Ok(PyArray { inner: array });
        }

        // Try as list
        if let Ok(list) = data.cast::<PyList>() {
            let vec: Vec<f64> = list.extract()?;
            return Ok(PyArray {
                inner: Array::from_vec(vec),
            });
        }

        // Try as tuple
        if let Ok(tuple) = data.cast::<PyTuple>() {
            let vec: Vec<f64> = tuple.extract()?;
            return Ok(PyArray {
                inner: Array::from_vec(vec),
            });
        }

        Err(PyTypeError::new_err(
            "Expected list, tuple, or NumPy array (float32 or float64)",
        ))
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

    /// Get the data type
    #[getter]
    fn dtype(&self) -> &str {
        "float64"
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

    /// Flatten the array to 1D
    fn flatten(&self) -> PyResult<Self> {
        let size = self.inner.size();
        Ok(PyArray {
            inner: self.inner.clone().reshape(&[size]),
        })
    }

    /// Squeeze - remove dimensions of size 1
    fn squeeze(&self) -> PyResult<Self> {
        let shape: Vec<usize> = self
            .inner
            .shape()
            .iter()
            .copied()
            .filter(|&s| s != 1)
            .collect();
        if shape.is_empty() {
            Ok(PyArray {
                inner: self.inner.clone().reshape(&[1]),
            })
        } else {
            Ok(PyArray {
                inner: self.inner.clone().reshape(&shape),
            })
        }
    }

    /// Convert to a flat Python list
    pub fn tolist(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    /// Convert to NumPy array.
    ///
    /// Always copies once into an independently-owned NumPy buffer:
    /// NumRS2's `Array` is `Arc`-based copy-on-write (`Clone` is O(1) and
    /// shares storage), so handing NumPy that same buffer without copying
    /// would let NumPy code mutate memory another `Array` handle still
    /// expects to be untouched -- unsound under this crate's COW
    /// invariant, so a copy is unavoidable here regardless of whether this
    /// particular handle happens to be uniquely owned. `from_array` builds
    /// the NumPy array directly from the source view in one pass (already
    /// shaped correctly, so unlike the previous implementation this has no
    /// separate fallible reshape step to `.expect()`), rather than via an
    /// intermediate flat `Vec` + a second reshape-copy.
    ///
    /// Contrast `crate::python::fft`, whose `fft`/`ifft`/`rfft`/`fftn`/
    /// `ifftn`/`rfftn` return a genuine zero-copy NumPy array: those
    /// results are freshly computed with no other referents, so handing
    /// their buffer to NumPy via `IntoPyArray` is safe.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        PyArrayDyn::from_array(py, self.inner.array())
    }

    /// Copy the array
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Sum of all elements
    fn sum(&self) -> f64 {
        self.inner.to_vec().iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        let vec = self.inner.to_vec();
        if vec.is_empty() {
            0.0
        } else {
            vec.iter().sum::<f64>() / vec.len() as f64
        }
    }

    /// Minimum value
    fn min(&self) -> PyResult<f64> {
        self.inner
            .to_vec()
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Array is empty"))
    }

    /// Maximum value
    fn max(&self) -> PyResult<f64> {
        self.inner
            .to_vec()
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Array is empty"))
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
        format!(
            "Array(shape={:?}, dtype={}, size={})",
            self.shape(),
            self.dtype(),
            self.size()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Length for len() function
    fn __len__(&self) -> usize {
        self.inner.shape().first().copied().unwrap_or(0)
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

/// Create a 2D array with ones on the diagonal and zeros elsewhere
///
/// PyO3 0.29 does not infer a default of `None` for a trailing `Option<T>`
/// parameter on its own; without `#[pyo3(signature = ...)]` every
/// parameter -- `Option<T>` or not -- is required, so `eye(3)` from Python
/// would raise `TypeError: eye() missing 2 required positional arguments`
/// (confirmed empirically: `tests/python_smoke.py`'s `nr.eye(2)` failed
/// exactly this way before this attribute was added).
#[pyfunction]
#[pyo3(signature = (n, m=None, k=None))]
fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> PyArray {
    let m = m.unwrap_or(n);
    let k = k.unwrap_or(0);
    PyArray {
        inner: Array::eye(n, m, k),
    }
}

/// Create an identity matrix
#[pyfunction]
fn identity(n: usize) -> PyArray {
    PyArray {
        inner: Array::eye(n, n, 0),
    }
}

/// Create an array with evenly spaced values
#[pyfunction]
#[pyo3(signature = (start, stop, num, endpoint=None))]
fn linspace(start: f64, stop: f64, num: usize, endpoint: Option<bool>) -> PyArray {
    let endpoint = endpoint.unwrap_or(true);
    if endpoint {
        PyArray {
            inner: math::linspace(start, stop, num),
        }
    } else {
        // Without endpoint, spacing is (stop - start) / num
        let step = (stop - start) / num as f64;
        let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
        PyArray {
            inner: Array::from_vec(values),
        }
    }
}

/// Create an array with values in a range
#[pyfunction]
#[pyo3(signature = (start, stop, step=None))]
fn arange(start: f64, stop: f64, step: Option<f64>) -> PyArray {
    let step = step.unwrap_or(1.0);
    PyArray {
        inner: math::arange(start, stop, step),
    }
}

/// Create an array filled with a constant value
#[pyfunction]
fn full(shape: Vec<usize>, fill_value: f64) -> PyResult<PyArray> {
    let size: usize = shape.iter().product();
    let data = vec![fill_value; size];
    Ok(PyArray {
        inner: Array::from_vec_shape(data, &shape)?,
    })
}

/// Create an array filled with zeros (like another array)
#[pyfunction]
fn zeros_like(a: &PyArray) -> PyArray {
    zeros(a.shape())
}

/// Create an array filled with ones (like another array)
#[pyfunction]
fn ones_like(a: &PyArray) -> PyArray {
    ones(a.shape())
}

/// Concatenate arrays along an axis (default 0), matching NumPy's
/// `concatenate` (join along the given axis, preserving every other
/// dimension) rather than flattening every input first -- flattening was
/// the previous "simplified implementation", which silently discarded
/// shape/axis and only happened to be correct for 1-D inputs concatenated
/// along axis 0.
#[pyfunction]
#[pyo3(signature = (arrays, axis=None))]
fn concatenate(arrays: Vec<PyArray>, axis: Option<usize>) -> PyResult<PyArray> {
    let axis = axis.unwrap_or(0);
    let refs: Vec<&Array<f64>> = arrays.iter().map(|a| &a.inner).collect();
    let result = crate::array_ops::joining::concatenate(&refs, axis)
        .map_err(|e| PyValueError::new_err(format!("concatenate failed: {e}")))?;
    Ok(PyArray { inner: result })
}

/// Register array module functions and classes
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add class
    m.add_class::<PyArray>()?;

    // Add creation functions
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;

    Ok(())
}

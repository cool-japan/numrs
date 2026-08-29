//! Random number generation for Python bindings
//!
//! Binds `scirs2_core::random` behind a NumPy-`Generator`-shaped API:
//! [`PyGenerator`] (exposed to Python as `Array`'s sibling class
//! `random.Generator`) plus a `default_rng` factory, alongside the
//! original two legacy convenience functions `rand`/`randn` (moved here
//! unchanged from `crate::python::stats`, which used to own a token
//! two-function `random` submodule before this fuller one existed).
//!
//! `scirs2_core::random` exports two same-shaped but distinct `Random<R>`
//! types (the modern `core::Random`, re-exported as `CoreRandom`, and a
//! "legacy" `Random<R>` defined directly in `scirs2_core::random`'s own
//! `mod.rs` for backward compatibility). This module always spells the
//! legacy one out as `scirs2_core::random::Random` and avoids
//! `scirs2_core::random::*` glob imports, to keep which one is in use
//! unambiguous.
//!
//! `PyGenerator` always stores a *seeded* `Random<StdRng>`, never
//! `Random<ThreadRng>`: `ThreadRng` is thread-local and `!Send`, and a
//! `#[pyclass]`'s contents must be `Send` (Python can move an object
//! between OS threads). An unseeded `Generator()` draws one `u64` from
//! `scirs2_core::random::thread_rng()` (the crate's own thread-local RNG)
//! and feeds it to `Random::seed`, so entropy still comes from a proper
//! thread-local source without ever storing a non-`Send` RNG.

use crate::array::Array;
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::random::rngs::{StdRng, ThreadRng};
use scirs2_core::random::{Normal, Random, Uniform};

/// Build a seeded generator, drawing entropy from the thread-local RNG when
/// `seed` is `None`. See the module doc comment for why this is always
/// `Random<StdRng>`, never `Random<ThreadRng>`.
fn generator_from_seed(seed: Option<u64>) -> Random<StdRng> {
    let s = seed.unwrap_or_else(|| scirs2_core::random::thread_rng().random::<u64>());
    Random::<ThreadRng>::seed(s)
}

/// NumPy-style random number generator (`numpy.random.Generator`).
///
/// Create one directly (`Generator(seed=42)`) or via `default_rng`; both
/// are equivalent. Every sampling method takes an optional `size`: `None`
/// (the default) returns a Python `float`, matching NumPy's own scalar
/// collapse, while a shape sequence returns an `Array` of that shape.
#[pyclass(name = "Generator")]
pub struct PyGenerator {
    rng: Random<StdRng>,
}

impl PyGenerator {
    /// Shared plumbing for every `size: Option<Vec<usize>> -> float | Array`
    /// sampling method below: draw one sample per element (or a single
    /// scalar when `size` is `None`) by repeatedly calling `f`.
    fn sample_scalar_or_array(
        &mut self,
        py: Python<'_>,
        size: Option<Vec<usize>>,
        mut f: impl FnMut(&mut Random<StdRng>) -> f64,
    ) -> PyResult<Py<PyAny>> {
        match size {
            None => Ok(f(&mut self.rng).into_pyobject(py)?.into_any().unbind()),
            Some(shape) => {
                let total: usize = shape.iter().product();
                let data: Vec<f64> = (0..total).map(|_| f(&mut self.rng)).collect();
                let arr = Array::from_vec_shape(data, &shape)?;
                let py_arr = PyArray { inner: arr };
                py_arr.into_pyobject(py).map(|b| b.into_any().unbind())
            }
        }
    }

    /// Shared plumbing for `permutation`/`shuffle`: a fresh `Array` with the
    /// same shape as `arr`, whose axis-0 "rows" (the whole array, for 1-D
    /// input) have been reordered by an independently-shuffled permutation.
    fn permuted_copy(&mut self, arr: &Array<f64>) -> PyResult<Array<f64>> {
        let shape = arr.shape();
        if shape.is_empty() {
            return Err(PyValueError::new_err(
                "permutation/shuffle require an array with at least 1 dimension",
            ));
        }
        let n = shape[0];
        let row_size: usize = shape[1..].iter().product::<usize>().max(1);
        let data = arr.to_vec();

        let mut order: Vec<usize> = (0..n).collect();
        self.rng.shuffle(&mut order);

        let mut result = Vec::with_capacity(data.len());
        for &i in &order {
            result.extend_from_slice(&data[i * row_size..(i + 1) * row_size]);
        }
        Ok(Array::from_vec_shape(result, &shape)?)
    }
}

#[pymethods]
impl PyGenerator {
    /// Create a new generator. With no seed, entropy comes from the
    /// process's thread-local RNG; a given `seed` makes the sequence of
    /// every subsequent draw fully deterministic and reproducible.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        PyGenerator {
            rng: generator_from_seed(seed),
        }
    }

    /// Re-seed this generator in place (subsequent draws restart from the
    /// deterministic sequence for `seed`).
    fn seed(&mut self, seed: u64) {
        self.rng = Random::<ThreadRng>::seed(seed);
    }

    /// Draw sample(s) from a uniform distribution over `[low, high)`.
    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(
        &mut self,
        py: Python<'_>,
        low: f64,
        high: f64,
        size: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        // Written via `partial_cmp` rather than `!(low < high)`: for `f64`
        // (only `PartialOrd`, not `Ord`) those aren't equivalent when NaN is
        // involved -- `low < high` and `low >= high` are BOTH `false` for
        // NaN, so negating the first to reject invalid ranges must not be
        // rewritten as the second, which would silently accept it instead.
        if !matches!(low.partial_cmp(&high), Some(std::cmp::Ordering::Less)) {
            return Err(PyValueError::new_err("uniform requires low < high"));
        }
        let dist = Uniform::new(low, high)
            .map_err(|e| PyValueError::new_err(format!("Invalid uniform range: {e}")))?;
        self.sample_scalar_or_array(py, size, |rng| rng.sample(dist))
    }

    /// Draw sample(s) from a normal (Gaussian) distribution.
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(
        &mut self,
        py: Python<'_>,
        loc: f64,
        scale: f64,
        size: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let dist = Normal::new(loc, scale)
            .map_err(|e| PyValueError::new_err(format!("Invalid normal parameters: {e}")))?;
        self.sample_scalar_or_array(py, size, |rng| rng.sample(dist))
    }

    /// Draw sample(s) from the standard normal distribution (mean 0, scale 1).
    #[pyo3(signature = (size=None))]
    fn standard_normal(&mut self, py: Python<'_>, size: Option<Vec<usize>>) -> PyResult<Py<PyAny>> {
        self.normal(py, 0.0, 1.0, size)
    }

    /// Draw sample(s) uniformly from `[0.0, 1.0)`.
    #[pyo3(signature = (size=None))]
    fn random(&mut self, py: Python<'_>, size: Option<Vec<usize>>) -> PyResult<Py<PyAny>> {
        self.sample_scalar_or_array(py, size, |rng| rng.random_f64())
    }

    /// Draw random integer sample(s) from `[low, high)`, or `[0, low)` when
    /// `high` is omitted (matching `numpy.random.Generator.integers`).
    /// Returned as `float`/`Array` of floats, like every other NumRS2
    /// numeric API in these bindings; values stay exact for any range that
    /// fits an `f64`'s 53-bit mantissa.
    #[pyo3(signature = (low, high=None, size=None))]
    fn integers(
        &mut self,
        py: Python<'_>,
        low: i64,
        high: Option<i64>,
        size: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let (lo, hi) = match high {
            Some(h) => (low, h),
            None => (0, low),
        };
        if lo >= hi {
            return Err(PyValueError::new_err(
                "integers requires low < high (or, with high omitted, low > 0)",
            ));
        }
        self.sample_scalar_or_array(py, size, |rng| rng.random_range(lo..hi) as f64)
    }

    /// Return a randomly permuted copy of `x`.
    ///
    /// `x` may be a non-negative `int` `n` (returns a shuffled
    /// `arange(n)`) or an `Array` (returns a shuffled copy along its first
    /// axis, leaving each "row" -- the whole element for a 1-D array --
    /// intact), matching `numpy.random.Generator.permutation`.
    fn permutation(&mut self, x: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        if let Ok(n) = x.extract::<usize>() {
            let mut idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
            self.rng.shuffle(&mut idx);
            return Ok(PyArray {
                inner: Array::from_vec(idx),
            });
        }
        if let Ok(arr) = x.extract::<PyArray>() {
            let inner = self.permuted_copy(&arr.inner)?;
            return Ok(PyArray { inner });
        }
        Err(PyValueError::new_err(
            "permutation expects a non-negative int or an Array",
        ))
    }

    /// Shuffle `x` in place along its first axis (returns `None`, like
    /// `numpy.random.Generator.shuffle`).
    fn shuffle(&mut self, x: &mut PyArray) -> PyResult<()> {
        x.inner = self.permuted_copy(&x.inner)?;
        Ok(())
    }
}

/// Create a new [`PyGenerator`] (NumPy's `numpy.random.default_rng` idiom).
#[pyfunction]
#[pyo3(signature = (seed=None))]
fn default_rng(seed: Option<u64>) -> PyGenerator {
    PyGenerator {
        rng: generator_from_seed(seed),
    }
}

/// Generate random samples from a standard normal distribution (legacy
/// `numpy.random.randn`-style convenience function; prefer
/// `default_rng().standard_normal(size)` in new code for a reproducible,
/// seedable generator).
#[pyfunction]
fn randn(size: Vec<usize>) -> PyResult<PyArray> {
    let dist = Normal::new(0.0, 1.0).map_err(|e| {
        PyValueError::new_err(format!("Failed to create normal distribution: {}", e))
    })?;
    let total_size: usize = size.iter().product();
    let mut rng = scirs2_core::random::thread_rng();
    let data: Vec<f64> = (0..total_size).map(|_| rng.sample(dist)).collect();
    Ok(PyArray {
        inner: Array::from_vec_shape(data, &size)?,
    })
}

/// Generate random samples from a uniform `[0, 1)` distribution (legacy
/// `numpy.random.rand`-style convenience function; prefer
/// `default_rng().random(size)` in new code for a reproducible, seedable
/// generator).
#[pyfunction]
fn rand(size: Vec<usize>) -> PyResult<PyArray> {
    let total_size: usize = size.iter().product();
    let mut rng = scirs2_core::random::thread_rng();
    let data: Vec<f64> = (0..total_size).map(|_| rng.random::<f64>()).collect();
    Ok(PyArray {
        inner: Array::from_vec_shape(data, &size)?,
    })
}

/// Register the `random` submodule: `Generator`, `default_rng`, and the
/// legacy `rand`/`randn` convenience functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let random_module = PyModule::new(m.py(), "random")?;

    random_module.add_class::<PyGenerator>()?;
    random_module.add_function(wrap_pyfunction!(default_rng, m)?)?;
    random_module.add_function(wrap_pyfunction!(randn, m)?)?;
    random_module.add_function(wrap_pyfunction!(rand, m)?)?;

    m.add_submodule(&random_module)?;

    Ok(())
}

//! FFT operations for Python bindings.
//!
//! Binds `crate::fft`'s NumPy-parameter-complete wrappers
//! (`fft_with`/`ifft_with`/`rfft_with`/`irfft_with` for the 1-D family,
//! `fftn`/`ifftn`/`rfftn`/`irfftn` for the N-D family) under their plain
//! NumPy names (`fft`, `ifft`, ...), rather than reimplementing axis/norm
//! handling here.
//!
//! ## Complex numbers: raw NumPy arrays, not `Array`
//!
//! [`PyArray`] (`nr.Array`) wraps `crate::array::Array<f64>` -- real only.
//! FFT output is fundamentally complex, so every function that produces a
//! spectrum (`fft`, `ifft`, `rfft`, `fftn`, `ifftn`, `rfftn`) returns a
//! genuine NumPy `complex128` array instead of an `Array`; symmetrically,
//! `ifft`/`irfft`/`ifftn`/`irfftn` (whose natural input *is* a spectrum)
//! accept a raw NumPy array rather than an `Array`. Only the two purely
//! real-valued inverse-real-FFT functions (`irfft`, `irfftn`) return an
//! `Array`. A round trip is therefore written as
//! `nr.fft.ifft(nr.fft.fft(arr))`, entirely in NumPy space for the
//! spectrum, matching how `numpy.fft` itself behaves (its arrays just
//! don't distinguish real from complex the way this crate's `Array` does).
//!
//! This is also this bindings crate's one genuinely zero-copy NumPy
//! transfer: the spectrum is a freshly built buffer with no other
//! referents (unlike `Array`, which is `Arc`-shared copy-on-write), so
//! handing it to NumPy via `IntoPyArray` moves its allocation directly
//! into the NumPy array with no copy at all. Contrast `Array.to_numpy`
//! (`src/python/array.rs`), which always copies because `Array` might be
//! shared.
//!
//! Verified against `scirs2-fft`'s own `convert_to_complex`/
//! `try_as_complex` helpers (which every function bound here funnels
//! through): passing an already-complex `Complex64` slice/array through
//! these generic, `NumCast`-bounded functions preserves the full complex
//! value (both real and imaginary parts) rather than truncating to the
//! real part, so `ifft`/`irfft`/`ifftn`/`irfftn` on a real `fft`/`rfft`
//! output round-trip correctly.
//!
//! ## Scope
//!
//! `fft`/`ifft`/`rfft`/`irfft` only accept 1-D input (matching the
//! underlying `crate::fft::*_with` functions, which operate on flat
//! slices); higher-rank input is rejected with a message pointing at
//! `fftn`/`ifftn`/`rfftn`/`irfftn`, which are N-D natively.

use crate::kernels::borrow::operand;
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::Complex64;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};

/// Borrow a NumPy array view as a flat slice at zero cost when it is
/// already contiguous (the common case: e.g. the freshly built output of
/// [`fft`]/[`rfft`] fed straight into [`ifft`]/[`irfft`]), falling back to
/// one copy otherwise. Mirrors `kernels::borrow::operand`'s
/// zero-copy-or-logical-order strategy for `crate::array::Array`, applied
/// here to a borrowed `PyReadonlyArray`'s view instead.
fn flat_or_owned<'a>(
    view: scirs2_core::ndarray::ArrayViewD<'a, Complex64>,
) -> std::borrow::Cow<'a, [Complex64]> {
    // `ArrayView::to_slice()` (unlike the generic `ArrayBase::as_slice()`)
    // transfers the view's own `'a` to the returned slice instead of tying
    // it to this `&view` borrow, which is what makes the `Cow::Borrowed`
    // arm below (a slice living past this function returning) sound.
    match view.to_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        // `ArrayBase::to_vec()` only exists for `Ix1` (see `impl_1d.rs`);
        // `.iter().cloned().collect()` is the ND-generic equivalent,
        // walking the view in logical (row-major) order regardless of its
        // strides -- the same fallback `kernels::borrow::operand` uses.
        None => std::borrow::Cow::Owned(view.iter().cloned().collect()),
    }
}

/// Reject anything but a 1-D `Array`, pointing at the N-D entry point.
fn require_1d(x: &crate::array::Array<f64>, nd_name: &str) -> PyResult<()> {
    if x.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "expected a 1-D array (got {}-D); use {} for multi-dimensional input",
            x.ndim(),
            nd_name
        )));
    }
    Ok(())
}

/// 1-D FFT: `nr.fft.fft(x, n=None, axis=None, norm=None)`.
///
/// See the module doc comment for why this returns a raw NumPy
/// `complex128` array (zero-copy) instead of an `Array`.
#[pyfunction]
#[pyo3(signature = (x, n=None, axis=None, norm=None))]
fn fft<'py>(
    py: Python<'py>,
    x: &PyArray,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_1d(&x.inner, "fftn")?;
    let op = operand(&x.inner);
    let result = crate::fft::fft_with(&op, n, axis, norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("fft failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// 1-D inverse FFT: `nr.fft.ifft(x, n=None, axis=None, norm=None)`.
///
/// `x` is a NumPy `complex128` array (typically the output of [`fft`] or
/// [`rfft`], possibly further processed); see the module doc comment.
#[pyfunction]
#[pyo3(signature = (x, n=None, axis=None, norm=None))]
fn ifft<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, Complex64>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let view = x.as_array();
    if view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "expected a 1-D array (got {}-D); use ifftn for multi-dimensional input",
            view.ndim()
        )));
    }
    let data = flat_or_owned(view);
    let result = crate::fft::ifft_with(&data, n, axis, norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("ifft failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// 1-D real-input FFT: `nr.fft.rfft(x, n=None, axis=None, norm=None)`.
///
/// Returns only the non-redundant half of the spectrum (length `n/2 + 1`),
/// as a raw NumPy `complex128` array.
#[pyfunction]
#[pyo3(signature = (x, n=None, axis=None, norm=None))]
fn rfft<'py>(
    py: Python<'py>,
    x: &PyArray,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_1d(&x.inner, "rfftn")?;
    let op = operand(&x.inner);
    let result = crate::fft::rfft_with(&op, n, axis, norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("rfft failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// 1-D inverse real FFT: `nr.fft.irfft(x, n=None, axis=None, norm=None)`.
///
/// `x` is a NumPy `complex128` array holding the non-redundant half of a
/// spectrum (typically [`rfft`]'s output); `n` is the length of the real
/// output signal (default: `2 * (len(x) - 1)`, NumPy's own convention).
/// Returns a real `Array`.
#[pyfunction]
#[pyo3(signature = (x, n=None, axis=None, norm=None))]
fn irfft(
    x: PyReadonlyArrayDyn<'_, Complex64>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<String>,
) -> PyResult<PyArray> {
    let view = x.as_array();
    if view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "expected a 1-D array (got {}-D); use irfftn for multi-dimensional input",
            view.ndim()
        )));
    }
    let data = flat_or_owned(view);
    let result = crate::fft::irfft_with(&data, n, axis, norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("irfft failed: {e}")))?;
    Ok(PyArray {
        inner: crate::array::Array::from_vec(result),
    })
}

/// N-dimensional FFT: `nr.fft.fftn(x, s=None, axes=None, norm=None)`.
///
/// `s` (per-axis output sizes) and `axes` (which axes to transform) follow
/// NumPy's exact conventions; both default to every axis of `x` at its own
/// size. Returns a raw NumPy `complex128` array (see the module doc
/// comment for why).
#[pyfunction]
#[pyo3(signature = (x, s=None, axes=None, norm=None))]
fn fftn<'py>(
    py: Python<'py>,
    x: &PyArray,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let result = crate::fft::fftn(
        x.inner.array(),
        s.as_deref(),
        axes.as_deref(),
        norm.as_deref(),
    )
    .map_err(|e| PyValueError::new_err(format!("fftn failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// N-dimensional inverse FFT: `nr.fft.ifftn(x, s=None, axes=None, norm=None)`.
///
/// `x` is a NumPy `complex128` array (typically [`fftn`]'s output); see the
/// module doc comment.
///
/// Unlike [`ifft`]/[`irfft`] (which borrow the input via [`flat_or_owned`]
/// at zero cost whenever it is contiguous), this always pays one copy on
/// the way in: `crate::fft::ifftn` takes an owned `ArrayD<Complex64>`, not
/// a borrowable view or slice, so there is no representation here for
/// `flat_or_owned`'s "borrow when contiguous" trick to produce -- an owned
/// `ArrayD` has to be materialized (`view.to_owned()`) regardless of the
/// input's own contiguity. See `crate::python::mod`'s copy-vs-view table.
#[pyfunction]
#[pyo3(signature = (x, s=None, axes=None, norm=None))]
fn ifftn<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, Complex64>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let view = x.as_array();
    let owned = view.to_owned();
    let result = crate::fft::ifftn(&owned, s.as_deref(), axes.as_deref(), norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("ifftn failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// N-dimensional real-input FFT: `nr.fft.rfftn(x, s=None, axes=None, norm=None)`.
///
/// Like [`rfft`], only the non-redundant half of the *last* transformed
/// axis is kept. Returns a raw NumPy `complex128` array.
#[pyfunction]
#[pyo3(signature = (x, s=None, axes=None, norm=None))]
fn rfftn<'py>(
    py: Python<'py>,
    x: &PyArray,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<String>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let result = crate::fft::rfftn(
        x.inner.array(),
        s.as_deref(),
        axes.as_deref(),
        norm.as_deref(),
    )
    .map_err(|e| PyValueError::new_err(format!("rfftn failed: {e}")))?;
    Ok(result.into_pyarray(py))
}

/// N-dimensional inverse real FFT: `nr.fft.irfftn(x, s=None, axes=None, norm=None)`.
///
/// `x` is a NumPy `complex128` array (typically [`rfftn`]'s output); `s`
/// gives the real output shape (default follows NumPy's `2 * (m - 1)` rule
/// on the last transformed axis). Returns a real `Array`.
///
/// Always copies its input on the way in, for the same reason as
/// [`ifftn`]: `crate::fft::irfftn` also takes an owned `ArrayD`.
#[pyfunction]
#[pyo3(signature = (x, s=None, axes=None, norm=None))]
fn irfftn(
    x: PyReadonlyArrayDyn<'_, Complex64>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<String>,
) -> PyResult<PyArray> {
    let view = x.as_array();
    let owned = view.to_owned();
    let result = crate::fft::irfftn(&owned, s.as_deref(), axes.as_deref(), norm.as_deref())
        .map_err(|e| PyValueError::new_err(format!("irfftn failed: {e}")))?;
    let shape = result.shape().to_vec();
    let (vec, _offset) = result.into_raw_vec_and_offset();
    Ok(PyArray {
        inner: crate::array::Array::from_vec_shape(vec, &shape)?,
    })
}

/// Register the `fft` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let fft_module = PyModule::new(m.py(), "fft")?;

    fft_module.add_function(wrap_pyfunction!(fft, m)?)?;
    fft_module.add_function(wrap_pyfunction!(ifft, m)?)?;
    fft_module.add_function(wrap_pyfunction!(rfft, m)?)?;
    fft_module.add_function(wrap_pyfunction!(irfft, m)?)?;
    fft_module.add_function(wrap_pyfunction!(fftn, m)?)?;
    fft_module.add_function(wrap_pyfunction!(ifftn, m)?)?;
    fft_module.add_function(wrap_pyfunction!(rfftn, m)?)?;
    fft_module.add_function(wrap_pyfunction!(irfftn, m)?)?;

    m.add_submodule(&fft_module)?;

    Ok(())
}

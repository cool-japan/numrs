//! Neural network operations for Python bindings

use crate::array::Array;
use crate::kernels::borrow::operand;
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};

/// Apply a 1-D NumRS2 core op (an activation function, dropout, ...) to an
/// array of any rank.
///
/// Every op this is used for (`relu`, `sigmoid`, `tanh`, `dropout`, ...) is
/// a pure element-wise `mapv`/per-element closure underneath -- see e.g.
/// `relu`/`relu_2d`'s byte-for-byte identical bodies in
/// `src/nn/activation.rs` -- so the 1-D core function already computes the
/// right answer for any shape once given a flat view of the data; the
/// `ArrayView1`-only signature is an API restriction, not a mathematical
/// one. This routes any-rank input through that same 1-D core function by
/// borrowing the array's data in logical order (zero-copy when contiguous,
/// via `kernels::borrow::operand`; a single materializing copy otherwise,
/// e.g. after `.transpose()`) and rebuilding an array with the original
/// shape from the (freshly allocated, so offset-0) result.
fn apply_elementwise_1d(
    x: &Array<f64>,
    op_name: &str,
    f: impl FnOnce(&ArrayView1<f64>) -> crate::nn::NnResult<Array1<f64>>,
) -> PyResult<Array<f64>> {
    let shape = x.shape();
    let op = operand(x);
    let view = ArrayView1::from(&*op);
    let result = f(&view).map_err(|e| PyValueError::new_err(format!("{op_name} failed: {e}")))?;
    let (vec, _offset) = result.into_raw_vec_and_offset();
    Array::from_vec_shape(vec, &shape).map_err(PyErr::from)
}

/// Convert a NumPy-style (possibly negative) axis to a `0..ndim` index.
fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let resolved = if axis < 0 { axis + ndim as isize } else { axis };
    if resolved < 0 || resolved as usize >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for an array of dimension {ndim}"
        )));
    }
    Ok(resolved as usize)
}

/// Borrow `arr` (which must be exactly 2-D) as an `ArrayView2`.
fn to_array_view2(arr: &Array<f64>) -> Result<ArrayView2<f64>, String> {
    if arr.ndim() != 2 {
        return Err(format!("Expected 2D array, got {}D", arr.ndim()));
    }
    arr.data
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|e| format!("Failed to convert to 2D view: {}", e))
}

/// Apply ReLU activation function
#[pyfunction]
fn relu(x: &PyArray) -> PyResult<PyArray> {
    use crate::nn::activation;
    let inner = apply_elementwise_1d(&x.inner, "ReLU", activation::relu)?;
    Ok(PyArray { inner })
}

/// Apply sigmoid activation function
#[pyfunction]
fn sigmoid(x: &PyArray) -> PyResult<PyArray> {
    use crate::nn::activation;
    let inner = apply_elementwise_1d(&x.inner, "Sigmoid", activation::sigmoid)?;
    Ok(PyArray { inner })
}

/// Apply tanh activation function
#[pyfunction]
fn tanh(x: &PyArray) -> PyResult<PyArray> {
    use crate::nn::activation;
    let inner = apply_elementwise_1d(&x.inner, "Tanh", activation::tanh)?;
    Ok(PyArray { inner })
}

/// Apply softmax activation function along `axis` (default: the last axis).
///
/// Supports arrays of any rank >= 1: every 1-D lane along `axis` is
/// normalized independently by the same `crate::nn::activation::softmax`
/// 1-D core op the old 2-D-only path used, generalized to N-D via manual
/// strided indexing (the same technique `stats::median`'s `axis` branch in
/// this crate already uses for the same purpose).
///
/// Note: this honors a caller-supplied `axis` for 2-D input too. The
/// previous implementation silently hardcoded axis 1 (last axis) for every
/// 2-D array regardless of what `axis` was passed; `axis=0` now actually
/// normalizes down the columns, matching NumPy/SciPy convention.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn softmax(x: &PyArray, axis: Option<isize>) -> PyResult<PyArray> {
    use crate::nn::activation;

    let shape = x.inner.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "Softmax requires an array with at least 1 dimension",
        ));
    }
    let ax = normalize_axis(axis.unwrap_or(-1), ndim)?;
    let axis_len = shape[ax];
    if axis_len == 0 {
        return Err(PyValueError::new_err("Softmax requires a non-empty axis"));
    }

    let mut out_shape = shape.clone();
    out_shape.remove(ax);
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    let op = operand(&x.inner);
    let data: &[f64] = &op;
    let mut result = vec![0.0_f64; data.len()];

    for out_i in 0..out_size {
        // Decode `out_i` into a multi-index over `out_shape` (row-major).
        let mut out_idx = vec![0usize; out_shape.len()];
        let mut tmp = out_i;
        for d in (0..out_shape.len()).rev() {
            out_idx[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        // Gather the lane along `ax` at this outer position, in original
        // (row-major, `shape`-strided) linear order.
        let mut lane = Vec::with_capacity(axis_len);
        let mut lane_linear = Vec::with_capacity(axis_len);
        for j in 0..axis_len {
            let mut full_idx = out_idx.clone();
            full_idx.insert(ax, j);
            let mut linear = 0usize;
            let mut stride = 1usize;
            for d in (0..shape.len()).rev() {
                linear += full_idx[d] * stride;
                stride *= shape[d];
            }
            lane.push(data[linear]);
            lane_linear.push(linear);
        }

        let view = ArrayView1::from(&lane[..]);
        let sm = activation::softmax(&view)
            .map_err(|e| PyValueError::new_err(format!("Softmax failed: {}", e)))?;
        for (j, &linear) in lane_linear.iter().enumerate() {
            result[linear] = sm[j];
        }
    }

    let arr = Array::from_vec_shape(result, &shape)?;
    Ok(PyArray { inner: arr })
}

/// Compute mean squared error loss.
///
/// `L = mean((predictions - targets)^2)` over every element, which is
/// invariant to how the elements are arranged into dimensions -- so, unlike
/// the loss below, this needs no per-rank special casing: any two
/// same-shaped arrays of any rank work.
#[pyfunction]
fn mse_loss(predictions: &PyArray, targets: &PyArray) -> PyResult<f64> {
    use crate::nn::loss;
    use crate::nn::ReductionMode;

    if predictions.inner.shape() != targets.inner.shape() {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            predictions.inner.shape(),
            targets.inner.shape()
        )));
    }

    let pred_op = operand(&predictions.inner);
    let targ_op = operand(&targets.inner);
    let pred_view = ArrayView1::from(&*pred_op);
    let targ_view = ArrayView1::from(&*targ_op);

    loss::mse_loss(&targ_view, &pred_view, ReductionMode::Mean)
        .map_err(|e| PyValueError::new_err(format!("MSE loss calculation failed: {}", e)))
}

/// Compute cross-entropy loss between `predictions` and `targets`.
///
/// * 1-D input: binary cross-entropy, elementwise over the whole array.
/// * N-D input (N >= 2): categorical cross-entropy treating the *last* axis
///   as the class-probability distribution (matching this module's
///   `softmax`'s default `axis=-1`) and every leading axis as "batch" --
///   the input is reshaped to `(prod(leading dims), n_classes)` and the
///   mean is taken over that flattened batch. A plain `(batch, classes)`
///   2-D array behaves exactly as before; higher-rank input (e.g.
///   `(batch, seq_len, classes)`) now works instead of being rejected.
///
/// Note: the previous implementation passed `predictions` where the core
/// `binary_cross_entropy`/`categorical_cross_entropy` functions expect
/// `y_true` (and vice versa). Both formulas are asymmetric in their two
/// array arguments, so this silently computed the wrong loss value; it is
/// fixed here.
#[pyfunction]
fn cross_entropy_loss(predictions: &PyArray, targets: &PyArray) -> PyResult<f64> {
    use crate::nn::loss;
    use crate::nn::ReductionMode;

    if predictions.inner.shape() != targets.inner.shape() {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            predictions.inner.shape(),
            targets.inner.shape()
        )));
    }

    let ndim = predictions.inner.ndim();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "Cross-entropy loss requires an array with at least 1 dimension",
        ));
    }

    if ndim == 1 {
        let pred_op = operand(&predictions.inner);
        let targ_op = operand(&targets.inner);
        let pred_view = ArrayView1::from(&*pred_op);
        let targ_view = ArrayView1::from(&*targ_op);
        loss::binary_cross_entropy(&targ_view, &pred_view, ReductionMode::Mean).map_err(|e| {
            PyValueError::new_err(format!("Cross-entropy loss calculation failed: {}", e))
        })
    } else {
        let shape = predictions.inner.shape();
        let n_classes = shape[ndim - 1];
        let batch = shape[..ndim - 1].iter().product::<usize>().max(1);

        let pred_flat = operand(&predictions.inner).to_vec();
        let targ_flat = operand(&targets.inner).to_vec();
        let pred_2d = Array::from_vec_shape(pred_flat, &[batch, n_classes])?;
        let targ_2d = Array::from_vec_shape(targ_flat, &[batch, n_classes])?;

        let pred_view = to_array_view2(&pred_2d).map_err(PyValueError::new_err)?;
        let targ_view = to_array_view2(&targ_2d).map_err(PyValueError::new_err)?;

        loss::categorical_cross_entropy(&targ_view, &pred_view, ReductionMode::Mean).map_err(|e| {
            PyValueError::new_err(format!("Cross-entropy loss calculation failed: {}", e))
        })
    }
}

/// Apply dropout (training mode simulation)
#[pyfunction]
fn dropout(x: &PyArray, p: f64) -> PyResult<PyArray> {
    if !(0.0..1.0).contains(&p) {
        return Err(PyValueError::new_err(
            "Dropout probability must be in [0, 1)",
        ));
    }

    use crate::nn::normalization;
    let inner = apply_elementwise_1d(&x.inner, "Dropout", |view| {
        normalization::dropout(view, p, true)
    })?;
    Ok(PyArray { inner })
}

/// Apply batch normalization.
///
/// Treats the *last* axis as "features" (scaled/shifted independently, each
/// with its own batch mean/variance) and every leading axis as "batch" --
/// matching NumPy/ML convention and exactly how the previous 2-D-only
/// implementation already treated a `(batch, features)` array. Any array of
/// rank >= 2 is reshaped to `(prod(leading dims), features)`, normalized,
/// and reshaped back, so `(batch, features)` behaves exactly as before and
/// higher-rank input (e.g. `(batch, height, width, channels)`) now works
/// instead of being rejected.
#[pyfunction]
#[pyo3(signature = (x, eps=None))]
fn batch_norm(x: &PyArray, eps: Option<f64>) -> PyResult<PyArray> {
    let eps = eps.unwrap_or(1e-5);
    use crate::nn::normalization;

    let shape = x.inner.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(PyValueError::new_err(
            "Batch normalization requires an array with at least 2 dimensions (..., features)",
        ));
    }

    let n_features = shape[ndim - 1];
    let batch = shape[..ndim - 1].iter().product::<usize>().max(1);

    let flat = operand(&x.inner).to_vec();
    let x_2d = Array::from_vec_shape(flat, &[batch, n_features])?;
    let view = to_array_view2(&x_2d).map_err(PyValueError::new_err)?;

    // Learnable parameters (gamma and beta), initialized to 1 and 0.
    let gamma = Array1::from_elem(n_features, 1.0);
    let beta = Array1::from_elem(n_features, 0.0);

    let result_nd = normalization::batch_norm_1d(&view, &gamma.view(), &beta.view(), eps)
        .map_err(|e| PyValueError::new_err(format!("Batch normalization failed: {}", e)))?;

    let (vec, _offset) = result_nd.into_raw_vec_and_offset();
    let inner = Array::from_vec_shape(vec, &shape)?;
    Ok(PyArray { inner })
}

/// Register neural network functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create nn submodule
    let nn_module = PyModule::new(m.py(), "nn")?;

    // Add activation functions
    nn_module.add_function(wrap_pyfunction!(relu, m)?)?;
    nn_module.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    nn_module.add_function(wrap_pyfunction!(tanh, m)?)?;
    nn_module.add_function(wrap_pyfunction!(softmax, m)?)?;

    // Add loss functions
    nn_module.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    nn_module.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;

    // Add normalization functions
    nn_module.add_function(wrap_pyfunction!(dropout, m)?)?;
    nn_module.add_function(wrap_pyfunction!(batch_norm, m)?)?;

    m.add_submodule(&nn_module)?;

    Ok(())
}

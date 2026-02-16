//! Neural network operations for Python bindings

use crate::array::Array;
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::ndarray::{ArrayView1, ArrayView2, IxDyn};

/// Helper function to convert Array<f64> to ArrayView1<f64>
fn to_array_view1(arr: &Array<f64>) -> Result<ArrayView1<f64>, String> {
    if arr.ndim() != 1 {
        return Err(format!("Expected 1D array, got {}D", arr.ndim()));
    }
    arr.data
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix1>()
        .map_err(|e| format!("Failed to convert to 1D view: {}", e))
}

/// Helper function to convert Array<f64> to ArrayView2<f64>
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

    if x.inner.ndim() == 1 {
        let view = to_array_view1(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::relu(&view)
            .map_err(|e| PyValueError::new_err(format!("ReLU failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else if x.inner.ndim() == 2 {
        let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::relu_2d(&view)
            .map_err(|e| PyValueError::new_err(format!("ReLU failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else {
        Err(PyValueError::new_err(format!(
            "ReLU only supports 1D and 2D arrays, got {}D",
            x.inner.ndim()
        )))
    }
}

/// Apply sigmoid activation function
#[pyfunction]
fn sigmoid(x: &PyArray) -> PyResult<PyArray> {
    use crate::nn::activation;

    if x.inner.ndim() == 1 {
        let view = to_array_view1(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::sigmoid(&view)
            .map_err(|e| PyValueError::new_err(format!("Sigmoid failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else if x.inner.ndim() == 2 {
        let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::sigmoid_2d(&view)
            .map_err(|e| PyValueError::new_err(format!("Sigmoid failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else {
        Err(PyValueError::new_err(format!(
            "Sigmoid only supports 1D and 2D arrays, got {}D",
            x.inner.ndim()
        )))
    }
}

/// Apply tanh activation function
#[pyfunction]
fn tanh(x: &PyArray) -> PyResult<PyArray> {
    use crate::nn::activation;

    if x.inner.ndim() == 1 {
        let view = to_array_view1(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::tanh(&view)
            .map_err(|e| PyValueError::new_err(format!("Tanh failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else if x.inner.ndim() == 2 {
        let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::tanh_2d(&view)
            .map_err(|e| PyValueError::new_err(format!("Tanh failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else {
        Err(PyValueError::new_err(format!(
            "Tanh only supports 1D and 2D arrays, got {}D",
            x.inner.ndim()
        )))
    }
}

/// Apply softmax activation function
#[pyfunction]
fn softmax(x: &PyArray, axis: Option<isize>) -> PyResult<PyArray> {
    use crate::nn::activation;
    let _axis = axis.unwrap_or(-1);

    if x.inner.ndim() == 1 {
        let view = to_array_view1(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = activation::softmax(&view)
            .map_err(|e| PyValueError::new_err(format!("Softmax failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else if x.inner.ndim() == 2 {
        let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;
        // For 2D, apply softmax along last axis (axis 1)
        let result_nd = activation::softmax_2d(&view, 1)
            .map_err(|e| PyValueError::new_err(format!("Softmax failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else {
        Err(PyValueError::new_err(format!(
            "Softmax only supports 1D and 2D arrays, got {}D",
            x.inner.ndim()
        )))
    }
}

/// Compute mean squared error loss
#[pyfunction]
fn mse_loss(predictions: &PyArray, targets: &PyArray) -> PyResult<f64> {
    use crate::nn::loss;
    use crate::nn::ReductionMode;

    if predictions.inner.ndim() == 1 {
        let pred_view = to_array_view1(&predictions.inner).map_err(PyValueError::new_err)?;
        let targ_view = to_array_view1(&targets.inner).map_err(PyValueError::new_err)?;
        loss::mse_loss(&pred_view, &targ_view, ReductionMode::Mean)
            .map_err(|e| PyValueError::new_err(format!("MSE loss calculation failed: {}", e)))
    } else if predictions.inner.ndim() == 2 {
        let pred_view = to_array_view2(&predictions.inner).map_err(PyValueError::new_err)?;
        let targ_view = to_array_view2(&targets.inner).map_err(PyValueError::new_err)?;
        loss::mse_loss_2d(&pred_view, &targ_view, ReductionMode::Mean)
            .map_err(|e| PyValueError::new_err(format!("MSE loss calculation failed: {}", e)))
    } else {
        Err(PyValueError::new_err(format!(
            "MSE loss only supports 1D and 2D arrays, got {}D",
            predictions.inner.ndim()
        )))
    }
}

/// Compute cross-entropy loss
#[pyfunction]
fn cross_entropy_loss(predictions: &PyArray, targets: &PyArray) -> PyResult<f64> {
    use crate::nn::loss;
    use crate::nn::ReductionMode;

    if predictions.inner.ndim() == 1 {
        let pred_view = to_array_view1(&predictions.inner).map_err(PyValueError::new_err)?;
        let targ_view = to_array_view1(&targets.inner).map_err(PyValueError::new_err)?;
        loss::binary_cross_entropy(&pred_view, &targ_view, ReductionMode::Mean).map_err(|e| {
            PyValueError::new_err(format!("Cross-entropy loss calculation failed: {}", e))
        })
    } else if predictions.inner.ndim() == 2 {
        let pred_view = to_array_view2(&predictions.inner).map_err(PyValueError::new_err)?;
        let targ_view = to_array_view2(&targets.inner).map_err(PyValueError::new_err)?;
        loss::categorical_cross_entropy(&pred_view, &targ_view, ReductionMode::Mean).map_err(|e| {
            PyValueError::new_err(format!("Cross-entropy loss calculation failed: {}", e))
        })
    } else {
        Err(PyValueError::new_err(format!(
            "Cross-entropy loss only supports 1D and 2D arrays, got {}D",
            predictions.inner.ndim()
        )))
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

    if x.inner.ndim() == 1 {
        let view = to_array_view1(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = normalization::dropout(&view, p, true)
            .map_err(|e| PyValueError::new_err(format!("Dropout failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else if x.inner.ndim() == 2 {
        let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;
        let result_nd = normalization::dropout_2d(&view, p, true)
            .map_err(|e| PyValueError::new_err(format!("Dropout failed: {}", e)))?;
        let result = Array::from_ndarray(result_nd.into_dyn());
        Ok(PyArray { inner: result })
    } else {
        Err(PyValueError::new_err(format!(
            "Dropout only supports 1D and 2D arrays, got {}D",
            x.inner.ndim()
        )))
    }
}

/// Apply batch normalization
#[pyfunction]
fn batch_norm(x: &PyArray, eps: Option<f64>) -> PyResult<PyArray> {
    let eps = eps.unwrap_or(1e-5);
    use crate::nn::normalization;
    use scirs2_core::ndarray::Array1;

    if x.inner.ndim() != 2 {
        return Err(PyValueError::new_err(
            "Batch normalization requires a 2D array",
        ));
    }

    let view = to_array_view2(&x.inner).map_err(PyValueError::new_err)?;

    // Create learnable parameters (gamma and beta) initialized to 1 and 0
    let n_features = x.inner.shape()[1];
    let gamma = Array1::from_elem(n_features, 1.0);
    let beta = Array1::from_elem(n_features, 0.0);

    let result_nd = normalization::batch_norm_1d(&view, &gamma.view(), &beta.view(), eps)
        .map_err(|e| PyValueError::new_err(format!("Batch normalization failed: {}", e)))?;
    let result = Array::from_ndarray(result_nd.into_dyn());
    Ok(PyArray { inner: result })
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

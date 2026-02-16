//! I/O operations for Python bindings

use crate::io::{npy_npz, SerializeFormat};
use crate::python::array::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::path::PathBuf;

/// Save array to NPY format
#[pyfunction]
fn save_npy(file: String, arr: &PyArray) -> PyResult<()> {
    let path = PathBuf::from(file);
    let mut writer = File::create(&path)
        .map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;
    npy_npz::serialize_to_file(&arr.inner, &mut writer, SerializeFormat::Npy)
        .map_err(|e| PyValueError::new_err(format!("Failed to save NPY file: {}", e)))
}

/// Load array from NPY format
#[pyfunction]
fn load_npy(file: String) -> PyResult<PyArray> {
    let path = PathBuf::from(file);
    let mut reader = File::open(&path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
    let arr = npy_npz::deserialize_from_file(&mut reader, SerializeFormat::Npy)
        .map_err(|e| PyValueError::new_err(format!("Failed to load NPY file: {}", e)))?;
    Ok(PyArray { inner: arr })
}

/// Save array to CSV format
#[pyfunction]
fn save_csv(file: String, arr: &PyArray) -> PyResult<()> {
    let path = PathBuf::from(file);
    arr.inner
        .to_file(&path, SerializeFormat::Csv)
        .map_err(|e| PyValueError::new_err(format!("Failed to save CSV file: {}", e)))
}

/// Load array from CSV format
#[pyfunction]
fn load_csv(file: String) -> PyResult<PyArray> {
    use crate::array::Array;
    let path = PathBuf::from(file);
    let arr = Array::<f64>::from_file(&path, SerializeFormat::Csv)
        .map_err(|e| PyValueError::new_err(format!("Failed to load CSV file: {}", e)))?;
    Ok(PyArray { inner: arr })
}

/// Save array to JSON format
#[pyfunction]
fn save_json(file: String, arr: &PyArray) -> PyResult<()> {
    let path = PathBuf::from(file);
    arr.inner
        .to_file(&path, SerializeFormat::Json)
        .map_err(|e| PyValueError::new_err(format!("Failed to save JSON file: {}", e)))
}

/// Load array from JSON format
#[pyfunction]
fn load_json(file: String) -> PyResult<PyArray> {
    use crate::array::Array;
    let path = PathBuf::from(file);
    let arr = Array::<f64>::from_file(&path, SerializeFormat::Json)
        .map_err(|e| PyValueError::new_err(format!("Failed to load JSON file: {}", e)))?;
    Ok(PyArray { inner: arr })
}

/// Register I/O functions
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create io submodule
    let io_module = PyModule::new(m.py(), "io")?;

    // Add NPY functions
    io_module.add_function(wrap_pyfunction!(save_npy, m)?)?;
    io_module.add_function(wrap_pyfunction!(load_npy, m)?)?;

    // Add CSV functions
    io_module.add_function(wrap_pyfunction!(save_csv, m)?)?;
    io_module.add_function(wrap_pyfunction!(load_csv, m)?)?;

    // Add JSON functions
    io_module.add_function(wrap_pyfunction!(save_json, m)?)?;
    io_module.add_function(wrap_pyfunction!(load_json, m)?)?;

    m.add_submodule(&io_module)?;

    Ok(())
}

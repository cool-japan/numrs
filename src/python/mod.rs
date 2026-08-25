//! Python bindings for NumRS2 via PyO3
//!
//! This module provides comprehensive Python bindings for NumRS2's functionality,
//! enabling seamless integration with Python and NumPy.
//!
//! ## Module Organization
//!
//! - `array` - Core array operations and creation
//! - `linalg` - Linear algebra operations
//! - `stats` - Statistical functions and distributions
//! - `random` - Random number generation (`Generator`, `default_rng`, ...)
//! - `fft` - Fast Fourier Transform operations
//! - `optimize` - Optimization algorithms
//! - `nn` - Neural network primitives
//! - `symbolic` - Symbolic computation
//! - `io` - Data I/O formats
//! - `error` - Error conversion utilities
//!
//! ## Zero-copy vs. copy: per API
//!
//! `Array` (`PyArray`, in `array.rs`) is `Arc`-backed copy-on-write: `Clone`
//! is O(1) and shares the underlying buffer with every other handle to it.
//! That invariant is exactly what makes an *unconditional* copy necessary
//! at most of the NumPy boundary below -- handing NumPy the same buffer
//! without copying would let NumPy-side mutation corrupt memory another
//! `Array` handle still expects to be untouched. The one direction that
//! genuinely needs no copy is a freshly computed result with no other
//! referents (e.g. an FFT spectrum), which can be moved into NumPy outright.
//!
//! | API | Direction | Cost | Why |
//! |---|---|---|---|
//! | `Array(data)` (`__new__`) | NumPy (`float32`/`float64`)/list/tuple -> `Array` | 1 copy | Materializes NumRS2's own owned storage from a view it does not control (stride-aware via `.as_array()`, so a non-contiguous/transposed NumPy input works too, not just a C-contiguous one); a `float32` input additionally pays a per-element widening cast to this crate's `f64` storage. |
//! | `Array.to_numpy` | `Array` -> NumPy `float64` | 1 copy | `Array` may be `Arc`-shared; NumPy needs sole, mutable ownership of what it's given, so this always copies regardless of whether this particular handle happens to be uniquely owned right now. `Array` is `f64`-only storage, so unlike the constructor above there is no `float32` *output* path -- only `float32` NumPy *input* is accepted (and upcast). |
//! | `fft`/`rfft`/`fftn`/`rfftn` (forward) | -> NumPy `complex128` | 0 copy | Output is a brand-new buffer built to hand straight to NumPy (`IntoPyArray`) with no other referent; nothing to protect by copying. |
//! | `ifft`/`irfft` (1-D inverse) | NumPy `complex128` -> Rust `&[Complex64]` | 0 copy when contiguous, else 1 | `fft::flat_or_owned` borrows the input view's slice directly when it is contiguous (the common case: feeding one of these functions' own output back in), and falls back to one `Vec`-materializing copy for a non-contiguous view. |
//! | `ifftn`/`irfftn` (N-D inverse) | NumPy `complex128` -> Rust `ArrayD` | 1 copy, always | The underlying `crate::fft::{ifftn, irfftn}` take an owned `ArrayD<T>` (not a borrowable view or slice), so there is no representation to borrow into regardless of the input's contiguity; unlike the 1-D inverse path, this one cannot skip the copy. |
//! | `random.Generator` sampling, array creation (`zeros`, `arange`, ...) | Rust `Vec` -> `Array` | 0 extra copy | Built directly as `Array` via `from_vec`/`from_vec_shape`, which take ownership of the `Vec` rather than copying out of it. |

use pyo3::prelude::*;

pub mod array;
pub mod error;
pub mod fft;
pub mod io;
pub mod linalg;
pub mod nn;
pub mod optimize;
pub mod random;
pub mod stats;
pub mod symbolic;

/// NumRS2 Python module
#[pymodule]
fn _numrs2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add array class and functions
    array::register(m)?;

    // Add linear algebra functions
    linalg::register(m)?;

    // Add statistics functions
    stats::register(m)?;

    // Add random number generation (Generator, default_rng, rand, randn)
    random::register(m)?;

    // Add FFT functions
    fft::register(m)?;

    // Add optimization functions
    optimize::register(m)?;

    // Add neural network functions
    nn::register(m)?;

    // Add symbolic computation functions
    symbolic::register(m)?;

    // Add I/O functions
    io::register(m)?;

    Ok(())
}

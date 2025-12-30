// Matrix module for NumRS2
// Provides matrix-specific functionality and operations

mod banded;
mod matrix_class;
pub mod special;

pub use banded::BandedMatrix;
pub use matrix_class::{
    asmatrix, asmatrix_from_nested, matrix, matrix_from_nested, matrix_from_scalar, Matrix,
};
pub use special::*;

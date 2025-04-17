// Matrix module for NumRS2
// Provides matrix-specific functionality and operations

mod matrix_class;
mod banded;
pub mod special;

pub use matrix_class::Matrix;
pub use banded::BandedMatrix;
pub use special::*;
// Matrix module for NumRS2
// Provides matrix-specific functionality and operations

mod banded;
mod matrix_class;
pub mod special;

pub use banded::BandedMatrix;
pub use matrix_class::Matrix;
pub use special::*;

//! Array manipulation operations
//!
//! This module provides NumPy-like array manipulation functions including:
//!
//! ## Transform Operations
//! - [`roll`] - Roll array elements along an axis
//! - [`flip`] - Reverse the order of elements along an axis
//! - [`flipud`] - Flip array in up/down direction
//! - [`fliplr`] - Flip array in left/right direction
//! - [`rot90`] - Rotate array by 90 degrees
//! - [`expand_dims`] - Expand the shape of an array
//! - [`squeeze`] - Remove axes of length 1
//! - [`ravel`] - Return a flattened array (view when possible)
//! - [`flatten`] - Return a flattened copy of the array
//!
//! ## Modify Operations
//! - [`delete`] - Delete sub-arrays along an axis
//! - [`insert`] - Insert values along an axis
//! - [`pad`] - Pad an array
//! - [`trim_zeros`] - Trim leading/trailing zeros
//! - [`extract`] - Extract elements by condition
//! - [`place`] - Place values by condition
//! - [`put`] - Put values at indices
//! - [`compress`] - Select slices by condition
//!
//! ## Bit Operations
//! - [`packbits`] - Pack binary values into uint8
//! - [`unpackbits`] - Unpack uint8 to binary values
//!
//! ## Index Operations
//! - [`unravel_index`] - Convert flat indices to coordinates
//! - [`ravel_multi_index`] - Convert coordinates to flat indices
//!
//! ## Triangle Operations
//! - [`tril`] - Lower triangle of an array
//! - [`triu`] - Upper triangle of an array
//!
//! ## Append Operations
//! - [`append`] - Append values to an array

mod bits_index;
mod modify;
mod pad;
mod transform;
mod triangle;

// Re-export all public functions from submodules

// Transform operations
pub use transform::{expand_dims, flatten, flip, fliplr, flipud, ravel, roll, rot90, squeeze};

// Modify operations
pub use modify::{compress, delete, extract, insert, place, put, trim_zeros};

// Pad operations
pub use pad::pad;

// Bit and index operations
pub use bits_index::{packbits, ravel_multi_index, unpackbits, unravel_index};

// Triangle and append operations
pub use triangle::{append, tril, triu};

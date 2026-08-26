//! Indexing module for NumRS2 arrays
//!
//! This module provides comprehensive indexing capabilities for arrays, including:
//! - Basic indexing with integer indices
//! - Slicing with ranges and steps
//! - Boolean (mask) indexing
//! - Fancy indexing with index arrays
//! - NewAxis and Ellipsis support
//!
//! # Examples
//!
//! ```
//! use numrs2::prelude::*;
//!
//! // Create a 2D array
//! let arr = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
//!
//! // Basic indexing
//! let elem = arr.get(&[0, 1]).expect("valid index should succeed");
//! assert_eq!(elem, 2);
//!
//! // Slicing
//! let slice = arr.index(&[IndexSpec::All, IndexSpec::Slice(0, Some(2), None)])
//!     .expect("valid slice should succeed");
//! assert_eq!(slice.shape(), vec![2, 2]);
//! ```

mod array_methods;
mod core;
mod fancy;
mod index_utils;
mod take_put;

#[cfg(test)]
mod tests;

use crate::array::Array;
use crate::error::Result;
use std::ops::Range;

// Re-export all public items.
//
// `array_methods`, `core`, and `fancy` contain only `impl Array<T>` blocks
// (inherent methods, reachable wherever `Array` is already in scope) with
// no standalone `pub` items -- a `pub use self::x::*` of any of them is an
// always-empty glob, so they are intentionally not re-exported here.
pub use self::index_utils::*;
pub use self::take_put::*;

/// Represents an index specification for a single dimension
#[derive(Clone)]
pub enum IndexSpec {
    /// A single index
    Index(usize),
    /// A range of indices with start, end, and optional step
    Slice(usize, Option<usize>, Option<usize>),
    /// A set of arbitrary indices
    Indices(Vec<usize>),
    /// A boolean mask
    Mask(Vec<bool>),
    /// All indices
    All,
    /// Ellipsis (...) - expands to the number of : needed for selection
    Ellipsis,
    /// NewAxis (np.newaxis) - inserts a new axis of length 1
    NewAxis,
}

impl IndexSpec {
    /// Create a new slice specification
    pub fn slice(start: usize, end: Option<usize>, step: Option<usize>) -> Self {
        IndexSpec::Slice(start, end, step)
    }

    /// Create a new index specification from a range
    pub fn from_range(range: Range<usize>) -> Self {
        IndexSpec::Slice(range.start, Some(range.end), None)
    }

    /// Create a new index specification from a vector of indices
    pub fn from_indices(indices: Vec<usize>) -> Self {
        IndexSpec::Indices(indices)
    }

    /// Create a new index specification from a boolean mask
    pub fn from_mask(mask: Vec<bool>) -> Self {
        IndexSpec::Mask(mask)
    }

    /// Create an ellipsis index specification
    pub fn ellipsis() -> Self {
        IndexSpec::Ellipsis
    }

    /// Create a new axis index specification (inserts dimension of size 1)
    pub fn newaxis() -> Self {
        IndexSpec::NewAxis
    }
}

/// Insert new axes at specified output positions in the array
///
/// This helper function is used by the indexing system to handle `IndexSpec::NewAxis`.
/// It inserts axes of size 1 at the specified final output positions.
///
/// # Arguments
/// * `arr` - The array to modify
/// * `output_positions` - The final output positions for new axes (e.g., [0, 2] means
///   dims 0 and 2 of the result will be size 1 newaxis)
///
/// # Returns
/// A new array with additional axes of size 1 at the specified positions
pub(crate) fn insert_newaxis<T: Clone + num_traits::Zero>(
    arr: &Array<T>,
    output_positions: &[usize],
) -> Result<Array<T>> {
    if output_positions.is_empty() {
        return Ok(arr.clone());
    }

    // Convert output_positions to a set for fast lookup
    let newaxis_set: std::collections::HashSet<usize> = output_positions.iter().copied().collect();

    // Calculate the new total dimensions
    let new_ndim = arr.ndim() + output_positions.len();

    // Build the new shape by interleaving original dims and newaxis
    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut orig_dim_idx = 0;

    for out_dim in 0..new_ndim {
        if newaxis_set.contains(&out_dim) {
            // This position is a newaxis
            new_shape.push(1);
        } else {
            // This position comes from the original array
            if orig_dim_idx < arr.ndim() {
                new_shape.push(arr.shape()[orig_dim_idx]);
                orig_dim_idx += 1;
            } else {
                // Shouldn't happen if positions are correct
                new_shape.push(1);
            }
        }
    }

    // Reshape the array to the new shape
    Ok(arr.reshape(&new_shape))
}

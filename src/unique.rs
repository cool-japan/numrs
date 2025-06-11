use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Type alias for unique operation tuple results
pub type UniqueTuple<T> = (Array<T>, Array<usize>, Array<usize>, Array<usize>);

/// Find the unique elements of an array.
///
/// This function can return the unique elements of an array, optionally
/// along a specified axis. It can also return the indices of the first
/// occurrences of the unique values, and the indices to reconstruct the
/// original array.
///
/// # Parameters
///
/// * `a` - Input array
/// * `axis` - Optional axis along which to find unique elements
/// * `return_index` - If true, also return the indices of the first occurrences
/// * `return_inverse` - If true, also return the indices to reconstruct the original array
/// * `return_counts` - If true, also return the counts of each unique value
///
/// # Returns
///
/// A UniqueResult struct containing some or all of:
/// * The unique values in the array (required)
/// * The indices of the first unique values (if return_index is true)
/// * The indices to reconstruct the original array (if return_inverse is true)
/// * The counts of each unique value (if return_counts is true)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Simple use case
/// let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]).reshape(&[6]);
/// let result = unique(&a, None, None, None, None).unwrap();
/// assert_eq!(result.values.to_vec(), vec![1, 2, 3, 4]);
///
/// // With indices of first occurrences
/// let result = unique(&a, None, Some(true), None, None).unwrap();
/// let (values, indices) = result.values_indices().unwrap();
/// assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
/// assert_eq!(indices.to_vec(), vec![0, 1, 2, 5]); // Positions where each unique value first appears
///
/// // With counts
/// let result = unique(&a, None, None, None, Some(true)).unwrap();
/// let (values, counts) = result.values_counts().unwrap();
/// assert_eq!(counts.to_vec(), vec![2, 2, 1, 1]); // 1 appears twice, 2 appears twice, etc.
///
/// // Along a specific axis (for 2D arrays)
/// let b = Array::from_vec(vec![1, 2, 3, 2, 1, 4]).reshape(&[2, 3]);
/// let result = unique(&b, Some(0), None, None, None).unwrap();
/// // Unique rows
/// ```
pub fn unique<T>(
    a: &Array<T>,
    axis: Option<usize>,
    return_index: Option<bool>,
    return_inverse: Option<bool>,
    return_counts: Option<bool>,
) -> Result<UniqueResult<T>>
where
    T: Clone + Hash + Eq + Debug + Zero,
{
    // If no axis is provided, flatten the array and find unique elements
    if axis.is_none() {
        let flat_data = a.to_vec();

        // Track unique elements, their first indices, and counts
        let mut unique_elements = Vec::new();
        let mut first_indices = Vec::new();
        let mut inverse_indices = vec![0; flat_data.len()];
        let mut value_to_index = HashMap::new();

        // Process each element
        for (i, value) in flat_data.iter().enumerate() {
            if let Some(&idx) = value_to_index.get(value) {
                // Element already seen
                inverse_indices[i] = idx;
            } else {
                // New unique element
                let new_idx = unique_elements.len();
                unique_elements.push(value.clone());
                first_indices.push(i);
                value_to_index.insert(value, new_idx);
                inverse_indices[i] = new_idx;
            }
        }

        // Calculate counts if needed
        let counts = if return_counts.unwrap_or(false) {
            let mut counts_vec = vec![0; unique_elements.len()];
            for &idx in &inverse_indices {
                counts_vec[idx] += 1;
            }
            Some(Array::from_vec(counts_vec))
        } else {
            None
        };

        // Construct the result
        let unique_array = Array::from_vec(unique_elements);

        return Ok(UniqueResult {
            values: unique_array,
            indices: if return_index.unwrap_or(false) {
                Some(Array::from_vec(first_indices))
            } else {
                None
            },
            inverse: if return_inverse.unwrap_or(false) {
                Some(Array::from_vec(inverse_indices))
            } else {
                None
            },
            counts,
        });
    }

    // Process along a specific axis
    let axis_val = axis.unwrap();
    if axis_val >= a.ndim() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Axis {} out of bounds for array of dimension {}",
            axis_val,
            a.ndim()
        )));
    }

    // Get the shape
    let shape = a.shape();

    // For 1D arrays, axis=0 is the same as no axis
    if shape.len() == 1 && axis_val == 0 {
        return unique(a, None, return_index, return_inverse, return_counts);
    }

    // For higher dimensions, we need to find unique subarrays along the specified axis

    // We'll convert each subarray along the axis to a hashable representation
    // This is a simplified approach - for better performance we would use more
    // efficient data structures and algorithms

    // Get the size of the axis and calculate the shape of each subarray
    let axis_len = shape[axis_val];

    // Create a vector to hold each subarray along the axis
    let mut subarrays = Vec::with_capacity(axis_len);
    let mut subarray_hashes = Vec::with_capacity(axis_len);

    // Extract subarrays along the specified axis
    for i in 0..axis_len {
        // Get the subarray
        let subarray = a.slice(axis_val, i)?;

        // Convert to a hashable representation (serialize to a vector)
        let hash_rep = subarray.to_vec();

        subarrays.push(subarray);
        subarray_hashes.push(hash_rep);
    }

    // Find unique subarrays
    let mut unique_indices = Vec::new();
    let mut index_map = HashMap::new();
    let mut inverse = vec![0; axis_len];
    let mut seen = HashSet::new();

    for i in 0..axis_len {
        let hash_rep = &subarray_hashes[i];

        if !seen.contains(hash_rep) {
            // This is a new unique subarray
            let idx = unique_indices.len();
            unique_indices.push(i);
            index_map.insert(hash_rep, idx);
            seen.insert(hash_rep.clone());
            inverse[i] = idx;
        } else {
            // This subarray has been seen before
            let idx = *index_map.get(hash_rep).unwrap();
            inverse[i] = idx;
        }
    }

    // Calculate counts if needed
    let counts = if return_counts.unwrap_or(false) {
        let mut counts_vec = vec![0; unique_indices.len()];
        for &idx in &inverse {
            counts_vec[idx] += 1;
        }
        Some(Array::from_vec(counts_vec))
    } else {
        None
    };

    // Create the output arrays

    // Create a new shape for the output with the axis dimension set to the number of unique subarrays
    let mut output_shape = shape.clone();
    output_shape[axis_val] = unique_indices.len();

    // Create the result array by concatenating the unique subarrays along the axis
    let mut unique_subarrays = Vec::with_capacity(unique_indices.len());
    for &idx in &unique_indices {
        unique_subarrays.push(&subarrays[idx]);
    }

    // Use the concatenate function to join the unique subarrays
    let values = if !unique_subarrays.is_empty() {
        // For now, convert the subarrays to a 1D array for each unique subarray
        // A better implementation would use proper array concatenation along the specified axis
        let mut unique_data = Vec::new();
        for &idx in &unique_indices {
            unique_data.extend_from_slice(&subarray_hashes[idx]);
        }
        Array::from_vec(unique_data).reshape(&output_shape)
    } else {
        // Empty result
        Array::zeros(&output_shape)
    };

    Ok(UniqueResult {
        values,
        indices: if return_index.unwrap_or(false) {
            Some(Array::from_vec(unique_indices))
        } else {
            None
        },
        inverse: if return_inverse.unwrap_or(false) {
            Some(Array::from_vec(inverse))
        } else {
            None
        },
        counts,
    })
}

/// Output type for unique function to handle variable return types
pub struct UniqueResult<T> {
    pub values: Array<T>,
    pub indices: Option<Array<usize>>,
    pub inverse: Option<Array<usize>>,
    pub counts: Option<Array<usize>>,
}

impl<T: Clone> UniqueResult<T> {
    /// Get the unique values only
    pub fn values(self) -> Array<T> {
        self.values
    }

    /// Get a tuple of (values, indices) if indices were requested
    pub fn values_indices(self) -> Result<(Array<T>, Array<usize>)> {
        match self.indices {
            Some(indices) => Ok((self.values, indices)),
            None => Err(NumRs2Error::InvalidOperation(
                "indices were not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, inverse) if inverse was requested
    pub fn values_inverse(self) -> Result<(Array<T>, Array<usize>)> {
        match self.inverse {
            Some(inverse) => Ok((self.values, inverse)),
            None => Err(NumRs2Error::InvalidOperation(
                "inverse was not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, counts) if counts were requested
    pub fn values_counts(self) -> Result<(Array<T>, Array<usize>)> {
        match self.counts {
            Some(counts) => Ok((self.values, counts)),
            None => Err(NumRs2Error::InvalidOperation(
                "counts were not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, indices, inverse) if both were requested
    pub fn values_indices_inverse(self) -> Result<(Array<T>, Array<usize>, Array<usize>)> {
        match (self.indices, self.inverse) {
            (Some(indices), Some(inverse)) => Ok((self.values, indices, inverse)),
            _ => Err(NumRs2Error::InvalidOperation(
                "either indices or inverse were not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, indices, counts) if both were requested
    pub fn values_indices_counts(self) -> Result<(Array<T>, Array<usize>, Array<usize>)> {
        match (self.indices, self.counts) {
            (Some(indices), Some(counts)) => Ok((self.values, indices, counts)),
            _ => Err(NumRs2Error::InvalidOperation(
                "either indices or counts were not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, inverse, counts) if both were requested
    pub fn values_inverse_counts(self) -> Result<(Array<T>, Array<usize>, Array<usize>)> {
        match (self.inverse, self.counts) {
            (Some(inverse), Some(counts)) => Ok((self.values, inverse, counts)),
            _ => Err(NumRs2Error::InvalidOperation(
                "either inverse or counts were not requested in the unique call".to_string(),
            )),
        }
    }

    /// Get a tuple of (values, indices, inverse, counts) if all were requested
    pub fn values_indices_inverse_counts(self) -> Result<UniqueTuple<T>> {
        match (self.indices, self.inverse, self.counts) {
            (Some(indices), Some(inverse), Some(counts)) => {
                Ok((self.values, indices, inverse, counts))
            }
            _ => Err(NumRs2Error::InvalidOperation(
                "not all of indices, inverse, and counts were requested in the unique call"
                    .to_string(),
            )),
        }
    }
}

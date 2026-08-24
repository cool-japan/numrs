use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Optimized version of the unique function to find unique elements of an array.
///
/// This version includes several optimizations over the standard unique function:
/// - Parallel processing for large arrays
/// - Pre-allocated buffers to reduce memory allocations
/// - Early capacity estimation based on input size
/// - SIMD-friendly memory access patterns where possible
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
pub fn unique_optimized<T>(
    a: &Array<T>,
    axis: Option<usize>,
    return_index: Option<bool>,
    return_inverse: Option<bool>,
    return_counts: Option<bool>,
) -> Result<UniqueResult<T>>
where
    T: Clone + Hash + Eq + Debug + Zero + Send + Sync,
{
    // Helper functions
    fn estimate_capacity(array_size: usize) -> usize {
        // Heuristic: for random data, expect about 90% of elements to be unique
        // for small arrays, just use the full size
        if array_size < 1000 {
            array_size
        } else {
            (array_size as f64 * 0.9) as usize
        }
    }

    // If no axis is provided, flatten the array and find unique elements
    if axis.is_none() {
        let flat_data = a.to_vec();
        let array_size = flat_data.len();

        // Optimize for large arrays by using parallel processing
        if array_size > 10000 {
            return unique_optimized_large(
                &flat_data,
                array_size,
                return_index,
                return_inverse,
                return_counts,
            );
        }

        // For smaller arrays, use a more efficient sequential approach with pre-allocation
        let estimated_capacity = estimate_capacity(array_size);

        // Pre-allocate with estimated capacity
        let mut unique_elements = Vec::with_capacity(estimated_capacity);
        let mut first_indices = if return_index.unwrap_or(false) {
            Vec::with_capacity(estimated_capacity)
        } else {
            Vec::new()
        };
        let mut inverse_indices = if return_inverse.unwrap_or(false) {
            vec![0; array_size]
        } else {
            Vec::new()
        };
        let need_inverse = return_inverse.unwrap_or(false);

        // Use HashMap with capacity hint
        let mut value_to_index = HashMap::with_capacity(estimated_capacity);

        // Process each element
        for (i, value) in flat_data.iter().enumerate() {
            if let Some(&idx) = value_to_index.get(value) {
                // Element already seen
                if need_inverse {
                    inverse_indices[i] = idx;
                }
            } else {
                // New unique element
                let new_idx = unique_elements.len();
                unique_elements.push(value.clone());
                if return_index.unwrap_or(false) {
                    first_indices.push(i);
                }
                value_to_index.insert(value, new_idx);
                if need_inverse {
                    inverse_indices[i] = new_idx;
                }
            }
        }

        // Calculate counts if needed
        let counts = if return_counts.unwrap_or(false) {
            let mut counts_vec = vec![0; unique_elements.len()];
            if need_inverse {
                // If we already computed inverse indices, use them
                for &idx in &inverse_indices {
                    counts_vec[idx] += 1;
                }
            } else {
                // Otherwise, count directly from original data
                for value in flat_data.iter() {
                    let idx = *value_to_index
                        .get(value)
                        .expect("value must exist in value_to_index map");
                    counts_vec[idx] += 1;
                }
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
    let axis_val = axis.expect("axis must be Some at this point (None case handled above)");
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
        return unique_optimized(a, None, return_index, return_inverse, return_counts);
    }

    // For higher dimensions, we need to find unique subarrays along the specified axis

    // Get the size of the axis and calculate the shape of each subarray
    let axis_len = shape[axis_val];

    // Optimize memory allocation for subarrays
    let mut subarrays = Vec::with_capacity(axis_len);
    let mut subarray_hashes = Vec::with_capacity(axis_len);

    // Extract subarrays along the specified axis
    for i in 0..axis_len {
        // Get the subarray
        let subarray = a.slice(axis_val, i)?;

        // Convert to a hashable representation
        let hash_rep = subarray.to_vec();

        subarrays.push(subarray);
        subarray_hashes.push(hash_rep);
    }

    // Estimate capacity for unique subarrays
    let estimated_capacity = estimate_capacity(axis_len);

    // Find unique subarrays with pre-allocation
    let mut unique_indices = Vec::with_capacity(estimated_capacity);
    let mut index_map = HashMap::with_capacity(estimated_capacity);
    let mut inverse = if return_inverse.unwrap_or(false) {
        vec![0; axis_len]
    } else {
        Vec::new()
    };
    let need_inverse = return_inverse.unwrap_or(false);
    let mut seen = HashSet::with_capacity(estimated_capacity);

    for i in 0..axis_len {
        let hash_rep = &subarray_hashes[i];

        if !seen.contains(hash_rep) {
            // This is a new unique subarray
            let idx = unique_indices.len();
            unique_indices.push(i);
            index_map.insert(hash_rep, idx);
            seen.insert(hash_rep.clone());
            if need_inverse {
                inverse[i] = idx;
            }
        } else {
            // This subarray has been seen before
            if need_inverse {
                let idx = *index_map
                    .get(hash_rep)
                    .expect("hash_rep must exist in index_map for seen subarrays");
                inverse[i] = idx;
            }
        }
    }

    // Calculate counts if needed
    let counts = if return_counts.unwrap_or(false) {
        let mut counts_vec = vec![0; unique_indices.len()];
        if need_inverse {
            for &idx in &inverse {
                counts_vec[idx] += 1;
            }
        } else {
            for hash_rep in &subarray_hashes {
                if let Some(&idx) = index_map.get(hash_rep) {
                    counts_vec[idx] += 1;
                }
            }
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
        Array::from_vec_shape(unique_data, &output_shape)?
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

// Special optimized implementation for large arrays using parallel processing
fn unique_optimized_large<T>(
    flat_data: &[T],
    array_size: usize,
    return_index: Option<bool>,
    return_inverse: Option<bool>,
    return_counts: Option<bool>,
) -> Result<UniqueResult<T>>
where
    T: Clone + Hash + Eq + Debug + Send + Sync,
{
    let need_index = return_index.unwrap_or(false);
    let need_inverse = return_inverse.unwrap_or(false);
    let need_counts = return_counts.unwrap_or(false);

    // Use atomic counter for thread-safe indexing
    let unique_counter = AtomicUsize::new(0);

    // Create a shared HashMap for value-to-index mapping
    // Using estimated capacity heuristic
    let estimated_capacity = (array_size as f64 * 0.9) as usize;
    let value_to_index = std::sync::RwLock::new(HashMap::with_capacity(estimated_capacity));

    // First pass: identify unique elements and assign indices
    let mut unique_elements = Vec::with_capacity(estimated_capacity);
    let mut first_indices = if need_index {
        Vec::with_capacity(estimated_capacity)
    } else {
        Vec::new()
    };

    // Using batched processing for better performance
    let batch_size = std::cmp::max(1, array_size / scirs2_core::parallel_ops::num_threads());
    let batches = flat_data.chunks(batch_size);

    // Process each batch for unique values
    batches.enumerate().for_each(|(batch_idx, batch)| {
        let mut local_uniques = HashMap::new();
        let base_index = batch_idx * batch_size;

        // First pass within batch to find local unique elements
        for (local_idx, value) in batch.iter().enumerate() {
            let global_idx = base_index + local_idx;
            if !local_uniques.contains_key(value) {
                local_uniques.insert(value.clone(), global_idx);
            }
        }

        // Second pass: merge with global unique set
        let mut value_map = value_to_index
            .write()
            .expect("value_to_index RwLock poisoned: failed to acquire write lock");
        for (value, local_first_idx) in local_uniques {
            if !value_map.contains_key(&value) {
                let new_idx = unique_counter.fetch_add(1, Ordering::SeqCst);
                value_map.insert(value.clone(), new_idx);

                // This is thread-safe because each value is processed only by the thread that first discovers it
                synchronized_push(&mut unique_elements, value);
                if need_index {
                    synchronized_push(&mut first_indices, local_first_idx);
                }
            }
        }
    });

    // Create inverse indices if needed
    let inverse_indices = if need_inverse {
        let value_map = value_to_index
            .read()
            .expect("value_to_index RwLock poisoned: failed to acquire read lock");
        flat_data
            .par_iter()
            .map(|value| {
                *value_map
                    .get(value)
                    .expect("value must exist in value_map for inverse indices")
            })
            .collect()
    } else {
        Vec::new()
    };

    // Calculate counts if needed
    let counts = if need_counts {
        let mut counts_vec = vec![0; unique_elements.len()];

        if need_inverse {
            // If we already have inverse indices, use them
            for &idx in &inverse_indices {
                counts_vec[idx] += 1;
            }
        } else {
            // Otherwise, count directly using value map
            let value_map = value_to_index
                .read()
                .expect("value_to_index RwLock poisoned: failed to acquire read lock");

            // Use thread-local counters and then merge
            let local_counts = flat_data
                .par_iter()
                .map(|value| {
                    let idx = *value_map
                        .get(value)
                        .expect("value must exist in value_map for counting");
                    (idx, 1)
                })
                .collect::<Vec<(usize, usize)>>();

            // Aggregate counts
            for (idx, count) in local_counts {
                counts_vec[idx] += count;
            }
        }

        Some(Array::from_vec(counts_vec))
    } else {
        None
    };

    // Construct the result
    let unique_array = Array::from_vec(unique_elements);

    Ok(UniqueResult {
        values: unique_array,
        indices: if need_index {
            Some(Array::from_vec(first_indices))
        } else {
            None
        },
        inverse: if need_inverse {
            Some(Array::from_vec(inverse_indices))
        } else {
            None
        },
        counts,
    })
}

// Helper function for thread-safe vector push
fn synchronized_push<T: Send + Clone>(vec: &mut Vec<T>, value: T) {
    // For simplicity, using a mutex, but a lock-free implementation would be better
    // in a production environment
    let mutex = std::sync::Mutex::new(());
    let guard = mutex.lock().expect("Mutex poisoned in synchronized_push");
    vec.push(value);
    drop(guard);
}

// Helper function for atomic increment (unused but kept for reference)
// fn synchronized_increment(vec: &mut [usize], idx: usize) {
//     // Using atomic operations for thread safety
//     let ptr = &mut vec[idx] as *mut usize;
//     // This is safe because we're incrementing a single unique index per thread
//     unsafe {
//         let atomic_ref = AtomicUsize::from_ptr(ptr);
//         atomic_ref.fetch_add(1, Ordering::Relaxed);
//     }
// }

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
    pub fn values_indices_inverse_counts(self) -> Result<crate::unique::UniqueTuple<T>> {
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

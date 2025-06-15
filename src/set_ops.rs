//! Set operations for arrays
//!
//! This module provides set-theoretic operations on arrays, similar to NumPy's set operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Type alias for unique function return values
/// Returns (unique_values, indices, inverse_indices, counts)
type UniqueResult<T> = (Array<T>, Array<usize>, Array<usize>, Array<usize>);

/// Find the intersection of two arrays
///
/// Return the sorted, unique values that are in both input arrays.
///
/// # Parameters
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// A new array containing the intersection of the two input arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 3, 4, 3]);
/// let b = Array::from_vec(vec![3, 1, 2, 1]);
/// let result = intersect1d(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 3]);
/// ```
pub fn intersect1d<T: Clone + Eq + Hash + Ord>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>> {
    let set1: HashSet<T> = ar1.to_vec().into_iter().collect();
    let set2: HashSet<T> = ar2.to_vec().into_iter().collect();

    let mut intersection: Vec<T> = set1.intersection(&set2).cloned().collect();
    intersection.sort();

    Ok(Array::from_vec(intersection))
}

/// Find the union of two arrays
///
/// Return the unique, sorted array of values that are in either of the two input arrays.
///
/// # Parameters
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// A new array containing the union of the two input arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 2, 4]);
/// let b = Array::from_vec(vec![2, 3, 5, 7, 5]);
/// let result = union1d(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 7]);
/// ```
pub fn union1d<T: Clone + Eq + Hash + Ord>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>> {
    let set1: HashSet<T> = ar1.to_vec().into_iter().collect();
    let set2: HashSet<T> = ar2.to_vec().into_iter().collect();

    let mut union: Vec<T> = set1.union(&set2).cloned().collect();
    union.sort();

    Ok(Array::from_vec(union))
}

/// Find the set difference of two arrays
///
/// Return the unique values in ar1 that are not in ar2.
///
/// # Parameters
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// A new array containing values in ar1 that are not in ar2
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 2, 4, 1]);
/// let b = Array::from_vec(vec![3, 4, 5, 6]);
/// let result = setdiff1d(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 2]);
/// ```
pub fn setdiff1d<T: Clone + Eq + Hash + Ord>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>> {
    let set1: HashSet<T> = ar1.to_vec().into_iter().collect();
    let set2: HashSet<T> = ar2.to_vec().into_iter().collect();

    let mut difference: Vec<T> = set1.difference(&set2).cloned().collect();
    difference.sort();

    Ok(Array::from_vec(difference))
}

/// Find the set exclusive-or of two arrays
///
/// Return the sorted, unique values that are in only one (not both) of the input arrays.
///
/// # Parameters
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// A new array containing the symmetric difference of the two input arrays
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 2, 4]);
/// let b = Array::from_vec(vec![2, 3, 5, 7, 5]);
/// let result = setxor1d(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 4, 5, 7]);
/// ```
pub fn setxor1d<T: Clone + Eq + Hash + Ord>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>> {
    let set1: HashSet<T> = ar1.to_vec().into_iter().collect();
    let set2: HashSet<T> = ar2.to_vec().into_iter().collect();

    let mut symmetric_diff: Vec<T> = set1.symmetric_difference(&set2).cloned().collect();
    symmetric_diff.sort();

    Ok(Array::from_vec(symmetric_diff))
}

/// Test whether each element of a 1-D array is also present in a second array
///
/// Returns a boolean array the same length as ar1 that is True
/// where an element of ar1 is in ar2 and False otherwise.
///
/// # Parameters
///
/// * `ar1` - Input array
/// * `ar2` - Array of values against which to test each value of ar1
///
/// # Returns
///
/// A boolean array with the same shape as ar1
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
/// let b = Array::from_vec(vec![2, 4, 6]);
/// let result = in1d(&a, &b).unwrap();
/// assert_eq!(result.to_vec(), vec![false, true, false, true, false, true]);
/// ```
pub fn in1d<T: Clone + Eq + Hash>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<bool>> {
    let set2: HashSet<T> = ar2.to_vec().into_iter().collect();
    let result: Vec<bool> = ar1.to_vec().iter().map(|x| set2.contains(x)).collect();

    Ok(Array::from_vec(result))
}

/// Calculates `element in test_elements`, broadcasting over `element` only
///
/// Returns a boolean array of the same shape as `element` that is True
/// where an element of `element` is in `test_elements` and False otherwise.
///
/// # Parameters
///
/// * `element` - Input array
/// * `test_elements` - Array of values against which to test each value of element
/// * `assume_unique` - If True, the input arrays are both assumed to be unique,
///   which can speed up the calculation. Default: False
/// * `invert` - If True, the returned boolean array is inverted (False where an
///   element of element is in test_elements). Default: False
///
/// # Returns
///
/// A boolean array with the same shape as element
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let element = Array::from_vec(vec![0, 1, 2, 5, 0]).reshape(&[5]);
/// let test_elements = Array::from_vec(vec![0, 2, 5, 7, 9]);
/// let result = isin(&element, &test_elements, false, false).unwrap();
/// assert_eq!(result.to_vec(), vec![true, false, true, true, true]);
///
/// // With invert=true
/// let result_inv = isin(&element, &test_elements, false, true).unwrap();
/// assert_eq!(result_inv.to_vec(), vec![false, true, false, false, false]);
/// ```
pub fn isin<T: Clone + Eq + Hash>(
    element: &Array<T>,
    test_elements: &Array<T>,
    assume_unique: bool,
    invert: bool,
) -> Result<Array<bool>> {
    let test_set: HashSet<T> = if assume_unique {
        // If we assume uniqueness, we can skip the deduplication step
        test_elements.to_vec().into_iter().collect()
    } else {
        test_elements.to_vec().into_iter().collect()
    };

    let result: Vec<bool> = element
        .to_vec()
        .iter()
        .map(|x| {
            let contains = test_set.contains(x);
            if invert {
                !contains
            } else {
                contains
            }
        })
        .collect();

    let result_array = Array::from_vec(result);

    // Preserve the original shape
    Ok(result_array.reshape(&element.shape()))
}

/// Find unique elements and their indices
///
/// # Parameters
///
/// * `ar` - Input array
/// * `return_index` - If True, also return the indices of ar (along the specified axis) that
///   result in the unique array
/// * `return_inverse` - If True, also return the indices to reconstruct the original array
/// * `return_counts` - If True, also return the number of times each unique item appears
///
/// # Returns
///
/// A tuple containing:
/// - unique: The sorted unique values
/// - unique_indices: The indices of the first occurrences of the unique values (if return_index=True)
/// - unique_inverse: The indices to reconstruct the original array (if return_inverse=True)
/// - unique_counts: The number of times each unique value appears (if return_counts=True)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 1, 2, 2, 3, 3]);
/// let (unique, indices, inverse, counts) = unique_with_options(&a, true, true, true).unwrap();
/// assert_eq!(unique.to_vec(), vec![1, 2, 3]);
/// assert_eq!(indices.to_vec(), vec![0, 2, 4]);
/// assert_eq!(inverse.to_vec(), vec![0, 0, 1, 1, 2, 2]);
/// assert_eq!(counts.to_vec(), vec![2, 2, 2]);
/// ```
pub fn unique_with_options<T: Clone + Eq + Hash + Ord + Debug>(
    ar: &Array<T>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> Result<UniqueResult<T>> {
    let data = ar.to_vec();
    let mut seen = std::collections::HashMap::new();
    let mut unique_values = Vec::new();
    let mut first_indices = Vec::new();
    let _counts: Vec<usize> = Vec::new();

    // Build unique values and track their first occurrences and counts
    for (i, value) in data.iter().enumerate() {
        if let Some((_first_idx, count)) = seen.get_mut(value) {
            *count += 1;
        } else {
            seen.insert(value.clone(), (i, 1));
            unique_values.push(value.clone());
            first_indices.push(i);
        }
    }

    // Sort by the unique values to get consistent ordering
    let mut sorted_indices: Vec<usize> = (0..unique_values.len()).collect();
    sorted_indices.sort_by(|&a, &b| unique_values[a].cmp(&unique_values[b]));

    let sorted_unique: Vec<T> = sorted_indices
        .iter()
        .map(|&i| unique_values[i].clone())
        .collect();

    // Build the outputs
    let unique_array = Array::from_vec(sorted_unique.clone());

    let indices_array = if return_index {
        let sorted_first_indices: Vec<usize> =
            sorted_indices.iter().map(|&i| first_indices[i]).collect();
        Array::from_vec(sorted_first_indices)
    } else {
        Array::from_vec(vec![])
    };

    let inverse_array = if return_inverse {
        // Create a mapping from unique values to their positions in the sorted unique array
        let value_to_pos: std::collections::HashMap<&T, usize> = sorted_unique
            .iter()
            .enumerate()
            .map(|(pos, val)| (val, pos))
            .collect();

        let inverse: Vec<usize> = data
            .iter()
            .map(|val| *value_to_pos.get(val).unwrap())
            .collect();

        Array::from_vec(inverse)
    } else {
        Array::from_vec(vec![])
    };

    let counts_array = if return_counts {
        let sorted_counts: Vec<usize> = sorted_unique
            .iter()
            .map(|val| seen.get(val).unwrap().1)
            .collect();
        Array::from_vec(sorted_counts)
    } else {
        Array::from_vec(vec![])
    };

    Ok((unique_array, indices_array, inverse_array, counts_array))
}

/// Find unique elements with multi-dimensional support and axis parameter
///
/// # Parameters
///
/// * `ar` - Input array
/// * `axis` - The axis along which to find unique values. If None, the array is flattened first
/// * `return_index` - If True, also return the indices of ar that result in the unique array
/// * `return_inverse` - If True, also return the indices to reconstruct the original array
/// * `return_counts` - If True, also return the number of times each unique item appears
///
/// # Returns
///
/// A tuple containing:
/// - unique: The unique values
/// - unique_indices: The indices of the first occurrences (if return_index=True)
/// - unique_inverse: The indices to reconstruct the original array (if return_inverse=True)
/// - unique_counts: The number of times each unique value appears (if return_counts=True)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // 1D case (same as unique_with_options)
/// let a = Array::from_vec(vec![3, 1, 2, 1, 3, 2, 1]);
/// let (unique, indices, inverse, counts) = unique_axis(&a, None, true, true, true).unwrap();
/// assert_eq!(unique.to_vec(), vec![1, 2, 3]);
///
/// // 2D case along axis 0 (find unique rows)
/// let b = Array::from_vec(vec![1, 2, 3, 1, 2, 3, 4, 5, 6]).reshape(&[3, 3]);
/// let (unique_rows, _, _, _) = unique_axis(&b, Some(0), false, false, false).unwrap();
/// assert_eq!(unique_rows.shape(), vec![2, 3]); // Two unique rows
///
/// // 2D case along axis 1 (find unique columns)
/// let c = Array::from_vec(vec![1, 2, 1, 3, 4, 3]).reshape(&[2, 3]);
/// let (unique_cols, _, _, _) = unique_axis(&c, Some(1), false, false, false).unwrap();
/// assert_eq!(unique_cols.shape(), vec![2, 2]); // Two unique columns
/// ```
pub fn unique_axis<T: Clone + Eq + Hash + Ord + Debug + num_traits::Zero>(
    ar: &Array<T>,
    axis: Option<usize>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> Result<UniqueResult<T>> {
    match axis {
        None => {
            // No axis specified, flatten the array and use existing unique_with_options
            unique_with_options(ar, return_index, return_inverse, return_counts)
        }
        Some(ax) => {
            if ax >= ar.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} is out of bounds for array with {} dimensions",
                    ax,
                    ar.ndim()
                )));
            }

            let shape = ar.shape();
            let axis_dim = shape[ax];

            // Extract slices along the specified axis
            let mut slices = Vec::with_capacity(axis_dim);
            for i in 0..axis_dim {
                // Create index specs to select along the axis
                let mut index_specs = vec![crate::indexing::IndexSpec::All; ar.ndim()];
                index_specs[ax] = crate::indexing::IndexSpec::Index(i);

                // Get the slice at this index
                let slice = ar.index(&index_specs)?;
                slices.push(slice);
            }

            // Convert slices to comparable form (vectors)
            let slice_vecs: Vec<Vec<T>> = slices.iter().map(|s| s.to_vec()).collect();

            // Track unique slices and their metadata
            let mut seen = std::collections::HashMap::new();
            let mut unique_slices = Vec::new();
            let mut first_indices = Vec::new();

            // Find unique slices
            for (i, slice_vec) in slice_vecs.iter().enumerate() {
                if let Some((_first_idx, count)) = seen.get_mut::<Vec<T>>(slice_vec) {
                    *count += 1;
                } else {
                    seen.insert(slice_vec.clone(), (i, 1));
                    unique_slices.push(slices[i].clone());
                    first_indices.push(i);
                }
            }

            // Sort by lexicographic order of the slice data
            let mut sorted_indices: Vec<usize> = (0..unique_slices.len()).collect();
            sorted_indices.sort_by(|&a, &b| {
                let slice_a = unique_slices[a].to_vec();
                let slice_b = unique_slices[b].to_vec();
                slice_a.cmp(&slice_b)
            });

            let sorted_unique_slices: Vec<Array<T>> = sorted_indices
                .iter()
                .map(|&i| unique_slices[i].clone())
                .collect();

            // Build the unique array by stacking unique slices
            let unique_array = if !sorted_unique_slices.is_empty() {
                crate::array_ops::stack(&sorted_unique_slices.iter().collect::<Vec<_>>(), ax)?
            } else {
                // Handle empty case
                let mut empty_shape = shape.clone();
                empty_shape[ax] = 0;
                Array::zeros(&empty_shape)
            };

            // Build indices array
            let indices_array = if return_index {
                let sorted_first_indices: Vec<usize> =
                    sorted_indices.iter().map(|&i| first_indices[i]).collect();
                Array::from_vec(sorted_first_indices)
            } else {
                Array::from_vec(vec![])
            };

            // Build inverse array
            let inverse_array = if return_inverse {
                // Create mapping from slice to position in sorted unique array
                let mut slice_to_pos = std::collections::HashMap::new();
                for (pos, &sorted_idx) in sorted_indices.iter().enumerate() {
                    let slice_vec = unique_slices[sorted_idx].to_vec();
                    slice_to_pos.insert(slice_vec, pos);
                }

                let inverse: Vec<usize> = slice_vecs
                    .iter()
                    .map(|slice_vec| *slice_to_pos.get(slice_vec).unwrap())
                    .collect();

                Array::from_vec(inverse)
            } else {
                Array::from_vec(vec![])
            };

            // Build counts array
            let counts_array = if return_counts {
                let sorted_counts: Vec<usize> = sorted_indices
                    .iter()
                    .map(|&i| {
                        let slice_vec = unique_slices[i].to_vec();
                        seen.get(&slice_vec).unwrap().1
                    })
                    .collect();
                Array::from_vec(sorted_counts)
            } else {
                Array::from_vec(vec![])
            };

            Ok((unique_array, indices_array, inverse_array, counts_array))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect1d() {
        let a = Array::from_vec(vec![1, 3, 4, 3]);
        let b = Array::from_vec(vec![3, 1, 2, 1]);
        let result = intersect1d(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_union1d() {
        let a = Array::from_vec(vec![1, 2, 3, 2, 4]);
        let b = Array::from_vec(vec![2, 3, 5, 7, 5]);
        let result = union1d(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 7]);
    }

    #[test]
    fn test_setdiff1d() {
        let a = Array::from_vec(vec![1, 2, 3, 2, 4, 1]);
        let b = Array::from_vec(vec![3, 4, 5, 6]);
        let result = setdiff1d(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![1, 2]);
    }

    #[test]
    fn test_setxor1d() {
        let a = Array::from_vec(vec![1, 2, 3, 2, 4]);
        let b = Array::from_vec(vec![2, 3, 5, 7, 5]);
        let result = setxor1d(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![1, 4, 5, 7]);
    }

    #[test]
    fn test_in1d() {
        let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let b = Array::from_vec(vec![2, 4, 6]);
        let result = in1d(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![false, true, false, true, false, true]);
    }

    #[test]
    fn test_isin() {
        let element = Array::from_vec(vec![0, 1, 2, 5, 0]);
        let test_elements = Array::from_vec(vec![0, 2, 5, 7, 9]);
        let result = isin(&element, &test_elements, false, false).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, true, true, true]);

        // With invert=true
        let result_inv = isin(&element, &test_elements, false, true).unwrap();
        assert_eq!(result_inv.to_vec(), vec![false, true, false, false, false]);
    }

    #[test]
    fn test_unique_with_options() {
        let a = Array::from_vec(vec![1, 1, 2, 2, 3, 3]);
        let (unique, indices, inverse, counts) = unique_with_options(&a, true, true, true).unwrap();
        assert_eq!(unique.to_vec(), vec![1, 2, 3]);
        assert_eq!(indices.to_vec(), vec![0, 2, 4]);
        assert_eq!(inverse.to_vec(), vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(counts.to_vec(), vec![2, 2, 2]);
    }
}

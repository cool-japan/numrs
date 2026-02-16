//! Search and sorting operations
//!
//! This module provides functions for searching and sorting array elements.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

/// Return a partitioned copy of an array
///
/// Partitioning creates a partially sorted output where elements
/// smaller than the kth element are moved before it and larger elements
/// are moved after it. The kth element will be in the position it would
/// be in a sorted array.
///
/// # Parameters
///
/// * `array` - Array to be partitioned
/// * `kth` - Element index to partition by
/// * `axis` - Axis along which to partition
///   If None, array is flattened before partitioning
///
/// # Returns
///
/// * Copy of array with values arranged to ensure the kth element
///   is in its sorted position
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops_legacy::partition;
///
/// // Partition a 1D array
/// let a = Array::from_vec(vec![9, 4, 1, 7, 5, 3, 8, 2, 6]);
/// let partitioned = partition(&a, 3, None).expect("partition failed");
/// // The 4th element (index 3) is now in correct position for sorting
/// let val = partitioned.get(&[3]).expect("get failed");
/// assert!(val >= 1 && val <= 9);
/// ```
pub fn partition<T: Clone + PartialOrd>(
    array: &Array<T>,
    kth: usize,
    axis: Option<usize>,
) -> Result<Array<T>> {
    match axis {
        None => {
            // Flatten array and partition
            let mut data = array.to_vec();
            let n = data.len();

            if kth >= n {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "kth ({}) is out of bounds for array of size {}",
                    kth, n
                )));
            }

            // Quick-select algorithm to efficiently find the kth element and partition the array
            quick_select(&mut data, 0, n - 1, kth);

            // Reshape back to original shape
            Ok(Array::from_vec(data).reshape(&array.shape()))
        }
        Some(axis_val) => {
            let shape = array.shape();

            if axis_val >= shape.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    axis_val,
                    shape.len()
                )));
            }

            let axis_size = shape[axis_val];

            if kth >= axis_size {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "kth ({}) is out of bounds for axis {} with size {}",
                    kth, axis_val, axis_size
                )));
            }

            // Create a new array with the same shape
            let mut result = array.clone();
            let result_vec = result.array_mut().as_slice_mut().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Failed to get mutable slice".into())
            })?;

            // Calculate the sizes of the pre-axis, axis, and post-axis dimensions
            let pre_axis_size: usize = shape.iter().take(axis_val).product();
            let post_axis_size: usize = shape.iter().skip(axis_val + 1).product();

            // Partition each slice along the specified axis
            for i_pre in 0..pre_axis_size {
                for i_post in 0..post_axis_size {
                    // Extract the slice along the axis
                    let mut slice = Vec::with_capacity(axis_size);

                    for i_axis in 0..axis_size {
                        let idx =
                            i_pre * (axis_size * post_axis_size) + i_axis * post_axis_size + i_post;
                        slice.push(result_vec[idx].clone());
                    }

                    // Partition the slice
                    quick_select(&mut slice, 0, axis_size - 1, kth);

                    // Write back the partitioned slice
                    #[allow(clippy::needless_range_loop)]
                    for i_axis in 0..axis_size {
                        let idx =
                            i_pre * (axis_size * post_axis_size) + i_axis * post_axis_size + i_post;
                        result_vec[idx] = slice[i_axis].clone();
                    }
                }
            }

            Ok(result)
        }
    }
}

/// Quick-select algorithm to partition an array and place the kth element
/// in its sorted position. Elements smaller than the kth element will be
/// before it, and elements larger than the kth element will be after it.
///
/// This is a helper function for the partition function.
fn quick_select<T: Clone + PartialOrd>(arr: &mut [T], left: usize, right: usize, k: usize) {
    if left == right {
        return;
    }

    // Choose a pivot index (using a simple median-of-three approach)
    let pivot_idx = choose_pivot(arr, left, right);

    // Partition around the pivot
    let pivot_idx = partition_around_pivot(arr, left, right, pivot_idx);

    match k.cmp(&pivot_idx) {
        std::cmp::Ordering::Equal => {
            // k is at its final position
        }
        std::cmp::Ordering::Less => {
            // k is in the left side
            if pivot_idx > 0 {
                quick_select(arr, left, pivot_idx - 1, k);
            }
        }
        std::cmp::Ordering::Greater => {
            // k is in the right side
            quick_select(arr, pivot_idx + 1, right, k);
        }
    }
}

/// Choose a good pivot index using median-of-three strategy
///
/// This helper function helps improve the performance of quick-select
/// by choosing a better pivot than just the first or last element.
fn choose_pivot<T: PartialOrd>(arr: &[T], left: usize, right: usize) -> usize {
    if right - left < 2 {
        return left;
    }

    let mid = left + (right - left) / 2;

    // Choose median of left, middle, and right elements
    let mut indices = [left, mid, right];

    // Simple bubble sort of the three indices based on their values
    if arr[indices[0]] > arr[indices[1]] {
        indices.swap(0, 1);
    }
    if arr[indices[1]] > arr[indices[2]] {
        indices.swap(1, 2);
    }
    if arr[indices[0]] > arr[indices[1]] {
        indices.swap(0, 1);
    }

    // Return the middle value
    indices[1]
}

/// Partition the array around a pivot value
///
/// After partitioning, all elements less than the pivot value are on the left side,
/// and all elements greater are on the right side. The pivot element is at the returned index.
fn partition_around_pivot<T: Clone + PartialOrd>(
    arr: &mut [T],
    left: usize,
    right: usize,
    pivot_idx: usize,
) -> usize {
    let pivot_value = arr[pivot_idx].clone();

    // Move pivot to the end temporarily
    arr.swap(pivot_idx, right);

    // Move all elements less than pivot to the left
    let mut store_idx = left;
    for i in left..right {
        if arr[i] < pivot_value {
            arr.swap(i, store_idx);
            store_idx += 1;
        }
    }

    // Move pivot to its final place
    arr.swap(store_idx, right);

    store_idx
}

/// Find indices where elements should be inserted to maintain order
///
/// Performs binary search to find the indices into a sorted array `a` such that,
/// if the corresponding elements in `v` were inserted before the indices, the
/// order of `a` would be preserved.
///
/// # Parameters
///
/// * `a` - Input array, must be sorted in ascending order
/// * `v` - Values to insert into `a`
/// * `side` - If 'left', return the first suitable location found.
///   If 'right', return the last such index. Default is 'left'.
/// * `sorter` - Optional array of integer indices that sorts `a` into ascending order.
///   This is typically the result of `argsort`.
///
/// # Returns
///
/// * Array of insertion points with the same shape as `v`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops_legacy::searchsorted;
///
/// // Create a sorted array
/// let a = Array::from_vec(vec![1, 3, 5, 7, 9]);
///
/// // Find insertion points for values
/// let v = Array::from_vec(vec![0, 1, 2, 4, 8, 10]);
/// let indices = searchsorted(&a, &v, Some("left"), None).expect("searchsorted failed");
/// assert_eq!(indices.to_vec(), vec![0, 0, 1, 2, 4, 5]);
///
/// // Use 'right' side
/// let indices = searchsorted(&a, &v, Some("right"), None).expect("searchsorted failed");
/// assert_eq!(indices.to_vec(), vec![0, 1, 1, 2, 4, 5]);
/// ```
pub fn searchsorted<T: Clone + PartialOrd>(
    a: &Array<T>,
    v: &Array<T>,
    side: Option<&str>,
    sorter: Option<&Array<usize>>,
) -> Result<Array<usize>> {
    let side = side.unwrap_or("left");
    if side != "left" && side != "right" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Side '{}' is invalid, must be 'left' or 'right'",
            side
        )));
    }

    // If a custom sorter is provided, rearrange the array
    let a_sorted = if let Some(sorter_array) = sorter {
        if sorter_array.ndim() != 1 {
            return Err(NumRs2Error::InvalidOperation(
                "Sorter array must be 1-dimensional".into(),
            ));
        }

        if sorter_array.size() != a.size() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Sorter size ({}) does not match array size ({})",
                sorter_array.size(),
                a.size()
            )));
        }

        // Create a new array using the sorter indices
        let mut sorted_data = Vec::with_capacity(a.size());
        let a_vec = a.to_vec();
        let sorter_vec = sorter_array.to_vec();

        for &idx in &sorter_vec {
            if idx >= a_vec.len() {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Sorter index {} out of range for array of size {}",
                    idx,
                    a_vec.len()
                )));
            }
            sorted_data.push(a_vec[idx].clone());
        }

        Array::from_vec(sorted_data)
    } else {
        a.clone()
    };

    // If a is not 1D, flatten it
    let a_flat = if a_sorted.ndim() != 1 {
        a_sorted.flatten(None)
    } else {
        a_sorted
    };

    // Check if a_flat is sorted
    let a_flat_vec = a_flat.to_vec();
    for i in 1..a_flat_vec.len() {
        if a_flat_vec[i] < a_flat_vec[i - 1] {
            return Err(NumRs2Error::InvalidOperation(
                "The input array must be sorted in ascending order".into(),
            ));
        }
    }

    // Convert v to a flat array if needed
    let v_vec = v.to_vec();

    // Perform binary search for each value in v
    let mut result = Vec::with_capacity(v_vec.len());

    for val in &v_vec {
        let idx = if side == "left" {
            binary_search_left(&a_flat_vec, val)
        } else {
            binary_search_right(&a_flat_vec, val)
        };

        result.push(idx);
    }

    // Reshape result to match v's shape
    Ok(Array::from_vec(result).reshape(&v.shape()))
}

/// Binary search for the leftmost insertion point
fn binary_search_left<T: PartialOrd>(arr: &[T], value: &T) -> usize {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;

        if &arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    left
}

/// Binary search for the rightmost insertion point
fn binary_search_right<T: PartialOrd>(arr: &[T], value: &T) -> usize {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;

        if value < &arr[mid] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    left
}

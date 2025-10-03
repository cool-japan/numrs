use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{NumCast, ToPrimitive, Zero};
use scirs2_core::Complex;

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
/// use numrs2::array_ops::sorting::partition;
///
/// // Partition a 1D array
/// let a = Array::from_vec(vec![9.0, 4.0, 1.0, 7.0, 5.0, 3.0, 8.0, 2.0, 6.0]);
/// let partitioned = partition(&a, 3, None).unwrap();
/// // The 4th element (index 3) is now 4.0, all elements before are <= 4.0,
/// // and all elements after are >= 4.0
/// assert_eq!(partitioned.get(&[3]).unwrap(), 4.0);
/// for i in 0..3 {
///     assert!(partitioned.get(&[i]).unwrap() <= 4.0);
/// }
/// for i in 4..9 {
///     assert!(partitioned.get(&[i]).unwrap() >= 4.0);
/// }
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
/// use numrs2::array_ops::sorting::searchsorted;
///
/// // Create a sorted array
/// let a = Array::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);
///
/// // Find insertion points for values
/// let v = Array::from_vec(vec![0.0, 1.0, 2.0, 4.0, 8.0, 10.0]);
/// let indices = searchsorted(&a, &v, Some("left"), None).unwrap();
/// assert_eq!(indices.to_vec(), vec![0, 0, 1, 2, 4, 5]);
///
/// // Use 'right' side
/// let indices = searchsorted(&a, &v, Some("right"), None).unwrap();
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

/// Generic sorting function that can handle both real and complex numbers
///
/// This is a convenience function that automatically chooses the appropriate
/// sorting algorithm based on the data type.
///
/// # Parameters
///
/// * `array` - Input array to sort
/// * `kind` - Sort algorithm: "quicksort", "mergesort", or "heapsort"
///   If None, uses "quicksort" as default
///
/// # Returns
///
/// * Sorted copy of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::sorting::sort;
///
/// let a = Array::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
/// let sorted = sort(&a, Some("mergesort")).unwrap();
/// assert_eq!(sorted.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
/// ```
pub fn sort<T: Clone + PartialOrd>(array: &Array<T>, kind: Option<&str>) -> Result<Array<T>> {
    let sort_kind = kind.unwrap_or("quicksort");

    match sort_kind {
        "mergesort" => msort(array),
        "quicksort" => {
            // Use standard library sort (which is typically introsort)
            let mut data = array.to_vec();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(Array::from_vec(data).reshape(&array.shape()))
        }
        "heapsort" => {
            // Implement heap sort
            let mut data = array.to_vec();
            heap_sort(&mut data);
            Ok(Array::from_vec(data).reshape(&array.shape()))
        }
        _ => Err(NumRs2Error::InvalidOperation(format!(
            "Unknown sort kind: {}. Must be 'quicksort', 'mergesort', or 'heapsort'",
            sort_kind
        ))),
    }
}

/// Heap sort implementation for in-place sorting
fn heap_sort<T: Clone + PartialOrd>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    // Build max heap
    for i in (0..len / 2).rev() {
        heapify(arr, len, i);
    }

    // Extract elements from heap one by one
    for i in (1..len).rev() {
        arr.swap(0, i);
        heapify(arr, i, 0);
    }
}

/// Heapify a subtree rooted at index i
fn heapify<T: Clone + PartialOrd>(arr: &mut [T], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;

    // If left child is larger than root
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }

    // If right child is larger than largest so far
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }

    // If largest is not root
    if largest != i {
        arr.swap(i, largest);
        heapify(arr, n, largest);
    }
}

/// Sort a complex array using the absolute value as the key
///
/// Complex numbers are sorted first by their absolute value (magnitude),
/// and then by their argument (angle) for numbers with the same magnitude.
/// This provides a consistent ordering for complex numbers.
///
/// # Parameters
///
/// * `array` - Input array of complex numbers to sort
///
/// # Returns
///
/// * Sorted copy of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::sorting::sort_complex;
/// use scirs2_core::Complex;
///
/// let a = Array::from_vec(vec![
///     Complex::new(3.0, 4.0),  // magnitude 5.0
///     Complex::new(1.0, 0.0),  // magnitude 1.0
///     Complex::new(0.0, 1.0),  // magnitude 1.0
///     Complex::new(2.0, 0.0),  // magnitude 2.0
/// ]);
/// let sorted = sort_complex(&a).unwrap();
/// // Should be sorted by magnitude: [1+0i, 0+1i, 2+0i, 3+4i]
/// ```
pub fn sort_complex<T>(array: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Clone + PartialOrd + num_traits::Float,
{
    // For now, implement for 1D arrays and flatten multi-dimensional arrays
    let flattened = if array.ndim() == 1 {
        array.clone()
    } else {
        array.flatten(None)
    };

    let mut data = flattened.to_vec();

    // Sort by magnitude first, then by argument for equal magnitudes
    data.sort_by(|a, b| {
        let mag_a = a.norm();
        let mag_b = b.norm();

        match mag_a
            .partial_cmp(&mag_b)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => {
                // Same magnitude, sort by argument (angle)
                let arg_a = a.arg();
                let arg_b = b.arg();
                arg_a
                    .partial_cmp(&arg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other,
        }
    });

    // Reshape back to original shape
    Ok(Array::from_vec(data).reshape(&array.shape()))
}

/// Count number of occurrences of each value in array of non-negative integers
///
/// The number of bins (of size 1) is one larger than the largest value in x.
/// If minlength is specified, there will be at least this number of bins in the
/// output array (though it will be longer if necessary, depending on the contents of x).
/// Each bin gives the number of occurrences of its index value in x.
///
/// # Parameters
///
/// * `x` - Input array of non-negative integers
/// * `weights` - Optional weights array of the same shape as x
/// * `minlength` - Minimum number of bins for output array
///
/// # Returns
///
/// Array of counts. The length of the output is equal to max(x) + 1 if x is non-empty,
/// and at least minlength.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::sorting::bincount;
///
/// // Basic bincount
/// let x = Array::from_vec(vec![0, 1, 1, 3, 2, 1, 7]);
/// let counts: Array<i32> = bincount(&x, None, None).unwrap();
/// // counts = [1, 3, 1, 1, 0, 0, 0, 1] (8 elements, up to max value 7)
/// assert_eq!(counts.shape(), vec![8]);
/// assert_eq!(counts.get(&[1]).unwrap(), 3); // value 1 appears 3 times
///
/// // With weights
/// let weights = Array::from_vec(vec![0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 2.0]);
/// let weighted_counts: Array<f64> = bincount(&x, Some(&weights), None).unwrap();
/// assert_eq!(weighted_counts.get(&[1]).unwrap(), 1.5); // sum of weights where x=1
/// ```
pub fn bincount<T, W>(
    x: &Array<T>,
    weights: Option<&Array<W>>,
    minlength: Option<usize>,
) -> Result<Array<W>>
where
    T: Clone + ToPrimitive + PartialOrd + Zero,
    W: Clone + Zero + std::ops::AddAssign + NumCast,
{
    if x.shape().len() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "bincount requires 1D input array".to_string(),
        ));
    }

    let x_data = x.to_vec();
    let _n = x_data.len();

    // Check for negative values
    for val in &x_data {
        if *val < T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "bincount requires non-negative integers".to_string(),
            ));
        }
    }

    // Find maximum value to determine output size
    let max_val = x_data
        .iter()
        .filter_map(|v| v.to_usize())
        .max()
        .unwrap_or(0);

    let output_len = std::cmp::max(max_val + 1, minlength.unwrap_or(0));

    // Initialize output array
    let mut counts = vec![W::zero(); output_len];

    if let Some(w) = weights {
        if w.shape() != x.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x.shape().to_vec(),
                actual: w.shape().to_vec(),
            });
        }

        let w_data = w.to_vec();
        for (i, val) in x_data.iter().enumerate() {
            if let Some(idx) = val.to_usize() {
                if idx < output_len {
                    counts[idx] += w_data[i].clone();
                }
            }
        }
    } else {
        // Without weights, count occurrences
        for val in &x_data {
            if let Some(idx) = val.to_usize() {
                if idx < output_len {
                    counts[idx] += W::from(1).unwrap_or(W::zero());
                }
            }
        }
    }

    Ok(Array::from_vec(counts))
}

/// Return the indices of the bins to which each value in input array belongs
///
/// Each value in x is assigned to a bin such that bin[i-1] <= x < bin[i].
/// If values in x are beyond the bounds of bins, 0 or len(bins) is returned as appropriate.
///
/// # Parameters
///
/// * `x` - Input array to be binned
/// * `bins` - Array of bin edges (must be 1-D and monotonic)
/// * `right` - If false (default), bins[i-1] <= x < bins[i]
///   If true, bins[i-1] < x <= bins[i]
///
/// # Returns
///
/// Array of indices into bins, same shape as x
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]);
/// let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
/// let indices = digitize(&x, &bins, false).unwrap();
/// // indices = [1, 4, 3, 2]
/// // 0.2 is in bin 1 [0.0, 1.0)
/// // 6.4 is in bin 4 [4.0, 10.0)
/// // 3.0 is in bin 3 [2.5, 4.0)
/// // 1.6 is in bin 2 [1.0, 2.5)
/// assert_eq!(indices.to_vec(), vec![1, 4, 3, 2]);
/// ```
pub fn digitize<T>(x: &Array<T>, bins: &Array<T>, right: bool) -> Result<Array<usize>>
where
    T: Clone + PartialOrd,
{
    if bins.shape().len() != 1 {
        return Err(NumRs2Error::InvalidOperation(
            "bins must be 1-dimensional".to_string(),
        ));
    }

    let bins_data = bins.to_vec();
    if bins_data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "bins cannot be empty".to_string(),
        ));
    }

    // Check that bins are monotonic
    let mut increasing = true;
    let mut decreasing = true;
    for i in 1..bins_data.len() {
        if bins_data[i] <= bins_data[i - 1] {
            increasing = false;
        }
        if bins_data[i] >= bins_data[i - 1] {
            decreasing = false;
        }
    }

    if !increasing && !decreasing {
        return Err(NumRs2Error::InvalidOperation(
            "bins must be monotonically increasing or decreasing".to_string(),
        ));
    }

    let x_data = x.to_vec();
    let mut indices = Vec::with_capacity(x_data.len());

    for val in x_data {
        let idx = if increasing {
            // Use binary search for increasing bins
            if right {
                binary_search_right(&bins_data, &val)
            } else {
                binary_search_left(&bins_data, &val)
            }
        } else {
            // For decreasing bins, we need to reverse the logic
            let n = bins_data.len();
            if right {
                n - binary_search_left(&bins_data.iter().rev().cloned().collect::<Vec<_>>(), &val)
            } else {
                n - binary_search_right(&bins_data.iter().rev().cloned().collect::<Vec<_>>(), &val)
            }
        };

        indices.push(idx);
    }

    Ok(Array::from_vec(indices).reshape(&x.shape()))
}

/// Perform an indirect stable sort using a sequence of keys
///
/// Given multiple sorting keys defined by the array `keys`, lexsort returns
/// an array of indices that describes the sort order by multiple columns.
/// The last key in the sequence is used as the primary sort key,
/// the second-to-last as the secondary sort key, and so on.
///
/// # Parameters
///
/// * `keys` - An array of keys to sort by. Each row is a sorting key.
///   The last row is the primary sort key.
///
/// # Returns
///
/// * Array of indices that sorts the keys along the given axis
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::sorting::lexsort;
///
/// // Sort by two keys: first by the second row (primary), then by the first row (secondary)
/// let keys = Array::from_vec(vec![
///     1, 0, 1, 0, 1,  // Secondary sort key
///     2, 2, 1, 1, 3,  // Primary sort key  
/// ]).reshape(&[2, 5]);
///
/// let indices = lexsort(&keys).unwrap();
/// // Elements will be sorted first by second row: [1,1,2,2,3]
/// // Then by first row within ties: [(1,0),(0,1),(0,2),(1,2),(1,3)]
/// assert_eq!(indices.to_vec(), vec![3, 2, 1, 0, 4]);
/// ```
pub fn lexsort<T: Clone + PartialOrd + Zero>(keys: &Array<T>) -> Result<Array<usize>> {
    let shape = keys.shape();

    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "lexsort requires a 2D array of keys".to_string(),
        ));
    }

    let n_keys = shape[0];
    let n_items = shape[1];

    if n_items == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    // Initialize indices
    let mut indices: Vec<usize> = (0..n_items).collect();

    // Sort by each key from first to last (which becomes primary to secondary)
    // This works because stable_sort_by preserves the order of equal elements
    for key_idx in 0..n_keys {
        let key_row_data: Vec<T> = (0..n_items)
            .map(|i| keys.get(&[key_idx, i]).unwrap())
            .collect();

        // Stable sort preserves relative order of equal elements
        indices.sort_by(|&a, &b| {
            key_row_data[a]
                .partial_cmp(&key_row_data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    Ok(Array::from_vec(indices))
}

/// Return a sorted copy of an array using merge sort
///
/// This is a stable sort that performs merge sort along the last axis.
/// The advantage of merge sort is that it has guaranteed O(n log n) performance
/// and is stable (maintains relative order of equal elements).
///
/// # Parameters
///
/// * `array` - Input array to sort
///
/// # Returns
///
/// * Sorted copy of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::sorting::msort;
///
/// let a = Array::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
/// let sorted = msort(&a).unwrap();
/// assert_eq!(sorted.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
/// ```
pub fn msort<T: Clone + PartialOrd>(array: &Array<T>) -> Result<Array<T>> {
    // For now, implement for 1D arrays and flatten multi-dimensional arrays
    let flattened = if array.ndim() == 1 {
        array.clone()
    } else {
        array.flatten(None)
    };

    let mut data = flattened.to_vec();
    merge_sort(&mut data);

    // Reshape back to original shape
    Ok(Array::from_vec(data).reshape(&array.shape()))
}

/// Merge sort implementation for in-place sorting
fn merge_sort<T: Clone + PartialOrd>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    let mid = len / 2;
    merge_sort(&mut arr[..mid]);
    merge_sort(&mut arr[mid..]);

    // Merge the two sorted halves
    let mut temp = Vec::with_capacity(len);

    let (left, right) = arr.split_at(mid);
    let mut l = 0;
    let mut r = 0;

    while l < left.len() && r < right.len() {
        if left[l] <= right[r] {
            temp.push(left[l].clone());
            l += 1;
        } else {
            temp.push(right[r].clone());
            r += 1;
        }
    }

    // Add remaining elements
    while l < left.len() {
        temp.push(left[l].clone());
        l += 1;
    }

    while r < right.len() {
        temp.push(right[r].clone());
        r += 1;
    }

    // Copy back to original array
    for (i, item) in temp.into_iter().enumerate() {
        arr[i] = item;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_partition_1d() {
        let a = Array::from_vec(vec![9, 4, 1, 7, 5, 3, 8, 2, 6]);
        let partitioned = partition(&a, 3, None).unwrap();

        // The 4th element (index 3) should be in its sorted position
        let kth_element = partitioned.get(&[3]).unwrap();

        // Check that all elements before index 3 are <= kth_element
        for i in 0..3 {
            assert!(partitioned.get(&[i]).unwrap() <= kth_element);
        }

        // Check that all elements after index 3 are >= kth_element
        for i in 4..9 {
            assert!(partitioned.get(&[i]).unwrap() >= kth_element);
        }
    }

    #[test]
    fn test_searchsorted_left() {
        let a = Array::from_vec(vec![1, 3, 5, 7, 9]);
        let v = Array::from_vec(vec![0, 1, 2, 4, 8, 10]);
        let indices = searchsorted(&a, &v, Some("left"), None).unwrap();
        assert_eq!(indices.to_vec(), vec![0, 0, 1, 2, 4, 5]);
    }

    #[test]
    fn test_searchsorted_right() {
        let a = Array::from_vec(vec![1, 3, 5, 7, 9]);
        let v = Array::from_vec(vec![0, 1, 2, 4, 8, 10]);
        let indices = searchsorted(&a, &v, Some("right"), None).unwrap();
        assert_eq!(indices.to_vec(), vec![0, 1, 1, 2, 4, 5]);
    }

    #[test]
    fn test_searchsorted_duplicates() {
        let a = Array::from_vec(vec![1, 1, 1, 3, 3, 5]);
        let v = Array::from_vec(vec![1, 3]);

        let indices_left = searchsorted(&a, &v, Some("left"), None).unwrap();
        assert_eq!(indices_left.to_vec(), vec![0, 3]); // First occurrence

        let indices_right = searchsorted(&a, &v, Some("right"), None).unwrap();
        assert_eq!(indices_right.to_vec(), vec![3, 5]); // After last occurrence
    }

    #[test]
    fn test_binary_search_functions() {
        let arr = vec![1, 3, 5, 7, 9];

        assert_eq!(binary_search_left(&arr, &0), 0);
        assert_eq!(binary_search_left(&arr, &1), 0);
        assert_eq!(binary_search_left(&arr, &2), 1);
        assert_eq!(binary_search_left(&arr, &10), 5);

        assert_eq!(binary_search_right(&arr, &0), 0);
        assert_eq!(binary_search_right(&arr, &1), 1);
        assert_eq!(binary_search_right(&arr, &2), 1);
        assert_eq!(binary_search_right(&arr, &10), 5);
    }

    #[test]
    fn test_bincount() {
        // Basic bincount
        let x = Array::from_vec(vec![0, 1, 1, 3, 2, 1, 7]);
        let counts: Array<i32> = bincount(&x, None, None).unwrap();
        assert_eq!(counts.shape(), vec![8]);
        assert_eq!(counts.to_vec(), vec![1, 3, 1, 1, 0, 0, 0, 1]);

        // With minlength
        let counts: Array<i32> = bincount(&x, None, Some(10)).unwrap();
        assert_eq!(counts.shape(), vec![10]);
        assert_eq!(counts.to_vec(), vec![1, 3, 1, 1, 0, 0, 0, 1, 0, 0]);

        // With weights
        let weights = Array::from_vec(vec![0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 2.0]);
        let weighted_counts: Array<f64> = bincount(&x, Some(&weights), None).unwrap();
        assert_eq!(weighted_counts.shape(), vec![8]);
        assert_eq!(
            weighted_counts.to_vec(),
            vec![0.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0]
        );

        // Empty array
        let empty: Array<i32> = Array::from_vec(vec![]);
        let counts: Array<i32> = bincount(&empty, None::<&Array<i32>>, Some(5)).unwrap();
        assert_eq!(counts.to_vec(), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_digitize() {
        // Basic digitize with increasing bins
        let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]);
        let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
        let indices = digitize(&x, &bins, false).unwrap();
        assert_eq!(indices.to_vec(), vec![1, 4, 3, 2]);

        // Test boundary values
        let x = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
        let indices = digitize(&x, &bins, false).unwrap();
        assert_eq!(indices.to_vec(), vec![0, 1, 2, 3, 4]);

        // Test with right=true
        let indices = digitize(&x, &bins, true).unwrap();
        assert_eq!(indices.to_vec(), vec![1, 2, 3, 4, 5]);

        // Test values outside bounds
        let x = Array::from_vec(vec![-1.0, 15.0]);
        let indices = digitize(&x, &bins, false).unwrap();
        assert_eq!(indices.to_vec(), vec![0, 5]);

        // Test decreasing bins
        let bins_dec = Array::from_vec(vec![10.0, 4.0, 2.5, 1.0, 0.0]);
        let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]);
        let indices = digitize(&x, &bins_dec, false).unwrap();
        assert_eq!(indices.to_vec(), vec![4, 1, 2, 3]);

        // Test 2D array
        let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6]).reshape(&[2, 2]);
        let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0]);
        let indices = digitize(&x, &bins, false).unwrap();
        assert_eq!(indices.shape(), vec![2, 2]);
        assert_eq!(indices.to_vec(), vec![1, 4, 3, 2]);
    }

    #[test]
    fn test_msort() {
        // Test basic merge sort
        let a = Array::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        let sorted = msort(&a).unwrap();
        assert_eq!(sorted.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Test empty array
        let empty: Array<i32> = Array::from_vec(vec![]);
        let sorted_empty = msort(&empty).unwrap();
        assert_eq!(sorted_empty.to_vec(), Vec::<i32>::new());

        // Test single element
        let single = Array::from_vec(vec![42]);
        let sorted_single = msort(&single).unwrap();
        assert_eq!(sorted_single.to_vec(), vec![42]);

        // Test with floating point numbers
        let float_arr = Array::from_vec(vec![3.14, 2.71, 1.41, 1.73]);
        let sorted_float = msort(&float_arr).unwrap();
        assert_eq!(sorted_float.to_vec(), vec![1.41, 1.73, 2.71, 3.14]);
    }

    #[test]
    fn test_sort_complex() {
        // Test complex number sorting
        let a = Array::from_vec(vec![
            Complex::new(3.0, 4.0), // magnitude 5.0
            Complex::new(1.0, 0.0), // magnitude 1.0
            Complex::new(0.0, 1.0), // magnitude 1.0
            Complex::new(2.0, 0.0), // magnitude 2.0
        ]);
        let sorted = sort_complex(&a).unwrap();

        // Check that magnitudes are in ascending order
        let magnitudes: Vec<f64> = sorted.to_vec().iter().map(|c| c.norm()).collect();
        for i in 1..magnitudes.len() {
            assert!(magnitudes[i] >= magnitudes[i - 1]);
        }

        // Test with equal magnitudes sorted by argument
        let b = Array::from_vec(vec![
            Complex::new(1.0, 0.0),  // arg = 0
            Complex::new(0.0, 1.0),  // arg = π/2
            Complex::new(-1.0, 0.0), // arg = π
            Complex::new(0.0, -1.0), // arg = -π/2 or 3π/2
        ]);
        let sorted_b = sort_complex(&b).unwrap();

        // All should have magnitude 1, sorted by argument
        for val in sorted_b.to_vec() {
            assert!((val.norm() - 1.0_f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_generic_sort() {
        // Test quicksort
        let a = Array::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        let sorted = sort(&a, Some("quicksort")).unwrap();
        assert_eq!(sorted.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Test mergesort
        let sorted_merge = sort(&a, Some("mergesort")).unwrap();
        assert_eq!(sorted_merge.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Test heapsort
        let sorted_heap = sort(&a, Some("heapsort")).unwrap();
        assert_eq!(sorted_heap.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Test default (quicksort)
        let sorted_default = sort(&a, None).unwrap();
        assert_eq!(sorted_default.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Test invalid sort kind
        let result = sort(&a, Some("invalid"));
        assert!(result.is_err());
    }

    #[test]
    fn test_lexsort_basic() {
        // Test lexicographic sorting
        let keys = Array::from_vec(vec![
            1, 0, 1, 0, 1, // Secondary sort key
            2, 2, 1, 1, 3, // Primary sort key
        ])
        .reshape(&[2, 5]);

        let indices = lexsort(&keys).unwrap();
        // Should sort by primary key (row 1) first: [1,1,2,2,3]
        // Then by secondary key (row 0) within ties
        assert_eq!(indices.to_vec(), vec![3, 2, 1, 0, 4]);
    }
}

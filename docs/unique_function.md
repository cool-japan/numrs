# Unique Function Documentation

## Overview

The `unique` function in NumRS2 finds unique elements in an array. It provides functionality similar to NumPy's `numpy.unique()` function, with support for axis-specific operations and multiple return values.

Two implementations are available:
- `unique`: Standard implementation
- `unique_optimized`: Performance-optimized implementation for most use cases

## Basic Usage

```rust
use numrs2::prelude::*;

// Create an array with duplicate values
let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]);

// Find unique values
let result = unique(&a, None, None, None, None).unwrap();
// result.values contains: [1, 2, 3, 4]
```

## Function Signature

```rust
pub fn unique<T>(
    a: &Array<T>,
    axis: Option<usize>,
    return_index: Option<bool>,
    return_inverse: Option<bool>,
    return_counts: Option<bool>,
) -> Result<UniqueResult<T>>
where
    T: Clone + Hash + Eq + Debug + Zero,
```

### Parameters

- **a** - Input array
- **axis** - Optional axis along which to find unique elements
- **return_index** - If true, also return the indices of the first occurrences of the unique values
- **return_inverse** - If true, also return the indices to reconstruct the original array
- **return_counts** - If true, also return the counts of each unique value

### Return Value

The function returns a `UniqueResult<T>` struct that contains some or all of:
- **values** - The unique values in the array (always returned)
- **indices** - The indices of the first unique values (if `return_index` is true)
- **inverse** - The indices to reconstruct the original array (if `return_inverse` is true)
- **counts** - The counts of each unique value (if `return_counts` is true)

## Examples

### Finding Unique Values in a 1D Array

```rust
use numrs2::prelude::*;

// Create an array with duplicate values
let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]);

// Find unique values
let result = unique(&a, None, None, None, None).unwrap();
assert_eq!(result.values.to_vec(), vec![1, 2, 3, 4]);
```

### Getting Indices of First Occurrences

```rust
// Find unique values and indices of their first occurrences
let result = unique(&a, None, Some(true), None, None).unwrap();
let (values, indices) = result.values_indices().unwrap();
assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
assert_eq!(indices.to_vec(), vec![0, 1, 2, 5]);
```

### Getting Counts of Unique Values

```rust
// Find unique values and their counts
let result = unique(&a, None, None, None, Some(true)).unwrap();
let (values, counts) = result.values_counts().unwrap();
assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
assert_eq!(counts.to_vec(), vec![2, 2, 1, 1]);
```

### Reconstructing Original Array with Inverse Indices

```rust
// Find unique values and inverse indices
let result = unique(&a, None, None, Some(true), None).unwrap();
let (values, inverse) = result.values_inverse().unwrap();

// Reconstruction example
let values_vec = values.to_vec();
let inverse_vec = inverse.to_vec();
let mut reconstructed = Vec::with_capacity(a.size());
for &idx in &inverse_vec {
    reconstructed.push(values_vec[idx]);
}
assert_eq!(reconstructed, a.to_vec());
```

### Working with 2D Arrays and Axis Parameter

```rust
// Create a 2D array
let b = Array::from_vec(vec![1, 2, 3, 1, 2, 3, 7, 8, 9]).reshape(&[3, 3]);
// The array looks like:
// [1, 2, 3]
// [1, 2, 3]
// [7, 8, 9]

// Find unique rows (axis=0)
let result = unique(&b, Some(0), None, None, None).unwrap();
// result.values contains 2 unique rows: [1, 2, 3] and [7, 8, 9]
assert_eq!(result.values.shape(), vec![2, 3]);

// Find unique columns (axis=1) by using transpose
let bt = b.transpose();
let result = unique(&bt, Some(0), None, None, None).unwrap();
```

### Getting Multiple Return Values

```rust
// Get all return values
let result = unique(&a, None, Some(true), Some(true), Some(true)).unwrap();
let (values, indices, inverse, counts) = result.values_indices_inverse_counts().unwrap();
```

## UniqueResult API

The `UniqueResult<T>` struct provides methods to access the results:

```rust
// Basic access to the values
let values = result.values;

// Access combinations of results (if requested):
let (values, indices) = result.values_indices().unwrap();
let (values, inverse) = result.values_inverse().unwrap();
let (values, counts) = result.values_counts().unwrap();
let (values, indices, inverse) = result.values_indices_inverse().unwrap();
let (values, indices, counts) = result.values_indices_counts().unwrap();
let (values, inverse, counts) = result.values_inverse_counts().unwrap();
let (values, indices, inverse, counts) = result.values_indices_inverse_counts().unwrap();
```

## Optimized Implementation

The optimized implementation (`unique_optimized`) uses a variety of techniques to improve performance:

```rust
use numrs2::prelude::*;

// Use the optimized version
let result = unique_optimized(&a, None, None, None, None).unwrap();
```

### Performance Characteristics

The optimized version is generally faster, especially for:
- Small arrays (1,000 elements): ~66% faster
- Medium arrays (10,000 elements): ~17% faster
- Operations with axis parameter: ~25% faster
- Operations with all return options: ~22% faster

For very large arrays (100,000+ elements), the standard version may perform better due to current parallelization overhead in the optimized implementation.

## Edge Cases

- **Empty Arrays**: Returns an empty result
- **Single Element Arrays**: Returns a single unique value
- **Arrays with All Identical Elements**: Returns a single unique value
- **Invalid Axis**: Returns an error if the axis is out of bounds

## Type Constraints

The `unique` function works with types that implement:
- `Clone`: For creating new arrays
- `Hash`: For efficient uniqueness detection
- `Eq`: For equality comparison
- `Debug`: For error messages and debugging
- `Zero`: For creating zero-initialized arrays

This includes most numeric types (integers, floating-point types) and types like `String` that implement these traits.

## Performance Considerations

- For performance-critical code, use `unique_optimized` for small to medium arrays
- For large arrays, benchmark both implementations to determine which performs better
- When using the axis parameter, be aware that performance depends on array shape and memory layout
- Requesting all return values (indices, inverse, counts) has minimal overhead compared to just requesting values
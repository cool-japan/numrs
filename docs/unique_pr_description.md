# Implement `unique` function with axis parameter support

## Overview

This PR implements the `unique` function for NumRS2, providing functionality equivalent to NumPy's `numpy.unique()`. The implementation includes support for the axis parameter and various return options.

## Features

- **Core functionality**: Find unique elements in arrays
- **Axis parameter**: Support finding unique subarrays along a specific axis
- **Return options**:
  - `return_index`: Indices of first occurrences of unique values
  - `return_inverse`: Indices to reconstruct the original array
  - `return_counts`: Counts of each unique value
- **Result handling**: `UniqueResult<T>` struct with methods to extract different combinations of results
- **Optimized implementation**: `unique_optimized` function with significant performance improvements

## Implementation Details

- Created a new module `unique.rs` for the standard implementation
- Created `unique_optimized.rs` for the performance-optimized version
- Added comprehensive tests in `test_unique.rs`
- Created example programs demonstrating usage
- Added performance benchmarks

## Benchmarks

| Scenario | Size | Standard | Optimized | Improvement |
|----------|------|----------|-----------|-------------|
| Small arrays | 1,000 | 2.72 ms | 0.92 ms | 66.11% faster |
| Medium arrays | 10,000 | 4.15 ms | 3.40 ms | 17.44% faster |
| Large arrays | 100,000 | 37.40 ms | 45.60 ms | 21.91% slower |
| With axis=0 | 10,000 | 5.90 ms | 4.40 ms | 24.93% faster |
| All return options | 10,000 | 5.00 ms | 3.90 ms | 21.82% faster |

## Usage Examples

```rust
use numrs2::prelude::*;

// Basic usage
let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]);
let result = unique(&a, None, None, None, None).unwrap();
// result.values contains: [1, 2, 3, 4]

// With return options
let result = unique(&a, None, Some(true), Some(true), Some(true)).unwrap();
let (values, indices, inverse, counts) = result.values_indices_inverse_counts().unwrap();

// With axis parameter (2D array)
let b = Array::from_vec(vec![1, 2, 3, 1, 2, 3, 7, 8, 9]).reshape(&[3, 3]);
let result = unique(&b, Some(0), None, None, None).unwrap();
// Finds unique rows
```

## Documentation

- Added comprehensive function and method documentation
- Created detailed usage examples
- Included technical design document
- Added performance considerations

## Testing

- Created test cases covering basic functionality
- Added tests for 2D arrays with axis parameter
- Added tests for edge cases (empty arrays, all identical elements)
- Added tests for each return option
- Created comprehensive examples and benchmarks

## Changes

- Created `src/unique.rs` module
- Created `src/unique_optimized.rs` module
- Added `unique` and `unique_optimized` to prelude
- Added test file `tests/test_unique.rs`
- Added example programs `examples/unique_example.rs`, `examples/unique_simple_benchmark.rs`, and `examples/unique_optimized_benchmark.rs`
- Added documentation in `docs/unique_function.md` and `docs/unique_design.md`
- Updated `TODO.md` to mark this feature as completed

## Next Steps

- Further tune parallel processing for large arrays
- Add specialized implementations for common numeric types
- Consider SIMD optimizations for performance-critical paths
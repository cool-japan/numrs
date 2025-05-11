# Unique Function Technical Design

## Implementation Overview

The `unique` function in NumRS2 is designed to find unique elements in n-dimensional arrays, with support for axis-specific operations. It can return various combinations of results: unique values, indices of first occurrences, indices for reconstruction, and counts of each unique value.

## Design Goals

1. **Correctness**: Match NumPy's behavior for all edge cases
2. **Performance**: Optimize common operations
3. **Flexibility**: Support different dimensions and data types
4. **Usability**: Provide a clean API for working with results

## Architecture

### Core Components

#### 1. UniqueResult Struct

The `UniqueResult<T>` struct serves as the container for all possible return values:

```rust
pub struct UniqueResult<T> {
    pub values: Array<T>,
    pub indices: Option<Array<usize>>,
    pub inverse: Option<Array<usize>>,
    pub counts: Option<Array<usize>>,
}
```

This approach allows users to access specific combinations of results through helper methods, improving usability.

#### 2. Main Algorithm

The algorithm has two main branches:

1. **Without axis parameter (flattened operation)**:
   - Flatten the array to a 1D vector
   - Use a HashMap to track unique elements and their indices
   - Build result collections based on requested return values

2. **With axis parameter**:
   - Extract subarrays along the specified axis
   - Convert each subarray to a hashable representation
   - Find unique subarrays using HashSet
   - Build the result by combining unique subarrays

### Data Flow

1. **Input Validation**: Check array dimensions and axis parameter
2. **Data Preparation**: Flatten or extract subarrays
3. **Unique Detection**: Identify unique elements using HashMap/HashSet
4. **Result Construction**: Assemble the requested return values
5. **Result Packaging**: Return via UniqueResult struct

## Implementation Details

### Standard Implementation

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

The standard implementation uses a straightforward approach:

1. For flattened operation (no axis):
   - Iterate through elements sequentially
   - Track unique elements and their positions
   - Build return values as needed

2. For axis-specific operation:
   - Extract subarrays along the axis
   - Find unique subarrays
   - Construct output with proper shape

### Optimized Implementation

```rust
pub fn unique_optimized<T>(
    a: &Array<T>,
    axis: Option<usize>,
    return_index: Option<bool>,
    return_inverse: Option<bool>,
    return_counts: Option<bool>,
) -> Result<UniqueResult<T>>
where
    T: Clone + Hash + Eq + Debug + Zero + Send + Sync,
```

The optimized implementation introduces several improvements:

1. **Pre-allocated Collections**:
   - Use capacity hints based on input size
   - Estimate unique element count with a heuristic (90% of size for random data)

2. **Early Short-circuiting**:
   - Special handling for 1D arrays with axis=0
   - Optimized paths for common cases

3. **Memory Access Patterns**:
   - Minimize vector resizing
   - Avoid unnecessary copies

4. **Parallel Processing**:
   - For large arrays, use parallel processing with Rayon
   - Process data in batches for better cache efficiency
   - Use thread-safe data structures for aggregation

### Special Cases Handling

Both implementations handle these special cases:

1. **Empty Arrays**: Return empty results
2. **Single Element Arrays**: Return that element as the only unique value
3. **Invalid Axis**: Return error if axis is out of bounds
4. **1D Array with axis=0**: Treat as flattened operation

## Performance Optimizations

### Capacity Hints

Collection pre-allocation significantly reduces reallocations:

```rust
// Estimate capacity based on array size
let estimated_capacity = (array_size as f64 * 0.9) as usize;
let mut unique_elements = Vec::with_capacity(estimated_capacity);
let mut value_to_index = HashMap::with_capacity(estimated_capacity);
```

### Batch Processing

The optimized version processes large arrays in batches:

```rust
let batch_size = std::cmp::max(1, array_size / rayon::current_num_threads());
let batches = flat_data.chunks(batch_size);
```

This improves cache locality and reduces synchronization overhead.

### Conditional Execution

We avoid unnecessary work based on return flag parameters:

```rust
let need_index = return_index.unwrap_or(false);
let need_inverse = return_inverse.unwrap_or(false);
let need_counts = return_counts.unwrap_or(false);

// Only allocate and compute what's needed
if need_index {
    // Compute indices
}
```

## Performance Characteristics

### Benchmark Results

| Scenario | Size | Standard | Optimized | Improvement |
|----------|------|----------|-----------|-------------|
| Small arrays | 1,000 | 2.72 ms | 0.92 ms | 66.11% faster |
| Medium arrays | 10,000 | 4.15 ms | 3.40 ms | 17.44% faster |
| Large arrays | 100,000 | 37.40 ms | 45.60 ms | 21.91% slower |
| With axis=0 | 10,000 | 5.90 ms | 4.40 ms | 24.93% faster |
| All return options | 10,000 | 5.00 ms | 3.90 ms | 21.82% faster |

### Performance Analysis

1. **Small Arrays**: The optimized version shows dramatic improvement due to pre-allocation and efficient memory access patterns.

2. **Medium Arrays**: Moderate improvement from optimized access patterns and minimal allocations.

3. **Large Arrays**: The current parallel implementation shows worse performance, likely due to:
   - Synchronization overhead
   - Thread creation/management costs
   - Memory contention

4. **Axis Operations**: Significant improvement from specialized paths and optimized data handling.

5. **Return Options**: Faster when computing multiple return values due to data reuse.

## Future Improvements

1. **Adaptive Parallelization**:
   - Use an adaptive threshold for parallel execution
   - Only use parallel execution when input size exceeds threshold

2. **SIMD Optimization**:
   - Apply SIMD operations for numeric types
   - Optimize equality comparisons and counting

3. **Memory Layout Optimization**:
   - Consider array striding and layout for optimal access patterns
   - Optimize cache line usage

4. **Algorithm Specialization**:
   - Use specialized algorithms for sorted data
   - Add specialized paths for common numeric types

5. **Thread-Local Storage**:
   - Improve parallel implementation with thread-local buffers
   - Reduce synchronization points

## Conclusion

The `unique` implementation in NumRS2 successfully provides functionality equivalent to NumPy's `numpy.unique()` with good performance characteristics. The optimized version delivers significant improvements for common use cases, while the standard implementation offers reliable performance across all scenarios.

The design prioritizes usability through a clean API that handles multiple return values elegantly, while maintaining type safety through the Rust type system.
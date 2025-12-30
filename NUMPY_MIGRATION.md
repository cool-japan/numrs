# NumPy to NumRS2 Migration Guide

This guide helps NumPy users transition to NumRS2, highlighting key differences and similarities between the two libraries.

## Overview

NumRS2 aims to provide NumPy's functionality in Rust with idiomatic Rust patterns and performance benefits. While the API is designed to be familiar to NumPy users, there are necessarily some differences due to:

1. Language differences between Python and Rust
2. Rust's static typing vs Python's dynamic typing
3. Rust's ownership and borrowing system
4. Performance considerations specific to Rust

## Key Concepts Mapping

| NumPy Concept | NumRS2 Equivalent | Notes |
|---------------|-------------------|-------|
| `ndarray` | `Array` | Core array type in NumRS2 |
| `dtype` | Rust types + `Type` enum | NumRS2 uses Rust's static type system |
| NumPy functions | NumRS2 methods | Most operations are implemented as methods on `Array` |
| Broadcasting | Broadcasting | Similar semantics but explicit method names in some cases |
| Views vs Copies | Borrowed vs Owned | Follows Rust's ownership model |

## Basic Operations Comparison

### Array Creation

**NumPy:**
```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4])
b = np.zeros((2, 3))
c = np.ones((2, 2))
d = np.eye(3)
e = np.linspace(0, 1, 5)
```

**NumRS2:**
```rust
use numrs2::prelude::*;

// Create arrays
let a = Array::from_vec(vec![1, 2, 3, 4]);
let b = Array::zeros(&[2, 3]);
let c = Array::ones(&[2, 2]);
let d = Array::eye(3);
let e = Array::linspace(0.0, 1.0, 5)?;
```

### Array Operations

**NumPy:**
```python
# Element-wise operations
result = a + b
result = a * b
result = np.sqrt(a)

# Matrix operations
result = a.dot(b)
result = a @ b  # Matrix multiplication
```

**NumRS2:**
```rust
// Element-wise operations
let result = a.add(&b)?;  // Using method
let result = &a + &b;     // Using operator overloading
let result = a.multiply(&b)?;
let result = a.sqrt()?;

// Matrix operations
let result = a.dot(&b)?;
let result = a.matmul(&b)?;  // Matrix multiplication
```

### Indexing and Slicing

**NumPy:**
```python
# Indexing
value = a[0]
row = a[1, :]
column = a[:, 1]

# Fancy indexing
subset = a[[0, 2, 3]]
mask = a > 2
filtered = a[mask]
```

**NumRS2:**
```rust
// Indexing
let value = a.get(&[0])?;
let row = a.slice(&[1..2, ..]);
let column = a.slice(&[.., 1..2]);

// Advanced indexing
let subset = a.take(&Array::from_vec(vec![0, 2, 3]))?;
let mask = a.gt(&2)?;
let filtered = a.compress(&mask)?;
```

## Error Handling

A major difference between NumPy and NumRS2 is error handling:

**NumPy** often raises exceptions for invalid operations:
```python
try:
    result = np.array([1, 2, 3]).reshape((2, 2))  # Will raise ValueError
except ValueError as e:
    print(f"Error: {e}")
```

**NumRS2** uses Rust's `Result` type:
```rust
match array.reshape(&[2, 2]) {
    Ok(reshaped) => println!("Reshaped: {}", reshaped),
    Err(e) => println!("Error: {}", e),
}

// Or with the ? operator
let reshaped = array.reshape(&[2, 2])?;
```

## Common Patterns

### Broadcasting

**NumPy** has implicit broadcasting:
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
c = a + b  # b is broadcast to shape (2, 2)
```

**NumRS2** offers both methods:
```rust
// Explicit broadcasting
let c = a.add_broadcast(&b)?;

// Or with compatible shapes
let c = &a + &b;  // If shapes are broadcast-compatible
```

### Reduction Operations

**NumPy:**
```python
sum_all = np.sum(a)
sum_axis0 = np.sum(a, axis=0)
mean_axis1 = np.mean(a, axis=1)
```

**NumRS2:**
```rust
let sum_all = a.sum()?;
let sum_axis0 = a.sum_axis(0)?;
let mean_axis1 = a.mean_axis(1)?;
```

### Linear Algebra

**NumPy:**
```python
from numpy import linalg

# Decompositions
u, s, vh = linalg.svd(a)
eigenvalues, eigenvectors = linalg.eig(a)

# Other operations
inv_a = linalg.inv(a)
det_a = linalg.det(a)
```

**NumRS2:**
```rust
// Decompositions
let (u, s, vt) = a.svd_compute()?;
let (eigenvalues, eigenvectors) = a.eig()?;

// Other operations
let inv_a = a.inv()?;
let det_a = a.det()?;
```

## Performance Considerations

### Memory Management

NumRS2 offers more explicit control over memory with features like:

```rust
// Create array with specific memory alignment
let aligned = Array::with_alignment::<f64>(&[1000, 1000], 64)?;

// Memory-mapped arrays for large datasets
let mmap_array = MmapArray::new::<f64>("data.bin", &[1000, 1000])?;

// Custom allocators for specialized workloads
let arena_array = Array::with_allocator::<f64>(&[5000, 5000], ArenaAllocator::new())?;
```

### Parallelism

NumRS2 provides explicit control over parallel execution:

```rust
// With automatic parallelization
let result = a.parallel_map(|x| x.sqrt())?;

// With custom threshold
let result = a.parallel_map_with_threshold(|x| x.sqrt(), 1000)?;

// With custom scheduling strategy
let strategy = SchedulingStrategy::work_stealing();
let result = a.parallel_map_with_strategy(|x| x.sqrt(), strategy)?;
```

## Tips for Migration

1. **Start small**: Begin by migrating isolated numerical computation functions
2. **Use explicit methods**: Prefer explicit method calls until you're comfortable with NumRS2's behavior
3. **Handle errors properly**: Always handle potential errors using Rust's error handling patterns
4. **Benchmark and optimize**: Use Rust's and NumRS2's performance features to optimize your code
5. **Use type annotations**: Take advantage of Rust's type system for correctness and documentation
6. **Leverage Rust's ecosystem**: Integrate with other Rust crates for I/O, visualization, and more

## Common Gotchas

1. **Indexing**: NumRS2 uses `&[i, j]` style indexing vs NumPy's `[i, j]`
2. **Error handling**: Operations that can fail return `Result<T>` and require handling
3. **Memory ownership**: Be aware of Rust's ownership rules and when operations create new arrays
4. **Type specificity**: NumRS2 requires specific type information where NumPy would infer types
5. **Mutability**: Mutable operations in NumRS2 require explicit mut references

## Examples

See the [examples directory](examples/) for side-by-side comparisons of NumPy and NumRS2 code for common tasks.

## Function Name Equivalents

| NumPy Function | NumRS2 Equivalent |
|----------------|-------------------|
| `np.array()` | `Array::from_vec()` |
| `np.zeros()` | `Array::zeros()` |
| `np.ones()` | `Array::ones()` |
| `np.eye()` | `Array::eye()` |
| `np.linspace()` | `Array::linspace()` |
| `np.arange()` | `Array::range()` |
| `np.reshape()` | `array.reshape()` |
| `np.transpose()` | `array.transpose()` |
| `np.concatenate()` | `Array::concatenate()` |
| `np.stack()` | `Array::stack()` |
| `np.vstack()` | `Array::vstack()` |
| `np.hstack()` | `Array::hstack()` |
| `np.split()` | `array.split()` |
| `np.random.rand()` | `random::random()` |
| `np.random.randn()` | `random::normal()` |

## Further Resources

- [NumRS2 User Guide](GUIDE.md)
- [API Documentation](https://docs.rs/numrs2)
- [Example Gallery](examples/README.md)
- [NumPy Documentation](https://numpy.org/doc/stable/)
# NumRS2 Python Guide

Complete guide to using NumRS2 from Python via PyO3 bindings.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [NumPy Compatibility](#numpy-compatibility)
5. [Performance](#performance)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (for building from source)
- maturin (Python package builder)

### Installing from Source

```bash
# Install maturin
pip install maturin

# Clone the repository
git clone https://github.com/cool-japan/numrs
cd numrs

# Build and install in development mode
maturin develop --release --features python

# Or build a wheel
maturin build --release --features python
pip install target/wheels/numrs2-*.whl
```

### Verify Installation

```python
import numrs2 as nr
print(nr.__version__)
```

## Quick Start

### Basic Array Operations

```python
import numrs2 as nr
import numpy as np

# Create arrays
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = nr.zeros([3, 4])
c = nr.ones([2, 3])
d = nr.eye(3)

# Array properties
print(a.shape)  # [5]
print(a.ndim)   # 1
print(a.size)   # 5
print(a.dtype)  # 'float64'

# Operations
a_sum = a.sum()
a_mean = a.mean()
a_min = a.min()
a_max = a.max()

# Reshape and transpose
matrix = a.reshape([5, 1])
transposed = matrix.transpose()

# Convert to NumPy
numpy_array = a.to_numpy(None)
```

### Element-wise Operations

```python
x = nr.array([1.0, 2.0, 3.0])
y = nr.array([4.0, 5.0, 6.0])

# Arithmetic
z = x + y  # [5.0, 7.0, 9.0]
z = x - y  # [-3.0, -3.0, -3.0]
z = x * y  # [4.0, 10.0, 18.0]
z = x / y  # [0.25, 0.4, 0.5]

# Negation
neg_x = -x  # [-1.0, -2.0, -3.0]
```

## API Reference

### Array Creation

#### `array(data)`
Create an array from Python list, tuple, or NumPy array.

```python
a = nr.array([1.0, 2.0, 3.0])
b = nr.array(np.array([4.0, 5.0, 6.0]))
```

#### `zeros(shape)`
Create an array filled with zeros.

```python
z = nr.zeros([2, 3])  # 2x3 array of zeros
```

#### `ones(shape)`
Create an array filled with ones.

```python
o = nr.ones([3, 2])  # 3x2 array of ones
```

#### `eye(n, m=None, k=None)`
Create a 2D array with ones on the diagonal.

```python
I = nr.eye(3)  # 3x3 identity matrix
```

#### `identity(n)`
Create an n×n identity matrix.

```python
I = nr.identity(4)  # 4x4 identity matrix
```

#### `linspace(start, stop, num, endpoint=True)`
Create an array with evenly spaced values.

```python
x = nr.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]
```

#### `arange(start, stop, step=None)`
Create an array with values in a range.

```python
r = nr.arange(0.0, 10.0, 2.0)  # [0.0, 2.0, 4.0, 6.0, 8.0]
```

#### `full(shape, fill_value)`
Create an array filled with a constant value.

```python
f = nr.full([2, 3], 7.0)  # 2x3 array filled with 7.0
```

#### `zeros_like(a)`, `ones_like(a)`
Create arrays with the same shape as another array.

```python
a = nr.array([1.0, 2.0, 3.0])
z = nr.zeros_like(a)  # Same shape as a, filled with zeros
```

### Array Methods

#### `reshape(shape)`
Reshape array to new dimensions.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = a.reshape([2, 3])  # 2x3 matrix
```

#### `transpose()`
Transpose array (swap dimensions).

```python
a = nr.zeros([2, 3])
b = a.transpose()  # Shape: [3, 2]
```

#### `flatten()`
Flatten array to 1D.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
b = a.flatten()  # Shape: [4]
```

#### `squeeze()`
Remove dimensions of size 1.

```python
a = nr.zeros([3, 1, 4, 1])
b = a.squeeze()  # Shape: [3, 4]
```

#### `tolist()`
Convert to Python list.

```python
a = nr.array([1.0, 2.0, 3.0])
lst = a.tolist()  # [1.0, 2.0, 3.0]
```

#### `to_numpy(py)`
Convert to NumPy array.

```python
a = nr.array([1.0, 2.0, 3.0])
np_array = a.to_numpy(None)
```

### Linear Algebra (`nr.linalg`)

#### `matmul(a, b)`
Matrix multiplication.

```python
A = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
B = nr.array([5.0, 6.0, 7.0, 8.0]).reshape([2, 2])
C = nr.linalg.matmul(A, B)
# Or: C = nr.matmul(A, B)
```

#### `dot(a, b)`
Dot product.

```python
u = nr.array([1.0, 2.0, 3.0])
v = nr.array([4.0, 5.0, 6.0])
result = nr.linalg.dot(u, v)  # 32.0
# Or: result = nr.dot(u, v)
```

#### `det(a)`
Matrix determinant.

```python
M = nr.array([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
d = nr.linalg.det(M)
```

#### `trace(a)`
Matrix trace (sum of diagonal elements).

```python
M = nr.eye(3)
t = nr.linalg.trace(M)  # 3.0
```

#### `inv(a)`
Matrix inverse.

```python
M = nr.array([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
M_inv = nr.linalg.inv(M)
```

#### `solve(a, b)`
Solve linear system Ax = b.

```python
A = nr.array([3.0, 1.0, 1.0, 2.0]).reshape([2, 2])
b = nr.array([9.0, 8.0])
x = nr.linalg.solve(A, b)
```

#### `eigvals(a)`
Compute eigenvalues.

```python
M = nr.array([4.0, -2.0, 1.0, 1.0]).reshape([2, 2])
vals = nr.linalg.eigvals(M)
```

#### `eig(a)`
Eigendecomposition (eigenvalues and eigenvectors).

```python
M = nr.array([4.0, -2.0, 1.0, 1.0]).reshape([2, 2])
vals, vecs = nr.linalg.eig(M)
```

#### `svd(a, full_matrices=True)`
Singular Value Decomposition.

```python
A = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
U, S, Vt = nr.linalg.svd(A)
```

#### `qr(a)`
QR decomposition.

```python
A = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
Q, R = nr.linalg.qr(A)
```

#### `cholesky(a)`
Cholesky decomposition (for positive definite matrices).

```python
# A must be positive definite
L = nr.linalg.cholesky(A)
```

#### `lu(a)`
LU decomposition.

```python
A = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
P, L, U = nr.linalg.lu(A)
```

#### `norm(a, ord=None)`
Matrix or vector norm.

```python
v = nr.array([3.0, 4.0])
n = nr.linalg.norm(v)  # 5.0 (L2 norm)
```

#### `cond(a)`
Condition number.

```python
M = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
c = nr.linalg.cond(M)
```

#### `matrix_rank(a, tol=None)`
Matrix rank.

```python
M = nr.array([1.0, 2.0, 2.0, 4.0]).reshape([2, 2])
r = nr.linalg.matrix_rank(M)  # 1 (rank-deficient)
```

### Statistics (`nr.stats`)

#### `mean(a, axis=None)`
Arithmetic mean.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
m = nr.stats.mean(a)  # 3.0
```

#### `median(a, axis=None)`
Median value.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
med = nr.stats.median(a)  # 3.0
```

#### `std(a, axis=None, ddof=0)`
Standard deviation.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
s = nr.stats.std(a)
```

#### `var(a, axis=None, ddof=0)`
Variance.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
v = nr.stats.var(a)
```

#### `corrcoef(x, y=None)`
Correlation coefficient.

```python
x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = nr.array([2.0, 4.0, 6.0, 8.0, 10.0])
corr_matrix = nr.stats.corrcoef(x, y)
```

#### `histogram(a, bins=None, range=None)`
Compute histogram.

```python
a = nr.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
counts, edges = nr.stats.histogram(a, bins=3)
```

#### `percentile(a, q)`
Compute percentile.

```python
a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
p50 = nr.stats.percentile(a, 50.0)  # Median
```

### Random Numbers (`nr.random`)

#### `randn(size)`
Random samples from standard normal distribution.

```python
samples = nr.random.randn([100])  # 100 samples from N(0, 1)
```

#### `rand(size)`
Random samples from uniform [0, 1) distribution.

```python
samples = nr.random.rand([10, 10])  # 10x10 uniform random
```

### Neural Networks (`nr.nn`)

#### Activation Functions

```python
x = nr.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU: max(0, x)
y = nr.nn.relu(x)

# Sigmoid: 1 / (1 + exp(-x))
y = nr.nn.sigmoid(x)

# Tanh: tanh(x)
y = nr.nn.tanh(x)

# Softmax (outputs sum to 1)
logits = nr.array([1.0, 2.0, 3.0])
probs = nr.nn.softmax(logits)
```

#### Loss Functions

```python
predictions = nr.array([1.5, 2.3, 3.1])
targets = nr.array([1.0, 2.0, 3.0])

# Mean squared error
mse = nr.nn.mse_loss(predictions, targets)

# Cross-entropy
ce = nr.nn.cross_entropy_loss(predictions, targets)
```

#### Normalization

```python
x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Dropout (probability p)
dropped = nr.nn.dropout(x, p=0.5)

# Batch normalization
normalized = nr.nn.batch_norm(x, eps=1e-5)
```

### I/O (`nr.io`)

#### NPY Format (NumPy Binary)

```python
# Save
a = nr.array([1.0, 2.0, 3.0])
nr.io.save_npy("array.npy", a)

# Load
b = nr.io.load_npy("array.npy")
```

#### CSV Format

```python
# Save
a = nr.array([1.0, 2.0, 3.0])
nr.io.save_csv("array.csv", a)

# Load
b = nr.io.load_csv("array.csv")
```

#### JSON Format

```python
# Save
a = nr.array([1.0, 2.0, 3.0])
nr.io.save_json("array.json", a)

# Load
b = nr.io.load_json("array.json")
```

## NumPy Compatibility

### API Mapping

| NumPy | NumRS2 | Notes |
|-------|--------|-------|
| `np.array()` | `nr.array()` | Fully compatible |
| `np.zeros()` | `nr.zeros()` | Fully compatible |
| `np.ones()` | `nr.ones()` | Fully compatible |
| `np.eye()` | `nr.eye()` | Fully compatible |
| `np.linspace()` | `nr.linspace()` | Fully compatible |
| `np.arange()` | `nr.arange()` | Fully compatible |
| `arr.reshape()` | `arr.reshape()` | Fully compatible |
| `arr.T` | `arr.transpose()` | Use method instead of property |
| `np.matmul()` | `nr.matmul()` | Fully compatible |
| `np.linalg.det()` | `nr.linalg.det()` | Fully compatible |
| `np.linalg.inv()` | `nr.linalg.inv()` | Fully compatible |
| `np.mean()` | `nr.stats.mean()` | In stats submodule |
| `np.std()` | `nr.stats.std()` | In stats submodule |

### Zero-Copy Conversion

NumRS2 supports efficient conversion to/from NumPy:

```python
import numpy as np
import numrs2 as nr

# NumPy -> NumRS2
np_array = np.array([1.0, 2.0, 3.0])
nr_array = nr.array(np_array)

# NumRS2 -> NumPy
back_to_numpy = nr_array.to_numpy(None)
```

## Performance

### Benchmarks

NumRS2 leverages Rust's performance and SIMD optimizations:

- **Array operations**: Comparable to NumPy for large arrays
- **Linear algebra**: Uses OxiBLAS (pure Rust BLAS implementation)
- **Memory safety**: No segfaults or buffer overflows

### Performance Tips

1. **Use vectorized operations** instead of Python loops
2. **Minimize conversions** between NumPy and NumRS2
3. **Preallocate arrays** when possible

```python
# Good: Vectorized
result = nr.array([1.0, 2.0, 3.0]) * 2.0

# Avoid: Python loops
result = nr.array([x * 2.0 for x in [1.0, 2.0, 3.0]])
```

## Advanced Usage

### Type Hints

NumRS2 provides type stubs for IDE support:

```python
from numrs2 import Array
import numrs2 as nr

def process_data(arr: Array) -> Array:
    return arr * 2.0

result: Array = nr.zeros([10])
```

### Error Handling

NumRS2 converts Rust errors to Python exceptions:

```python
try:
    M = nr.array([1.0, 0.0, 0.0, 0.0]).reshape([2, 2])
    inv = nr.linalg.inv(M)  # Singular matrix
except ValueError as e:
    print(f"Error: {e}")
```

## Troubleshooting

### Installation Issues

**Problem**: `maturin: command not found`

```bash
pip install maturin
```

**Problem**: Build fails with "Rust compiler not found"

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Runtime Issues

**Problem**: `ImportError: cannot import name '_numrs2'`

Solution: Rebuild the package:
```bash
maturin develop --release --features python
```

**Problem**: Dimension mismatch errors

Solution: Check array shapes before operations:
```python
print(f"Shape: {arr.shape}")
```

### Performance Issues

**Problem**: Operations slower than NumPy

Solution:
- Ensure you're using release build (`--release` flag)
- Check if SIMD optimizations are enabled
- Profile your code to identify bottlenecks

## Examples

See the `examples/python/` directory for complete examples:

- `basic_usage.py` - Array creation and manipulation
- `linear_algebra.py` - Matrix operations and decompositions
- `neural_networks.py` - Neural network primitives
- `data_io.py` - Saving and loading arrays
- `numpy_migration.py` - NumPy to NumRS2 migration guide

## Further Reading

- [NumRS2 GitHub Repository](https://github.com/cool-japan/numrs)
- [NumRS2 Rust Documentation](https://docs.rs/numrs2)
- [SciRS2 Ecosystem](https://github.com/cool-japan/scirs2)

## Contributing

We welcome contributions! Please see our contributing guidelines in the main repository.

## License

NumRS2 is licensed under Apache-2.0.

---

**Version**: 0.2.0
**Last Updated**: 2026-02-09

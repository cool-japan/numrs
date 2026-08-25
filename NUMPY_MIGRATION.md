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
| `np.quantile()` / `np.percentile()` | `stats::quantile()` / `stats::percentile()` |
| `np.histogramdd()` | `stats::histogramdd()` |
| `np.pad()` | `array_ops::manipulation::pad::pad()` |
| `np.fft.fftn()` / `ifftn()` / `rfftn()` / `irfftn()` | `fft::numpy_parity::{fftn, ifftn, rfftn, irfftn}` |
| `numpy.linalg.multi_dot()` | `linalg::multi_dot()` |
| `numpy.linalg.tensorsolve()` / `tensorinv()` | `linalg::tensorsolve()` / `tensorinv()` |
| `numpy.ma.array()` | `masked::MaskedArray::new()` |
| `numpy.polynomial.Chebyshev` / `Legendre` / `Hermite` / `HermiteE` / `Laguerre` | `new_modules::polynomial::{Chebyshev, Legendre, Hermite, HermiteE, Laguerre}` |
| `np.random.SeedSequence()` | `random::SeedSequence::new()` |
| `np.random.Generator(np.random.Philox(seed))` | `random::philox_seed_rng()` |
| `Generator.permuted()` | `Generator::permuted()` (see the Exactness caveat in TODO.md) |

## New in 0.4.1: Additional NumPy Parity

The APIs below landed in the 0.4.1 production-hardening pass (see CHANGELOG.md for the full,
verified list). Each snippet's status is marked explicitly: **doctest** means it is copied
verbatim from an existing `///` example already compiled and run as part of this crate's own
`cargo test --doc`; **signature-verified** means the function/type signature was read directly
from source but the snippet itself was not independently compiled for this guide.

### ufunc `reduce` / `accumulate` / `outer` / `at`

**NumPy:**
```python
import numpy as np

row_sums = np.add.reduce(a, axis=1)          # like a.sum(axis=1)
col_max = np.maximum.reduce(a, axis=0)
running = np.add.accumulate(a, axis=0)       # like np.cumsum
outer = np.multiply.outer(a, b)
np.add.at(a, [0, 0, 1], [10, 20, 30])        # unbuffered scatter-add
```

**NumRS2** (`numrs2::ufunc_ops`) — *doctest, from `ufunc_reduce`'s own example*:
```rust
use numrs2::array::Array;
use numrs2::ufunc_ops::{ufunc_reduce, UfuncOp};

let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

// np.add.reduce(a, axis=1) == [6.0, 15.0]
let row_sums = ufunc_reduce(UfuncOp::Add, &a, Some(1), false, None)
    .expect("reduce should succeed");
assert_eq!(row_sums.to_vec(), vec![6.0, 15.0]);

// np.maximum.reduce(np.array([]), initial=5) == 5.0 (no identity, but `initial` covers it)
let empty: Array<f64> = Array::from_vec(vec![]);
let with_initial = ufunc_reduce(UfuncOp::Maximum, &empty, None, false, Some(5.0))
    .expect("initial makes an empty maximum.reduce succeed");
assert_eq!(with_initial.to_vec(), vec![5.0]);
```
`ufunc_accumulate(op, a, axis)`, `ufunc_outer(op, a, b)`, `ufunc_reduceat(op, a, indices, axis)`,
and `ufunc_at(op, a, indices, b)` (in-place, `np.add.at`-style scatter) round out the family —
*signature-verified*, same module.

### N-D FFT: `fftn` / `ifftn` / `rfftn` / `irfftn`

**NumPy:**
```python
spectrum = np.fft.fftn(a, s=None, axes=None, norm=None)
recovered = np.fft.ifftn(spectrum)
real_spectrum = np.fft.rfftn(a)
```

**NumRS2** (`numrs2::fft::numpy_parity`) — *signature-verified*:
```rust
use numrs2::fft::numpy_parity::{fftn, ifftn, rfftn};

let spectrum = fftn(&a, /* s */ None, /* axes */ None, /* norm */ None)?;
let recovered = ifftn(&spectrum, None, None, None)?;
let real_spectrum = rfftn(&a, None, None, None)?;
```
These are correctness wrappers over `scirs2_fft` that work around two confirmed upstream
normalization bugs (see CHANGELOG.md's Fixed and Known Upstream Issues sections) — prefer them
over calling `scirs2_fft::fftn` directly.

### `pad`: new modes and `reflect_type`

**NumPy:**
```python
np.pad(a, (2, 3), mode="constant", constant_values=(0, 0))
np.pad(b, ((1, 1), (2, 2)), mode="edge")
np.pad(c, (1, 1), mode="reflect", reflect_type="odd")
```

**NumRS2** (`numrs2::array_ops::manipulation::pad`) — *doctest, from `pad`'s own example*:
```rust
use numrs2::prelude::*;

// Pad 1D array with constant value
let a = Array::from_vec(vec![1, 2, 3]);
let result =
    pad(&a, &[(2, 3)], "constant", Some((0, 0)), None, None).expect("operation should succeed");
assert_eq!(result.to_vec(), vec![0, 0, 1, 2, 3, 0, 0, 0]);

// Pad 2D array with edge values
let b = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
let result =
    pad(&b, &[(1, 1), (2, 2)], "edge", None, None, None).expect("operation should succeed");
assert_eq!(result.shape(), vec![4, 6]);
```
`mode` now covers all 11 of NumPy's modes (`"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`,
`"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`); the last
parameter, `reflect_type`, selects `"even"` (default) or `"odd"` for `"reflect"`/`"symmetric"`.

### `quantile` / `percentile`: 9 new interpolation methods

**NumPy (>= 1.22):**
```python
np.quantile(a, 0.5, method="median_unbiased")
```

**NumRS2** (re-exported from the `quantile` submodule as `numrs2::stats::quantile`) — *signature-verified*:
```rust
use numrs2::stats::quantile; // the function, re-exported from src/stats/quantile.rs

let q = Array::from_vec(vec![0.5]);
let result = quantile(&a, &q, Some("median_unbiased"))?;
```
`method` accepts NumPy's 9 Hyndman & Fan methods (`"inverted_cdf"`, `"averaged_inverted_cdf"`,
`"closest_observation"`, `"interpolated_inverted_cdf"`, `"hazen"`, `"weibull"`, `"linear"`
[default], `"median_unbiased"`, `"normal_unbiased"`) plus the 4 legacy methods (`"lower"`,
`"higher"`, `"nearest"`, `"midpoint"`).

### `histogramdd` and `density=`

**NumPy:**
```python
hist, edges = np.histogramdd(data, bins=[2, 2])
```

**NumRS2** (`numrs2::stats::histogramdd`) — *doctest, from `histogramdd`'s own example*:
```rust
use numrs2::prelude::*;
use numrs2::stats::histogramdd;

let data = Array::from_vec(vec![
    0.0, 0.0,
    0.5, 0.5,
    1.0, 1.0,
    0.3, 0.7,
]).reshape(&[4, 2]);

let (hist, edges) = histogramdd(&data, &[2, 2], None, None, None).expect("histogramdd should succeed");
assert_eq!(hist.shape(), vec![2, 2]);
```
`histogram`/`histogram2d`/`histogramdd` all take a trailing `density: Option<bool>` —
`Some(true)` matches `density=True`.

### `multi_dot`, `tensorsolve`, `tensorinv`

**NumPy:**
```python
from numpy.linalg import multi_dot, tensorsolve, tensorinv

result = multi_dot([a, b, c])
x = tensorsolve(a, b)
a_inv = tensorinv(a, ind=1)
```

**NumRS2** (`numrs2::linalg`) — *doctest, from each function's own example*:
```rust
use numrs2::prelude::*;
use numrs2::linalg::{multi_dot, tensorsolve, tensorinv};

let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
let c = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0]).reshape(&[2, 2]);
let result = multi_dot(&[&a, &b, &c]).expect("multi_dot should succeed");

// tensorsolve/tensorinv degenerate to ordinary solve/inverse for 2-D square input.
let ts_a = Array::from_vec(vec![2.0, 0.0, 0.0, 4.0]).reshape(&[2, 2]);
let ts_b = Array::from_vec(vec![4.0, 8.0]);
let x = tensorsolve(&ts_a, &ts_b, None).expect("tensorsolve should succeed"); // [2.0, 2.0]

let ti_a = Array::from_vec(vec![4.0, 0.0, 0.0, 2.0]).reshape(&[2, 2]);
let a_inv = tensorinv(&ti_a, 1).expect("tensorinv should succeed"); // [[0.25, 0], [0, 0.5]]
```
`norm` also gained ord `-2` and `"nuc"` (nuclear norm).

### Masked arrays (`numpy.ma` parity)

**NumPy:**
```python
import numpy.ma as ma

m = ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
m.std(); m.var(); m.median(); m.ptp()
m.argmin()   # returns 0 for an all-masked array -- see the note below
```

**NumRS2** (`numrs2::masked::MaskedArray`) — *signature-verified*:
```rust
use numrs2::masked::MaskedArray;
use numrs2::array::Array;

let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let mask = Array::from_vec(vec![false, true, false, false]);
let m = MaskedArray::new(data, Some(mask), None)?;

let s = m.std(None, false)?;
let med = m.median(None, false)?;
let idx = m.argmin(None, false)?;   // Err if every element in the reduced lane is masked
```

**Deviation from `numpy.ma`:** a fully-masked lane's `argmin`/`argmax` is an `Err` here, not
NumPy's degenerate (and silently ambiguous) index `0` — see TODO.md's "Intentional NumPy
deviations". `std`/`var`/`prod`/`ptp`/`any`/`all`/`sort`/`cumsum`/`dot`/`concatenate` are also now
implemented, plus `Sub`/`Div`/comparison operators and `axis=` support throughout.

### Polynomial classes (`numpy.polynomial`)

**NumPy:**
```python
from numpy.polynomial import Chebyshev

p = Chebyshev([1.0, 2.0, 3.0])   # 1 + 2*T_1(x) + 3*T_2(x), domain/window default to [-1, 1]
y = p(0.5)
roots = p.roots()
```

**NumRS2** (`numrs2::new_modules::polynomial`) — *signature-verified*:
```rust
use numrs2::new_modules::polynomial::Chebyshev;

let p = Chebyshev::from_coef(vec![1.0, 2.0, 3.0]); // default domain/window [-1, 1]
let y = p.eval(0.5);
let roots = p.roots()?;
```
`Legendre`, `Hermite` (physicists'), `HermiteE` (probabilists'), and `Laguerre` (default domain
`[0, 1]`, matching NumPy) share the same `new`/`from_coef`/`eval`/`eval_array`/`fit`/`roots`/
`deriv`/`integ` interface — coefficients are **ascending order**, matching
`numpy.polynomial.Chebyshev` etc. (the opposite convention from this crate's older, descending-order
`polyfit`/`polyval`/`polyder` free functions).

### Random: `SeedSequence`, `Generator.spawn`, `Philox`, `SFC64`, `permuted`

**NumPy:**
```python
import numpy as np

ss = np.random.SeedSequence(42)
children = ss.spawn(4)
rngs = [np.random.Generator(np.random.Philox(s)) for s in children]
a.mean(axis=1)  # for comparison
shuffled = rng.permuted(a, axis=1)
```

**NumRS2** (`numrs2::random`) — *doctest for the base case (`sfc64_rng`/`philox_seed_rng`), from
their own examples; `spawn`/`permuted` composition signature-verified*:
```rust
use numrs2::random::{philox_seed_rng, philox_from_seed_sequence, sfc64_rng, SeedSequence};

// Philox4x64 reproduces `np.random.Philox(seed=seed)`'s raw output exactly.
let rng = philox_seed_rng(42);
let random_array = rng.random::<f64>(&[3, 3])?;

// SFC64, similarly seeded:
let rng2 = sfc64_rng();
let random_array2 = rng2.random::<f64>(&[3, 3])?;

// SeedSequence + spawn, for independent parallel streams:
let parent = philox_from_seed_sequence(SeedSequence::new(42));
let children = parent.spawn(4)?;   // Vec<Generator<Philox4x64BitGenerator>>

// Generator::permuted: independent per-lane shuffle (see the Exactness caveat in TODO.md --
// this is not bit-identical to `np.random.Generator.permuted`, on any bit generator).
let shuffled = rng.permuted(&a, Some(1))?;
```

## Further Resources

- [API Documentation](https://docs.rs/numrs2)
- [Changelog](CHANGELOG.md) — full, verified per-release list of additions, changes, and fixes
- [Development Roadmap](TODO.md) — current status and known deferred items
- [Benchmarking Guide](BENCHMARKING.md)
- [Example Gallery](examples/README.md)
- [NumPy Documentation](https://numpy.org/doc/stable/)
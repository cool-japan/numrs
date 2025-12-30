# SciRS2 Ecosystem Policy for NumRS2

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**NumRS2 is part of the SciRS2 ecosystem and MUST follow all SciRS2 ecosystem policies.** This document establishes how NumRS2 integrates with and extends SciRS2 while maintaining NumPy API compatibility.

## Table of Contents

1. [Core Ecosystem Principles](#core-ecosystem-principles)
2. [Pure Rust Migration (v0.1.1)](#pure-rust-migration-v011)
3. [Dependency Abstraction Policy](#dependency-abstraction-policy)
4. [Technical Policies](#technical-policies)
5. [NumRS2-Specific Implementation](#numrs2-specific-implementation)
6. [Dependency Configuration](#dependency-configuration)
7. [Enforcement and Compliance](#enforcement-and-compliance)

---

## Core Ecosystem Principles

### 1. NumRS2's Role in the SciRS2 Ecosystem

- NumRS2 is a **SciRS2 ecosystem project** providing NumPy-compatible API
- NumRS2 builds upon SciRS2's scientific computing foundation
- NumRS2 **MUST follow all SciRS2 ecosystem policies** (see SciRS2 repository SCIRS2_POLICY.md)
- NumRS2 extends SciRS2 capabilities with NumPy-specific API patterns

### 2. Architectural Hierarchy

```
NumRS2 (NumPy-compatible API layer)
    ↓ builds upon and follows policies of
SciRS2 Ecosystem (scirs2-core, scirs2-stats, scirs2-linalg, etc.)
    ↓ provides abstractions for
External Libraries (ndarray, rand, OxiBLAS, etc.)
```

### 3. Dependency Abstraction Policy (Mandatory)

- NumRS2 **MUST NEVER** use external dependencies directly (rand, ndarray, nalgebra, etc.)
- NumRS2 **MUST** use SciRS2-Core abstractions for all external functionality
- NumRS2 **MUST** follow the layered architecture of the SciRS2 ecosystem

---

## Pure Rust Migration (v0.1.1)

**Major architectural changes in v0.1.1 (December 2025):**

### OxiBLAS Migration - Pure Rust BLAS/LAPACK

**REMOVED Dependencies:**
- ❌ `ndarray-linalg` - Replaced with scirs2-linalg independent implementation
- ❌ `openblas-src` / `blas-src` / `lapack-src` - System BLAS libraries
- ❌ `accelerate-src` - macOS Accelerate Framework bindings
- ❌ `intel-mkl-src` - Intel MKL bindings
- ❌ `netlib-src` - Netlib reference implementation

**ADDED Dependencies (via scirs2-core):**
- ✅ `oxiblas-ndarray` v0.1.2+ - Pure Rust ndarray integration
- ✅ `oxiblas-blas` v0.1.2+ - Pure Rust BLAS implementation
- ✅ `oxiblas-lapack` v0.1.2+ - Pure Rust LAPACK implementation (supports Complex<f64>)

**Benefits:**
- 🚀 **Zero System Dependencies** - No need to install OpenBLAS, MKL, or system BLAS
- 🔧 **Easy Cross-Compilation** - Pure Rust works on all platforms
- 📦 **Simplified Builds** - No C/Fortran compiler required
- 🔒 **Complete Control** - Full Rust ecosystem integration
- ⚡ **SIMD Optimized** - Performance competitive with native BLAS

### Oxicode Migration - SIMD-Optimized Serialization

**REMOVED Dependencies:**
- ❌ `bincode` - Generic binary serialization

**ADDED Dependencies (COOLJAPAN Policy):**
- ✅ `oxicode` v0.1.1+ - SIMD-optimized binary serialization
- ✅ `oxicode_derive` - Derive macros for custom types

**Benefits:**
- ⚡ **SIMD Acceleration** - Up to 4x faster than bincode
- 🎯 **Scientific Data Optimized** - Specialized for numeric arrays
- 🔒 **Type Safe** - Compile-time serialization verification

---

## Dependency Abstraction Policy

### Core Principle: Layered Abstraction Architecture

The SciRS2 ecosystem follows a strict layered architecture where only the core crate can use external dependencies directly, while all other crates (including NumRS2) must use SciRS2-Core abstractions.

### Policy: No Direct External Dependencies

**Applies to:** All NumRS2 code
- All production code in `src/`
- All tests in `tests/`
- All examples in `examples/`
- All benchmarks in `bench/`

#### Prohibited Direct Dependencies in Cargo.toml:

```toml
# ❌ FORBIDDEN in NumRS2
[dependencies]
rand = { workspace = true }              # ❌ Use scirs2-core instead
rand_distr = { workspace = true }        # ❌ Use scirs2-core instead
ndarray = { workspace = true }           # ❌ Use scirs2-core instead
ndarray-rand = { workspace = true }      # ❌ Use scirs2-core instead
ndarray-linalg = { workspace = true }    # ❌ REMOVED v0.1.1 - scirs2-linalg independent
num-traits = { workspace = true }        # ❌ Use scirs2-core instead
num-complex = { workspace = true }       # ❌ Use scirs2-core instead
nalgebra = { workspace = true }          # ❌ Use scirs2-core instead
rayon = { workspace = true }             # ❌ Use scirs2-core instead
bincode = { workspace = true }           # ❌ REMOVED v0.1.1 - Use oxicode instead
openblas-src = { workspace = true }      # ❌ REMOVED v0.1.1 - Use OxiBLAS instead
blas-src = { workspace = true }          # ❌ REMOVED v0.1.1 - Use OxiBLAS instead
lapack-src = { workspace = true }        # ❌ REMOVED v0.1.1 - Use OxiBLAS instead
```

#### Required Core Dependencies:

```toml
# ✅ REQUIRED in NumRS2
[dependencies]
scirs2-core = { workspace = true, features = ["random", "array", "simd", "parallel", "linalg"] }
scirs2-stats = { workspace = true }
scirs2-linalg = { workspace = true }
scirs2-fft = { workspace = true }
scirs2-signal = { workspace = true }
scirs2-special = { workspace = true }
scirs2-ndimage = { workspace = true }
scirs2-spatial = { workspace = true }
```

#### Prohibited Direct Imports in Code:

```rust
// ❌ FORBIDDEN in NumRS2
use rand::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Beta, Normal, StudentT};  // Use scirs2_core::random instead
use ndarray::*;
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s};  // Macros now available through scirs2_core
use num_complex::Complex;
use num_traits::*;
use rayon::prelude::*;
// etc.
```

#### Required SciRS2-Core Abstractions:

```rust
// ✅ REQUIRED in all NumRS2 code (production, tests, examples, benchmarks)

// === Random Number Generation ===
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
// Includes: thread_rng, Rng, SliceRandom, etc.
// All distributions: Beta, Cauchy, ChiSquared, Normal, StudentT, Weibull, etc.

// === Array Operations ===
use scirs2_core::ndarray::*;          // Complete ndarray ecosystem
// Includes: Array, Array1, Array2, ArrayView, array!, s!, azip! macros
// Includes: ndarray-rand, ndarray-stats, ndarray-npy when array feature enabled
// NOTE: ndarray-linalg removed v0.1.1 - scirs2-linalg provides independent implementation

// === Numerical Traits ===
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
// Includes: Float, Zero, One, Num, Complex, etc.

// === SIMD Operations ===
use scirs2_core::simd_ops::*;         // SimdUnifiedOps trait

// === Parallel Operations ===
use scirs2_core::parallel_ops::*;     // Rayon through core

// === Advanced Types ===
use scirs2_core::array::*;            // Scientific array types (MaskedArray, RecordArray)
use scirs2_core::linalg::*;           // Linear algebra (via OxiBLAS)
```

### Complete Dependency Mapping

| External Crate | SciRS2-Core Module | Note |
|----------------|-------------------|------|
| `rand` | `scirs2_core::random` | Full functionality |
| `rand_distr` | `scirs2_core::random` | All distributions |
| `ndarray` | `scirs2_core::ndarray` | Full functionality |
| `ndarray-rand` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-stats` | `scirs2_core::ndarray` | Via `array` feature |
| ~~`ndarray-linalg`~~ | N/A | **REMOVED** - scirs2-linalg independent |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |
| `rayon` | `scirs2_core::parallel_ops` | Parallel processing |
| `oxiblas-*` | `scirs2_core::linalg` | Pure Rust BLAS/LAPACK (v0.1.2+) |
| ~~`bincode`~~ | N/A | **REPLACED** by `oxicode` v0.1.1+ |
| `oxicode` | Direct usage | COOLJAPAN Policy - SIMD serialization |

---

## Technical Policies

All technical policies from the SciRS2 ecosystem SCIRS2_POLICY.md apply to NumRS2.

### 1. SIMD Operations Policy

**Mandatory Rules:**
1. **ALWAYS use `scirs2_core::simd_ops::SimdUnifiedOps`** for all SIMD operations
2. **NEVER implement custom SIMD** code in NumRS2
3. **ALWAYS provide scalar fallbacks** through the unified trait

```rust
// ✅ CORRECT - NumRS2 usage
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn numpy_add<T: SimdUnifiedOps>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> Array1<T> {
    T::simd_add(a, b)
}

// ❌ FORBIDDEN - Custom SIMD
// use wide::f32x8;  // FORBIDDEN in NumRS2
```

### 2. Parallel Processing Policy

**Mandatory Rules:**
1. **ALWAYS use `scirs2_core::parallel_ops`** for all parallel operations
2. **NEVER add direct `rayon` dependency** to NumRS2
3. **ALWAYS import via `use scirs2_core::parallel_ops::*`**

```rust
// ✅ CORRECT
use scirs2_core::parallel_ops::*;

let results: Vec<f64> = (0..n)
    .into_par_iter()
    .map(|i| compute(i))
    .collect();

// ❌ FORBIDDEN
// use rayon::prelude::*;  // FORBIDDEN
```

### 3. BLAS Operations Policy

**Mandatory Rules:**
1. **ALL BLAS operations go through `scirs2-core`**
2. **NEVER add direct BLAS dependencies** (openblas-src, blas-src, etc.)
3. **Backend: OxiBLAS (Pure Rust)** - v0.1.2+ via scirs2-core

**v0.1.1+ (Current):**
- ✅ **All Platforms: OxiBLAS (Pure Rust BLAS/LAPACK)** - Default and recommended
  - No system dependencies required
  - Cross-compilation friendly
  - Complete Rust ecosystem integration
  - SIMD optimized (AVX2/NEON)

**Legacy (Removed in v0.1.1):**
- ❌ ~~OpenBLAS~~ **REMOVED**
- ❌ ~~Accelerate Framework~~ **REMOVED**
- ❌ ~~Intel MKL~~ **REMOVED**

### 4. Platform Detection Policy

```rust
// ✅ CORRECT
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::detect();
if caps.simd_available {
    // Use SIMD path
}

// ❌ FORBIDDEN
// if is_x86_feature_detected!("avx2") {  // FORBIDDEN
```

### 5. Error Handling Policy

```rust
// ✅ CORRECT
use scirs2_core::error::CoreError;
use scirs2_core::validation::{check_positive, check_finite};

#[derive(Debug, thiserror::Error)]
pub enum NumRs2Error {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("NumPy API error: {0}")]
    NumpyApi(String),
}

// Use core validation
check_positive(value, "parameter_name")?;
check_finite(&array)?;
```

---

## NumRS2-Specific Implementation

### Required Import Patterns (Mandatory)

```rust
// ✅ REQUIRED - All NumRS2 code, tests, examples, benchmarks
use scirs2_core::random::*;           // All random functionality
use scirs2_core::ndarray::*;          // All ndarray + macros (array!, s!)
use scirs2_core::simd_ops::*;         // SIMD operations
use scirs2_core::parallel_ops::*;     // Parallel operations
use scirs2_stats::*;                  // Statistical functions
use scirs2_linalg::*;                 // Linear algebra

// ❌ FORBIDDEN - Direct external dependencies
// use rand::*;                       // FORBIDDEN
// use ndarray::*;                    // FORBIDDEN
// use rayon::prelude::*;             // FORBIDDEN
// use openblas_src::*;               // FORBIDDEN
// use bincode::*;                    // FORBIDDEN - Use oxicode instead
```

### Random Number Generation

```rust
// ✅ CORRECT - All distributions available through scirs2_core
use scirs2_core::random::*;

pub fn generate_normal_samples(n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

// ❌ FORBIDDEN - Direct rand usage
// use rand::thread_rng;  // FORBIDDEN
// use rand_distr::Normal;  // FORBIDDEN
```

### Array Operations

```rust
// ✅ CORRECT - All array operations through scirs2_core::ndarray
use scirs2_core::ndarray::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn numpy_add<T: SimdUnifiedOps>(
    a: &ArrayView1<T>,
    b: &ArrayView1<T>
) -> Array1<T> {
    // Use SciRS2's SIMD-optimized operations
    T::simd_add(a, b)
}

pub fn numpy_matmul<T: SimdUnifiedOps>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>
) -> Array2<T> {
    // Use SciRS2's BLAS-backed operations (via OxiBLAS)
    a.dot(b)
}
```

### Test and Example Code (Mandatory)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, s};  // array! and s! macros work
    use scirs2_core::random::*;

    #[test]
    fn test_numpy_compatible_op() {
        let mut rng = thread_rng();
        let arr = array![1.0, 2.0, 3.0];  // array! macro from scirs2_core
        let slice = arr.slice(s![..]);    // s! macro from scirs2_core

        // All operations through scirs2-core
        assert_eq!(arr.len(), 3);
    }
}
```

---

## Dependency Configuration

### Required Cargo.toml Configuration

```toml
[dependencies]
# ===================================================================
# SCIRS2 ECOSYSTEM DEPENDENCIES (MANDATORY) - MUST USE WORKSPACE
# ===================================================================
# ALL external functionality (ndarray, rand, rayon, BLAS) goes through scirs2-core
scirs2-core = { workspace = true, features = ["random", "array", "simd", "parallel", "linalg"] }
scirs2-stats = { workspace = true }
scirs2-linalg = { workspace = true }
scirs2-fft = { workspace = true }
scirs2-signal = { workspace = true }
scirs2-special = { workspace = true }
scirs2-ndimage = { workspace = true }
scirs2-spatial = { workspace = true }

# ===================================================================
# ALLOWED DEPENDENCIES (Non-conflicting)
# ===================================================================
num-traits = "0.2.19"  # Numeric traits (compatible with SciRS2)
thiserror = "2.0.17"   # Error handling
oxicode = { version = "0.1.1", features = ["serde"] }  # COOLJAPAN Policy

# ===================================================================
# FORBIDDEN (Removed per SCIRS2 POLICY)
# ===================================================================
# ❌ ndarray = "0.17"        # FORBIDDEN - Use scirs2_core::ndarray
# ❌ rand = "0.9"            # FORBIDDEN - Use scirs2_core::random
# ❌ rand_distr = "0.5"      # FORBIDDEN - Use scirs2_core::random
# ❌ rayon = "1.11"          # FORBIDDEN - Use scirs2_core::parallel_ops
# ❌ num-complex = "0.4"     # FORBIDDEN - Use scirs2_core::numeric
# ❌ nalgebra = "0.33"       # FORBIDDEN - BLAS through scirs2-core
# ❌ bincode = "2.0"         # FORBIDDEN - Use oxicode instead
# ❌ openblas-src = "*"      # FORBIDDEN - Use OxiBLAS (pure Rust)
# ❌ ndarray-linalg = "*"    # FORBIDDEN - Use scirs2-linalg
```

### Workspace Configuration

```toml
[workspace.dependencies]
scirs2-core = { version = "0.1.1" }
scirs2-stats = { version = "0.1.1" }
scirs2-linalg = { version = "0.1.1" }
scirs2-fft = { version = "0.1.1" }
scirs2-signal = { version = "0.1.1" }
scirs2-special = { version = "0.1.1" }
scirs2-ndimage = { version = "0.1.1" }
scirs2-spatial = { version = "0.1.1" }
```

---

## Enforcement and Compliance

### Mandatory Compliance Checks

1. **No Direct External Dependencies**
   - ❌ Direct imports of `rand`, `ndarray`, `rayon`, BLAS libraries FORBIDDEN
   - ✅ All external functionality through `scirs2_core` abstractions

2. **All Code Must Use SciRS2 Abstractions**
   - Production code, tests, examples, benchmarks
   - No exceptions for "temporary" or "experimental" code

3. **Code Review Requirements**
   - All PRs checked for policy violations
   - Reject any direct external dependency usage
   - Ensure proper scirs2_core abstractions

### CI Pipeline Checks

```bash
# Check for forbidden imports
rg "use (rand|ndarray|rayon|openblas|bincode)::" --type rust
# Should return no matches in NumRS2 code
```

---

## Benefits of Ecosystem Compliance

By following the SciRS2 ecosystem policies, NumRS2 gains:

1. **Unified Performance**: Benefit from SciRS2's optimizations automatically
2. **Easier Maintenance**: Updates in scirs2-core benefit NumRS2 immediately
3. **Consistent Behavior**: Same optimizations as other SciRS2 projects
4. **Better Testing**: Leverage scirs2-core's testing of critical operations
5. **Improved Portability**: Platform-specific code handled by core
6. **Version Control**: Simplified dependency management through workspace
7. **Type Safety**: Consistent types across the SciRS2 ecosystem
8. **NumPy Compatibility**: Focus NumRS2 on API, let SciRS2 handle computation
9. **Pure Rust**: No C/C++ dependencies via OxiBLAS
10. **Cross-Platform**: Easy compilation on all platforms

---

## Current Status (v0.1.1)

### Policy Compliance
- ✅ **100% Compliant** - All code uses SciRS2 abstractions
- ✅ **Zero Direct Dependencies** - All external libs through scirs2-core
- ✅ **Pure Rust** - OxiBLAS v0.1.2, Oxicode v0.1.1
- ✅ **Production Ready** - 1,111+ tests passing

### SciRS2 Integration
- ✅ scirs2-core v0.1.1 - Foundation (random, ndarray, SIMD, parallel, linalg via OxiBLAS)
- ✅ scirs2-stats v0.1.1 - Statistical operations
- ✅ scirs2-linalg v0.1.1 - Linear algebra (independent implementation)
- ✅ scirs2-fft v0.1.1 - FFT operations
- ✅ scirs2-signal v0.1.1 - Signal processing
- ✅ scirs2-special v0.1.1 - Special functions
- ✅ scirs2-ndimage v0.1.1 - N-dimensional image processing
- ✅ scirs2-spatial v0.1.1 - Spatial algorithms

---

## Conclusion

**NumRS2 is part of the SciRS2 ecosystem and MUST follow all SciRS2 policies.**

NumRS2's mission is to provide a NumPy-compatible API layer on top of SciRS2's scientific computing foundation. This requires strict adherence to SciRS2 ecosystem policies to ensure consistency, performance, and maintainability.

For complete technical policies, see the SciRS2 repository SCIRS2_POLICY.md

---

**Document Version**: 3.0 - Pure Rust Era (v0.1.1)
**Effective Date**: SciRS2 v0.1.1 (December 2025)
**Last Updated**: 2025-12-30
**Status**: Active - Mandatory Compliance
**Parent Policy**: SciRS2 SCIRS2_POLICY.md v3.0.0

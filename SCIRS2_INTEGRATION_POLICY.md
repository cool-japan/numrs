# SciRS2 Ecosystem Policy for NumRS2

## 📦 Version Tracking Strategy

**NumRS2 tracks SciRS2 development versions using local path dependencies.**

This allows NumRS2 to:
- Stay synchronized with latest SciRS2 features and API changes
- Catch compatibility issues early before crates.io releases
- Provide feedback to SciRS2 team during development

**Current Configuration:**
```toml
[workspace.dependencies]
scirs2-core = { path = "../scirs/scirs2-core", version = "0.1.0-beta.4" }
scirs2-stats = { path = "../scirs/scirs2-stats", version = "0.1.0-beta.4" }
scirs2-linalg = { path = "../scirs/scirs2-linalg", version = "0.1.0-beta.4" }
```

**Note**: When SciRS2 beta.4 is published to crates.io, these can be switched back to registry dependencies.

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**NumRS2 is part of the SciRS2 ecosystem and MUST follow all SciRS2 ecosystem policies.** This document establishes how NumRS2 integrates with and extends SciRS2 while maintaining NumPy API compatibility.

## Core Ecosystem Principles

### 1. **NumRS2's Role in the SciRS2 Ecosystem**
- NumRS2 is a **SciRS2 ecosystem project** providing NumPy-compatible API
- NumRS2 builds upon SciRS2's scientific computing foundation
- NumRS2 **MUST follow all SciRS2 ecosystem policies** (see main SCIRS2_POLICY.md)
- NumRS2 extends SciRS2 capabilities with NumPy-specific API patterns

### 2. **Dependency Abstraction Policy (Mandatory)**
- NumRS2 **MUST NEVER** use external dependencies directly (rand, ndarray, nalgebra, etc.)
- NumRS2 **MUST** use SciRS2-Core abstractions for all external functionality
- NumRS2 **MUST** follow the layered architecture of the SciRS2 ecosystem

### 3. **Architectural Hierarchy**
```
NumRS2 (NumPy-compatible API layer)
    ↓ builds upon and follows policies of
SciRS2 Ecosystem (scirs2-core, scirs2-stats, scirs2-linalg, etc.)
    ↓ provides abstractions for
External Libraries (ndarray, rand, BLAS, etc.)
```

## 🔄 Beta.4 API Changes and Fixes

### Gamma Distribution Fix
**Issue**: Beta.3 had a scale parameter bug in Gamma distribution (passing `1/scale` instead of `scale`).

**Resolution**: Beta.4 fixed this bug. NumRS2 removed the workaround:

```rust
// ❌ OLD (Beta.3 workaround):
let corrected_scale = 1.0 / scale_f64;  // Workaround for beta.3 bug
let dist = Gamma::new(shape_f64, corrected_scale, 0.0)?;

// ✅ NEW (Beta.4 - bug fixed):
let dist = Gamma::new(shape_f64, scale_f64, 0.0)?;  // Direct scale parameter
```

**Impact**: All gamma distribution tests now pass with correct parameterization.

### API Changes from Beta.3 → Beta.4
1. **Gamma Distribution**: Fixed scale parameter (no longer inverted)
2. **Feature names**: Confirmed `"linalg"` feature (not `"blas"`)
3. **Complex types**: Direct re-export as `scirs2_core::Complex` (not `scirs2_core::complex::Complex`)

## Part I: SciRS2 Ecosystem Integration

### Required Dependencies (Mandatory Core Abstractions)

NumRS2 **MUST** use the following SciRS2-Core modules for all external functionality:

#### **MANDATORY: scirs2-core abstractions**
```rust
// ✅ REQUIRED - All external functionality through scirs2-core
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
use scirs2_core::ndarray::*;          // Complete ndarray functionality + macros (array!, s!)
use scirs2_core::array::*;            // Scientific array types (MaskedArray, etc.)
use scirs2_core::complex::*;          // Instead of num_complex::*
use scirs2_core::linalg::*;           // Core linear algebra

// ❌ FORBIDDEN - Direct external dependencies
// use rand::*;                       // FORBIDDEN
// use ndarray::*;                    // FORBIDDEN
// use num_complex::*;                // FORBIDDEN
```

### Required SciRS2 Crates for NumRS2

#### **ESSENTIAL (Always Required)**

1. **`scirs2-core`** - FOUNDATION (v0.1.0-beta.3+)
   - **Use Cases**: All external dependency abstractions (rand, ndarray, BLAS, SIMD)
   - **NumRS2 Usage**: Core array operations, random number generation, SIMD, parallel ops
   - **Status**: ✅ REQUIRED - Mandatory foundation
   - **Policy**: ALL imports must go through scirs2-core abstractions

2. **`scirs2-stats`** - STATISTICAL OPERATIONS
   - **Use Cases**: Statistical distributions, descriptive statistics, hypothesis testing
   - **NumRS2 Usage**: `numpy.random` compatibility, statistical functions
   - **Status**: ✅ REQUIRED - NumPy statistical compatibility

3. **`scirs2-linalg`** - LINEAR ALGEBRA
   - **Use Cases**: Matrix operations, decompositions, eigenvalue problems
   - **NumRS2 Usage**: `numpy.linalg` compatibility
   - **Status**: ✅ REQUIRED - NumPy linalg compatibility

#### **CONDITIONALLY REQUIRED (Add as NumRS2 features expand)**

4. **`scirs2-autograd`** - AUTOMATIC DIFFERENTIATION
   - **Status**: ⚠️ Add when NumRS2 implements gradient-based operations
   - **Note**: Basic ndarray operations available via `scirs2_core::ndarray`

5. **`scirs2-signal`** - SIGNAL PROCESSING
   - **Status**: ⚠️ Add when implementing `numpy.fft` compatibility

6. **`scirs2-sparse`** - SPARSE MATRICES
   - **Status**: ⚠️ Add when implementing `scipy.sparse` compatibility

#### **NOT CURRENTLY REQUIRED**
- `scirs2-neural`, `scirs2-graph`, `scirs2-cluster` - Outside NumRS2's core scope

## Part II: Technical Policies (NumRS2 Must Follow)

All technical policies from the main SCIRS2_POLICY.md apply to NumRS2. Key policies:

### 1. SIMD Operations Policy

**Mandatory Rules for NumRS2:**
1. **ALWAYS use `scirs2_core::simd_ops::SimdUnifiedOps`** for all SIMD operations
2. **NEVER implement custom SIMD** code in NumRS2 modules
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

**Mandatory Rules for NumRS2:**
1. **ALWAYS use `scirs2_core::parallel_ops`** for all parallel operations
2. **NEVER add direct `rayon` dependency** to NumRS2 Cargo.toml
3. **ALWAYS import via `use scirs2_core::parallel_ops::*`**

```rust
// ✅ CORRECT - NumRS2 usage
use scirs2_core::parallel_ops::*;

let results: Vec<f64> = (0..n)
    .into_par_iter()
    .map(|i| compute(i))
    .collect();

// ❌ FORBIDDEN - Direct Rayon
// use rayon::prelude::*;  // FORBIDDEN
```

### 3. BLAS Operations Policy

**Mandatory Rules for NumRS2:**
1. **ALL BLAS operations go through `scirs2-core`**
2. **NEVER add direct BLAS dependencies** (openblas-src, blas-src, etc.)
3. **Backend selection handled by scirs2-core** platform configuration

### 4. Platform Detection Policy

```rust
// ✅ CORRECT - NumRS2 usage
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::detect();
if caps.simd_available {
    // Use SIMD path
}

// ❌ FORBIDDEN - Custom detection
// if is_x86_feature_detected!("avx2") {  // FORBIDDEN
```

### 5. Error Handling Policy

```rust
// ✅ CORRECT - NumRS2 errors extend core errors
use scirs2_core::error::CoreError;
use scirs2_core::validation::{check_positive, check_finite};

#[derive(Debug, thiserror::Error)]
pub enum NumRS2Error {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("NumPy API error: {0}")]
    NumpyApi(String),
}

// Use core validation
check_positive(value, "parameter_name")?;
check_finite(&array)?;
```

## Part III: NumRS2-Specific Implementation Guidelines

### Import Patterns (Mandatory)

```rust
// ✅ REQUIRED - All NumRS2 code, tests, examples
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

### Random Number Generation (Updated for v0.1.0-beta.3+)

```rust
// ✅ CORRECT - All distributions available
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

## Part IV: NumRS2-Specific Implementation Patterns

### NumPy API Compatibility Layer

NumRS2's role is to provide NumPy-compatible APIs while using SciRS2 as the computational backend.

```rust
// ✅ CORRECT - NumRS2 wraps SciRS2 with NumPy API
use scirs2_core::ndarray::*;
use scirs2_stats::*;

/// NumPy-compatible mean function
pub fn mean<T: ScientificNumber>(arr: &ArrayView1<T>) -> T {
    // Use SciRS2's optimized implementation
    scirs2_stats::mean(arr)
}

/// NumPy-compatible std function (with ddof parameter)
pub fn std<T: ScientificNumber>(arr: &ArrayView1<T>, ddof: usize) -> T {
    // Wrap SciRS2 with NumPy-compatible parameters
    scirs2_stats::std_with_ddof(arr, ddof)
}
```

### Array Operations Pattern

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
    // Use SciRS2's BLAS-backed operations
    a.dot(b)
}
```

### Statistical Functions Pattern

```rust
// ✅ CORRECT - Statistical operations through scirs2-stats
use scirs2_stats::*;
use scirs2_core::ndarray::*;

pub mod numpy_stats {
    use super::*;

    /// NumPy-compatible statistics namespace
    pub fn mean<T: ScientificNumber>(arr: &ArrayView1<T>) -> T {
        scirs2_stats::mean(arr)
    }

    pub fn var<T: ScientificNumber>(arr: &ArrayView1<T>, ddof: usize) -> T {
        scirs2_stats::variance_with_ddof(arr, ddof)
    }
}
```

### Linear Algebra Pattern

```rust
// ✅ CORRECT - Linear algebra through scirs2-linalg
use scirs2_linalg::*;
use scirs2_core::ndarray::*;

pub mod numpy_linalg {
    use super::*;

    /// NumPy-compatible linalg namespace
    pub fn svd<T: Float>(matrix: &ArrayView2<T>) -> (Array2<T>, Array1<T>, Array2<T>) {
        scirs2_linalg::decomposition::svd(matrix)
    }

    pub fn inv<T: Float>(matrix: &ArrayView2<T>) -> Array2<T> {
        scirs2_linalg::matrix::inverse(matrix)
    }
}
```

### Performance Hierarchy (Simplified)

NumRS2 follows SciRS2's performance policies:
1. **Use SciRS2's optimized implementations** (SIMD, BLAS, parallel)
2. **Wrap with NumPy-compatible API** (parameter names, behavior)
3. **Document any behavioral differences** from NumPy

## Part V: Dependency Configuration

### Required Cargo.toml Configuration

```toml
[dependencies]
# MANDATORY: SciRS2 ecosystem dependencies (NOT optional)
scirs2-core = { workspace = true, features = ["random", "ndarray", "simd", "parallel", "blas"] }
scirs2-stats = { workspace = true }
scirs2-linalg = { workspace = true }

# FORBIDDEN: No direct external dependencies
# rand = "0.8"              # FORBIDDEN - use scirs2_core::random
# ndarray = "0.15"          # FORBIDDEN - use scirs2_core::ndarray
# rayon = "1.10"            # FORBIDDEN - use scirs2_core::parallel_ops
# openblas-src = "0.10"     # FORBIDDEN - BLAS through scirs2-core
```

### Workspace Configuration (Cargo.toml at workspace root)

```toml
[workspace.dependencies]
scirs2-core = "0.1.0-beta.3"
scirs2-stats = "0.1.0-beta.3"
scirs2-linalg = "0.1.0-beta.3"
```

## Part VI: Enforcement and Compliance

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

### CI Pipeline Checks (Planned)

```bash
# Check for forbidden imports
rg "use (rand|ndarray|rayon|openblas)" --type rust
# Should return no matches in NumRS2 code
```

## Part VII: Migration Status and Next Steps

### Current Integration Status (v0.1.0-beta.3)

- ✅ `scirs2-core` - Foundation (random, ndarray, SIMD, parallel, BLAS)
- ✅ `scirs2-stats` - Statistical operations
- ✅ `scirs2-linalg` - Linear algebra operations

### Migration Required

NumRS2 code must be audited and updated to:
1. Remove all direct `rand` imports → use `scirs2_core::random`
2. Remove all direct `ndarray` imports → use `scirs2_core::ndarray`
3. Remove all direct `rayon` imports → use `scirs2_core::parallel_ops`
4. Update all tests and examples to use SciRS2 abstractions
5. Update Cargo.toml to remove forbidden dependencies

### Refactoring Priority

Follow the priority order from SCIRS2_POLICY.md:
1. SIMD implementations → `scirs2_core::simd_ops`
2. Parallel operations → `scirs2_core::parallel_ops`
3. Random number generation → `scirs2_core::random`
4. Array operations → `scirs2_core::ndarray`
5. BLAS operations → through scirs2-core
6. Error types → base on `scirs2_core::error`

## Part VIII: Benefits of Ecosystem Compliance

By following the SciRS2 ecosystem policies, NumRS2 gains:

1. **Unified Performance**: Benefit from SciRS2's optimizations automatically
2. **Easier Maintenance**: Updates in scirs2-core benefit NumRS2 immediately
3. **Consistent Behavior**: Same optimizations as other SciRS2 projects
4. **Better Testing**: Leverage scirs2-core's testing of critical operations
5. **Improved Portability**: Platform-specific code handled by core
6. **Version Control**: Simplified dependency management through workspace
7. **Type Safety**: Consistent types across the SciRS2 ecosystem
8. **NumPy Compatibility**: Focus NumRS2 on API, let SciRS2 handle computation

## Conclusion

**NumRS2 is part of the SciRS2 ecosystem and MUST follow all SciRS2 policies.**

NumRS2's mission is to provide a NumPy-compatible API layer on top of SciRS2's scientific computing foundation. This requires strict adherence to SciRS2 ecosystem policies to ensure consistency, performance, and maintainability.

For complete technical policies, see: `~/work/scirs/SCIRS2_POLICY.md`

---

**Document Version**: 2.0 - Aligned with SciRS2 Ecosystem Policy
**Effective Date**: SciRS2 v0.1.0-beta.3
**Last Updated**: 2025-09-30
**Status**: Active - Mandatory Compliance
**Owner**: NumRS2 Architecture Team
**Parent Policy**: SciRS2 SCIRS2_POLICY.md v2.0.0
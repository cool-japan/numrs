# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands
- Build project: `cargo build`
- Run all tests: `cargo test`
- Run specific test: `cargo test <test_name>`
- Run examples: `cargo run --example basic_usage`
- Benchmarks: `cargo bench`

## Important Workflow Guidelines
- **Always verify successful build before committing**: Run `cargo build` and `cargo test` to confirm code compiles and tests pass before any commit
- **Commit & Push**: Only after verifying successful builds and tests, create commits with descriptive messages and push changes
- **Documentation updates**: Only update documentation (especially TODO.md) after confirming successful builds
- **Work-in-progress code**: Mark features as "in progress" in documentation if build is not yet successful
- **Warning elimination**: Treat warnings as errors - resolve all warnings before committing
- **Code verification**: Thoroughly verify build success before updating documentation and committing changes

## Code Style Guidelines
- **Imports**: standard lib → third-party → internal crate (`crate::`)
- **Error handling**: Use `Result<T>` and `?` operator, never unwrap in production code
- **Naming**: PascalCase for types, snake_case for functions/variables
- **Formatting**: 4-space indentation, trailing commas in multi-line structs
- **Documentation**: Triple-slash `///` doc comments for all public items
- **Performance**: Use SIMD when possible, remember to check alignment
- **Testing**: Write unit tests for all new functionality

Always leverage existing abstractions: array operations, BLAS integration, and SIMD optimizations.

## SCIRS2 Ecosystem Policy (MANDATORY)

**NumRS2 is part of the SciRS2 ecosystem and MUST follow all SciRS2 ecosystem policies.**

### Core Requirements

1. **Mandatory SciRS2 Foundation**: NumRS2 builds upon SciRS2's scientific computing foundation
2. **NO Direct External Dependencies**: NEVER use `rand`, `ndarray`, `rayon`, BLAS libraries directly
3. **ALWAYS Use SciRS2-Core Abstractions**: All external functionality through `scirs2_core::*`
4. **Version**: Currently using SciRS2 0.1.0-rc.4

### Required Import Patterns

```rust
// ✅ REQUIRED - All NumRS2 code, tests, examples
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
use scirs2_core::ndarray::*;          // Complete ndarray functionality + macros (array!, s!)
use scirs2_core::simd_ops::*;         // SIMD operations
use scirs2_core::parallel_ops::*;     // Parallel operations
use scirs2_stats::*;                  // Statistical functions
use scirs2_linalg::*;                 // Linear algebra

// ❌ FORBIDDEN - Direct external dependencies
// use rand::*;                       // FORBIDDEN
// use ndarray::*;                    // FORBIDDEN
// use rayon::prelude::*;             // FORBIDDEN
```

### Policy Documents

- **NumRS2 Specific**: See `SCIRS2_INTEGRATION_POLICY.md` for NumRS2-specific integration patterns
- **Main Ecosystem Policy**: See `~/work/scirs/SCIRS2_POLICY.md` for complete technical policies
- **Mandatory Compliance**: All code must comply with SciRS2 ecosystem policies

### Key Policies

1. **SIMD Operations**: ALWAYS use `scirs2_core::simd_ops::SimdUnifiedOps` trait
2. **Parallel Processing**: ALWAYS use `scirs2_core::parallel_ops` (NEVER direct rayon)
3. **Random Numbers**: ALWAYS use `scirs2_core::random` (NEVER direct rand/rand_distr)
4. **Array Operations**: ALWAYS use `scirs2_core::ndarray` (NEVER direct ndarray)
5. **BLAS Operations**: ALL through scirs2-core (NEVER direct BLAS dependencies)
6. **Error Handling**: Base on `scirs2_core::error::CoreError`

When writing NumRS2 code, ALWAYS use SciRS2 abstractions - no exceptions for tests, examples, or benchmarks.
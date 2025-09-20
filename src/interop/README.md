# NumRS2 Interoperability Module

This directory contains modules for integrating NumRS2 with other Rust numerical computing libraries.

## Current Integrations

- `ndarray_compat.rs`: Integration with ndarray
- `nalgebra_compat.rs`: Integration with nalgebra
- `scirs_compat.rs`: Integration with SciRS2 (in progress)

## SciRS2 Integration Status

The SciRS2 integration has been updated to use the correct module paths with SciRS2 beta.2. The implementation provides adapter functions for advanced statistical distributions from SciRS2.

### Current Features

1. **Module Paths**: The imports in `scirs_compat.rs` have been updated to match the actual module structure of SciRS2 beta.2:
   - `scirs2_stats::distributions::continuous` for most continuous distributions
   - `scirs2_core::array::Array` for array types
   - `scirs2_core::random::Generator` for random generators

2. **Distribution Implementations**: The following distributions have been implemented:
   - Noncentral Chi-Square (`NoncentralChiSquared`)
   - Noncentral F (`FNoncentral`)
   - Von Mises (`VonMises`)
   - Maxwell-Boltzmann (`MaxwellBoltzmann`)
   - Truncated Normal (`TruncatedNormal`)
   - Multivariate Normal with Rotation (`MultivariateNormal`)

3. **Comprehensive Tests**: Each distribution has been tested to ensure it produces statistically valid outputs.

### Remaining Tasks

1. **Type Conversions**: The conversion functions between NumRS2 and SciRS2 arrays have been implemented but need more extensive testing.

2. **Additional Distributions**: More distributions from SciRS2 could be added in the future.

3. **Performance Optimization**: Further optimization of the conversion between NumRS2 and SciRS2 arrays could improve performance.

4. **Run Integration Tests**: After the implementation is complete, run the full integration tests in `tests/test_scirs_integration.rs`.

### Resources

- SciRS2 Core: `~/.cargo/registry/src/index.crates.io-*/scirs2-core-0.1.0-beta.2/`
- SciRS2 Stats: `~/.cargo/registry/src/index.crates.io-*/scirs2-stats-0.1.0-beta.2/`

## Usage

To use the SciRS2 integration, add the following to your `Cargo.toml`:

```toml
[dependencies]
numrs2 = { version = "0.1.0-beta.2", features = ["scirs"] }
```

Then import the SciRS2 compatibility module:

```rust
use numrs2::interop::scirs_compat::*;
```

See the examples in `/examples/random_distributions_example.rs` for usage examples.
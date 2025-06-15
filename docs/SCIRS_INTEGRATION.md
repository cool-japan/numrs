# SciRS2 Integration in NumRS2

This document describes the integration between NumRS2 and SciRS2, focusing on advanced statistical distributions and how to use them in your projects.

## Overview

NumRS2 integrates with SciRS2 to provide advanced statistical distributions that are available in SciPy but not directly implemented in NumRS2. This integration is optional and enabled through a Cargo feature flag.

## Available Distributions

When the `scirs` feature is enabled, the following advanced distributions become available:

1. **Noncentral Chi-square**: `noncentral_chisquare(df, nonc, shape)`
2. **Noncentral F**: `noncentral_f(dfnum, dfden, nonc, shape)`
3. **Von Mises**: `vonmises(mu, kappa, shape)`
4. **Maxwell**: `maxwell(scale, shape)`
5. **Truncated Normal**: `truncated_normal(mean, std, low, high, shape)`
6. **Multivariate Normal with Rotation**: `multivariate_normal_with_rotation(means, cov, size, rotation)`

## How to Enable SciRS2 Integration

To enable SciRS2 integration:

1. Add SciRS2 dependencies to your `Cargo.toml`:
   ```toml
   [dependencies]
   numrs2 = { version = "0.1.0-alpha.4", features = ["scirs"] }
   ```

2. Or, when building NumRS2 directly:
   ```bash
   cargo build --features scirs
   ```

## Usage Examples

Here's how to use the SciRS2 integrated distributions:

```rust
use numrs2::array::Array;
use numrs2::random::distributions::set_seed;

// Import the SciRS2 integration module
#[cfg(feature = "scirs")]
use numrs2::interop::scirs_compat::*;

// Set a seed for reproducibility
set_seed(12345);

// Generate samples from a noncentral chi-square distribution
#[cfg(feature = "scirs")]
let samples = noncentral_chisquare(2.0, 1.0, &[10]).unwrap();

// Generate samples from a von Mises distribution
#[cfg(feature = "scirs")]
let samples = vonmises(0.0, 1.0, &[10]).unwrap();

// Generate samples from a truncated normal distribution
#[cfg(feature = "scirs")]
let samples = truncated_normal(0.0, 1.0, -2.0, 2.0, &[10]).unwrap();

// Generate samples from a multivariate normal with rotation
#[cfg(feature = "scirs")]
let mean = vec![0.0, 0.0];
#[cfg(feature = "scirs")]
let cov_data = vec![1.0, 0.5, 0.5, 1.0];
#[cfg(feature = "scirs")]
let cov = Array::from_vec(cov_data).reshape(&[2, 2]);
#[cfg(feature = "scirs")]
let samples = multivariate_normal_with_rotation(&mean, &cov, Some(&[5]), None).unwrap();
```

## Conditional Compilation

The integration code uses conditional compilation to ensure that there are no runtime errors when the `scirs` feature is disabled. When SciRS2 is not available, placeholder functions return a clear error message indicating that the feature needs to be enabled.

```rust
// When SciRS2 is not available
if let Err(e) = noncentral_chisquare(2.0, 1.0, &[10]) {
    println!("Error: {}", e); 
    // Outputs: "Error: Not implemented: Noncentral chi-square distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml."
}
```

## Implementation Details

The integration is implemented in the `src/interop/scirs_compat.rs` module, which provides:

1. **Type conversions** between NumRS2 and SciRS2 arrays
2. **Parameter validation** to ensure inputs meet distribution requirements  
3. **Error handling** with detailed messages when operations fail
4. **Conditional compilation** to provide graceful degradation when SciRS2 is unavailable

## Testing

The SciRS2 integration includes:

1. **Unit tests**: In `tests/test_scirs_integration.rs` that verify the distributions produce valid outputs
2. **Reference tests**: In `tests/test_distribution_reference.rs` that validate outputs against NumPy/SciPy reference data

## Future Development

## Current Status

The SciRS2 integration has been updated to work with version 0.1.0-alpha.4:

1. **Module imports fixed**: The imports in `src/interop/scirs_compat.rs` have been updated to match the actual SciRS2 module structure in version 0.1.0-alpha.4:
   - `scirs2_stats::distributions::continuous` for most continuous distributions
   - `scirs2_core::array::Array` for array types
   - `scirs2_core::random::Generator` for random generators

2. **Distribution implementations completed**: The implementations of the SciRS2 adapter functions have been connected to the appropriate SciRS2 distributions:
   - `NoncentralChiSquared` (replaces `NoncentralChiSquare`)
   - `FNoncentral` (replaces `NoncentralF`)
   - `VonMises` (unchanged)
   - `MaxwellBoltzmann` (replaces `Maxwell`)
   - `TruncatedNormal` (unchanged)
   - `MultivariateNormal` (replaces `MultivariateDist::multivariate_normal`)

3. **Array conversions implemented**: The conversion between NumRS2 and SciRS2 array types has been implemented and tested.

See the `src/interop/README.md` file for more details on the current status and next steps.

## Long-term Enhancements

Future enhancements to the SciRS2 integration may include:

1. **Additional distributions** from SciRS2 as they become available
2. **Performance optimizations** for the conversion between NumRS2 and SciRS2 arrays
3. **More integrated functions** beyond just statistical distributions

## Troubleshooting

Common issues and solutions:

1. **Missing SciRS2 error**:
   ```
   Error: Not implemented: Noncentral chi-square distribution requires SciRS2 integration. Enable the 'scirs' feature in Cargo.toml.
   ```
   **Solution**: Add the `scirs` feature to your dependencies or build command.

2. **Version compatibility issues**:
   If you encounter errors related to SciRS2 versions, ensure you're using compatible versions.
   NumRS2 is tested with SciRS2 v0.1.0-alpha.4 or newer.

## Contributing

Contributions to enhance the SciRS2 integration are welcome! Areas that would benefit from contribution include:

1. Additional distribution implementations
2. Enhanced type conversion utilities
3. Performance improvements
4. Additional testing and validation
# Installing NumRS2

This guide provides detailed instructions for installing NumRS2 in various environments.

## Prerequisites

Before installing NumRS2, ensure you have:

1. **Rust Toolchain**: NumRS2 requires Rust stable 1.65 or later
   - Install via [rustup](https://rustup.rs/) (recommended)
   - Verify with: `rustc --version`

2. **C/C++ Compiler Toolchain**:
   - Linux: GCC or Clang
   - macOS: Xcode Command Line Tools
   - Windows: Microsoft Visual C++ Build Tools or MinGW-w64

3. **Linear Algebra Libraries** (for BLAS/LAPACK features):
   - Linux: OpenBLAS, ATLAS, or Intel MKL
   - macOS: Accelerate Framework (built-in) or OpenBLAS
   - Windows: OpenBLAS or Intel MKL

## Installation Methods

### Method 1: From crates.io (Recommended)

The simplest way to install NumRS2 is directly from [crates.io](https://crates.io/):

1. Add NumRS2 to your `Cargo.toml`:
   ```toml
   [dependencies]
   numrs2 = "0.1.0-alpha.2"
   ```

2. Or use Cargo to add it directly:
   ```bash
   cargo add numrs2
   ```

### Method 2: From GitHub

To install the latest development version:

```bash
cargo add numrs2 --git https://github.com/cool-japan/numrs.git
```

### Method 3: Building from Source

For maximum control or to contribute to development:

1. Clone the repository:
   ```bash
   git clone https://github.com/cool-japan/numrs.git
   cd numrs
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run tests to verify the installation:
   ```bash
   cargo test
   ```

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential libopenblas-dev liblapack-dev
   ```

2. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. Add NumRS2 to your project's `Cargo.toml`

#### Fedora/RHEL/CentOS

1. Install system dependencies:
   ```bash
   sudo dnf install gcc-c++ make openblas-devel lapack-devel
   ```

2. Follow the same Rust installation and project setup as above

### macOS

1. Install developer tools:
   ```bash
   xcode-select --install
   ```

2. Install additional dependencies (optional, for better performance):
   ```bash
   brew install openblas lapack
   ```

3. Install Rust and add NumRS2 to your project

### Windows

1. Install Rust with rustup from https://rustup.rs/
   - Choose the MSVC toolchain (default)

2. For BLAS/LAPACK support:
   - Install OpenBLAS: See https://github.com/OpenBLAS/OpenBLAS/wiki/Installation-Guide
   - Set the environment variable for OpenBLAS:
     ```
     set OPENBLAS_DIR=C:\path\to\openblas
     ```

3. Add NumRS2 to your project's `Cargo.toml`

## Verifying Installation

To verify your installation:

1. Create a new test project:
   ```bash
   cargo new numrs_test
   cd numrs_test
   ```

2. Add NumRS2 to dependencies in `Cargo.toml`:
   ```toml
   [dependencies]
   numrs2 = "0.1.0-alpha.2"
   ```

3. Replace `src/main.rs` content with:
   ```rust
   use numrs2::prelude::*;
   
   fn main() -> Result<()> {
       // Create a simple array
       let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
       println!("Array:\n{}", arr);
       
       // Perform a simple operation
       let result = arr.mean()?;
       println!("Mean value: {}", result);
       
       Ok(())
   }
   ```

4. Run the test program:
   ```bash
   cargo run
   ```

If no errors occur and you see the array and its mean value printed, the installation was successful.

## Configuration Options

NumRS2 can be configured with various feature flags in your `Cargo.toml`:

```toml
[dependencies]
numrs2 = { version = "0.1.0-alpha.2", features = ["blas", "lapack", "serde"] }
```

Available features:
- `blas`: Enable BLAS integration for linear algebra (on by default)
- `lapack`: Enable LAPACK for advanced linear algebra operations (on by default)
- `serde`: Add serialization/deserialization support with serde
- `simd`: Enable SIMD optimizations (on by default)
- `rayon`: Enable parallel computing capabilities (on by default)
- `npy`: Support for NumPy's .npy/.npz file formats

## Troubleshooting

### Common Issues

#### Missing BLAS/LAPACK Libraries

Error: `cannot find -lopenblas` or similar

Solution:
- Linux: Install the development packages mentioned above
- macOS: Install via Homebrew or use Apple's Accelerate Framework
- Windows: Check environment variables and library paths

#### Compilation Errors

Error: `error: failed to run custom build command for numrs2`

Solution:
- Ensure you have the latest stable Rust toolchain: `rustup update stable`
- Check for missing system dependencies
- Try with minimal features: `cargo build --no-default-features`

#### Runtime Panics

If you experience runtime panics, check:
- Array dimensions match requirements for operations
- No division by zero or other numerical issues
- Input data is valid

### Getting Help

If you encounter issues not covered here:
- Check the [GitHub issues](https://github.com/cool-japan/numrs/issues) for similar problems
- Open a new issue with detailed information about your environment and the error
- Join the community discussion on [GitHub Discussions](https://github.com/cool-japan/numrs/discussions)

## Uninstallation

To remove NumRS2 from your project, simply remove it from your `Cargo.toml` dependencies.

If you installed development tools specifically for NumRS2 and no longer need them:
- Uninstall Rust (if not needed): `rustup self uninstall`
- System dependencies can be removed using your package manager if not needed for other purposes
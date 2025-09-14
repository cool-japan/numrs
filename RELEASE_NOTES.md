# NumRS2 v0.1.0-beta.1 Release Notes

**Release Date:** September 15, 2025
**First Beta Release** - Dependency Modernization & Stability

## 🎯 **Release Highlights**

This is the **first beta release** of NumRS2, focusing on dependency modernization and stability improvements while maintaining full backward compatibility. This release prepares NumRS2 for production use with updated dependencies and enhanced integration with the SciRS2 ecosystem.

### 📦 **Dependency Updates**

- **SciRS2 Integration**: Updated all scirs2-* dependencies from 0.1.0-alpha.5 to 0.1.0-beta.1
  - Enhanced scientific computing capabilities
  - Improved SIMD acceleration through SciRS2-Core
  - Better statistical distribution support
- **Core Dependencies**:
  - `rand`: 0.9.0 → 0.9.2 (per CLAUDE.md requirements)
  - `rand_distr`: 0.5.0 → 0.5.1 (improved distributions)
  - `nalgebra`: 0.32.3 → 0.34.0 (major version upgrade)
  - `criterion`: 0.5.1 → 0.7.0 (enhanced benchmarking)
  - `csv`: 1.3.0 → 1.3.1 (improved CSV processing)
  - `zip`: 0.6.6 → 5.1.1 (enhanced NPZ file support)
- **Security & Performance**: Updated 100+ transitive dependencies for improved security and performance

### 🛠️ **Technical Fixes**

- **Compatibility**: Resolved bincode 2.0 API breaking changes by maintaining compatibility with bincode 1.3.3
- **Type Safety**: Fixed zip 5.1 FileOptions type annotations for proper NPZ file handling
- **Build**: Fixed SIMD verification test type annotation error for improved compilation

### ✅ **Quality Assurance**

- **100% Test Pass Rate**: All 586 tests pass successfully (0 failed, 1 ignored)
- **Zero Regressions**: No functionality regressions detected from dependency updates
- **API Stability**: Maintained full API compatibility and feature set
- **Integration Verified**: Confirmed scirs2 integration compatibility with beta.1 versions

## 📊 **Production Readiness**

### ✅ **Ready for Production Use**
- **Core Operations**: N-dimensional arrays, broadcasting, mathematical functions
- **Linear Algebra**: Matrix operations, decompositions, eigenvalues, SVD, QR, LU, Cholesky
- **Statistical Analysis**: Comprehensive statistics, random distributions, hypothesis testing
- **Performance**: SIMD acceleration, parallel processing, GPU support (optional)
- **I/O Operations**: NPY/NPZ, CSV, JSON, binary formats
- **Integration**: SciRS2, nalgebra, ndarray ecosystem compatibility

### 🔬 **Scientific Computing Features**
- **Advanced Distributions**: Noncentral chi-square, noncentral F, von Mises, Maxwell-Boltzmann
- **Signal Processing**: FFT, windowing functions, convolution, correlation
- **Polynomial Operations**: Interpolation, fitting, root finding, evaluation
- **Financial Mathematics**: Bond pricing, options pricing, risk calculations
- **Sparse Arrays**: Memory-efficient sparse matrix operations

### 🚀 **Performance Optimizations**
- **SIMD Acceleration**: Automatic CPU feature detection (AVX2, AVX512, NEON)
- **Parallel Processing**: Adaptive work distribution with Rayon
- **Memory Optimization**: Cache-friendly layouts, custom allocators
- **GPU Acceleration**: Optional WGPU-based GPU computing

## 🔧 **Breaking Changes**

**None** - This release maintains full backward compatibility with alpha.5.

## 🎯 **Migration from Alpha.5**

Update your `Cargo.toml`:

```toml
[dependencies]
numrs2 = "0.1.0-beta.1"
```

No code changes required - all APIs remain stable.

## 🚀 **What's Next**

The **beta phase** will focus on:

1. **API Stabilization**: Finalizing public APIs for 1.0 release
2. **Performance Optimization**: Advanced SIMD and GPU acceleration
3. **Documentation**: Comprehensive guides and tutorials
4. **Community Feedback**: Incorporating user feedback and requests
5. **Production Hardening**: Additional edge case testing and optimization

## 📚 **Resources**

- **Documentation**: [docs.rs/numrs2](https://docs.rs/numrs2)
- **Repository**: [github.com/cool-japan/numrs](https://github.com/cool-japan/numrs)
- **Examples**: See `examples/` directory for comprehensive usage examples
- **Migration Guide**: See `NUMPY_MIGRATION.md` for NumPy users

## 🙏 **Acknowledgments**

Special thanks to the Rust scientific computing community and all contributors who helped make this release possible. The dependency updates in this release build upon the excellent work of the nalgebra, rand, and broader Rust ecosystem maintainers.

---

**Download**: Available on [crates.io](https://crates.io/crates/numrs2)
**License**: Apache-2.0 OR MIT
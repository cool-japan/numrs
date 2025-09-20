# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Released]

## [0.1.0-beta.2] - 2025-09-20

### Updated
- **Dependencies**: Updated all scirs2-* dependencies from 0.1.0-beta.1 to 0.1.0-beta.2 for better integration
- **Version**: Bumped version to 0.1.0-beta.2 for second beta release

### Added
- **Mathematical Functions**: Extended mathematical operations with over 100 new functions
- **Error Handling**: Comprehensive error recovery system with context-aware messages
- **Advanced Indexing**: Boolean indexing, fancy indexing, and multi-dimensional slicing capabilities
- **Financial Computing**: Options pricing models, bond valuation, and financial metrics
- **Signal Processing**: Enhanced FFT, convolution, correlation, and filtering operations
- **Testing Framework**: Comprehensive testing utilities for numerical validation

### Improved
- **Performance**: Optimized SIMD operations for AVX512 and ARM NEON architectures
- **Memory Management**: Enhanced memory alignment and cache utilization
- **Documentation**: Expanded examples and migration guides
- **NumPy Compatibility**: Improved compatibility layer for easier migration

## [0.1.0-beta.1] - 2025-09-15

### Updated
- **Dependencies**: Updated scirs2-* dependencies from 0.1.0-alpha.5 to 0.1.0-beta.1 for enhanced SciRS2 integration
- **Dependencies**: Updated rand from 0.9.0 to 0.9.2 (per CLAUDE.md requirements)
- **Dependencies**: Updated rand_distr from 0.5.0 to 0.5.1 for improved random number generation
- **Dependencies**: Updated nalgebra from 0.32.3 to 0.34.0 (major version upgrade)
- **Dependencies**: Updated criterion from 0.5.1 to 0.7.0 (major version upgrade for better benchmarking)
- **Dependencies**: Updated csv from 1.3.0 to 1.3.1 for improved CSV processing
- **Dependencies**: Updated zip from 0.6.6 to 5.1.1 (major version upgrade for NPZ file support)
- **Dependencies**: Updated 100+ transitive dependencies via cargo update for improved security and performance

### Fixed
- **Build**: Fixed SIMD verification test type annotation error for improved compilation
- **Compatibility**: Resolved bincode 2.0 API breaking changes by maintaining compatibility with bincode 1.3.3
- **Types**: Fixed zip 5.1 FileOptions type annotations for proper NPZ file handling

### Verified
- **Testing**: All 586 tests pass successfully (0 failed, 1 ignored)
- **Stability**: No regressions detected from dependency updates
- **Integration**: Verified scirs2 integration compatibility with beta.1 versions
- **API**: Maintained full API compatibility and feature set

This beta release focuses on dependency modernization and stability improvements while maintaining full backward compatibility.


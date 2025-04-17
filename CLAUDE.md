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
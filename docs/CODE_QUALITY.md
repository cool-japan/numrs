# Code Quality Guidelines for NumRS2

This document outlines the code quality standards for the NumRS2 project, with a focus on using Clippy for code quality checks.

## Overview

NumRS2 aims to maintain high code quality standards to ensure that the library is reliable, performant, and maintainable. We use Rust's Clippy linter to enforce these standards and catch common programming errors.

## Using Clippy

Clippy is a collection of lints to catch common mistakes and improve your Rust code. It's an essential tool for ensuring code quality in NumRS2.

### Running Clippy Locally

To run Clippy on the NumRS2 codebase, use the provided script:

```bash
./scripts/clippy_check.sh
```

To automatically fix eligible issues:

```bash
./scripts/clippy_check.sh --fix
```

### Clippy in CI

Clippy checks are automatically run in CI for all pull requests. Pull requests that introduce new Clippy warnings will be flagged, and maintainers will ask you to address these issues before merging.

## Clippy Configuration

NumRS2 uses a customized Clippy configuration that balances strict code quality with the practical needs of a numerical computing library:

### Enforced Lints

- `clippy::all`: All default clippy lints
- `clippy::pedantic`: Stricter, more pedantic lints
- `clippy::nursery`: Experimental lints for early adopters
- `clippy::unwrap_used`: Avoid unwrap() which can panic
- `clippy::expect_used`: Avoid expect() which can panic

### Allowed Exceptions

Numerical computing code sometimes has legitimate reasons to use patterns that Clippy warns about:

- `clippy::cast_possible_truncation`: Allowed due to common numerical type conversions
- `clippy::cast_precision_loss`: Allowed due to numerical algorithm requirements
- `clippy::similar_names`: Allowed as math variables often use similar names
- `clippy::too_many_arguments`: Allowed for complex numerical functions with many parameters

## Code Quality Best Practices

Beyond Clippy, NumRS2 follows these general code quality practices:

### Error Handling

- Use `Result` for operations that can fail
- Avoid `unwrap()` and `expect()` in production code
- Provide detailed error messages that explain what went wrong
- Consider error contexts (what happened and why it failed)

### Documentation

- Document all public API items with clear explanations
- Include examples in documentation comments
- Note any performance implications or algorithmic complexity
- Explain mathematical concepts where relevant

### Testing

- Write unit tests for all functionality
- Include edge cases in tests
- Use property-based testing for mathematical properties
- Test numerical stability with extreme values

### Performance

- Be mindful of allocations, especially in hot code paths
- Use the most efficient algorithm for the task
- Consider SIMD optimizations for numerical code
- Profile before optimizing

## Special Considerations for Numerical Code

Numerical computing has some special considerations for code quality:

### Numerical Stability

- Be aware of floating-point precision limitations
- Use stable algorithms for numerical operations
- Test with extreme values to ensure stability
- Document known limitations or edge cases

### Type Conversions

- Be explicit about type conversions
- Document any potential loss of precision
- Consider using generic programming with trait bounds

### Performance Optimization

- Balance readability with performance
- Document performance characteristics
- Use benchmarks to validate optimizations

## Reviewing Code for Quality

When reviewing NumRS2 code (your own or others'), ask these questions:

1. Does the code follow Rust's idioms and best practices?
2. Is the code well-documented, including mathematical concepts?
3. Are there appropriate tests, including edge cases?
4. Is the error handling comprehensive and user-friendly?
5. Is the code numerically stable across different inputs?
6. Does the code balance performance with readability?
7. Are there any Clippy warnings that should be addressed?

## Conclusion

Maintaining high code quality is essential for NumRS2 to be a reliable, performant numerical computing library. By following these guidelines and using Clippy, we can ensure that NumRS2 meets the needs of its users while being maintainable and extensible.
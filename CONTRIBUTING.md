# Contributing to NumRS2

Thank you for your interest in contributing to NumRS2! This document provides guidelines and workflows for contributing to the project.

NumRS2 is a comprehensive numerical computing library for Rust, aiming to provide NumPy-like functionality with Rust's performance and safety benefits. We welcome contributions of all kinds, from bug fixes to documentation improvements to feature implementations.

## Ways to Contribute

You can contribute to NumRS2 in many ways:

- **Code maintenance and development**: Implement new features or fix bugs
- **Documentation**: Improve existing docs or write new guides and examples
- **Testing**: Add tests for existing functionality or new features
- **Benchmarking**: Help measure and improve performance
- **Examples**: Create examples demonstrating library usage
- **Reviews**: Review pull requests from other contributors

## Development Process

### Setting Up Your Development Environment

1. **Fork the repository**:
   Go to https://github.com/cool-japan/numrs and click the "fork" button to create your own copy of the project.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/numrs.git
   cd numrs
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/cool-japan/numrs.git
   ```

4. **Pull the latest changes**:
   ```bash
   git checkout main
   git pull upstream main
   ```

### Development Workflow

1. **Create a branch for your feature or bugfix**:
   ```bash
   git checkout -b my-feature-branch
   ```

2. **Make your changes**:
   - Write code that follows Rust style guidelines
   - Add appropriate documentation and tests
   - Run tests to ensure your changes don't break existing functionality

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

4. **Run tests locally**:
   ```bash
   cargo test
   ```

5. **Push your changes**:
   ```bash
   git push origin my-feature-branch
   ```

6. **Create a Pull Request**:
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your branch and submit the pull request with a clear description

### Pull Request Guidelines

When submitting a pull request:

1. **Describe your changes**: Provide a clear description of what your pull request does, and why it's valuable.
2. **Reference issues**: If your PR is related to an issue, reference it with "Fixes #123" or "Related to #123".
3. **Keep it focused**: Each PR should focus on a single feature or bugfix.
4. **Add tests**: Ensure your code is tested. If you're fixing a bug, add a test that would have caught the bug.
5. **Update documentation**: If your changes require it, update relevant documentation.

## Coding Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` to format your code
- Run `clippy` to check for common mistakes and non-idiomatic code
- Aim for zero warnings

### Documentation

- Document all public API items with rustdoc comments
- For mathematical functions, include the mathematical formula and references
- Provide examples for non-trivial functionality
- Keep documentation up-to-date with code changes

### Testing

- Write unit tests for all new functionality
- For mathematical operations, include property-based tests where applicable
- Test edge cases appropriately
- Aim for high test coverage (ideally 100% for new code)

## Review Process

Once you submit a pull request:

1. Maintainers and contributors will review your code
2. Automated tests will run to verify your changes
3. Feedback may be provided, requiring changes to your PR
4. Once approved, your PR will be merged into the main codebase

Don't be discouraged if your PR isn't accepted immediately! The review process helps ensure high quality code and is an opportunity for learning.

## Getting Help

If you need help with contributing:

- Open an issue on GitHub with your question
- Reach out through the project's communication channels
- Look for "good first issue" labels for places to start

Thank you for contributing to NumRS2!
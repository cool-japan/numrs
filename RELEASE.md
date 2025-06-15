# Release Process for NumRS2

This document outlines the process for releasing new versions of NumRS2.

## Version Numbering

NumRS2 follows [Semantic Versioning](https://semver.org/):

- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (X.Y.0)**: New functionality in a backward-compatible manner
- **Patch version (X.Y.Z)**: Backward-compatible bug fixes

## Release Prerequisites

Before beginning a release, ensure you have:

1. Proper Rust toolchain installed (stable version)
2. Cargo and related tools (`cargo-release` is recommended)
3. GitHub CLI (for creating release tags)
4. Write access to the NumRS2 repository and crates.io
5. All tests pass on the main branch
6. Documentation is up-to-date

## Release Checklist

### 1. Pre-release Verification

- [ ] Ensure all tests pass: `cargo test`
- [ ] Check the code with clippy: `cargo clippy`
- [ ] Verify documentation builds correctly
- [ ] Create a release branch: `git checkout -b release-vX.Y.Z`
- [ ] Update version numbers in:
  - [ ] `Cargo.toml`
  - [ ] `README.md`
  - [ ] Any other version references in documentation

### 2. Update Changelog

- [ ] Update `CHANGELOG.md` with all notable changes since the last release
- [ ] Organize changes under categories:
  - **Added**: New features
  - **Changed**: Changes in existing functionality
  - **Deprecated**: Soon-to-be removed features
  - **Removed**: Removed features
  - **Fixed**: Bug fixes
  - **Security**: Vulnerability fixes
- [ ] Include PR and issue numbers where appropriate

### 3. Final Review

- [ ] Run the full test suite one final time
- [ ] Build the documentation: `cargo doc --no-deps --open`
- [ ] Check all examples: `cargo run --example basic_usage`
- [ ] Verify any platform-specific functionality if applicable

### 4. Create the Release

- [ ] Commit all changes: `git commit -am "Prepare release vX.Y.Z"`
- [ ] Push the release branch: `git push origin release-vX.Y.Z`
- [ ] Create a pull request for the release
- [ ] After approval and merge, tag the release:
  ```
  git checkout main
  git pull
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  git push origin vX.Y.Z
  ```

### 5. Publish to crates.io

- [ ] Publish the crate:
  ```
  cargo login
  cargo publish
  ```
- [ ] Verify the release is visible on [crates.io](https://crates.io/crates/numrs2)

### 6. Create GitHub Release

- [ ] Create a new GitHub release from the tag
- [ ] Copy the relevant CHANGELOG entries to the release description
- [ ] Attach any binary artifacts if applicable

### 7. Update Documentation

- [ ] If applicable, update documentation on the official website
- [ ] Ensure docs.rs has the latest documentation

### 8. Post-release Tasks

- [ ] Announce the release on relevant channels (if applicable)
- [ ] Create an issue to track any follow-up tasks from the release
- [ ] Bump version numbers to next development version (X.Y.Z+1-dev) in the main branch

## Current Release Status

### Version 0.1.0-alpha.4 (Released: June 15, 2025)

This release focuses on comprehensive NumPy parity roadmap and enhanced documentation:

**Key Features:**
- Comprehensive NumPy parity roadmap in TODO.md
- Detailed implementation plan for complete NumPy compatibility
- Phase-based development strategy for systematic feature implementation
- Enhanced documentation of current implementation status (80-85% NumPy parity)
- Success metrics and quality standards for NumPy parity achievement

**Architecture Improvements:**
- Complete restructuring of TODO.md into comprehensive NumPy parity initiative
- Improved development priorities with focus on missing 15-20% of NumPy functionality
- Enhanced contribution guidelines for NumPy compatibility
- Improved technical implementation documentation

## Release Cadence

NumRS2 aims for a regular release schedule:

- **Minor releases**: Every 2-3 months with new features
- **Patch releases**: As needed for bug fixes
- **Major releases**: When significant API changes are required, with ample advance notice

## Long-term Support

- LTS releases will be designated as needed
- Security updates and critical bug fixes will be backported to LTS releases

## Handling Special Releases

### Pre-releases

For pre-releases (alpha/beta/rc):

1. Follow the same process but use appropriate suffixes (e.g., `1.0.0-alpha.1`)
2. Clearly mark these as non-production ready in release notes

### Emergency Fixes

For emergency/security fixes:

1. Create a branch from the relevant release tag
2. Apply the minimal fix required
3. Follow the normal release process but increment only the patch version
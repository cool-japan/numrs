//! Build script for numrs2.
//!
//! This exists for exactly one reason: when the `python` feature is enabled,
//! pyo3 is built with its "extension-module" feature, which deliberately
//! suppresses linking against libpython -- the `_Py*` symbols are meant to be
//! resolved by the CPython interpreter that dlopen()s the module, not bound at
//! build time. macOS's Mach-O linker rejects the resulting undefined symbols
//! ("Undefined symbols for architecture arm64: _PyBaseObject_Type, ...") unless
//! it is given `-undefined dynamic_lookup`, so we ask pyo3-build-config to emit
//! that link argument. maturin passes the same flag when it builds the wheel,
//! which is why `maturin build` always worked on macOS while a plain
//! `cargo build --features python` did not.
//!
//! See .cargo/config.toml for the companion rustflags: this call emits
//! `cargo:rustc-cdylib-link-arg=...`, which reaches the **cdylib only**, while
//! `cargo test` / `cargo nextest run --all-features` also links a unit-test
//! binary that compiles src/python/mod.rs's #[pymodule] into its own objects
//! and needs the same flag. Neither part covers the other's link products.
//!
//! Non-macOS targets are unaffected: add_extension_module_link_args() is a
//! no-op everywhere except macOS and wasm32-unknown-emscripten.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "python")]
    pyo3_build_config::add_extension_module_link_args();
}

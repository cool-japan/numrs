//! WASM integration test harness.
//!
//! Wires the previously-orphaned `tests/wasm/{test_wasm_array,test_wasm_linalg,
//! test_wasm_stats}.rs` into an actual cargo test target. Those three files
//! were already written against `wasm-bindgen-test` (the right framework),
//! but `tests/wasm/mod.rs` was never `mod`-included by any top-level harness
//! or `[[test]]` entry, so none of them were ever compiled.
//!
//! All three modules (and this file) are `#![cfg(target_arch = "wasm32")]`,
//! so on a normal host build this whole target compiles to an empty shell --
//! no special `required-features` gating is needed for that, unlike the GPU
//! harness. Run for real with (see `scripts/build-wasm.sh`):
//!   `wasm-pack test --headless --firefox --features wasm`
//!
//! `wasm_bindgen_test_configure!(run_in_browser)` must appear exactly once
//! per test binary, so it lives here rather than in each module file (it
//! used to be duplicated three times across the never-compiled files, which
//! would have been three expansions in one crate the moment they were wired
//! into a single binary like this one).
#![cfg(target_arch = "wasm32")]

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "wasm/test_wasm_array.rs"]
mod test_wasm_array;

#[path = "wasm/test_wasm_linalg.rs"]
mod test_wasm_linalg;

#[path = "wasm/test_wasm_stats.rs"]
mod test_wasm_stats;

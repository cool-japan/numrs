//! GPU integration test harness.
//!
//! Wires every file under `tests/gpu/` into an actual `cargo test`/
//! `cargo nextest` target. Each of those files existed on disk as ordinary
//! Rust source but was never `mod`-included by any top-level `tests/*.rs`
//! harness or `[[test]]` entry, so none of them were ever compiled --
//! silently dead test code. `tests/gpu/mod.rs`, which used to declare all
//! five as a `#[cfg(test)]` module tree that nothing included either, has
//! been deleted now that this file supersedes it.
//!
//! Gated by `required-features = ["gpu"]` in `Cargo.toml` (see the
//! `[[test]]` entry there), so this target is skipped entirely -- not
//! compiled at all, not just a no-op -- under default features. Run with:
//!   `cargo nextest run --features gpu --test gpu_integration`

#[path = "gpu/test_gpu_ops.rs"]
mod test_gpu_ops;

#[path = "gpu/test_gpu_linalg.rs"]
mod test_gpu_linalg;

#[path = "gpu/test_gpu_memory.rs"]
mod test_gpu_memory;

#[path = "gpu/test_compute.rs"]
mod test_compute;

#[path = "gpu/test_batching.rs"]
mod test_batching;

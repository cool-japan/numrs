//! GPU integration test harness.
//!
//! Wires the previously-orphaned `tests/gpu/{test_gpu_ops,test_gpu_linalg,
//! test_gpu_memory}.rs` into an actual `cargo test`/`cargo nextest` target.
//! Those three files existed on disk but were never `mod`-included by any
//! top-level `tests/*.rs` harness or `[[test]]` entry, so they were never
//! compiled -- silently dead test code. `tests/gpu/mod.rs` also declares
//! `test_compute` and `test_batching`, which are intentionally NOT wired
//! here (out of scope for this pass; see the W6-TESTS report).
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

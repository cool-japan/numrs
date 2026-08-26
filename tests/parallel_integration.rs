//! Parallel-computing integration test harness.
//!
//! Wires the previously-orphaned `tests/parallel/` tree (`mod.rs` plus 12
//! test files covering adaptive scheduling, load balancing, NUMA awareness,
//! work stealing, thread affinity, metrics/monitoring, scalability and
//! stress) into an actual cargo test target. `tests/parallel/mod.rs` was
//! never `mod`-included by any top-level harness or `[[test]]` entry, so
//! none of these files were ever compiled (never since 0.2.0).
//!
//! `numrs2::parallel` is always compiled (not feature-gated), so unlike the
//! GPU/I-O harnesses this one needs no `required-features` -- it runs under
//! default features. Loading `tests/parallel/mod.rs` by its on-disk path
//! (rather than re-declaring each of the 12 `mod test_*;` lines here) keeps
//! that file meaningful as the single place that lists which test files are
//! wired, and lets those files share the `wait_for_drain` helper defined
//! there via `super::`. Run with:
//!   `cargo nextest run --test parallel_integration`
#[path = "parallel/mod.rs"]
mod parallel;

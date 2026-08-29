//! I/O format integration test harness.
//!
//! Wires the previously-orphaned `tests/io/{test_bson,test_matlab,
//! test_messagepack,test_netcdf}.rs` into an actual cargo test target. Those
//! four files were already written against the real, current
//! `numrs2::io::{bson_format,matlab,messagepack,netcdf}` APIs, but
//! `tests/io/` was never `mod`-included by any top-level harness or
//! `[[test]]` entry, so none of them were ever compiled (never since
//! 0.2.0).
//!
//! Each format lives behind its own optional Cargo feature (see
//! `tests/io/test_*.rs`, which individually `#[cfg(feature = "...")]`-gate
//! their imports and `#[test]` functions), so each submodule here is gated
//! the same way: with none of the four features on, this file compiles to
//! an empty shell, not a stub that hits missing symbols. `required-features`
//! in Cargo.toml's `[[test]]` entry additionally skips building this target
//! at all under default features. Run with:
//!   `cargo nextest run --features io-all --test io_integration`
//! (or `--features bson,matlab,messagepack,netcdf` -- `io-all` is a superset
//! that also turns on `arrow`/`parquet`, unrelated to these four formats).

#[cfg(feature = "bson")]
#[path = "io/test_bson.rs"]
mod test_bson;

#[cfg(feature = "matlab")]
#[path = "io/test_matlab.rs"]
mod test_matlab;

#[cfg(feature = "messagepack")]
#[path = "io/test_messagepack.rs"]
mod test_messagepack;

#[cfg(feature = "netcdf")]
#[path = "io/test_netcdf.rs"]
mod test_netcdf;

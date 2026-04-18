//! Comprehensive IO Benchmarks for NumRS2
//!
//! This benchmark suite tests NPY and NPZ format IO performance including:
//! - NPY serialization (write) for varying array sizes
//! - NPY deserialization (read) for varying array sizes
//! - NPZ multi-array write (uncompressed and compressed)
//! - NPZ multi-array read
//! - Full NPY roundtrip (write + read cycle)
//!
//! All benchmarks follow COOLJAPAN policies: no unwrap(), snake_case, no warnings.
//! In-memory IO is used throughout (Cursor<Vec<u8>>) — no filesystem access.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::io::{
    deserialize_from_file, load_all_npz_arrays, save_npz_arrays, serialize_to_file, SerializeFormat,
};
use numrs2::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::io::Cursor;

/// Benchmark serializing a 1D f64 array to NPY format.
/// Sizes: 1_000, 10_000, 100_000, 1_000_000 elements.
/// Throughput is reported in bytes (f64 = 8 bytes per element).
fn bench_npy_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_write");

    for n in [1_000usize, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes((*n * 8) as u64));
        let array = Array::from_vec(vec![0.0f64; *n]);

        group.bench_with_input(BenchmarkId::new("f64_1d", n), n, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(Vec::new());
                if let Ok(()) = serialize_to_file(&array, &mut cursor, SerializeFormat::Npy) {
                    black_box(cursor.into_inner());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark deserializing a 1D f64 array from NPY format.
/// Pre-serializes to a buffer outside the timing loop; only read is measured.
/// Sizes: 1_000, 10_000, 100_000, 1_000_000 elements.
fn bench_npy_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_read");

    for n in [1_000usize, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes((*n * 8) as u64));

        // Pre-build the buffer outside the benchmark loop.
        let array = Array::from_vec(vec![0.0f64; *n]);
        let mut write_cursor = Cursor::new(Vec::new());
        if let Ok(()) = serialize_to_file(&array, &mut write_cursor, SerializeFormat::Npy) {
            let buffer = write_cursor.into_inner();

            group.bench_with_input(BenchmarkId::new("f64_1d", n), n, |b, _| {
                b.iter(|| {
                    // Create a fresh cursor over the slice on every iteration (zero-alloc).
                    let cursor = Cursor::new(buffer.as_slice());
                    if let Ok(arr) = deserialize_from_file::<f64, _>(cursor, SerializeFormat::Npy) {
                        black_box(arr);
                    }
                });
            });
        }
    }

    group.finish();
}

/// Benchmark writing a HashMap of 3 named f64 arrays to NPZ (uncompressed).
/// Total element counts: 10_000, 100_000, 1_000_000 (split evenly across 3 arrays).
/// Throughput is reported in bytes for the total data.
fn bench_npz_write_uncompressed(c: &mut Criterion) {
    let mut group = c.benchmark_group("npz_write_uncompressed");

    for total in [10_000usize, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes((*total * 8) as u64));
        let per_array = total / 3;

        let mut arrays: HashMap<String, Array<f64>> = HashMap::new();
        arrays.insert(
            "arr_a".to_string(),
            Array::from_vec(vec![0.0f64; per_array]),
        );
        arrays.insert(
            "arr_b".to_string(),
            Array::from_vec(vec![1.0f64; per_array]),
        );
        arrays.insert(
            "arr_c".to_string(),
            Array::from_vec(vec![2.0f64; per_array]),
        );

        group.bench_with_input(BenchmarkId::new("3_arrays", total), total, |b, _| {
            b.iter(|| {
                let cursor = Cursor::new(Vec::new());
                if let Ok(()) = save_npz_arrays(&arrays, cursor, false) {
                    black_box(());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark writing a HashMap of 3 named f64 arrays to NPZ (compressed / DEFLATE via OxiARC).
/// Same sizes as the uncompressed benchmark — shows compression overhead vs throughput.
/// Total element counts: 10_000, 100_000, 1_000_000 (split evenly across 3 arrays).
fn bench_npz_write_compressed(c: &mut Criterion) {
    let mut group = c.benchmark_group("npz_write_compressed");

    for total in [10_000usize, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes((*total * 8) as u64));
        let per_array = total / 3;

        let mut arrays: HashMap<String, Array<f64>> = HashMap::new();
        arrays.insert(
            "arr_a".to_string(),
            Array::from_vec(vec![0.0f64; per_array]),
        );
        arrays.insert(
            "arr_b".to_string(),
            Array::from_vec(vec![1.0f64; per_array]),
        );
        arrays.insert(
            "arr_c".to_string(),
            Array::from_vec(vec![2.0f64; per_array]),
        );

        group.bench_with_input(BenchmarkId::new("3_arrays", total), total, |b, _| {
            b.iter(|| {
                let cursor = Cursor::new(Vec::new());
                if let Ok(()) = save_npz_arrays(&arrays, cursor, true) {
                    black_box(());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark loading all arrays from a multi-array NPZ file.
/// Pre-saves the NPZ to a buffer outside the timing loop; only the read is measured.
/// Total element counts: 10_000, 100_000 elements (split evenly across 3 arrays).
fn bench_npz_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("npz_read");

    for total in [10_000usize, 100_000].iter() {
        group.throughput(Throughput::Bytes((*total * 8) as u64));
        let per_array = total / 3;

        let mut arrays: HashMap<String, Array<f64>> = HashMap::new();
        arrays.insert(
            "arr_a".to_string(),
            Array::from_vec(vec![0.0f64; per_array]),
        );
        arrays.insert(
            "arr_b".to_string(),
            Array::from_vec(vec![1.0f64; per_array]),
        );
        arrays.insert(
            "arr_c".to_string(),
            Array::from_vec(vec![2.0f64; per_array]),
        );

        // Pre-build the NPZ buffer outside the benchmark loop.
        // save_npz_arrays takes writer by value (W: Write+Seek), so use Cursor<&mut Vec<u8>>
        // which also satisfies Write+Seek while keeping ownership of the backing Vec.
        let mut npz_buf: Vec<u8> = Vec::new();
        if let Ok(()) = save_npz_arrays(&arrays, Cursor::new(&mut npz_buf), false) {
            group.bench_with_input(BenchmarkId::new("3_arrays", total), total, |b, _| {
                b.iter(|| {
                    // Fresh zero-alloc cursor slice on every iteration.
                    let cursor = Cursor::new(npz_buf.as_slice());
                    if let Ok(result) = load_all_npz_arrays::<f64, _>(cursor) {
                        black_box(result);
                    }
                });
            });
        }
    }

    group.finish();
}

/// Benchmark the full NPY write+read cycle for 100_000 f64 elements.
/// Shows total IO overhead for a complete roundtrip.
fn bench_npy_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_roundtrip");
    let n: usize = 100_000;
    group.throughput(Throughput::Bytes((n * 8) as u64));

    let array = Array::from_vec(vec![0.0f64; n]);

    group.bench_function("f64_100k", |b| {
        b.iter(|| {
            // Write phase.
            let mut cursor = Cursor::new(Vec::new());
            if let Ok(()) = serialize_to_file(&array, &mut cursor, SerializeFormat::Npy) {
                let buffer = cursor.into_inner();
                // Read phase.
                let read_cursor = Cursor::new(buffer.as_slice());
                if let Ok(arr) = deserialize_from_file::<f64, _>(read_cursor, SerializeFormat::Npy)
                {
                    black_box(arr);
                }
            }
        });
    });

    group.finish();
}

criterion_group!(
    io_benches,
    bench_npy_write,
    bench_npy_read,
    bench_npz_write_uncompressed,
    bench_npz_write_compressed,
    bench_npz_read,
    bench_npy_roundtrip,
);
criterion_main!(io_benches);

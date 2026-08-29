//! Benchmarks for FFT operations in NumRS2
//!
//! This file contains benchmarks for various FFT operations
//! to track performance and identify bottlenecks.

#![allow(deprecated)]
#![allow(clippy::result_large_err)]

#[macro_use]
extern crate criterion;
use criterion::{BenchmarkId, Criterion};

use numrs2::prelude::*;
use std::f64::consts::PI;
use std::hint::black_box;

/// Benchmark 1D FFT for different sizes
fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d");

    // Test FFT with different sizes (all powers of 2)
    for size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark FFT
        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| black_box(FFT::fft(&signal)))
        });

        // Benchmark IFFT (with a complex input)
        let complex_signal = signal.fft().unwrap();
        group.bench_with_input(BenchmarkId::new("ifft", size), size, |b, _| {
            b.iter(|| black_box(FFT::ifft(&complex_signal)))
        });
    }

    group.finish();
}

/// Benchmark real FFT operations
fn bench_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft");

    // Test RFFT with different sizes
    for size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark RFFT (real FFT)
        group.bench_with_input(BenchmarkId::new("rfft", size), size, |b, _| {
            b.iter(|| black_box(FFT::rfft(&signal)))
        });

        // Benchmark IRFFT (inverse real FFT)
        let rfft_result = signal.rfft().unwrap();
        group.bench_with_input(BenchmarkId::new("irfft", size), size, |b, _| {
            b.iter(|| black_box(FFT::irfft(&rfft_result, *size)))
        });
    }

    group.finish();
}

/// Benchmark 2D FFT operations
fn bench_fft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_2d");

    // Test 2D FFT with different sizes
    for size in [8, 16, 32, 64, 128].iter() {
        // Create a random 2D array
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size, *size]).unwrap();

        // Benchmark 2D FFT
        group.bench_with_input(BenchmarkId::new("fft2", size), size, |b, _| {
            b.iter(|| black_box(FFT::fft2(&signal)))
        });

        // Benchmark 2D IFFT
        let fft2_result = signal.fft2().unwrap();
        group.bench_with_input(BenchmarkId::new("ifft2", size), size, |b, _| {
            b.iter(|| black_box(FFT::ifft2(&fft2_result)))
        });

        // Benchmark 2D real FFT
        group.bench_with_input(BenchmarkId::new("rfft2", size), size, |b, _| {
            b.iter(|| black_box(FFT::rfft2(&signal)))
        });

        // Benchmark 2D inverse real FFT
        let rfft2_result = signal.rfft2().unwrap();
        group.bench_with_input(BenchmarkId::new("irfft2", size), size, |b, _| {
            b.iter(|| black_box(FFT::irfft2(&rfft2_result, &[*size, *size])))
        });
    }

    group.finish();
}

/// Benchmark window functions
fn bench_window_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_functions");

    // Test window functions with different sizes
    for size in [16, 64, 256, 1024, 4096].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark different window types
        for window_type in ["rectangular", "hann", "hamming", "blackman"].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("window_{}", window_type), size),
                size,
                |b, _| b.iter(|| black_box(FFT::apply_window(&signal, window_type))),
            );
        }
    }

    group.finish();
}

/// Benchmark FFT shift operations
fn bench_fft_shift(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_shift");

    // Test FFT shift with different sizes
    for size in [16, 64, 256, 1024, 4096].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark fftshift
        group.bench_with_input(BenchmarkId::new("fftshift", size), size, |b, _| {
            b.iter(|| black_box(FFT::fftshift(&signal)))
        });

        // Benchmark ifftshift
        group.bench_with_input(BenchmarkId::new("ifftshift", size), size, |b, _| {
            b.iter(|| black_box(FFT::ifftshift(&signal)))
        });
    }

    group.finish();
}

/// Benchmark FFT frequency axis generation
fn bench_fft_freq(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_freq");

    // Test frequency axis generation with different sizes
    for size in [16, 64, 256, 1024, 4096].iter() {
        let sample_rate = 1000.0; // Hz

        // Benchmark fftfreq
        group.bench_with_input(BenchmarkId::new("fftfreq", size), size, |b, _| {
            b.iter(|| black_box(FFT::fftfreq(*size, 1.0 / sample_rate)))
        });

        // Benchmark rfftfreq
        group.bench_with_input(BenchmarkId::new("rfftfreq", size), size, |b, _| {
            b.iter(|| black_box(FFT::rfftfreq(*size, 1.0 / sample_rate)))
        });
    }

    group.finish();
}

/// Benchmark power spectrum calculation
fn bench_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_spectrum");

    // Test power spectrum calculation with different sizes
    for size in [16, 64, 256, 1024, 4096].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark power spectrum
        group.bench_with_input(BenchmarkId::new("power_spectrum", size), size, |b, _| {
            b.iter(|| black_box(FFT::power_spectrum(&signal)))
        });
    }

    group.finish();
}

/// Benchmark FFT on different signal types
fn bench_fft_signal_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_signal_types");

    // Define size for all tests
    let size = 1024;

    // Create different signal types

    // 1. Random signal
    let rng = random::default_rng();
    let random_signal = rng.random::<f64>(&[size]).unwrap();

    // 2. Sinusoidal signal
    let mut sine_data = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        sine_data.push((2.0 * PI * 10.0 * t).sin());
    }
    let sine_signal = Array::from_vec(sine_data);

    // 3. Square wave
    let mut square_data = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        square_data.push(if (t * 10.0) % 1.0 < 0.5 { 1.0 } else { -1.0 });
    }
    let square_signal = Array::from_vec(square_data);

    // 4. Impulse (delta function)
    let mut impulse_data = vec![0.0; size];
    impulse_data[0] = 1.0;
    let impulse_signal = Array::from_vec(impulse_data);

    // 5. Complex exponential - commented out as FFT::fft expects real input
    // let mut complex_exp_data = Vec::with_capacity(size);
    // for i in 0..size {
    //     let angle = 2.0 * PI * 10.0 * i as f64 / size as f64;
    //     complex_exp_data.push(Complex64::new(angle.cos(), angle.sin()));
    // }
    // let complex_exp_signal = Array::from_vec(complex_exp_data);

    // Benchmark FFT on different signal types
    group.bench_function("fft_random", |b| {
        b.iter(|| black_box(FFT::fft(&random_signal)))
    });

    group.bench_function("fft_sine", |b| b.iter(|| black_box(FFT::fft(&sine_signal))));

    group.bench_function("fft_square", |b| {
        b.iter(|| black_box(FFT::fft(&square_signal)))
    });

    group.bench_function("fft_impulse", |b| {
        b.iter(|| black_box(FFT::fft(&impulse_signal)))
    });

    // group.bench_function("fft_complex_exp", |b| {
    //     b.iter(|| black_box(FFT::fft(&complex_exp_signal)))
    // });

    group.finish();
}

/// Benchmark end-to-end FFT workflow
fn bench_fft_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_workflow");

    // Test end-to-end FFT workflow with different sizes
    for size in [64, 256, 1024].iter() {
        // Define the workflow benchmark
        group.bench_with_input(BenchmarkId::new("fft_workflow", size), size, |b, _| {
            b.iter(|| {
                // Create a random signal
                let rng = random::default_rng();
                let signal = rng.random::<f64>(&[*size]).unwrap();

                // Apply window function
                let windowed = black_box(FFT::apply_window(&signal, "hann").unwrap());

                // Compute FFT
                let fft_result = black_box(FFT::fft(&windowed).unwrap());

                // Compute power spectrum
                let power = black_box(Array::from_vec(
                    fft_result
                        .to_vec()
                        .iter()
                        .map(|val| val.norm_sqr())
                        .collect(),
                ));

                // Find peak frequency
                let mut max_power = 0.0;
                let mut max_idx = 0;
                for (i, &val) in power.to_vec().iter().enumerate() {
                    if val > max_power {
                        max_power = val;
                        max_idx = i;
                    }
                }

                // Compute frequency axis
                let freq_axis = black_box(FFT::fftfreq(*size, 1.0 / 1000.0).unwrap());

                // Get the peak frequency
                black_box(freq_axis.to_vec()[max_idx])
            })
        });
    }

    group.finish();
}

/// Compare performance between Array::fft and FFT::fft
fn bench_fft_implementation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_implementation");

    // Test both implementations with different sizes
    for size in [64, 256, 1024].iter() {
        // Create a random signal
        let rng = random::default_rng();
        let signal = rng.random::<f64>(&[*size]).unwrap();

        // Benchmark Array::fft (method)
        group.bench_with_input(BenchmarkId::new("array_fft", size), size, |b, _| {
            b.iter(|| black_box(signal.fft()))
        });

        // Benchmark FFT::fft (static function)
        group.bench_with_input(BenchmarkId::new("fft_fft", size), size, |b, _| {
            b.iter(|| black_box(FFT::fft(&signal)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fft_1d,
    bench_rfft,
    bench_fft_2d,
    bench_window_functions,
    bench_fft_shift,
    bench_fft_freq,
    bench_power_spectrum,
    bench_fft_signal_types,
    bench_fft_workflow,
    bench_fft_implementation,
);
criterion_main!(benches);

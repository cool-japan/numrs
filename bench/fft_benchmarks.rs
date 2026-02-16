//! Comprehensive FFT Benchmarks for NumRS2
//!
//! This benchmark suite tests all FFT operations including:
//! - FFT/IFFT (1D) - various sizes (powers of 2)
//! - FFT/IFFT (2D) - various sizes
//! - Real FFT (RFFT)
//! - DCT/DST transforms (if available)
//! - Window functions
//! - FFT utilities
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::prelude::*;
use std::f64::consts::PI;
use std::hint::black_box;

/// Benchmark 1D FFT for various sizes
fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d");

    // Test FFT with different sizes (all powers of 2)
    for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::fft(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark 1D IFFT for various sizes
fn bench_ifft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_1d");

    for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("ifft", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                if let Ok(fft_result) = signal.fft() {
                    b.iter(|| {
                        if let Ok(result) = FFT::ifft(&fft_result) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark real FFT (RFFT) for various sizes
fn bench_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft");

    for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("rfft", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::rfft(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark inverse real FFT (IRFFT) for various sizes
fn bench_irfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("irfft");

    for size in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("irfft", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                if let Ok(rfft_result) = signal.rfft() {
                    b.iter(|| {
                        if let Ok(result) = FFT::irfft(&rfft_result, s) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark 2D FFT for various sizes
fn bench_fft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_2d");

    for size in [8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("fft2", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::fft2(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark 2D IFFT for various sizes
fn bench_ifft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_2d");

    for size in [8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("ifft2", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s, s]) {
                if let Ok(fft2_result) = signal.fft2() {
                    b.iter(|| {
                        if let Ok(result) = FFT::ifft2(&fft2_result) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark 2D real FFT for various sizes
fn bench_rfft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft_2d");

    for size in [8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("rfft2", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s, s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::rfft2(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark 2D inverse real FFT for various sizes
fn bench_irfft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("irfft_2d");

    for size in [8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("irfft2", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s, s]) {
                if let Ok(rfft2_result) = signal.rfft2() {
                    b.iter(|| {
                        if let Ok(result) = FFT::irfft2(&rfft2_result, &[s, s]) {
                            black_box(result);
                        }
                    });
                }
            }
        });
    }

    group.finish();
}

/// Benchmark window functions
fn bench_window_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_functions");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        // Rectangular window
        group.bench_with_input(BenchmarkId::new("rectangular", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::apply_window(&signal, "rectangular") {
                        black_box(result);
                    }
                });
            }
        });

        // Hann window
        group.bench_with_input(BenchmarkId::new("hann", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::apply_window(&signal, "hann") {
                        black_box(result);
                    }
                });
            }
        });

        // Hamming window
        group.bench_with_input(BenchmarkId::new("hamming", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::apply_window(&signal, "hamming") {
                        black_box(result);
                    }
                });
            }
        });

        // Blackman window
        group.bench_with_input(BenchmarkId::new("blackman", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::apply_window(&signal, "blackman") {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark FFT shift operations
fn bench_fft_shift(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_shift");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("fftshift", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::fftshift(&signal) {
                        black_box(result);
                    }
                });
            }
        });

        group.bench_with_input(BenchmarkId::new("ifftshift", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::ifftshift(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark FFT frequency axis generation
fn bench_fft_freq(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_freq");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let sample_rate = 1000.0; // Hz

        group.bench_with_input(BenchmarkId::new("fftfreq", size), size, |b, &s| {
            b.iter(|| {
                if let Ok(result) = FFT::fftfreq(s, 1.0 / sample_rate) {
                    black_box(result);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("rfftfreq", size), size, |b, &s| {
            b.iter(|| {
                if let Ok(result) = FFT::rfftfreq(s, 1.0 / sample_rate) {
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark power spectrum calculation
fn bench_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_spectrum");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.bench_with_input(BenchmarkId::new("power_spectrum", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = FFT::power_spectrum(&signal) {
                        black_box(result);
                    }
                });
            }
        });
    }

    group.finish();
}

/// Benchmark FFT on different signal types
fn bench_fft_signal_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_signal_types");

    let size = 1024;

    // Random signal
    let rng = random::default_rng();
    if let Ok(random_signal) = rng.random::<f64>(&[size]) {
        group.bench_function("fft_random", |b| {
            b.iter(|| {
                if let Ok(result) = FFT::fft(&random_signal) {
                    black_box(result);
                }
            });
        });
    }

    // Sinusoidal signal
    let mut sine_data = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        sine_data.push((2.0 * PI * 10.0 * t).sin());
    }
    let sine_signal = Array::from_vec(sine_data);
    group.bench_function("fft_sine", |b| {
        b.iter(|| {
            if let Ok(result) = FFT::fft(&sine_signal) {
                black_box(result);
            }
        });
    });

    // Square wave
    let mut square_data = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        square_data.push(if (t * 10.0) % 1.0 < 0.5 { 1.0 } else { -1.0 });
    }
    let square_signal = Array::from_vec(square_data);
    group.bench_function("fft_square", |b| {
        b.iter(|| {
            if let Ok(result) = FFT::fft(&square_signal) {
                black_box(result);
            }
        });
    });

    // Impulse (delta function)
    let mut impulse_data = vec![0.0; size];
    impulse_data[0] = 1.0;
    let impulse_signal = Array::from_vec(impulse_data);
    group.bench_function("fft_impulse", |b| {
        b.iter(|| {
            if let Ok(result) = FFT::fft(&impulse_signal) {
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark end-to-end FFT workflow
fn bench_fft_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_workflow");

    for size in [256, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("complete_workflow", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let rng = random::default_rng();
                    if let Ok(signal) = rng.random::<f64>(&[s]) {
                        // Apply window
                        if let Ok(windowed) = FFT::apply_window(&signal, "hann") {
                            // Compute FFT
                            if let Ok(fft_result) = FFT::fft(&windowed) {
                                // Compute power spectrum
                                let power_vec: Vec<f64> = fft_result
                                    .to_vec()
                                    .iter()
                                    .map(|val| val.norm_sqr())
                                    .collect();
                                let power = Array::from_vec(power_vec);

                                // Find peak
                                let mut max_power = 0.0;
                                let mut max_idx = 0;
                                for (i, &val) in power.to_vec().iter().enumerate() {
                                    if val > max_power {
                                        max_power = val;
                                        max_idx = i;
                                    }
                                }

                                // Get frequency axis
                                if let Ok(freq_axis) = FFT::fftfreq(s, 1.0 / 1000.0) {
                                    let peak_freq = freq_axis.to_vec()[max_idx];
                                    black_box(peak_freq);
                                }
                            }
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Compare FFT implementations
fn bench_fft_implementation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_implementation_comparison");

    for size in [256, 1024, 4096].iter() {
        // Method syntax
        group.bench_with_input(BenchmarkId::new("array_method_fft", size), size, |b, &s| {
            let rng = random::default_rng();
            if let Ok(signal) = rng.random::<f64>(&[s]) {
                b.iter(|| {
                    if let Ok(result) = signal.fft() {
                        black_box(result);
                    }
                });
            }
        });

        // Static function syntax
        group.bench_with_input(
            BenchmarkId::new("static_function_fft", size),
            size,
            |b, &s| {
                let rng = random::default_rng();
                if let Ok(signal) = rng.random::<f64>(&[s]) {
                    b.iter(|| {
                        if let Ok(result) = FFT::fft(&signal) {
                            black_box(result);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark FFT with different data types
fn bench_fft_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_data_types");

    let size = 1024;

    // f64
    group.bench_function("fft_f64", |b| {
        let rng = random::default_rng();
        if let Ok(signal) = rng.random::<f64>(&[size]) {
            b.iter(|| {
                if let Ok(result) = FFT::fft(&signal) {
                    black_box(result);
                }
            });
        }
    });

    // Note: FFT currently requires T: From<f64>, so f32 is not supported directly
    // Removed f32 benchmark to avoid compilation error

    group.finish();
}

criterion_group!(
    benches,
    bench_fft_1d,
    bench_ifft_1d,
    bench_rfft,
    bench_irfft,
    bench_fft_2d,
    bench_ifft_2d,
    bench_rfft_2d,
    bench_irfft_2d,
    bench_window_functions,
    bench_fft_shift,
    bench_fft_freq,
    bench_power_spectrum,
    bench_fft_signal_types,
    bench_fft_workflow,
    bench_fft_implementation_comparison,
    bench_fft_data_types,
);

criterion_main!(benches);

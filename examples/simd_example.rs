#![allow(deprecated)]

use numrs2::prelude::*;
use numrs2::simd::{simd_add, simd_mul, simd_div, simd_exp, simd_log, simd_sqrt, simd_sum, simd_prod};

fn main() {
    println!("NumRS SIMD Operations Example");
    println!("===========================");

    // Detect CPU features and select SIMD implementation
    let features = detect_cpu_features();
    let implementation = get_simd_implementation();

    println!("Detected CPU features:");
    println!("  SSE2: {}", features.sse2);
    println!("  SSE3: {}", features.sse3);
    println!("  SSSE3: {}", features.ssse3);
    println!("  SSE4.1: {}", features.sse4_1);
    println!("  SSE4.2: {}", features.sse4_2);
    println!("  AVX: {}", features.avx);
    println!("  AVX2: {}", features.avx2);
    println!("  FMA: {}", features.fma);
    println!("  AVX-512F: {}", features.avx512f);
    println!("  NEON: {}", features.neon);
    println!("  SVE: {}", features.sve);

    println!("\nSelected SIMD implementation: {}", implementation.name());

    // Create arrays
    let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let b = Array::from_vec(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[2, 3]);

    println!("\nArray a:");
    println!("{}", a);

    println!("\nArray b:");
    println!("{}", b);

    // SIMD addition
    let c = simd_add(&a, &b).unwrap();
    println!("\nSIMD Add (a + b):");
    println!("{}", c);

    // SIMD multiplication
    let d = simd_mul(&a, &b).unwrap();
    println!("\nSIMD Multiply (a * b):");
    println!("{}", d);

    // SIMD division
    let e = simd_div(&a, &b).unwrap();
    println!("\nSIMD Divide (a / b):");
    println!("{}", e);

    // SIMD exponential
    let f = simd_exp(&a);
    println!("\nSIMD Exp (exp(a)):");
    println!("{}", f);

    // SIMD logarithm
    let g = simd_log(&b);
    println!("\nSIMD Log (log(b)):");
    println!("{}", g);

    // SIMD square root
    let h = simd_sqrt(&a);
    println!("\nSIMD Sqrt (sqrt(a)):");
    println!("{}", h);

    // SIMD reduction operations
    let sum = simd_sum(&a);
    println!("\nSIMD Sum of all elements in a: {}", sum);

    let prod = simd_prod(&a);
    println!("SIMD Product of all elements in a: {}", prod);

    // Performance comparison
    println!("\nPerformance Comparison");
    println!("===========================");

    // Create a larger array for performance testing
    let large_array_size = 1_000_000;
    let large_array = Array::<f64>::ones(&[large_array_size]);

    // Time standard operations
    let start = std::time::Instant::now();
    let _ = large_array.add(&large_array);
    let standard_duration = start.elapsed();
    println!("Standard addition time: {:?}", standard_duration);

    // Time SIMD operations
    let start = std::time::Instant::now();
    let _ = simd_add(&large_array, &large_array).unwrap();
    let simd_duration = start.elapsed();
    println!("SIMD addition time: {:?}", simd_duration);

    // Calculate speedup
    let speedup = standard_duration.as_secs_f64() / simd_duration.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
}

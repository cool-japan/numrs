//! SIMD Operations Example
//!
//! This example demonstrates the SIMD-accelerated operations available in NumRS2
//! using scirs2-core's SimdUnifiedOps trait.

use numrs2::prelude::*;
use numrs2::simd::{get_simd_implementation_name, SimdOps};

fn main() {
    println!("NumRS SIMD Operations Example");
    println!("=============================");

    // Get SIMD implementation info
    let implementation_name = get_simd_implementation_name();
    println!("\nUsing SIMD implementation: {}", implementation_name);

    // Create arrays
    let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let b = Array::from_vec(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[2, 3]);

    println!("\nArray a:");
    println!("{}", a);

    println!("\nArray b:");
    println!("{}", b);

    // SIMD operations using SimdOps trait
    println!("\n--- SIMD Operations via SimdOps Trait ---");

    // Flatten arrays for SIMD operations
    let a_flat = a.flatten(None);
    let b_flat = b.flatten(None);

    // SIMD addition
    let c = a_flat.simd_add(&b_flat).unwrap();
    println!("\nSIMD Add (a + b): {:?}", c.to_vec());

    // SIMD multiplication
    let d = a_flat.simd_mul(&b_flat).unwrap();
    println!("SIMD Multiply (a * b): {:?}", d.to_vec());

    // SIMD division
    let e = a_flat.simd_div(&b_flat).unwrap();
    println!("SIMD Divide (a / b): {:?}", e.to_vec());

    // SIMD dot product
    let dot_result = a_flat.simd_dot(&b_flat).unwrap();
    println!("SIMD Dot product (a · b): {}", dot_result);

    // SIMD reduction operations
    let sum = a_flat.simd_sum();
    println!("\nSIMD Sum of all elements in a: {}", sum);

    let mean = a_flat.simd_mean();
    println!("SIMD Mean of all elements in a: {}", mean);

    // SIMD scalar operations
    let scaled = a_flat.simd_mul_scalar(2.0);
    println!("\nSIMD Scalar multiply (a * 2): {:?}", scaled.to_vec());

    let shifted = a_flat.simd_add_scalar(10.0);
    println!("SIMD Scalar add (a + 10): {:?}", shifted.to_vec());

    // SIMD FMA (fused multiply-add)
    let mul_factor = Array::from_vec(vec![2.0f64; 6]);
    let add_factor = Array::from_vec(vec![1.0f64; 6]);
    let fma_result = a_flat.simd_fma(&mul_factor, &add_factor).unwrap();
    println!("SIMD FMA (a * 2 + 1): {:?}", fma_result.to_vec());

    // Performance comparison
    println!("\n--- Performance Comparison ---");

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
    let _ = large_array.simd_add(&large_array).unwrap();
    let simd_duration = start.elapsed();
    println!("SIMD addition time: {:?}", simd_duration);

    // Calculate speedup
    if simd_duration.as_nanos() > 0 {
        let speedup = standard_duration.as_secs_f64() / simd_duration.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    } else {
        println!("SIMD was too fast to measure accurately!");
    }

    // Demonstrate platform capabilities
    println!("\n--- Platform Capabilities ---");
    println!(
        "This NumRS2 build uses scirs2-core's SimdUnifiedOps trait,\n\
         which automatically selects the best SIMD implementation\n\
         for your platform (AVX2, AVX-512, NEON, etc.)"
    );
}

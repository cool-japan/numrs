#![feature(test)]

extern crate numrs2;
extern crate serde_json;
extern crate test;

use numrs2::array::Array;
use numrs2::random::distributions::*;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use test::Bencher;

// Sample sizes for benchmarks
const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 10_000;
const LARGE_SIZE: usize = 100_000;
const MATRIX_SMALL: [usize; 2] = [100, 100];
const MATRIX_MEDIUM: [usize; 2] = [1000, 1000];
const MATRIX_LARGE: [usize; 2] = [5000, 5000];

// Optionally load NumPy benchmark results for comparison
fn load_numpy_results() -> Option<serde_json::Value> {
    let numpy_file = Path::new("bench/numpy/numpy_benchmark_results.json");
    if numpy_file.exists() {
        match File::open(numpy_file) {
            Ok(file) => {
                let reader = BufReader::new(file);
                match serde_json::from_reader(reader) {
                    Ok(value) => Some(value),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    } else {
        None
    }
}

// Print comparison with NumPy if results are available
fn print_comparison(bench_name: &str, numrs_time_ns: u64) {
    if let Some(numpy_results) = load_numpy_results() {
        // Try to find the corresponding NumPy benchmark
        // Convert bench name from Rust style to Python style
        // e.g. "bench_zeros_medium" -> "zeros_medium"
        let numpy_name = bench_name.replace("bench_", "");

        // Find the category
        let categories = [
            "array_creation",
            "array_operations",
            "linear_algebra",
            "distributions",
        ];
        for category in &categories {
            if let Some(cat_results) = numpy_results[category].as_object() {
                if let Some(result) = cat_results.get(&numpy_name) {
                    if let Some(numpy_time_ms) = result["mean"].as_f64() {
                        // Convert NumRS2 time from ns to ms for comparison
                        let numrs_time_ms = numrs_time_ns as f64 / 1_000_000.0;

                        // Calculate speedup/slowdown
                        let ratio = numpy_time_ms / numrs_time_ms;

                        let comparison = if ratio > 1.0 {
                            format!("NumRS2 is {:.2}x faster than NumPy", ratio)
                        } else {
                            format!("NumRS2 is {:.2}x slower than NumPy", 1.0 / ratio)
                        };

                        println!(
                            "Benchmark: {} - NumRS2: {:.2} ms, NumPy: {:.2} ms - {}",
                            numpy_name, numrs_time_ms, numpy_time_ms, comparison
                        );
                        return;
                    }
                }
            }
        }
    }
}

// Array Creation benchmarks
#[bench]
fn bench_zeros_small(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::zeros(&[SMALL_SIZE]);
    });

    print_comparison("zeros_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_zeros_medium(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::zeros(&[MEDIUM_SIZE]);
    });

    print_comparison("zeros_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_zeros_large(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::zeros(&[LARGE_SIZE]);
    });

    print_comparison("zeros_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_ones_small(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::ones(&[SMALL_SIZE]);
    });

    print_comparison("ones_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_ones_medium(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::ones(&[MEDIUM_SIZE]);
    });

    print_comparison("ones_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_ones_large(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::ones(&[LARGE_SIZE]);
    });

    print_comparison("ones_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_random_small(b: &mut Bencher) {
    b.iter(|| {
        let _ = uniform(0.0, 1.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison("random_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_random_medium(b: &mut Bencher) {
    b.iter(|| {
        let _ = uniform(0.0, 1.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison("random_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_random_large(b: &mut Bencher) {
    b.iter(|| {
        let _ = uniform(0.0, 1.0, &[LARGE_SIZE]).unwrap();
    });

    print_comparison("random_large", b.bench(test::black_box).unwrap().median);
}

// 2D array creation
#[bench]
fn bench_zeros_matrix_small(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::zeros(&MATRIX_SMALL);
    });

    print_comparison(
        "zeros_matrix_small",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_zeros_matrix_medium(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::zeros(&MATRIX_MEDIUM);
    });

    print_comparison(
        "zeros_matrix_medium",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_ones_matrix_small(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::ones(&MATRIX_SMALL);
    });

    print_comparison(
        "ones_matrix_small",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_ones_matrix_medium(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::ones(&MATRIX_MEDIUM);
    });

    print_comparison(
        "ones_matrix_medium",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_identity_small(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::identity(100);
    });

    print_comparison("identity_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_identity_medium(b: &mut Bencher) {
    b.iter(|| {
        let _arr = Array::<f64>::identity(1000);
    });

    print_comparison("identity_medium", b.bench(test::black_box).unwrap().median);
}

// Array operations
#[bench]
fn bench_add_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);
    let b = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _c = &a + &b;
    });

    print_comparison("add_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_add_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);
    let b = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _c = &a + &b;
    });

    print_comparison("add_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_add_large(b: &mut Bencher) {
    let a = Array::<f64>::random(&[LARGE_SIZE]);
    let b = Array::<f64>::random(&[LARGE_SIZE]);

    b.iter(|| {
        let _c = &a + &b;
    });

    print_comparison("add_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_multiply_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);
    let b = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _c = &a * &b;
    });

    print_comparison("multiply_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_multiply_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);
    let b = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _c = &a * &b;
    });

    print_comparison("multiply_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_multiply_large(b: &mut Bencher) {
    let a = Array::<f64>::random(&[LARGE_SIZE]);
    let b = Array::<f64>::random(&[LARGE_SIZE]);

    b.iter(|| {
        let _c = &a * &b;
    });

    print_comparison("multiply_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_sqrt_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.sqrt();
    });

    print_comparison("sqrt_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_sqrt_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.sqrt();
    });

    print_comparison("sqrt_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_sqrt_large(b: &mut Bencher) {
    let a = Array::<f64>::random(&[LARGE_SIZE]);

    b.iter(|| {
        let _result = a.sqrt();
    });

    print_comparison("sqrt_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_exp_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.exp();
    });

    print_comparison("exp_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_exp_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.exp();
    });

    print_comparison("exp_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_log_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]) + 0.1; // Ensure positive values

    b.iter(|| {
        let _result = a.ln();
    });

    print_comparison("log_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_log_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]) + 0.1; // Ensure positive values

    b.iter(|| {
        let _result = a.ln();
    });

    print_comparison("log_medium", b.bench(test::black_box).unwrap().median);
}

// Reductions
#[bench]
fn bench_sum_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.sum();
    });

    print_comparison("sum_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_sum_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.sum();
    });

    print_comparison("sum_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_sum_large(b: &mut Bencher) {
    let a = Array::<f64>::random(&[LARGE_SIZE]);

    b.iter(|| {
        let _result = a.sum();
    });

    print_comparison("sum_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_mean_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.mean();
    });

    print_comparison("mean_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_mean_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.mean();
    });

    print_comparison("mean_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_mean_large(b: &mut Bencher) {
    let a = Array::<f64>::random(&[LARGE_SIZE]);

    b.iter(|| {
        let _result = a.mean();
    });

    print_comparison("mean_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_std_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.std();
    });

    print_comparison("std_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_std_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.std();
    });

    print_comparison("std_medium", b.bench(test::black_box).unwrap().median);
}

// Array manipulation
#[bench]
fn bench_reshape_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE]);

    b.iter(|| {
        let _result = a.reshape(&[SMALL_SIZE / 10, 10]);
    });

    print_comparison("reshape_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_reshape_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE]);

    b.iter(|| {
        let _result = a.reshape(&[MEDIUM_SIZE / 100, 100]);
    });

    print_comparison("reshape_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_transpose_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[SMALL_SIZE / 10, 10]);

    b.iter(|| {
        let _result = a.transpose();
    });

    print_comparison("transpose_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_transpose_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[MEDIUM_SIZE / 100, 100]);

    b.iter(|| {
        let _result = a.transpose();
    });

    print_comparison("transpose_medium", b.bench(test::black_box).unwrap().median);
}

// Linear algebra
#[bench]
fn bench_matmul_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&MATRIX_SMALL);
    let b = Array::<f64>::random(&MATRIX_SMALL);

    b.iter(|| {
        let _result = a.matmul(&b);
    });

    print_comparison("matmul_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_dot_small(b: &mut Bencher) {
    let a = Array::<f64>::random(&[100]);
    let b = Array::<f64>::random(&[100]);

    b.iter(|| {
        let _result = a.dot(&b);
    });

    print_comparison("dot_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_dot_medium(b: &mut Bencher) {
    let a = Array::<f64>::random(&[1000]);
    let b = Array::<f64>::random(&[1000]);

    b.iter(|| {
        let _result = a.dot(&b);
    });

    print_comparison("dot_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_svd_small(b: &mut Bencher) {
    use numrs2::linalg;
    let a = Array::<f64>::random(&[100, 100]);

    b.iter(|| {
        let _result = linalg::svd(&a);
    });

    print_comparison("svd_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_qr_small(b: &mut Bencher) {
    use numrs2::linalg;
    let a = Array::<f64>::random(&[100, 100]);

    b.iter(|| {
        let _result = linalg::qr(&a);
    });

    print_comparison("qr_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_cholesky_small(b: &mut Bencher) {
    use numrs2::linalg;
    // Create a positive definite matrix
    let a = Array::<f64>::random(&[100, 100]);
    let a_t = a.transpose();
    let pd_matrix = a.matmul(&a_t);

    b.iter(|| {
        let _result = linalg::cholesky(&pd_matrix);
    });

    print_comparison("cholesky_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_det_small(b: &mut Bencher) {
    use numrs2::linalg;
    let a = Array::<f64>::random(&[100, 100]);

    b.iter(|| {
        let _result = linalg::det(&a);
    });

    print_comparison("det_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_inv_small(b: &mut Bencher) {
    use numrs2::linalg;
    let a = Array::<f64>::random(&[100, 100]);

    b.iter(|| {
        let _result = linalg::inv(&a);
    });

    print_comparison("inv_small", b.bench(test::black_box).unwrap().median);
}

// Distributions
#[bench]
fn bench_normal_small(b: &mut Bencher) {
    b.iter(|| {
        let _result = normal(0.0, 1.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison("normal_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_normal_medium(b: &mut Bencher) {
    b.iter(|| {
        let _result = normal(0.0, 1.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison("normal_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_normal_large(b: &mut Bencher) {
    b.iter(|| {
        let _result = normal(0.0, 1.0, &[LARGE_SIZE]).unwrap();
    });

    print_comparison("normal_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_uniform_small(b: &mut Bencher) {
    b.iter(|| {
        let _result = uniform(0.0, 1.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison("uniform_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_uniform_medium(b: &mut Bencher) {
    b.iter(|| {
        let _result = uniform(0.0, 1.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison("uniform_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_uniform_large(b: &mut Bencher) {
    b.iter(|| {
        let _result = uniform(0.0, 1.0, &[LARGE_SIZE]).unwrap();
    });

    print_comparison("uniform_large", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_exponential_small(b: &mut Bencher) {
    b.iter(|| {
        let _result = exponential(1.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison(
        "exponential_small",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_exponential_medium(b: &mut Bencher) {
    b.iter(|| {
        let _result = exponential(1.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison(
        "exponential_medium",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_exponential_large(b: &mut Bencher) {
    b.iter(|| {
        let _result = exponential(1.0, &[LARGE_SIZE]).unwrap();
    });

    print_comparison(
        "exponential_large",
        b.bench(test::black_box).unwrap().median,
    );
}

#[bench]
fn bench_beta_small(b: &mut Bencher) {
    b.iter(|| {
        let _result = beta(2.0, 5.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison("beta_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_beta_medium(b: &mut Bencher) {
    b.iter(|| {
        let _result = beta(2.0, 5.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison("beta_medium", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_gamma_small(b: &mut Bencher) {
    b.iter(|| {
        let _result = gamma(2.0, 2.0, &[SMALL_SIZE]).unwrap();
    });

    print_comparison("gamma_small", b.bench(test::black_box).unwrap().median);
}

#[bench]
fn bench_gamma_medium(b: &mut Bencher) {
    b.iter(|| {
        let _result = gamma(2.0, 2.0, &[MEDIUM_SIZE]).unwrap();
    });

    print_comparison("gamma_medium", b.bench(test::black_box).unwrap().median);
}

// SciRS2 integration benchmarks for advanced distributions (only run when feature is enabled)
#[cfg(feature = "scirs")]
mod scirs_benchmarks {
    use super::*;
    use numrs2::interop::scirs_compat::*;

    #[bench]
    fn bench_noncentral_chisquare_small(b: &mut Bencher) {
        b.iter(|| {
            let _result = noncentral_chisquare(2.0, 1.0, &[SMALL_SIZE]).unwrap();
        });

        print_comparison(
            "noncentral_chisquare_small",
            b.bench(test::black_box).unwrap().median,
        );
    }

    #[bench]
    fn bench_noncentral_chisquare_medium(b: &mut Bencher) {
        b.iter(|| {
            let _result = noncentral_chisquare(2.0, 1.0, &[MEDIUM_SIZE]).unwrap();
        });

        print_comparison(
            "noncentral_chisquare_medium",
            b.bench(test::black_box).unwrap().median,
        );
    }

    #[bench]
    fn bench_noncentral_f_small(b: &mut Bencher) {
        b.iter(|| {
            let _result = noncentral_f(2.0, 5.0, 1.0, &[SMALL_SIZE]).unwrap();
        });

        print_comparison(
            "noncentral_f_small",
            b.bench(test::black_box).unwrap().median,
        );
    }

    #[bench]
    fn bench_noncentral_f_medium(b: &mut Bencher) {
        b.iter(|| {
            let _result = noncentral_f(2.0, 5.0, 1.0, &[MEDIUM_SIZE]).unwrap();
        });

        print_comparison(
            "noncentral_f_medium",
            b.bench(test::black_box).unwrap().median,
        );
    }

    #[bench]
    fn bench_vonmises_small(b: &mut Bencher) {
        b.iter(|| {
            let _result = vonmises(0.0, 1.0, &[SMALL_SIZE]).unwrap();
        });

        print_comparison("vonmises_small", b.bench(test::black_box).unwrap().median);
    }

    #[bench]
    fn bench_vonmises_medium(b: &mut Bencher) {
        b.iter(|| {
            let _result = vonmises(0.0, 1.0, &[MEDIUM_SIZE]).unwrap();
        });

        print_comparison("vonmises_medium", b.bench(test::black_box).unwrap().median);
    }
}

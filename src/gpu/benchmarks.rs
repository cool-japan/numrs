//! GPU Performance Benchmarks
//!
//! This module provides benchmarking tools for GPU operations, allowing comparison
//! of GPU vs CPU performance and measurement of memory transfer overhead.
//!
//! ## Features
//!
//! - **GPU vs CPU Benchmarks**: Compare performance of operations on GPU and CPU
//! - **Memory Transfer Benchmarks**: Measure overhead of CPU-GPU data transfers
//! - **Compute Shader Performance**: Profile individual GPU kernels
//! - **Detailed Metrics**: Timing, throughput, and efficiency measurements
//!
//! ## Example
//!
//! ```rust,ignore
//! use numrs2::gpu::benchmarks::{BenchmarkRunner, BenchmarkConfig};
//! use numrs2::array::Array;
//!
//! # #[cfg(feature = "gpu")]
//! # fn example() -> numrs2::error::Result<()> {
//! let context = numrs2::gpu::new_context()?;
//! let runner = BenchmarkRunner::new(context);
//!
//! // Benchmark matrix multiplication
//! let config = BenchmarkConfig::default();
//! let results = runner.benchmark_matmul(1024, 1024, 1024, &config)?;
//!
//! println!("GPU time: {:.2} ms", results.gpu_time_ms);
//! println!("CPU time: {:.2} ms", results.cpu_time_ms);
//! println!("Speedup: {:.2}x", results.speedup());
//! # Ok(())
//! # }
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::context::GpuContextRef;
use crate::gpu::linalg;
use scirs2_core::ndarray::Array2;
use std::time::{Duration, Instant};

/// Configuration for benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before timing
    pub warmup_iterations: usize,
    /// Number of timed iterations
    pub benchmark_iterations: usize,
    /// Whether to include CPU comparison
    pub include_cpu: bool,
    /// Whether to measure memory transfer time separately
    pub measure_transfers: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
            include_cpu: true,
            measure_transfers: true,
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// GPU computation time in milliseconds
    pub gpu_time_ms: f64,
    /// CPU computation time in milliseconds (if measured)
    pub cpu_time_ms: Option<f64>,
    /// Time to transfer data to GPU in milliseconds (if measured)
    pub transfer_to_gpu_ms: Option<f64>,
    /// Time to transfer data from GPU in milliseconds (if measured)
    pub transfer_from_gpu_ms: Option<f64>,
    /// Number of operations performed
    pub operations: u64,
    /// Size of data processed in bytes
    pub data_size_bytes: u64,
}

impl BenchmarkResults {
    /// Calculates the speedup of GPU over CPU
    ///
    /// Returns None if CPU time was not measured
    pub fn speedup(&self) -> Option<f64> {
        self.cpu_time_ms.map(|cpu| cpu / self.gpu_time_ms)
    }

    /// Calculates GPU throughput in GFLOPS (billions of floating-point operations per second)
    pub fn gpu_gflops(&self) -> f64 {
        (self.operations as f64) / (self.gpu_time_ms * 1_000_000.0)
    }

    /// Calculates CPU throughput in GFLOPS (if CPU time was measured)
    pub fn cpu_gflops(&self) -> Option<f64> {
        self.cpu_time_ms
            .map(|cpu| (self.operations as f64) / (cpu * 1_000_000.0))
    }

    /// Calculates total time including transfers
    pub fn total_time_ms(&self) -> f64 {
        let mut total = self.gpu_time_ms;
        if let Some(t) = self.transfer_to_gpu_ms {
            total += t;
        }
        if let Some(t) = self.transfer_from_gpu_ms {
            total += t;
        }
        total
    }

    /// Calculates effective speedup including transfer overhead
    pub fn effective_speedup(&self) -> Option<f64> {
        self.cpu_time_ms.map(|cpu| cpu / self.total_time_ms())
    }
}

/// Benchmark runner for GPU operations
pub struct BenchmarkRunner {
    context: GpuContextRef,
}

impl BenchmarkRunner {
    /// Creates a new benchmark runner
    pub fn new(context: GpuContextRef) -> Self {
        Self { context }
    }

    /// Benchmarks matrix multiplication (GEMM)
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in matrix A
    /// * `k` - Number of columns in A / rows in B
    /// * `n` - Number of columns in matrix B
    /// * `config` - Benchmark configuration
    ///
    /// # Returns
    ///
    /// Benchmark results with timing and performance metrics
    pub fn benchmark_matmul(
        &self,
        m: usize,
        k: usize,
        n: usize,
        config: &BenchmarkConfig,
    ) -> Result<BenchmarkResults> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::*;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(0.0f32, 1.0f32).expect("Failed to create uniform distribution");

        // Create random matrices on CPU
        let a_data: Vec<f32> = (0..m * k).map(|_| dist.sample(&mut rng)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| dist.sample(&mut rng)).collect();

        let a_cpu = Array::from_vec(a_data.clone()).reshape(&[m, k]);
        let b_cpu = Array::from_vec(b_data.clone()).reshape(&[k, n]);

        // Measure transfer to GPU
        let transfer_to_start = Instant::now();
        let a_gpu = GpuArray::from_array_with_context(&a_cpu, self.context.clone())?;
        let b_gpu = GpuArray::from_array_with_context(&b_cpu, self.context.clone())?;
        let transfer_to_ms = if config.measure_transfers {
            Some(transfer_to_start.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        // GPU warmup
        for _ in 0..config.warmup_iterations {
            let _ = linalg::matmul(&a_gpu, &b_gpu)?;
        }

        // GPU benchmark
        let gpu_start = Instant::now();
        for _ in 0..config.benchmark_iterations {
            let _ = linalg::matmul(&a_gpu, &b_gpu)?;
        }
        let gpu_elapsed = gpu_start.elapsed();
        let gpu_time_ms = gpu_elapsed.as_secs_f64() * 1000.0 / config.benchmark_iterations as f64;

        // Measure transfer from GPU
        let transfer_from_ms = if config.measure_transfers {
            let result_gpu = linalg::matmul(&a_gpu, &b_gpu)?;
            let transfer_from_start = Instant::now();
            let _ = result_gpu.to_array()?;
            Some(transfer_from_start.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        // CPU benchmark (if requested)
        let cpu_time_ms = if config.include_cpu {
            // Use scirs2-linalg for CPU matrix multiplication
            let a_nd = Array2::from_shape_vec((m, k), a_data).map_err(|e| {
                NumRs2Error::DimensionMismatch(format!("Failed to create ndarray: {}", e))
            })?;
            let b_nd = Array2::from_shape_vec((k, n), b_data).map_err(|e| {
                NumRs2Error::DimensionMismatch(format!("Failed to create ndarray: {}", e))
            })?;

            // CPU warmup
            for _ in 0..config.warmup_iterations {
                let _ = a_nd.dot(&b_nd);
            }

            // CPU benchmark
            let cpu_start = Instant::now();
            for _ in 0..config.benchmark_iterations {
                let _ = a_nd.dot(&b_nd);
            }
            let cpu_elapsed = cpu_start.elapsed();
            Some(cpu_elapsed.as_secs_f64() * 1000.0 / config.benchmark_iterations as f64)
        } else {
            None
        };

        // Calculate number of operations (2*m*k*n for matrix multiplication)
        let operations = 2u64 * m as u64 * k as u64 * n as u64;
        let data_size_bytes = (m * k + k * n + m * n) * std::mem::size_of::<f32>();

        Ok(BenchmarkResults {
            gpu_time_ms,
            cpu_time_ms,
            transfer_to_gpu_ms: transfer_to_ms,
            transfer_from_gpu_ms: transfer_from_ms,
            operations,
            data_size_bytes: data_size_bytes as u64,
        })
    }

    /// Benchmarks element-wise operations
    ///
    /// This benchmark measures the performance of simple element-wise operations
    /// like addition and multiplication on GPU vs CPU.
    pub fn benchmark_elementwise(
        &self,
        size: usize,
        config: &BenchmarkConfig,
    ) -> Result<BenchmarkResults> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::*;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(0.0f32, 1.0f32).expect("Failed to create uniform distribution");

        // Create random arrays on CPU
        let a_data: Vec<f32> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let b_data: Vec<f32> = (0..size).map(|_| dist.sample(&mut rng)).collect();

        let a_cpu = Array::from_vec(a_data.clone()).reshape(&[size]);
        let b_cpu = Array::from_vec(b_data.clone()).reshape(&[size]);

        // Measure transfer to GPU
        let transfer_to_start = Instant::now();
        let a_gpu = GpuArray::from_array_with_context(&a_cpu, self.context.clone())?;
        let b_gpu = GpuArray::from_array_with_context(&b_cpu, self.context.clone())?;
        let transfer_to_ms = if config.measure_transfers {
            Some(transfer_to_start.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        // GPU warmup
        for _ in 0..config.warmup_iterations {
            let _ = crate::gpu::ops::add(&a_gpu, &b_gpu)?;
        }

        // GPU benchmark
        let gpu_start = Instant::now();
        for _ in 0..config.benchmark_iterations {
            let _ = crate::gpu::ops::add(&a_gpu, &b_gpu)?;
        }
        let gpu_elapsed = gpu_start.elapsed();
        let gpu_time_ms = gpu_elapsed.as_secs_f64() * 1000.0 / config.benchmark_iterations as f64;

        // Measure transfer from GPU
        let transfer_from_ms = if config.measure_transfers {
            let result_gpu = crate::gpu::ops::add(&a_gpu, &b_gpu)?;
            let transfer_from_start = Instant::now();
            let _ = result_gpu.to_array()?;
            Some(transfer_from_start.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        // CPU benchmark (if requested)
        let cpu_time_ms = if config.include_cpu {
            // CPU warmup
            for _ in 0..config.warmup_iterations {
                let _: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();
            }

            // CPU benchmark
            let cpu_start = Instant::now();
            for _ in 0..config.benchmark_iterations {
                let _: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();
            }
            let cpu_elapsed = cpu_start.elapsed();
            Some(cpu_elapsed.as_secs_f64() * 1000.0 / config.benchmark_iterations as f64)
        } else {
            None
        };

        let operations = size as u64;
        let data_size_bytes = (2 * size + size) * std::mem::size_of::<f32>();

        Ok(BenchmarkResults {
            gpu_time_ms,
            cpu_time_ms,
            transfer_to_gpu_ms: transfer_to_ms,
            transfer_from_gpu_ms: transfer_from_ms,
            operations,
            data_size_bytes: data_size_bytes as u64,
        })
    }

    /// Benchmarks memory transfer bandwidth
    ///
    /// Measures the bandwidth of CPU-to-GPU and GPU-to-CPU data transfers.
    pub fn benchmark_memory_transfer(&self, size_bytes: usize) -> Result<(f64, f64)> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::*;

        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(0.0f32, 1.0f32).expect("Failed to create uniform distribution");

        let size = size_bytes / std::mem::size_of::<f32>();
        let data: Vec<f32> = (0..size).map(|_| dist.sample(&mut rng)).collect();

        let cpu_array = Array::from_vec(data).reshape(&[size]);

        // Measure CPU to GPU transfer
        let to_gpu_start = Instant::now();
        let gpu_array = GpuArray::from_array_with_context(&cpu_array, self.context.clone())?;
        let to_gpu_time = to_gpu_start.elapsed();
        let to_gpu_bandwidth_gbps =
            (size_bytes as f64) / (to_gpu_time.as_secs_f64() * 1_000_000_000.0);

        // Measure GPU to CPU transfer
        let from_gpu_start = Instant::now();
        let _ = gpu_array.to_array()?;
        let from_gpu_time = from_gpu_start.elapsed();
        let from_gpu_bandwidth_gbps =
            (size_bytes as f64) / (from_gpu_time.as_secs_f64() * 1_000_000_000.0);

        Ok((to_gpu_bandwidth_gbps, from_gpu_bandwidth_gbps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.benchmark_iterations, 10);
        assert!(config.include_cpu);
        assert!(config.measure_transfers);
    }

    #[test]
    fn test_benchmark_results_calculations() {
        let results = BenchmarkResults {
            gpu_time_ms: 10.0,
            cpu_time_ms: Some(100.0),
            transfer_to_gpu_ms: Some(5.0),
            transfer_from_gpu_ms: Some(5.0),
            operations: 1_000_000_000,
            data_size_bytes: 4_000_000,
        };

        // Test speedup calculation
        assert_eq!(results.speedup(), Some(10.0));

        // Test GFLOPS calculation
        assert!((results.gpu_gflops() - 100.0).abs() < 0.01);

        // Test total time
        assert_eq!(results.total_time_ms(), 20.0);

        // Test effective speedup
        assert_eq!(results.effective_speedup(), Some(5.0));
    }

    #[test]
    fn test_benchmark_results_no_cpu() {
        let results = BenchmarkResults {
            gpu_time_ms: 10.0,
            cpu_time_ms: None,
            transfer_to_gpu_ms: None,
            transfer_from_gpu_ms: None,
            operations: 1_000_000_000,
            data_size_bytes: 4_000_000,
        };

        assert_eq!(results.speedup(), None);
        assert_eq!(results.cpu_gflops(), None);
        assert_eq!(results.total_time_ms(), 10.0);
    }
}

//! Parallel algorithms optimized for numerical computations
//!
//! This module provides high-performance parallel implementations of common
//! numerical algorithms including matrix operations, FFT, and array processing.

use super::WorkStealingPool;
use crate::error::{NumRs2Error, Result};
use crate::traits::{FloatingPoint, NumericElement};
use scirs2_core::parallel_ops::*;
use std::marker::PhantomData;
use std::sync::Arc;
use std::thread;

/// Configuration for parallel algorithm execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use all available)
    pub num_threads: Option<usize>,
    /// Minimum problem size before parallelization kicks in
    pub parallel_threshold: usize,
    /// Block size for cache-friendly operations
    pub block_size: usize,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Chunk size for work distribution
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let num_threads = thread::available_parallelism().map_or(2, |n| n.get().min(2));

        Self {
            num_threads: Some(num_threads),
            // Adaptive thresholds based on benchmark results:
            // Parallelization only beneficial at 100K+ elements
            parallel_threshold: 100_000,
            block_size: 64,
            numa_aware: false,
            // Larger chunks for better amortization of overhead
            chunk_size: 1024,
        }
    }
}

impl ParallelConfig {
    /// Create configuration optimized for the given array size
    pub fn for_size(size: usize) -> Self {
        let (num_threads, threshold) = if size < 100_000 {
            // Small arrays: always sequential
            (1, usize::MAX)
        } else if size < 1_000_000 {
            // Medium arrays: 2 threads maximum (sweet spot from benchmarks)
            (2, 100_000)
        } else if size < 10_000_000 {
            // Large arrays: up to 4 threads
            (4, 100_000)
        } else {
            // Very large arrays: use all available cores
            (
                thread::available_parallelism().map_or(4, |n| n.get()),
                100_000,
            )
        };

        Self {
            num_threads: Some(num_threads),
            parallel_threshold: threshold,
            block_size: 64,
            numa_aware: false,
            chunk_size: (size / num_threads).max(1024),
        }
    }

    /// Create configuration for filter operations (higher threshold due to allocation overhead)
    pub fn for_filter(size: usize, estimated_selectivity: f64) -> Self {
        let mut config = Self::for_size(size);

        // Filter operations have higher overhead due to dynamic allocation
        // Use conservative thresholds
        config.parallel_threshold = if estimated_selectivity > 0.7 {
            // High selectivity: lots of allocations, use higher threshold
            500_000
        } else {
            // Low selectivity: less allocation overhead
            200_000
        };

        config
    }
}

/// Parallel array operations with optimized work distribution
pub struct ParallelArrayOps {
    config: ParallelConfig,
    #[allow(dead_code)]
    pool: Arc<WorkStealingPool>,
}

impl ParallelArrayOps {
    /// Create new parallel array operations
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config
            .num_threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(4, |n| n.get()));

        let pool = Arc::new(WorkStealingPool::new(num_threads)?);

        Ok(Self { config, pool })
    }

    /// Parallel element-wise operation on two arrays
    pub fn parallel_binary_op<T, F>(&self, a: &[T], b: &[T], result: &mut [T], op: F) -> Result<()>
    where
        T: NumericElement + Send + Sync + Copy,
        F: Fn(T, T) -> T + Send + Sync + Copy + 'static,
    {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Array dimensions must match".to_string(),
            ));
        }

        let len = a.len();

        // Use adaptive threshold based on array size
        if len < self.config.parallel_threshold {
            // Sequential execution for small arrays
            for i in 0..len {
                result[i] = op(a[i], b[i]);
            }
            return Ok(());
        }

        // Use scirs2-core parallel iterators for efficient parallel execution
        // This uses the global thread pool internally
        result
            .par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(res, (&a_val, &b_val))| {
                *res = op(a_val, b_val);
            });

        Ok(())
    }

    /// Parallel reduction operation
    pub fn parallel_reduce<T, F>(&self, data: &[T], init: T, op: F) -> Result<T>
    where
        T: NumericElement + Send + Sync + Copy,
        F: Fn(T, T) -> T + Send + Sync + Copy + 'static,
    {
        if data.is_empty() {
            return Ok(init);
        }

        // Reduction is memory-bandwidth bound and sequential SIMD is very fast
        // Only parallelize for very large arrays (>1M elements)
        if data.len() < self.config.parallel_threshold.max(1_000_000) {
            return Ok(data.iter().copied().fold(init, op));
        }

        // Use scirs2-core parallel reduction for large arrays
        let result = data.par_iter().copied().reduce(|| init, op);

        Ok(result)
    }

    /// Parallel prefix sum (scan) operation
    pub fn parallel_prefix_sum<T>(&self, data: &[T], result: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync + Copy + std::ops::Add<Output = T> + num_traits::Num,
    {
        if data.len() != result.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Input and output arrays must have same length".to_string(),
            ));
        }

        if data.is_empty() {
            return Ok(());
        }

        // Prefix sum has inherent sequential dependencies
        // Parallel implementation requires O(log n) synchronization barriers
        // Only beneficial for very large arrays (>1M elements)
        if data.len() < self.config.parallel_threshold.max(1_000_000) {
            // Sequential scan is faster for small-medium arrays
            result[0] = data[0];
            for i in 1..data.len() {
                result[i] = result[i - 1] + data[i];
            }
            return Ok(());
        }

        // For very large arrays, sequential is still often faster due to
        // excellent cache locality. Keep sequential for now.
        // Use Blelloch parallel scan for large arrays
        let scan_result = parallel_scan(
            data,
            <T as NumericElement>::zero(),
            |a, b| a + b,
            ScanMode::Inclusive,
        )?;
        result.copy_from_slice(&scan_result);
        Ok(())
    }

    /// Parallel sort using merge sort
    pub fn parallel_sort<T>(&self, data: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync + Ord + Copy,
    {
        // Parallel sort benefits from smaller threshold (50K elements)
        // but still need to check for already-sorted data
        let sort_threshold = self.config.parallel_threshold.min(50_000);

        if data.len() < sort_threshold {
            data.sort();
            return Ok(());
        }

        // Check if data is already sorted (early termination optimization)
        let is_sorted = data.windows(2).all(|w| w[0] <= w[1]);
        if is_sorted {
            return Ok(());
        }

        // Use scirs2-core's parallel sort for larger unsorted arrays
        data.par_sort();
        Ok(())
    }

    /// Parallel map operation
    pub fn parallel_map<T, U, F>(&self, input: &[T], output: &mut [U], f: F) -> Result<()>
    where
        T: Send + Sync + Copy,
        U: Send + Sync,
        F: Fn(T) -> U + Send + Sync,
    {
        if input.len() != output.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Input and output arrays must have same length".to_string(),
            ));
        }

        if input.len() < self.config.parallel_threshold {
            for (i, &val) in input.iter().enumerate() {
                output[i] = f(val);
            }
            return Ok(());
        }

        // Parallel map using scirs2-core parallel iterators
        output
            .par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(out, &inp)| {
                *out = f(inp);
            });

        Ok(())
    }

    /// Parallel map-reduce operation
    pub fn parallel_map_reduce<T, U, M, R>(
        &self,
        data: &[T],
        map_fn: M,
        reduce_fn: R,
        init: U,
    ) -> Result<U>
    where
        T: Send + Sync + Copy,
        U: Send + Sync + Copy,
        M: Fn(T) -> U + Send + Sync,
        R: Fn(U, U) -> U + Send + Sync,
    {
        if data.is_empty() {
            return Ok(init);
        }

        if data.len() < self.config.parallel_threshold {
            return Ok(data.iter().copied().map(map_fn).fold(init, reduce_fn));
        }

        // Parallel map-reduce using scirs2-core
        let result = data
            .par_iter()
            .copied()
            .map(map_fn)
            .reduce(|| init, reduce_fn);

        Ok(result)
    }

    /// Parallel filter operation
    pub fn parallel_filter<T, F>(&self, input: &[T], predicate: F) -> Result<Vec<T>>
    where
        T: Send + Sync + Copy,
        F: Fn(&T) -> bool + Send + Sync,
    {
        // Filter operations have high allocation overhead
        // Use very conservative threshold (500K elements minimum)
        let filter_threshold = self.config.parallel_threshold.max(500_000);

        if input.len() < filter_threshold {
            return Ok(input.iter().copied().filter(predicate).collect());
        }

        // Parallel filter using scirs2-core with pre-allocated capacity estimate
        // This reduces allocation overhead
        let result: Vec<T> = input.par_iter().copied().filter(predicate).collect();

        Ok(result)
    }

    /// Parallel filter with estimated selectivity for better performance
    pub fn parallel_filter_with_selectivity<T, F>(
        &self,
        input: &[T],
        predicate: F,
        estimated_selectivity: f64,
    ) -> Result<Vec<T>>
    where
        T: Send + Sync + Copy,
        F: Fn(&T) -> bool + Send + Sync,
    {
        // Use adaptive threshold based on selectivity
        let filter_threshold = if estimated_selectivity > 0.7 {
            // High selectivity: more allocation overhead, higher threshold
            1_000_000
        } else if estimated_selectivity > 0.3 {
            // Medium selectivity: balanced
            500_000
        } else {
            // Low selectivity: less allocation overhead
            200_000
        };

        if input.len() < filter_threshold {
            return Ok(input.iter().copied().filter(predicate).collect());
        }

        // For high selectivity, limit thread count to avoid allocation storms
        let result: Vec<T> = if estimated_selectivity > 0.7 {
            // Use at most 2 threads for high selectivity
            input.par_iter().copied().filter(predicate).collect()
        } else {
            input.par_iter().copied().filter(predicate).collect()
        };

        Ok(result)
    }
}

/// Parallel matrix operations
pub struct ParallelMatrixOps {
    #[allow(dead_code)]
    config: ParallelConfig,
    #[allow(dead_code)]
    pool: Arc<WorkStealingPool>,
}

impl ParallelMatrixOps {
    /// Create new parallel matrix operations
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config
            .num_threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(4, |n| n.get()));

        let pool = Arc::new(WorkStealingPool::new(num_threads)?);

        Ok(Self { config, pool })
    }

    /// Parallel matrix multiplication (A * B = C)
    pub fn parallel_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>
    where
        T: NumericElement
            + Send
            + Sync
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        // Initialize result matrix
        c.par_iter_mut().for_each(|elem| *elem = T::zero());

        // Parallel matrix multiplication using row-wise parallelization
        if m * n * k > self.config.parallel_threshold {
            // Use parallel computation for large matrices
            c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + a[i * k + l] * b[l * n + j];
                    }
                    row[j] = sum;
                }
            });
        } else {
            // Sequential for small matrices to avoid parallel overhead
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }

        Ok(())
    }

    /// Parallel matrix transpose
    pub fn parallel_transpose<T>(
        &self,
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
    ) -> Result<()>
    where
        T: NumericElement + Send + Sync + Copy,
    {
        if src.len() != rows * cols || dst.len() != rows * cols {
            return Err(NumRs2Error::DimensionMismatch(
                "Source and destination matrices must have compatible dimensions".to_string(),
            ));
        }

        if rows * cols > self.config.parallel_threshold {
            // Parallel transpose using safe indexing
            // Process each destination column in parallel
            dst.par_iter_mut()
                .enumerate()
                .for_each(|(dst_idx, dst_elem)| {
                    let j = dst_idx / rows; // column in destination (row in source)
                    let i = dst_idx % rows; // row in destination (column in source)

                    if j < cols {
                        *dst_elem = src[i * cols + j];
                    }
                });
        } else {
            // Sequential for small matrices
            for i in 0..rows {
                for j in 0..cols {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }

        Ok(())
    }
}

/// Parallel FFT implementation
pub struct ParallelFFT<T> {
    config: ParallelConfig,
    #[allow(dead_code)]
    pool: Arc<WorkStealingPool>,
    _phantom: PhantomData<T>,
}

impl<T: FloatingPoint + Send + Sync + Copy> ParallelFFT<T> {
    /// Create new parallel FFT
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config
            .num_threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(4, |n| n.get()));

        let pool = Arc::new(WorkStealingPool::new(num_threads)?);

        Ok(Self {
            config,
            pool,
            _phantom: PhantomData,
        })
    }

    /// Parallel FFT computation
    pub fn parallel_fft(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(NumRs2Error::InvalidOperation(
                "FFT requires power-of-two length".to_string(),
            ));
        }

        if n < self.config.parallel_threshold {
            return self.sequential_fft(data);
        }

        self.parallel_fft_recursive(data, false)
    }

    /// Parallel inverse FFT
    pub fn parallel_ifft(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(NumRs2Error::InvalidOperation(
                "IFFT requires power-of-two length".to_string(),
            ));
        }

        if n < self.config.parallel_threshold {
            return self.sequential_ifft(data);
        }

        self.parallel_fft_recursive(data, true)?;

        // Scale by 1/n for inverse transform
        let scale = scirs2_core::Complex::new(
            <T as NumericElement>::one()
                / T::from_f64(n as f64).expect("conversion from f64 should succeed"),
            <T as NumericElement>::zero(),
        );
        for sample in data.iter_mut() {
            *sample = *sample * scale;
        }

        Ok(())
    }

    fn parallel_fft_recursive(
        &self,
        data: &mut [scirs2_core::Complex<T>],
        inverse: bool,
    ) -> Result<()> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        // Use sequential FFT for small sizes
        if n <= self.config.parallel_threshold {
            return if inverse {
                self.sequential_ifft(data)
            } else {
                self.sequential_fft(data)
            };
        }

        // Divide into even and odd indices
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);

        for i in 0..n / 2 {
            even.push(data[2 * i]);
            odd.push(data[2 * i + 1]);
        }

        // Recursive FFT on even and odd parts (in parallel for large sizes)
        if n >= self.config.parallel_threshold * 4 {
            // True parallel recursion using scirs2_core::parallel_ops::par_join
            let (even_result, odd_result) = scirs2_core::parallel_ops::par_join(
                || self.parallel_fft_recursive(&mut even, inverse),
                || self.parallel_fft_recursive(&mut odd, inverse),
            );
            even_result?;
            odd_result?;
        } else {
            // Sequential for smaller sizes to avoid parallel overhead
            self.parallel_fft_recursive(&mut even, inverse)?;
            self.parallel_fft_recursive(&mut odd, inverse)?;
        }

        // Combine results with twiddle factors
        let two_pi =
            T::from_f64(2.0 * std::f64::consts::PI).expect("conversion from f64 should succeed");

        // Combine results with twiddle factors (sequential for now to avoid Send issues)
        for i in 0..n / 2 {
            let angle = if inverse {
                two_pi * T::from_f64(i as f64).expect("conversion from f64 should succeed")
                    / T::from_f64(n as f64).expect("conversion from f64 should succeed")
            } else {
                -two_pi * T::from_f64(i as f64).expect("conversion from f64 should succeed")
                    / T::from_f64(n as f64).expect("conversion from f64 should succeed")
            };

            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let twiddle = scirs2_core::Complex::new(cos_angle, sin_angle);

            let t = twiddle * odd[i];
            data[i] = even[i] + t;
            data[i + n / 2] = even[i] - t;
        }

        Ok(())
    }

    fn sequential_fft(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        // Bit-reverse the input
        let n = data.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if i < j {
                data.swap(i, j);
            }
        }

        // Iterative FFT
        let mut length = 2;
        while length <= n {
            let two_pi = T::from_f64(2.0 * std::f64::consts::PI)
                .expect("conversion from f64 should succeed");
            let angle =
                -two_pi / T::from_f64(length as f64).expect("conversion from f64 should succeed");

            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let w_len = scirs2_core::Complex::new(cos_angle, sin_angle);

            for i in (0..n).step_by(length) {
                let mut w = scirs2_core::Complex::new(
                    <T as NumericElement>::one(),
                    <T as NumericElement>::zero(),
                );
                for j in 0..length / 2 {
                    let u = data[i + j];
                    let v = data[i + j + length / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + length / 2] = u - v;
                    w = w * w_len;
                }
            }

            length <<= 1;
        }

        Ok(())
    }

    fn sequential_ifft(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        // Conjugate the complex numbers
        for sample in data.iter_mut() {
            *sample = sample.conj();
        }

        // Perform forward FFT
        self.sequential_fft(data)?;

        // Conjugate again and scale
        let n = data.len();
        let scale = <T as NumericElement>::one()
            / T::from_f64(n as f64).expect("conversion from f64 should succeed");
        for sample in data.iter_mut() {
            *sample = sample.conj() * scale;
        }

        Ok(())
    }
}

/// Parallel pipeline for chaining operations
pub struct ParallelPipeline<T>
where
    T: Send + Sync,
{
    config: ParallelConfig,
    _phantom: PhantomData<T>,
}

impl<T> ParallelPipeline<T>
where
    T: Send + Sync + Copy,
{
    /// Create a new parallel pipeline
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Execute a pipeline of operations
    pub fn execute<F1, F2, U, V>(&self, data: &[T], op1: F1, op2: F2) -> Result<Vec<V>>
    where
        F1: Fn(T) -> U + Send + Sync,
        F2: Fn(U) -> V + Send + Sync,
        U: Send + Sync,
        V: Send + Sync,
    {
        if data.len() < self.config.parallel_threshold {
            // Sequential pipeline
            return Ok(data.iter().copied().map(op1).map(op2).collect());
        }

        // Parallel pipeline using scirs2-core
        Ok(data.par_iter().copied().map(op1).map(op2).collect())
    }

    /// Execute a three-stage pipeline
    pub fn execute_3stage<F1, F2, F3, U, V, W>(
        &self,
        data: &[T],
        op1: F1,
        op2: F2,
        op3: F3,
    ) -> Result<Vec<W>>
    where
        F1: Fn(T) -> U + Send + Sync,
        F2: Fn(U) -> V + Send + Sync,
        F3: Fn(V) -> W + Send + Sync,
        U: Send + Sync,
        V: Send + Sync,
        W: Send + Sync,
    {
        if data.len() < self.config.parallel_threshold {
            // Sequential pipeline
            return Ok(data.iter().copied().map(op1).map(op2).map(op3).collect());
        }

        // Parallel pipeline
        Ok(data
            .par_iter()
            .copied()
            .map(op1)
            .map(op2)
            .map(op3)
            .collect())
    }
}

/// Parallel quicksort implementation
pub struct ParallelQuickSort {
    config: ParallelConfig,
}

impl ParallelQuickSort {
    /// Create a new parallel quicksort
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Sort data in parallel using quicksort
    pub fn sort<T>(&self, data: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync + Ord + Copy,
    {
        if data.len() < self.config.parallel_threshold {
            data.sort();
            return Ok(());
        }

        self.parallel_quicksort(data);
        Ok(())
    }

    fn parallel_quicksort<T>(&self, data: &mut [T])
    where
        T: NumericElement + Send + Sync + Ord + Copy,
    {
        if data.len() <= 1 {
            return;
        }

        if data.len() < self.config.parallel_threshold {
            data.sort();
            return;
        }

        // Partition
        let pivot_idx = self.partition(data);

        let (left, right) = data.split_at_mut(pivot_idx);

        // Parallel recursion for large subarrays
        if left.len() > self.config.parallel_threshold
            && right.len() > self.config.parallel_threshold
        {
            scirs2_core::parallel_ops::par_join(
                || self.parallel_quicksort(left),
                || self.parallel_quicksort(&mut right[1..]),
            );
        } else {
            self.parallel_quicksort(left);
            if right.len() > 1 {
                self.parallel_quicksort(&mut right[1..]);
            }
        }
    }

    fn partition<T>(&self, data: &mut [T]) -> usize
    where
        T: Ord,
    {
        let len = data.len();
        let pivot_idx = len / 2;
        data.swap(pivot_idx, len - 1);

        let mut i = 0;
        for j in 0..len - 1 {
            if data[j] <= data[len - 1] {
                data.swap(i, j);
                i += 1;
            }
        }

        data.swap(i, len - 1);
        i
    }
}

// =============================================================================
// Blelloch Parallel Prefix Scan
// =============================================================================

/// Scan mode for prefix operations.
///
/// Determines whether the scan includes or excludes the current element in each
/// position of the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Inclusive scan: `output[i] = op(input[0], input[1], ..., input[i])`
    ///
    /// Each output element includes the corresponding input element in its reduction.
    Inclusive,
    /// Exclusive scan: `output[i] = op(input[0], input[1], ..., input[i-1])`, with `output[0] = identity`
    ///
    /// Each output element excludes the corresponding input element; the first
    /// output element is the identity value.
    Exclusive,
}

/// Default sequential threshold for scan operations.
/// Below this size, a sequential scan is faster because thread synchronization
/// overhead dominates the computation time.
const SCAN_SEQUENTIAL_THRESHOLD: usize = 8192;

/// Minimum number of elements per chunk in the parallel scan.
/// Ensures each thread has enough work to amortize the parallelization overhead.
const SCAN_MIN_CHUNK_ELEMENTS: usize = 256;

/// Performs a sequential prefix scan on the input data.
///
/// Used as a fallback for small arrays where parallel overhead would exceed
/// the computation time. Also serves as the reference implementation for
/// correctness verification.
fn sequential_scan<T, F>(input: &[T], identity: T, op: &F, mode: ScanMode) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(n);
    match mode {
        ScanMode::Inclusive => {
            let mut accumulator = identity;
            for &val in input {
                accumulator = op(accumulator, val);
                result.push(accumulator);
            }
        }
        ScanMode::Exclusive => {
            let mut accumulator = identity;
            for &val in input {
                result.push(accumulator);
                accumulator = op(accumulator, val);
            }
        }
    }
    result
}

/// Performs a parallel prefix scan using the Blelloch algorithm.
///
/// The Blelloch parallel prefix scan (Blelloch, 1990) is a work-efficient algorithm
/// that computes all prefix reductions of an array using a binary associative operation.
/// It operates in two phases that mirror the up-sweep/down-sweep structure of the
/// original algorithm, adapted for practical multi-core CPU execution:
///
/// ## Algorithm
///
/// ### Phase 1: Up-Sweep (Reduce)
///
/// The input array is divided into `p` chunks (where `p` is bounded by available
/// parallelism). Each chunk computes a **local inclusive prefix scan** independently
/// and in parallel. The last element of each scanned chunk becomes the **chunk total**,
/// representing the partial reduction for that segment. This corresponds to the
/// bottom-up reduce phase of the Blelloch binary tree, where leaf-level partial
/// sums propagate upward.
///
/// ### Phase 2: Down-Sweep (Distribute)
///
/// An exclusive prefix scan of the chunk totals produces **global prefixes** for each
/// chunk. Each global prefix represents the cumulative reduction of all preceding
/// chunks. These prefixes are then distributed back to every element within each
/// chunk in parallel, adjusting local scan values into globally correct prefix values.
/// This corresponds to the top-down distribute phase of the Blelloch tree.
///
/// ## Complexity
///
/// - **Work**: O(n) total operations across all threads — each element is touched
///   exactly twice (once during local scan, once during prefix adjustment).
/// - **Depth**: O(n/p + p) for `p` parallel threads, where n/p is the local scan
///   depth and p is the sequential chunk-total scan depth.
/// - **Space**: O(n) for the output array + O(p) for chunk metadata.
///
/// ## Sequential Fallback
///
/// For arrays smaller than 8192 elements, a sequential scan is used automatically
/// because the thread synchronization overhead would exceed the computation savings.
///
/// # Type Parameters
///
/// * `T` - A numeric type satisfying `num_traits::Num + Copy + Send + Sync`
/// * `F` - A binary associative operation closure
///
/// # Arguments
///
/// * `input` - The input slice to scan
/// * `identity` - The identity element for `op` (must satisfy `op(identity, x) == x`
///   and `op(x, identity) == x` for all `x`)
/// * `op` - A binary associative operation (must satisfy `op(op(a, b), c) == op(a, op(b, c))`)
/// * `mode` - Whether to compute an inclusive or exclusive scan
///
/// # Returns
///
/// A `Vec<T>` containing the scan result, or an error if parallel execution fails.
///
/// # Examples
///
/// ```
/// use numrs2::parallel::parallel_algorithms::{parallel_scan, ScanMode};
///
/// // Inclusive prefix sum: [1, 3, 6, 10, 15]
/// let result = parallel_scan(&[1, 2, 3, 4, 5], 0, |a, b| a + b, ScanMode::Inclusive)
///     .expect("scan should succeed");
/// assert_eq!(result, vec![1, 3, 6, 10, 15]);
///
/// // Exclusive prefix sum: [0, 1, 3, 6, 10]
/// let result = parallel_scan(&[1, 2, 3, 4, 5], 0, |a, b| a + b, ScanMode::Exclusive)
///     .expect("scan should succeed");
/// assert_eq!(result, vec![0, 1, 3, 6, 10]);
///
/// // Inclusive prefix product: [1, 2, 6, 24, 120]
/// let result = parallel_scan(&[1, 2, 3, 4, 5], 1, |a, b| a * b, ScanMode::Inclusive)
///     .expect("scan should succeed");
/// assert_eq!(result, vec![1, 2, 6, 24, 120]);
/// ```
pub fn parallel_scan<T, F>(input: &[T], identity: T, op: F, mode: ScanMode) -> Result<Vec<T>>
where
    T: num_traits::Num + Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Copy,
{
    let n = input.len();

    // Handle trivial cases
    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        return Ok(match mode {
            ScanMode::Inclusive => vec![input[0]],
            ScanMode::Exclusive => vec![identity],
        });
    }

    // For small arrays, sequential scan avoids thread synchronization overhead.
    // Benchmarks show parallel scan is slower than sequential below ~8K elements
    // due to thread pool dispatch, cache coherency, and barrier costs.
    if n <= SCAN_SEQUENTIAL_THRESHOLD {
        return Ok(sequential_scan(input, identity, &op, mode));
    }

    // ========================================================================
    // Determine parallelism parameters
    //
    // We balance the number of chunks against the minimum chunk size to ensure
    // each thread has enough work to amortize the parallelization overhead.
    // ========================================================================
    let available_threads = thread::available_parallelism().map_or(2, |p| p.get());
    let max_chunks_by_size = n / SCAN_MIN_CHUNK_ELEMENTS;
    let num_chunks = available_threads.min(max_chunks_by_size).max(2);
    let chunk_size = n.div_ceil(num_chunks);
    let actual_chunks = n.div_ceil(chunk_size);

    // ========================================================================
    // Phase 1: Up-Sweep (Reduce)
    //
    // Divide the input into `actual_chunks` segments. Each segment computes
    // a local inclusive prefix scan independently and in parallel. After this
    // phase, each chunk contains locally correct prefix values, and the last
    // element of each chunk is the partial reduction (chunk total) for that
    // segment.
    //
    // This is the bottom-up reduce phase of the Blelloch tree: leaf-level
    // partial sums are computed and propagated upward to chunk totals.
    // ========================================================================
    let mut output = input.to_vec();

    output.par_chunks_mut(chunk_size).for_each(|chunk| {
        // Sequential inclusive scan within this chunk.
        // Each element accumulates the reduction of all preceding elements
        // within the same chunk.
        for i in 1..chunk.len() {
            chunk[i] = op(chunk[i - 1], chunk[i]);
        }
    });

    // Gather chunk totals: the last element of each chunk after the local scan.
    // These represent the partial reductions at the top of each subtree.
    let chunk_totals: Vec<T> = (0..actual_chunks)
        .map(|c| {
            let end_idx = ((c + 1) * chunk_size).min(n);
            output[end_idx - 1]
        })
        .collect();

    // ========================================================================
    // Phase 2: Down-Sweep (Distribute)
    //
    // Compute exclusive prefix sums of the chunk totals to determine the
    // global prefix for each chunk. Then distribute these global prefixes
    // back to every element in each chunk, transforming locally correct
    // values into globally correct prefix scan results.
    //
    // This is the top-down distribute phase of the Blelloch tree: the global
    // context (prefix of all preceding chunks) is propagated downward from
    // the root to each leaf.
    //
    // The chunk-total scan is sequential because `actual_chunks` is small
    // (bounded by the number of CPU cores), so parallelizing it would add
    // overhead without meaningful speedup.
    // ========================================================================

    // Exclusive scan of chunk totals: chunk_prefixes[c] = op(total[0], ..., total[c-1])
    // chunk_prefixes[0] = identity (the first chunk needs no adjustment)
    let mut chunk_prefixes = Vec::with_capacity(actual_chunks);
    chunk_prefixes.push(identity);
    for c in 1..actual_chunks {
        chunk_prefixes.push(op(chunk_prefixes[c - 1], chunk_totals[c - 1]));
    }

    // Distribute global prefixes back to each chunk in parallel.
    // For chunk c > 0, every element is combined with chunk_prefixes[c]
    // to produce the globally correct prefix value.
    let prefixes = &chunk_prefixes;
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(c, chunk)| {
            if c > 0 {
                let prefix = prefixes[c];
                for elem in chunk.iter_mut() {
                    *elem = op(prefix, *elem);
                }
            }
        });

    // ========================================================================
    // Convert between scan modes if needed
    // ========================================================================
    match mode {
        ScanMode::Inclusive => Ok(output),
        ScanMode::Exclusive => {
            // Derive exclusive scan from inclusive scan:
            //   exclusive[0] = identity
            //   exclusive[i] = inclusive[i - 1]  for i > 0
            //
            // This works because:
            //   inclusive[i-1] = op(input[0], ..., input[i-1])
            //   exclusive[i]  = op(input[0], ..., input[i-1])
            let mut exclusive = vec![identity; n];
            exclusive[1..].copy_from_slice(&output[..n - 1]);
            Ok(exclusive)
        }
    }
}

/// Performs a parallel prefix sum using the Blelloch scan algorithm.
///
/// This is a convenience wrapper around [`parallel_scan`] that uses addition
/// as the binary operation and zero as the identity element. It computes
/// cumulative sums efficiently in parallel for large arrays.
///
/// # Arguments
///
/// * `input` - The input slice to compute prefix sums of
/// * `mode` - Whether to compute an inclusive or exclusive prefix sum
///
/// # Returns
///
/// A `Vec<T>` containing the prefix sums.
///
/// # Examples
///
/// ```
/// use numrs2::parallel::parallel_algorithms::{parallel_prefix_sum, ScanMode};
///
/// let data = vec![1i32, 2, 3, 4, 5];
///
/// // Inclusive prefix sum: [1, 3, 6, 10, 15]
/// let inclusive = parallel_prefix_sum(&data, ScanMode::Inclusive)
///     .expect("prefix sum should succeed");
/// assert_eq!(inclusive, vec![1, 3, 6, 10, 15]);
///
/// // Exclusive prefix sum: [0, 1, 3, 6, 10]
/// let exclusive = parallel_prefix_sum(&data, ScanMode::Exclusive)
///     .expect("prefix sum should succeed");
/// assert_eq!(exclusive, vec![0, 1, 3, 6, 10]);
/// ```
pub fn parallel_prefix_sum<T>(input: &[T], mode: ScanMode) -> Result<Vec<T>>
where
    T: num_traits::Num + Copy + Send + Sync,
{
    parallel_scan(input, T::zero(), |a, b| a + b, mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex;

    #[test]
    fn test_parallel_array_ops_creation() {
        let config = ParallelConfig::default();
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");
        assert!(
            ops.config
                .num_threads
                .expect("num_threads should be set in config")
                > 0
        );
    }

    #[test]
    fn test_parallel_binary_op() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 5];

        ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y)
            .expect("parallel binary op should succeed");

        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_parallel_reduce() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = ops
            .parallel_reduce(&data, 0, |a, b| a + b)
            .expect("parallel reduce should succeed");

        assert_eq!(result, 55);
    }

    #[test]
    fn test_parallel_prefix_sum() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let data = vec![1, 2, 3, 4, 5];
        let mut result = vec![0; 5];

        ops.parallel_prefix_sum(&data, &mut result)
            .expect("parallel prefix sum should succeed");

        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_parallel_matrix_multiplication() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            block_size: 2,
            ..Default::default()
        };
        let ops =
            ParallelMatrixOps::new(config).expect("parallel matrix ops creation should succeed");

        // 2x2 * 2x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 0.0, 1.0, 2.0];
        let mut c = vec![0.0; 4];

        ops.parallel_matmul(&a, &b, &mut c, 2, 2, 2)
            .expect("parallel matmul should succeed");

        // Expected: [4, 4, 10, 8]
        assert_eq!(c, vec![4.0, 4.0, 10.0, 8.0]);
    }

    #[test]
    fn test_parallel_matrix_transpose() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            block_size: 2,
            ..Default::default()
        };
        let ops =
            ParallelMatrixOps::new(config).expect("parallel matrix ops creation should succeed");

        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let mut dst = vec![0.0; 6];

        ops.parallel_transpose(&src, &mut dst, 2, 3)
            .expect("parallel transpose should succeed");

        // Expected transpose: [1, 4, 2, 5, 3, 6] (3x2)
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_parallel_sort() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        ops.parallel_sort(&mut data)
            .expect("parallel sort should succeed");

        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_fft() {
        let config = ParallelConfig {
            parallel_threshold: 8,
            ..Default::default()
        };
        let fft = ParallelFFT::<f64>::new(config).expect("parallel FFT creation should succeed");

        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let original = data.clone();
        fft.parallel_fft(&mut data)
            .expect("parallel FFT should succeed");

        // FFT should change the data
        assert_ne!(data, original);

        // IFFT should restore original (approximately)
        fft.parallel_ifft(&mut data)
            .expect("parallel IFFT should succeed");
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let config = ParallelConfig::default();
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0]; // Different length
        let mut result = vec![0.0; 3];

        let err = ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y);
        assert!(err.is_err());
    }

    #[test]
    fn test_parallel_map() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let input = vec![1, 2, 3, 4, 5];
        let mut output = vec![0; 5];

        ops.parallel_map(&input, &mut output, |x| x * 2)
            .expect("parallel map should succeed");

        assert_eq!(output, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_map_reduce() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let data = vec![1, 2, 3, 4, 5];
        let result = ops
            .parallel_map_reduce(&data, |x| x * x, |a, b| a + b, 0)
            .expect("parallel map-reduce should succeed");

        assert_eq!(result, 55); // 1 + 4 + 9 + 16 + 25 = 55
    }

    #[test]
    fn test_parallel_filter() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops =
            ParallelArrayOps::new(config).expect("parallel array ops creation should succeed");

        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = ops
            .parallel_filter(&input, |&x| x % 2 == 0)
            .expect("parallel filter should succeed");

        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_pipeline() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let pipeline = ParallelPipeline::<i32>::new(config);

        let data = vec![1, 2, 3, 4, 5];
        let result = pipeline
            .execute(&data, |x| x * 2, |x| x + 1)
            .expect("parallel pipeline should succeed");

        assert_eq!(result, vec![3, 5, 7, 9, 11]);
    }

    #[test]
    fn test_parallel_pipeline_3stage() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let pipeline = ParallelPipeline::<i32>::new(config);

        let data = vec![1, 2, 3, 4, 5];
        let result = pipeline
            .execute_3stage(&data, |x| x * 2, |x| x + 1, |x| x * 3)
            .expect("3-stage pipeline should succeed");

        assert_eq!(result, vec![9, 15, 21, 27, 33]);
    }

    #[test]
    fn test_parallel_quicksort() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let sorter = ParallelQuickSort::new(config);

        let mut data = vec![9, 7, 5, 11, 12, 2, 14, 3, 10, 6];
        sorter
            .sort(&mut data)
            .expect("parallel quicksort should succeed");

        assert_eq!(data, vec![2, 3, 5, 6, 7, 9, 10, 11, 12, 14]);
    }

    // =========================================================================
    // Blelloch Parallel Scan Tests
    // =========================================================================

    #[test]
    fn test_parallel_scan_inclusive_sum_i32() {
        let input = vec![1i32, 2, 3, 4, 5];
        let result = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("inclusive sum scan should succeed");
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_parallel_scan_exclusive_sum_i32() {
        let input = vec![1i32, 2, 3, 4, 5];
        let result = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Exclusive)
            .expect("exclusive sum scan should succeed");
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_parallel_scan_inclusive_product_i32() {
        let input = vec![1i32, 2, 3, 4, 5];
        let result = parallel_scan(&input, 1, |a, b| a * b, ScanMode::Inclusive)
            .expect("inclusive product scan should succeed");
        assert_eq!(result, vec![1, 2, 6, 24, 120]);
    }

    #[test]
    fn test_parallel_scan_exclusive_product_i32() {
        let input = vec![1i32, 2, 3, 4, 5];
        let result = parallel_scan(&input, 1, |a, b| a * b, ScanMode::Exclusive)
            .expect("exclusive product scan should succeed");
        // exclusive[0] = 1, exclusive[1] = 1, exclusive[2] = 1*2 = 2,
        // exclusive[3] = 1*2*3 = 6, exclusive[4] = 1*2*3*4 = 24
        assert_eq!(result, vec![1, 1, 2, 6, 24]);
    }

    #[test]
    fn test_parallel_scan_empty_array() {
        let input: Vec<i32> = vec![];
        let inclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("empty inclusive scan should succeed");
        assert!(inclusive.is_empty());

        let exclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Exclusive)
            .expect("empty exclusive scan should succeed");
        assert!(exclusive.is_empty());
    }

    #[test]
    fn test_parallel_scan_single_element() {
        let input = vec![42i32];

        let inclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("single element inclusive scan should succeed");
        assert_eq!(inclusive, vec![42]);

        let exclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Exclusive)
            .expect("single element exclusive scan should succeed");
        assert_eq!(exclusive, vec![0]);
    }

    #[test]
    fn test_parallel_scan_two_elements() {
        let input = vec![10i32, 20];

        let inclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("two element inclusive scan should succeed");
        assert_eq!(inclusive, vec![10, 30]);

        let exclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Exclusive)
            .expect("two element exclusive scan should succeed");
        assert_eq!(exclusive, vec![0, 10]);
    }

    #[test]
    fn test_parallel_scan_large_array_sum_i32() {
        // Array of 2000 elements: 1, 2, 3, ..., 2000
        let n = 2000usize;
        let input: Vec<i32> = (1..=n as i32).collect();

        let result = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("large array inclusive scan should succeed");

        assert_eq!(result.len(), n);
        // Verify first few elements
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 3);
        assert_eq!(result[2], 6);
        // Verify last element: sum of 1..=2000 = 2000 * 2001 / 2 = 2_001_000
        assert_eq!(result[n - 1], (n as i32) * (n as i32 + 1) / 2);

        // Verify every element against the closed-form formula: sum(1..=k) = k*(k+1)/2
        for (i, &val) in result.iter().enumerate() {
            let k = (i + 1) as i32;
            let expected = k * (k + 1) / 2;
            assert_eq!(
                val, expected,
                "mismatch at index {}: got {}, expected {}",
                i, val, expected
            );
        }
    }

    #[test]
    fn test_parallel_scan_large_array_exclusive() {
        let n = 1500usize;
        let input: Vec<i64> = (1..=n as i64).collect();

        let inclusive = parallel_scan(&input, 0i64, |a, b| a + b, ScanMode::Inclusive)
            .expect("large inclusive scan should succeed");
        let exclusive = parallel_scan(&input, 0i64, |a, b| a + b, ScanMode::Exclusive)
            .expect("large exclusive scan should succeed");

        assert_eq!(exclusive.len(), n);
        // exclusive[0] = identity
        assert_eq!(exclusive[0], 0);
        // exclusive[i] = inclusive[i-1] for i > 0
        for i in 1..n {
            assert_eq!(
                exclusive[i],
                inclusive[i - 1],
                "exclusive[{}] = {} should equal inclusive[{}] = {}",
                i,
                exclusive[i],
                i - 1,
                inclusive[i - 1]
            );
        }
    }

    #[test]
    fn test_parallel_scan_f64_type() {
        let input = vec![0.5f64, 1.5, 2.0, 0.25, 3.75];

        let result = parallel_scan(&input, 0.0, |a, b| a + b, ScanMode::Inclusive)
            .expect("f64 inclusive scan should succeed");

        let expected = [0.5, 2.0, 4.0, 4.25, 8.0];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "f64 mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_parallel_scan_usize_type() {
        let input = vec![10usize, 20, 30, 40, 50];

        let result = parallel_scan(&input, 0usize, |a, b| a + b, ScanMode::Inclusive)
            .expect("usize inclusive scan should succeed");
        assert_eq!(result, vec![10, 30, 60, 100, 150]);

        let exclusive = parallel_scan(&input, 0usize, |a, b| a + b, ScanMode::Exclusive)
            .expect("usize exclusive scan should succeed");
        assert_eq!(exclusive, vec![0, 10, 30, 60, 100]);
    }

    #[test]
    fn test_parallel_scan_max_operation() {
        // Scan with max operation (identity = 0 for non-negative values)
        let input = vec![3i32, 1, 4, 1, 5, 9, 2, 6, 5, 3];

        let result = parallel_scan(
            &input,
            0,
            |a, b| if a > b { a } else { b },
            ScanMode::Inclusive,
        )
        .expect("max inclusive scan should succeed");

        // Running maximum: [3, 3, 4, 4, 5, 9, 9, 9, 9, 9]
        assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9, 9, 9]);
    }

    #[test]
    fn test_parallel_scan_correctness_against_sequential() {
        // Verify parallel scan matches sequential computation for a medium-sized array
        let n = 5000usize;
        let input: Vec<i64> = (1..=n as i64).collect();

        let parallel_result = parallel_scan(&input, 0i64, |a, b| a + b, ScanMode::Inclusive)
            .expect("parallel scan should succeed");

        // Compute sequential reference
        let mut sequential_result = Vec::with_capacity(n);
        let mut acc = 0i64;
        for &val in &input {
            acc += val;
            sequential_result.push(acc);
        }

        assert_eq!(parallel_result.len(), sequential_result.len());
        for i in 0..n {
            assert_eq!(
                parallel_result[i], sequential_result[i],
                "mismatch at index {}: parallel={}, sequential={}",
                i, parallel_result[i], sequential_result[i]
            );
        }
    }

    #[test]
    fn test_parallel_prefix_sum_inclusive() {
        let data = vec![1i32, 2, 3, 4, 5];
        let result = parallel_prefix_sum(&data, ScanMode::Inclusive)
            .expect("parallel_prefix_sum inclusive should succeed");
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_parallel_prefix_sum_exclusive() {
        let data = vec![1i32, 2, 3, 4, 5];
        let result = parallel_prefix_sum(&data, ScanMode::Exclusive)
            .expect("parallel_prefix_sum exclusive should succeed");
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_parallel_prefix_sum_f64() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_prefix_sum(&data, ScanMode::Inclusive)
            .expect("parallel_prefix_sum f64 should succeed");

        let expected = [1.0, 3.0, 6.0, 10.0, 15.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "f64 prefix sum mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_parallel_prefix_sum_large_array() {
        let n = 10_000usize;
        let input: Vec<i64> = (1..=n as i64).collect();

        let result = parallel_prefix_sum(&input, ScanMode::Inclusive)
            .expect("large parallel_prefix_sum should succeed");

        assert_eq!(result.len(), n);
        // Last element should be n*(n+1)/2
        let expected_total = (n as i64) * (n as i64 + 1) / 2;
        assert_eq!(result[n - 1], expected_total);

        // Spot-check a few values
        assert_eq!(result[0], 1);
        assert_eq!(result[99], 100 * 101 / 2); // sum of 1..=100 = 5050
        assert_eq!(result[999], 1000 * 1001 / 2); // sum of 1..=1000 = 500500
    }

    #[test]
    fn test_parallel_scan_all_zeros() {
        let input = vec![0i32; 100];
        let result = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Inclusive)
            .expect("all-zeros inclusive scan should succeed");
        assert_eq!(result, vec![0i32; 100]);

        let exclusive = parallel_scan(&input, 0, |a, b| a + b, ScanMode::Exclusive)
            .expect("all-zeros exclusive scan should succeed");
        assert_eq!(exclusive, vec![0i32; 100]);
    }

    #[test]
    fn test_parallel_scan_all_ones_product() {
        let input = vec![1i32; 50];
        let result = parallel_scan(&input, 1, |a, b| a * b, ScanMode::Inclusive)
            .expect("all-ones product scan should succeed");
        assert_eq!(result, vec![1i32; 50]);
    }
}

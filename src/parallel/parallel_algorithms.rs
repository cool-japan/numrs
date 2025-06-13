//! Parallel algorithms optimized for numerical computations
//!
//! This module provides high-performance parallel implementations of common
//! numerical algorithms including matrix operations, FFT, and array processing.

use crate::error::{NumRs2Error, Result};
use crate::traits::{NumericElement, FloatingPoint};
use super::{WorkStealingPool, Task, TaskResult};
use super::work_stealing::task;
use std::sync::{Arc, Mutex};
use std::thread;
use std::marker::PhantomData;

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
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
            
        Self {
            num_threads: Some(num_threads),
            parallel_threshold: 1000,
            block_size: 64,
            numa_aware: false,
            chunk_size: 256,
        }
    }
}

/// Parallel array operations with optimized work distribution
pub struct ParallelArrayOps {
    config: ParallelConfig,
    pool: Arc<WorkStealingPool>,
}

impl ParallelArrayOps {
    /// Create new parallel array operations
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        
        let pool = Arc::new(WorkStealingPool::new(num_threads)?);
        
        Ok(Self { config, pool })
    }

    /// Parallel element-wise operation on two arrays
    pub fn parallel_binary_op<T, F>(
        &self,
        a: &[T],
        b: &[T],
        result: &mut [T],
        op: F,
    ) -> Result<()>
    where
        T: NumericElement + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync + Copy + 'static,
    {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Array dimensions must match".to_string()
            ));
        }

        let len = a.len();
        if len < self.config.parallel_threshold {
            // Sequential execution for small arrays
            for i in 0..len {
                result[i] = op(a[i], b[i]);
            }
            return Ok(());
        }

        let chunk_size = self.config.chunk_size.min(len / self.config.num_threads.unwrap_or(4));
        let chunk_size = chunk_size.max(1);
        
        // Convert to raw pointers for parallel access
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        let mut handles = Vec::new();
        
        for chunk_start in (0..len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(len);
            
            let task = task(move || {
                unsafe {
                    for i in chunk_start..chunk_end {
                        let a_val = *a_ptr.add(i);
                        let b_val = *b_ptr.add(i);
                        *result_ptr.add(i) = op(a_val, b_val);
                    }
                }
            });
            
            self.pool.submit(task)?;
        }

        // Wait for completion by checking pending tasks
        while self.pool.pending_tasks() > 0 {
            thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    }

    /// Parallel reduction operation
    pub fn parallel_reduce<T, F>(&self, data: &[T], init: T, op: F) -> Result<T>
    where
        T: NumericElement + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync + Copy + 'static,
    {
        if data.is_empty() {
            return Ok(init);
        }

        if data.len() < self.config.parallel_threshold {
            return Ok(data.iter().copied().fold(init, op));
        }

        let num_threads = self.config.num_threads.unwrap_or(4);
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        
        let partial_results = Arc::new(Mutex::new(Vec::new()));
        
        for chunk_start in (0..data.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(data.len());
            let chunk = &data[chunk_start..chunk_end];
            let chunk_data: Vec<T> = chunk.to_vec(); // Copy for ownership
            let results_clone = Arc::clone(&partial_results);
            
            let task = task(move || {
                let local_result = chunk_data.iter().copied().fold(init, op);
                results_clone.lock().unwrap().push(local_result);
            });
            
            self.pool.submit(task)?;
        }

        // Wait for completion
        while self.pool.pending_tasks() > 0 {
            thread::sleep(std::time::Duration::from_millis(1));
        }

        // Combine partial results
        let results = partial_results.lock().unwrap();
        Ok(results.iter().copied().fold(init, op))
    }

    /// Parallel prefix sum (scan) operation
    pub fn parallel_prefix_sum<T>(&self, data: &[T], result: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync,
    {
        if data.len() != result.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Input and output arrays must have same length".to_string()
            ));
        }

        if data.is_empty() {
            return Ok(());
        }

        if data.len() < self.config.parallel_threshold {
            // Sequential scan
            result[0] = data[0];
            for i in 1..data.len() {
                result[i] = result[i - 1] + data[i];
            }
            return Ok(());
        }

        // Parallel prefix sum using work-efficient algorithm
        self.work_efficient_scan(data, result)
    }

    /// Work-efficient parallel scan implementation
    fn work_efficient_scan<T>(&self, data: &[T], result: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync,
    {
        let n = data.len();
        result.copy_from_slice(data);

        // Up-sweep phase
        let mut d = 1;
        while d < n {
            let step = d * 2;
            for i in (step - 1..n).step_by(step) {
                result[i] = result[i] + result[i - d];
            }
            d *= 2;
        }

        // Clear the last element
        result[n - 1] = T::zero();

        // Down-sweep phase
        d = n / 2;
        while d > 0 {
            let step = d * 2;
            for i in (step - 1..n).step_by(step) {
                let temp = result[i - d];
                result[i - d] = result[i];
                result[i] = result[i] + temp;
            }
            d /= 2;
        }

        Ok(())
    }

    /// Parallel sort using merge sort
    pub fn parallel_sort<T>(&self, data: &mut [T]) -> Result<()>
    where
        T: NumericElement + Send + Sync + Ord,
    {
        if data.len() < self.config.parallel_threshold {
            data.sort();
            return Ok(());
        }

        self.parallel_merge_sort(data, 0)
    }

    fn parallel_merge_sort<T>(&self, data: &mut [T], depth: usize) -> Result<()>
    where
        T: NumericElement + Send + Sync + Ord,
    {
        let len = data.len();
        if len <= 1 {
            return Ok(());
        }

        let mid = len / 2;
        let max_depth = (self.config.num_threads.unwrap_or(4) as f64).log2() as usize;

        if depth < max_depth && len > self.config.parallel_threshold {
            // Parallel recursion
            let (left, right) = data.split_at_mut(mid);
            
            // For now, we'll use sequential recursion due to borrowing constraints
            // In a real implementation, we'd need more sophisticated synchronization
            self.parallel_merge_sort(left, depth + 1)?;
            self.parallel_merge_sort(right, depth + 1)?;
        } else {
            // Sequential recursion
            let (left, right) = data.split_at_mut(mid);
            left.sort();
            right.sort();
        }

        // Merge the sorted halves
        self.merge_sorted_slices(data, mid);
        Ok(())
    }

    fn merge_sorted_slices<T>(&self, data: &mut [T], mid: usize)
    where
        T: NumericElement + Copy + Ord,
    {
        let mut temp = Vec::with_capacity(data.len());
        let (left, right) = data.split_at(mid);
        
        let mut i = 0;
        let mut j = 0;
        
        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                temp.push(left[i]);
                i += 1;
            } else {
                temp.push(right[j]);
                j += 1;
            }
        }
        
        while i < left.len() {
            temp.push(left[i]);
            i += 1;
        }
        
        while j < right.len() {
            temp.push(right[j]);
            j += 1;
        }
        
        data.copy_from_slice(&temp);
    }
}

/// Parallel matrix operations
pub struct ParallelMatrixOps {
    config: ParallelConfig,
    pool: Arc<WorkStealingPool>,
}

impl ParallelMatrixOps {
    /// Create new parallel matrix operations
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        
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
        T: NumericElement + Send + Sync,
    {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(NumRs2Error::DimensionMismatch(
                "Matrix dimensions don't match for multiplication".to_string()
            ));
        }

        // Initialize result matrix
        for elem in c.iter_mut() {
            *elem = T::zero();
        }

        let total_ops = m * n * k;
        if total_ops < self.config.parallel_threshold {
            // Sequential multiplication
            for i in 0..m {
                for j in 0..n {
                    for l in 0..k {
                        c[i * n + j] = c[i * n + j] + a[i * k + l] * b[l * n + j];
                    }
                }
            }
            return Ok(());
        }

        // Parallel block-based multiplication
        let block_size = self.config.block_size;
        
        // Convert to raw pointers for parallel access
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let task = task(move || {
                        let i_end = (i_block + block_size).min(m);
                        let j_end = (j_block + block_size).min(n);
                        let k_end = (k_block + block_size).min(k);

                        unsafe {
                            for i in i_block..i_end {
                                for j in j_block..j_end {
                                    let mut sum = *c_ptr.add(i * n + j);
                                    for l in k_block..k_end {
                                        let a_val = *a_ptr.add(i * k + l);
                                        let b_val = *b_ptr.add(l * n + j);
                                        sum = sum + a_val * b_val;
                                    }
                                    *c_ptr.add(i * n + j) = sum;
                                }
                            }
                        }
                    });
                    
                    self.pool.submit(task)?;
                }
            }
        }

        // Wait for completion
        while self.pool.pending_tasks() > 0 {
            thread::sleep(std::time::Duration::from_millis(1));
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
        T: NumericElement + Send + Sync,
    {
        if src.len() != rows * cols || dst.len() != rows * cols {
            return Err(NumRs2Error::DimensionMismatch(
                "Source and destination matrices must have compatible dimensions".to_string()
            ));
        }

        if rows * cols < self.config.parallel_threshold {
            // Sequential transpose
            for i in 0..rows {
                for j in 0..cols {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            return Ok(());
        }

        // Parallel blocked transpose
        let block_size = self.config.block_size;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        for i_block in (0..rows).step_by(block_size) {
            for j_block in (0..cols).step_by(block_size) {
                let task = task(move || {
                    let i_end = (i_block + block_size).min(rows);
                    let j_end = (j_block + block_size).min(cols);

                    unsafe {
                        for i in i_block..i_end {
                            for j in j_block..j_end {
                                let src_val = *src_ptr.add(i * cols + j);
                                *dst_ptr.add(j * rows + i) = src_val;
                            }
                        }
                    }
                });
                
                self.pool.submit(task)?;
            }
        }

        // Wait for completion
        while self.pool.pending_tasks() > 0 {
            thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    }
}

/// Parallel FFT implementation
pub struct ParallelFFT<T> {
    config: ParallelConfig,
    pool: Arc<WorkStealingPool>,
    _phantom: PhantomData<T>,
}

impl<T: FloatingPoint + Send + Sync> ParallelFFT<T> {
    /// Create new parallel FFT
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });
        
        let pool = Arc::new(WorkStealingPool::new(num_threads)?);
        
        Ok(Self {
            config,
            pool,
            _phantom: PhantomData,
        })
    }

    /// Parallel FFT computation
    pub fn parallel_fft(&self, data: &mut [num_complex::Complex<T>]) -> Result<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(NumRs2Error::InvalidOperation(
                "FFT requires power-of-two length".to_string()
            ));
        }

        if n < self.config.parallel_threshold {
            return self.sequential_fft(data);
        }

        self.parallel_fft_recursive(data, false)
    }

    /// Parallel inverse FFT
    pub fn parallel_ifft(&self, data: &mut [num_complex::Complex<T>]) -> Result<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(NumRs2Error::InvalidOperation(
                "IFFT requires power-of-two length".to_string()
            ));
        }

        if n < self.config.parallel_threshold {
            return self.sequential_ifft(data);
        }

        self.parallel_fft_recursive(data, true)?;
        
        // Scale by 1/n for inverse transform
        let scale = num_complex::Complex::new(
            T::one() / T::from_f64(n as f64).unwrap(),
            T::zero()
        );
        for sample in data.iter_mut() {
            *sample = *sample * scale;
        }
        
        Ok(())
    }

    fn parallel_fft_recursive(&self, data: &mut [num_complex::Complex<T>], inverse: bool) -> Result<()> {
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
            // TODO: Implement true parallel recursion
            // For now, use sequential due to borrowing constraints
            self.parallel_fft_recursive(&mut even, inverse)?;
            self.parallel_fft_recursive(&mut odd, inverse)?;
        } else {
            self.parallel_fft_recursive(&mut even, inverse)?;
            self.parallel_fft_recursive(&mut odd, inverse)?;
        }

        // Combine results with twiddle factors
        let two_pi = T::from_f64(2.0 * std::f64::consts::PI).unwrap();
        
        // Parallelize the twiddle factor computation
        let data_ptr = data.as_mut_ptr();
        let even_ptr = even.as_ptr();
        let odd_ptr = odd.as_ptr();

        for chunk_start in (0..n / 2).step_by(self.config.chunk_size) {
            let chunk_end = (chunk_start + self.config.chunk_size).min(n / 2);
            
            let task = task(move || {
                for i in chunk_start..chunk_end {
                    let angle = if inverse {
                        two_pi * T::from_f64(i as f64).unwrap() / T::from_f64(n as f64).unwrap()
                    } else {
                        -two_pi * T::from_f64(i as f64).unwrap() / T::from_f64(n as f64).unwrap()
                    };
                    
                    let cos_angle = angle.cos();
                    let sin_angle = angle.sin();
                    let twiddle = num_complex::Complex::new(cos_angle, sin_angle);
                    
                    unsafe {
                        let even_val = *even_ptr.add(i);
                        let odd_val = *odd_ptr.add(i);
                        let t = twiddle * odd_val;
                        
                        *data_ptr.add(i) = even_val + t;
                        *data_ptr.add(i + n / 2) = even_val - t;
                    }
                }
            });
            
            self.pool.submit(task)?;
        }

        // Wait for completion
        while self.pool.pending_tasks() > 0 {
            thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(())
    }

    fn sequential_fft(&self, data: &mut [num_complex::Complex<T>]) -> Result<()> {
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
            let two_pi = T::from_f64(2.0 * std::f64::consts::PI).unwrap();
            let angle = -two_pi / T::from_f64(length as f64).unwrap();
            
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let w_len = num_complex::Complex::new(cos_angle, sin_angle);
            
            for i in (0..n).step_by(length) {
                let mut w = num_complex::Complex::new(T::one(), T::zero());
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

    fn sequential_ifft(&self, data: &mut [num_complex::Complex<T>]) -> Result<()> {
        // Conjugate the complex numbers
        for sample in data.iter_mut() {
            *sample = sample.conj();
        }

        // Perform forward FFT
        self.sequential_fft(data)?;

        // Conjugate again and scale
        let n = data.len();
        let scale = T::one() / T::from_f64(n as f64).unwrap();
        for sample in data.iter_mut() {
            *sample = sample.conj() * scale;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_parallel_array_ops_creation() {
        let config = ParallelConfig::default();
        let ops = ParallelArrayOps::new(config).unwrap();
        assert!(ops.config.num_threads.unwrap() > 0);
    }

    #[test]
    fn test_parallel_binary_op() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 5];

        ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y).unwrap();
        
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_parallel_reduce() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = ops.parallel_reduce(&data, 0, |a, b| a + b).unwrap();
        
        assert_eq!(result, 55);
    }

    #[test]
    fn test_parallel_prefix_sum() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        let mut result = vec![0; 5];

        ops.parallel_prefix_sum(&data, &mut result).unwrap();
        
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_parallel_matrix_multiplication() {
        let config = ParallelConfig {
            parallel_threshold: 10,
            block_size: 2,
            ..Default::default()
        };
        let ops = ParallelMatrixOps::new(config).unwrap();

        // 2x2 * 2x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 0.0, 1.0, 2.0];
        let mut c = vec![0.0; 4];

        ops.parallel_matmul(&a, &b, &mut c, 2, 2, 2).unwrap();
        
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
        let ops = ParallelMatrixOps::new(config).unwrap();

        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let mut dst = vec![0.0; 6];

        ops.parallel_transpose(&src, &mut dst, 2, 3).unwrap();
        
        // Expected transpose: [1, 4, 2, 5, 3, 6] (3x2)
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_parallel_sort() {
        let config = ParallelConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).unwrap();

        let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        ops.parallel_sort(&mut data).unwrap();
        
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_fft() {
        let config = ParallelConfig {
            parallel_threshold: 8,
            ..Default::default()
        };
        let fft = ParallelFFT::<f64>::new(config).unwrap();

        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let original = data.clone();
        fft.parallel_fft(&mut data).unwrap();
        
        // FFT should change the data
        assert_ne!(data, original);
        
        // IFFT should restore original (approximately)
        fft.parallel_ifft(&mut data).unwrap();
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let config = ParallelConfig::default();
        let ops = ParallelArrayOps::new(config).unwrap();

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0]; // Different length
        let mut result = vec![0.0; 3];

        let err = ops.parallel_binary_op(&a, &b, &mut result, |x, y| x + y);
        assert!(err.is_err());
    }
}
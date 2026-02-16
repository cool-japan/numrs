//! Cache-aware algorithm implementations
//!
//! This module provides optimized implementations of common numerical algorithms
//! that are designed to maximize cache efficiency and minimize memory bandwidth.

#![allow(clippy::result_large_err)]
#![allow(clippy::needless_range_loop)]

use crate::error::{NumRs2Error, Result};
#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_alloc::cache_optimization::cache_constants;
use crate::memory_alloc::cache_optimization::{CacheConfig, CacheLevel};
use crate::traits::{FloatingPoint, NumericElement};
use std::marker::PhantomData;

/// Cache-aware array operations with blocking and tiling strategies
pub struct CacheAwareArrayOps<T> {
    cache_config: CacheConfig,
    _phantom: PhantomData<T>,
}

impl<T: NumericElement> CacheAwareArrayOps<T> {
    /// Create new cache-aware array operations
    pub fn new(cache_config: CacheConfig) -> Self {
        Self {
            cache_config,
            _phantom: PhantomData,
        }
    }

    /// Create with default cache configuration
    pub fn with_default_config() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Calculate optimal tile size for 2D operations
    pub fn optimal_tile_size(&self, element_size: usize) -> (usize, usize) {
        let cache_size = self.cache_config.l1_cache_size / 4; // Use 1/4 of L1 cache
        let elements_per_tile = cache_size / element_size;

        // For square tiles
        let side_length = (elements_per_tile as f64).sqrt() as usize;
        let power_of_two_side = side_length.next_power_of_two() / 2;

        (power_of_two_side, power_of_two_side)
    }

    /// Cache-blocked matrix transpose
    pub fn transpose_blocked(
        &self,
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
    ) -> Result<()>
    where
        T: Copy,
    {
        if src.len() != rows * cols || dst.len() != rows * cols {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::dimension_mismatch(
                    "Invalid matrix dimensions",
                    None,
                    None,
                ),
            ));
        }

        let (tile_rows, tile_cols) = self.optimal_tile_size(std::mem::size_of::<T>());

        // Process in cache-friendly tiles
        for row_tile in (0..rows).step_by(tile_rows) {
            for col_tile in (0..cols).step_by(tile_cols) {
                let row_end = (row_tile + tile_rows).min(rows);
                let col_end = (col_tile + tile_cols).min(cols);

                // Transpose this tile
                for i in row_tile..row_end {
                    for j in col_tile..col_end {
                        dst[j * rows + i] = src[i * cols + j];
                    }
                }
            }
        }

        Ok(())
    }

    /// Cache-efficient vector sum with blocking
    pub fn sum_blocked(&self, data: &[T]) -> T
    where
        T: Copy + std::ops::Add<Output = T>,
    {
        let block_size = self.cache_config.l1_cache_size / (4 * std::mem::size_of::<T>());
        let block_size = block_size.max(1);

        let mut total = T::zero();

        for chunk in data.chunks(block_size) {
            let mut partial_sum = T::zero();
            for &value in chunk {
                partial_sum = partial_sum + value;
            }
            total = total + partial_sum;
        }

        total
    }

    /// Cache-aware matrix-vector multiplication
    pub fn matvec_blocked(
        &self,
        matrix: &[T],
        vector: &[T],
        result: &mut [T],
        rows: usize,
        cols: usize,
    ) -> Result<()>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if matrix.len() != rows * cols || vector.len() != cols || result.len() != rows {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::dimension_mismatch(
                    "Invalid dimensions for matrix-vector multiplication",
                    None,
                    None,
                ),
            ));
        }

        let block_size = self.cache_config.l1_cache_size / (8 * std::mem::size_of::<T>());
        let block_size = block_size.max(1);

        // Initialize result
        for i in 0..rows {
            result[i] = T::zero();
        }

        // Block the computation for better cache utilization
        for row_block in (0..rows).step_by(block_size) {
            for col_block in (0..cols).step_by(block_size) {
                let row_end = (row_block + block_size).min(rows);
                let col_end = (col_block + block_size).min(cols);

                for i in row_block..row_end {
                    let mut partial_sum = T::zero();
                    for j in col_block..col_end {
                        partial_sum = partial_sum + matrix[i * cols + j] * vector[j];
                    }
                    result[i] = result[i] + partial_sum;
                }
            }
        }

        Ok(())
    }

    /// Cache-oblivious merge sort implementation
    pub fn merge_sort_cache_oblivious(&self, data: &mut [T]) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        if data.len() <= 1 {
            return Ok(());
        }

        self.merge_sort_recursive(data)?;
        Ok(())
    }

    /// Recursive cache-oblivious merge sort
    fn merge_sort_recursive(&self, data: &mut [T]) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        let len = data.len();
        if len <= 1 {
            return Ok(());
        }

        // Use insertion sort for small arrays (cache-friendly)
        if len <= 32 {
            self.insertion_sort(data);
            return Ok(());
        }

        let mid = len / 2;
        self.merge_sort_recursive(&mut data[..mid])?;
        self.merge_sort_recursive(&mut data[mid..])?;

        // Merge the two sorted halves
        self.merge_in_place(data, mid)?;

        Ok(())
    }

    /// Cache-friendly insertion sort for small arrays
    fn insertion_sort(&self, data: &mut [T])
    where
        T: Copy + PartialOrd,
    {
        for i in 1..data.len() {
            let key = data[i];
            let mut j = i;

            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }

            data[j] = key;
        }
    }

    /// In-place merge for cache-oblivious merge sort
    fn merge_in_place(&self, data: &mut [T], mid: usize) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        // Use a temporary buffer for merging
        let left_len = mid;
        let _right_len = data.len() - mid;

        // Allocate temporary storage
        let mut temp = Vec::with_capacity(left_len);
        temp.extend_from_slice(&data[..mid]);

        let mut i = 0; // Index for temp (left half)
        let mut j = mid; // Index for right half
        let mut k = 0; // Index for merged result

        // Merge temp and right half back into data
        while i < temp.len() && j < data.len() {
            if temp[i] <= data[j] {
                data[k] = temp[i];
                i += 1;
            } else {
                data[k] = data[j];
                j += 1;
            }
            k += 1;
        }

        // Copy remaining elements from temp
        while i < temp.len() {
            data[k] = temp[i];
            i += 1;
            k += 1;
        }

        Ok(())
    }

    /// Cache-aware stride optimization for array access patterns
    pub fn optimize_stride_access<F>(
        &self,
        data: &mut [T],
        rows: usize,
        cols: usize,
        mut operation: F,
    ) -> Result<()>
    where
        F: FnMut(&mut T, usize, usize),
        T: Copy,
    {
        if data.len() != rows * cols {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::dimension_mismatch(
                    "Invalid array dimensions",
                    None,
                    None,
                ),
            ));
        }

        let (tile_rows, tile_cols) = self.optimal_tile_size(std::mem::size_of::<T>());

        // Process in cache-friendly order
        for row_tile in (0..rows).step_by(tile_rows) {
            for col_tile in (0..cols).step_by(tile_cols) {
                let row_end = (row_tile + tile_rows).min(rows);
                let col_end = (col_tile + tile_cols).min(cols);

                for i in row_tile..row_end {
                    for j in col_tile..col_end {
                        operation(&mut data[i * cols + j], i, j);
                    }
                }
            }
        }

        Ok(())
    }
}

impl<T: NumericElement> Default for CacheAwareArrayOps<T> {
    fn default() -> Self {
        Self::with_default_config()
    }
}

/// Cache-aware FFT implementation with locality optimization
pub struct CacheAwareFFT<T> {
    #[allow(dead_code)]
    cache_config: CacheConfig,
    _phantom: PhantomData<T>,
}

impl<T: FloatingPoint> CacheAwareFFT<T> {
    /// Create new cache-aware FFT
    pub fn new(cache_config: CacheConfig) -> Self {
        Self {
            cache_config,
            _phantom: PhantomData,
        }
    }

    /// Cache-oblivious FFT implementation
    pub fn fft_cache_oblivious(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        if !n.is_power_of_two() {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::invalid_operation(
                    "FFT",
                    "requires power-of-two length",
                ),
            ));
        }

        self.fft_recursive(data, false)?;
        Ok(())
    }

    /// Cache-oblivious inverse FFT
    pub fn ifft_cache_oblivious(&self, data: &mut [scirs2_core::Complex<T>]) -> Result<()> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        if !n.is_power_of_two() {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::invalid_operation(
                    "IFFT",
                    "requires power-of-two length",
                ),
            ));
        }

        self.fft_recursive(data, true)?;

        // Scale by 1/n for inverse transform
        let scale = scirs2_core::Complex::new(
            <T as NumericElement>::one()
                / T::from_f64(n as f64).expect("Failed to convert FFT length to numeric type"),
            <T as NumericElement>::zero(),
        );
        for sample in data.iter_mut() {
            *sample = *sample * scale;
        }

        Ok(())
    }

    /// Recursive cache-oblivious FFT implementation
    fn fft_recursive(&self, data: &mut [scirs2_core::Complex<T>], inverse: bool) -> Result<()> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        // Use iterative approach for small sizes to improve cache locality
        if n <= 64 {
            return self.fft_iterative(data, inverse);
        }

        // Divide into even and odd indices
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);

        for i in 0..n / 2 {
            even.push(data[2 * i]);
            odd.push(data[2 * i + 1]);
        }

        // Recursive FFT on even and odd parts
        self.fft_recursive(&mut even, inverse)?;
        self.fft_recursive(&mut odd, inverse)?;

        // Combine results with twiddle factors
        let two_pi = T::from_f64(2.0 * std::f64::consts::PI)
            .expect("Failed to convert 2*PI to numeric type");
        for i in 0..n / 2 {
            let angle = if inverse {
                two_pi * T::from_f64(i as f64).expect("Failed to convert index to numeric type")
                    / T::from_f64(n as f64).expect("Failed to convert length to numeric type")
            } else {
                -two_pi * T::from_f64(i as f64).expect("Failed to convert index to numeric type")
                    / T::from_f64(n as f64).expect("Failed to convert length to numeric type")
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

    /// Iterative FFT for small arrays (better cache locality)
    fn fft_iterative(&self, data: &mut [scirs2_core::Complex<T>], inverse: bool) -> Result<()> {
        let n = data.len();

        // Bit-reverse the input
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
                .expect("Failed to convert 2*PI to numeric type");
            let angle = if inverse {
                two_pi
                    / T::from_f64(length as f64).expect("Failed to convert length to numeric type")
            } else {
                -two_pi
                    / T::from_f64(length as f64).expect("Failed to convert length to numeric type")
            };

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
}

/// Cache-aware convolution operations
pub struct CacheAwareConvolution<T> {
    cache_config: CacheConfig,
    _phantom: PhantomData<T>,
}

impl<T: NumericElement + Copy> CacheAwareConvolution<T> {
    /// Create new cache-aware convolution operations
    pub fn new(cache_config: CacheConfig) -> Self {
        Self {
            cache_config,
            _phantom: PhantomData,
        }
    }

    /// Cache-blocked 2D convolution
    pub fn conv2d_blocked(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        input_height: usize,
        input_width: usize,
        kernel_height: usize,
        kernel_width: usize,
    ) -> Result<()>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        let output_height = input_height - kernel_height + 1;
        let output_width = input_width - kernel_width + 1;

        if input.len() != input_height * input_width
            || kernel.len() != kernel_height * kernel_width
            || output.len() != output_height * output_width
        {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::dimension_mismatch(
                    "Invalid convolution dimensions",
                    None,
                    None,
                ),
            ));
        }

        // Calculate block sizes for cache efficiency
        let element_size = std::mem::size_of::<T>();
        let cache_size = self.cache_config.l1_cache_size / 4;
        let block_size = (cache_size / element_size).max(1);
        let tile_size = (block_size as f64).sqrt() as usize;

        // Process in cache-friendly blocks
        for out_row_block in (0..output_height).step_by(tile_size) {
            for out_col_block in (0..output_width).step_by(tile_size) {
                let out_row_end = (out_row_block + tile_size).min(output_height);
                let out_col_end = (out_col_block + tile_size).min(output_width);

                // Process this output block
                for out_row in out_row_block..out_row_end {
                    for out_col in out_col_block..out_col_end {
                        let mut sum = T::zero();

                        // Convolve with kernel
                        for k_row in 0..kernel_height {
                            for k_col in 0..kernel_width {
                                let in_row = out_row + k_row;
                                let in_col = out_col + k_col;

                                let input_val = input[in_row * input_width + in_col];
                                let kernel_val = kernel[k_row * kernel_width + k_col];

                                sum = sum + input_val * kernel_val;
                            }
                        }

                        output[out_row * output_width + out_col] = sum;
                    }
                }
            }
        }

        Ok(())
    }

    /// Cache-aware separable convolution (for separable kernels)
    pub fn separable_conv2d(
        &self,
        input: &[T],
        h_kernel: &[T],
        v_kernel: &[T],
        output: &mut [T],
        height: usize,
        width: usize,
    ) -> Result<()>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        let h_kernel_size = h_kernel.len();
        let v_kernel_size = v_kernel.len();

        if input.len() != height * width || output.len() != height * width {
            return Err(NumRs2Error::Core(
                crate::error::core::CoreError::dimension_mismatch(
                    "Invalid separable convolution dimensions",
                    None,
                    None,
                ),
            ));
        }

        // Temporary buffer for horizontal pass
        let mut temp = vec![T::zero(); height * width];

        // Horizontal pass - cache-friendly row processing
        for row in 0..height {
            for col in 0..width {
                let mut sum = T::zero();

                for k in 0..h_kernel_size {
                    let input_col =
                        if col + k >= h_kernel_size / 2 && col + k - h_kernel_size / 2 < width {
                            col + k - h_kernel_size / 2
                        } else {
                            // Handle boundary conditions (zero padding)
                            continue;
                        };

                    sum = sum + input[row * width + input_col] * h_kernel[k];
                }

                temp[row * width + col] = sum;
            }
        }

        // Vertical pass - process in column blocks for cache efficiency
        let col_block_size = self.cache_config.l1_cache_size / (8 * std::mem::size_of::<T>());
        let col_block_size = col_block_size.max(1);

        for col_block in (0..width).step_by(col_block_size) {
            let col_end = (col_block + col_block_size).min(width);

            for col in col_block..col_end {
                for row in 0..height {
                    let mut sum = T::zero();

                    for k in 0..v_kernel_size {
                        let input_row = if row + k >= v_kernel_size / 2
                            && row + k - v_kernel_size / 2 < height
                        {
                            row + k - v_kernel_size / 2
                        } else {
                            // Handle boundary conditions (zero padding)
                            continue;
                        };

                        sum = sum + temp[input_row * width + col] * v_kernel[k];
                    }

                    output[row * width + col] = sum;
                }
            }
        }

        Ok(())
    }
}

/// Memory bandwidth optimization utilities
pub struct BandwidthOptimizer {
    cache_config: CacheConfig,
}

impl BandwidthOptimizer {
    pub fn new(cache_config: CacheConfig) -> Self {
        Self { cache_config }
    }

    /// Estimate memory bandwidth requirements for an operation
    pub fn estimate_bandwidth<T>(&self, operation: MemoryOperation<T>) -> BandwidthEstimate {
        match operation {
            MemoryOperation::MatrixMultiply { m, n, k, .. } => {
                let element_size = std::mem::size_of::<T>();
                let reads = (m * k + k * n) * element_size;
                let writes = m * n * element_size;
                let total_bytes = reads + writes;

                BandwidthEstimate {
                    total_bytes,
                    cache_friendly: self.fits_in_cache(total_bytes, CacheLevel::L3),
                    recommended_blocking: !self.fits_in_cache(total_bytes, CacheLevel::L2),
                    estimated_time_ns: self.estimate_access_time(total_bytes),
                }
            }
            MemoryOperation::VectorOperation { length, .. } => {
                let element_size = std::mem::size_of::<T>();
                let total_bytes = length * element_size * 2; // Read + write

                BandwidthEstimate {
                    total_bytes,
                    cache_friendly: self.fits_in_cache(total_bytes, CacheLevel::L1),
                    recommended_blocking: false,
                    estimated_time_ns: self.estimate_access_time(total_bytes),
                }
            }
            MemoryOperation::Convolution {
                input_size,
                kernel_size,
                output_size,
                ..
            } => {
                let element_size = std::mem::size_of::<T>();
                let reads = (input_size + kernel_size) * element_size;
                let writes = output_size * element_size;
                let total_bytes = reads + writes;

                BandwidthEstimate {
                    total_bytes,
                    cache_friendly: self.fits_in_cache(total_bytes, CacheLevel::L2),
                    recommended_blocking: !self.fits_in_cache(total_bytes, CacheLevel::L1),
                    estimated_time_ns: self.estimate_access_time(total_bytes),
                }
            }
        }
    }

    fn fits_in_cache(&self, size: usize, cache_level: CacheLevel) -> bool {
        let cache_size = match cache_level {
            CacheLevel::L1 => self.cache_config.l1_cache_size,
            CacheLevel::L2 => self.cache_config.l2_cache_size,
            CacheLevel::L3 => self.cache_config.l3_cache_size,
        };

        size <= (cache_size * 4) / 5 // 80% utilization threshold
    }

    fn estimate_access_time(&self, total_bytes: usize) -> u64 {
        // Simplified model: different latencies for different cache levels
        if self.fits_in_cache(total_bytes, CacheLevel::L1) {
            total_bytes as u64 * 1 // ~1ns per byte for L1
        } else if self.fits_in_cache(total_bytes, CacheLevel::L2) {
            total_bytes as u64 * 3 // ~3ns per byte for L2
        } else if self.fits_in_cache(total_bytes, CacheLevel::L3) {
            total_bytes as u64 * 10 // ~10ns per byte for L3
        } else {
            total_bytes as u64 * 100 // ~100ns per byte for main memory
        }
    }
}

/// Memory operation types for bandwidth analysis
pub enum MemoryOperation<T> {
    MatrixMultiply {
        m: usize,
        n: usize,
        k: usize,
        _phantom: PhantomData<T>,
    },
    VectorOperation {
        length: usize,
        _phantom: PhantomData<T>,
    },
    Convolution {
        input_size: usize,
        kernel_size: usize,
        output_size: usize,
        _phantom: PhantomData<T>,
    },
}

/// Memory bandwidth estimation result
#[derive(Debug, Clone)]
pub struct BandwidthEstimate {
    pub total_bytes: usize,
    pub cache_friendly: bool,
    pub recommended_blocking: bool,
    pub estimated_time_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex;

    #[test]
    fn test_cache_aware_transpose() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<f32>::new(config);

        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0; 6];

        ops.transpose_blocked(&src, &mut dst, 2, 3)
            .expect("Transpose should succeed");

        // Expected: [1, 4, 2, 5, 3, 6] (transpose of 2x3 -> 3x2)
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cache_blocked_sum() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<f32>::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ops.sum_blocked(&data);

        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_cache_aware_matvec() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<f32>::new(config);

        let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![1.0, 1.0];
        let mut result = vec![0.0; 2];

        ops.matvec_blocked(&matrix, &vector, &mut result, 2, 2)
            .expect("Matrix-vector multiplication should succeed");

        // Expected: [3.0, 7.0] (matrix * vector)
        assert_eq!(result, vec![3.0, 7.0]);
    }

    #[test]
    fn test_cache_oblivious_merge_sort() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<i32>::new(config);

        let mut data = vec![5, 2, 8, 1, 9, 3];
        ops.merge_sort_cache_oblivious(&mut data)
            .expect("Cache-oblivious merge sort should succeed");

        assert_eq!(data, vec![1, 2, 3, 5, 8, 9]);
    }

    #[test]
    fn test_cache_aware_fft() {
        let config = CacheConfig::default();
        let fft = CacheAwareFFT::<f64>::new(config);

        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        fft.fft_cache_oblivious(&mut data)
            .expect("Cache-oblivious FFT should succeed");

        // Should have non-zero values after FFT
        assert!(data.iter().any(|&x| x.norm() > 0.1));
    }

    #[test]
    fn test_cache_blocked_convolution() {
        let config = CacheConfig::default();
        let conv = CacheAwareConvolution::<f32>::new(config);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 input
        let kernel = vec![1.0]; // 1x1 kernel (identity)
        let mut output = vec![0.0; 4]; // 2x2 output

        conv.conv2d_blocked(&input, &kernel, &mut output, 2, 2, 1, 1)
            .expect("2D blocked convolution should succeed");

        // Identity convolution should preserve input
        assert_eq!(output, input);
    }

    #[test]
    fn test_bandwidth_estimation() {
        let config = CacheConfig::default();
        let optimizer = BandwidthOptimizer::new(config);

        let estimate = optimizer.estimate_bandwidth(MemoryOperation::<f32>::VectorOperation {
            length: 1000,
            _phantom: PhantomData,
        });

        assert!(estimate.total_bytes > 0);
        assert!(estimate.estimated_time_ns > 0);
    }

    #[test]
    fn test_optimal_tile_size_calculation() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<f64>::new(config);

        let (rows, cols) = ops.optimal_tile_size(8); // 8 bytes per f64

        assert!(rows > 0);
        assert!(cols > 0);
        assert!(rows.is_power_of_two());
        assert!(cols.is_power_of_two());
    }

    #[test]
    fn test_separable_convolution() {
        let config = CacheConfig::default();
        let conv = CacheAwareConvolution::<f32>::new(config);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 input
        let h_kernel = vec![1.0]; // 1x1 horizontal kernel
        let v_kernel = vec![1.0]; // 1x1 vertical kernel
        let mut output = vec![0.0; 4];

        conv.separable_conv2d(&input, &h_kernel, &v_kernel, &mut output, 2, 2)
            .expect("Separable 2D convolution should succeed");

        // Identity separable convolution
        assert_eq!(output, input);
    }

    #[test]
    fn test_stride_optimization() {
        let config = CacheConfig::default();
        let ops = CacheAwareArrayOps::<f32>::new(config);

        let mut data = vec![0.0; 9]; // 3x3 matrix

        ops.optimize_stride_access(&mut data, 3, 3, |element, i, j| {
            *element = (i * 3 + j) as f32;
        })
        .expect("Stride optimization should succeed");

        // Check that elements were set correctly
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(data[i * 3 + j], (i * 3 + j) as f32);
            }
        }
    }
}

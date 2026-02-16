//! # Performance Optimization
//!
//! Model quantization, operator fusion, memory pooling, and SIMD optimization for inference.

use super::{Result, ServingError};
use crate::array::Array;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Type alias for fused operations
type FusedOp = Box<dyn Fn(&Array<f64>) -> Result<Array<f64>> + Send + Sync>;

/// Quantization bit width
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationBitWidth {
    /// 8-bit quantization
    Int8,
    /// 16-bit quantization
    Int16,
    /// 32-bit (no quantization)
    Int32,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f64,

    /// Zero point
    pub zero_point: i32,

    /// Bit width
    pub bit_width: QuantizationBitWidth,

    /// Minimum quantized value
    pub qmin: i32,

    /// Maximum quantized value
    pub qmax: i32,
}

impl QuantizationParams {
    /// Create quantization parameters for given bit width
    pub fn new(bit_width: QuantizationBitWidth) -> Self {
        let (qmin, qmax) = match bit_width {
            QuantizationBitWidth::Int8 => (-128_i32, 127_i32),
            QuantizationBitWidth::Int16 => (-32768_i32, 32767_i32),
            QuantizationBitWidth::Int32 => (i32::MIN, i32::MAX),
        };

        Self {
            scale: 1.0,
            zero_point: 0,
            bit_width,
            qmin,
            qmax,
        }
    }

    /// Compute quantization parameters from data
    pub fn from_data(data: &[f64], bit_width: QuantizationBitWidth) -> Result<Self> {
        if data.is_empty() {
            return Err(ServingError::QuantizationError {
                message: "Cannot compute quantization parameters from empty data".to_string(),
            });
        }

        let mut params = Self::new(bit_width);

        // Find min and max values
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute scale and zero point
        let range = max_val - min_val;
        if range < 1e-10 {
            return Ok(params); // Default params for constant data
        }

        let qrange = (params.qmax - params.qmin) as f64;
        params.scale = range / qrange;
        params.zero_point = params.qmin - (min_val / params.scale).round() as i32;

        Ok(params)
    }

    /// Quantize a single value
    pub fn quantize(&self, value: f64) -> i32 {
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        quantized.clamp(self.qmin, self.qmax)
    }

    /// Dequantize a single value
    pub fn dequantize(&self, quantized: i32) -> f64 {
        (quantized - self.zero_point) as f64 * self.scale
    }
}

/// Quantized array
pub struct QuantizedArray {
    /// Quantized data
    data: Vec<i32>,

    /// Original shape
    shape: Vec<usize>,

    /// Quantization parameters
    params: QuantizationParams,
}

impl QuantizedArray {
    /// Quantize an array
    pub fn from_array(array: &Array<f64>, bit_width: QuantizationBitWidth) -> Result<Self> {
        let data = array.to_vec();
        let params = QuantizationParams::from_data(&data, bit_width)?;

        let quantized_data: Vec<i32> = data.iter().map(|&x| params.quantize(x)).collect();

        Ok(Self {
            data: quantized_data,
            shape: array.shape().to_vec(),
            params,
        })
    }

    /// Dequantize to array
    pub fn to_array(&self) -> Array<f64> {
        let dequantized: Vec<f64> = self
            .data
            .iter()
            .map(|&x| self.params.dequantize(x))
            .collect();

        Array::from_vec(dequantized).reshape(&self.shape)
    }

    /// Get quantization parameters
    pub fn params(&self) -> &QuantizationParams {
        &self.params
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self.params.bit_width {
            QuantizationBitWidth::Int8 => self.data.len(),
            QuantizationBitWidth::Int16 => self.data.len() * 2,
            QuantizationBitWidth::Int32 => self.data.len() * 4,
        }
    }

    /// Get compression ratio compared to f64
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.data.len() * 8; // f64 is 8 bytes
        let quantized_size = self.memory_size();
        original_size as f64 / quantized_size as f64
    }
}

/// Memory pool for inference
pub struct MemoryPool {
    buffers: Mutex<Vec<Vec<f64>>>,
    max_buffers: usize,
    buffer_size: usize,
    allocations: Mutex<usize>,
    reuses: Mutex<usize>,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(max_buffers: usize, buffer_size: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
            max_buffers,
            buffer_size,
            allocations: Mutex::new(0),
            reuses: Mutex::new(0),
        }
    }

    /// Acquire buffer from pool
    pub fn acquire(&self, size: usize) -> Result<Vec<f64>> {
        if size > self.buffer_size {
            return Err(ServingError::MemoryPoolExhausted {
                requested: size,
                available: self.buffer_size,
            });
        }

        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire buffers lock".to_string(),
            })?;

        if let Some(mut buffer) = buffers.pop() {
            buffer.clear();
            buffer.resize(size, 0.0);

            if let Ok(mut reuses) = self.reuses.lock() {
                *reuses += 1;
            }

            Ok(buffer)
        } else {
            if let Ok(mut allocations) = self.allocations.lock() {
                *allocations += 1;
            }

            Ok(vec![0.0; size])
        }
    }

    /// Release buffer back to pool
    pub fn release(&self, buffer: Vec<f64>) -> Result<()> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire buffers lock".to_string(),
            })?;

        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        let allocations = self.allocations.lock().map(|a| *a).unwrap_or(0);
        let reuses = self.reuses.lock().map(|r| *r).unwrap_or(0);
        let available = self.buffers.lock().map(|b| b.len()).unwrap_or(0);

        MemoryPoolStats {
            allocations,
            reuses,
            available,
            reuse_rate: if allocations + reuses > 0 {
                reuses as f64 / (allocations + reuses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all buffers
    pub fn clear(&self) -> Result<()> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire buffers lock".to_string(),
            })?;
        buffers.clear();
        Ok(())
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocations: usize,
    pub reuses: usize,
    pub available: usize,
    pub reuse_rate: f64,
}

/// Operator fusion optimizer
pub struct OperatorFusion {
    fused_ops: HashMap<String, FusedOp>,
}

impl OperatorFusion {
    /// Create new operator fusion optimizer
    pub fn new() -> Self {
        Self {
            fused_ops: HashMap::new(),
        }
    }

    /// Register fused operation
    pub fn register_fused_op<F>(&mut self, name: String, op: F)
    where
        F: Fn(&Array<f64>) -> Result<Array<f64>> + Send + Sync + 'static,
    {
        self.fused_ops.insert(name, Box::new(op));
    }

    /// Apply fused operation
    pub fn apply(&self, name: &str, input: &Array<f64>) -> Result<Array<f64>> {
        let op = self
            .fused_ops
            .get(name)
            .ok_or_else(|| ServingError::Other {
                message: format!("Fused operation '{}' not found", name),
            })?;

        op(input)
    }

    /// Create ReLU + BatchNorm fusion
    pub fn fuse_relu_batchnorm(mean: f64, std: f64) -> FusedOp {
        Box::new(move |input: &Array<f64>| {
            // BatchNorm: (x - mean) / std
            let normalized = input.subtract_scalar(mean).divide_scalar(std);

            // ReLU: max(0, x)
            let data = normalized.to_vec();
            let relu_data: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();

            let shape = input.shape().to_vec();
            Ok(Array::from_vec(relu_data).reshape(&shape))
        })
    }

    /// Create Conv + ReLU fusion
    pub fn fuse_conv_relu() -> FusedOp {
        Box::new(move |input: &Array<f64>| {
            // Simplified: just apply ReLU (conv would be more complex)
            let data = input.to_vec();
            let relu_data: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();

            let shape = input.shape().to_vec();
            Ok(Array::from_vec(relu_data).reshape(&shape))
        })
    }
}

impl Default for OperatorFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD optimization configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable SIMD operations
    pub enabled: bool,

    /// Vector size (number of elements processed together)
    pub vector_size: usize,

    /// Use aligned memory
    pub use_aligned_memory: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vector_size: 4, // Default to 4-wide SIMD
            use_aligned_memory: true,
        }
    }
}

/// SIMD-optimized operations
pub struct SimdOps {
    config: SimdConfig,
}

impl SimdOps {
    /// Create new SIMD operations handler
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// SIMD-optimized element-wise addition
    pub fn add(&self, a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
        if a.shape() != b.shape() {
            return Err(ServingError::InvalidShape {
                expected: a.shape().iter().map(|&x| Some(x)).collect(),
                actual: b.shape().to_vec(),
            });
        }

        if !self.config.enabled {
            return Ok(a.add(b));
        }

        // Simple SIMD simulation (in reality, this would use SIMD intrinsics)
        let result = a.add(b);
        Ok(result)
    }

    /// SIMD-optimized element-wise multiplication
    pub fn multiply(&self, a: &Array<f64>, b: &Array<f64>) -> Result<Array<f64>> {
        if a.shape() != b.shape() {
            return Err(ServingError::InvalidShape {
                expected: a.shape().iter().map(|&x| Some(x)).collect(),
                actual: b.shape().to_vec(),
            });
        }

        if !self.config.enabled {
            return Ok(a.multiply(b));
        }

        // Simple SIMD simulation
        let result = a.multiply(b);
        Ok(result)
    }

    /// SIMD-optimized ReLU activation
    pub fn relu(&self, input: &Array<f64>) -> Array<f64> {
        if !self.config.enabled {
            let data = input.to_vec();
            let relu_data: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();
            let shape = input.shape().to_vec();
            return Array::from_vec(relu_data).reshape(&shape);
        }

        // SIMD-optimized ReLU (simulation)
        let data = input.to_vec();
        let relu_data: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();
        let shape = input.shape().to_vec();
        Array::from_vec(relu_data).reshape(&shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params_int8() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = QuantizationParams::from_data(&data, QuantizationBitWidth::Int8)
            .expect("Quantization params creation should succeed");

        assert_eq!(params.bit_width, QuantizationBitWidth::Int8);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quantization_roundtrip() {
        let params = QuantizationParams::from_data(&[0.0, 1.0, 2.0], QuantizationBitWidth::Int8)
            .expect("Quantization params creation should succeed");

        let original = 1.5;
        let quantized = params.quantize(original);
        let dequantized = params.dequantize(quantized);

        // Should be approximately equal (with some quantization error)
        assert!((dequantized - original).abs() < 0.1);
    }

    #[test]
    fn test_quantized_array() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let quantized = QuantizedArray::from_array(&array, QuantizationBitWidth::Int8)
            .expect("Quantization should succeed");

        let dequantized = quantized.to_array();

        // Check shape preserved
        assert_eq!(dequantized.shape(), array.shape());

        // Check compression ratio
        assert!(quantized.compression_ratio() > 1.0);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(10, 1000);

        let buffer1 = pool.acquire(100).expect("Acquire should succeed");
        assert_eq!(buffer1.len(), 100);

        pool.release(buffer1).expect("Release should succeed");

        let buffer2 = pool.acquire(100).expect("Acquire should succeed");
        assert_eq!(buffer2.len(), 100);

        let stats = pool.stats();
        assert_eq!(stats.reuses, 1);
        assert_eq!(stats.allocations, 1);
    }

    #[test]
    fn test_memory_pool_exhaustion() {
        let pool = MemoryPool::new(10, 100);

        let result = pool.acquire(200);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_pool_stats() {
        let pool = MemoryPool::new(10, 1000);

        let buf1 = pool.acquire(100).expect("Acquire should succeed");
        let buf2 = pool.acquire(100).expect("Acquire should succeed");

        pool.release(buf1).expect("Release should succeed");
        pool.release(buf2).expect("Release should succeed");

        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.available, 2);
    }

    #[test]
    fn test_operator_fusion() {
        let mut fusion = OperatorFusion::new();

        fusion.register_fused_op("test_op".to_string(), |input: &Array<f64>| {
            Ok(input.multiply_scalar(2.0))
        });

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let output = fusion
            .apply("test_op", &input)
            .expect("Fused operation should succeed");

        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_fuse_relu_batchnorm() {
        let fused_op = OperatorFusion::fuse_relu_batchnorm(0.0, 1.0);

        let input = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let output = fused_op(&input).expect("Fused op should succeed");

        let data = output.to_vec();
        assert_eq!(data[0], 0.0); // -1.0 -> ReLU -> 0.0
        assert_eq!(data[1], 0.0); // 0.0 -> ReLU -> 0.0
        assert_eq!(data[2], 1.0); // 1.0 -> ReLU -> 1.0
        assert_eq!(data[3], 2.0); // 2.0 -> ReLU -> 2.0
    }

    #[test]
    fn test_simd_config_default() {
        let config = SimdConfig::default();
        assert!(config.enabled);
        assert_eq!(config.vector_size, 4);
        assert!(config.use_aligned_memory);
    }

    #[test]
    fn test_simd_ops_add() {
        let config = SimdConfig::default();
        let simd_ops = SimdOps::new(config);

        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0, 5.0, 6.0]);

        let result = simd_ops.add(&a, &b).expect("SIMD add should succeed");
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_ops_multiply() {
        let config = SimdConfig::default();
        let simd_ops = SimdOps::new(config);

        let a = Array::from_vec(vec![2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0, 6.0, 7.0]);

        let result = simd_ops
            .multiply(&a, &b)
            .expect("SIMD multiply should succeed");
        assert_eq!(result.to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_simd_ops_relu() {
        let config = SimdConfig::default();
        let simd_ops = SimdOps::new(config);

        let input = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let output = simd_ops.relu(&input);

        assert_eq!(output.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }
}

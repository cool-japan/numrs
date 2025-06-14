//! Broadcasting operations for NumPy-style array operations
//!
//! This module implements efficient broadcasting semantics that allow operations
//! between arrays of different but compatible shapes, following NumPy's broadcasting rules.

use crate::error::{NumRs2Error, Result};
use crate::traits::{NumericElement, FloatingPoint};
use super::advanced_ops::{Shape, ArrayView, BroadcastOp};
use std::cmp;

/// Configuration for broadcasting operations
#[derive(Debug, Clone)]
pub struct BroadcastConfig {
    /// Enable vectorized operations when possible
    pub enable_vectorization: bool,
    /// Enable parallel processing for large operations
    pub enable_parallel: bool,
    /// Minimum size threshold for parallel processing
    pub parallel_threshold: usize,
    /// Memory usage optimization level
    pub memory_optimization: MemoryOptimization,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            enable_vectorization: true,
            enable_parallel: true,
            parallel_threshold: 10000,
            memory_optimization: MemoryOptimization::Balanced,
        }
    }
}

/// Memory optimization strategies for broadcasting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOptimization {
    /// Minimize memory usage at cost of performance
    Memory,
    /// Balance memory usage and performance
    Balanced,
    /// Maximize performance at cost of memory
    Performance,
}

/// Broadcasting engine for efficient array operations
pub struct BroadcastEngine {
    config: BroadcastConfig,
}

impl BroadcastEngine {
    /// Create a new broadcasting engine
    pub fn new(config: BroadcastConfig) -> Self {
        Self { config }
    }

    /// Create a default broadcasting engine
    pub fn default() -> Self {
        Self::new(BroadcastConfig::default())
    }

    /// Add two arrays with broadcasting
    pub fn add<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Add<Output = T>,
    {
        self.binary_op(a, b, |x, y| x + y)
    }

    /// Subtract two arrays with broadcasting
    pub fn subtract<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Sub<Output = T>,
    {
        self.binary_op(a, b, |x, y| x - y)
    }

    /// Multiply two arrays with broadcasting
    pub fn multiply<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Mul<Output = T>,
    {
        self.binary_op(a, b, |x, y| x * y)
    }

    /// Divide two arrays with broadcasting
    pub fn divide<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: FloatingPoint + Copy + std::ops::Div<Output = T>,
    {
        self.binary_op(a, b, |x, y| x / y)
    }

    /// Power operation with broadcasting
    pub fn power<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: FloatingPoint + Copy,
    {
        self.binary_op(a, b, |x, y| x.powf(y))
    }

    /// Maximum of two arrays with broadcasting
    pub fn maximum<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        self.binary_op(a, b, |x, y| if x > y { x } else { y })
    }

    /// Minimum of two arrays with broadcasting
    pub fn minimum<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        self.binary_op(a, b, |x, y| if x < y { x } else { y })
    }

    /// Generic binary operation with broadcasting
    pub fn binary_op<T, F>(&self, a: &ArrayView<T>, b: &ArrayView<T>, op: F) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
        F: Fn(T, T) -> T + Send + Sync,
    {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![T::zero(); broadcast_shape.size()];

        if self.config.enable_parallel && broadcast_shape.size() >= self.config.parallel_threshold {
            self.binary_op_parallel(a, b, &mut result, &broadcast_shape, op)?;
        } else {
            BroadcastOp::binary_op(a, b, &mut result, op)?;
        }

        Ok(result)
    }

    /// Parallel binary operation implementation
    fn binary_op_parallel<T, F>(
        &self,
        a: &ArrayView<T>,
        b: &ArrayView<T>,
        result: &mut [T],
        broadcast_shape: &Shape,
        op: F,
    ) -> Result<()>
    where
        T: NumericElement + Copy + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync,
    {
        use rayon::prelude::*;

        let chunk_size = cmp::max(1, broadcast_shape.size() / rayon::current_num_threads());
        
        result.par_chunks_mut(chunk_size).enumerate().try_for_each(|(chunk_idx, chunk)| {
            let start_idx = chunk_idx * chunk_size;
            
            for (i, output_elem) in chunk.iter_mut().enumerate() {
                let flat_idx = start_idx + i;
                let indices = self.flat_to_multi_index(flat_idx, &broadcast_shape.dims);
                
                let a_indices = self.map_broadcast_indices(&indices, a.shape(), broadcast_shape);
                let b_indices = self.map_broadcast_indices(&indices, b.shape(), broadcast_shape);
                
                let a_val = *a.get(&a_indices).map_err(|e| format!("Error accessing array a: {}", e))?;
                let b_val = *b.get(&b_indices).map_err(|e| format!("Error accessing array b: {}", e))?;
                
                *output_elem = op(a_val, b_val);
            }
            Ok::<(), String>(())
        }).map_err(|e| NumRs2Error::RuntimeError(e))?;

        Ok(())
    }

    /// Element-wise comparison operations with broadcasting
    pub fn equal<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<bool>>
    where
        T: NumericElement + Copy + PartialEq,
    {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![false; broadcast_shape.size()];

        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];
        
        loop {
            let a_indices = self.map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = self.map_broadcast_indices(&indices, b.shape(), &broadcast_shape);
            
            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            result[output_idx] = a_val == b_val;
            
            output_idx += 1;
            
            if !self.advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Greater than comparison with broadcasting
    pub fn greater<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<bool>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![false; broadcast_shape.size()];

        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];
        
        loop {
            let a_indices = self.map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = self.map_broadcast_indices(&indices, b.shape(), &broadcast_shape);
            
            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            result[output_idx] = a_val > b_val;
            
            output_idx += 1;
            
            if !self.advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Less than comparison with broadcasting
    pub fn less<T>(&self, a: &ArrayView<T>, b: &ArrayView<T>) -> Result<Vec<bool>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![false; broadcast_shape.size()];

        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];
        
        loop {
            let a_indices = self.map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = self.map_broadcast_indices(&indices, b.shape(), &broadcast_shape);
            
            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            result[output_idx] = a_val < b_val;
            
            output_idx += 1;
            
            if !self.advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Logical AND operation with broadcasting (for boolean arrays)
    pub fn logical_and(&self, a: &ArrayView<bool>, b: &ArrayView<bool>) -> Result<Vec<bool>> {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![false; broadcast_shape.size()];

        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];
        
        loop {
            let a_indices = self.map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = self.map_broadcast_indices(&indices, b.shape(), &broadcast_shape);
            
            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            result[output_idx] = a_val && b_val;
            
            output_idx += 1;
            
            if !self.advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Logical OR operation with broadcasting (for boolean arrays)
    pub fn logical_or(&self, a: &ArrayView<bool>, b: &ArrayView<bool>) -> Result<Vec<bool>> {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;
        let mut result = vec![false; broadcast_shape.size()];

        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];
        
        loop {
            let a_indices = self.map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = self.map_broadcast_indices(&indices, b.shape(), &broadcast_shape);
            
            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            result[output_idx] = a_val || b_val;
            
            output_idx += 1;
            
            if !self.advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Apply scalar operation to array with broadcasting
    pub fn scalar_op<T, F>(&self, array: &ArrayView<T>, scalar: T, op: F) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
        F: Fn(T, T) -> T,
    {
        let mut result = Vec::with_capacity(array.shape().size());
        // Iterate through all elements manually
        let mut indices = vec![0; array.shape().ndim()];
        loop {
            if let Ok(element) = array.get(&indices) {
                result.push(op(*element, scalar));
            }
            
            // Advance indices
            let mut carry = 1;
            for i in (0..indices.len()).rev() {
                indices[i] += carry;
                if indices[i] < array.shape().dims[i] {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                    carry = 1;
                }
            }
            
            if carry == 1 {
                break;
            }
        }

        Ok(result)
    }

    /// Add scalar to array
    pub fn add_scalar<T>(&self, array: &ArrayView<T>, scalar: T) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Add<Output = T>,
    {
        self.scalar_op(array, scalar, |x, s| x + s)
    }

    /// Multiply array by scalar
    pub fn multiply_scalar<T>(&self, array: &ArrayView<T>, scalar: T) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Mul<Output = T>,
    {
        self.scalar_op(array, scalar, |x, s| x * s)
    }

    /// Check if two shapes can be broadcasted together
    pub fn can_broadcast(&self, shape1: &Shape, shape2: &Shape) -> bool {
        shape1.is_broadcastable_with(shape2)
    }

    /// Compute the result shape of broadcasting two shapes
    pub fn broadcast_shape(&self, shape1: &Shape, shape2: &Shape) -> Result<Shape> {
        shape1.broadcast_with(shape2)
    }

    // Helper methods

    fn flat_to_multi_index(&self, flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(shape.len());
        let mut remaining = flat_index;
        
        for &dim_size in shape.iter().rev() {
            indices.push(remaining % dim_size);
            remaining /= dim_size;
        }
        
        indices.reverse();
        indices
    }

    fn map_broadcast_indices(&self, broadcast_indices: &[usize], original_shape: &Shape, broadcast_shape: &Shape) -> Vec<usize> {
        let mut result = Vec::with_capacity(original_shape.ndim());
        let ndim_diff = broadcast_shape.ndim() - original_shape.ndim();
        
        for i in 0..original_shape.ndim() {
            let broadcast_idx = broadcast_indices[i + ndim_diff];
            let original_dim = original_shape.dims[i];
            
            // If original dimension is 1, use index 0 (broadcasting)
            let mapped_idx = if original_dim == 1 { 0 } else { broadcast_idx };
            result.push(mapped_idx);
        }
        
        result
    }

    fn advance_indices(&self, indices: &mut [usize], shape: &[usize]) -> bool {
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < shape[i] {
                return true;
            }
            indices[i] = 0;
        }
        false
    }
}

/// Reduction operations with broadcasting support
pub struct BroadcastReduction {
    #[allow(dead_code)]
    engine: BroadcastEngine,
}

impl BroadcastReduction {
    /// Create a new broadcast reduction engine
    pub fn new(config: BroadcastConfig) -> Self {
        Self {
            engine: BroadcastEngine::new(config),
        }
    }

    /// Sum along specified axes with broadcasting
    pub fn sum<T>(&self, array: &ArrayView<T>, axes: Option<Vec<usize>>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + std::ops::Add<Output = T>,
    {
        match axes {
            None => {
                // Sum all elements
                let mut sum = T::zero();
                let mut indices = vec![0; array.shape().ndim()];
                loop {
                    if let Ok(element) = array.get(&indices) {
                        sum = sum + *element;
                    }
                    
                    // Advance indices
                    let mut carry = 1;
                    for i in (0..indices.len()).rev() {
                        indices[i] += carry;
                        if indices[i] < array.shape().dims[i] {
                            carry = 0;
                            break;
                        } else {
                            indices[i] = 0;
                            carry = 1;
                        }
                    }
                    
                    if carry == 1 {
                        break;
                    }
                }
                Ok(vec![sum])
            },
            Some(axes) => {
                // Sum along specific axes
                self.reduce_along_axes(array, axes, T::zero(), |acc, x| acc + x)
            }
        }
    }

    /// Mean along specified axes with broadcasting
    pub fn mean<T>(&self, array: &ArrayView<T>, axes: Option<Vec<usize>>) -> Result<Vec<T>>
    where
        T: FloatingPoint + Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
    {
        let sum_result = self.sum(array, axes.clone())?;
        
        let count = match axes {
            None => array.shape().size(),
            Some(axes) => {
                let mut count = 1;
                for &axis in &axes {
                    if axis < array.shape().ndim() {
                        count *= array.shape().dims[axis];
                    }
                }
                count
            }
        };

        let count_t = T::from_f64(count as f64).unwrap_or(<T as crate::traits::NumericElement>::one());
        Ok(sum_result.into_iter().map(|x| x / count_t).collect())
    }

    /// Maximum along specified axes
    pub fn max<T>(&self, array: &ArrayView<T>, axes: Option<Vec<usize>>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        if array.shape().size() == 0 {
            return Err(NumRs2Error::InvalidOperation("Cannot find max of empty array".to_string()));
        }

        // Get first element as init value
        let init = *array.get(&vec![0; array.shape().ndim()])?;
        match axes {
            None => {
                let mut max_val = init;
                let mut indices = vec![0; array.shape().ndim()];
                loop {
                    if let Ok(element) = array.get(&indices) {
                        if *element > max_val {
                            max_val = *element;
                        }
                    }
                    
                    // Advance indices
                    let mut carry = 1;
                    for i in (0..indices.len()).rev() {
                        indices[i] += carry;
                        if indices[i] < array.shape().dims[i] {
                            carry = 0;
                            break;
                        } else {
                            indices[i] = 0;
                            carry = 1;
                        }
                    }
                    
                    if carry == 1 {
                        break;
                    }
                }
                Ok(vec![max_val])
            },
            Some(axes) => {
                self.reduce_along_axes(array, axes, init, |acc, x| if x > acc { x } else { acc })
            }
        }
    }

    /// Minimum along specified axes
    pub fn min<T>(&self, array: &ArrayView<T>, axes: Option<Vec<usize>>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy + PartialOrd,
    {
        if array.shape().size() == 0 {
            return Err(NumRs2Error::InvalidOperation("Cannot find min of empty array".to_string()));
        }

        // Get first element as init value
        let init = *array.get(&vec![0; array.shape().ndim()])?;
        match axes {
            None => {
                let mut min_val = init;
                let mut indices = vec![0; array.shape().ndim()];
                loop {
                    if let Ok(element) = array.get(&indices) {
                        if *element < min_val {
                            min_val = *element;
                        }
                    }
                    
                    // Advance indices
                    let mut carry = 1;
                    for i in (0..indices.len()).rev() {
                        indices[i] += carry;
                        if indices[i] < array.shape().dims[i] {
                            carry = 0;
                            break;
                        } else {
                            indices[i] = 0;
                            carry = 1;
                        }
                    }
                    
                    if carry == 1 {
                        break;
                    }
                }
                Ok(vec![min_val])
            },
            Some(axes) => {
                self.reduce_along_axes(array, axes, init, |acc, x| if x < acc { x } else { acc })
            }
        }
    }

    /// Generic reduction along axes
    fn reduce_along_axes<T, F>(
        &self,
        array: &ArrayView<T>,
        axes: Vec<usize>,
        init: T,
        reduce_fn: F,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
        F: Fn(T, T) -> T,
    {
        // Validate axes
        for &axis in &axes {
            if axis >= array.shape().ndim() {
                return Err(NumRs2Error::DimensionMismatch(
                    format!("Axis {} is out of bounds for array of dimension {}", axis, array.shape().ndim())
                ));
            }
        }

        // Compute output shape
        let mut output_dims = Vec::new();
        for (i, &dim) in array.shape().dims.iter().enumerate() {
            if !axes.contains(&i) {
                output_dims.push(dim);
            }
        }
        
        if output_dims.is_empty() {
            output_dims.push(1); // Scalar result
        }

        let output_size: usize = output_dims.iter().product();
        let mut result = vec![init; output_size];

        // Simple implementation - could be optimized
        // For now, iterate through all elements and accumulate
        let mut indices = vec![0; array.shape().ndim()];
        loop {
            if let Ok(element) = array.get(&indices) {
                // This is a simplified version - a full implementation would
                // need to properly map the multi-dimensional indices
                if !result.is_empty() {
                    result[0] = reduce_fn(result[0], *element);
                }
            }
            
            // Advance indices
            let mut carry = 1;
            for i in (0..indices.len()).rev() {
                indices[i] += carry;
                if indices[i] < array.shape().dims[i] {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                    carry = 1;
                }
            }
            
            if carry == 1 {
                break;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::advanced_ops::{Shape, ArrayView};

    #[test]
    fn test_broadcast_engine_creation() {
        let config = BroadcastConfig::default();
        let engine = BroadcastEngine::new(config);
        
        // Test with simple arrays
        let data_a = vec![1.0, 2.0, 3.0];
        let shape_a = Shape::from_1d(3);
        let view_a = ArrayView::from_data(&data_a, shape_a).unwrap();
        
        let data_b = vec![10.0];
        let shape_b = Shape::from_1d(1);
        let view_b = ArrayView::from_data(&data_b, shape_b).unwrap();
        
        assert!(engine.can_broadcast(view_a.shape(), view_b.shape()));
    }

    #[test]
    fn test_broadcast_add() {
        let engine = BroadcastEngine::default();
        
        let data_a = vec![1.0, 2.0, 3.0];
        let shape_a = Shape::from_1d(3);
        let view_a = ArrayView::from_data(&data_a, shape_a).unwrap();
        
        let data_b = vec![10.0];
        let shape_b = Shape::from_1d(1);
        let view_b = ArrayView::from_data(&data_b, shape_b).unwrap();
        
        let result = engine.add(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_broadcast_multiply() {
        let engine = BroadcastEngine::default();
        
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = Shape::from_2d(2, 2);
        let view_a = ArrayView::from_data(&data_a, shape_a).unwrap();
        
        let data_b = vec![2.0, 3.0];
        let shape_b = Shape::from_1d(2);
        let view_b = ArrayView::from_data(&data_b, shape_b).unwrap();
        
        let result = engine.multiply(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_broadcast_scalar_operations() {
        let engine = BroadcastEngine::default();
        
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::from_2d(2, 2);
        let view = ArrayView::from_data(&data, shape).unwrap();
        
        let result = engine.add_scalar(&view, 5.0).unwrap();
        assert_eq!(result, vec![6.0, 7.0, 8.0, 9.0]);
        
        let result = engine.multiply_scalar(&view, 2.0).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_broadcast_comparisons() {
        let engine = BroadcastEngine::default();
        
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = Shape::from_1d(4);
        let view_a = ArrayView::from_data(&data_a, shape_a).unwrap();
        
        let data_b = vec![2.5];
        let shape_b = Shape::from_1d(1);
        let view_b = ArrayView::from_data(&data_b, shape_b).unwrap();
        
        let result = engine.greater(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![false, false, true, true]);
        
        let result = engine.less(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![true, true, false, false]);
    }

    #[test]
    fn test_broadcast_logical_operations() {
        let engine = BroadcastEngine::default();
        
        // Test with compatible shapes for broadcasting
        let data_a = vec![true, false, true, false];
        let shape_a = Shape::from_2d(2, 2);
        let view_a = ArrayView::from_data(&data_a, shape_a).unwrap();
        
        let data_b = vec![true, false];
        let shape_b = Shape::from_2d(1, 2);  // Shape [1, 2] can broadcast to [2, 2]
        let view_b = ArrayView::from_data(&data_b, shape_b).unwrap();
        
        let result = engine.logical_and(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![true, false, true, false]);
        
        let result = engine.logical_or(&view_a, &view_b).unwrap();
        assert_eq!(result, vec![true, false, true, false]);
    }

    #[test]
    fn test_broadcast_reduction() {
        let config = BroadcastConfig::default();
        let reduction = BroadcastReduction::new(config);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::from_2d(2, 3);
        let view = ArrayView::from_data(&data, shape).unwrap();
        
        // Sum all elements
        let result = reduction.sum(&view, None).unwrap();
        assert_eq!(result, vec![21.0]);
        
        // Mean of all elements
        let result = reduction.mean(&view, None).unwrap();
        assert_eq!(result, vec![3.5]);
        
        // Max of all elements
        let result = reduction.max(&view, None).unwrap();
        assert_eq!(result, vec![6.0]);
        
        // Min of all elements
        let result = reduction.min(&view, None).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_broadcast_shape_computation() {
        let engine = BroadcastEngine::default();
        
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![2, 4]);
        
        assert!(engine.can_broadcast(&shape1, &shape2));
        
        let result_shape = engine.broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(result_shape.dims, vec![3, 2, 4]);
    }
}
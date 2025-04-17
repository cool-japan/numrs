use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use rayon::prelude::*;

/// Utilities for parallel computation
pub fn parallel_map<T, U, F>(array: &Array<T>, f: F) -> Array<U>
where
    T: Send + Sync + Clone,
    U: Send + Clone,
    F: Fn(T) -> U + Send + Sync,
{
    let vec_data = array.to_vec();
    let result: Vec<U> = vec_data
        .par_iter()
        .map(|x| f(x.clone()))
        .collect();
    
    let shape = array.shape();
    Array::from_vec(result).reshape(&shape)
}

/// Memory layout optimization utilities
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
}

/// Optimizes the memory layout of an array for a specific operation
pub fn optimize_layout<T: Clone>(array: &Array<T>, layout: MemoryLayout) -> Array<T> {
    // In a real implementation, this would convert between row-major and column-major layouts
    // For this example, we'll just return a clone of the array
    match layout {
        MemoryLayout::RowMajor => array.clone(),
        MemoryLayout::ColumnMajor => {
            // For column-major, we could transpose and use specialized algorithms
            // Here we just return the original array for simplicity
            array.clone()
        }
    }
}

/// Checks if an array can be operated on in-place
pub fn can_operate_inplace<T>(_array: &Array<T>) -> bool {
    // This would check if the array is contiguous and has the right memory layout
    // For this example, we'll just return true
    true
}

/// Broadcasting utilities
pub fn broadcast_arrays<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }
    
    // Determine the broadcast shape
    let mut broadcast_shape = Vec::new();
    for array in arrays {
        let shape = array.shape();
        if broadcast_shape.is_empty() {
            broadcast_shape = shape.clone();
        } else {
            // Compute the broadcast shape
            let mut new_shape = Vec::new();
            let max_dims = broadcast_shape.len().max(shape.len());
            
            // Pad shapes with 1s
            let padded_a = pad_shape(&broadcast_shape, max_dims);
            let padded_b = pad_shape(&shape, max_dims);
            
            // Compute the broadcast shape
            for i in 0..max_dims {
                let dim_a = padded_a[i];
                let dim_b = padded_b[i];
                
                if dim_a == 1 {
                    new_shape.push(dim_b);
                } else if dim_b == 1 {
                    new_shape.push(dim_a);
                } else if dim_a == dim_b {
                    new_shape.push(dim_a);
                } else {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: broadcast_shape,
                        actual: shape.clone(),
                    });
                }
            }
            
            broadcast_shape = new_shape;
        }
    }
    
    // Broadcast each array to the broadcast shape
    let mut result = Vec::new();
    for array in arrays {
        result.push(broadcast_to(array, &broadcast_shape)?);
    }
    
    Ok(result)
}

fn pad_shape(shape: &[usize], target_len: usize) -> Vec<usize> {
    let mut padded = vec![1; target_len];
    let offset = target_len - shape.len();
    for (i, &dim) in shape.iter().enumerate() {
        padded[i + offset] = dim;
    }
    padded
}

fn broadcast_to<T: Clone>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    let orig_shape = array.shape();
    
    // Check if broadcasting is possible
    if orig_shape == shape {
        return Ok(array.clone());
    }
    
    let padded_orig = pad_shape(&orig_shape, shape.len());
    
    for i in 0..shape.len() {
        if padded_orig[i] != 1 && padded_orig[i] != shape[i] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: orig_shape,
            });
        }
    }
    
    // For this example, we'll create a new array with the broadcast shape
    // In a real implementation, we'd use ndarray's broadcasting capabilities
    let orig_data = array.to_vec();
    let mut result_data = Vec::new();
    
    // This is a simplified broadcasting implementation
    // A real implementation would be more efficient
    let size: usize = shape.iter().product();
    result_data.reserve(size);
    
    // Create a simple mapping from result indices to original indices
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    let mut orig_strides = vec![1; padded_orig.len()];
    for i in (0..padded_orig.len() - 1).rev() {
        orig_strides[i] = orig_strides[i + 1] * padded_orig[i + 1];
    }
    
    // Fill the result array
    for i in 0..size {
        let mut orig_idx = 0;
        let mut idx = i;
        
        for j in 0..shape.len() {
            let dim_idx = idx / strides[j];
            idx %= strides[j];
            
            if padded_orig[j] > 1 {
                orig_idx += dim_idx * orig_strides[j];
            }
        }
        
        result_data.push(orig_data[orig_idx].clone());
    }
    
    Ok(Array::from_vec(result_data).reshape(shape))
}

/// Type conversion utilities
pub fn astype<T: Clone, U: Clone + From<T>>(array: &Array<T>) -> Array<U> {
    let data = array.to_vec();
    let converted: Vec<U> = data.into_iter().map(U::from).collect();
    Array::from_vec(converted).reshape(&array.shape())
}

// Specialized optimizations for common operations
pub fn fast_sum<T: Float + Send + Sync>(array: &Array<T>) -> T {
    let data = array.to_vec();
    data.par_iter()
        .cloned()
        .reduce(|| T::zero(), |a, b| a + b)
}
//! Array operations - mapping, aggregation, and element-wise operations
//!
//! This module contains methods for:
//! - Element-wise operations (map, par_map)
//! - Aggregation operations (sum, product, sum_axis)
//! - Binary operations (zip_with, broadcast_op)
//! - Scalar operations (scalar_mul, scalar_div)

use super::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{One, Zero};
use scirs2_core::ndarray::Array1;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::ops::{Add, Div, Mul, Sub};

impl<T: Clone> Array<T> {
    /// Perform element-wise multiplication by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new array with each element multiplied by the scalar
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Clone + Mul<Output = T>,
    {
        self.map(|x| x * scalar.clone())
    }

    /// Perform element-wise division by a scalar
    ///
    /// # Parameters
    ///
    /// * `scalar` - The scalar value to divide by
    ///
    /// # Returns
    ///
    /// A new array with each element divided by the scalar
    pub fn scalar_div(&self, scalar: T) -> Self
    where
        T: Clone + Div<Output = T>,
    {
        self.map(|x| x / scalar.clone())
    }

    /// Calculate the sum of all elements in the array
    ///
    /// # Returns
    ///
    /// The sum of all elements
    ///
    /// # Performance
    ///
    /// Optimized to avoid unnecessary allocations by iterating directly over the array.
    pub fn sum_all(&self) -> T
    where
        T: Clone + Add<Output = T> + Zero,
    {
        // Direct iteration without to_vec() allocation
        self.data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the sum along the specified axis
    ///
    /// # Parameters
    ///
    /// * `axis` - The axis along which to sum
    ///
    /// # Returns
    ///
    /// A new array with the specified axis removed
    pub fn sum_axis(&self, axis: usize) -> Result<Self>
    where
        T: Clone + Add<Output = T> + Zero,
    {
        let axis_val = axis;
        if axis_val >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis_val,
                self.ndim()
            )));
        }

        let shape = self.shape();
        let axis_size = shape[axis_val];

        // Calculate the shape of the result
        let mut result_shape = shape.clone();
        result_shape.remove(axis_val);

        // Initialize the result array
        let mut result = Self::zeros(&result_shape);

        // Get the raw data
        let data = self.to_vec();

        // Helper function to calculate indices
        let mut indices = vec![0; shape.len()];
        let mut result_indices = vec![0; result_shape.len()];

        // Calculate the total number of elements in the result
        let result_size = result.size();

        // For each position in the result array
        for i in 0..result_size {
            // Convert flat index to multi-dimensional indices
            let mut remainder = i;
            for j in (0..result_shape.len()).rev() {
                result_indices[j] = remainder % result_shape[j];
                remainder /= result_shape[j];
            }

            // Copy the result indices to the array indices, accounting for the removed axis
            let mut result_idx = 0;
            for (j, idx) in indices.iter_mut().enumerate() {
                if j == axis_val {
                    *idx = 0; // Start at 0 for the axis we're summing
                } else {
                    *idx = result_indices[result_idx];
                    result_idx += 1;
                }
            }

            // Sum along the specified axis
            let mut sum = T::zero();
            for k in 0..axis_size {
                indices[axis_val] = k;

                // Calculate the flat index in the original data
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..shape.len()).rev() {
                    flat_idx += indices[j] * stride;
                    stride *= shape[j];
                }

                sum = sum + data[flat_idx].clone();
            }

            // Set the result value
            result.set(&result_indices, sum)?;
        }

        Ok(result)
    }

    /// Apply a function to each element of the array in parallel
    pub fn par_map<F, U>(&self, f: F) -> Array<U>
    where
        T: Send + Sync + Clone,
        U: Send + Clone,
        F: Fn(T) -> U + Send + Sync,
    {
        let vec_data = self.to_vec();
        let result: Vec<U> = vec_data.par_iter().map(|x| f(x.clone())).collect();

        Array::from_vec(result).reshape(&self.shape())
    }

    /// Apply a function to each element of the array
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        U: Clone,
        F: Fn(T) -> U,
        T: Clone,
    {
        let vec_data = self.to_vec();
        let result: Vec<U> = vec_data.iter().map(|x| f(x.clone())).collect();

        Array::from_vec(result).reshape(&self.shape())
    }

    /// Apply a function to corresponding elements of two arrays with broadcasting
    pub fn zip_with<F, U, V>(&self, other: &Array<U>, f: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(T, U) -> V,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply function directly without broadcasting
        if a_shape == b_shape {
            let self_data = self.to_vec();
            let other_data = other.to_vec();

            let result: Vec<V> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(a, b)| f(a.clone(), b.clone()))
                .collect();

            return Ok(Array::from_vec(result).reshape(&self.shape()));
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Now apply the function to the broadcasted arrays (which have the same shape)
        let self_data = self_broadcast.to_vec();
        let other_data = other_broadcast.to_vec();

        let result: Vec<V> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| f(a.clone(), b.clone()))
            .collect();

        Ok(Array::from_vec(result).reshape(&broadcast_shape))
    }

    /// Broadcast binary operation between two arrays of potentially different shapes
    pub fn broadcast_op<F, U, V>(&self, other: &Array<U>, op: F) -> Result<Array<V>>
    where
        T: Clone,
        U: Clone,
        V: Clone,
        F: Fn(&Array<T>, &Array<U>) -> Array<V>,
    {
        let a_shape = self.shape();
        let b_shape = other.shape();

        // If shapes are equal, apply operation directly
        if a_shape == b_shape {
            return Ok(op(self, other));
        }

        // Calculate broadcast shape
        let broadcast_shape = Self::broadcast_shape(&a_shape, &b_shape)?;

        // Broadcast both arrays to the new shape
        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        // Apply the operation on the broadcasted arrays
        Ok(op(&self_broadcast, &other_broadcast))
    }
}

// Add sum and product methods
impl<T> Array<T>
where
    T: Clone + Add<Output = T> + Zero + Mul<Output = T> + One + 'static,
{
    /// Calculate the sum of all elements in the array
    /// Uses SimdUnifiedOps for SIMD acceleration on supported platforms
    pub fn sum(&self) -> T {
        // Use SIMD for f64 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let data: Vec<f64> = self
                .to_vec()
                .iter()
                .map(|x| {
                    // Safe type conversion through f64
                    let ptr = x as *const T as *const f64;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f64::simd_sum(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        // Use SIMD for f32 arrays with sufficient size via SimdUnifiedOps
        if self.len() >= 64 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let data: Vec<f32> = self
                .to_vec()
                .iter()
                .map(|x| {
                    let ptr = x as *const T as *const f32;
                    unsafe { *ptr }
                })
                .collect();
            let nd_array = Array1::from_vec(data);
            let result = f32::simd_sum(&nd_array.view());
            return unsafe { std::mem::transmute_copy(&result) };
        }
        let data = self.to_vec();
        data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Calculate the product of all elements in the array
    /// Note: Product reduction doesn't have direct SIMD support, uses scalar fallback
    pub fn product(&self) -> T {
        let data = self.to_vec();
        data.iter().fold(T::one(), |acc, x| acc * x.clone())
    }
}

// Non-Result returning versions for convenience (assumes same shape)
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add arrays without broadcasting (for convenience)
    pub fn add(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data + &other.data;
        Array { data: result }
    }

    /// Add arrays with broadcasting
    pub fn add_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data + &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract arrays without broadcasting (for convenience)
    pub fn subtract(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data - &other.data;
        Array { data: result }
    }

    /// Subtract arrays with broadcasting
    pub fn subtract_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data - &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply arrays without broadcasting (for convenience)
    pub fn multiply(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data * &other.data;
        Array { data: result }
    }

    /// Multiply arrays with broadcasting
    pub fn multiply_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data * &b.data;
            Array { data: result }
        })
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide arrays without broadcasting (for convenience)
    pub fn divide(&self, other: &Array<T>) -> Array<T> {
        let result = &self.data / &other.data;
        Array { data: result }
    }

    /// Divide arrays with broadcasting
    pub fn divide_broadcast(&self, other: &Array<T>) -> Result<Array<T>> {
        self.broadcast_op(other, |a, b| {
            let result = &a.data / &b.data;
            Array { data: result }
        })
    }
}

// Implement scalar operations
impl<T: Clone + Add<Output = T>> Array<T> {
    /// Add a scalar to the array (element-wise)
    pub fn add_scalar(&self, scalar: T) -> Self {
        self.map(|x| x + scalar.clone())
    }
}

impl<T: Clone + Sub<Output = T>> Array<T> {
    /// Subtract a scalar from the array (element-wise)
    pub fn subtract_scalar(&self, scalar: T) -> Self {
        self.map(|x| x - scalar.clone())
    }
}

impl<T: Clone + Mul<Output = T>> Array<T> {
    /// Multiply the array by a scalar (element-wise)
    pub fn multiply_scalar(&self, scalar: T) -> Self {
        self.map(|x| x * scalar.clone())
    }
}

impl<T: Clone + Div<Output = T>> Array<T> {
    /// Divide the array by a scalar (element-wise)
    pub fn divide_scalar(&self, scalar: T) -> Self {
        self.map(|x| x / scalar.clone())
    }
}

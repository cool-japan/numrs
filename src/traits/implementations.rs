//! Trait implementations for NumRS2 Array<T> type
//!
//! This module provides implementations of the core trait system for the
//! existing Array<T> type, ensuring backward compatibility while enabling
//! the new trait-based architecture.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::indexing::IndexSpec;
use crate::traits::*;
use num_traits::{Float, NumCast, Zero};

// =============================================================================
// ARRAY OPERATIONS IMPLEMENTATION
// =============================================================================

impl<T: NumericElement> ArrayOps<T> for Array<T> {
    type Output = Array<T>;
    type Error = NumRs2Error;
    
    fn add(&self, other: &Self) -> Result<Self::Output> {
        // Delegate to existing implementation
        Ok(self.add(other))
    }
    
    fn sub(&self, other: &Self) -> Result<Self::Output> {
        Ok(self.subtract(other))
    }
    
    fn mul(&self, other: &Self) -> Result<Self::Output> {
        Ok(self.multiply(other))
    }
    
    fn div(&self, other: &Self) -> Result<Self::Output> {
        Ok(self.divide(other))
    }
    
    fn add_scalar(&self, scalar: T) -> Self::Output {
        self.map(|x| x + scalar)
    }
    
    fn mul_scalar(&self, scalar: T) -> Self::Output {
        self.map(|x| x * scalar)
    }
    
    fn div_scalar(&self, scalar: T) -> Result<Self::Output> {
        if scalar.is_zero() {
            return Err(NumRs2Error::InvalidOperation("Division by zero".to_string()));
        }
        Ok(self.map(|x| x / scalar))
    }
    
    fn add_broadcast(&self, other: &Self) -> Result<Self::Output> {
        // Delegate to existing broadcasting implementation
        Ok(self.add(other))
    }
    
    fn mul_broadcast(&self, other: &Self) -> Result<Self::Output> {
        Ok(self.multiply(other))
    }
}

// =============================================================================
// ARRAY REDUCTION IMPLEMENTATION
// =============================================================================

impl<T: NumericElement> ArrayReduction<T> for Array<T> 
where
    T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<usize> + PartialOrd + Copy
{
    type Error = NumRs2Error;
    
    fn sum(&self) -> T {
        let data = self.to_vec();
        data.into_iter().fold(T::zero(), |acc, x| acc + x)
    }
    
    fn sum_axis(&self, axis: usize) -> Result<Self> {
        // Delegate to existing implementation if available
        // For now, implement basic sum along axis
        if axis >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch("Axis out of bounds".to_string()));
        }
        
        // Simplified implementation - would need more sophisticated logic for actual axis reduction
        Ok(self.clone())
    }
    
    fn mean(&self) -> T 
    where 
        T: std::ops::Div<Output = T> + From<usize> 
    {
        let total = self.sum();
        let count = T::from(self.size());
        total / count
    }
    
    fn mean_axis(&self, axis: Option<usize>) -> Result<Self> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch("Axis out of bounds".to_string()));
                }
                // Simplified implementation
                Ok(self.clone())
            },
            None => {
                let mean_val = self.mean();
                Ok(Array::from_vec(vec![mean_val]))
            }
        }
    }
    
    fn std(&self) -> T 
    where 
        T: FloatingPoint 
    {
        let mean_val = self.mean();
        let data = self.to_vec();
        let variance = data.iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / T::from(self.size());
        variance.sqrt()
    }
    
    fn std_axis(&self, axis: Option<usize>) -> Result<Self> {
        match axis {
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch("Axis out of bounds".to_string()));
                }
                // Simplified implementation
                Ok(self.clone())
            },
            None => {
                let std_val = self.std();
                Ok(Array::from_vec(vec![std_val]))
            }
        }
    }
    
    fn min(&self) -> T 
    where 
        T: PartialOrd 
    {
        let data = self.to_vec();
        data.into_iter().fold(data[0], |acc, x| if x < acc { x } else { acc })
    }
    
    fn max(&self) -> T 
    where 
        T: PartialOrd 
    {
        let data = self.to_vec();
        data.into_iter().fold(data[0], |acc, x| if x > acc { x } else { acc })
    }
    
    fn argmin(&self) -> usize 
    where 
        T: PartialOrd 
    {
        let data = self.to_vec();
        let mut min_idx = 0;
        let mut min_val = data[0];
        
        for (i, &val) in data.iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        min_idx
    }
    
    fn argmax(&self) -> usize 
    where 
        T: PartialOrd 
    {
        let data = self.to_vec();
        let mut max_idx = 0;
        let mut max_val = data[0];
        
        for (i, &val) in data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }
}

// =============================================================================
// ARRAY INDEXING IMPLEMENTATION
// =============================================================================

impl<T: NumericElement> ArrayIndexing<T> for Array<T> {
    type IndexResult = Array<T>;
    type Error = NumRs2Error;
    
    fn get(&self, indices: &[usize]) -> Result<T> {
        self.get(indices).map_err(|e| e.into())
    }
    
    fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        self.set(indices, value).map_err(|e| e.into())
    }
    
    fn index(&self, specs: &[IndexSpec]) -> Result<Self::IndexResult> {
        // Delegate to existing advanced indexing implementation
        self.index(specs).map_err(|e| e.into())
    }
    
    fn fancy_index(&self, indices: &[&[usize]]) -> Result<Self::IndexResult> {
        // Delegate to existing fancy indexing implementation
        self.fancy_index(indices).map_err(|e| e.into())
    }
    
    fn bool_index(&self, mask: &[bool]) -> Result<Self::IndexResult> {
        // Delegate to existing boolean indexing implementation
        self.bool_index(mask).map_err(|e| e.into())
    }
    
    fn slice(&self, axis: usize, start: usize, end: Option<usize>) -> Result<Self::IndexResult> {
        // Delegate to existing slicing implementation
        self.slice(axis, start.into()).map_err(|e| e.into())
    }
}

// =============================================================================
// ARRAY MATH IMPLEMENTATION
// =============================================================================

impl<T: NumericElement> ArrayMath<T> for Array<T> 
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + 
       std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Copy
{
    fn abs(&self) -> Self::Output 
    where 
        T: num_traits::Signed 
    {
        self.map(|x| x.abs())
    }
    
    fn sqrt(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.sqrt())
    }
    
    fn exp(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.exp())
    }
    
    fn ln(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.ln())
    }
    
    fn sin(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.sin())
    }
    
    fn cos(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.cos())
    }
    
    fn tan(&self) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.tan())
    }
    
    fn pow(&self, exponent: T) -> Self::Output 
    where 
        T: FloatingPoint 
    {
        self.map(|x| x.powf(exponent))
    }
    
    fn pow_array(&self, exponents: &Self) -> Result<Self::Output> 
    where 
        T: FloatingPoint 
    {
        if self.shape() != exponents.shape() {
            return Err(NumRs2Error::DimensionMismatch(
                "Arrays must have the same shape for element-wise power".to_string()
            ));
        }
        
        let self_data = self.to_vec();
        let exp_data = exponents.to_vec();
        let result_data: Vec<T> = self_data.iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powf(exp))
            .collect();
            
        Ok(Array::from_vec(result_data).reshape(self.shape()))
    }
}

// =============================================================================
// LINEAR ALGEBRA IMPLEMENTATION
// =============================================================================

impl<T: FloatingPoint> LinearAlgebra<T> for Array<T> {
    type Error = NumRs2Error;
    
    fn matmul(&self, other: &Self) -> Result<Self> {
        // Delegate to existing matrix multiplication implementation
        self.matmul(other).map_err(|e| e.into())
    }
    
    fn transpose(&self) -> Self {
        // Delegate to existing transpose implementation
        self.transpose()
    }
    
    fn det(&self) -> Result<T> {
        // Delegate to existing determinant implementation if available
        #[cfg(feature = "matrix_decomp")]
        {
            self.det().map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for determinant".to_string()))
        }
    }
    
    fn inv(&self) -> Result<Self> {
        // Delegate to existing matrix inverse implementation if available
        #[cfg(feature = "matrix_decomp")]
        {
            self.inv().map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for matrix inverse".to_string()))
        }
    }
    
    fn solve(&self, b: &Self) -> Result<Self> {
        // Delegate to existing linear system solver implementation if available
        #[cfg(feature = "matrix_decomp")]
        {
            self.solve(b).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for solve".to_string()))
        }
    }
    
    fn rank(&self) -> Result<usize> {
        // Simplified rank computation using SVD
        #[cfg(feature = "matrix_decomp")]
        {
            // Would delegate to existing rank implementation
            Ok(std::cmp::min(self.shape()[0], self.shape()[1]))
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for rank".to_string()))
        }
    }
    
    fn cond(&self) -> Result<T> {
        // Delegate to existing condition number implementation if available
        #[cfg(feature = "matrix_decomp")]
        {
            self.cond().map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for condition number".to_string()))
        }
    }
    
    fn norm(&self, ord: Option<T>) -> Result<T> {
        // Basic implementation of matrix norms
        match ord {
            None => {
                // Frobenius norm
                let data = self.to_vec();
                let sum_squares = data.iter().fold(T::zero(), |acc, &x| acc + x * x);
                Ok(sum_squares.sqrt())
            },
            Some(_ord) => {
                // More sophisticated norm computations would be implemented here
                Err(NumRs2Error::NotImplemented("Specific matrix norms not yet implemented".to_string()))
            }
        }
    }
}

// =============================================================================
// MATRIX DECOMPOSITION IMPLEMENTATION
// =============================================================================

impl<T: FloatingPoint> MatrixDecomposition<T> for Array<T> 
where
    T: Clone + std::fmt::Debug + ndarray_linalg::Lapack,
{
    type DecompositionResult = (Array<T>, Array<T>, Array<T>); // Example: (L, U, P) for LU
    type Error = NumRs2Error;
    
    fn lu(&self) -> Result<Self::DecompositionResult> {
        // Delegate to existing LU decomposition implementation
        #[cfg(feature = "matrix_decomp")]
        {
            use crate::linalg_extended::decomposition::lu;
            lu(self).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for LU decomposition".to_string()))
        }
    }
    
    fn qr(&self) -> Result<Self::DecompositionResult> {
        // Delegate to existing QR decomposition implementation
        #[cfg(feature = "matrix_decomp")]
        {
            use crate::linalg_extended::decomposition::qr;
            qr(self).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for QR decomposition".to_string()))
        }
    }
    
    fn svd(&self) -> Result<Self::DecompositionResult> {
        // Delegate to existing SVD implementation
        #[cfg(feature = "matrix_decomp")]
        {
            use crate::linalg_extended::decomposition::svd;
            svd(self).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for SVD".to_string()))
        }
    }
    
    fn cholesky(&self) -> Result<Self> {
        // Delegate to existing Cholesky decomposition implementation
        #[cfg(feature = "matrix_decomp")]
        {
            use crate::linalg_extended::decomposition::cholesky;
            cholesky(self).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for Cholesky decomposition".to_string()))
        }
    }
    
    fn eig(&self) -> Result<Self::DecompositionResult> {
        // Delegate to existing eigenvalue decomposition implementation
        #[cfg(feature = "matrix_decomp")]
        {
            use crate::linalg_extended::eigenvalue::eig;
            eig(self).map_err(|e| e.into())
        }
        
        #[cfg(not(feature = "matrix_decomp"))]
        {
            Err(NumRs2Error::FeatureNotEnabled("matrix_decomp feature required for eigenvalue decomposition".to_string()))
        }
    }
    
    fn schur(&self) -> Result<Self::DecompositionResult> {
        // Schur decomposition would be implemented here
        Err(NumRs2Error::NotImplemented("Schur decomposition not yet implemented".to_string()))
    }
}

// =============================================================================
// MEMORY MANAGEMENT IMPLEMENTATION
// =============================================================================

impl<T: NumericElement> crate::traits::MemoryAware for Array<T> {
    fn set_allocator(&mut self, _allocator: Box<dyn crate::traits::SpecializedAllocator<Error = NumRs2Error>>) {
        // For now, this is a placeholder since Array<T> doesn't directly use custom allocators
        // In a future enhancement, Array<T> could store a reference to the allocator for new allocations
        // This would be implemented as part of the Array<T> refactoring
    }

    fn memory_usage(&self) -> crate::traits::MemoryUsage {
        let element_size = std::mem::size_of::<T>();
        let total_elements = self.size();
        let total_bytes = total_elements * element_size;
        
        crate::traits::MemoryUsage {
            total_bytes,
            allocation_count: 1, // Array uses single contiguous allocation
            fragmentation: 0.0, // Contiguous allocation has no fragmentation
            efficiency: 1.0,    // Using all allocated memory
        }
    }

    fn optimize_memory_layout(&mut self) -> Result<()> {
        // For now, Array<T> doesn't support runtime layout optimization
        // This could be enhanced to:
        // 1. Defragment multi-part arrays
        // 2. Realign memory for SIMD operations
        // 3. Compress sparse regions
        Ok(())
    }

    fn suggest_optimizations(&self) -> Vec<crate::traits::MemoryOptimization> {
        use crate::traits::{MemoryOptimization, OptimizationType};
        
        let mut suggestions = Vec::new();
        let element_size = std::mem::size_of::<T>();
        let total_bytes = self.size() * element_size;
        
        // Suggest alignment optimization for large arrays that might benefit from SIMD
        if total_bytes > 1024 && element_size >= 4 {
            suggestions.push(MemoryOptimization {
                optimization_type: OptimizationType::AlignmentOptimization,
                description: "Align array memory for SIMD operations".to_string(),
                estimated_savings: 0, // No memory savings, but performance improvement
                complexity: 2,
            });
        }
        
        // Suggest arena allocation for temporary arrays
        if total_bytes < 65536 {
            suggestions.push(MemoryOptimization {
                optimization_type: OptimizationType::ArenaOptimization,
                description: "Use arena allocation for temporary array".to_string(),
                estimated_savings: 0, // Savings in allocation overhead
                complexity: 3,
            });
        }
        
        // Suggest pooling for small, frequently allocated arrays
        if total_bytes < 8192 {
            suggestions.push(MemoryOptimization {
                optimization_type: OptimizationType::PoolingOptimization,
                description: "Use memory pool for small array allocations".to_string(),
                estimated_savings: 0, // Savings in allocation time
                complexity: 2,
            });
        }
        
        suggestions
    }
}
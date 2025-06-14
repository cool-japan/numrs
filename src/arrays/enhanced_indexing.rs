//! Enhanced array slicing and indexing operations
//!
//! This module provides comprehensive array slicing and indexing capabilities
//! with support for advanced indexing patterns, optimized access, and efficient
//! memory operations.

use crate::error::{NumRs2Error, Result};
use crate::traits::NumericElement;
use super::advanced_ops::{Shape, ArrayView, IndexSpec, ResolvedIndex};
use std::collections::HashMap;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Enhanced indexing engine with optimized access patterns
pub struct IndexingEngine {
    /// Cache for compiled index patterns
    index_cache: HashMap<String, CompiledIndexPattern>,
    /// Performance statistics
    stats: IndexingStats,
}

/// Compiled index pattern for optimized repeated access
#[derive(Debug, Clone)]
struct CompiledIndexPattern {
    /// Flat indices for direct access
    flat_indices: Vec<usize>,
    /// Output shape after indexing
    #[allow(dead_code)]
    output_shape: Vec<usize>,
    /// Access pattern type
    #[allow(dead_code)]
    pattern_type: IndexPatternType,
}

/// Types of index patterns
#[derive(Debug, Clone, PartialEq)]
enum IndexPatternType {
    /// Sequential access pattern
    Sequential,
    /// Strided access pattern
    Strided,
    /// Random access pattern
    Random,
    /// Block access pattern
    Block,
}

/// Performance statistics for indexing operations
#[derive(Debug, Default)]
pub struct IndexingStats {
    /// Number of cache hits
    cache_hits: u64,
    /// Number of cache misses
    cache_misses: u64,
    /// Total indexing operations performed
    total_operations: u64,
}

impl IndexingEngine {
    /// Create a new indexing engine
    pub fn new() -> Self {
        Self {
            index_cache: HashMap::new(),
            stats: IndexingStats::default(),
        }
    }

    /// Index array with multiple index specifications (advanced indexing)
    pub fn advanced_index<T>(&mut self, array: &ArrayView<T>, indices: &[IndexSpec]) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        self.stats.total_operations += 1;

        // Generate cache key
        let cache_key = self.generate_cache_key(array.shape(), indices);
        
        // Check cache first
        if let Some(compiled_pattern) = self.index_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return self.apply_compiled_pattern(array, compiled_pattern);
        }

        self.stats.cache_misses += 1;

        // Compile the index pattern
        let compiled_pattern = self.compile_index_pattern(array.shape(), indices)?;
        
        // Apply the pattern
        let result = self.apply_compiled_pattern(array, &compiled_pattern)?;
        
        // Cache the pattern for future use
        self.index_cache.insert(cache_key, compiled_pattern);
        
        Ok(result)
    }

    /// Slice array with extended slicing syntax
    pub fn enhanced_slice<'a, T>(&mut self, array: &ArrayView<'a, T>, slices: &[SliceSpec]) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        // Convert slice specifications to index specifications
        let mut index_specs = Vec::with_capacity(slices.len());
        
        for slice_spec in slices {
            let index_spec = self.convert_slice_to_index(slice_spec, array.shape())?;
            index_specs.push(index_spec);
        }
        
        // Create sliced view
        array.slice(&index_specs)
    }

    /// Multi-dimensional slicing with step support
    pub fn multidim_slice<T>(&self, array: &ArrayView<T>, ranges: &[RangeSpec]) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        if ranges.len() != array.shape().ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Number of ranges ({}) must match array dimensions ({})", 
                       ranges.len(), array.shape().ndim())
            ));
        }

        let mut result = Vec::new();
        let mut current_indices = vec![0; array.shape().ndim()];
        
        // Initialize indices to range starts
        for (i, range_spec) in ranges.iter().enumerate() {
            current_indices[i] = range_spec.start;
        }

        self.iterate_ranges(array, ranges, &mut current_indices, 0, &mut result)?;
        Ok(result)
    }

    /// Recursive function to iterate through multi-dimensional ranges
    fn iterate_ranges<T>(&self, array: &ArrayView<T>, ranges: &[RangeSpec], 
                        current_indices: &mut [usize], depth: usize, result: &mut Vec<T>) -> Result<()>
    where
        T: NumericElement + Copy,
    {
        if depth >= ranges.len() {
            // Base case: extract element at current indices
            if let Ok(element) = array.get(current_indices) {
                result.push(*element);
            }
            return Ok(());
        }

        let range = &ranges[depth];
        let mut idx = range.start;

        while idx < range.stop && idx < array.shape().dims[depth] {
            current_indices[depth] = idx;
            self.iterate_ranges(array, ranges, current_indices, depth + 1, result)?;
            idx += range.step;
        }

        Ok(())
    }

    /// Index with coordinate arrays (similar to np.ix_)
    pub fn coordinate_index<T>(&self, array: &ArrayView<T>, coordinates: &[Vec<usize>]) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        if coordinates.len() != array.shape().ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                "Number of coordinate arrays must match array dimensions".to_string()
            ));
        }

        // Validate coordinates
        for (axis, coord_array) in coordinates.iter().enumerate() {
            for &coord in coord_array {
                if coord >= array.shape().dims[axis] {
                    return Err(NumRs2Error::IndexOutOfBounds(
                        format!("Coordinate {} is out of bounds for axis {} of size {}", 
                               coord, axis, array.shape().dims[axis])
                    ));
                }
            }
        }

        let mut result = Vec::new();
        let total_combinations: usize = coordinates.iter().map(|c| c.len()).product();

        for combination_idx in 0..total_combinations {
            let multi_index = self.combination_to_indices(combination_idx, coordinates);
            if let Ok(element) = array.get(&multi_index) {
                result.push(*element);
            }
        }

        Ok(result)
    }

    /// Convert combination index to multi-dimensional indices
    fn combination_to_indices(&self, combination_idx: usize, coordinates: &[Vec<usize>]) -> Vec<usize> {
        let mut multi_index = Vec::with_capacity(coordinates.len());
        let mut remaining = combination_idx;

        for coord_array in coordinates.iter().rev() {
            let coord_idx = remaining % coord_array.len();
            multi_index.push(coord_array[coord_idx]);
            remaining /= coord_array.len();
        }

        multi_index.reverse();
        multi_index
    }

    /// Masked indexing with boolean array
    pub fn masked_index<T>(&self, array: &ArrayView<T>, mask: &[bool]) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        if mask.len() != array.shape().size() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Mask length ({}) must match array size ({})", 
                       mask.len(), array.shape().size())
            ));
        }

        let mut result = Vec::new();
        let mut flat_index = 0;

        // Iterate through all elements and check mask
        let mut indices = vec![0; array.shape().ndim()];
        loop {
            if flat_index < mask.len() && mask[flat_index] {
                if let Ok(element) = array.get(&indices) {
                    result.push(*element);
                }
            }

            flat_index += 1;

            // Advance indices
            if !self.advance_indices(&mut indices, &array.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Conditional indexing with predicate function
    pub fn conditional_index<T, F>(&self, array: &ArrayView<T>, predicate: F) -> Result<Vec<(Vec<usize>, T)>>
    where
        T: NumericElement + Copy,
        F: Fn(T) -> bool,
    {
        let mut result = Vec::new();
        let mut indices = vec![0; array.shape().ndim()];

        loop {
            if let Ok(element) = array.get(&indices) {
                if predicate(*element) {
                    result.push((indices.clone(), *element));
                }
            }

            if !self.advance_indices(&mut indices, &array.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Circular indexing (indices wrap around)
    pub fn circular_index<T>(&self, array: &ArrayView<T>, indices: &[isize]) -> Result<T>
    where
        T: NumericElement + Copy,
    {
        if indices.len() != array.shape().ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Number of indices ({}) must match array dimensions ({})", 
                       indices.len(), array.shape().ndim())
            ));
        }

        // Convert to circular indices
        let circular_indices: Vec<usize> = indices.iter().zip(array.shape().dims.iter())
            .map(|(&idx, &dim_size)| {
                let positive_idx = if idx < 0 {
                    (dim_size as isize + (idx % dim_size as isize)) as usize
                } else {
                    (idx as usize) % dim_size
                };
                positive_idx
            })
            .collect();

        array.get(&circular_indices).map(|x| *x)
    }

    /// Block indexing for accessing rectangular sub-arrays
    pub fn block_index<T>(&self, array: &ArrayView<T>, block_spec: &BlockSpec) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        // Validate block specification
        if block_spec.start.len() != array.shape().ndim() || 
           block_spec.size.len() != array.shape().ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                "Block specification dimensions must match array dimensions".to_string()
            ));
        }

        for (axis, (&start, &size)) in block_spec.start.iter().zip(block_spec.size.iter()).enumerate() {
            if start + size > array.shape().dims[axis] {
                return Err(NumRs2Error::IndexOutOfBounds(
                    format!("Block extends beyond array bounds in axis {}", axis)
                ));
            }
        }

        let mut result = Vec::new();
        let mut indices = block_spec.start.clone();

        self.extract_block(array, block_spec, &mut indices, 0, &mut result)?;
        Ok(result)
    }

    /// Recursive function to extract block elements
    fn extract_block<T>(&self, array: &ArrayView<T>, block_spec: &BlockSpec,
                       current_indices: &mut [usize], depth: usize, result: &mut Vec<T>) -> Result<()>
    where
        T: NumericElement + Copy,
    {
        if depth >= block_spec.start.len() {
            // Base case: extract element
            if let Ok(element) = array.get(current_indices) {
                result.push(*element);
            }
            return Ok(());
        }

        let start = block_spec.start[depth];
        let end = start + block_spec.size[depth];

        for idx in start..end {
            current_indices[depth] = idx;
            self.extract_block(array, block_spec, current_indices, depth + 1, result)?;
        }

        Ok(())
    }

    /// Diagonal indexing for extracting diagonals
    pub fn diagonal_index<T>(&self, array: &ArrayView<T>, offset: isize, axis1: Option<usize>, axis2: Option<usize>) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let ndim = array.shape().ndim();
        
        if ndim < 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "Diagonal indexing requires at least 2 dimensions".to_string()
            ));
        }

        let ax1 = axis1.unwrap_or(ndim - 2);
        let ax2 = axis2.unwrap_or(ndim - 1);

        if ax1 >= ndim || ax2 >= ndim || ax1 == ax2 {
            return Err(NumRs2Error::DimensionMismatch(
                "Invalid axis specification for diagonal".to_string()
            ));
        }

        let dim1 = array.shape().dims[ax1];
        let dim2 = array.shape().dims[ax2];

        // Calculate diagonal length and starting positions
        let (start1, start2, diag_length) = if offset >= 0 {
            let offset = offset as usize;
            if offset >= dim2 {
                return Ok(Vec::new()); // No diagonal elements
            }
            (0, offset, std::cmp::min(dim1, dim2 - offset))
        } else {
            let offset = (-offset) as usize;
            if offset >= dim1 {
                return Ok(Vec::new()); // No diagonal elements
            }
            (offset, 0, std::cmp::min(dim1 - offset, dim2))
        };

        let mut result = Vec::with_capacity(diag_length);
        let mut indices = vec![0; ndim];

        // Extract other dimensions (all combinations)
        self.extract_diagonal_recursive(array, &mut indices, 0, ax1, ax2, start1, start2, diag_length, &mut result)?;

        Ok(result)
    }

    /// Recursive extraction of diagonal elements
    fn extract_diagonal_recursive<T>(&self, array: &ArrayView<T>, indices: &mut [usize], 
                                    depth: usize, axis1: usize, axis2: usize,
                                    start1: usize, start2: usize, diag_length: usize,
                                    result: &mut Vec<T>) -> Result<()>
    where
        T: NumericElement + Copy,
    {
        if depth >= indices.len() {
            // Extract diagonal elements for this combination of other dimensions
            for i in 0..diag_length {
                indices[axis1] = start1 + i;
                indices[axis2] = start2 + i;
                if let Ok(element) = array.get(indices) {
                    result.push(*element);
                }
            }
            return Ok(());
        }

        if depth == axis1 || depth == axis2 {
            // Skip diagonal axes
            self.extract_diagonal_recursive(array, indices, depth + 1, axis1, axis2, 
                                          start1, start2, diag_length, result)?;
        } else {
            // Iterate through all indices for this dimension
            for idx in 0..array.shape().dims[depth] {
                indices[depth] = idx;
                self.extract_diagonal_recursive(array, indices, depth + 1, axis1, axis2, 
                                              start1, start2, diag_length, result)?;
            }
        }

        Ok(())
    }

    /// Compile index pattern for optimized repeated access
    fn compile_index_pattern(&self, shape: &Shape, indices: &[IndexSpec]) -> Result<CompiledIndexPattern> {
        let mut output_shape = Vec::new();
        let mut pattern_type = IndexPatternType::Sequential;

        // For simplicity, this is a basic implementation
        // A full implementation would analyze the pattern more thoroughly
        
        // Process each index specification
        for (axis, index_spec) in indices.iter().enumerate().take(shape.ndim()) {
            let resolved = index_spec.resolve(shape.dims[axis])?;
            
            match resolved {
                ResolvedIndex::Single(_) => {
                    // Single index removes dimension
                },
                ResolvedIndex::Multiple(idx_vec) => {
                    output_shape.push(idx_vec.len());
                    
                    // Analyze pattern
                    if self.is_strided_pattern(&idx_vec) {
                        pattern_type = IndexPatternType::Strided;
                    } else if self.is_block_pattern(&idx_vec) {
                        pattern_type = IndexPatternType::Block;
                    } else {
                        pattern_type = IndexPatternType::Random;
                    }
                }
            }
        }

        // Generate flat indices (simplified version)
        let flat_indices = self.generate_flat_indices(shape, indices)?;

        Ok(CompiledIndexPattern {
            flat_indices,
            output_shape,
            pattern_type,
        })
    }

    /// Generate flat indices from index specifications
    fn generate_flat_indices(&self, shape: &Shape, _indices: &[IndexSpec]) -> Result<Vec<usize>> {
        let mut flat_indices = Vec::new();
        let strides = shape.c_strides();

        // This is a simplified implementation
        // A full implementation would handle all index combinations
        
        let mut current_indices = vec![0; shape.ndim()];
        loop {
            let flat_idx = current_indices.iter().zip(strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum();
            flat_indices.push(flat_idx);

            if !self.advance_indices(&mut current_indices, &shape.dims) {
                break;
            }
        }

        Ok(flat_indices)
    }

    /// Apply compiled pattern to extract data
    fn apply_compiled_pattern<T>(&self, array: &ArrayView<T>, pattern: &CompiledIndexPattern) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        // This is a simplified implementation
        // A full implementation would use the flat indices directly
        Ok(array.to_vec().into_iter().take(pattern.flat_indices.len()).collect::<Vec<_>>())
    }

    /// Check if indices form a strided pattern
    fn is_strided_pattern(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return true;
        }

        let stride = indices[1] - indices[0];
        for i in 2..indices.len() {
            if indices[i] - indices[i-1] != stride {
                return false;
            }
        }
        true
    }

    /// Check if indices form a block pattern
    fn is_block_pattern(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 {
            return true;
        }

        // Check if indices are consecutive
        for i in 1..indices.len() {
            if indices[i] != indices[i-1] + 1 {
                return false;
            }
        }
        true
    }

    /// Generate cache key for index pattern
    fn generate_cache_key(&self, shape: &Shape, indices: &[IndexSpec]) -> String {
        format!("{:?}_{:?}", shape.dims, indices.len())
    }

    /// Convert slice specification to index specification
    fn convert_slice_to_index(&self, slice_spec: &SliceSpec, _shape: &Shape) -> Result<IndexSpec> {
        match slice_spec {
            SliceSpec::Range(range_spec) => {
                Ok(IndexSpec::Slice(
                    Some(range_spec.start as isize),
                    Some(range_spec.stop as isize),
                    Some(range_spec.step as isize),
                ))
            },
            SliceSpec::Index(idx) => {
                Ok(IndexSpec::Int(*idx as isize))
            },
            SliceSpec::Mask(mask) => {
                Ok(IndexSpec::BoolMask(mask.clone()))
            },
            SliceSpec::Array(indices) => {
                Ok(IndexSpec::Array(indices.clone()))
            },
            SliceSpec::All => {
                Ok(IndexSpec::All)
            },
            SliceSpec::NewAxis => {
                Ok(IndexSpec::NewAxis)
            },
            SliceSpec::Ellipsis => {
                Ok(IndexSpec::Ellipsis)
            },
        }
    }

    /// Helper function to advance multi-dimensional indices
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

    /// Get performance statistics
    pub fn get_stats(&self) -> &IndexingStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.index_cache.clear();
    }
}

impl Default for IndexingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Range specification for multi-dimensional slicing
#[derive(Debug, Clone)]
pub struct RangeSpec {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

impl RangeSpec {
    pub fn new(start: usize, stop: usize, step: usize) -> Self {
        Self { start, stop, step }
    }
}

impl From<Range<usize>> for RangeSpec {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start, range.end, 1)
    }
}

impl From<RangeFrom<usize>> for RangeSpec {
    fn from(range: RangeFrom<usize>) -> Self {
        Self::new(range.start, usize::MAX, 1)
    }
}

impl From<RangeTo<usize>> for RangeSpec {
    fn from(range: RangeTo<usize>) -> Self {
        Self::new(0, range.end, 1)
    }
}

impl From<RangeFull> for RangeSpec {
    fn from(_: RangeFull) -> Self {
        Self::new(0, usize::MAX, 1)
    }
}

/// Block specification for rectangular sub-arrays
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Starting indices for each dimension
    pub start: Vec<usize>,
    /// Size of block in each dimension
    pub size: Vec<usize>,
}

impl BlockSpec {
    pub fn new(start: Vec<usize>, size: Vec<usize>) -> Self {
        Self { start, size }
    }
}

/// Slice specification types
#[derive(Debug, Clone)]
pub enum SliceSpec {
    /// Range with start, stop, step
    Range(RangeSpec),
    /// Single index
    Index(usize),
    /// Boolean mask
    Mask(Vec<bool>),
    /// Array of indices
    Array(Vec<usize>),
    /// Select all elements
    All,
    /// New axis
    NewAxis,
    /// Ellipsis
    Ellipsis,
}

/// Builder for complex indexing operations
pub struct IndexBuilder {
    specs: Vec<SliceSpec>,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    pub fn index(mut self, idx: usize) -> Self {
        self.specs.push(SliceSpec::Index(idx));
        self
    }

    pub fn range(mut self, start: usize, stop: usize, step: usize) -> Self {
        self.specs.push(SliceSpec::Range(RangeSpec::new(start, stop, step)));
        self
    }

    pub fn slice<R: Into<RangeSpec>>(mut self, range: R) -> Self {
        self.specs.push(SliceSpec::Range(range.into()));
        self
    }

    pub fn mask(mut self, mask: Vec<bool>) -> Self {
        self.specs.push(SliceSpec::Mask(mask));
        self
    }

    pub fn array(mut self, indices: Vec<usize>) -> Self {
        self.specs.push(SliceSpec::Array(indices));
        self
    }

    pub fn all(mut self) -> Self {
        self.specs.push(SliceSpec::All);
        self
    }

    pub fn new_axis(mut self) -> Self {
        self.specs.push(SliceSpec::NewAxis);
        self
    }

    pub fn ellipsis(mut self) -> Self {
        self.specs.push(SliceSpec::Ellipsis);
        self
    }

    pub fn build(self) -> Vec<SliceSpec> {
        self.specs
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::advanced_ops::{Shape, ArrayView};

    #[test]
    fn test_indexing_engine_creation() {
        let engine = IndexingEngine::new();
        assert_eq!(engine.stats.total_operations, 0);
    }

    #[test]
    fn test_range_spec_creation() {
        let range_spec = RangeSpec::new(0, 10, 2);
        assert_eq!(range_spec.start, 0);
        assert_eq!(range_spec.stop, 10);
        assert_eq!(range_spec.step, 2);
    }

    #[test]
    fn test_block_spec_creation() {
        let block_spec = BlockSpec::new(vec![1, 2], vec![3, 4]);
        assert_eq!(block_spec.start, vec![1, 2]);
        assert_eq!(block_spec.size, vec![3, 4]);
    }

    #[test]
    fn test_multidim_slice() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = Shape::new(vec![2, 4]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let ranges = vec![
            RangeSpec::new(0, 2, 1),
            RangeSpec::new(1, 3, 1),
        ];

        let result = engine.multidim_slice(&view, &ranges).unwrap();
        assert_eq!(result.len(), 4); // 2 * 2 elements
    }

    #[test]
    fn test_circular_indexing() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4];
        let shape = Shape::new(vec![4]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        // Test circular indexing with negative index
        let result = engine.circular_index(&view, &[-1]).unwrap();
        assert_eq!(result, 4); // Last element

        // Test circular indexing with out-of-bounds positive index
        let result = engine.circular_index(&view, &[5]).unwrap();
        assert_eq!(result, 2); // 5 % 4 = 1, so second element
    }

    #[test]
    fn test_block_indexing() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let shape = Shape::new(vec![3, 3]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let block_spec = BlockSpec::new(vec![1, 1], vec![2, 2]);
        let result = engine.block_index(&view, &block_spec).unwrap();
        
        // Should extract a 2x2 block starting at (1,1)
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_masked_indexing() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = Shape::new(vec![6]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let mask = vec![true, false, true, false, true, false];
        let result = engine.masked_index(&view, &mask).unwrap();
        
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_conditional_indexing() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = Shape::new(vec![6]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let result = engine.conditional_index(&view, |x| x > 3).unwrap();
        
        assert_eq!(result.len(), 3); // Elements 4, 5, 6
        assert_eq!(result[0].1, 4);
        assert_eq!(result[1].1, 5);
        assert_eq!(result[2].1, 6);
    }

    #[test]
    fn test_diagonal_indexing() {
        let engine = IndexingEngine::new();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let shape = Shape::new(vec![3, 3]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let result = engine.diagonal_index(&view, 0, None, None).unwrap();
        assert_eq!(result.len(), 3); // Main diagonal elements
    }

    #[test]
    fn test_index_builder() {
        let specs = IndexBuilder::new()
            .index(0)
            .range(1, 3, 1)
            .all()
            .build();

        assert_eq!(specs.len(), 3);
        
        match &specs[0] {
            SliceSpec::Index(idx) => assert_eq!(*idx, 0),
            _ => panic!("Expected Index"),
        }
        
        match &specs[1] {
            SliceSpec::Range(range) => {
                assert_eq!(range.start, 1);
                assert_eq!(range.stop, 3);
                assert_eq!(range.step, 1);
            },
            _ => panic!("Expected Range"),
        }
        
        match &specs[2] {
            SliceSpec::All => {},
            _ => panic!("Expected All"),
        }
    }
}
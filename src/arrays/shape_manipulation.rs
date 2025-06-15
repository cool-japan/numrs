//! Enhanced shape manipulation and multidimensional views
//!
//! This module provides comprehensive shape manipulation capabilities including
//! advanced reshaping, view system, stride calculations, and layout optimization.

use super::advanced_ops::{ArrayView, Shape};
use crate::error::{NumRs2Error, Result};
use crate::traits::NumericElement;
use std::collections::HashMap;

/// Memory layout for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// C-style contiguous (row-major)
    C,
    /// Fortran-style contiguous (column-major)
    Fortran,
    /// Custom stride pattern
    Custom,
    /// Non-contiguous layout
    Strided,
}

/// Advanced shape manipulation engine
pub struct ShapeEngine {
    /// Cache for computed strides
    stride_cache: HashMap<(Vec<usize>, MemoryLayout), Vec<usize>>,
}

impl ShapeEngine {
    /// Create a new shape engine
    pub fn new() -> Self {
        Self {
            stride_cache: HashMap::new(),
        }
    }

    /// Compute optimal strides for a given shape and layout
    pub fn compute_strides(&mut self, shape: &[usize], layout: MemoryLayout) -> Vec<usize> {
        let cache_key = (shape.to_vec(), layout);

        if let Some(cached_strides) = self.stride_cache.get(&cache_key) {
            return cached_strides.clone();
        }

        let strides = match layout {
            MemoryLayout::C => self.compute_c_strides(shape),
            MemoryLayout::Fortran => self.compute_fortran_strides(shape),
            MemoryLayout::Custom => self.compute_optimal_strides(shape),
            MemoryLayout::Strided => self.compute_default_strides(shape),
        };

        self.stride_cache.insert(cache_key, strides.clone());
        strides
    }

    /// Compute C-style (row-major) strides
    fn compute_c_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    /// Compute Fortran-style (column-major) strides
    fn compute_fortran_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in 1..shape.len() {
            strides[i] = strides[i - 1] * shape[i - 1];
        }

        strides
    }

    /// Compute optimal strides for cache efficiency
    fn compute_optimal_strides(&self, shape: &[usize]) -> Vec<usize> {
        // For optimal cache performance, order dimensions by size (smallest stride for largest dimension)
        let mut dim_sizes: Vec<(usize, usize)> = shape
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();
        dim_sizes.sort_by_key(|&(_, size)| std::cmp::Reverse(size));

        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;

        for &(dim_idx, dim_size) in &dim_sizes {
            strides[dim_idx] = current_stride;
            current_stride *= dim_size;
        }

        strides
    }

    /// Compute default strides (same as C-style)
    fn compute_default_strides(&self, shape: &[usize]) -> Vec<usize> {
        self.compute_c_strides(shape)
    }

    /// Check if a reshape operation is valid
    pub fn can_reshape(&self, current_shape: &[usize], new_shape: &[usize]) -> bool {
        let current_size: usize = current_shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        current_size == new_size
    }

    /// Create a reshaped view if possible
    pub fn reshape_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        new_shape: &[usize],
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        if !self.can_reshape(&view.shape().dims, new_shape) {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Cannot reshape array from {:?} to {:?}: incompatible sizes",
                view.shape().dims,
                new_shape
            )));
        }

        // Check if the view is C-contiguous (required for simple reshape)
        if !view.is_c_contiguous() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot reshape non-contiguous view. Use copy() first.".to_string(),
            ));
        }

        let new_shape_obj = Shape::new(new_shape.to_vec());
        view.reshape(new_shape_obj)
    }

    /// Transpose an array view with specified axes
    pub fn transpose_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axes: Option<Vec<usize>>,
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        view.transpose(axes)
    }

    /// Create a view with swapped axes
    pub fn swapaxes_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axis1: usize,
        axis2: usize,
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        let ndim = view.shape().ndim();

        if axis1 >= ndim || axis2 >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axes {} and {} are out of bounds for array of dimension {}",
                axis1, axis2, ndim
            )));
        }

        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(axis1, axis2);

        self.transpose_view(view, Some(axes))
    }

    /// Move axis to a new position
    pub fn moveaxis_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        source: usize,
        destination: usize,
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        let ndim = view.shape().ndim();

        if source >= ndim || destination >= ndim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Source axis {} or destination axis {} is out of bounds for array of dimension {}",
                source, destination, ndim
            )));
        }

        let mut axes: Vec<usize> = (0..ndim).collect();
        let removed = axes.remove(source);
        axes.insert(destination, removed);

        self.transpose_view(view, Some(axes))
    }

    /// Roll array elements along a given axis
    pub fn roll_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        shift: isize,
        axis: Option<usize>,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        match axis {
            Some(ax) => self.roll_along_axis(view, shift, ax),
            None => self.roll_flattened(view, shift),
        }
    }

    /// Roll elements along a specific axis
    fn roll_along_axis<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        shift: isize,
        axis: usize,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        if axis >= view.shape().ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Axis {} is out of bounds for array of dimension {}",
                axis,
                view.shape().ndim()
            )));
        }

        let axis_size = view.shape().dims[axis];
        let effective_shift =
            ((shift % axis_size as isize) + axis_size as isize) as usize % axis_size;

        let mut result = Vec::with_capacity(view.shape().size());
        let mut indices = vec![0; view.shape().ndim()];

        loop {
            // Calculate rolled index for the specified axis
            let original_axis_idx = indices[axis];
            let rolled_axis_idx = (original_axis_idx + effective_shift) % axis_size;

            // Create rolled indices
            let mut rolled_indices = indices.clone();
            rolled_indices[axis] = rolled_axis_idx;

            // Get element at rolled position
            if let Ok(element) = view.get(&rolled_indices) {
                result.push(*element);
            }

            // Advance indices
            if !self.advance_indices(&mut indices, &view.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Roll elements in flattened array
    fn roll_flattened<'a, T>(&self, view: &ArrayView<'a, T>, shift: isize) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let flat_data = view.to_vec();
        let size = flat_data.len();

        if size == 0 {
            return Ok(flat_data);
        }

        let effective_shift = ((shift % size as isize) + size as isize) as usize % size;
        let mut result = Vec::with_capacity(size);

        // Roll the flattened array
        for i in 0..size {
            let src_idx = (i + size - effective_shift) % size;
            result.push(flat_data[src_idx]);
        }

        Ok(result)
    }

    /// Flip array along specified axes
    pub fn flip_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axes: Option<Vec<usize>>,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let axes_to_flip = match axes {
            Some(ax) => ax,
            None => (0..view.shape().ndim()).collect(),
        };

        // Validate axes
        for &axis in &axes_to_flip {
            if axis >= view.shape().ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} is out of bounds for array of dimension {}",
                    axis,
                    view.shape().ndim()
                )));
            }
        }

        let mut result = Vec::with_capacity(view.shape().size());
        let mut indices = vec![0; view.shape().ndim()];

        loop {
            // Create flipped indices
            let mut flipped_indices = indices.clone();
            for &axis in &axes_to_flip {
                flipped_indices[axis] = view.shape().dims[axis] - 1 - indices[axis];
            }

            // Get element at flipped position
            if let Ok(element) = view.get(&flipped_indices) {
                result.push(*element);
            }

            // Advance indices
            if !self.advance_indices(&mut indices, &view.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Rotate array by 90 degrees
    pub fn rot90_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        k: i32,
        axes: Option<(usize, usize)>,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        if view.shape().ndim() < 2 {
            return Err(NumRs2Error::DimensionMismatch(
                "rot90 requires at least 2 dimensions".to_string(),
            ));
        }

        let (axis1, axis2) = axes.unwrap_or((0, 1));

        if axis1 >= view.shape().ndim() || axis2 >= view.shape().ndim() || axis1 == axis2 {
            return Err(NumRs2Error::DimensionMismatch(
                "Invalid rotation axes".to_string(),
            ));
        }

        // Normalize k to range [0, 4)
        let k_norm = ((k % 4) + 4) % 4;

        match k_norm {
            0 => Ok(view.to_vec()), // No rotation
            1 => self.rotate_90_once(view, axis1, axis2),
            2 => self.rotate_180(view, axis1, axis2),
            3 => self.rotate_270(view, axis1, axis2),
            _ => unreachable!(),
        }
    }

    /// Rotate by 90 degrees once
    fn rotate_90_once<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axis1: usize,
        axis2: usize,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let mut result = Vec::with_capacity(view.shape().size());
        let mut indices = vec![0; view.shape().ndim()];

        loop {
            // For 90-degree rotation: (i, j) -> (j, -i-1) = (j, rows-1-i)
            let mut rotated_indices = indices.clone();
            let old_i = indices[axis1];
            let old_j = indices[axis2];

            rotated_indices[axis1] = old_j;
            rotated_indices[axis2] = view.shape().dims[axis1] - 1 - old_i;

            if let Ok(element) = view.get(&rotated_indices) {
                result.push(*element);
            }

            if !self.advance_indices(&mut indices, &view.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Rotate by 180 degrees
    fn rotate_180<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axis1: usize,
        axis2: usize,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let mut result = Vec::with_capacity(view.shape().size());
        let mut indices = vec![0; view.shape().ndim()];

        loop {
            // For 180-degree rotation: (i, j) -> (-i-1, -j-1) = (rows-1-i, cols-1-j)
            let mut rotated_indices = indices.clone();
            rotated_indices[axis1] = view.shape().dims[axis1] - 1 - indices[axis1];
            rotated_indices[axis2] = view.shape().dims[axis2] - 1 - indices[axis2];

            if let Ok(element) = view.get(&rotated_indices) {
                result.push(*element);
            }

            if !self.advance_indices(&mut indices, &view.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Rotate by 270 degrees (or -90 degrees)
    fn rotate_270<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axis1: usize,
        axis2: usize,
    ) -> Result<Vec<T>>
    where
        T: NumericElement + Copy,
    {
        let mut result = Vec::with_capacity(view.shape().size());
        let mut indices = vec![0; view.shape().ndim()];

        loop {
            // For 270-degree rotation: (i, j) -> (-j-1, i) = (cols-1-j, i)
            let mut rotated_indices = indices.clone();
            let old_i = indices[axis1];
            let old_j = indices[axis2];

            rotated_indices[axis1] = view.shape().dims[axis2] - 1 - old_j;
            rotated_indices[axis2] = old_i;

            if let Ok(element) = view.get(&rotated_indices) {
                result.push(*element);
            }

            if !self.advance_indices(&mut indices, &view.shape().dims) {
                break;
            }
        }

        Ok(result)
    }

    /// Squeeze array dimensions (remove dimensions of size 1)
    pub fn squeeze_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axes: Option<Vec<usize>>,
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        let axes_to_squeeze = match axes {
            Some(ax) => {
                // Validate specified axes
                for &axis in &ax {
                    if axis >= view.shape().ndim() {
                        return Err(NumRs2Error::DimensionMismatch(format!(
                            "Axis {} is out of bounds",
                            axis
                        )));
                    }
                    if view.shape().dims[axis] != 1 {
                        return Err(NumRs2Error::DimensionMismatch(format!(
                            "Cannot squeeze axis {} with size {}",
                            axis,
                            view.shape().dims[axis]
                        )));
                    }
                }
                ax
            }
            None => {
                // Find all axes with size 1
                view.shape()
                    .dims
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &size)| if size == 1 { Some(i) } else { None })
                    .collect()
            }
        };

        // Create new shape by removing squeezed dimensions
        let new_dims: Vec<usize> = view
            .shape()
            .dims
            .iter()
            .enumerate()
            .filter_map(|(i, &size)| {
                if axes_to_squeeze.contains(&i) {
                    None
                } else {
                    Some(size)
                }
            })
            .collect();

        if new_dims.is_empty() {
            // If all dimensions are squeezed, return scalar (1D array with one element)
            let new_shape = Shape::new(vec![1]);
            view.reshape(new_shape)
        } else {
            let new_shape = Shape::new(new_dims);
            view.reshape(new_shape)
        }
    }

    /// Expand array dimensions (add dimensions of size 1)
    pub fn expand_dims_view<'a, T>(
        &self,
        view: &ArrayView<'a, T>,
        axes: Vec<usize>,
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        let mut new_dims = view.shape().dims.clone();
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_unstable();

        // Validate axes
        for &axis in &sorted_axes {
            if axis > new_dims.len() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} is out of bounds for expansion",
                    axis
                )));
            }
        }

        // Insert new dimensions in reverse order to maintain correct indices
        for &axis in sorted_axes.iter().rev() {
            new_dims.insert(axis, 1);
        }

        let new_shape = Shape::new(new_dims);
        view.reshape(new_shape)
    }

    /// Check if array is broadcastable to a target shape
    pub fn is_broadcastable(&self, source_shape: &[usize], target_shape: &[usize]) -> bool {
        let max_ndim = std::cmp::max(source_shape.len(), target_shape.len());

        for i in 0..max_ndim {
            let src_dim = if i < source_shape.len() {
                source_shape[source_shape.len() - i - 1]
            } else {
                1
            };
            let tgt_dim = if i < target_shape.len() {
                target_shape[target_shape.len() - i - 1]
            } else {
                1
            };

            if src_dim != tgt_dim && src_dim != 1 && tgt_dim != 1 {
                return false;
            }
        }

        true
    }

    /// Analyze memory layout efficiency
    pub fn analyze_layout_efficiency(&self, shape: &[usize], strides: &[usize]) -> LayoutAnalysis {
        let c_strides = self.compute_c_strides(shape);
        let f_strides = self.compute_fortran_strides(shape);

        let is_c_contiguous = strides == c_strides;
        let is_f_contiguous = strides == f_strides;
        let is_contiguous = is_c_contiguous || is_f_contiguous;

        // Calculate stride efficiency (how close to optimal)
        let total_elements: usize = shape.iter().product();
        let memory_span = self.calculate_memory_span(shape, strides);
        let efficiency = if memory_span > 0 {
            total_elements as f64 / memory_span as f64
        } else {
            0.0
        };

        // Detect common patterns
        let layout_pattern = if is_c_contiguous {
            LayoutPattern::CContiguous
        } else if is_f_contiguous {
            LayoutPattern::FortranContiguous
        } else if self.is_unit_stride_pattern(strides) {
            LayoutPattern::UnitStride
        } else if self.has_regular_pattern(strides) {
            LayoutPattern::Regular
        } else {
            LayoutPattern::Irregular
        };

        LayoutAnalysis {
            is_contiguous,
            is_c_contiguous,
            is_f_contiguous,
            efficiency,
            layout_pattern,
            memory_span,
            recommended_layout: self.recommend_layout(shape, strides),
        }
    }

    /// Calculate memory span (highest address - lowest address + 1)
    fn calculate_memory_span(&self, shape: &[usize], strides: &[usize]) -> usize {
        if shape.is_empty() {
            return 0;
        }

        let mut min_offset = 0;
        let mut max_offset = 0;

        for (_, (&dim_size, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
            if dim_size > 1 {
                let offset = (dim_size - 1) * stride;
                if stride > 0 {
                    max_offset += offset;
                } else {
                    min_offset += offset;
                }
            }
        }

        max_offset - min_offset + 1
    }

    /// Check if strides follow unit stride pattern
    fn is_unit_stride_pattern(&self, strides: &[usize]) -> bool {
        strides.contains(&1)
    }

    /// Check if strides have regular pattern
    fn has_regular_pattern(&self, strides: &[usize]) -> bool {
        if strides.len() < 2 {
            return true;
        }

        // Check if strides are in geometric progression
        for i in 1..strides.len() {
            if strides[i] == 0 || strides[i - 1] == 0 {
                continue;
            }
            // Simple regularity check - more sophisticated analysis could be added
            if strides[i] > strides[i - 1] * 10 || strides[i - 1] > strides[i] * 10 {
                return false;
            }
        }

        true
    }

    /// Recommend optimal layout
    fn recommend_layout(&self, shape: &[usize], current_strides: &[usize]) -> MemoryLayout {
        let c_strides = self.compute_c_strides(shape);
        let f_strides = self.compute_fortran_strides(shape);

        let is_c_contiguous = current_strides == c_strides;
        let is_f_contiguous = current_strides == f_strides;

        // Calculate simple efficiency metric without recursion
        let total_elements: usize = shape.iter().product();
        let memory_span = self.calculate_memory_span(shape, current_strides);
        let efficiency = if memory_span > 0 {
            total_elements as f64 / memory_span as f64
        } else {
            0.0
        };

        if efficiency > 0.9 {
            if is_c_contiguous {
                MemoryLayout::C
            } else if is_f_contiguous {
                MemoryLayout::Fortran
            } else {
                MemoryLayout::Custom
            }
        } else {
            // Recommend based on access patterns (heuristic)
            if shape.len() <= 2 {
                MemoryLayout::C
            } else {
                MemoryLayout::Custom // Use cache-optimized layout
            }
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
}

impl Default for ShapeEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis of memory layout efficiency
#[derive(Debug, Clone)]
pub struct LayoutAnalysis {
    /// Whether the layout is contiguous in memory
    pub is_contiguous: bool,
    /// Whether the layout is C-contiguous
    pub is_c_contiguous: bool,
    /// Whether the layout is Fortran-contiguous
    pub is_f_contiguous: bool,
    /// Memory utilization efficiency (0.0 to 1.0)
    pub efficiency: f64,
    /// Pattern classification
    pub layout_pattern: LayoutPattern,
    /// Total memory span used
    pub memory_span: usize,
    /// Recommended layout for optimization
    pub recommended_layout: MemoryLayout,
}

/// Memory layout patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutPattern {
    /// C-style contiguous
    CContiguous,
    /// Fortran-style contiguous
    FortranContiguous,
    /// Has unit stride in at least one dimension
    UnitStride,
    /// Regular stride pattern
    Regular,
    /// Irregular stride pattern
    Irregular,
}

/// Advanced view system for efficient array operations
pub struct ViewSystem {
    shape_engine: ShapeEngine,
}

impl ViewSystem {
    /// Create a new view system
    pub fn new() -> Self {
        Self {
            shape_engine: ShapeEngine::new(),
        }
    }

    /// Create an optimized view with the best layout for the given operations
    pub fn create_optimized_view<'a, T>(
        &mut self,
        data: &'a [T],
        shape: &[usize],
        intended_operations: &[ViewOperation],
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement,
    {
        let optimal_layout = self.determine_optimal_layout(shape, intended_operations);
        let strides = self.shape_engine.compute_strides(shape, optimal_layout);

        let shape_obj = Shape::new(shape.to_vec());
        ArrayView::new(data, shape_obj, strides, 0)
    }

    /// Determine optimal layout based on intended operations
    fn determine_optimal_layout(
        &self,
        shape: &[usize],
        operations: &[ViewOperation],
    ) -> MemoryLayout {
        let mut score_c = 0;
        let mut score_fortran = 0;
        let mut score_custom = 0;

        for op in operations {
            match op {
                ViewOperation::RowAccess => score_c += 2,
                ViewOperation::ColumnAccess => score_fortran += 2,
                ViewOperation::RandomAccess => score_custom += 1,
                ViewOperation::SequentialScan => score_c += 1,
                ViewOperation::Transpose => score_fortran += 1,
                ViewOperation::MatrixMultiply => {
                    score_c += 1;
                    score_fortran += 1;
                }
                ViewOperation::Reduction => score_c += 1,
                ViewOperation::Broadcasting => score_custom += 2,
            }
        }

        // Consider shape characteristics
        if shape.len() > 2 {
            score_custom += 1;
        }

        if score_custom > score_c && score_custom > score_fortran {
            MemoryLayout::Custom
        } else if score_fortran > score_c {
            MemoryLayout::Fortran
        } else {
            MemoryLayout::C
        }
    }

    /// Create a view chain for complex operations
    pub fn create_view_chain<'a, T>(
        &mut self,
        initial_view: ArrayView<'a, T>,
        operations: &[ViewChainOperation],
    ) -> Result<ArrayView<'a, T>>
    where
        T: NumericElement + Copy,
    {
        let mut current_view = initial_view;

        for operation in operations {
            current_view = match operation {
                ViewChainOperation::Reshape(new_shape) => {
                    self.shape_engine.reshape_view(&current_view, new_shape)?
                }
                ViewChainOperation::Transpose(axes) => self
                    .shape_engine
                    .transpose_view(&current_view, axes.clone())?,
                ViewChainOperation::SwapAxes(ax1, ax2) => {
                    self.shape_engine.swapaxes_view(&current_view, *ax1, *ax2)?
                }
                ViewChainOperation::MoveAxis(src, dst) => {
                    self.shape_engine.moveaxis_view(&current_view, *src, *dst)?
                }
                ViewChainOperation::Squeeze(axes) => self
                    .shape_engine
                    .squeeze_view(&current_view, axes.clone())?,
                ViewChainOperation::ExpandDims(axes) => self
                    .shape_engine
                    .expand_dims_view(&current_view, axes.clone())?,
            };
        }

        Ok(current_view)
    }
}

impl Default for ViewSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Operations that can be performed on views
#[derive(Debug, Clone)]
pub enum ViewOperation {
    /// Access rows sequentially
    RowAccess,
    /// Access columns sequentially
    ColumnAccess,
    /// Random element access
    RandomAccess,
    /// Sequential scan through all elements
    SequentialScan,
    /// Transpose operation
    Transpose,
    /// Matrix multiplication
    MatrixMultiply,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
    /// Broadcasting operations
    Broadcasting,
}

/// Operations in a view chain
#[derive(Debug, Clone)]
pub enum ViewChainOperation {
    /// Reshape to new dimensions
    Reshape(Vec<usize>),
    /// Transpose with optional axis specification
    Transpose(Option<Vec<usize>>),
    /// Swap two axes
    SwapAxes(usize, usize),
    /// Move axis from source to destination
    MoveAxis(usize, usize),
    /// Squeeze dimensions
    Squeeze(Option<Vec<usize>>),
    /// Expand dimensions
    ExpandDims(Vec<usize>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::advanced_ops::{ArrayView, Shape};

    #[test]
    fn test_shape_engine_creation() {
        let engine = ShapeEngine::new();
        assert!(engine.stride_cache.is_empty());
    }

    #[test]
    fn test_c_strides_computation() {
        let mut engine = ShapeEngine::new();
        let shape = [2, 3, 4];
        let strides = engine.compute_strides(&shape, MemoryLayout::C);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_fortran_strides_computation() {
        let mut engine = ShapeEngine::new();
        let shape = [2, 3, 4];
        let strides = engine.compute_strides(&shape, MemoryLayout::Fortran);
        assert_eq!(strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_reshape_validation() {
        let engine = ShapeEngine::new();
        assert!(engine.can_reshape(&[2, 3], &[6]));
        assert!(engine.can_reshape(&[2, 3], &[3, 2]));
        assert!(!engine.can_reshape(&[2, 3], &[7]));
    }

    #[test]
    fn test_broadcastability_check() {
        let engine = ShapeEngine::new();
        assert!(engine.is_broadcastable(&[3, 1, 4], &[2, 4]));
        assert!(engine.is_broadcastable(&[1, 4], &[3, 4]));
        assert!(!engine.is_broadcastable(&[3, 4], &[5, 4]));
    }

    #[test]
    fn test_layout_analysis() {
        let mut engine = ShapeEngine::new();
        let shape = [3, 4];
        let c_strides = engine.compute_strides(&shape, MemoryLayout::C);

        let analysis = engine.analyze_layout_efficiency(&shape, &c_strides);
        assert!(analysis.is_c_contiguous);
        assert!(analysis.is_contiguous);
        assert!(analysis.efficiency > 0.9);
    }

    #[test]
    fn test_view_system_creation() {
        let mut view_system = ViewSystem::new();
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = [2, 3];
        let operations = vec![ViewOperation::RowAccess];

        let view = view_system
            .create_optimized_view(&data, &shape, &operations)
            .unwrap();
        assert_eq!(view.shape().dims, vec![2, 3]);
    }

    #[test]
    fn test_squeeze_operation() {
        let engine = ShapeEngine::new();
        let data = vec![1, 2, 3, 4];
        let shape = Shape::new(vec![1, 2, 1, 2]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let squeezed = engine.squeeze_view(&view, None).unwrap();
        assert_eq!(squeezed.shape().dims, vec![2, 2]);
    }

    #[test]
    fn test_expand_dims_operation() {
        let engine = ShapeEngine::new();
        let data = vec![1, 2, 3, 4];
        let shape = Shape::new(vec![2, 2]);
        let view = ArrayView::from_data(&data, shape).unwrap();

        let expanded = engine.expand_dims_view(&view, vec![1]).unwrap();
        assert_eq!(expanded.shape().dims, vec![2, 1, 2]);
    }
}

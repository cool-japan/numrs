//! Advanced array operations: broadcasting, fancy indexing, and views
//!
//! This module implements NumPy-style advanced array operations that are essential
//! for high-level numerical computing, including shape manipulation, broadcasting
//! semantics, and efficient memory views.

use crate::error::{NumRs2Error, Result};
use crate::traits::NumericElement;
use std::collections::HashSet;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Shape information for multidimensional arrays
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Create a 1D shape
    pub fn from_1d(size: usize) -> Self {
        Self { dims: vec![size] }
    }

    /// Create a 2D shape
    pub fn from_2d(rows: usize, cols: usize) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shapes are compatible for broadcasting
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let max_ndim = std::cmp::max(self.ndim(), other.ndim());

        for i in 0..max_ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - i - 1]
            } else {
                1
            };
            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - i - 1]
            } else {
                1
            };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Compute the broadcasted shape
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        if !self.is_broadcastable_with(other) {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                self.dims, other.dims
            )));
        }

        let max_ndim = std::cmp::max(self.ndim(), other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - i - 1]
            } else {
                1
            };
            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - i - 1]
            } else {
                1
            };
            result_dims.push(std::cmp::max(dim1, dim2));
        }

        result_dims.reverse();
        Ok(Shape::new(result_dims))
    }

    /// Get strides for C-contiguous layout
    pub fn c_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Get strides for Fortran-contiguous layout
    pub fn f_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in 1..self.ndim() {
            strides[i] = strides[i - 1] * self.dims[i - 1];
        }
        strides
    }

    /// Reshape to a new shape with the same total size
    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Shape> {
        let new_size: usize = new_dims.iter().product();
        if new_size != self.size() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Cannot reshape size {} to size {}",
                self.size(),
                new_size
            )));
        }
        Ok(Shape::new(new_dims))
    }

    /// Transpose the shape by swapping dimensions
    pub fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Shape> {
        let axes = axes.unwrap_or_else(|| (0..self.ndim()).rev().collect());

        if axes.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                "Number of axes must match number of dimensions".to_string(),
            ));
        }

        // Check for duplicate axes
        let mut seen = HashSet::new();
        for &axis in &axes {
            if axis >= self.ndim() {
                return Err(NumRs2Error::DimensionMismatch(format!(
                    "Axis {} is out of bounds for array of dimension {}",
                    axis,
                    self.ndim()
                )));
            }
            if !seen.insert(axis) {
                return Err(NumRs2Error::DimensionMismatch(
                    "Duplicate axis in transpose".to_string(),
                ));
            }
        }

        let new_dims = axes.iter().map(|&i| self.dims[i]).collect();
        Ok(Shape::new(new_dims))
    }
}

/// Index specification for advanced indexing
#[derive(Debug, Clone)]
pub enum IndexSpec {
    /// Single integer index
    Int(isize),
    /// Single integer index (alias for compatibility)
    Index(isize),
    /// Slice with start, stop, step
    Slice(Option<isize>, Option<isize>, Option<isize>),
    /// Array of indices (fancy indexing)
    Array(Vec<usize>),
    /// Array of indices (alias for compatibility)
    Indices(Vec<usize>),
    /// Boolean mask
    BoolMask(Vec<bool>),
    /// Boolean mask (alias for compatibility)
    Mask(Vec<bool>),
    /// Select all elements (equivalent to :)
    All,
    /// Ellipsis (...) - expand to fill remaining dimensions
    Ellipsis,
    /// New axis (None in NumPy)
    NewAxis,
}

impl IndexSpec {
    /// Create a slice from a range
    pub fn from_range<R>(range: R, _axis_size: usize) -> Self
    where
        R: Into<SliceInfo>,
    {
        let slice_info = range.into();
        Self::Slice(slice_info.start, slice_info.stop, slice_info.step)
    }

    /// Resolve the index specification to concrete indices
    pub fn resolve(&self, axis_size: usize) -> Result<ResolvedIndex> {
        match self {
            IndexSpec::Int(idx) | IndexSpec::Index(idx) => {
                let resolved_idx = if *idx < 0 {
                    axis_size.saturating_sub((-idx) as usize)
                } else {
                    *idx as usize
                };

                if resolved_idx >= axis_size {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} is out of bounds for axis of size {}",
                        idx, axis_size
                    )));
                }

                Ok(ResolvedIndex::Single(resolved_idx))
            }
            IndexSpec::Slice(start, stop, step) => {
                let step = step.unwrap_or(1);
                if step == 0 {
                    return Err(NumRs2Error::InvalidOperation(
                        "Slice step cannot be zero".to_string(),
                    ));
                }

                let start_resolved = start
                    .map(|s| {
                        if s < 0 {
                            axis_size.saturating_sub((-s) as usize)
                        } else {
                            s as usize
                        }
                    })
                    .unwrap_or(if step > 0 {
                        0
                    } else {
                        axis_size.saturating_sub(1)
                    });

                let stop_resolved = stop
                    .map(|s| {
                        if s < 0 {
                            axis_size.saturating_sub((-s) as usize)
                        } else {
                            s as usize
                        }
                    })
                    .unwrap_or(if step > 0 { axis_size } else { 0 });

                let indices = if step > 0 {
                    (start_resolved..stop_resolved.min(axis_size))
                        .step_by(step as usize)
                        .collect()
                } else {
                    let mut indices = Vec::new();
                    let mut current = start_resolved.min(axis_size);
                    while current > stop_resolved && current < axis_size {
                        indices.push(current);
                        if current < (-step) as usize {
                            break;
                        }
                        current -= (-step) as usize;
                    }
                    indices
                };

                Ok(ResolvedIndex::Multiple(indices))
            }
            IndexSpec::Array(indices) | IndexSpec::Indices(indices) => {
                // Validate all indices are in bounds
                for &idx in indices {
                    if idx >= axis_size {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "Index {} is out of bounds for axis of size {}",
                            idx, axis_size
                        )));
                    }
                }
                Ok(ResolvedIndex::Multiple(indices.clone()))
            }
            IndexSpec::BoolMask(mask) | IndexSpec::Mask(mask) => {
                if mask.len() != axis_size {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Boolean mask length {} doesn't match axis size {}",
                        mask.len(),
                        axis_size
                    )));
                }
                let indices = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if b { Some(i) } else { None })
                    .collect();
                Ok(ResolvedIndex::Multiple(indices))
            }
            IndexSpec::All => {
                // Select all elements (equivalent to 0..axis_size)
                let indices = (0..axis_size).collect();
                Ok(ResolvedIndex::Multiple(indices))
            }
            IndexSpec::Ellipsis | IndexSpec::NewAxis => {
                // These are handled at a higher level during index processing
                Ok(ResolvedIndex::Multiple(vec![]))
            }
        }
    }
}

/// Resolved index after processing IndexSpec
#[derive(Debug, Clone)]
pub enum ResolvedIndex {
    /// Single index
    Single(usize),
    /// Multiple indices
    Multiple(Vec<usize>),
}

/// Slice information helper
pub struct SliceInfo {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: Option<isize>,
}

impl From<Range<usize>> for SliceInfo {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: Some(range.start as isize),
            stop: Some(range.end as isize),
            step: Some(1),
        }
    }
}

impl From<RangeFrom<usize>> for SliceInfo {
    fn from(range: RangeFrom<usize>) -> Self {
        Self {
            start: Some(range.start as isize),
            stop: None,
            step: Some(1),
        }
    }
}

impl From<RangeTo<usize>> for SliceInfo {
    fn from(range: RangeTo<usize>) -> Self {
        Self {
            start: None,
            stop: Some(range.end as isize),
            step: Some(1),
        }
    }
}

impl From<RangeFull> for SliceInfo {
    fn from(_: RangeFull) -> Self {
        Self {
            start: None,
            stop: None,
            step: Some(1),
        }
    }
}

impl From<RangeInclusive<usize>> for SliceInfo {
    fn from(range: RangeInclusive<usize>) -> Self {
        Self {
            start: Some(*range.start() as isize),
            stop: Some(*range.end() as isize + 1),
            step: Some(1),
        }
    }
}

impl From<RangeToInclusive<usize>> for SliceInfo {
    fn from(range: RangeToInclusive<usize>) -> Self {
        Self {
            start: None,
            stop: Some(range.end as isize + 1),
            step: Some(1),
        }
    }
}

/// Array view that provides access to array data without owning it
#[derive(Debug, Clone)]
pub struct ArrayView<'a, T> {
    data: &'a [T],
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

impl<'a, T> ArrayView<'a, T> {
    /// Create a new array view
    pub fn new(data: &'a [T], shape: Shape, strides: Vec<usize>, offset: usize) -> Result<Self> {
        if strides.len() != shape.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                "Strides length must match number of dimensions".to_string(),
            ));
        }

        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Create a C-contiguous view
    pub fn from_data(data: &'a [T], shape: Shape) -> Result<Self> {
        let strides = shape.c_strides();
        Self::new(data, shape, strides, 0)
    }

    /// Get the shape of the view
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the strides of the view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if indices.len() != self.shape.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.shape.ndim(),
                indices.len()
            )));
        }

        let mut flat_index = self.offset;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape.dims[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} is out of bounds for dimension {} of size {}",
                    idx, i, self.shape.dims[i]
                )));
            }
            flat_index += idx * stride;
        }

        if flat_index >= self.data.len() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Computed index {} is out of bounds for data of size {}",
                flat_index,
                self.data.len()
            )));
        }

        Ok(&self.data[flat_index])
    }

    /// Create a slice view using index specifications
    pub fn slice(&self, indices: &[IndexSpec]) -> Result<ArrayView<'a, T>> {
        if indices.len() > self.shape.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                "Too many indices for array".to_string(),
            ));
        }

        let mut new_shape_dims = Vec::new();
        let mut new_strides = Vec::new();
        let mut new_offset = self.offset;
        let mut axis = 0;
        let mut ellipsis_used = false;

        for index_spec in indices {
            match index_spec {
                IndexSpec::Ellipsis => {
                    if ellipsis_used {
                        return Err(NumRs2Error::InvalidOperation(
                            "Only one ellipsis allowed".to_string(),
                        ));
                    }
                    ellipsis_used = true;

                    // Calculate how many axes to skip
                    let remaining_specs = indices.len() - new_shape_dims.len() - 1;
                    let axes_to_add = self.shape.ndim().saturating_sub(remaining_specs);

                    for _ in 0..axes_to_add {
                        if axis < self.shape.ndim() {
                            new_shape_dims.push(self.shape.dims[axis]);
                            new_strides.push(self.strides[axis]);
                            axis += 1;
                        }
                    }
                }
                IndexSpec::NewAxis => {
                    new_shape_dims.push(1);
                    new_strides.push(0); // Stride of 0 for new axis
                }
                _ => {
                    if axis >= self.shape.ndim() {
                        return Err(NumRs2Error::DimensionMismatch(
                            "Index beyond array dimensions".to_string(),
                        ));
                    }

                    let resolved = index_spec.resolve(self.shape.dims[axis])?;
                    match resolved {
                        ResolvedIndex::Single(idx) => {
                            new_offset += idx * self.strides[axis];
                            // Single index removes the dimension
                        }
                        ResolvedIndex::Multiple(indices) => {
                            if let IndexSpec::Slice(start, _, step) = index_spec {
                                let start = start.unwrap_or(0) as usize;
                                let step = step.unwrap_or(1) as usize;
                                new_offset += start * self.strides[axis];
                                new_shape_dims.push(indices.len());
                                new_strides.push(self.strides[axis] * step);
                            } else {
                                // For fancy indexing, we'd need a more complex implementation
                                new_shape_dims.push(indices.len());
                                new_strides.push(self.strides[axis]);
                            }
                        }
                    }
                    axis += 1;
                }
            }
        }

        // Add remaining dimensions if no ellipsis was used
        while axis < self.shape.ndim() {
            new_shape_dims.push(self.shape.dims[axis]);
            new_strides.push(self.strides[axis]);
            axis += 1;
        }

        let new_shape = Shape::new(new_shape_dims);
        Self::new(self.data, new_shape, new_strides, new_offset)
    }

    /// Transpose the view
    pub fn transpose(&self, axes: Option<Vec<usize>>) -> Result<ArrayView<'a, T>> {
        let new_shape = self.shape.transpose(axes.clone())?;
        let axes = axes.unwrap_or_else(|| (0..self.shape.ndim()).rev().collect());

        let new_strides = axes.iter().map(|&i| self.strides[i]).collect();

        Self::new(self.data, new_shape, new_strides, self.offset)
    }

    /// Reshape the view (only works if the view is contiguous)
    pub fn reshape(&self, new_shape: Shape) -> Result<ArrayView<'a, T>> {
        if new_shape.size() != self.shape.size() {
            return Err(NumRs2Error::DimensionMismatch(
                "Cannot reshape: size mismatch".to_string(),
            ));
        }

        // Check if the view is C-contiguous
        if !self.is_c_contiguous() {
            return Err(NumRs2Error::InvalidOperation(
                "Can only reshape C-contiguous views".to_string(),
            ));
        }

        let new_strides = new_shape.c_strides();
        Self::new(self.data, new_shape, new_strides, self.offset)
    }

    /// Check if the view is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        let expected_strides = self.shape.c_strides();
        self.strides == expected_strides
    }

    /// Check if the view is Fortran-contiguous
    pub fn is_f_contiguous(&self) -> bool {
        let expected_strides = self.shape.f_strides();
        self.strides == expected_strides
    }

    /// Get an iterator over all elements in the view in C-order (row-major)
    pub fn iter(&self) -> ArrayViewIterator<'a, T> {
        ArrayViewIterator::new(self)
    }
}

impl<T: Clone> ArrayView<'_, T> {
    /// Convert the view to a owned vector (collects all elements)
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.shape.size());
        // Simple implementation: iterate through all valid indices
        let mut indices = vec![0; self.shape.ndim()];
        loop {
            if let Ok(element) = self.get(&indices) {
                result.push(element.clone());
            }

            // Advance indices
            let mut carry = 1;
            for i in (0..indices.len()).rev() {
                indices[i] += carry;
                if indices[i] < self.shape.dims[i] {
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
        result
    }
}

/// Iterator over elements of an `ArrayView` in C-order (row-major).
///
/// The lifetime `'a` ties this iterator to the underlying slice, ensuring
/// the references remain valid for as long as the iterator is alive.
pub struct ArrayViewIterator<'a, T> {
    data: &'a [T],
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
    current_indices: Vec<usize>,
    finished: bool,
}

impl<'a, T> ArrayViewIterator<'a, T> {
    fn new(view: &ArrayView<'a, T>) -> Self {
        let current_indices = vec![0; view.shape.ndim()];
        let finished = view.shape.size() == 0;
        Self {
            data: view.data,
            shape: view.shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
            current_indices,
            finished,
        }
    }

    /// Compute the flat data index for the current multi-dimensional indices.
    fn current_flat_index(&self) -> usize {
        let mut flat_index = self.offset;
        for (&idx, &stride) in self.current_indices.iter().zip(self.strides.iter()) {
            flat_index += idx * stride;
        }
        flat_index
    }

    /// Advance the multi-dimensional index counter by one step in C-order.
    /// Returns `true` when the iteration has been exhausted after this advance.
    fn advance_indices(&mut self) -> bool {
        let mut carry = 1usize;
        for i in (0..self.current_indices.len()).rev() {
            self.current_indices[i] += carry;
            if self.current_indices[i] < self.shape.dims[i] {
                carry = 0;
                break;
            } else {
                self.current_indices[i] = 0;
                carry = 1;
            }
        }
        carry == 1
    }
}

impl<'a, T> Iterator for ArrayViewIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let flat_index = self.current_flat_index();

        // Advance indices before returning, marking finished if we wrap around.
        if self.advance_indices() {
            self.finished = true;
        }

        self.data.get(flat_index)
    }
}

/// Broadcasting operation helper
pub struct BroadcastOp;

impl BroadcastOp {
    /// Apply binary operation with broadcasting
    pub fn binary_op<T, F>(
        a: &ArrayView<T>,
        b: &ArrayView<T>,
        output: &mut [T],
        op: F,
    ) -> Result<()>
    where
        T: NumericElement + Copy,
        F: Fn(T, T) -> T,
    {
        let broadcast_shape = a.shape().broadcast_with(b.shape())?;

        if output.len() != broadcast_shape.size() {
            return Err(NumRs2Error::DimensionMismatch(
                "Output buffer size doesn't match broadcast shape".to_string(),
            ));
        }

        // Simple implementation - could be optimized with SIMD
        let mut output_idx = 0;
        let mut indices = vec![0; broadcast_shape.ndim()];

        loop {
            // Map broadcast indices to original array indices
            let a_indices = Self::map_broadcast_indices(&indices, a.shape(), &broadcast_shape);
            let b_indices = Self::map_broadcast_indices(&indices, b.shape(), &broadcast_shape);

            let a_val = *a.get(&a_indices)?;
            let b_val = *b.get(&b_indices)?;
            output[output_idx] = op(a_val, b_val);

            output_idx += 1;

            // Advance indices
            if !Self::advance_indices(&mut indices, &broadcast_shape.dims) {
                break;
            }
        }

        Ok(())
    }

    /// Apply unary operation with broadcasting/reshaping
    pub fn unary_op<T, F>(
        input: &ArrayView<T>,
        output: &mut [T],
        target_shape: &Shape,
        op: F,
    ) -> Result<()>
    where
        T: NumericElement + Copy,
        F: Fn(T) -> T,
    {
        if !input.shape().is_broadcastable_with(target_shape) {
            return Err(NumRs2Error::DimensionMismatch(
                "Input shape is not broadcastable to target shape".to_string(),
            ));
        }

        if output.len() != target_shape.size() {
            return Err(NumRs2Error::DimensionMismatch(
                "Output buffer size doesn't match target shape".to_string(),
            ));
        }

        let mut output_idx = 0;
        let mut indices = vec![0; target_shape.ndim()];

        loop {
            let input_indices = Self::map_broadcast_indices(&indices, input.shape(), target_shape);
            let input_val = *input.get(&input_indices)?;
            output[output_idx] = op(input_val);

            output_idx += 1;

            if !Self::advance_indices(&mut indices, &target_shape.dims) {
                break;
            }
        }

        Ok(())
    }

    fn map_broadcast_indices(
        broadcast_indices: &[usize],
        original_shape: &Shape,
        broadcast_shape: &Shape,
    ) -> Vec<usize> {
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

    fn advance_indices(indices: &mut [usize], shape: &[usize]) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::from_2d(3, 4);
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.size(), 12);
        assert_eq!(shape.dims, vec![3, 4]);
    }

    #[test]
    fn test_shape_broadcasting() {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![2, 4]);

        assert!(shape1.is_broadcastable_with(&shape2));

        let broadcast_shape = shape1
            .broadcast_with(&shape2)
            .expect("test: operation should succeed");
        assert_eq!(broadcast_shape.dims, vec![3, 2, 4]);
    }

    #[test]
    fn test_shape_broadcasting_incompatible() {
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![5, 4]);

        assert!(!shape1.is_broadcastable_with(&shape2));
        assert!(shape1.broadcast_with(&shape2).is_err());
    }

    #[test]
    fn test_shape_strides() {
        let shape = Shape::new(vec![2, 3, 4]);

        let c_strides = shape.c_strides();
        assert_eq!(c_strides, vec![12, 4, 1]);

        let f_strides = shape.f_strides();
        assert_eq!(f_strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_shape_transpose() {
        let shape = Shape::new(vec![2, 3, 4]);

        // Default transpose (reverse axes)
        let transposed = shape
            .transpose(None)
            .expect("test: operation should succeed");
        assert_eq!(transposed.dims, vec![4, 3, 2]);

        // Custom transpose
        let transposed = shape
            .transpose(Some(vec![1, 0, 2]))
            .expect("test: operation should succeed");
        assert_eq!(transposed.dims, vec![3, 2, 4]);
    }

    #[test]
    fn test_index_spec_resolution() {
        // Test integer index
        let spec = IndexSpec::Int(2);
        let resolved = spec.resolve(5).expect("test: operation should succeed");
        assert!(matches!(resolved, ResolvedIndex::Single(2)));

        // Test negative index
        let spec = IndexSpec::Int(-1);
        let resolved = spec.resolve(5).expect("test: operation should succeed");
        assert!(matches!(resolved, ResolvedIndex::Single(4)));

        // Test slice
        let spec = IndexSpec::Slice(Some(1), Some(4), Some(2));
        let resolved = spec.resolve(5).expect("test: operation should succeed");
        if let ResolvedIndex::Multiple(indices) = resolved {
            assert_eq!(indices, vec![1, 3]);
        } else {
            panic!("Expected Multiple indices");
        }
    }

    #[test]
    fn test_array_view_creation() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = Shape::from_2d(2, 3);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        assert_eq!(view.shape().dims, vec![2, 3]);
        assert_eq!(
            view.get(&[0, 0]).expect("test: operation should succeed"),
            &1
        );
        assert_eq!(
            view.get(&[1, 2]).expect("test: operation should succeed"),
            &6
        );
    }

    #[test]
    fn test_array_view_slicing() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let shape = Shape::new(vec![3, 4]);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        // Slice first row
        let row_slice = view
            .slice(&[IndexSpec::Int(0), IndexSpec::Slice(None, None, None)])
            .expect("test: operation should succeed");
        assert_eq!(row_slice.shape().dims, vec![4]);
        assert_eq!(
            row_slice.get(&[0]).expect("test: operation should succeed"),
            &1
        );
        assert_eq!(
            row_slice.get(&[3]).expect("test: operation should succeed"),
            &4
        );
    }

    #[test]
    fn test_array_view_transpose() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = Shape::from_2d(2, 3);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let transposed = view
            .transpose(None)
            .expect("test: operation should succeed");
        assert_eq!(transposed.shape().dims, vec![3, 2]);
        assert_eq!(
            transposed
                .get(&[0, 0])
                .expect("test: operation should succeed"),
            &1
        );
        assert_eq!(
            transposed
                .get(&[2, 1])
                .expect("test: operation should succeed"),
            &6
        );
    }

    #[test]
    fn test_array_view_iterator() {
        let data = vec![1, 2, 3, 4];
        let shape = Shape::from_2d(2, 2);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        // Verify iter() returns elements in C-order (row-major): 1, 2, 3, 4
        let elements: Vec<&i32> = view.iter().collect();
        assert_eq!(elements, vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_array_view_iter_1d() {
        let data = vec![10, 20, 30, 40, 50];
        let shape = Shape::from_1d(5);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let elements: Vec<i32> = view.iter().copied().collect();
        assert_eq!(elements, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_array_view_iter_empty() {
        let data: Vec<i32> = vec![];
        let shape = Shape::new(vec![0]);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let elements: Vec<&i32> = view.iter().collect();
        assert!(elements.is_empty());
    }

    #[test]
    fn test_array_view_iter_3d() {
        // 2x2x3 array filled with 0..12
        let data: Vec<i32> = (0..12).collect();
        let shape = Shape::new(vec![2, 2, 3]);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let elements: Vec<i32> = view.iter().copied().collect();
        assert_eq!(elements, (0..12).collect::<Vec<i32>>());
    }

    #[test]
    fn test_array_view_iter_sum() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::from_2d(2, 3);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let sum: f64 = view.iter().copied().sum();
        assert!((sum - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_broadcast_binary_op() {
        let data_a = vec![1.0, 2.0, 3.0];
        let shape_a = Shape::from_1d(3);
        let view_a =
            ArrayView::from_data(&data_a, shape_a).expect("test: operation should succeed");

        let data_b = vec![10.0];
        let shape_b = Shape::from_1d(1);
        let view_b =
            ArrayView::from_data(&data_b, shape_b).expect("test: operation should succeed");

        let mut output = vec![0.0; 3];
        BroadcastOp::binary_op(&view_a, &view_b, &mut output, |a, b| a + b)
            .expect("test: operation should succeed");

        assert_eq!(output, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_broadcast_unary_op() {
        let data = vec![1.0, 2.0];
        let shape = Shape::from_1d(2);
        let view = ArrayView::from_data(&data, shape).expect("test: operation should succeed");

        let target_shape = Shape::from_2d(2, 2);
        let mut output = vec![0.0; 4];

        BroadcastOp::unary_op(&view, &mut output, &target_shape, |x| x * 2.0)
            .expect("test: operation should succeed");
        assert_eq!(output, vec![2.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_fancy_indexing() {
        let _data = [10, 20, 30, 40, 50];
        let indices = vec![0, 2, 4];
        let spec = IndexSpec::Array(indices);

        let resolved = spec.resolve(5).expect("test: operation should succeed");
        if let ResolvedIndex::Multiple(indices) = resolved {
            assert_eq!(indices, vec![0, 2, 4]);
        } else {
            panic!("Expected Multiple indices");
        }
    }

    #[test]
    fn test_boolean_indexing() {
        let mask = vec![true, false, true, false, true];
        let spec = IndexSpec::BoolMask(mask);

        let resolved = spec.resolve(5).expect("test: operation should succeed");
        if let ResolvedIndex::Multiple(indices) = resolved {
            assert_eq!(indices, vec![0, 2, 4]);
        } else {
            panic!("Expected Multiple indices");
        }
    }
}

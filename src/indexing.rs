use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use std::ops::Range;

/// Represents an index specification for a single dimension
#[derive(Clone)]
pub enum IndexSpec {
    /// A single index
    Index(usize),
    /// A range of indices with start, end, and optional step
    Slice(usize, Option<usize>, Option<usize>),
    /// A set of arbitrary indices
    Indices(Vec<usize>),
    /// A boolean mask
    Mask(Vec<bool>),
    /// All indices
    All,
    /// Ellipsis (...) - expands to the number of : needed for selection
    Ellipsis,
}

impl IndexSpec {
    /// Create a new slice specification
    pub fn slice(start: usize, end: Option<usize>, step: Option<usize>) -> Self {
        IndexSpec::Slice(start, end, step)
    }
    
    /// Create a new index specification from a range
    pub fn from_range(range: Range<usize>) -> Self {
        IndexSpec::Slice(range.start, Some(range.end), None)
    }
    
    /// Create a new index specification from a vector of indices
    pub fn from_indices(indices: Vec<usize>) -> Self {
        IndexSpec::Indices(indices)
    }
    
    /// Create a new index specification from a boolean mask
    pub fn from_mask(mask: Vec<bool>) -> Self {
        IndexSpec::Mask(mask)
    }
    
    /// Create an ellipsis index specification
    pub fn ellipsis() -> Self {
        IndexSpec::Ellipsis
    }
}

impl<T: Clone + num_traits::Zero> Array<T> {
    /// Get an element at the specified indices
    /// 
    /// # Arguments
    /// * `indices` - A slice of indices, one for each dimension
    /// 
    /// # Returns
    /// * `Ok(T)` - The element at the specified indices
    /// * `Err(NumRsError)` - If the indices are out of bounds
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Expected {} indices, got {}", self.ndim(), indices.len())
            ));
        }
        
        // Check if indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumRs2Error::IndexOutOfBounds(
                    format!("Index {} is out of bounds for dimension {} with size {}", 
                            idx, i, self.shape()[i])
                ));
            }
        }
        
        // Get the element
        let value = self.array().get(indices).ok_or_else(|| {
            NumRs2Error::IndexOutOfBounds(
                format!("Failed to get element at indices {:?}", indices)
            )
        })?;
        
        Ok(value.clone())
    }
    
    /// Index into the array using boolean array or indices
    /// 
    /// # Arguments
    /// * `index_specs` - A slice of index specifications, one for each dimension
    /// 
    /// # Returns
    /// * `Ok(Array<T>)` - The indexed array
    /// * `Err(NumRsError)` - If the indices are invalid
    pub fn index(&self, index_specs: &[IndexSpec]) -> Result<Self>
    where 
        T: Clone
    {
        if index_specs.is_empty() {
            return Ok(self.clone());
        }
        
        if index_specs.len() > self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Too many indices: expected at most {}, got {}", 
                        self.ndim(), index_specs.len())
            ));
        }
        
        // Handle boolean indexing first
        for (dim, spec) in index_specs.iter().enumerate() {
            if let IndexSpec::Mask(mask) = spec {
                return self.bool_index(dim, mask);
            }
        }
        
        // Handle fancy indexing (integer array indexing)
        let has_fancy_indexing = index_specs.iter().any(|spec| {
            matches!(spec, IndexSpec::Indices(_))
        });
        
        if has_fancy_indexing {
            return self.fancy_index(index_specs);
        }
        
        // Handle basic indexing (integer and slice indexing)
        let mut shape = Vec::new();
        let mut ndarray_indices = Vec::with_capacity(self.ndim());
        
        // Process explicitly provided indices
        for (dim, spec) in index_specs.iter().enumerate() {
            match spec {
                IndexSpec::Index(idx) => {
                    if *idx >= self.shape()[dim] {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("Index {} is out of bounds for dimension {} with size {}", 
                                    idx, dim, self.shape()[dim])
                        ));
                    }
                    ndarray_indices.push(SliceInfoElem::Index(*idx as isize));
                },
                IndexSpec::Slice(start, end, step) => {
                    let dim_size = self.shape()[dim];
                    let end_idx = end.unwrap_or(dim_size);
                    let step_size = step.unwrap_or(1);
                    
                    if *start >= dim_size {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("Start index {} is out of bounds for dimension {} with size {}", 
                                    start, dim, dim_size)
                        ));
                    }
                    
                    if end_idx > dim_size {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("End index {} is out of bounds for dimension {} with size {}", 
                                    end_idx, dim, dim_size)
                        ));
                    }
                    
                    if step_size == 0 {
                        return Err(NumRs2Error::InvalidOperation(
                            "Step size cannot be zero".to_string()
                        ));
                    }
                    
                    // Calculate the size of this dimension in the result
                    let slice_size = if end_idx > *start {
                        (end_idx - *start + step_size - 1) / step_size
                    } else {
                        0
                    };
                    
                    shape.push(slice_size);
                    ndarray_indices.push(SliceInfoElem::Slice { 
                        start: *start as isize, 
                        end: Some(end_idx as isize), 
                        step: step_size as isize 
                    });
                },
                IndexSpec::All => {
                    shape.push(self.shape()[dim]);
                    ndarray_indices.push(SliceInfoElem::Slice { 
                        start: 0, 
                        end: Some(self.shape()[dim] as isize), 
                        step: 1 
                    });
                },
                IndexSpec::Indices(_) | IndexSpec::Mask(_) => {
                    // These should have been handled above
                    unreachable!();
                },
                IndexSpec::Ellipsis => {
                    // Ellipsis is handled separately below
                }
            }
        }
        
        // Process ellipsis if present
        let ellipsis_idx = index_specs.iter().position(|spec| matches!(spec, IndexSpec::Ellipsis));
        
        if let Some(idx) = ellipsis_idx {
            // Calculate how many dimensions need to be filled
            let num_dims_provided = index_specs.len() - 1; // -1 for the ellipsis
            let num_dims_needed = self.ndim();
            let additional_dims = if num_dims_needed > num_dims_provided {
                num_dims_needed - num_dims_provided
            } else {
                0
            };
            
            let mut expanded_indices = Vec::with_capacity(self.ndim());
            
            // Add indices before ellipsis
            expanded_indices.extend_from_slice(&ndarray_indices[0..idx]);
            
            // Add full slices for each expanded dimension
            for dim in 0..additional_dims {
                let actual_dim = idx + dim;
                if actual_dim < self.shape().len() {
                    expanded_indices.push(SliceInfoElem::Slice { 
                        start: 0, 
                        end: Some(self.shape()[actual_dim] as isize), 
                        step: 1 
                    });
                }
            }
            
            // Add indices after ellipsis
            if idx < ndarray_indices.len() {
                expanded_indices.extend_from_slice(&ndarray_indices[idx..]);
            }
            
            // Replace the original indices with expanded ones
            ndarray_indices = expanded_indices;
        } else {
            // Fill in remaining dimensions with full slices
            for dim in index_specs.len()..self.ndim() {
                shape.push(self.shape()[dim]);
                ndarray_indices.push(SliceInfoElem::Slice { 
                    start: 0, 
                    end: Some(self.shape()[dim] as isize), 
                    step: 1 
                });
            }
        }
        
        // Create the slice information
        let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(ndarray_indices)
            .map_err(|_| NumRs2Error::InvalidOperation("Failed to create slice info".to_string()))?;
        
        // Slice the array
        let result = self.array().slice(slice_info).into_owned().into_dyn();
        
        Ok(Self::from_ndarray(result))
    }
    
    /// Index into the array using a boolean mask for a specific dimension
    fn bool_index(&self, dim: usize, mask: &[bool]) -> Result<Self>
    where 
        T: Clone
    {
        if dim >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Dimension {} is out of bounds for array with {} dimensions", 
                        dim, self.ndim())
            ));
        }
        
        if mask.len() != self.shape()[dim] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![self.shape()[dim]],
                actual: vec![mask.len()],
            });
        }
        
        // Count true values to determine output size
        let true_count = mask.iter().filter(|&&m| m).count();
        
        // If no true values, return empty array with appropriate shape
        if true_count == 0 {
            let mut result_shape = self.shape().clone();
            result_shape[dim] = 0;
            return Ok(Self::zeros(&result_shape));
        }
        
        // Convert boolean mask to indices
        let indices: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        
        // Use the indices for indexing
        let mut index_specs = vec![IndexSpec::All; self.ndim()];
        index_specs[dim] = IndexSpec::Indices(indices);
        
        // Call fancy_index to perform the actual indexing
        self.fancy_index(&index_specs)
    }
    
    /// Index into the array using arrays of indices (fancy indexing)
    fn fancy_index(&self, index_specs: &[IndexSpec]) -> Result<Self>
    where 
        T: Clone
    {
        // Count fancy indexing dimensions
        let fancy_dims: Vec<usize> = index_specs.iter()
            .enumerate()
            .filter_map(|(i, spec)| if matches!(spec, IndexSpec::Indices(_)) { Some(i) } else { None })
            .collect();
        
        if fancy_dims.is_empty() {
            return Err(NumRs2Error::InvalidOperation("No fancy indexing found".to_string()))
        }
        
        // Support multiple fancy indexing dimensions
        if fancy_dims.len() > 1 {
            return self.multi_fancy_index(index_specs, &fancy_dims);
        }
        
        let fancy_dim = fancy_dims[0];
        
        let idx_vec = match &index_specs[fancy_dim] {
            IndexSpec::Indices(idx) => idx,
            _ => unreachable!(),
        };
        
        if idx_vec.is_empty() {
            // Handle empty indices case - return empty array with appropriate shape
            let mut result_shape = self.shape().clone();
            result_shape[fancy_dim] = 0;
            return Ok(Self::zeros(&result_shape));
        }
        
        // Check if indices are within bounds
        for &idx in idx_vec.iter() {
            if idx >= self.shape()[fancy_dim] {
                return Err(NumRs2Error::IndexOutOfBounds(
                    format!("Index {} is out of bounds for dimension {} with size {}", 
                            idx, fancy_dim, self.shape()[fancy_dim])
                ));
            }
        }
        
        // Calculate output shape
        let mut output_shape = self.shape().clone();
        output_shape[fancy_dim] = idx_vec.len();
        
        // Create a result array with the right shape
        let mut result = Self::zeros(&output_shape);
        
        // For each index in the fancy indexing dimension
        for (new_idx, &orig_idx) in idx_vec.iter().enumerate() {
            // Create index specs to select a single slice from the original array
            let mut slice_specs = index_specs.to_vec();
            slice_specs[fancy_dim] = IndexSpec::Index(orig_idx);
            
            // Get the slice
            let slice = self.index(&slice_specs)?;
            
            // Create index specs to select the target position in the result array
            let mut result_specs = vec![IndexSpec::All; self.ndim()];
            result_specs[fancy_dim] = IndexSpec::Index(new_idx);
            
            // Copy slice data to the appropriate position in the result
            let mut target_idx = vec![0; self.ndim()];
            for i in 0..slice.size() {
                // Compute multi-dimensional index for the slice
                let mut slice_idx = vec![0; slice.ndim()];
                let mut tmp = i;
                for dim in (0..slice.ndim()).rev() {
                    slice_idx[dim] = tmp % slice.shape()[dim];
                    tmp /= slice.shape()[dim];
                }
                
                // Compute corresponding index in the result array
                for dim in 0..self.ndim() {
                    if dim == fancy_dim {
                        target_idx[dim] = new_idx;
                    } else {
                        let slice_dim = if dim < fancy_dim { dim } else { dim - 1 };
                        if slice_dim < slice_idx.len() {
                            target_idx[dim] = slice_idx[slice_dim];
                        } else {
                            target_idx[dim] = 0;
                        }
                    }
                }
                
                // Get value from slice and set in result
                let value = slice.get(&slice_idx)?;
                result.set(&target_idx, value)?;
            }
        }
        
        Ok(result)
    }
    
    /// Handle indexing with multiple fancy indices
    fn multi_fancy_index(&self, index_specs: &[IndexSpec], fancy_dims: &[usize]) -> Result<Self>
    where 
        T: Clone
    {
        // Extract all indices sets
        let indices_sets: Vec<&Vec<usize>> = fancy_dims.iter()
            .map(|&dim| {
                match &index_specs[dim] {
                    IndexSpec::Indices(idx) => idx,
                    _ => unreachable!(),
                }
            })
            .collect();
        
        // Find the broadcast shape of all indices
        let broadcast_size = indices_sets.iter()
            .map(|idx| idx.len())
            .max()
            .unwrap_or(0);
        
        // Verify all indices are broadcastable (they must have the same size or be size 1)
        for indices in &indices_sets {
            if indices.len() != broadcast_size && indices.len() != 1 {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![broadcast_size],
                    actual: vec![indices.len()],
                });
            }
        }
        
        // Create a result array to hold the elements
        let mut result_data = Vec::with_capacity(broadcast_size);
        
        // For each index in the broadcast shape
        for i in 0..broadcast_size {
            // Create a complete set of indices by selecting from each dimension
            let mut all_indices = vec![0; self.ndim()];
            
            for (&dim, indices) in fancy_dims.iter().zip(&indices_sets) {
                // Handle broadcasting of size 1 indices
                let idx_i = if indices.len() == 1 { 0 } else { i };
                
                // Check bounds
                if indices[idx_i] >= self.shape()[dim] {
                    return Err(NumRs2Error::IndexOutOfBounds(
                        format!("Index {} is out of bounds for dimension {} with size {}", 
                                indices[idx_i], dim, self.shape()[dim])
                    ));
                }
                
                all_indices[dim] = indices[idx_i];
            }
            
            // Set the non-fancy dims to their default values (first element)
            for dim in 0..self.ndim() {
                if !fancy_dims.contains(&dim) {
                    match &index_specs[dim] {
                        IndexSpec::All => {},
                        IndexSpec::Index(idx) => all_indices[dim] = *idx,
                        IndexSpec::Slice(start, _, _) => all_indices[dim] = *start,
                        _ => {}
                    }
                }
            }
            
            // Get and store the element
            let element = self.get(&all_indices)?;
            result_data.push(element);
        }
        
        // Create the result array with the broadcast shape
        Ok(Array::from_vec(result_data))
    }
    
    /// Set values in the array using a boolean mask
    pub fn set_mask(&mut self, mask: &Array<bool>, values: &Array<T>) -> Result<()>
    where 
        T: Clone
    {
        // Check that the mask shape is compatible with the array shape
        if self.shape() != mask.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: mask.shape(),
            });
        }
        
        // Count the number of True values in the mask
        let true_count = mask.array().iter().filter(|&&x| x).count();
        
        // Check that the values array has the right number of elements
        if values.size() != true_count {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![true_count],
                actual: vec![values.size()],
            });
        }
        
        // Set the values
        let mut value_idx = 0;
        let values_vec = values.to_vec();
        
        for (idx, &masked) in mask.array().iter().enumerate() {
            if masked {
                // Get the corresponding multi-dimensional index
                let mut multi_idx = Vec::with_capacity(self.ndim());
                let mut remaining = idx;
                
                for dim_size in self.shape().iter().rev() {
                    multi_idx.insert(0, remaining % dim_size);
                    remaining /= dim_size;
                }
                
                // Set the value
                if let Some(elem) = self.array_mut().get_mut(multi_idx.as_slice()) {
                    *elem = values_vec[value_idx].clone();
                }
                
                value_idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Get a diagonal view of a 2D array
    pub fn diag(&self) -> Result<Self>
    where 
        T: Clone
    {
        if self.ndim() != 2 {
            return Err(NumRs2Error::DimensionMismatch(
                format!("diag requires a 2D array, got {}D", self.ndim())
            ));
        }
        
        let shape = self.shape();
        let min_dim = shape[0].min(shape[1]);
        
        let mut result = Vec::with_capacity(min_dim);
        
        for i in 0..min_dim {
            result.push(self.get(&[i, i])?);
        }
        
        Ok(Self::from_vec(result))
    }
    
    /// Extract the diagonal from an array
    /// 
    /// # Parameters
    /// 
    /// * `offset` - Offset of the diagonal from the main diagonal (0 = main diagonal, +ve = above, -ve = below)
    /// * `axis1` - First axis of the 2D subarray to extract diagonal from (default 0)
    /// * `axis2` - Second axis of the 2D subarray to extract diagonal from (default 1)
    /// 
    /// # Returns
    /// 
    /// Array containing the diagonal elements
    /// 
    /// # Examples
    /// 
    /// ```
    /// use numrs2::prelude::*;
    /// 
    /// // Create a 3x3 array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    /// 
    /// // Extract the main diagonal
    /// let diag = a.diagonal(0, None, None).unwrap();
    /// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
    /// 
    /// // Extract the diagonal above the main diagonal
    /// let diag_above = a.diagonal(1, None, None).unwrap();
    /// assert_eq!(diag_above.to_vec(), vec![2, 6]);
    /// 
    /// // Extract the diagonal below the main diagonal
    /// let diag_below = a.diagonal(-1, None, None).unwrap();
    /// assert_eq!(diag_below.to_vec(), vec![4, 8]);
    /// ```
    pub fn diagonal(&self, offset: isize, axis1: Option<usize>, axis2: Option<usize>) -> Result<Self>
    where 
        T: Clone
    {
        // Default axes
        let ax1 = axis1.unwrap_or(0);
        let ax2 = axis2.unwrap_or(1);
        
        let ndim = self.ndim();
        
        if ax1 >= ndim || ax2 >= ndim {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Axes ({}, {}) out of bounds for array of dimension {}", ax1, ax2, ndim)
            ));
        }
        
        if ax1 == ax2 {
            return Err(NumRs2Error::InvalidOperation(
                format!("Axes ({}, {}) cannot be identical", ax1, ax2)
            ));
        }
        
        // Get the dimensions of the 2D subarray defined by the axes
        let shape = self.shape();
        let dim1 = shape[ax1];
        let dim2 = shape[ax2];
        
        // Calculate the min and max valid indices based on offset
        let max_diag_len = if offset >= 0 {
            ((dim2 - offset as usize).min(dim1)) as isize
        } else {
            ((dim1 - (-offset) as usize).min(dim2)) as isize
        };
        
        if max_diag_len <= 0 {
            // Empty diagonal
            return Ok(Self::from_vec(vec![]));
        }
        
        // Create the result array
        let mut result = Vec::with_capacity(max_diag_len as usize);
        
        // Extract the diagonal elements
        for i in 0..max_diag_len {
            let mut idx = vec![0; ndim];
            
            if offset >= 0 {
                idx[ax1] = i as usize;
                idx[ax2] = (i + offset) as usize;
            } else {
                idx[ax1] = (i - offset) as usize;
                idx[ax2] = i as usize;
            }
            
            result.push(self.get(&idx)?);
        }
        
        Ok(Self::from_vec(result))
    }
    
    /// Take elements from an array along an axis
    /// 
    /// # Parameters
    /// 
    /// * `indices` - The indices of the values to extract
    /// * `axis` - The axis over which to select values (default is None which flattens the array)
    /// 
    /// # Returns
    /// 
    /// An array of values at the given indices along the given axis
    /// 
    /// # Examples
    /// 
    /// ```
    /// use numrs2::prelude::*;
    /// 
    /// // Create a simple array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// 
    /// // Take elements at indices 0, 2, 4
    /// let result = a.take(&Array::from_vec(vec![0, 2, 4]), None).unwrap();
    /// assert_eq!(result.to_vec(), vec![1, 3, 5]);
    /// 
    /// // With a 2D array
    /// let b = a.reshape(&[5, 2]);
    /// 
    /// // Take rows at indices 0, 2, 4
    /// let rows = b.take(&Array::from_vec(vec![0, 2, 4]), Some(0)).unwrap();
    /// assert_eq!(rows.shape(), vec![3, 2]);
    /// 
    /// // Take columns at index 1
    /// let cols = b.take(&Array::from_vec(vec![1]), Some(1)).unwrap();
    /// assert_eq!(cols.shape(), vec![5, 1]);
    /// ```
    pub fn take(&self, indices: &Self, axis: Option<usize>) -> Result<Self>
    where 
        T: Clone + ToString
    {
        // Validate indices are integers
        for i in 0..indices.size() {
            if let Ok(idx) = indices.array().as_slice().unwrap()[i].to_string().parse::<usize>() {
                if idx >= self.size() {
                    return Err(NumRs2Error::IndexOutOfBounds(
                        format!("Index {} is out of bounds for axis with size {}", idx, self.size())
                    ));
                }
            } else {
                return Err(NumRs2Error::InvalidOperation(
                    "Indices must be integers".to_string()
                ));
            }
        }
        
        match axis {
            None => {
                // Flatten the array and take elements directly
                let flat_data = self.to_vec();
                let mut result = Vec::with_capacity(indices.size());
                
                for i in 0..indices.size() {
                    // Parse index since we can't directly cast
                    let idx_str = indices.array().as_slice().unwrap()[i].to_string();
                    let idx = idx_str.parse::<usize>().unwrap();
                    
                    if idx >= flat_data.len() {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("Index {} is out of bounds for axis with size {}", idx, flat_data.len())
                        ));
                    }
                    
                    result.push(flat_data[idx].clone());
                }
                
                Ok(Self::from_vec(result))
            },
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(
                        format!("Axis {} is out of bounds for array with {} dimensions", ax, self.ndim())
                    ));
                }
                
                let shape = self.shape();
                let axis_dim = shape[ax];
                
                // Create the output shape
                let mut out_shape = shape.clone();
                out_shape[ax] = indices.size();
                
                let mut result_data = Vec::with_capacity(self.size() / axis_dim * indices.size());
                
                // This is a simplified implementation - a more efficient approach would
                // use strided views rather than extracting slices for each index
                
                for i in 0..indices.size() {
                    // Parse index since we can't directly cast
                    let idx_str = indices.array().as_slice().unwrap()[i].to_string();
                    let idx = idx_str.parse::<usize>().unwrap();
                    
                    if idx >= axis_dim {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("Index {} is out of bounds for axis {} with size {}", idx, ax, axis_dim)
                        ));
                    }
                    
                    // Create index specs to select along the axis
                    let mut index_specs = vec![IndexSpec::All; self.ndim()];
                    index_specs[ax] = IndexSpec::Index(idx);
                    
                    // Get the slice at this index
                    let slice = self.index(&index_specs)?;
                    
                    // Append the slice's data to the result
                    result_data.extend(slice.to_vec());
                }
                
                // Reshape to the correct output shape
                Ok(Self::from_vec(result_data).reshape(&out_shape))
            }
        }
    }
    
    /// Choose elements from a list of choices based on an index array
    /// 
    /// # Parameters
    /// 
    /// * `choices` - List of arrays to choose from
    /// * `mode` - Specifies how indices outside bounds are handled ('raise', 'wrap', 'clip')
    /// 
    /// # Returns
    /// 
    /// An array of elements chosen from the choices at the specified indices
    /// 
    /// # Examples
    /// 
    /// ```ignore
    /// use numrs2::prelude::*;
    /// 
    /// // Create an index array
    /// let a = Array::from_vec(vec![0, 1, 2, 1, 0]);
    /// 
    /// // Create choice arrays
    /// let c1 = Array::from_vec(vec![10, 20, 30, 40, 50]);
    /// let c2 = Array::from_vec(vec![100, 200, 300, 400, 500]);
    /// let c3 = Array::from_vec(vec![1000, 2000, 3000, 4000, 5000]);
    /// 
    /// // Select elements based on the indices
    /// let result = a.choose(&[&c1, &c2, &c3], None).unwrap();
    /// assert_eq!(result.to_vec(), vec![10, 200, 3000, 200, 10]);
    /// ```
    pub fn choose(&self, choices: &[&Self], mode: Option<&str>) -> Result<Self>
    where 
        T: Clone + ToString
    {
        if choices.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "No choices provided".to_string()
            ));
        }
        
        let n_choices = choices.len();
        
        // Check consistency of choices shapes
        let first_shape = choices[0].shape();
        for (_i, choice) in choices.iter().enumerate().skip(1) {
            if choice.shape() != first_shape {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: first_shape.clone(),
                    actual: choice.shape()
                });
            }
        }
        
        // Check if this array can be used as indices
        let my_shape = self.shape();
        let mut result_data = Vec::with_capacity(self.size());
        
        let handle_mode = mode.unwrap_or("raise");
        
        // For each element in this array, choose from the choices
        for i in 0..self.size() {
            // Get the index value
            let idx_value = self.array().as_slice().unwrap()[i].to_string().parse::<isize>();
            
            if idx_value.is_err() {
                return Err(NumRs2Error::InvalidOperation(
                    "Index values must be integers".to_string()
                ));
            }
            
            let mut idx = idx_value.unwrap();
            
            // Apply the mode
            match handle_mode {
                "raise" => {
                    if idx < 0 || idx >= n_choices as isize {
                        return Err(NumRs2Error::IndexOutOfBounds(
                            format!("Index {} is out of bounds for choices with size {}", idx, n_choices)
                        ));
                    }
                },
                "wrap" => {
                    // Wrap around
                    idx = ((idx % n_choices as isize) + n_choices as isize) % n_choices as isize;
                },
                "clip" => {
                    // Clip to bounds
                    if idx < 0 {
                        idx = 0;
                    } else if idx >= n_choices as isize {
                        idx = (n_choices - 1) as isize;
                    }
                },
                _ => {
                    return Err(NumRs2Error::InvalidOperation(
                        format!("Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'", handle_mode)
                    ));
                }
            }
            
            // Get the value from the appropriate choice array
            let flat_idx = i;
            result_data.push(choices[idx as usize].array().as_slice().unwrap()[flat_idx].clone());
        }
        
        Ok(Self::from_vec(result_data).reshape(&my_shape))
    }
    
    /// Select elements from an array using a boolean mask
    /// 
    /// # Parameters
    /// 
    /// * `condition` - Boolean mask array of the same shape as self
    /// * `axis` - Axis along which to select elements (None for flattened array)
    /// 
    /// # Returns
    /// 
    /// Array of selected elements
    /// 
    /// # Examples
    /// 
    /// ```
    /// use numrs2::prelude::*;
    /// 
    /// // Create a simple array
    /// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// 
    /// // Create a condition (select even numbers)
    /// let condition = a.map(|x| x % 2 == 0);
    /// 
    /// // Compress the array using the condition
    /// let result = a.compress(&condition, None).unwrap();
    /// assert_eq!(result.to_vec(), vec![2, 4, 6, 8, 10]);
    /// 
    /// // With a 2D array
    /// let b = a.reshape(&[5, 2]);
    /// let cond_axis = Array::from_vec(vec![true, false, true, false, true]);
    /// 
    /// // Select rows with the condition
    /// let compressed = b.compress(&cond_axis, Some(0)).unwrap();
    /// assert_eq!(compressed.shape(), vec![3, 2]);
    /// ```
    pub fn compress<U>(&self, condition: &Array<U>, axis: Option<usize>) -> Result<Self>
    where 
        T: Clone + ToString,
        U: Clone + ToString
    {
        // Check if condition contains boolean values
        for i in 0..condition.size() {
            let val_str = condition.array().as_slice().unwrap()[i].to_string();
            if val_str != "true" && val_str != "false" {
                return Err(NumRs2Error::InvalidOperation(
                    "Condition must contain boolean values".to_string()
                ));
            }
        }
        
        match axis {
            None => {
                // Flatten both arrays and select elements where condition is true
                let flat_data = self.to_vec();
                let flat_condition: Vec<bool> = condition.to_vec()
                    .iter()
                    .map(|x| x.to_string() == "true")
                    .collect();
                
                if flat_data.len() != flat_condition.len() {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: vec![flat_data.len()],
                        actual: vec![flat_condition.len()],
                    });
                }
                
                let mut result = Vec::new();
                
                for (i, &cond) in flat_condition.iter().enumerate() {
                    if cond {
                        result.push(flat_data[i].clone());
                    }
                }
                
                Ok(Self::from_vec(result))
            },
            Some(ax) => {
                if ax >= self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(
                        format!("Axis {} is out of bounds for array with {} dimensions", ax, self.ndim())
                    ));
                }
                
                let shape = self.shape();
                let axis_dim = shape[ax];
                
                // The condition must have the same size as the specified axis
                let cond_size = condition.size();
                if cond_size != axis_dim {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: vec![axis_dim],
                        actual: vec![cond_size],
                    });
                }
                
                // Count true values to determine output size
                let true_count = condition.to_vec()
                    .iter()
                    .filter(|x| x.to_string() == "true")
                    .count();
                
                if true_count == 0 {
                    // No true values, return an empty array with appropriate shape
                    let mut out_shape = shape.clone();
                    out_shape[ax] = 0;
                    return Ok(Self::zeros(&out_shape));
                }
                
                // Create index specs
                let indices: Vec<usize> = condition.to_vec()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| if x.to_string() == "true" { Some(i) } else { None })
                    .collect();
                
                // Create the output shape
                let mut out_shape = shape.clone();
                out_shape[ax] = true_count;
                
                let mut result_data = Vec::new();
                
                // This is a simplified implementation - a more efficient approach would
                // use strided views rather than extracting slices for each index
                
                for &idx in &indices {
                    // Create index specs to select along the axis
                    let mut index_specs = vec![IndexSpec::All; self.ndim()];
                    index_specs[ax] = IndexSpec::Index(idx);
                    
                    // Get the slice at this index
                    let slice = self.index(&index_specs)?;
                    
                    // Append the slice's data to the result
                    result_data.extend(slice.to_vec());
                }
                
                // Reshape to the correct output shape
                Ok(Self::from_vec(result_data).reshape(&out_shape))
            }
        }
    }
}

/// Generate index arrays for fancy indexing
/// 
/// # Parameters
/// 
/// * `arrays` - A list of arrays to generate index grids from
/// 
/// # Returns
/// 
/// A list of N arrays, where N is the number of input arrays. The i-th output array is an array that
/// can be used for indexing the i-th dimension of an array.
/// 
/// # Examples
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// // Create arrays for indices
/// let a = Array::from_vec(vec![0, 1, 2]);
/// let b = Array::from_vec(vec![3, 4, 5]);
/// 
/// // Generate index arrays
/// let indices = ix_(&[&a, &b]).unwrap();
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);
/// assert_eq!(indices[1].shape(), vec![1, 3]);
/// 
/// // Can be used for fancy indexing
/// let data = Array::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(&[3, 3]);
/// // Would select elements at positions (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), etc.
/// ```
pub fn ix_<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }
    
    let n = arrays.len();
    let mut result = Vec::with_capacity(n);
    
    for (i, array) in arrays.iter().enumerate() {
        // Create a shape with 1s for all dimensions except the i-th
        let mut shape = vec![1; n];
        shape[i] = array.size();
        
        // Create a reshaped copy of the array
        let reshaped = array.reshape(&shape);
        result.push(reshaped);
    }
    
    Ok(result)
}

/// Set array values using indices
/// 
/// # Parameters
/// 
/// * `array` - Array to modify
/// * `indices` - Indices where values should be put
/// * `values` - Values to put at the indices
/// * `mode` - How out-of-bounds indices are handled ('raise', 'wrap', 'clip')
/// 
/// # Returns
/// 
/// Result<()> indicating success or error
/// 
/// # Examples
/// 
/// ```ignore
/// use numrs2::prelude::*;
/// 
/// // Create an array
/// let mut a = Array::zeros(&[5]);
/// 
/// // Set values at specific indices
/// let indices = Array::from_vec(vec![0, 2, 4]);
/// let values = Array::from_vec(vec![10, 20, 30]);
/// 
/// put(&mut a, &indices, &values, None).unwrap();
/// assert_eq!(a.to_vec(), vec![10, 0, 20, 0, 30]);
/// 
/// // Test with wrap mode
/// let mut b = Array::zeros(&[3]);
/// let indices = Array::from_vec(vec![0, 1, 2, 3, 4, 5]);
/// let values = Array::from_vec(vec![10, 20, 30, 40, 50, 60]);
/// 
/// put(&mut b, &indices, &values, Some("wrap")).unwrap();
/// // Indices 3,4,5 wrap around to 0,1,2
/// assert_eq!(b.to_vec(), vec![10 + 40, 20 + 50, 30 + 60]);
/// ```
pub fn put<T: Clone + ToString>(array: &mut Array<T>, indices: &Array<T>, values: &Array<T>, mode: Option<&str>) -> Result<()> {
    // Validate indices are integers
    for i in 0..indices.size() {
        if indices.array().as_slice().unwrap()[i].to_string().parse::<isize>().is_err() {
            return Err(NumRs2Error::InvalidOperation(
                "Indices must be integers".to_string()
            ));
        }
    }
    
    // Check that values has at least as many elements as indices
    let n_indices = indices.size();
    let n_values = values.size();
    
    if n_values < n_indices {
        return Err(NumRs2Error::InvalidOperation(
            format!("Not enough values ({}) to put at all indices ({})", n_values, n_indices)
        ));
    }
    
    let array_size = array.size();
    let handle_mode = mode.unwrap_or("raise");
    
    // Process each index
    for i in 0..n_indices {
        // Get the index value
        let idx_value = indices.array().as_slice().unwrap()[i].to_string().parse::<isize>().unwrap();
        
        // Apply the mode
        let idx = match handle_mode {
            "raise" => {
                if idx_value < 0 || idx_value >= array_size as isize {
                    return Err(NumRs2Error::IndexOutOfBounds(
                        format!("Index {} is out of bounds for array with size {}", idx_value, array_size)
                    ));
                }
                idx_value as usize
            },
            "wrap" => {
                // Wrap around
                (((idx_value % array_size as isize) + array_size as isize) % array_size as isize) as usize
            },
            "clip" => {
                // Clip to bounds
                if idx_value < 0 {
                    0
                } else if idx_value >= array_size as isize {
                    array_size - 1
                } else {
                    idx_value as usize
                }
            },
            _ => {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Invalid mode: {}. Must be one of 'raise', 'wrap', or 'clip'", handle_mode)
                ));
            }
        };
        
        // Compute the multi-dimensional index
        let shape = array.shape();
        let ndim = shape.len();
        
        let mut multi_idx = Vec::with_capacity(ndim);
        let mut temp = idx;
        
        for j in (0..ndim).rev() {
            if j == 0 {
                multi_idx.insert(0, temp);
            } else {
                let prod: usize = shape[1..=j].iter().product();
                multi_idx.insert(0, temp / prod);
                temp %= prod;
            }
        }
        
        // Get the value to put
        let value = values.array().as_slice().unwrap()[i % n_values].clone();
        
        // Set the value
        array.set(&multi_idx, value)?;
    }
    
    Ok(())
}

/// Create index arrays for the nth dimension in an n-dimensional grid from shape
///
/// # Parameters
///
/// * `shape` - The shape of the grid
///
/// # Returns
///
/// A vector of arrays containing indices for each dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create index arrays for a 2D grid
/// let indices = indices_grid::<usize>(&[3, 2]).unwrap();
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].shape(), vec![3, 1]);  // Column vector of row indices
/// assert_eq!(indices[1].shape(), vec![1, 2]);  // Row vector of column indices
/// ```
pub fn indices_grid<T: Clone + num_traits::Zero + num_traits::One + num_traits::NumCast>(shape: &[usize]) -> Result<Vec<Array<T>>> {
    if shape.is_empty() {
        return Ok(vec![]);
    }
    
    let mut result = Vec::with_capacity(shape.len());
    
    for (i, &dim) in shape.iter().enumerate() {
        // Create a shape with 1s except at position i
        let mut index_shape = vec![1; shape.len()];
        index_shape[i] = dim;
        
        // Create the index array
        let mut index_data = Vec::with_capacity(dim);
        for j in 0..dim {
            index_data.push(T::from(j).unwrap());
        }
        
        let index_array = Array::from_vec(index_data).reshape(&index_shape);
        result.push(index_array);
    }
    
    Ok(result)
}

/// Return the indices to access array elements that satisfy the given condition
///
/// # Parameters
///
/// * `shape` - Shape of the output array
/// * `mask_fn` - Function which given indices returns a boolean mask
///
/// # Returns
///
/// An array of indices that satisfy the condition
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create indices for the upper triangle of a 3x3 array
/// let indices = mask_indices(&[3, 3], |i| i[0] <= i[1]).unwrap();
/// assert_eq!(indices.len(), 2);
/// assert_eq!(indices[0].to_vec(), vec![0, 0, 0, 1, 1, 2]);
/// assert_eq!(indices[1].to_vec(), vec![0, 1, 2, 1, 2, 2]);
/// ```
pub fn mask_indices<F>(shape: &[usize], mask_fn: F) -> Result<Vec<Array<usize>>>
where
    F: Fn(&[usize]) -> bool
{
    if shape.is_empty() {
        return Ok(vec![]);
    }
    
    // Calculate the total number of elements
    let total_elements: usize = shape.iter().product();
    
    // Create arrays to store indices for each dimension
    let mut indices_vec: Vec<Vec<usize>> = vec![Vec::new(); shape.len()];
    
    // Iterate through all possible indices
    let mut indices = vec![0; shape.len()];
    for _ in 0..total_elements {
        // Check the mask function
        if mask_fn(&indices) {
            // Store the indices
            for (dim, &idx) in indices.iter().enumerate() {
                indices_vec[dim].push(idx);
            }
        }
        
        // Increment indices
        let mut carry = true;
        for dim in (0..shape.len()).rev() {
            if carry {
                indices[dim] += 1;
                carry = indices[dim] >= shape[dim];
                if carry {
                    indices[dim] = 0;
                }
            }
        }
    }
    
    // Convert to Array objects
    let result = indices_vec.into_iter()
        .map(Array::from_vec)
        .collect();
    
    Ok(result)
}

/// Set array elements using a mask array
/// 
/// # Parameters
/// 
/// * `array` - Array to modify
/// * `mask` - Boolean mask array of same shape as array
/// * `values` - Values to set at positions where mask is True
/// 
/// # Returns
/// 
/// Result<()> indicating success or error
/// 
/// # Examples
/// 
/// ```
/// use numrs2::prelude::*;
/// 
/// // Create a simple array
/// let mut a = Array::from_vec(vec![1, 2, 3, 4, 5]);
/// 
/// // Create a mask (select even indices)
/// let mask = Array::from_vec(vec![false, true, false, true, false]);
/// 
/// // Set values at masked positions
/// let values = Array::from_vec(vec![20, 40]);
/// 
/// putmask(&mut a, &mask, &values).unwrap();
/// assert_eq!(a.to_vec(), vec![1, 20, 3, 40, 5]);
/// ```
pub fn putmask<T: Clone + ToString, U: Clone + ToString>(array: &mut Array<T>, mask: &Array<U>, values: &Array<T>) -> Result<()> {
    // Check shapes
    if array.shape() != mask.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: array.shape(),
            actual: mask.shape(),
        });
    }
    
    // Check if mask contains boolean values
    for i in 0..mask.size() {
        let val_str = mask.array().as_slice().unwrap()[i].to_string();
        if val_str != "true" && val_str != "false" {
            return Err(NumRs2Error::InvalidOperation(
                "Mask must contain boolean values".to_string()
            ));
        }
    }
    
    // Count true values in mask to check against values size
    let true_count = mask.to_vec()
        .iter()
        .filter(|x| x.to_string() == "true")
        .count();
    
    let n_values = values.size();
    
    if n_values == 0 && true_count > 0 {
        return Err(NumRs2Error::InvalidOperation(
            "No values provided to fill masked elements".to_string()
        ));
    }
    
    // Process each element
    let mut value_idx = 0;
    
    for i in 0..array.size() {
        let mask_val = mask.array().as_slice().unwrap()[i].to_string() == "true";
        
        if mask_val {
            // Calculate the multi-dimensional index
            let shape = array.shape();
            let ndim = shape.len();
            
            let mut multi_idx = Vec::with_capacity(ndim);
            let mut temp = i;
            
            for j in (0..ndim).rev() {
                if j == 0 {
                    multi_idx.insert(0, temp);
                } else {
                    let prod: usize = shape[1..=j].iter().product();
                    multi_idx.insert(0, temp / prod);
                    temp %= prod;
                }
            }
            
            // Get the value to put (cycling if necessary)
            let value = values.array().as_slice().unwrap()[value_idx % n_values].clone();
            
            // Set the value
            array.set(&multi_idx, value)?;
            
            value_idx += 1;
        }
    }
    
    Ok(())
}
//! Fancy and boolean indexing for arrays
//!
//! This module provides advanced indexing capabilities:
//! - `bool_index()` - Index using boolean masks
//! - `fancy_index()` - Index using arrays of indices
//! - `multi_fancy_index()` - Index using multiple index arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::indexing::IndexSpec;

impl<T: Clone + num_traits::Zero> Array<T> {
    /// Index into the array using a boolean mask for a specific dimension
    pub(crate) fn bool_index(&self, dim: usize, mask: &[bool]) -> Result<Self>
    where
        T: Clone,
    {
        if dim >= self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Dimension {} is out of bounds for array with {} dimensions",
                dim,
                self.ndim()
            )));
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
        let indices: Vec<usize> = mask
            .iter()
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
    pub(crate) fn fancy_index(&self, index_specs: &[IndexSpec]) -> Result<Self>
    where
        T: Clone,
    {
        // Count fancy indexing dimensions
        let fancy_dims: Vec<usize> = index_specs
            .iter()
            .enumerate()
            .filter_map(|(i, spec)| {
                if matches!(spec, IndexSpec::Indices(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if fancy_dims.is_empty() {
            return Err(NumRs2Error::InvalidOperation(
                "No fancy indexing found".to_string(),
            ));
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
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx,
                    fancy_dim,
                    self.shape()[fancy_dim]
                )));
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
                #[allow(clippy::needless_range_loop)]
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
    pub(crate) fn multi_fancy_index(
        &self,
        index_specs: &[IndexSpec],
        fancy_dims: &[usize],
    ) -> Result<Self>
    where
        T: Clone,
    {
        // Extract all indices sets
        let indices_sets: Vec<&Vec<usize>> = fancy_dims
            .iter()
            .map(|&dim| match &index_specs[dim] {
                IndexSpec::Indices(idx) => idx,
                _ => unreachable!(),
            })
            .collect();

        // Find the broadcast shape of all indices
        let broadcast_size = indices_sets.iter().map(|idx| idx.len()).max().unwrap_or(0);

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
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "Index {} is out of bounds for dimension {} with size {}",
                        indices[idx_i],
                        dim,
                        self.shape()[dim]
                    )));
                }

                all_indices[dim] = indices[idx_i];
            }

            // Set the non-fancy dims to their default values (first element)
            for dim in 0..self.ndim() {
                if !fancy_dims.contains(&dim) {
                    match &index_specs[dim] {
                        IndexSpec::All => {}
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
}

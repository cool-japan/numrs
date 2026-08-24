//! Core indexing methods for arrays
//!
//! This module provides the fundamental indexing operations:
//! - `get()` - Get a single element at specified indices
//! - `index()` - Index into an array using IndexSpec specifications

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::indexing::{insert_newaxis, IndexSpec};
use scirs2_core::ndarray::{IxDyn, SliceInfo, SliceInfoElem};

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
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Check if indices are within bounds.
        //
        // Reads `self.array().shape()` -- a zero-allocation `&[usize]`
        // borrow -- rather than calling `self.shape()` per index, which
        // heap-allocates a fresh `Vec` via `to_vec()` on *every* call.
        // `get()` sits on the same hot per-element paths as `Array::set`
        // (paired `get`/`set` loops in hand-rolled linalg and iterative
        // solvers), so this allocation was paid twice per loop iteration.
        // Values are identical either way.
        let shape = self.array().shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx, i, shape[i]
                )));
            }
        }

        // Get the element
        let value = self.array().get(indices).ok_or_else(|| {
            NumRs2Error::IndexOutOfBounds(format!("Failed to get element at indices {:?}", indices))
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
        T: Clone,
    {
        if index_specs.is_empty() {
            return Ok(self.clone());
        }

        // Count NewAxis specs - they don't consume input dimensions
        let newaxis_count = index_specs
            .iter()
            .filter(|spec| matches!(spec, IndexSpec::NewAxis))
            .count();
        let ellipsis_count = index_specs
            .iter()
            .filter(|spec| matches!(spec, IndexSpec::Ellipsis))
            .count();

        // Actual indices for dimensions (excluding NewAxis and Ellipsis)
        let actual_index_count = index_specs.len() - newaxis_count - ellipsis_count;

        if actual_index_count > self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Too many indices: expected at most {}, got {}",
                self.ndim(),
                actual_index_count
            )));
        }

        // Handle NewAxis separately - calculate output positions for new axes
        let has_newaxis = newaxis_count > 0;
        let newaxis_output_positions: Vec<usize> = if has_newaxis {
            // Calculate the output position for each NewAxis based on how many
            // output-producing specs come before it in the index spec list
            let mut positions = Vec::new();
            let mut output_dim = 0;

            for spec in index_specs.iter() {
                match spec {
                    IndexSpec::NewAxis => {
                        // NewAxis inserts at the current output dimension
                        positions.push(output_dim);
                        output_dim += 1;
                    }
                    IndexSpec::Index(_) => {
                        // Index consumes an input dimension but produces no output dimension
                    }
                    IndexSpec::Slice(_, _, _) | IndexSpec::All => {
                        // Slice/All consumes an input dimension and produces an output dimension
                        output_dim += 1;
                    }
                    IndexSpec::Ellipsis => {
                        // Ellipsis expands to fill remaining dimensions
                        // Count how many dimensions it will expand to
                        let remaining_input_dims = self.ndim() - actual_index_count;
                        output_dim += remaining_input_dims;
                    }
                    IndexSpec::Indices(_) | IndexSpec::Mask(_) => {
                        // Fancy indexing - for now treat as producing one output dimension
                        output_dim += 1;
                    }
                }
            }
            positions
        } else {
            vec![]
        };

        // Filter out NewAxis specs for the main indexing operation
        let filtered_specs: Vec<IndexSpec> = index_specs
            .iter()
            .filter(|spec| !matches!(spec, IndexSpec::NewAxis))
            .cloned()
            .collect();

        // Handle boolean indexing first (only check filtered specs)
        for (dim, spec) in filtered_specs.iter().enumerate() {
            if let IndexSpec::Mask(mask) = spec {
                let result = self.bool_index(dim, mask)?;
                return if has_newaxis {
                    insert_newaxis(&result, &newaxis_output_positions)
                } else {
                    Ok(result)
                };
            }
        }

        // Handle fancy indexing (integer array indexing)
        let has_fancy_indexing = filtered_specs
            .iter()
            .any(|spec| matches!(spec, IndexSpec::Indices(_)));

        if has_fancy_indexing {
            let result = self.fancy_index(&filtered_specs)?;
            return if has_newaxis {
                insert_newaxis(&result, &newaxis_output_positions)
            } else {
                Ok(result)
            };
        }

        // Handle basic indexing (integer and slice indexing)
        let mut shape = Vec::new();
        let mut ndarray_indices = Vec::with_capacity(self.ndim());

        // Process explicitly provided indices
        for (dim, spec) in filtered_specs.iter().enumerate() {
            match spec {
                IndexSpec::Index(idx) => {
                    if *idx >= self.shape()[dim] {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "Index {} is out of bounds for dimension {} with size {}",
                            idx,
                            dim,
                            self.shape()[dim]
                        )));
                    }
                    ndarray_indices.push(SliceInfoElem::Index(*idx as isize));
                }
                IndexSpec::Slice(start, end, step) => {
                    let dim_size = self.shape()[dim];
                    let end_idx = end.unwrap_or(dim_size);
                    let step_size = step.unwrap_or(1);

                    if *start >= dim_size {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "Start index {} is out of bounds for dimension {} with size {}",
                            start, dim, dim_size
                        )));
                    }

                    if end_idx > dim_size {
                        return Err(NumRs2Error::IndexOutOfBounds(format!(
                            "End index {} is out of bounds for dimension {} with size {}",
                            end_idx, dim, dim_size
                        )));
                    }

                    if step_size == 0 {
                        return Err(NumRs2Error::InvalidOperation(
                            "Step size cannot be zero".to_string(),
                        ));
                    }

                    // Calculate the size of this dimension in the result
                    let slice_size = if end_idx > *start {
                        (end_idx - *start).div_ceil(step_size)
                    } else {
                        0
                    };

                    shape.push(slice_size);
                    ndarray_indices.push(SliceInfoElem::Slice {
                        start: *start as isize,
                        end: Some(end_idx as isize),
                        step: step_size as isize,
                    });
                }
                IndexSpec::All => {
                    shape.push(self.shape()[dim]);
                    ndarray_indices.push(SliceInfoElem::Slice {
                        start: 0,
                        end: Some(self.shape()[dim] as isize),
                        step: 1,
                    });
                }
                IndexSpec::Indices(_) | IndexSpec::Mask(_) => {
                    // INVARIANT: unreachable. Any `Mask` spec in
                    // `filtered_specs` triggers an early `return` a few
                    // lines above (the `bool_index` loop), and any
                    // `Indices` spec sets `has_fancy_indexing`, which also
                    // triggers an early `return` (via `fancy_index`) before
                    // this loop runs. So by the time this `match` executes,
                    // `filtered_specs` cannot contain either variant.
                    unreachable!();
                }
                IndexSpec::Ellipsis => {
                    // Ellipsis is handled separately below
                }
                IndexSpec::NewAxis => {
                    // INVARIANT: unreachable. `filtered_specs` is built a
                    // few lines above by filtering `NewAxis` out of
                    // `index_specs`, so it can never contain this variant.
                    unreachable!();
                }
            }
        }

        // Process ellipsis if present
        let ellipsis_idx = filtered_specs
            .iter()
            .position(|spec| matches!(spec, IndexSpec::Ellipsis));

        if let Some(idx) = ellipsis_idx {
            // Calculate how many dimensions need to be filled
            let num_dims_provided = filtered_specs.len() - 1; // -1 for the ellipsis
            let num_dims_needed = self.ndim();
            let additional_dims = num_dims_needed.saturating_sub(num_dims_provided);

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
                        step: 1,
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
            for dim in filtered_specs.len()..self.ndim() {
                shape.push(self.shape()[dim]);
                ndarray_indices.push(SliceInfoElem::Slice {
                    start: 0,
                    end: Some(self.shape()[dim] as isize),
                    step: 1,
                });
            }
        }

        // Create the slice information
        let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(ndarray_indices).map_err(|_| {
            NumRs2Error::InvalidOperation("Failed to create slice info".to_string())
        })?;

        // Slice the array
        let result = self.array().slice(slice_info).into_owned().into_dyn();

        let result_array = Self::from_ndarray(result);

        // Insert new axes if needed
        if has_newaxis {
            insert_newaxis(&result_array, &newaxis_output_positions)
        } else {
            Ok(result_array)
        }
    }
}

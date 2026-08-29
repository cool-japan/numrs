//! Tensor operations for Array
//! Includes Kronecker product, tensor dot product, and other tensor operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Einstein summation convention for tensor contractions
///
/// Evaluates the Einstein summation convention on the operands. This function provides
/// a general way to compute tensor contractions, element-wise products, matrix products,
/// traces, and more operations through index notation.
///
/// # Arguments
/// * `subscripts` - String specifying the subscripts for summation (e.g., "ij,jk->ik" for matrix multiplication)
/// * `operands` - Vector of arrays to operate on
///
/// # Returns
/// * `Result<Array<T>>` - The result of the Einstein summation
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensor_ops::einsum;
///
/// // Matrix multiplication: C_ik = A_ij * B_jk
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let result = einsum("ij,jk->ik", &[&a, &b]).expect("einsum should succeed for valid matrix multiplication");
///
/// // Trace: sum_i A_ii
/// let trace = einsum("ii->", &[&a]).expect("einsum should succeed for trace computation");
///
/// // Dot product: sum_i a_i * b_i
/// let v1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let v2 = Array::from_vec(vec![4.0, 5.0, 6.0]);
/// let dot = einsum("i,i->", &[&v1, &v2]).expect("einsum should succeed for dot product");
/// ```
pub fn einsum<T: Float + Clone + Debug + std::ops::AddAssign + 'static>(
    subscripts: &str,
    operands: &[&Array<T>],
) -> Result<Array<T>> {
    // Parse the subscripts string
    let parts: Vec<&str> = subscripts.split("->").collect();
    if parts.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "einsum subscripts must contain exactly one '->'".to_string(),
        ));
    }

    let input_spec = parts[0];
    let output_spec = parts[1];

    // Split input spec by comma to get individual operand specs
    let operand_specs: Vec<&str> = input_spec.split(',').collect();

    if operand_specs.len() != operands.len() {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Number of operand specs ({}) doesn't match number of operands ({})",
            operand_specs.len(),
            operands.len()
        )));
    }

    // Handle common cases with optimized implementations

    // Matrix multiplication: "ij,jk->ik"
    if operand_specs.len() == 2
        && operand_specs[0] == "ij"
        && operand_specs[1] == "jk"
        && output_spec == "ik"
    {
        return operands[0].matmul(operands[1]);
    }

    // Vector dot product: "i,i->"
    if operand_specs.len() == 2
        && operand_specs[0] == "i"
        && operand_specs[1] == "i"
        && output_spec.is_empty()
    {
        use crate::linalg::vector_ops::vdot;
        let result = vdot(operands[0], operands[1])?;
        return Ok(Array::from_vec(vec![result]));
    }

    // Trace: "ii->"
    if operand_specs.len() == 1 && operand_specs[0] == "ii" && output_spec.is_empty() {
        use crate::linalg::vector_ops::trace;
        let result = trace(operands[0])?;
        return Ok(Array::from_vec(vec![result]));
    }

    // Transpose: "ij->ji"
    if operand_specs.len() == 1 && operand_specs[0] == "ij" && output_spec == "ji" {
        return Ok(operands[0].transpose());
    }

    // Diagonal: "ii->i"
    if operand_specs.len() == 1 && operand_specs[0] == "ii" && output_spec == "i" {
        use crate::array_ops::diagonal::diag;
        return diag(operands[0], None);
    }

    // Outer product: "i,j->ij"
    if operand_specs.len() == 2
        && operand_specs[0] == "i"
        && operand_specs[1] == "j"
        && output_spec == "ij"
    {
        use crate::linalg::vector_ops::outer;
        return outer(operands[0], operands[1]);
    }

    // Element-wise multiplication: "ij,ij->ij"
    if operand_specs.len() == 2
        && operand_specs[0] == operand_specs[1]
        && operand_specs[0] == output_spec
    {
        // Element-wise multiplication
        let a_data = operands[0].to_vec();
        let b_data = operands[1].to_vec();
        let result_data: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        return Array::from_vec_shape(result_data, &operands[0].shape());
    }

    // Sum over axis: "ij->i" (sum over j) or "ij->j" (sum over i)
    if operand_specs.len() == 1 && operand_specs[0].len() == 2 && output_spec.len() == 1 {
        let input_chars: Vec<char> = operand_specs[0].chars().collect();
        // output_spec is guaranteed to have exactly 1 character (checked on line 133)
        let output_char = output_spec.chars().next().unwrap_or_default();

        if input_chars.contains(&output_char) {
            // Find which axis to sum over
            let sum_axis = if input_chars[0] == output_char { 1 } else { 0 };
            return operands[0].sum_axis(sum_axis);
        }
    }

    // For more complex cases, use a general (but slower) implementation
    einsum_general(subscripts, operands)
}

/// General implementation of einsum for arbitrary index patterns
fn einsum_general<T: Float + Clone + Debug + std::ops::AddAssign>(
    subscripts: &str,
    operands: &[&Array<T>],
) -> Result<Array<T>> {
    // Parse subscripts
    let parts: Vec<&str> = subscripts.split("->").collect();
    let input_spec = parts[0];
    let output_spec = parts[1];
    let operand_specs: Vec<&str> = input_spec.split(',').collect();

    // Collect all unique indices
    let mut all_indices = std::collections::HashSet::new();
    for spec in &operand_specs {
        for ch in spec.chars() {
            if ch.is_alphabetic() {
                all_indices.insert(ch);
            }
        }
    }

    // Determine output indices
    let output_indices: Vec<char> = output_spec.chars().filter(|c| c.is_alphabetic()).collect();

    // Determine summation indices (those not in output)
    let summation_indices: Vec<char> = all_indices
        .iter()
        .filter(|&&idx| !output_indices.contains(&idx))
        .copied()
        .collect();

    // Map indices to dimensions for each operand
    let mut index_sizes = std::collections::HashMap::new();

    for (op_idx, &operand) in operands.iter().enumerate() {
        let spec = operand_specs[op_idx];
        let shape = operand.shape();

        for (dim_idx, idx_char) in spec.chars().enumerate() {
            if idx_char.is_alphabetic() {
                let size = shape[dim_idx];

                // Check consistency
                if let Some(&existing_size) = index_sizes.get(&idx_char) {
                    if existing_size != size {
                        return Err(NumRs2Error::DimensionMismatch(format!(
                            "Index '{}' has inconsistent sizes: {} and {}",
                            idx_char, existing_size, size
                        )));
                    }
                } else {
                    index_sizes.insert(idx_char, size);
                }
            }
        }
    }

    // Determine output shape. Every output index must appear in at least
    // one input operand's spec (its size is otherwise undefined); indexing
    // `index_sizes` directly would panic on a HashMap miss for a
    // free/undeclared output index (e.g. subscripts like "ij->ik" where
    // 'k' never appears on the input side).
    let output_shape: Vec<usize> = output_indices
        .iter()
        .map(|&idx| {
            index_sizes.get(&idx).copied().ok_or_else(|| {
                NumRs2Error::InvalidOperation(format!(
                    "einsum output index '{}' does not appear in any input operand",
                    idx
                ))
            })
        })
        .collect::<Result<Vec<usize>>>()?;

    // Handle scalar output case
    let output_shape = if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    };

    // Create output array
    let mut result = Array::zeros(&output_shape);

    // Compute einsum using nested loops
    // This is a simple but inefficient implementation
    // A production implementation would optimize loop order and use blocking

    let total_output_size: usize = output_shape.iter().product();

    // `result` is write-only across this whole loop (exactly one `.set()`
    // per output element, never read back), so bulk-acquiring once here
    // replaces `total_output_size` `Arc::make_mut` calls with one.
    let result_arr = result.array_mut();

    for output_idx in 0..total_output_size {
        // Convert linear index to multi-dimensional indices for output
        let mut output_multi_idx = vec![0; output_shape.len()];
        let mut temp = output_idx;
        for i in (0..output_shape.len()).rev() {
            output_multi_idx[i] = temp % output_shape[i];
            temp /= output_shape[i];
        }

        // Map output indices to their values. This must happen
        // unconditionally: `output_multi_idx[i]` is always a valid index
        // for `idx_char` regardless of that axis's *size*. The previous
        // `output_shape[0] != 1` guard skipped population whenever the
        // FIRST output axis happened to have size 1 (e.g. "ij,jk->ik"
        // with i=1), even though later output axes (or operands sharing
        // one of these index letters) still needed the mapping -- any
        // operand referencing an unmapped index then panicked on the
        // `index_values[&idx_char]` lookup below.
        let mut index_values = std::collections::HashMap::new();
        for (i, &idx_char) in output_indices.iter().enumerate() {
            index_values.insert(idx_char, output_multi_idx[i]);
        }

        // Sum over all combinations of summation indices
        let mut sum = T::zero();

        // Calculate ranges for summation indices
        let summation_ranges: Vec<usize> = summation_indices
            .iter()
            .map(|&idx| index_sizes[&idx])
            .collect();

        if summation_ranges.is_empty() {
            // No summation needed, just multiply the elements
            let mut product = T::one();

            for (op_idx, &operand) in operands.iter().enumerate() {
                let spec = operand_specs[op_idx];
                let op_shape = operand.shape();

                // Build indices for this operand
                let mut op_indices = vec![0; op_shape.len()];
                for (dim_idx, idx_char) in spec.chars().enumerate() {
                    if idx_char.is_alphabetic() {
                        op_indices[dim_idx] = index_values[&idx_char];
                    }
                }

                product = product * operand.get(&op_indices)?;
            }

            sum += product;
        } else {
            // Iterate over all combinations of summation indices
            let total_summation_size: usize = summation_ranges.iter().product();

            for sum_idx in 0..total_summation_size {
                // Convert linear index to multi-dimensional indices for summation
                let mut sum_multi_idx = vec![0; summation_ranges.len()];
                let mut temp = sum_idx;
                for i in (0..summation_ranges.len()).rev() {
                    sum_multi_idx[i] = temp % summation_ranges[i];
                    temp /= summation_ranges[i];
                }

                // Update index values with summation indices
                for (i, &idx_char) in summation_indices.iter().enumerate() {
                    index_values.insert(idx_char, sum_multi_idx[i]);
                }

                // Compute product for this combination
                let mut product = T::one();

                for (op_idx, &operand) in operands.iter().enumerate() {
                    let spec = operand_specs[op_idx];
                    let op_shape = operand.shape();

                    // Build indices for this operand
                    let mut op_indices = vec![0; op_shape.len()];
                    for (dim_idx, idx_char) in spec.chars().enumerate() {
                        if idx_char.is_alphabetic() {
                            op_indices[dim_idx] = index_values[&idx_char];
                        }
                    }

                    product = product * operand.get(&op_indices)?;
                }

                sum += product;
            }
        }

        // Store result
        if output_shape[0] == 1 && output_shape.len() == 1 {
            // Scalar output
            result_arr[[0]] = sum;
        } else {
            *result_arr
                .get_mut(output_multi_idx.as_slice())
                .ok_or_else(|| {
                    NumRs2Error::IndexOutOfBounds(format!(
                        "Failed to set element at indices {:?}",
                        output_multi_idx
                    ))
                })? = sum;
        }
    }

    // If output was scalar, reshape to remove the dummy dimension
    Ok(result)
}

/// Compute the Kronecker product of two arrays
///
/// The Kronecker product is a matrix operation that takes two matrices A (m×n) and B (p×q)
/// and produces a matrix of size (mp)×(nq). Each element `A[i,j]` is multiplied by the entire
/// matrix B and placed at the appropriate block position in the result.
///
/// # Arguments
/// * `a` - First input array (must be 2D)
/// * `b` - Second input array (must be 2D)
///
/// # Returns
/// * `Result<Array<T>>` - The Kronecker product of the two input arrays
///
/// # Errors
/// * `DimensionMismatch` - If either input is not a 2D array
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensor_ops::kron;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let result = kron(&a, &b).expect("kron should succeed for valid 2D arrays");
/// // Result is a 4×4 matrix
/// ```
pub fn kron<T: Float + Clone + Debug>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    // Check that both inputs are 2D arrays
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "kron requires two 2D arrays".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Output shape is (a_rows * b_rows, a_cols * b_cols)
    let out_shape = [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]];
    let mut result = Array::zeros(&out_shape);

    // Extract the data
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let result_data = result.array_mut().as_slice_mut().ok_or_else(|| {
        NumRs2Error::ComputationError("array should have contiguous memory layout".to_string())
    })?;

    // Compute Kronecker product
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            let a_idx = i * a_shape[1] + j;
            let a_val = a_data[a_idx];

            // For each element in A, multiply by entire B matrix
            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    let b_idx = k * b_shape[1] + l;
                    let b_val = b_data[b_idx];

                    // Position in result array
                    let row = i * b_shape[0] + k;
                    let col = j * b_shape[1] + l;
                    let result_idx = row * out_shape[1] + col;

                    result_data[result_idx] = a_val * b_val;
                }
            }
        }
    }

    Ok(result)
}

/// Compute tensor dot product of two arrays along specified axes
///
/// The tensor dot product contracts specified axes of two tensors. It generalizes
/// matrix multiplication to higher-dimensional arrays by summing over specified axes.
///
/// # Arguments
/// * `a` - First input array
/// * `b` - Second input array  
/// * `axes` - Array of axes to contract (must have exactly 2 elements)
///
/// # Returns
/// * `Result<Array<T>>` - The tensor dot product result
///
/// # Errors
/// * `InvalidOperation` - If axes array doesn't have exactly 2 elements
/// * `DimensionMismatch` - If input arrays are not 2D or axes are out of bounds
/// * `ShapeMismatch` - If the contracted dimensions don't match
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::linalg::tensor_ops::tensordot;
///
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
/// let result = tensordot(&a, &b, &[1, 0]).expect("tensordot should succeed"); // Contract axis 1 of a with axis 0 of b
/// ```
pub fn tensordot<T: Float + Clone + Debug + 'static>(
    a: &Array<T>,
    b: &Array<T>,
    axes: &[usize],
) -> Result<Array<T>> {
    // Simplified version for 2 axes
    if axes.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "This implementation of tensordot only supports 2 axes".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_axis = axes[0];
    let b_axis = axes[1];

    if a_axis >= a_shape.len() || b_axis >= b_shape.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Axis out of bounds".to_string(),
        ));
    }

    // Check that contracted dimensions match
    if a_shape[a_axis] != b_shape[b_axis] {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![a_shape[a_axis]],
            actual: vec![b_shape[b_axis]],
        });
    }

    // Fast path: the common "matrix multiplication" shape (both operands
    // 2-D, contracting `a`'s last axis against `b`'s first) needs no
    // reshaping at all.
    if a_shape.len() == 2 && b_shape.len() == 2 && a_axis == 1 && b_axis == 0 {
        return a.matmul(b);
    }

    // General case: works for any dimensionality and any single-axis-pair
    // contraction. Move the contracted axis of `a` to its last position and
    // of `b` to its first, reshape both down to 2-D (every other axis
    // flattens in place, keeping its existing relative order), multiply,
    // then reshape the 2-D product back out to the combined shape. This is
    // the standard reduction of `tensordot` to `matmul`, and it handles a
    // 1-D `a` or `b` correctly too: the "other axes" side of that operand
    // then reshapes to a length-1 dimension, matching a plain
    // vector-matrix product (verified against `np.tensordot` for a 3-D x
    // 3-D contraction over non-trivial axes).
    //
    // This also replaces the old `a_axis == 0 && b_axis == 1` special case,
    // which transposed its `matmul` result one time too many: that
    // combination's contraction already lands in the right
    // (a`s-other-axes, b's-other-axes) order with no further transpose
    // needed, so the old code silently returned the correctly-computed
    // result with its two axes swapped (confirmed against
    // `np.tensordot(a, b, axes=([0], [1]))` on a non-square example).
    let a_moved = crate::array_ops::axis_ops::moveaxis(a, &[a_axis], &[a_shape.len() - 1])?;
    let b_moved = crate::array_ops::axis_ops::moveaxis(b, &[b_axis], &[0])?;

    let a_moved_shape = a_moved.shape();
    let b_moved_shape = b_moved.shape();
    let contracted_dim = a_moved_shape[a_moved_shape.len() - 1];
    let a_rows: usize = a_moved_shape[..a_moved_shape.len() - 1].iter().product();
    let b_cols: usize = b_moved_shape[1..].iter().product();

    let a_2d = a_moved.try_reshape(&[a_rows, contracted_dim])?;
    let b_2d = b_moved.try_reshape(&[contracted_dim, b_cols])?;
    let result_2d = a_2d.matmul(&b_2d)?;

    let mut out_shape: Vec<usize> = a_moved_shape[..a_moved_shape.len() - 1].to_vec();
    out_shape.extend_from_slice(&b_moved_shape[1..]);

    if out_shape.is_empty() {
        // Both operands were 1-D: a full vector-vector contraction. This
        // crate's scalar convention (see e.g. `einsum`'s "i,i->" case
        // above) is a length-1 array rather than a true 0-D one.
        result_2d.try_reshape(&[1])
    } else {
        result_2d.try_reshape(&out_shape)
    }
}

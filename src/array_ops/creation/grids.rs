//! Grid and mesh creation functions
//!
//! This module provides functions to create coordinate grids and meshes:
//! - `meshgrid` - Create coordinate matrices from coordinate vectors
//! - `mgrid` - Create a dense multi-dimensional meshgrid
//! - `ogrid` - Create an open (sparse) multi-dimensional meshgrid

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;

/// Create coordinate matrices from coordinate vectors
///
/// Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields
/// over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.
///
/// # Parameters
///
/// * `xi` - 1-D arrays representing the coordinates of a grid
/// * `indexing` - Cartesian ('xy', default) or matrix ('ij') indexing of output
/// * `sparse` - If true, return sparse output arrays
///
/// # Returns
///
/// For vectors x1, x2,..., xn with lengths Ni=len(xi), returns (N1, N2, N3,..., Nn) shaped arrays
/// if indexing='ij' or (N2, N1, N3,..., Nn) shaped arrays if indexing='xy' with the elements of xi
/// repeated to fill the matrix along the first dimension for x1, the second for x2 and so on.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::meshgrid;
///
/// // Create 2D coordinate matrices
/// let x = Array::from_vec(vec![0.0, 0.5, 1.0]);
/// let y = Array::from_vec(vec![0.0, 0.67, 1.33, 2.0]);
/// let grids = meshgrid(&[&x, &y], "xy", false).expect("operation should succeed");
/// let (xx, yy) = (&grids[0], &grids[1]);
///
/// assert_eq!(xx.shape(), vec![4, 3]); // Note: transposed for 'xy' indexing
/// assert_eq!(yy.shape(), vec![4, 3]);
/// ```
pub fn meshgrid<T>(xi: &[&Array<T>], indexing: &str, sparse: bool) -> Result<Vec<Array<T>>>
where
    T: Clone + num_traits::Zero + num_traits::One,
{
    if xi.is_empty() {
        return Ok(vec![]);
    }

    // Validate indexing parameter
    let use_matrix_indexing = match indexing {
        "ij" => true,
        "xy" => false,
        _ => {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Invalid indexing '{}'. Use 'xy' or 'ij'",
                indexing
            )))
        }
    };

    // Get dimensions
    let ndim = xi.len();
    let mut shape: Vec<usize> = xi.iter().map(|arr| arr.shape()[0]).collect();

    // For 'xy' indexing, swap first two dimensions
    if !use_matrix_indexing && ndim >= 2 {
        shape.swap(0, 1);
    }

    let mut grids = Vec::with_capacity(ndim);

    if sparse {
        // Create sparse grids - each grid has values only along its axis
        for (axis_idx, &arr) in xi.iter().enumerate() {
            let mut grid_shape = vec![1; ndim];

            if use_matrix_indexing {
                // For 'ij' indexing, axis_idx corresponds directly to shape index
                grid_shape[axis_idx] = arr.shape()[0];
            } else {
                // For 'xy' indexing, swap first two dimensions
                if ndim >= 2 {
                    if axis_idx == 0 {
                        grid_shape[1] = arr.shape()[0];
                    } else if axis_idx == 1 {
                        grid_shape[0] = arr.shape()[0];
                    } else {
                        grid_shape[axis_idx] = arr.shape()[0];
                    }
                } else {
                    grid_shape[axis_idx] = arr.shape()[0];
                }
            }

            let grid = arr.clone().reshape(&grid_shape);
            grids.push(grid);
        }
    } else {
        // Create full grids
        for (axis_idx, &arr) in xi.iter().enumerate() {
            let mut grid = Array::zeros(&shape);
            let arr_data = arr.to_vec();

            // Fill the grid by repeating values along appropriate dimensions
            let total_elements: usize = shape.iter().product();
            let mut indices = vec![0; ndim];

            for linear_idx in 0..total_elements {
                // Convert linear index to multi-dimensional indices
                let mut temp = linear_idx;
                for i in (0..ndim).rev() {
                    indices[i] = temp % shape[i];
                    temp /= shape[i];
                }

                // Determine which value from the input array to use
                let src_idx = if !use_matrix_indexing && ndim >= 2 {
                    // For xy indexing, swap interpretation of first two indices
                    if axis_idx == 0 {
                        indices[1]
                    } else if axis_idx == 1 {
                        indices[0]
                    } else {
                        indices[axis_idx]
                    }
                } else {
                    indices[axis_idx]
                };

                grid.set(&indices, arr_data[src_idx].clone())?;
            }

            grids.push(grid);
        }
    }

    Ok(grids)
}

/// Create a dense multi-dimensional "meshgrid"
///
/// Returns arrays representing coordinates of a multi-dimensional grid.
/// Similar to `meshgrid`, but with a more convenient interface for generating
/// equally spaced grids.
///
/// # Parameters
///
/// * `slices` - A vector of slice specifications. Each slice can be:
///   - A tuple of (start, stop, num) for equally spaced values
///   - A tuple of (start, stop, step) where step is negative to indicate step size
///
/// # Returns
///
/// Vector of arrays where each array represents coordinates along one dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::mgrid;
///
/// // Create a 2D grid from 0 to 1 with 3 points in each dimension
/// let grids = mgrid(&[(0.0, 1.0, 3.0), (0.0, 1.0, 3.0)]).expect("operation should succeed");
/// assert_eq!(grids.len(), 2);
/// assert_eq!(grids[0].shape(), vec![3, 3]);
/// assert_eq!(grids[1].shape(), vec![3, 3]);
/// ```
pub fn mgrid<T>(slices: &[(T, T, T)]) -> Result<Vec<Array<T>>>
where
    T: Float + Clone + PartialOrd + num_traits::FromPrimitive + 'static,
{
    use crate::array::Array;
    use crate::math::linspace;

    if slices.is_empty() {
        return Ok(vec![]);
    }

    // Create coordinate arrays for each dimension
    let mut coord_arrays = Vec::with_capacity(slices.len());

    for &(start, stop, num_or_step) in slices {
        let arr = if num_or_step > T::one() && num_or_step == num_or_step.floor() {
            // If num_or_step is an integer > 1, treat as number of points
            let num = num_or_step.to_usize().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Cannot convert num to usize".to_string())
            })?;
            linspace(start, stop, num)
        } else {
            // Otherwise treat as step size
            let step = num_or_step;
            if step == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "Step size cannot be zero".to_string(),
                ));
            }

            // Generate points with the step
            let mut points = Vec::new();
            let mut current = start;

            if step > T::zero() {
                while current <= stop {
                    points.push(current);
                    current = current + step;
                }
            } else if step < T::zero() {
                while current >= stop {
                    points.push(current);
                    current = current + step;
                }
            }
            Array::from_vec(points)
        };

        coord_arrays.push(arr);
    }

    // Use meshgrid with ij indexing to create the full grids
    meshgrid(&coord_arrays.iter().collect::<Vec<_>>(), "ij", false)
}

/// Create an open multi-dimensional "meshgrid"
///
/// Returns arrays representing coordinates of a multi-dimensional grid,
/// but with shape (1, 1, ..., 1, n_i, 1, ..., 1) for the i-th dimension.
/// This is memory-efficient for operations that support broadcasting.
///
/// # Parameters
///
/// * `slices` - A vector of slice specifications. Each slice can be:
///   - A tuple of (start, stop, num) for equally spaced values
///   - A tuple of (start, stop, step) where step is negative to indicate step size
///
/// # Returns
///
/// Vector of arrays where each array has values only along its respective dimension
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::creation::ogrid;
///
/// // Create a sparse 2D grid from 0 to 1 with 3 points in each dimension
/// let grids = ogrid(&[(0.0, 1.0, 3.0), (0.0, 1.0, 3.0)]).expect("operation should succeed");
/// assert_eq!(grids.len(), 2);
/// assert_eq!(grids[0].shape(), vec![3, 1]); // Values along first dimension
/// assert_eq!(grids[1].shape(), vec![1, 3]); // Values along second dimension
/// ```
pub fn ogrid<T>(slices: &[(T, T, T)]) -> Result<Vec<Array<T>>>
where
    T: Float + Clone + PartialOrd + num_traits::FromPrimitive + 'static,
{
    use crate::array::Array;
    use crate::math::linspace;

    if slices.is_empty() {
        return Ok(vec![]);
    }

    // Create coordinate arrays for each dimension
    let mut coord_arrays = Vec::with_capacity(slices.len());

    for &(start, stop, num_or_step) in slices {
        let arr = if num_or_step > T::one() && num_or_step == num_or_step.floor() {
            // If num_or_step is an integer > 1, treat as number of points
            let num = num_or_step.to_usize().ok_or_else(|| {
                NumRs2Error::InvalidOperation("Cannot convert num to usize".to_string())
            })?;
            linspace(start, stop, num)
        } else {
            // Otherwise treat as step size
            let step = num_or_step;
            if step == T::zero() {
                return Err(NumRs2Error::InvalidOperation(
                    "Step size cannot be zero".to_string(),
                ));
            }

            // Generate points with the step
            let mut points = Vec::new();
            let mut current = start;

            if step > T::zero() {
                while current <= stop {
                    points.push(current);
                    current = current + step;
                }
            } else if step < T::zero() {
                while current >= stop {
                    points.push(current);
                    current = current + step;
                }
            }
            Array::from_vec(points)
        };

        coord_arrays.push(arr);
    }

    // Use meshgrid with ij indexing and sparse=true
    meshgrid(&coord_arrays.iter().collect::<Vec<_>>(), "ij", true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meshgrid() {
        // Test 2D meshgrid with xy indexing
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0]);

        let grids = meshgrid(&[&x, &y], "xy", false).expect("operation should succeed");
        assert_eq!(grids.len(), 2);

        let xx = &grids[0];
        let yy = &grids[1];

        assert_eq!(xx.shape(), vec![2, 3]); // Transposed for xy
        assert_eq!(yy.shape(), vec![2, 3]);

        // Check xx values (x repeated along rows)
        assert_eq!(xx.get(&[0, 0]).expect("operation should succeed"), 1.0);
        assert_eq!(xx.get(&[0, 1]).expect("operation should succeed"), 2.0);
        assert_eq!(xx.get(&[0, 2]).expect("operation should succeed"), 3.0);
        assert_eq!(xx.get(&[1, 0]).expect("operation should succeed"), 1.0);
        assert_eq!(xx.get(&[1, 1]).expect("operation should succeed"), 2.0);
        assert_eq!(xx.get(&[1, 2]).expect("operation should succeed"), 3.0);

        // Check yy values (y repeated along columns)
        assert_eq!(yy.get(&[0, 0]).expect("operation should succeed"), 4.0);
        assert_eq!(yy.get(&[0, 1]).expect("operation should succeed"), 4.0);
        assert_eq!(yy.get(&[0, 2]).expect("operation should succeed"), 4.0);
        assert_eq!(yy.get(&[1, 0]).expect("operation should succeed"), 5.0);
        assert_eq!(yy.get(&[1, 1]).expect("operation should succeed"), 5.0);
        assert_eq!(yy.get(&[1, 2]).expect("operation should succeed"), 5.0);

        // Test with ij indexing
        let grids_ij = meshgrid(&[&x, &y], "ij", false).expect("operation should succeed");
        let xx_ij = &grids_ij[0];
        let yy_ij = &grids_ij[1];

        assert_eq!(xx_ij.shape(), vec![3, 2]); // Not transposed for ij
        assert_eq!(yy_ij.shape(), vec![3, 2]);

        // Test sparse meshgrid
        let sparse_grids = meshgrid(&[&x, &y], "xy", true).expect("operation should succeed");
        assert_eq!(sparse_grids.len(), 2);
        assert_eq!(sparse_grids[0].shape(), vec![1, 3]);
        assert_eq!(sparse_grids[1].shape(), vec![2, 1]);
    }

    #[test]
    fn test_mgrid() {
        use approx::assert_relative_eq;

        // Test 2D mgrid with number of points
        let grids = mgrid(&[(0.0, 1.0, 3.0), (0.0, 2.0, 3.0)]).expect("operation should succeed");
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 3]);
        assert_eq!(grids[1].shape(), vec![3, 3]);

        // Check first grid (x coordinates)
        assert_relative_eq!(
            grids[0].get(&[0, 0]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[0, 1]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[0, 2]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[1, 0]).expect("operation should succeed"),
            0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[2, 0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );

        // Check second grid (y coordinates)
        assert_relative_eq!(
            grids[1].get(&[0, 0]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[0, 1]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[0, 2]).expect("operation should succeed"),
            2.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[1, 0]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[2, 2]).expect("operation should succeed"),
            2.0,
            epsilon = 1e-10
        );

        // Test with step size
        let grids_step =
            mgrid(&[(0.0, 1.0, 0.5), (0.0, 2.0, 1.0)]).expect("operation should succeed");
        assert_eq!(grids_step.len(), 2);
        assert_eq!(grids_step[0].shape(), vec![3, 3]);
        assert_eq!(grids_step[1].shape(), vec![3, 3]);

        // Test 1D mgrid
        let grids_1d = mgrid(&[(0.0, 2.0, 5.0)]).expect("operation should succeed");
        assert_eq!(grids_1d.len(), 1);
        assert_eq!(grids_1d[0].shape(), vec![5]);

        // Test empty input
        let grids_empty = mgrid::<f64>(&[]).expect("operation should succeed");
        assert_eq!(grids_empty.len(), 0);
    }

    #[test]
    fn test_ogrid() {
        use approx::assert_relative_eq;

        // Test 2D ogrid (sparse grid)
        let grids = ogrid(&[(0.0, 1.0, 3.0), (0.0, 2.0, 3.0)]).expect("operation should succeed");
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), vec![3, 1]); // Values along first dimension
        assert_eq!(grids[1].shape(), vec![1, 3]); // Values along second dimension

        // Check first grid (x coordinates)
        assert_relative_eq!(
            grids[0].get(&[0, 0]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[1, 0]).expect("operation should succeed"),
            0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[0].get(&[2, 0]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );

        // Check second grid (y coordinates)
        assert_relative_eq!(
            grids[1].get(&[0, 0]).expect("operation should succeed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[0, 1]).expect("operation should succeed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            grids[1].get(&[0, 2]).expect("operation should succeed"),
            2.0,
            epsilon = 1e-10
        );

        // Test 3D ogrid
        let grids_3d = ogrid(&[(0.0, 1.0, 2.0), (0.0, 1.0, 2.0), (0.0, 1.0, 2.0)])
            .expect("operation should succeed");
        assert_eq!(grids_3d.len(), 3);
        assert_eq!(grids_3d[0].shape(), vec![2, 1, 1]);
        assert_eq!(grids_3d[1].shape(), vec![1, 2, 1]);
        assert_eq!(grids_3d[2].shape(), vec![1, 1, 2]);

        // Test with step size
        let grids_step =
            ogrid(&[(0.0, 1.0, 0.5), (0.0, 2.0, 1.0)]).expect("operation should succeed");
        assert_eq!(grids_step.len(), 2);
        assert_eq!(grids_step[0].shape(), vec![3, 1]);
        assert_eq!(grids_step[1].shape(), vec![1, 3]);
    }
}

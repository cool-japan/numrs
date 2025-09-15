use crate::array::Array;
use crate::array_ops::manipulation::ravel;
use crate::error::{NumRs2Error, Result};
use std::fmt::Display;

/// One-dimensional linear interpolation.
///
/// Returns the one-dimensional piecewise linear interpolant to a function
/// with given discrete data points (xp, fp), evaluated at x.
///
/// # Parameters
///
/// * `x` - The x-coordinates at which to evaluate the interpolated values.
/// * `xp` - The x-coordinates of the data points, must be increasing.
/// * `fp` - The y-coordinates of the data points, same length as `xp`.
/// * `left` - Value to return for `x < xp[0]`. If not provided, defaults to `fp[0]`.
/// * `right` - Value to return for `x > xp[last]`. If not provided, defaults to `fp[last]`.
/// * `period` - A period for the x-coordinates. This parameter allows making the interpolation periodic in the specified period.
///
/// # Returns
///
/// The interpolated values, same shape as `x`.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::conditional::interp;
///
/// let xp = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let fp = Array::from_vec(vec![3.0, 2.0, 0.0]);
/// let x = Array::from_vec(vec![0.0, 1.5, 2.0, 2.5, 3.0, 4.0]);
///
/// // Without explicitly specifying `left` and `right`
/// let y = interp(&x, &xp, &fp, None, None, None).unwrap();
/// assert_eq!(y.to_vec(), vec![3.0, 2.5, 2.0, 1.0, 0.0, 0.0]);
///
/// // With explicit `left` and `right` values
/// let y = interp(&x, &xp, &fp, Some(-5.0), Some(-1.0), None).unwrap();
/// assert_eq!(y.to_vec(), vec![-5.0, 2.5, 2.0, 1.0, 0.0, -1.0]);
/// ```
pub fn interp<T>(
    x: &Array<T>,
    xp: &Array<T>,
    fp: &Array<T>,
    left: Option<T>,
    right: Option<T>,
    period: Option<T>,
) -> Result<Array<T>>
where
    T: Clone
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float,
{
    // Validate inputs
    if xp.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "xp must be 1-dimensional".into(),
        ));
    }
    if fp.ndim() != 1 {
        return Err(NumRs2Error::DimensionMismatch(
            "fp must be 1-dimensional".into(),
        ));
    }
    if xp.len() != fp.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "xp and fp must have the same length".into(),
        ));
    }
    if xp.len() < 2 {
        return Err(NumRs2Error::ValueError(
            "xp and fp must have at least 2 elements".into(),
        ));
    }

    // Check if xp is strictly increasing
    for i in 1..xp.len() {
        if xp.get(&[i])? <= xp.get(&[i - 1])? {
            return Err(NumRs2Error::ValueError(
                "xp must be strictly increasing".into(),
            ));
        }
    }

    // Save original shape of x for reshaping result later
    let x_shape = x.shape().clone();

    // Flatten x for processing
    let x_flat = ravel(x, None)?;
    let mut result = Array::zeros(&x_flat.shape());

    // Get default left and right values
    let left_val = left.unwrap_or_else(|| fp.get(&[0]).unwrap());
    let right_val = right.unwrap_or_else(|| fp.get(&[fp.len() - 1]).unwrap());

    // Process each element in x
    for i in 0..x_flat.len() {
        let mut x_val = x_flat.get(&[i])?;

        // Handle periodicity if specified
        if let Some(ref p) = period {
            let p_val = *p;
            let xp_min = xp.get(&[0])?;
            let xp_max = xp.get(&[xp.len() - 1])?;
            let period_width = xp_max - xp_min;

            // Normalize x_val to be within [xp_min, xp_min + period)
            let mut x_norm = x_val;
            if x_norm >= xp_min + period_width || x_norm < xp_min {
                x_norm = xp_min + ((x_norm - xp_min) % p_val + p_val) % p_val;
            }
            x_val = x_norm;
        }

        // Out of bounds handling
        if x_val < xp.get(&[0])? {
            result.set(&[i], left_val)?;
            continue;
        }
        if x_val > xp.get(&[xp.len() - 1])? {
            result.set(&[i], right_val)?;
            continue;
        }

        // Binary search to find the interval containing x_val
        let mut low: usize = 0;
        let mut high: usize = xp.len() - 1;

        while low < high - 1 {
            let mid = (low + high) / 2;
            if x_val < xp.get(&[mid])? {
                high = mid;
            } else {
                low = mid;
            }
        }

        // Linear interpolation within the interval
        let x0 = xp.get(&[low])?;
        let x1 = xp.get(&[high])?;
        let y0 = fp.get(&[low])?;
        let y1 = fp.get(&[high])?;

        let t = (x_val - x0) / (x1 - x0);
        let interpolated = y0 * (T::one() - t) + y1 * t;

        result.set(&[i], interpolated)?;
    }

    // Reshape result back to original shape of x
    Ok(result.reshape(&x_shape))
}

/// Return elements chosen from x or y depending on condition
///
/// # Parameters
///
/// * `condition` - Where True, yield x, otherwise yield y
/// * `x` - Values to choose from where condition is True
/// * `y` - Values to choose from where condition is False
///
/// # Returns
///
/// A new array with values chosen from x or y based on condition
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let condition = Array::from_vec(vec![true, false, true, false]);
/// let x = Array::from_vec(vec![1, 2, 3, 4]);
/// let y = Array::from_vec(vec![10, 20, 30, 40]);
/// let result = where_cond(&condition, &x, &y).unwrap();
/// assert_eq!(result.to_vec(), vec![1, 20, 3, 40]);
///
/// // With broadcasting
/// let condition_2d = Array::from_vec(vec![true, false, true, false]).reshape(&[2, 2]);
/// let x_scalar = Array::from_vec(vec![100]);
/// let y_2d = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
/// let result_2d = where_cond(&condition_2d, &x_scalar, &y_2d).unwrap();
/// assert_eq!(result_2d.to_vec(), vec![100, 2, 100, 4]);
/// ```
pub fn where_cond<T: Clone + Display>(
    condition: &Array<bool>,
    x: &Array<T>,
    y: &Array<T>,
) -> Result<Array<T>> {
    // Get the shapes
    let cond_shape = condition.shape();
    let x_shape = x.shape();
    let y_shape = y.shape();

    // Calculate broadcast shape for all three arrays
    let broadcast_shape_xy = Array::<T>::broadcast_shape(&x_shape, &y_shape)?;
    let broadcast_shape = Array::<bool>::broadcast_shape(&cond_shape, &broadcast_shape_xy)?;

    // Broadcast all arrays to the common shape
    let cond_broadcast = condition.broadcast_to(&broadcast_shape)?;
    let x_broadcast = x.broadcast_to(&broadcast_shape)?;
    let y_broadcast = y.broadcast_to(&broadcast_shape)?;

    // Apply the conditional logic element-wise
    let cond_data = cond_broadcast.to_vec();
    let x_data = x_broadcast.to_vec();
    let y_data = y_broadcast.to_vec();

    let result_data: Vec<T> = cond_data
        .iter()
        .zip(x_data.iter())
        .zip(y_data.iter())
        .map(
            |((&cond, x_val), y_val)| {
                if cond {
                    x_val.clone()
                } else {
                    y_val.clone()
                }
            },
        )
        .collect();

    Ok(Array::from_vec(result_data).reshape(&broadcast_shape))
}

/// Select elements from choices array based on conditions
///
/// Given a list of conditions and a list of choices, return an array drawn from the elements in choices,
/// depending on the conditions.
///
/// # Parameters
///
/// * `condlist` - A list of boolean arrays. The length of condlist determines the number of conditions
/// * `choicelist` - A list of arrays from which to choose. Must have the same length as condlist
/// * `default` - The element to use if no condition is satisfied. If None, uses zero.
///
/// # Returns
///
/// A new array with elements selected from choicelist based on conditions
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create conditions
/// let x = Array::from_vec(vec![0, 1, 2, 3, 4, 5]);
/// let cond1 = x.map(|val| val < 3);
/// let cond2 = x.map(|val| val >= 3);
///
/// // Create choices
/// let choice1 = Array::from_vec(vec![10, 10, 10, 10, 10, 10]);
/// let choice2 = Array::from_vec(vec![20, 20, 20, 20, 20, 20]);
///
/// let result = select(&[&cond1, &cond2], &[&choice1, &choice2], Some(99)).unwrap();
/// assert_eq!(result.to_vec(), vec![10, 10, 10, 20, 20, 20]);
///
/// // When no condition matches, use default
/// let always_false = Array::from_vec(vec![false, false, false]);
/// let choice_unused = Array::from_vec(vec![1, 2, 3]);
/// let result_default = select(&[&always_false], &[&choice_unused], Some(99)).unwrap();
/// assert_eq!(result_default.to_vec(), vec![99, 99, 99]);
/// ```
pub fn select<T: Clone + num_traits::Zero>(
    condlist: &[&Array<bool>],
    choicelist: &[&Array<T>],
    default: Option<T>,
) -> Result<Array<T>> {
    if condlist.len() != choicelist.len() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and choicelist must have the same length".to_string(),
        ));
    }

    if condlist.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and choicelist cannot be empty".to_string(),
        ));
    }

    // Determine the broadcast shape
    let mut broadcast_shape = condlist[0].shape();
    for cond in condlist.iter() {
        broadcast_shape = Array::<bool>::broadcast_shape(&broadcast_shape, &cond.shape())?;
    }
    for choice in choicelist.iter() {
        broadcast_shape = Array::<T>::broadcast_shape(&broadcast_shape, &choice.shape())?;
    }

    // Broadcast all arrays to the common shape
    let mut cond_broadcasts = Vec::with_capacity(condlist.len());
    let mut choice_broadcasts = Vec::with_capacity(choicelist.len());

    for cond in condlist.iter() {
        cond_broadcasts.push(cond.broadcast_to(&broadcast_shape)?);
    }
    for choice in choicelist.iter() {
        choice_broadcasts.push(choice.broadcast_to(&broadcast_shape)?);
    }

    // Create result array with default values
    let default_val = default.unwrap_or_else(T::zero);
    let mut result = Array::full(&broadcast_shape, default_val);

    // Process each element
    let total_size = broadcast_shape.iter().product::<usize>();
    for i in 0..total_size {
        // Convert flat index to multi-dimensional index
        let mut indices = Vec::with_capacity(broadcast_shape.len());
        let mut temp = i;
        for &dim in broadcast_shape.iter().rev() {
            indices.insert(0, temp % dim);
            temp /= dim;
        }

        // Check conditions in order
        for (cond_broadcast, choice_broadcast) in
            cond_broadcasts.iter().zip(choice_broadcasts.iter())
        {
            let cond_val = cond_broadcast
                .array()
                .get(ndarray::IxDyn(&indices))
                .unwrap();
            if *cond_val {
                let choice_val = choice_broadcast
                    .array()
                    .get(ndarray::IxDyn(&indices))
                    .unwrap();
                result.set(&indices, choice_val.clone())?;
                break; // Take the first matching condition
            }
        }
    }

    Ok(result)
}

/// Construct an array from an index array and a list of arrays to choose from
///
/// Given an array of indices and a list of choices, construct a new array where each element
/// is taken from the choice array corresponding to the index at that position.
///
/// # Parameters
///
/// * `a` - Array of indices. Each value must be in the range 0..choices.len()
/// * `choices` - List of arrays to choose from. All arrays must be broadcastable to the same shape
/// * `mode` - How to handle out-of-bounds indices: "raise" (error), "clip" (clip to bounds), or "wrap" (wrap around)
///
/// # Returns
///
/// A new array with the same shape as `a`, where each element is chosen from the corresponding choice array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Simple example with 1D arrays
/// let indices = Array::from_vec(vec![0, 1, 0, 2]);
/// let choices = vec![
///     Array::from_vec(vec![10, 11, 12, 13]),
///     Array::from_vec(vec![20, 21, 22, 23]),
///     Array::from_vec(vec![30, 31, 32, 33]),
/// ];
/// let result = choose(&indices, &choices.iter().collect::<Vec<_>>(), "raise").unwrap();
/// assert_eq!(result.to_vec(), vec![10, 21, 12, 33]);
///
/// // Example with broadcasting
/// let indices = Array::from_vec(vec![0, 1, 1, 0]);
/// let choices = vec![
///     Array::from_vec(vec![5]),  // will broadcast
///     Array::from_vec(vec![7]),  // will broadcast
/// ];
/// let result = choose(&indices, &choices.iter().collect::<Vec<_>>(), "raise").unwrap();
/// assert_eq!(result.to_vec(), vec![5, 7, 7, 5]);
/// ```
pub fn choose<T: Clone + num_traits::Zero>(
    a: &Array<usize>,
    choices: &[&Array<T>],
    mode: &str,
) -> Result<Array<T>> {
    if choices.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "choices cannot be empty".to_string(),
        ));
    }

    let n_choices = choices.len();

    // Find the broadcast shape for all choice arrays and the index array
    let mut broadcast_shape = a.shape();
    for choice in choices.iter() {
        broadcast_shape = Array::<T>::broadcast_shape(&broadcast_shape, &choice.shape())?;
    }

    // Broadcast all choice arrays to the common shape
    let mut choice_broadcasts = Vec::with_capacity(n_choices);
    for choice in choices.iter() {
        choice_broadcasts.push(choice.broadcast_to(&broadcast_shape)?);
    }

    // Create result array
    let mut result_data = Vec::with_capacity(a.len());

    // Process each index
    for (i, &idx) in a.to_vec().iter().enumerate() {
        let actual_idx = match mode {
            "raise" => {
                if idx >= n_choices {
                    return Err(NumRs2Error::IndexOutOfBounds(format!(
                        "index {} is out of bounds for choices of size {}",
                        idx, n_choices
                    )));
                }
                idx
            }
            "clip" => {
                if idx >= n_choices {
                    n_choices - 1
                } else {
                    idx
                }
            }
            "wrap" => idx % n_choices,
            _ => {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Invalid mode '{}'. Use 'raise', 'clip', or 'wrap'",
                    mode
                )))
            }
        };

        // Get element from the appropriate choice array
        let chosen_array = &choice_broadcasts[actual_idx];

        // Calculate multi-dimensional indices from flat index i
        let mut indices = Vec::with_capacity(a.ndim());
        let mut temp = i;
        for &dim in a.shape().iter().rev() {
            indices.insert(0, temp % dim);
            temp /= dim;
        }

        // Get value from chosen array
        let value = chosen_array.get(&indices)?;
        result_data.push(value);
    }

    Ok(Array::from_vec(result_data).reshape(&a.shape()))
}

/// Evaluate a piecewise-defined function
///
/// Given a set of conditions and corresponding functions, evaluate the piecewise
/// function defined by:
///   - where condition[0] is True: func[0](x)
///   - where condition[1] is True: func[1](x)
///   - ...
///   - otherwise: fill_value
///
/// # Parameters
///
/// * `x` - The input array
/// * `condlist` - List of boolean arrays or functions that return boolean arrays when called with x
/// * `funclist` - List of functions to apply where the corresponding condition is True.
///   Each function must accept x and return an array of the same shape
/// * `fill_value` - Value to use where no condition is True (default is 0)
///
/// # Returns
///
/// A new array with the same shape as x, with values computed by the piecewise function
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::array_ops::conditional::piecewise;
///
/// // Simple piecewise constant function
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let cond1 = x.map(|val| val < 2.0);
/// let cond2 = x.map(|val| val >= 2.0);
///
/// let func = |arr: &Array<f64>| arr.map(|x| x * 2.0);
/// let result = piecewise(&x, &[&cond1, &cond2], &[&func, &func], Some(0.0)).unwrap();
/// // Should double values where conditions are met
/// ```
pub fn piecewise<T, F>(
    x: &Array<T>,
    condlist: &[&Array<bool>],
    funclist: &[&F],
    fill_value: Option<T>,
) -> Result<Array<T>>
where
    T: Clone + num_traits::Zero,
    F: Fn(&Array<T>) -> Array<T>,
{
    if condlist.len() != funclist.len() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and funclist must have the same length".to_string(),
        ));
    }

    if condlist.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "condlist and funclist cannot be empty".to_string(),
        ));
    }

    // Verify all conditions have compatible shapes with x
    for cond in condlist {
        if cond.shape() != x.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x.shape(),
                actual: cond.shape(),
            });
        }
    }

    // Initialize result with fill value
    let fill_val = fill_value.unwrap_or_else(T::zero);
    let mut result = Array::full(&x.shape(), fill_val);
    let mut mask_used = Array::full(&x.shape(), false);

    // Apply functions where conditions are true
    for (cond, func) in condlist.iter().zip(funclist.iter()) {
        let func_result = func(x);

        // Verify function result has correct shape
        if func_result.shape() != x.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x.shape(),
                actual: func_result.shape(),
            });
        }

        // Apply function result where condition is true and we haven't already set a value
        let cond_data = cond.to_vec();
        let mask_data = mask_used.to_vec();
        let func_data = func_result.to_vec();
        let mut result_data = result.to_vec();

        for i in 0..cond_data.len() {
            if cond_data[i] && !mask_data[i] {
                result_data[i] = func_data[i].clone();
            }
        }

        // Update result and mask
        result = Array::from_vec(result_data).reshape(&x.shape());

        // Update mask to track which elements have been set
        let mut new_mask_data = mask_used.to_vec();
        for i in 0..cond_data.len() {
            if cond_data[i] {
                new_mask_data[i] = true;
            }
        }
        mask_used = Array::from_vec(new_mask_data).reshape(&x.shape());
    }

    Ok(result)
}

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use rayon::prelude::*;

/// Utilities for parallel computation
pub fn parallel_map<T, U, F>(array: &Array<T>, f: F) -> Array<U>
where
    T: Send + Sync + Clone,
    U: Send + Clone,
    F: Fn(T) -> U + Send + Sync,
{
    let vec_data = array.to_vec();
    let result: Vec<U> = vec_data.par_iter().map(|x| f(x.clone())).collect();

    let shape = array.shape();
    Array::from_vec(result).reshape(&shape)
}

/// Memory layout optimization utilities
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
}

/// Optimizes the memory layout of an array for a specific operation
pub fn optimize_layout<T: Clone>(array: &Array<T>, layout: MemoryLayout) -> Array<T> {
    // In a real implementation, this would convert between row-major and column-major layouts
    // For this example, we'll just return a clone of the array
    match layout {
        MemoryLayout::RowMajor => array.clone(),
        MemoryLayout::ColumnMajor => {
            // For column-major, we could transpose and use specialized algorithms
            // Here we just return the original array for simplicity
            array.clone()
        }
    }
}

/// Checks if an array can be operated on in-place
pub fn can_operate_inplace<T>(_array: &Array<T>) -> bool {
    // This would check if the array is contiguous and has the right memory layout
    // For this example, we'll just return true
    true
}

/// Broadcasting utilities
pub fn broadcast_arrays<T: Clone>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    // Determine the broadcast shape
    let mut broadcast_shape = Vec::new();
    for array in arrays {
        let shape = array.shape();
        if broadcast_shape.is_empty() {
            broadcast_shape = shape.clone();
        } else {
            // Compute the broadcast shape
            let mut new_shape = Vec::new();
            let max_dims = broadcast_shape.len().max(shape.len());

            // Pad shapes with 1s
            let padded_a = pad_shape(&broadcast_shape, max_dims);
            let padded_b = pad_shape(&shape, max_dims);

            // Compute the broadcast shape
            for i in 0..max_dims {
                let dim_a = padded_a[i];
                let dim_b = padded_b[i];

                if dim_a == 1 {
                    new_shape.push(dim_b);
                } else if dim_b == 1 || dim_a == dim_b {
                    new_shape.push(dim_a);
                } else {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: broadcast_shape,
                        actual: shape.clone(),
                    });
                }
            }

            broadcast_shape = new_shape;
        }
    }

    // Broadcast each array to the broadcast shape
    let mut result = Vec::new();
    for array in arrays {
        result.push(broadcast_to(array, &broadcast_shape)?);
    }

    Ok(result)
}

fn pad_shape(shape: &[usize], target_len: usize) -> Vec<usize> {
    let mut padded = vec![1; target_len];
    let offset = target_len - shape.len();
    for (i, &dim) in shape.iter().enumerate() {
        padded[i + offset] = dim;
    }
    padded
}

fn broadcast_to<T: Clone>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    let orig_shape = array.shape();

    // Check if broadcasting is possible
    if orig_shape == shape {
        return Ok(array.clone());
    }

    let padded_orig = pad_shape(&orig_shape, shape.len());

    for i in 0..shape.len() {
        if padded_orig[i] != 1 && padded_orig[i] != shape[i] {
            return Err(NumRs2Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: orig_shape,
            });
        }
    }

    // For this example, we'll create a new array with the broadcast shape
    // In a real implementation, we'd use ndarray's broadcasting capabilities
    let orig_data = array.to_vec();
    let mut result_data = Vec::new();

    // This is a simplified broadcasting implementation
    // A real implementation would be more efficient
    let size: usize = shape.iter().product();
    result_data.reserve(size);

    // Create a simple mapping from result indices to original indices
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut orig_strides = vec![1; padded_orig.len()];
    for i in (0..padded_orig.len() - 1).rev() {
        orig_strides[i] = orig_strides[i + 1] * padded_orig[i + 1];
    }

    // Fill the result array
    for i in 0..size {
        let mut orig_idx = 0;
        let mut idx = i;

        for j in 0..shape.len() {
            let dim_idx = idx / strides[j];
            idx %= strides[j];

            if padded_orig[j] > 1 {
                orig_idx += dim_idx * orig_strides[j];
            }
        }

        result_data.push(orig_data[orig_idx].clone());
    }

    Ok(Array::from_vec(result_data).reshape(shape))
}

/// Type conversion utilities
pub fn astype<T: Clone, U: Clone + From<T>>(array: &Array<T>) -> Array<U> {
    let data = array.to_vec();
    let converted: Vec<U> = data.into_iter().map(U::from).collect();
    Array::from_vec(converted).reshape(&array.shape())
}

// Specialized optimizations for common operations
pub fn fast_sum<T: Float + Send + Sync>(array: &Array<T>) -> T {
    let data = array.to_vec();
    data.par_iter().cloned().reduce(|| T::zero(), |a, b| a + b)
}

/// Check if a value is a scalar (0-dimensional)
///
/// # Arguments
///
/// * `value` - Any value to check
///
/// # Returns
///
/// True if the value is a scalar, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::util::isscalar;
///
/// assert!(isscalar(&5.0));
/// assert!(isscalar(&42));
/// assert!(!isscalar(&Array::from_vec(vec![1, 2, 3])));
/// ```
pub fn isscalar<T>(_value: &T) -> bool {
    // In NumPy, a scalar is a 0-dimensional value
    // For Rust, we consider primitive types as scalars
    std::mem::size_of::<T>() <= 16 && !std::any::type_name::<T>().starts_with("numrs2::")
}

/// Check if a given array is a scalar (0-dimensional array)
///
/// # Arguments
///
/// * `array` - Array to check
///
/// # Returns
///
/// True if the array is 0-dimensional, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::util::isscalar_array;
///
/// let scalar_array = Array::from_vec(vec![42.0]).reshape(&[]);
/// let vector_array = Array::from_vec(vec![1.0, 2.0, 3.0]);
///
/// assert!(isscalar_array(&scalar_array));
/// assert!(!isscalar_array(&vector_array));
/// ```
pub fn isscalar_array<T: Clone>(array: &Array<T>) -> bool {
    array.ndim() == 0
}

/// Check if a data type can be cast to another data type
///
/// # Arguments
///
/// * `from_type` - Source type name
/// * `to_type` - Target type name
/// * `casting` - Casting rule ("no", "equiv", "safe", "same_kind", "unsafe")
///
/// # Returns
///
/// True if the cast is allowed under the given casting rule, false otherwise
///
/// # Examples
///
/// ```
/// use numrs2::util::can_cast;
///
/// assert!(can_cast("i32", "f64", "safe"));
/// assert!(!can_cast("f64", "i32", "safe"));
/// assert!(can_cast("f64", "i32", "unsafe"));
/// ```
pub fn can_cast(from_type: &str, to_type: &str, casting: &str) -> bool {
    match casting {
        "no" => from_type == to_type,
        "equiv" => from_type == to_type,
        "safe" => {
            // Define safe casting rules
            matches!(
                (from_type, to_type),
                ("i8", "i16" | "i32" | "i64" | "f32" | "f64")
                    | ("i16", "i32" | "i64" | "f32" | "f64")
                    | ("i32", "i64" | "f64")
                    | ("i64", "f64")
                    | (
                        "u8",
                        "u16" | "u32" | "u64" | "i16" | "i32" | "i64" | "f32" | "f64"
                    )
                    | ("u16", "u32" | "u64" | "i32" | "i64" | "f32" | "f64")
                    | ("u32", "u64" | "i64" | "f64")
                    | ("u64", "f64")
                    | ("f32", "f64")
            ) || from_type == to_type
        }
        "same_kind" => {
            // Allow within same kind (int->int, float->float, etc.)
            let from_kind = get_type_kind(from_type);
            let to_kind = get_type_kind(to_type);
            from_kind == to_kind || can_cast(from_type, to_type, "safe")
        }
        "unsafe" => true, // Allow all casts
        _ => false,
    }
}

/// Get the kind of a type (integer, float, complex, etc.)
fn get_type_kind(type_name: &str) -> &str {
    match type_name {
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "integer",
        "f32" | "f64" => "float",
        "bool" => "bool",
        _ => "other",
    }
}

/// Find the common type that can represent all input types
///
/// # Arguments
///
/// * `types` - Slice of type names
///
/// # Returns
///
/// The common type name that can represent all input types
///
/// # Examples
///
/// ```
/// use numrs2::util::common_type;
///
/// assert_eq!(common_type(&["i32", "f32"]), "f32");
/// assert_eq!(common_type(&["i32", "i64"]), "i64");
/// assert_eq!(common_type(&["f32", "f64"]), "f64");
/// ```
pub fn common_type(types: &[&str]) -> &'static str {
    if types.is_empty() {
        return "f64"; // Default
    }

    let mut result = find_common_type_static(types[0]);
    for &type_name in &types[1..] {
        result = find_common_type(result, type_name);
    }
    result
}

/// Find the common type between two types
fn find_common_type(type1: &str, type2: &str) -> &'static str {
    if type1 == type2 {
        return find_common_type_static(type1);
    }

    // Type promotion hierarchy
    let promotion_order = [
        "bool", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64",
    ];

    let pos1 = promotion_order.iter().position(|&x| x == type1);
    let pos2 = promotion_order.iter().position(|&x| x == type2);

    match (pos1, pos2) {
        (Some(p1), Some(p2)) => promotion_order[p1.max(p2)],
        _ => "f64", // Default to f64 for unknown types
    }
}

/// Convert a type name to static string
fn find_common_type_static(type_name: &str) -> &'static str {
    match type_name {
        "bool" => "bool",
        "i8" => "i8",
        "u8" => "u8",
        "i16" => "i16",
        "u16" => "u16",
        "i32" => "i32",
        "u32" => "u32",
        "i64" => "i64",
        "u64" => "u64",
        "f32" => "f32",
        "f64" => "f64",
        _ => "f64",
    }
}

/// Determine the result type of an operation on given types
///
/// # Arguments
///
/// * `types` - Slice of type names involved in the operation
///
/// # Returns
///
/// The result type name
///
/// # Examples
///
/// ```
/// use numrs2::util::result_type;
///
/// assert_eq!(result_type(&["i32", "f32"]), "f32");
/// assert_eq!(result_type(&["i32", "i64"]), "i64");
/// ```
pub fn result_type(types: &[&str]) -> &'static str {
    common_type(types)
}

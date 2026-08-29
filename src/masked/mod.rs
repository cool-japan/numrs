//! `numpy.ma`-style masked arrays.
//!
//! [`MaskedArray<T>`] pairs a data [`Array<T>`] with a boolean mask
//! [`Array<bool>`] of the same shape (`true` = masked / invalid) and an
//! optional fill value. This module is split across several files, all
//! part of the same `masked` module and all operating on the same
//! [`MaskedArray`] type -- splitting does not change any public path
//! (`numrs2::masked::MaskedArray`, re-exported at the crate root as
//! `numrs2::prelude::MaskedArray`, is unaffected by which physical file a
//! given inherent method lives in):
//!
//! - `mod.rs` (this file) -- the struct itself, construction, shape/mask
//!   accessors, `get`/`set`, `reshape`/`transpose`, `Display`/`Debug`.
//! - `ops` -- arithmetic operators (`Add`/`Sub`/`Mul`/`Div`, all
//!   mask-propagating) and the element-wise comparison methods
//!   (`equal`/`not_equal`/`less_than`/`less_equal`/`greater_than`/
//!   `greater_equal`, NumPy's `eq`/`ne`/`lt`/`le`/`gt`/`ge`).
//! - `reductions` -- the shared axis-walking engine plus every
//!   value-producing reduction: `mean`/`sum`/`min`/`max` (whole-array,
//!   pre-existing, `Option<T>`) and their `_axis`-suffixed
//!   `axis`+`keepdims`-aware siblings, plus the brand-new `std`/`var`/
//!   `prod`/`median`/`ptp`.
//! - `bool_ops` -- `any`/`all`, scoped to `MaskedArray<bool>`.
//! - `search` -- `argmin`/`argmax`/`cumsum`/`sort`.
//! - `linalg` -- `dot`/`concatenate`.

mod bool_ops;
mod linalg;
mod ops;
mod reductions;
mod search;

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::fmt;
use std::fmt::Debug;

/// Represents a masked array with data and a boolean mask
///
/// # Type Parameters
///
/// * `T` - The type of the data in the array
///
/// # Fields
///
/// * `data` - The underlying array of data
/// * `mask` - Boolean mask where true values indicate masked (invalid) elements
/// * `fill_value` - Value used to fill masked elements when accessed or for certain operations
#[derive(Clone)]
pub struct MaskedArray<T> {
    data: Array<T>,
    mask: Array<bool>,
    fill_value: T,
}

impl<T: Clone> MaskedArray<T> {
    /// Create a new masked array from data and mask arrays
    ///
    /// # Arguments
    ///
    /// * `data` - The data array
    /// * `mask` - Optional mask array (same shape as data). If None, all elements are valid (unmasked).
    /// * `fill_value` - Optional value to use for masked elements. If None, a default value is used.
    ///
    /// # Returns
    ///
    /// A new MaskedArray instance
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create data array
    /// let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    ///
    /// // Create mask for some elements (true = masked)
    /// let mask = Array::from_vec(vec![false, true, false, true, false]);
    ///
    /// // Create masked array
    /// let masked = MaskedArray::new(data, Some(mask), Some(0.0));
    /// ```
    pub fn new(data: Array<T>, mask: Option<Array<bool>>, fill_value: Option<T>) -> Result<Self>
    where
        T: Clone + Default,
    {
        let shape = data.shape();
        let mask_array = match mask {
            Some(m) => {
                if m.shape() != shape {
                    return Err(NumRs2Error::ShapeMismatch {
                        expected: shape,
                        actual: m.shape(),
                    });
                }
                m
            }
            None => Array::from_vec_shape(vec![false; data.size()], &shape)?,
        };

        let fill_val = fill_value.unwrap_or_default();

        Ok(Self {
            data,
            mask: mask_array,
            fill_value: fill_val,
        })
    }

    /// Create a masked array from a regular array with specified values masked
    ///
    /// # Arguments
    ///
    /// * `data` - The data array
    /// * `value` - Value to mask in the array
    /// * `fill_value` - Optional value to use for masked elements
    ///
    /// # Returns
    ///
    /// A new MaskedArray with elements equal to `value` masked
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create data array
    /// let data = Array::from_vec(vec![1.0, 2.0, -999.0, 4.0, -999.0]);
    ///
    /// // Create masked array with -999.0 values masked
    /// let masked = MaskedArray::masked_values(data, -999.0, Some(0.0))
    ///     .expect("masked_values should succeed");
    /// ```
    pub fn masked_values(data: Array<T>, value: T, fill_value: Option<T>) -> Result<Self>
    where
        T: Clone + Default + PartialEq,
    {
        let shape = data.shape();
        let mut mask_vec = Vec::with_capacity(data.size());

        // Create mask where elements equal to value are masked
        for elem in data.array().iter() {
            mask_vec.push(*elem == value);
        }

        let mask_array = Array::from_vec_shape(mask_vec, &shape)?;
        let fill_val = fill_value.unwrap_or_default();

        Ok(Self {
            data: data.clone(),
            mask: mask_array,
            fill_value: fill_val,
        })
    }

    /// Create a masked array from a regular array with invalid values masked
    ///
    /// This is a placeholder since Rust doesn't have a direct equivalent to NaN/Inf without generic constraints.
    /// Implementation would need to be specialized for floating-point types.
    ///
    /// # Arguments
    ///
    /// * `data` - The data array
    /// * `fill_value` - Optional value to use for masked elements
    ///
    /// # Returns
    ///
    /// A new MaskedArray with invalid elements masked
    pub fn masked_invalid(data: Array<f64>, fill_value: Option<f64>) -> Result<MaskedArray<f64>> {
        let shape = data.shape();
        let mut mask_vec = Vec::with_capacity(data.size());

        // Create mask where elements are NaN or Inf
        for &elem in data.array().iter() {
            mask_vec.push(elem.is_nan() || elem.is_infinite());
        }

        let mask_array = Array::from_vec_shape(mask_vec, &shape)?;
        let fill_val = fill_value.unwrap_or(0.0);

        Ok(MaskedArray {
            data: data.clone(),
            mask: mask_array,
            fill_value: fill_val,
        })
    }

    /// Create a masked array based on a condition
    ///
    /// # Arguments
    ///
    /// * `data` - The data array
    /// * `condition` - Boolean array of the same shape as data, where True values will be masked
    /// * `fill_value` - Optional value to use for masked elements
    ///
    /// # Returns
    ///
    /// A new MaskedArray with elements where condition is True masked
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create data array
    /// let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    ///
    /// // Create condition (mask elements > 3.0)
    /// let condition = data.map(|x| x > 3.0);
    ///
    /// // Create masked array with elements > 3.0 masked
    /// let masked = MaskedArray::masked_where(data, condition, Some(0.0))
    ///     .expect("masked_where should succeed with matching shapes");
    /// ```
    pub fn masked_where(
        data: Array<T>,
        condition: Array<bool>,
        fill_value: Option<T>,
    ) -> Result<Self>
    where
        T: Clone + Default,
    {
        if data.shape() != condition.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: data.shape(),
                actual: condition.shape(),
            });
        }

        let fill_val = fill_value.unwrap_or_default();

        Ok(Self {
            data: data.clone(),
            mask: condition,
            fill_value: fill_val,
        })
    }

    /// Create a masked array with all elements masked
    ///
    /// # Arguments
    ///
    /// * `data` - The data array
    /// * `fill_value` - Optional value to use for masked elements
    ///
    /// # Returns
    ///
    /// A new MaskedArray with all elements masked
    pub fn masked_all(data: Array<T>, fill_value: Option<T>) -> Result<Self>
    where
        T: Clone + Default,
    {
        let shape = data.shape();
        let mask_array = Array::from_vec_shape(vec![true; data.size()], &shape)?;
        let fill_val = fill_value.unwrap_or_default();

        Ok(Self {
            data: data.clone(),
            mask: mask_array,
            fill_value: fill_val,
        })
    }

    /// Get the underlying data array
    pub fn get_data(&self) -> &Array<T> {
        &self.data
    }

    /// Get the mask array
    pub fn get_mask(&self) -> &Array<bool> {
        &self.mask
    }

    /// Get the fill value
    pub fn get_fill_value(&self) -> T {
        self.fill_value.clone()
    }

    /// Set the fill value
    pub fn set_fill_value(&mut self, value: T) {
        self.fill_value = value;
    }

    /// Get the shape of the array
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Get the number of masked (invalid) elements
    pub fn count_masked(&self) -> usize {
        self.mask.array().iter().filter(|&&x| x).count()
    }

    /// Get the number of unmasked (valid) elements
    pub fn count_valid(&self) -> usize {
        self.size() - self.count_masked()
    }

    /// Return a copy of the array with masked values filled with the fill_value
    ///
    /// # Arguments
    ///
    /// * `fill_value` - Optional value to use for masked elements. If None, uses the array's fill_value.
    ///
    /// # Returns
    ///
    /// A regular Array with masked values replaced by the fill value
    pub fn filled(&self, fill_value: Option<T>) -> Array<T>
    where
        T: Clone,
    {
        let fill_val = fill_value.unwrap_or_else(|| self.fill_value.clone());
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut filled_vec = Vec::with_capacity(self.size());

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if *is_masked {
                filled_vec.push(fill_val.clone());
            } else {
                filled_vec.push(value.clone());
            }
        }

        Array::from_vec_shape(filled_vec, &self.shape()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Return a regular array of valid data (compressed to remove masked elements)
    ///
    /// # Returns
    ///
    /// An Array containing only the valid (unmasked) elements
    pub fn compressed(&self) -> Array<T>
    where
        T: Clone,
    {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let mut compressed_vec = Vec::new();

        for (value, is_masked) in data_op.iter().zip(mask_op.iter()) {
            if !*is_masked {
                compressed_vec.push(value.clone());
            }
        }

        Array::from_vec(compressed_vec)
    }

    /// Create a new MaskedArray with the mask hardened
    ///
    /// After hardening, masks cannot be changed
    ///
    /// # Returns
    ///
    /// A new MaskedArray with hardened mask
    pub fn harden_mask(&self) -> Self
    where
        T: Clone,
    {
        // In NumPy, this sets an internal flag that is consulted when masks are set
        // Here, we'll just make a copy to represent this concept
        self.clone()
    }

    /// Create a new MaskedArray with the mask softened
    ///
    /// After softening, masks can be changed
    ///
    /// # Returns
    ///
    /// A new MaskedArray with softened mask
    pub fn soften_mask(&self) -> Self
    where
        T: Clone,
    {
        // In NumPy, this sets an internal flag that is consulted when masks are set
        // Here, we'll just make a copy to represent this concept
        self.clone()
    }

    /// Get a value at the specified indices
    ///
    /// If the value is masked, returns the fill value
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices for each dimension
    ///
    /// # Returns
    ///
    /// The value at the specified indices, or the fill value if masked
    pub fn get(&self, indices: &[usize]) -> Result<T>
    where
        T: Clone,
    {
        // Check if indices are within bounds
        if indices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Check if indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx,
                    i,
                    self.shape()[i]
                )));
            }
        }

        // Check if the element is masked by accessing mask array directly
        let mask_array = self.mask.array();
        let mask_value = mask_array.get(indices).ok_or_else(|| {
            NumRs2Error::IndexOutOfBounds(format!("Failed to get mask at indices {:?}", indices))
        })?;

        if *mask_value {
            // Return the fill value
            Ok(self.fill_value.clone())
        } else {
            // Return the actual value by accessing data array directly
            let data_array = self.data.array();
            let data_value = data_array.get(indices).ok_or_else(|| {
                NumRs2Error::IndexOutOfBounds(format!(
                    "Failed to get data at indices {:?}",
                    indices
                ))
            })?;

            Ok(data_value.clone())
        }
    }

    /// Set a value at the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices for each dimension
    /// * `value` - The value to set
    /// * `mask` - Optional boolean indicating whether to mask this element
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn set(&mut self, indices: &[usize], value: T, mask: Option<bool>) -> Result<()>
    where
        T: Clone,
    {
        // Set the data value
        self.data.set(indices, value)?;

        // Update mask if provided
        if let Some(mask_value) = mask {
            self.mask.set(indices, mask_value)?;
        }

        Ok(())
    }

    /// Reshape the MaskedArray
    ///
    /// # Arguments
    ///
    /// * `shape` - New shape for the array
    ///
    /// # Returns
    ///
    /// A new MaskedArray with the same data but reshaped
    pub fn reshape(&self, shape: &[usize]) -> Self
    where
        T: Clone,
    {
        MaskedArray {
            data: self.data.reshape(shape),
            mask: self.mask.reshape(shape),
            fill_value: self.fill_value.clone(),
        }
    }

    /// Transpose the MaskedArray
    ///
    /// # Returns
    ///
    /// A new MaskedArray with dimensions reversed
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        MaskedArray {
            data: self.data.transpose(),
            mask: self.mask.transpose(),
            fill_value: self.fill_value.clone(),
        }
    }
}

// Display implementation for MaskedArray
impl<T: Clone + fmt::Display + Debug> fmt::Display for MaskedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data_op = crate::kernels::borrow::operand(&self.data);
        let mask_op = crate::kernels::borrow::operand(&self.mask);
        let shape = self.shape();

        writeln!(f, "MaskedArray(")?;

        // Simple display for 1D arrays
        if shape.len() == 1 {
            write!(f, "[")?;
            for (i, (val, &masked)) in data_op.iter().zip(mask_op.iter()).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if masked {
                    write!(f, "--")?;
                } else {
                    write!(f, "{}", val)?;
                }
            }
            writeln!(f, "]")?;
        } else {
            // More complex display for higher dimensions
            writeln!(f, "Shape: {:?}", shape)?;
            writeln!(f, "Masked count: {}", self.count_masked())?;
        }

        write!(f, "Fill value: {}", self.fill_value)?;

        Ok(())
    }
}

// Debug implementation for MaskedArray
impl<T: Clone + Debug> fmt::Debug for MaskedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MaskedArray")
            .field("shape", &self.shape())
            .field("masked_count", &self.count_masked())
            .field("fill_value", &self.fill_value)
            .finish()
    }
}

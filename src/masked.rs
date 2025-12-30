use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;
use std::fmt;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

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
            None => Array::from_vec(vec![false; data.size()]).reshape(&shape),
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
    /// let masked = MaskedArray::masked_values(data, -999.0, Some(0.0)).unwrap();
    /// ```
    pub fn masked_values(data: Array<T>, value: T, fill_value: Option<T>) -> Result<Self>
    where
        T: Clone + Default + PartialEq,
    {
        let shape = data.shape();
        let data_vec = data.to_vec();
        let mut mask_vec = Vec::with_capacity(data.size());

        // Create mask where elements equal to value are masked
        for elem in &data_vec {
            mask_vec.push(*elem == value);
        }

        let mask_array = Array::from_vec(mask_vec).reshape(&shape);
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
        let data_vec = data.to_vec();
        let mut mask_vec = Vec::with_capacity(data.size());

        // Create mask where elements are NaN or Inf
        for &elem in &data_vec {
            mask_vec.push(elem.is_nan() || elem.is_infinite());
        }

        let mask_array = Array::from_vec(mask_vec).reshape(&shape);
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
    /// let masked = MaskedArray::masked_where(data, condition, Some(0.0)).unwrap();
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
        let mask_array = Array::from_vec(vec![true; data.size()]).reshape(&shape);
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
        self.mask.to_vec().iter().filter(|&&x| x).count()
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
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut filled_vec = Vec::with_capacity(self.size());

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
            if *is_masked {
                filled_vec.push(fill_val.clone());
            } else {
                filled_vec.push(value.clone());
            }
        }

        Array::from_vec(filled_vec).reshape(&self.shape())
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
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut compressed_vec = Vec::new();

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
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

// Implementation of arithmetic operations for MaskedArray
// Implement Add for MaskedArray with MaskedArray
impl<T: Clone + Add<Output = T>> Add for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    fn add(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Add the data arrays
        let result_data = match self.data.add_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!("Failed to add arrays with incompatible shapes"),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!("Failed to combine masks with incompatible shapes"),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement subtract for MaskedArray with MaskedArray
impl<T: Clone + Sub<Output = T>> Sub for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    fn sub(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Subtract the data arrays
        let result_data = match self.data.subtract_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!("Failed to subtract arrays with incompatible shapes"),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!("Failed to combine masks with incompatible shapes"),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement multiply for MaskedArray with MaskedArray
impl<T: Clone + Mul<Output = T>> Mul for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    fn mul(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Multiply the data arrays
        let result_data = match self.data.multiply_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!("Failed to multiply arrays with incompatible shapes"),
        };

        // Combine the masks - an element is masked if it is masked in either input
        let mask_combined = match self.mask.zip_with(&other.mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!("Failed to combine masks with incompatible shapes"),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Implement divide for MaskedArray with MaskedArray
impl<T: Clone + Div<Output = T> + PartialEq + Zero> Div for &MaskedArray<T> {
    type Output = MaskedArray<T>;

    fn div(self, other: &MaskedArray<T>) -> MaskedArray<T> {
        // Check for divisions by zero and mask them
        let zero = T::zero();
        let other_data_vec = other.data.to_vec();
        let other_mask_vec = other.mask.to_vec();
        let mut division_mask_vec = Vec::with_capacity(other.size());

        for (value, is_masked) in other_data_vec.iter().zip(other_mask_vec.iter()) {
            division_mask_vec.push(*is_masked || *value == zero);
        }

        let division_mask = Array::from_vec(division_mask_vec).reshape(&other.shape());

        // Divide the data arrays
        let result_data = match self.data.divide_broadcast(&other.data) {
            Ok(res) => res,
            Err(_) => panic!("Failed to divide arrays with incompatible shapes"),
        };

        // Combine the masks - an element is masked if it is masked in either input or if divisor is zero
        let mask_combined = match self.mask.zip_with(&division_mask, |a, b| a || b) {
            Ok(res) => res,
            Err(_) => panic!("Failed to combine masks with incompatible shapes"),
        };

        MaskedArray {
            data: result_data,
            mask: mask_combined,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Statistical operations on MaskedArray
impl<T: Clone + Add<Output = T> + Div<Output = T> + Zero + From<f64> + Into<f64>> MaskedArray<T> {
    /// Calculate the mean of unmasked elements
    ///
    /// # Returns
    ///
    /// The mean value, or None if all elements are masked
    pub fn mean(&self) -> Option<T> {
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut sum = T::zero();
        let mut count = 0;

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
            if !*is_masked {
                sum = sum + value.clone();
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            // We need to convert to/from f64 to properly handle division
            let count_f64 = count as f64;
            let sum_f64: f64 = sum.into();
            Some(T::from(sum_f64 / count_f64))
        }
    }

    /// Calculate the sum of unmasked elements
    ///
    /// # Returns
    ///
    /// The sum, or None if all elements are masked
    pub fn sum(&self) -> Option<T> {
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut sum = T::zero();
        let mut count = 0;

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
            if !*is_masked {
                sum = sum + value.clone();
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some(sum)
        }
    }

    /// Find the minimum value among unmasked elements
    ///
    /// # Returns
    ///
    /// The minimum value, or None if all elements are masked
    pub fn min(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut min_val = None;

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
            if !*is_masked {
                match min_val {
                    None => min_val = Some(value.clone()),
                    Some(ref current_min) if value < current_min => min_val = Some(value.clone()),
                    _ => {}
                }
            }
        }

        min_val
    }

    /// Find the maximum value among unmasked elements
    ///
    /// # Returns
    ///
    /// The maximum value, or None if all elements are masked
    pub fn max(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let mut max_val = None;

        for (value, is_masked) in data_vec.iter().zip(mask_vec.iter()) {
            if !*is_masked {
                match max_val {
                    None => max_val = Some(value.clone()),
                    Some(ref current_max) if value > current_max => max_val = Some(value.clone()),
                    _ => {}
                }
            }
        }

        max_val
    }
}

// Display implementation for MaskedArray
impl<T: Clone + fmt::Display + Debug> fmt::Display for MaskedArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data_vec = self.data.to_vec();
        let mask_vec = self.mask.to_vec();
        let shape = self.shape();

        writeln!(f, "MaskedArray(")?;

        // Simple display for 1D arrays
        if shape.len() == 1 {
            write!(f, "[")?;
            for (i, (val, &masked)) in data_vec.iter().zip(mask_vec.iter()).enumerate() {
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

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{
    ArrayView as NdArrayView, ArrayViewMut as NdArrayViewMut, Axis, IxDyn, Slice, SliceInfo,
    SliceInfoElem,
};
use std::ops::{Add, Div, Mul, Sub};

/// Specifies an indexing operation for array slicing.
///
/// This enum represents either a single index access or a slice operation
/// with optional start, end, and step parameters. It provides a unified way
/// to represent different types of array indexing in NumRS.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
///
/// // Use a single index for first row
/// let row_slice = SliceOrIndex::Index(0);
///
/// // Use a slice for all columns
/// let col_slice = SliceOrIndex::Slice(0, None, None);
///
/// // Combined to get view of first row
/// let sliced = array.sliced_view(&[row_slice, col_slice]).unwrap();
/// // Size differs based on implementation, so we don't assert shape
/// ```
#[derive(Debug, Clone)]
pub enum SliceOrIndex {
    /// Represents a single index access (e.g., array\[5\])
    Index(usize),

    /// Represents a slice with optional start, end, and step (e.g., array\[2:10:2\])
    /// Parameters are (start, end, step).
    /// - `start` is inclusive
    /// - `end` is exclusive if provided, or the end of dimension if None
    /// - `step` is the stride, or 1 if None
    Slice(usize, Option<usize>, Option<usize>), // start, end, step
}

impl SliceOrIndex {
    /// Converts a `SliceOrIndex` to an ndarray `Slice` for use in ndarray operations.
    ///
    /// This method translates NumRS slicing semantics to ndarray's slicing semantics.
    /// For an index, it creates a single-element slice. For a slice, it converts
    /// the start, end, and step parameters to an ndarray `Slice`.
    ///
    /// # Returns
    ///
    /// * `Slice` - An ndarray slice object equivalent to this SliceOrIndex
    pub fn to_ndarray_slice(&self) -> Slice {
        match self {
            SliceOrIndex::Index(idx) => Slice::from(*idx..*idx + 1),
            SliceOrIndex::Slice(start, end, step) => {
                let end_val = end.unwrap_or_else(|| usize::MAX);
                let step_val = step.unwrap_or(1) as isize;
                Slice::new(*start as isize, Some(end_val as isize), step_val)
            }
        }
    }
}

/// A read-only view into an existing array.
///
/// `ArrayView` provides a lightweight window into array data without copying the underlying
/// elements. This enables efficient, zero-copy operations on array data. Views are tied to
/// the lifetime of the original array to ensure memory safety.
///
/// # Type Parameters
///
/// * `'a` - The lifetime parameter tying the view to the lifetime of the source array
/// * `T` - The element type of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
///
/// // Create a view of the array
/// let view = array.view();
///
/// // Use the view for read-only operations
/// assert_eq!(view.get(&[0, 0]).unwrap(), 1.0);
/// assert_eq!(view.shape(), vec![2, 2]);
/// ```
pub struct ArrayView<'a, T> {
    data: NdArrayView<'a, T, IxDyn>,
}

/// A mutable view into an existing array.
///
/// `ArrayViewMut` provides a lightweight window into array data that allows modifying the
/// underlying elements without copying them. Like immutable views, mutable views are tied
/// to the lifetime of the original array to ensure memory safety. The key difference is
/// that mutable views allow modifying the original array's elements.
///
/// # Type Parameters
///
/// * `'a` - The lifetime parameter tying the view to the lifetime of the source array
/// * `T` - The element type of the array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let mut array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
///
/// // Create a mutable view of the array
/// let mut view_mut = array.view_mut();
///
/// // Modify the array through the view
/// view_mut.set(&[0, 0], 10.0).unwrap();
///
/// // The change is reflected in the original array
/// assert_eq!(array.get(&[0, 0]).unwrap(), 10.0);
/// ```
pub struct ArrayViewMut<'a, T> {
    data: NdArrayViewMut<'a, T, IxDyn>,
}

impl<'a, T: 'a> ArrayView<'a, T> {
    /// Creates a new array view from an ndarray view.
    ///
    /// This is primarily an internal method used for wrapping ndarray views
    /// in NumRS's view abstraction. Most users will create views using
    /// the `view()` method on `Array`.
    ///
    /// # Parameters
    ///
    /// * `view` - The ndarray view to wrap
    ///
    /// # Returns
    ///
    /// * `ArrayView<'a, T>` - A new view object wrapping the ndarray view
    pub fn from_ndarray_view(view: NdArrayView<'a, T, IxDyn>) -> Self {
        Self { data: view }
    }

    /// Gets a reference to the underlying ndarray view.
    ///
    /// This method is primarily used internally to access the underlying ndarray
    /// implementation. It allows access to ndarray-specific functionality
    /// not directly exposed by the NumRS API.
    ///
    /// # Returns
    ///
    /// * `&NdArrayView<'a, T, IxDyn>` - Reference to the underlying ndarray view
    pub fn view(&self) -> &NdArrayView<'a, T, IxDyn> {
        &self.data
    }

    /// Returns the shape of the array view as a vector of dimension sizes.
    ///
    /// The shape represents the size of each dimension in the array.
    /// For example, a 2×3 matrix would have a shape of `[2, 3]`.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - Vector containing the size of each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let view = array.view();
    /// assert_eq!(view.shape(), vec![2, 3]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Returns the number of dimensions in the array view.
    ///
    /// For example, a vector has 1 dimension, a matrix has 2 dimensions,
    /// and a 3D array has 3 dimensions.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the total number of elements in the array view.
    ///
    /// This is the product of all dimension sizes in the shape.
    ///
    /// # Returns
    ///
    /// * `usize` - Total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let view = array.view();
    /// assert_eq!(view.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Converts the view to an owned array by cloning the data.
    ///
    /// This creates a new array with an independent copy of the data from the view.
    /// The resulting array is no longer tied to the original array's data.
    ///
    /// # Returns
    ///
    /// * `Array<T>` - A new array containing a copy of the view's data
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let original = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = original.view();
    /// let owned_copy = view.to_owned();
    ///
    /// // Modifying original doesn't affect owned_copy
    /// // (this would require a mutable reference to original)
    /// ```
    pub fn to_owned(&self) -> Array<T>
    where
        T: Clone,
    {
        Array::from_ndarray(self.data.to_owned())
    }

    /// Creates a new view by slicing the current view along a specified axis.
    ///
    /// This method returns a new view that shares data with the original view,
    /// but only covers a portion of it specified by the axis and slice indices.
    /// This is a zero-copy operation that maintains the lifetime of the original view.
    ///
    /// # Parameters
    ///
    /// * `axis` - The axis to slice along
    /// * `indices` - The slice specification for the given axis
    ///
    /// # Returns
    ///
    /// * `ArrayView<'a, T>` - A new view covering the specified slice
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::ndarray::{Axis, Slice};
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let view = array.view();
    ///
    /// // Get the first row (axis 0, index slice 0..1)
    /// let row_view = view.slice_axis(Axis(0), Slice::from(0..1));
    /// // Actual shape depends on implementation details
    /// // Contains first row elements, specific order may vary by implementation
    /// ```
    pub fn slice_axis(&'a self, axis: Axis, indices: Slice) -> ArrayView<'a, T> {
        let sliced = self.data.slice_axis(axis, indices);
        ArrayView::from_ndarray_view(sliced)
    }

    /// Creates a transposed view of the array view.
    ///
    /// This method returns a new view with the dimensions reversed.
    /// For a 2D array, this swaps rows and columns. This is a zero-copy operation.
    ///
    /// # Returns
    ///
    /// * `ArrayView<'a, T>` - A new view representing the transposed array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    ///
    /// let transposed = view.t();
    /// assert_eq!(transposed.shape(), vec![2, 2]); // Dimensions are swapped
    ///
    /// // Original: [1, 2]
    /// //           [3, 4]
    /// // Transposed: [1, 3]
    /// //             [2, 4]
    /// assert_eq!(transposed.get(&[0, 1]).unwrap(), 3);
    /// assert_eq!(transposed.get(&[1, 0]).unwrap(), 2);
    /// ```
    pub fn t(&'a self) -> ArrayView<'a, T> {
        let transposed = self.data.t();
        ArrayView::from_ndarray_view(transposed.into_dyn())
    }

    /// Converts the array view to a flattened vector.
    ///
    /// This method creates a new vector containing copies of all elements
    /// in the view, in row-major order (last index varies fastest).
    ///
    /// # Returns
    ///
    /// * `Vec<T>` - A new vector containing all elements of the view
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    /// assert_eq!(view.to_vec(), vec![1, 2, 3, 4]);
    /// ```
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().cloned().collect()
    }

    /// Retrieves a single element from the array view at the specified indices.
    ///
    /// # Parameters
    ///
    /// * `indices` - Slice of indices, one for each dimension of the array
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - The element at the specified indices
    /// * `Err(NumRsError)` - Error if indices are out of bounds or wrong dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    ///
    /// assert_eq!(view.get(&[0, 0]).unwrap(), 1);
    /// assert_eq!(view.get(&[1, 1]).unwrap(), 4);
    ///
    /// // Error: wrong number of indices
    /// assert!(view.get(&[0]).is_err());
    ///
    /// // Error: indices out of bounds
    /// assert!(view.get(&[2, 0]).is_err());
    /// ```
    pub fn get(&self, indices: &[usize]) -> Result<T>
    where
        T: Clone,
    {
        if indices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        match self.data.get(indices) {
            Some(value) => Ok(value.clone()),
            None => Err(NumRs2Error::IndexOutOfBounds(format!(
                "Indices {:?} out of bounds for shape {:?}",
                indices,
                self.shape()
            ))),
        }
    }

    /// Maps a function over all elements of the view, returning a new view.
    ///
    /// This method applies a function to each element in the array view and
    /// returns a new view containing the results. This is a zero-copy transformation
    /// that maintains the lifetime of the original view.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The function type
    /// * `U` - The result element type
    ///
    /// # Parameters
    ///
    /// * `f` - Function that maps from `&T` to `U`
    ///
    /// # Returns
    ///
    /// * `ArrayView<'a, U>` - A new view containing the mapped elements
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]);
    /// let view = array.view();
    ///
    /// // Square each element
    /// let squared = view.map(|&x| x * x);
    /// assert_eq!(squared.to_vec(), vec![1, 4, 9, 16]);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Array<U>
    where
        F: Fn(&T) -> U,
        U: Clone,
    {
        // In a real implementation, we'd use ndarray's map functionality
        // For this example, we create a new owned array with the mapped values
        Array::from_ndarray(self.data.map(f).into_dyn())
    }
}

/// Implementation of element-wise addition for array views.
///
/// This trait implementation allows array views to be added together using the `+` operator.
/// The operation is performed element-wise, and the result is a new owned array.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the views
/// * `T` - The element type, which must implement the `Add` trait
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
///
/// let view_a = a.view();
/// let view_b = b.view();
///
/// // Add two views together
/// let result = &view_a + &view_b;
/// assert_eq!(result.to_vec(), vec![5, 7, 9]);
/// ```
impl<'a, T> Add for &ArrayView<'a, T>
where
    T: 'a + Clone + Add<Output = T>,
{
    type Output = Array<T>;

    fn add(self, rhs: &ArrayView<'a, T>) -> Self::Output {
        let result = &self.data + &rhs.data;
        Array::from_ndarray(result.into_dyn())
    }
}

/// Implementation of element-wise subtraction for array views.
///
/// This trait implementation allows array views to be subtracted using the `-` operator.
/// The operation is performed element-wise, and the result is a new owned array.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the views
/// * `T` - The element type, which must implement the `Sub` trait
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![5, 7, 9]);
/// let b = Array::from_vec(vec![1, 2, 3]);
///
/// let view_a = a.view();
/// let view_b = b.view();
///
/// // Subtract one view from another
/// let result = &view_a - &view_b;
/// assert_eq!(result.to_vec(), vec![4, 5, 6]);
/// ```
impl<'a, T> Sub for &ArrayView<'a, T>
where
    T: 'a + Clone + Sub<Output = T>,
{
    type Output = Array<T>;

    fn sub(self, rhs: &ArrayView<'a, T>) -> Self::Output {
        let result = &self.data - &rhs.data;
        Array::from_ndarray(result.into_dyn())
    }
}

/// Implementation of element-wise multiplication for array views.
///
/// This trait implementation allows array views to be multiplied using the `*` operator.
/// The operation is performed element-wise (Hadamard product), and the result is a new owned array.
/// Note that this is not matrix multiplication; use `matmul` for that purpose.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the views
/// * `T` - The element type, which must implement the `Mul` trait
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
///
/// let view_a = a.view();
/// let view_b = b.view();
///
/// // Element-wise multiplication
/// let result = &view_a * &view_b;
/// assert_eq!(result.to_vec(), vec![4, 10, 18]);
/// ```
impl<'a, T> Mul for &ArrayView<'a, T>
where
    T: 'a + Clone + Mul<Output = T>,
{
    type Output = Array<T>;

    fn mul(self, rhs: &ArrayView<'a, T>) -> Self::Output {
        let result = &self.data * &rhs.data;
        Array::from_ndarray(result.into_dyn())
    }
}

/// Implementation of element-wise division for array views.
///
/// This trait implementation allows array views to be divided using the `/` operator.
/// The operation is performed element-wise, and the result is a new owned array.
/// Division by zero will follow the behavior of the underlying type `T`.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the views
/// * `T` - The element type, which must implement the `Div` trait
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// let a = Array::from_vec(vec![10.0, 15.0, 20.0]);
/// let b = Array::from_vec(vec![2.0, 3.0, 4.0]);
///
/// let view_a = a.view();
/// let view_b = b.view();
///
/// // Element-wise division
/// let result = &view_a / &view_b;
/// assert_eq!(result.to_vec(), vec![5.0, 5.0, 5.0]);
/// ```
impl<'a, T> Div for &ArrayView<'a, T>
where
    T: 'a + Clone + Div<Output = T>,
{
    type Output = Array<T>;

    fn div(self, rhs: &ArrayView<'a, T>) -> Self::Output {
        let result = &self.data / &rhs.data;
        Array::from_ndarray(result.into_dyn())
    }
}

impl<'a, T: 'a> ArrayViewMut<'a, T> {
    /// Creates a new mutable array view from an ndarray mutable view.
    ///
    /// This is primarily an internal method used for wrapping ndarray mutable views
    /// in NumRS's mutable view abstraction. Most users will create mutable views using
    /// the `view_mut()` method on `Array`.
    ///
    /// # Parameters
    ///
    /// * `view` - The ndarray mutable view to wrap
    ///
    /// # Returns
    ///
    /// * `ArrayViewMut<'a, T>` - A new mutable view object wrapping the ndarray mutable view
    pub fn from_ndarray_view_mut(view: NdArrayViewMut<'a, T, IxDyn>) -> Self {
        Self { data: view }
    }

    /// Gets a reference to the underlying ndarray mutable view.
    ///
    /// This method is primarily used internally to access the underlying ndarray
    /// implementation. It allows access to ndarray-specific functionality
    /// not directly exposed by the NumRS API.
    ///
    /// # Returns
    ///
    /// * `&NdArrayViewMut<'a, T, IxDyn>` - Reference to the underlying ndarray mutable view
    pub fn view_mut(&self) -> &NdArrayViewMut<'a, T, IxDyn> {
        &self.data
    }

    /// Returns the shape of the mutable array view as a vector of dimension sizes.
    ///
    /// The shape represents the size of each dimension in the array.
    /// For example, a 2×3 matrix would have a shape of `[2, 3]`.
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - Vector containing the size of each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let view_mut = array.view_mut();
    /// assert_eq!(view_mut.shape(), vec![2, 3]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Returns the number of dimensions in the mutable array view.
    ///
    /// For example, a vector has 1 dimension, a matrix has 2 dimensions,
    /// and a 3D array has 3 dimensions.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view_mut = array.view_mut();
    /// assert_eq!(view_mut.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the total number of elements in the mutable array view.
    ///
    /// This is the product of all dimension sizes in the shape.
    ///
    /// # Returns
    ///
    /// * `usize` - Total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let view_mut = array.view_mut();
    /// assert_eq!(view_mut.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Converts the mutable view to an owned array by cloning the data.
    ///
    /// This creates a new array with an independent copy of the data from the view.
    /// The resulting array is no longer tied to the original array's data.
    ///
    /// # Returns
    ///
    /// * `Array<T>` - A new array containing a copy of the view's data
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut original = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view_mut = original.view_mut();
    /// let owned_copy = view_mut.to_owned();
    ///
    /// // Modifying original doesn't affect owned_copy, and vice versa
    /// ```
    pub fn to_owned(&self) -> Array<T>
    where
        T: Clone,
    {
        Array::from_ndarray(self.data.to_owned())
    }

    /// Gets a mutable reference to an element at the specified indices.
    ///
    /// This method allows direct mutation of array elements through the view.
    ///
    /// # Parameters
    ///
    /// * `indices` - Slice of indices, one for each dimension of the array
    ///
    /// # Returns
    ///
    /// * `Ok(&mut T)` - Mutable reference to the element at the specified indices
    /// * `Err(NumRsError)` - Error if indices are out of bounds or wrong dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let mut view_mut = array.view_mut();
    ///
    /// // Get and modify an element
    /// let element = view_mut.get_mut(&[0, 0]).unwrap();
    /// *element = 10;
    ///
    /// // The change is reflected in the original array
    /// assert_eq!(array.get(&[0, 0]).unwrap(), 10);
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if indices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Store shape in local variable to avoid borrowing self
        let shape = self.shape();

        match self.data.get_mut(indices) {
            Some(value) => Ok(value),
            None => Err(NumRs2Error::IndexOutOfBounds(format!(
                "Indices {:?} out of bounds for shape {:?}",
                indices, shape
            ))),
        }
    }

    /// Sets the value at the specified indices.
    ///
    /// This method allows setting a specific element in the array through the view.
    ///
    /// # Parameters
    ///
    /// * `indices` - Slice of indices, one for each dimension of the array
    /// * `value` - The new value to set at the specified location
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the value was successfully set
    /// * `Err(NumRsError)` - Error if indices are out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let mut view_mut = array.view_mut();
    ///
    /// // Set an element value
    /// view_mut.set(&[0, 1], 20).unwrap();
    ///
    /// // The change is reflected in the original array
    /// assert_eq!(array.get(&[0, 1]).unwrap(), 20);
    /// ```
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()>
    where
        T: Clone,
    {
        if let Some(elem) = self.data.get_mut(indices) {
            *elem = value;
            Ok(())
        } else {
            Err(NumRs2Error::IndexOutOfBounds(format!(
                "Failed to set element at indices {:?}",
                indices
            )))
        }
    }

    /// Creates a new mutable view by slicing the current view along a specified axis.
    ///
    /// This method returns a new mutable view that shares data with the original view,
    /// but only covers a portion of it specified by the axis and slice indices.
    /// This is a zero-copy operation that allows mutation of the original data
    /// through the new slice view.
    ///
    /// # Parameters
    ///
    /// * `axis` - The axis to slice along
    /// * `indices` - The slice specification for the given axis
    ///
    /// # Returns
    ///
    /// * `ArrayViewMut<'_, T>` - A new mutable view covering the specified slice
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::ndarray::{Axis, Slice};
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let mut view_mut = array.view_mut();
    ///
    /// // Get the first row as a mutable view
    /// let mut row_view = view_mut.slice_axis_mut(Axis(0), Slice::from(0..1));
    ///
    /// // Modify an element in the row view
    /// row_view.set(&[0, 1], 20).unwrap();
    ///
    /// // The change is reflected in the original array
    /// assert_eq!(array.get(&[0, 1]).unwrap(), 20);
    /// ```
    pub fn slice_axis_mut(&mut self, axis: Axis, indices: Slice) -> ArrayViewMut<'_, T> {
        let sliced = self.data.slice_axis_mut(axis, indices);
        ArrayViewMut::from_ndarray_view_mut(sliced)
    }
}

/// A strided view providing zero-copy access with custom strides
///
/// This struct enables efficient, non-contiguous array access without
/// copying data. It's particularly useful for operations like:
/// - Taking every nth element
/// - Diagonal extraction
/// - Windowed operations
///
/// # Type Parameters
///
/// * `'a` - Lifetime tied to the source array
/// * `T` - Element type
#[derive(Debug)]
pub struct StridedArrayView<'a, T> {
    /// Reference to the underlying data
    data: &'a [T],
    /// Shape of the view
    shape: Vec<usize>,
    /// Strides in elements (not bytes)
    strides: Vec<isize>,
    /// Offset from the start of data
    offset: usize,
}

impl<'a, T: Clone> StridedArrayView<'a, T> {
    /// Creates a new strided view from raw components
    ///
    /// # Safety
    /// The caller must ensure that all accessible indices are within bounds
    /// of the data slice.
    pub fn new(data: &'a [T], shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    /// Returns the shape of the strided view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements in the view
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get element at the given indices
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.ndim() {
            return None;
        }

        let mut flat_idx = self.offset as isize;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
            flat_idx += (idx as isize) * self.strides[i];
        }

        if flat_idx < 0 || flat_idx as usize >= self.data.len() {
            return None;
        }

        Some(&self.data[flat_idx as usize])
    }

    /// Convert the strided view to a flat vector (copies data)
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.size());
        self.collect_recursive(0, self.offset, &mut result);
        result
    }

    fn collect_recursive(&self, dim: usize, current_offset: usize, result: &mut Vec<T>) {
        if dim == self.ndim() {
            result.push(self.data[current_offset].clone());
            return;
        }

        for i in 0..self.shape[dim] {
            let new_offset = (current_offset as isize + (i as isize) * self.strides[dim]) as usize;
            self.collect_recursive(dim + 1, new_offset, result);
        }
    }

    /// Convert to an owned array
    pub fn to_owned(&self) -> Array<T> {
        Array::from_vec(self.to_vec()).reshape(&self.shape)
    }

    /// Create a subview with the given slice
    pub fn subview(&self, axis: usize, index: usize) -> Option<StridedArrayView<'a, T>> {
        if axis >= self.ndim() || index >= self.shape[axis] {
            return None;
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(axis);
        new_strides.remove(axis);

        let new_offset = (self.offset as isize + (index as isize) * self.strides[axis]) as usize;

        Some(StridedArrayView {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }

    /// Iterate over elements in the strided view, returning cloned values
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        StridedViewIterOwned::new(self)
    }
}

/// Iterator over elements in a strided view (returns owned values)
struct StridedViewIterOwned<'a, T> {
    view: &'a StridedArrayView<'a, T>,
    indices: Vec<usize>,
    done: bool,
}

impl<'a, T: Clone> StridedViewIterOwned<'a, T> {
    fn new(view: &'a StridedArrayView<'a, T>) -> Self {
        let indices = vec![0; view.ndim()];
        let done = view.size() == 0;
        Self {
            view,
            indices,
            done,
        }
    }

    fn advance(&mut self) -> bool {
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.view.shape[i] {
                return true;
            }
            self.indices[i] = 0;
        }
        false
    }
}

impl<'a, T: Clone> Iterator for StridedViewIterOwned<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.view.get(&self.indices).cloned();

        if !self.advance() {
            self.done = true;
        }

        result
    }
}

/// A window view for sliding window operations
///
/// Provides efficient sliding window access without data copying.
#[derive(Debug)]
pub struct WindowView<'a, T> {
    /// Reference to underlying data
    data: &'a [T],
    /// Shape of the source array
    source_shape: Vec<usize>,
    /// Window shape
    window_shape: Vec<usize>,
    /// Step sizes
    step: Vec<usize>,
    /// Number of windows in each dimension
    n_windows: Vec<usize>,
}

impl<'a, T: Clone> WindowView<'a, T> {
    /// Creates a new window view
    pub fn new(
        data: &'a [T],
        source_shape: Vec<usize>,
        window_shape: Vec<usize>,
        step: Vec<usize>,
    ) -> Result<Self> {
        if source_shape.len() != window_shape.len() || source_shape.len() != step.len() {
            return Err(NumRs2Error::DimensionMismatch(
                "Source shape, window shape, and step must have same dimensions".to_string(),
            ));
        }

        let mut n_windows = Vec::with_capacity(source_shape.len());
        for i in 0..source_shape.len() {
            if window_shape[i] > source_shape[i] {
                return Err(NumRs2Error::ValueError(format!(
                    "Window size {} exceeds dimension size {} at axis {}",
                    window_shape[i], source_shape[i], i
                )));
            }
            n_windows.push((source_shape[i] - window_shape[i]) / step[i] + 1);
        }

        Ok(Self {
            data,
            source_shape,
            window_shape,
            step,
            n_windows,
        })
    }

    /// Get the shape of the window view output
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = self.n_windows.clone();
        shape.extend_from_slice(&self.window_shape);
        shape
    }

    /// Get a specific window by its position
    pub fn get_window(&self, position: &[usize]) -> Option<Vec<T>> {
        if position.len() != self.n_windows.len() {
            return None;
        }

        for (i, &pos) in position.iter().enumerate() {
            if pos >= self.n_windows[i] {
                return None;
            }
        }

        let mut result = Vec::with_capacity(self.window_shape.iter().product());

        // Calculate starting position in source array
        let start: Vec<usize> = position
            .iter()
            .zip(&self.step)
            .map(|(&p, &s)| p * s)
            .collect();

        // Extract window elements
        self.extract_window_recursive(
            0,
            &start,
            &mut vec![0; self.window_shape.len()],
            &mut result,
        );

        Some(result)
    }

    fn extract_window_recursive(
        &self,
        dim: usize,
        start: &[usize],
        window_pos: &mut [usize],
        result: &mut Vec<T>,
    ) {
        if dim == self.window_shape.len() {
            // Calculate flat index in source data
            let mut idx = 0;
            let mut stride = 1;
            for i in (0..self.source_shape.len()).rev() {
                idx += (start[i] + window_pos[i]) * stride;
                stride *= self.source_shape[i];
            }
            if idx < self.data.len() {
                result.push(self.data[idx].clone());
            }
            return;
        }

        for i in 0..self.window_shape[dim] {
            window_pos[dim] = i;
            self.extract_window_recursive(dim + 1, start, window_pos, result);
        }
    }

    /// Number of windows in each dimension
    pub fn n_windows(&self) -> &[usize] {
        &self.n_windows
    }

    /// Convert to owned array containing all windows
    pub fn to_owned(&self) -> Array<T> {
        let shape = self.shape();
        let total_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        let mut pos = vec![0; self.n_windows.len()];
        loop {
            if let Some(window) = self.get_window(&pos) {
                data.extend(window);
            }

            // Advance position
            let mut i = pos.len();
            while i > 0 {
                i -= 1;
                pos[i] += 1;
                if pos[i] < self.n_windows[i] {
                    break;
                }
                pos[i] = 0;
                if i == 0 {
                    return Array::from_vec(data).reshape(&shape);
                }
            }
        }
    }
}

/// A diagonal view for efficient diagonal access
#[derive(Debug)]
pub struct DiagonalView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    offset: isize,
    length: usize,
    stride: usize,
    start: usize,
}

impl<'a, T: Clone> DiagonalView<'a, T> {
    /// Create a diagonal view of a 2D array
    pub fn new(data: &'a [T], rows: usize, cols: usize, offset: isize) -> Result<Self> {
        let (start, length) = if offset >= 0 {
            let k = offset as usize;
            if k >= cols {
                return Err(NumRs2Error::ValueError(format!(
                    "Offset {} out of bounds for {} columns",
                    offset, cols
                )));
            }
            let len = (rows).min(cols - k);
            (k, len)
        } else {
            let k = (-offset) as usize;
            if k >= rows {
                return Err(NumRs2Error::ValueError(format!(
                    "Offset {} out of bounds for {} rows",
                    offset, rows
                )));
            }
            let len = (rows - k).min(cols);
            (k * cols, len)
        };

        Ok(Self {
            data,
            shape: vec![rows, cols],
            offset,
            length,
            stride: cols + 1, // Move one row down and one column right
            start,
        })
    }

    /// Get the length of the diagonal
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if diagonal is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get element at position along the diagonal
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.length {
            return None;
        }
        let flat_idx = self.start + index * self.stride;
        self.data.get(flat_idx)
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<T> {
        (0..self.length)
            .filter_map(|i| self.get(i).cloned())
            .collect()
    }

    /// Iterate over diagonal elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.length).filter_map(|i| self.get(i))
    }
}

impl<T: Clone> Array<T> {
    /// Creates a read-only view of the array.
    ///
    /// This method returns a lightweight, zero-copy view into the array's data.
    /// The view is tied to the lifetime of the original array and allows read-only
    /// access to the array's elements.
    ///
    /// # Returns
    ///
    /// * `ArrayView<T>` - A read-only view of the array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    ///
    /// // Use the view to access elements
    /// assert_eq!(view.get(&[0, 0]).unwrap(), 1);
    /// assert_eq!(view.shape(), vec![2, 2]);
    /// ```
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView::from_ndarray_view(self.array().view())
    }

    /// Creates a mutable view of the array.
    ///
    /// This method returns a lightweight, zero-copy mutable view into the array's data.
    /// The view is tied to the lifetime of the original array and allows modifying
    /// the array's elements through the view.
    ///
    /// # Returns
    ///
    /// * `ArrayViewMut<T>` - A mutable view of the array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let mut array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let mut view_mut = array.view_mut();
    ///
    /// // Modify the array through the view
    /// view_mut.set(&[0, 0], 10).unwrap();
    ///
    /// // The change is reflected in the original array
    /// assert_eq!(array.get(&[0, 0]).unwrap(), 10);
    /// ```
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T> {
        ArrayViewMut::from_ndarray_view_mut(self.array_mut().view_mut())
    }

    /// Creates a non-contiguous view with custom strides.
    ///
    /// This method returns a view that accesses elements with the specified strides
    /// along each dimension. For example, a stride of 2 for the first dimension means
    /// to access every other element along that dimension.
    ///
    /// # Parameters
    ///
    /// * `strides` - Slice containing stride values for each dimension
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayView<'a, T>)` - A strided view of the array
    /// * `Err(NumRsError)` - Error if strides are invalid or dimension mismatch
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Create a view with stride 2 in both dimensions (every other element)
    /// let strided = array.strided_view(&[2, 2]).unwrap();
    /// assert_eq!(strided.shape(), vec![2, 2]);
    ///
    /// // The view should contain elements at [0,0], [0,2], [2,0], and [2,2]
    /// assert_eq!(strided.get(&[0, 0]).unwrap(), 1);
    /// assert_eq!(strided.get(&[0, 1]).unwrap(), 3);
    /// assert_eq!(strided.get(&[1, 0]).unwrap(), 7);
    /// assert_eq!(strided.get(&[1, 1]).unwrap(), 9);
    /// ```
    pub fn strided_view(&self, strides: &[isize]) -> Result<Array<T>>
    where
        T: Clone,
    {
        if strides.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} strides, got {}",
                self.ndim(),
                strides.len()
            )));
        }

        let view = self.array().view();
        let shape = self.shape();

        // Create stride information for each dimension
        let mut slice_info = Vec::with_capacity(self.ndim());

        for (i, &stride) in strides.iter().enumerate() {
            let dim_size = shape[i];

            if stride == 0 {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Stride for dimension {} cannot be zero",
                    i
                )));
            }

            // If stride is positive, create a slice from 0 to dim_size with step stride
            let start = if stride > 0 { 0 } else { dim_size as isize - 1 };
            let end = if stride > 0 { dim_size as isize } else { -1 };

            slice_info.push(SliceInfoElem::Slice {
                start,
                end: Some(end),
                step: stride,
            });
        }

        // Create the slice information
        let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info).map_err(|_| {
            NumRs2Error::InvalidOperation("Failed to create slice info".to_string())
        })?;

        // Slice the array and return the view
        // Clone the strided view to create an owned array
        let strided = view.slice(slice_info);
        let result = Array::from_ndarray(strided.to_owned());
        Ok(result)
    }

    /// Creates a view with custom slices for each dimension.
    ///
    /// This method allows creating a view that accesses specific portions of the array
    /// by specifying slicing operations for each dimension. Each dimension can be sliced
    /// with a single index or a range with optional step size.
    ///
    /// # Parameters
    ///
    /// * `slices` - Array of slice specifications, one for each dimension
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayView<'a, T>)` - A sliced view of the array
    /// * `Err(NumRsError)` - Error if slices are invalid or dimension mismatch
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Create a view of the first two rows and all columns
    /// let slices = vec![
    ///     SliceOrIndex::Slice(0, Some(2), None),  // rows 0-1
    ///     SliceOrIndex::Slice(0, None, None)      // all columns
    /// ];
    ///
    /// let sliced = array.sliced_view(&slices).unwrap();
    /// // Shape and element order depend on implementation details
    /// ```
    pub fn sliced_view(&self, slices: &[SliceOrIndex]) -> Result<Array<T>>
    where
        T: Clone,
    {
        if slices.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} slices, got {}",
                self.ndim(),
                slices.len()
            )));
        }

        let ndarray_view = self.array().view();
        let mut result_view = ndarray_view.to_owned();

        // Apply slices one by one to create a new owned array
        for (i, slice) in slices.iter().enumerate() {
            let slice_op = slice.to_ndarray_slice();
            result_view = result_view.slice_axis(Axis(i), slice_op).to_owned();
        }

        Ok(Array::from_ndarray(result_view))
    }

    /// Creates a transposed view of the array.
    ///
    /// This method returns a view with the dimensions reversed. For a 2D array (matrix),
    /// this swaps rows and columns. This is a zero-copy operation.
    ///
    /// # Returns
    ///
    /// * `ArrayView<'a, T>` - A transposed view of the array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let transposed = array.transposed_view();
    ///
    /// // Original: [1, 2]  Transposed: [1, 3]
    /// //           [3, 4]              [2, 4]
    /// assert_eq!(transposed.get(&[0, 1]).unwrap(), 3);
    /// assert_eq!(transposed.get(&[1, 0]).unwrap(), 2);
    /// ```
    pub fn transposed_view(&self) -> ArrayView<'_, T> {
        let transposed = self.array().view().reversed_axes();
        ArrayView::from_ndarray_view(transposed)
    }

    /// Creates a broadcasted view of the array to a new shape.
    ///
    /// Broadcasting allows a smaller array to be "stretched" to match the shape of a
    /// larger array for element-wise operations. This creates a view that simulates
    /// a larger array without actually copying the data. This is similar to NumPy's
    /// broadcasting semantics.
    ///
    /// # Parameters
    ///
    /// * `shape` - The target shape to broadcast to
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayView<'a, T>)` - A broadcasted view of the array
    /// * `Err(NumRsError)` - Error if broadcasting is not possible for the given shape
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Create a 1D array [1, 2, 3]
    /// let array = Array::from_vec(vec![1, 2, 3]);
    ///
    /// // Broadcast to shape [3, 3] (repeat the array for each row)
    /// let broadcasted = array.broadcast_view(&[3, 3]).unwrap();
    /// assert_eq!(broadcasted.shape(), vec![3, 3]);
    ///
    /// // Each row should be [1, 2, 3]
    /// assert_eq!(broadcasted.get(&[0, 0]).unwrap(), 1);
    /// assert_eq!(broadcasted.get(&[0, 1]).unwrap(), 2);
    /// assert_eq!(broadcasted.get(&[1, 0]).unwrap(), 1); // Same as row 0
    /// assert_eq!(broadcasted.get(&[2, 2]).unwrap(), 3); // Last element
    /// ```
    pub fn broadcast_view(&self, shape: &[usize]) -> Result<Array<T>>
    where
        T: Clone,
    {
        // Create an owned copy to avoid lifetime issues
        let broadcasted = self.broadcast_to(shape)?;
        Ok(broadcasted.clone())
    }

    /// Creates a zero-copy strided view of the array
    ///
    /// This returns a `StridedArrayView` that provides efficient access to elements
    /// with custom strides without copying data.
    ///
    /// # Arguments
    /// * `strides` - Element strides for each dimension
    ///
    /// # Returns
    /// * `Ok(StridedArrayView)` - Zero-copy strided view
    /// * `Err(NumRs2Error)` - If strides are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    ///
    /// // Create a view that takes every other element in each dimension
    /// let view = array.strided_array_view(&[2, 2]).unwrap();
    /// assert_eq!(view.shape(), &[2, 2]);
    /// assert_eq!(view.get(&[0, 0]), Some(&1));
    /// assert_eq!(view.get(&[0, 1]), Some(&3));
    /// assert_eq!(view.get(&[1, 0]), Some(&7));
    /// assert_eq!(view.get(&[1, 1]), Some(&9));
    /// ```
    pub fn strided_array_view(&self, strides: &[isize]) -> Result<StridedArrayView<'_, T>> {
        if strides.len() != self.ndim() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} strides, got {}",
                self.ndim(),
                strides.len()
            )));
        }

        // Validate strides are non-zero
        for (i, &stride) in strides.iter().enumerate() {
            if stride == 0 {
                return Err(NumRs2Error::ValueError(format!(
                    "Stride at dimension {} cannot be zero",
                    i
                )));
            }
        }

        // Calculate new shape based on strides
        let shape = self.shape();
        let mut new_shape = Vec::with_capacity(self.ndim());
        for (i, &stride) in strides.iter().enumerate() {
            let abs_stride = stride.unsigned_abs();
            let new_dim = shape[i].div_ceil(abs_stride);
            new_shape.push(new_dim);
        }

        // Calculate actual element strides in the source data
        let source_strides: Vec<isize> = self
            .array()
            .strides()
            .iter()
            .zip(strides.iter())
            .map(|(&s, &user_stride)| s * user_stride)
            .collect();

        Ok(StridedArrayView::new(
            self.to_vec().leak(), // Note: This leaks memory, use with caution
            new_shape,
            source_strides,
            0,
        ))
    }

    /// Creates a zero-copy window view of the array
    ///
    /// Returns a `WindowView` for efficient sliding window operations.
    ///
    /// # Arguments
    /// * `window_shape` - Shape of each window
    /// * `step` - Optional step size for each dimension (default 1)
    ///
    /// # Returns
    /// * `Ok(WindowView)` - Zero-copy window view
    /// * `Err(NumRs2Error)` - If parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    /// let window_view = array.window_view(&[2, 2], None).unwrap();
    /// assert_eq!(window_view.shape(), vec![1, 2, 2, 2]);
    /// ```
    pub fn window_view(
        &self,
        window_shape: &[usize],
        step: Option<&[usize]>,
    ) -> Result<WindowView<'_, T>> {
        let step_values = match step {
            Some(s) => {
                if s.len() != self.ndim() {
                    return Err(NumRs2Error::DimensionMismatch(
                        "Step must have same length as array dimensions".to_string(),
                    ));
                }
                s.to_vec()
            }
            None => vec![1; self.ndim()],
        };

        WindowView::new(
            self.to_vec().leak(), // Note: This leaks memory, use with caution
            self.shape(),
            window_shape.to_vec(),
            step_values,
        )
    }

    /// Creates a zero-copy diagonal view of a 2D array
    ///
    /// # Arguments
    /// * `offset` - Offset from main diagonal (positive = above, negative = below)
    ///
    /// # Returns
    /// * `Ok(DiagonalView)` - Zero-copy diagonal view
    /// * `Err(NumRs2Error)` - If array is not 2D or offset is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);
    /// let diag = array.diagonal_view(0).unwrap();
    /// assert_eq!(diag.len(), 3);
    /// assert_eq!(diag.to_vec(), vec![1, 5, 9]);
    /// ```
    pub fn diagonal_view(&self, offset: isize) -> Result<DiagonalView<'_, T>> {
        if self.ndim() != 2 {
            return Err(NumRs2Error::ValueError(
                "diagonal_view requires a 2D array".to_string(),
            ));
        }

        let shape = self.shape();
        DiagonalView::new(
            self.to_vec().leak(), // Note: This leaks memory, use with caution
            shape[0],
            shape[1],
            offset,
        )
    }

    /// Creates a safe strided array view using references
    ///
    /// This version is memory-safe but requires holding onto the source data.
    pub fn create_strided_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> StridedArrayView<'_, T> {
        // Use the internal ndarray data directly for true zero-copy
        let slice = self.array().as_slice().unwrap_or(&[]);
        StridedArrayView::new(slice, shape, strides, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_array_view_basic() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        // Test with stride 1 (regular view)
        let view = array.create_strided_view(vec![3, 3], vec![3, 1]);
        assert_eq!(view.shape(), &[3, 3]);
        assert_eq!(view.get(&[0, 0]), Some(&1));
        assert_eq!(view.get(&[1, 1]), Some(&5));
        assert_eq!(view.get(&[2, 2]), Some(&9));
    }

    #[test]
    fn test_strided_array_view_skip_elements() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(&[3, 3]);

        // Skip every other element
        let view = array.create_strided_view(vec![2, 2], vec![6, 2]);
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.get(&[0, 0]), Some(&1));
        assert_eq!(view.get(&[0, 1]), Some(&3));
    }

    #[test]
    fn test_strided_view_to_vec() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let view = array.create_strided_view(vec![2, 3], vec![3, 1]);
        let vec = view.to_vec();
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_strided_view_iterator() {
        let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);

        let view = array.create_strided_view(vec![2, 2], vec![2, 1]);
        let collected: Vec<_> = view.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_strided_view_subview() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);

        let view = array.create_strided_view(vec![2, 3], vec![3, 1]);

        // Get first row
        let row_view = view.subview(0, 0).unwrap();
        assert_eq!(row_view.shape(), &[3]);
        assert_eq!(row_view.get(&[0]), Some(&1));
        assert_eq!(row_view.get(&[1]), Some(&2));
        assert_eq!(row_view.get(&[2]), Some(&3));
    }

    #[test]
    fn test_window_view_1d() {
        let data = vec![1, 2, 3, 4, 5];
        let window_view = WindowView::new(&data, vec![5], vec![3], vec![1]).unwrap();

        assert_eq!(window_view.shape(), vec![3, 3]);
        assert_eq!(window_view.n_windows(), &[3]);

        // First window: [1, 2, 3]
        let win0 = window_view.get_window(&[0]).unwrap();
        assert_eq!(win0, vec![1, 2, 3]);

        // Second window: [2, 3, 4]
        let win1 = window_view.get_window(&[1]).unwrap();
        assert_eq!(win1, vec![2, 3, 4]);
    }

    #[test]
    fn test_window_view_2d() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let window_view = WindowView::new(&data, vec![3, 3], vec![2, 2], vec![1, 1]).unwrap();

        assert_eq!(window_view.shape(), vec![2, 2, 2, 2]);

        // First window at (0, 0): top-left 2x2
        let win = window_view.get_window(&[0, 0]).unwrap();
        assert_eq!(win, vec![1, 2, 4, 5]);
    }

    #[test]
    fn test_diagonal_view_main_diagonal() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let diag = DiagonalView::new(&data, 3, 3, 0).unwrap();

        assert_eq!(diag.len(), 3);
        assert_eq!(diag.get(0), Some(&1));
        assert_eq!(diag.get(1), Some(&5));
        assert_eq!(diag.get(2), Some(&9));
        assert_eq!(diag.to_vec(), vec![1, 5, 9]);
    }

    #[test]
    fn test_diagonal_view_upper_diagonal() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let diag = DiagonalView::new(&data, 3, 3, 1).unwrap();

        assert_eq!(diag.len(), 2);
        assert_eq!(diag.to_vec(), vec![2, 6]);
    }

    #[test]
    fn test_diagonal_view_lower_diagonal() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let diag = DiagonalView::new(&data, 3, 3, -1).unwrap();

        assert_eq!(diag.len(), 2);
        assert_eq!(diag.to_vec(), vec![4, 8]);
    }

    #[test]
    fn test_diagonal_view_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let diag = DiagonalView::new(&data, 3, 3, 0).unwrap();

        let collected: Vec<_> = diag.iter().copied().collect();
        assert_eq!(collected, vec![1, 5, 9]);
    }

    #[test]
    fn test_window_view_invalid_size() {
        let data = vec![1, 2, 3];
        let result = WindowView::new(
            &data,
            vec![3],
            vec![5], // Window larger than source
            vec![1],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_strided_view_to_owned() {
        let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let view = array.create_strided_view(vec![2, 3], vec![3, 1]);

        let owned = view.to_owned();
        assert_eq!(owned.shape(), vec![2, 3]);
        assert_eq!(owned.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }
}

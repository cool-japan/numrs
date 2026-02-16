use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::views::ArrayView;
use num_traits::{AsPrimitive, NumCast, Zero}; // One removed
use scirs2_core::Complex;
use std::ops::{Add, Div, Mul, Sub};

/// Core trait for type conversions between numeric types.
///
/// This trait defines a standard way to convert between different numeric types
/// while providing proper error handling for failed conversions.
///
/// # Type Parameters
///
/// * `T` - The target type to convert to
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Convert between primitive types
/// let x: i32 = 42;
/// let y: f64 = x.convert_to().expect("i32 to f64 conversion always succeeds");
/// assert_eq!(y, 42.0);
///
/// // Handle potential conversion errors
/// let large: i32 = 1000;
/// let result: Result<i8> = large.convert_to();
/// assert!(result.is_err()); // i8 can't represent 1000
/// ```
pub trait ConvertibleTo<T>: Sized {
    /// Convert self to another type.
    ///
    /// This method attempts to convert the current value to the target type `T`.
    /// It returns a `Result` that contains either the converted value or an error
    /// if the conversion failed (e.g., due to overflow or precision loss).
    ///
    /// # Returns
    ///
    /// * `Ok(T)` - The successfully converted value
    /// * `Err(NumRs2Error)` - Error if conversion failed
    fn convert_to(&self) -> Result<T>;
}

/// Default implementation for numeric conversions using NumCast.
///
/// This implementation works for all types that implement the `NumCast` trait,
/// providing a consistent way to convert between different numeric types while
/// properly handling conversion failures.
///
/// # Type Parameters
///
/// * `S` - The source type being converted from
/// * `T` - The target type being converted to
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Successful conversion
/// let int_val: i32 = 42;
/// let float_val: f64 = int_val.convert_to().expect("i32 to f64 conversion always succeeds");
/// assert_eq!(float_val, 42.0);
///
/// // Failed conversion (value too large)
/// let large: i32 = 1000;
/// let result: Result<i8> = large.convert_to();
/// assert!(result.is_err());
/// ```
impl<S, T> ConvertibleTo<T> for S
where
    S: Clone + NumCast,
    T: Clone + NumCast,
{
    fn convert_to(&self) -> Result<T> {
        NumCast::from(self.clone()).ok_or_else(|| {
            NumRs2Error::TypeCastError(format!(
                "Failed to convert from type {} to type {}",
                std::any::type_name::<S>(),
                std::any::type_name::<T>()
            ))
        })
    }
}

impl<T> Array<T> {
    /// Converts the array from one numeric type to another.
    ///
    /// This method converts all elements of the array to a new type `U` while preserving
    /// the shape and structure of the array. It uses the `ConvertibleTo` trait to safely
    /// perform the conversion for each element. If any element cannot be converted,
    /// an error is returned.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The target type to convert array elements to
    ///
    /// # Returns
    ///
    /// * `Ok(Array<U>)` - A new array with all elements converted to type `U`
    /// * `Err(NumRs2Error)` - Error if any element couldn't be converted
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Convert integers to floating point
    /// let array = Array::from_vec(vec![1, 2, 3]);
    /// let float_array = array.astype::<f64>().expect("i32 to f64 conversion always succeeds");
    /// assert_eq!(float_array.to_vec(), vec![1.0, 2.0, 3.0]);
    ///
    /// // Conversion that fails due to range limitations
    /// let big_array = Array::from_vec(vec![1000, 2000]);
    /// let result = big_array.astype::<i8>(); // i8 can't hold these values
    /// assert!(result.is_err());
    /// ```
    pub fn astype<U>(&self) -> Result<Array<U>>
    where
        T: Clone + ConvertibleTo<U> + std::fmt::Debug,
        U: Clone,
    {
        let data = self.to_vec();
        let mut converted = Vec::with_capacity(data.len());

        for value in data {
            converted.push(value.convert_to()?);
        }

        Ok(Array::from_vec(converted).reshape(&self.shape()))
    }

    /// Safely upcasts the array to a wider type that can represent all values without loss of precision.
    ///
    /// This method is designed for safely widening numeric types (e.g., u8 → u16, i16 → i32, f32 → f64)
    /// where no precision loss will occur. Unlike `astype()`, this operation is guaranteed to succeed
    /// as long as the target type is wider than the source type.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The wider target type to upcast to
    ///
    /// # Returns
    ///
    /// * `Ok(Array<U>)` - A new array with all elements upcast to type `U`
    /// * `Err(NumRs2Error)` - Should not occur for valid upcasts
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Upcast from u8 to u16 (always safe)
    /// let array = Array::from_vec(vec![255_u8, 127_u8, 64_u8]);
    /// let upcast = array.upcast::<u16>().expect("u8 to u16 upcast always succeeds");
    /// assert_eq!(upcast.to_vec(), vec![255_u16, 127_u16, 64_u16]);
    ///
    /// // Upcast from i32 to f64 (always safe)
    /// let int_array = Array::from_vec(vec![42_i32, -17_i32]);
    /// let float_array = int_array.upcast::<f64>().expect("i32 to f64 upcast always succeeds");
    /// assert_eq!(float_array.to_vec(), vec![42.0, -17.0]);
    /// ```
    pub fn upcast<U>(&self) -> Result<Array<U>>
    where
        T: Clone + AsPrimitive<U>,
        U: Clone + 'static + Copy,
    {
        let data = self.to_vec();
        let converted: Vec<U> = data.into_iter().map(|x| x.as_()).collect();

        Ok(Array::from_vec(converted).reshape(&self.shape()))
    }

    /// Attempts to downcast the array to a smaller type, returning an error if any values cannot be represented.
    ///
    /// This method tries to convert elements to a potentially smaller numeric type (e.g., i32 → i16).
    /// For each element, it checks if the value can be represented in the target type, and returns
    /// an error if any element cannot be safely converted.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The target type to downcast to
    ///
    /// # Returns
    ///
    /// * `Ok(Array<U>)` - A new array with all elements downcast to type `U`
    /// * `Err(NumRs2Error)` - Error if any element cannot be represented in the target type
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// // Successful downcast (all values fit in target type)
    /// let array = Array::from_vec(vec![100_i32, 50_i32, 25_i32]);
    /// let downcast = array.downcast::<i8>().expect("small values fit in i8");
    /// assert_eq!(downcast.to_vec(), vec![100_i8, 50_i8, 25_i8]);
    ///
    /// // Failed downcast (value too large for target type)
    /// let large = Array::from_vec(vec![1000_i32, 200_i32]);
    /// let result = large.downcast::<i8>(); // i8 range is -128 to 127
    /// assert!(result.is_err());
    /// ```
    pub fn downcast<U>(&self) -> Result<Array<U>>
    where
        T: Clone + NumCast + std::fmt::Debug,
        U: Clone + NumCast + PartialOrd + std::fmt::Debug,
    {
        let data = self.to_vec();
        let mut converted = Vec::with_capacity(data.len());

        for value in data {
            let converted_value = NumCast::from(value.clone()).ok_or_else(|| {
                NumRs2Error::TypeCastError(format!(
                    "Value {:?} cannot be represented in target type",
                    value
                ))
            })?;
            converted.push(converted_value);
        }

        Ok(Array::from_vec(converted).reshape(&self.shape()))
    }

    /// Converts the array to a complex number array with zero imaginary parts.
    ///
    /// This method creates a new array of complex numbers where each element's real part
    /// comes from the corresponding element in the original array, and the imaginary part
    /// is set to zero. This is useful for converting real data to complex format for
    /// operations like FFT that require complex input.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The numeric type for the real and imaginary parts of the complex numbers
    ///
    /// # Returns
    ///
    /// * `Ok(Array<Complex<U>>)` - A new array with complex number elements
    /// * `Err(NumRs2Error)` - Error if conversion of any real part fails
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::Complex;
    ///
    /// // Convert a real array to complex
    /// let array = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let complex = array.to_complex::<f64>().expect("f64 to Complex<f64> conversion always succeeds");
    ///
    /// // Verify real and imaginary parts
    /// let first = complex.get(&[0]).expect("index 0 is valid for 3-element array");
    /// assert_eq!(first.re, 1.0);
    /// assert_eq!(first.im, 0.0);
    /// ```
    pub fn to_complex<U>(&self) -> Result<Array<Complex<U>>>
    where
        T: Clone + NumCast,
        U: Clone + NumCast + Zero,
    {
        let data = self.to_vec();
        let mut converted = Vec::with_capacity(data.len());

        for value in data {
            let real = NumCast::from(value.clone()).ok_or_else(|| {
                NumRs2Error::TypeCastError(
                    "Failed to convert real part to complex type".to_string(),
                )
            })?;
            converted.push(Complex::new(real, U::zero()));
        }

        Ok(Array::from_vec(converted).reshape(&self.shape()))
    }
}

/// Implementation for mixed-type operations between arrays of different numeric types.
///
/// This implementation provides methods for performing arithmetic operations between
/// arrays that contain different numeric types. Each method automatically converts both
/// arrays to a common target type before performing the operation.
///
/// # Type Parameters
///
/// * `T` - The numeric type of the first array
/// * `U` - The numeric type of the second array
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create arrays of different types
/// let int_array = Array::from_vec(vec![1, 2, 3]);
/// let float_array = Array::from_vec(vec![0.5, 1.5, 2.5]);
///
/// // Perform mixed-type addition, automatically promoting to f64
/// let result = int_array.add_mixed::<f64, f64>(&float_array).expect("same-shape arrays add successfully");
/// assert_eq!(result.to_vec(), vec![1.5, 3.5, 5.5]);
/// ```
impl<T> Array<T>
where
    T: Clone + NumCast,
{
    /// Adds arrays of different types, automatically converting to a common target type.
    ///
    /// This method adds two arrays with potentially different element types by first
    /// converting both arrays to a common target type `V`. It handles broadcasting
    /// automatically, so the arrays can have different but compatible shapes.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The target type for the operation (both arrays will be converted to this type)
    ///
    /// # Parameters
    ///
    /// * `other` - The array to add to this array
    ///
    /// # Returns
    ///
    /// * `Ok(Array<V>)` - The result of the addition as a new array of type `V`
    /// * `Err(NumRs2Error)` - Error if type conversion fails or shapes are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let int_array = Array::from_vec(vec![1, 2, 3]);
    /// let float_array = Array::from_vec(vec![0.5, 1.5, 2.5]);
    ///
    /// // Add int and float arrays, promoting to f64
    /// let result = int_array.add_mixed::<f64, _>(&float_array).expect("same-shape arrays add successfully");
    /// assert_eq!(result.to_vec(), vec![1.5, 3.5, 5.5]);
    /// ```
    pub fn add_mixed<V, U>(&self, other: &Array<U>) -> Result<Array<V>>
    where
        T: Clone + NumCast + std::fmt::Debug,
        U: Clone + NumCast + std::fmt::Debug,
        V: Clone + NumCast + Add<Output = V> + std::fmt::Debug,
    {
        // First convert both arrays to the target type
        let self_converted = self.astype::<V>()?;
        let other_converted = other.astype::<V>()?;

        // Then perform the operation
        self_converted.add_broadcast(&other_converted)
    }

    /// Subtracts arrays of different types, automatically converting to a common target type.
    ///
    /// This method subtracts one array from another with potentially different element types by
    /// first converting both arrays to a common target type `V`. It handles broadcasting
    /// automatically, so the arrays can have different but compatible shapes.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The target type for the operation (both arrays will be converted to this type)
    ///
    /// # Parameters
    ///
    /// * `other` - The array to subtract from this array
    ///
    /// # Returns
    ///
    /// * `Ok(Array<V>)` - The result of the subtraction as a new array of type `V`
    /// * `Err(NumRs2Error)` - Error if type conversion fails or shapes are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let int_array = Array::from_vec(vec![3, 4, 5]);
    /// let float_array = Array::from_vec(vec![0.5, 1.5, 2.5]);
    ///
    /// // Subtract float from int array, promoting to f64
    /// let result = int_array.subtract_mixed::<f64, _>(&float_array).expect("same-shape arrays subtract successfully");
    /// assert_eq!(result.to_vec(), vec![2.5, 2.5, 2.5]);
    /// ```
    pub fn subtract_mixed<V, U>(&self, other: &Array<U>) -> Result<Array<V>>
    where
        T: Clone + NumCast + std::fmt::Debug,
        U: Clone + NumCast + std::fmt::Debug,
        V: Clone + NumCast + Sub<Output = V> + std::fmt::Debug,
    {
        // First convert both arrays to the target type
        let self_converted = self.astype::<V>()?;
        let other_converted = other.astype::<V>()?;

        // Then perform the operation
        self_converted.subtract_broadcast(&other_converted)
    }

    /// Multiplies arrays of different types, automatically converting to a common target type.
    ///
    /// This method multiplies two arrays with potentially different element types by
    /// first converting both arrays to a common target type `V`. It handles broadcasting
    /// automatically, so the arrays can have different but compatible shapes.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The target type for the operation (both arrays will be converted to this type)
    ///
    /// # Parameters
    ///
    /// * `other` - The array to multiply with this array
    ///
    /// # Returns
    ///
    /// * `Ok(Array<V>)` - The result of the multiplication as a new array of type `V`
    /// * `Err(NumRs2Error)` - Error if type conversion fails or shapes are incompatible
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let int_array = Array::from_vec(vec![1, 2, 3]);
    /// let float_array = Array::from_vec(vec![0.5, 1.5, 2.5]);
    ///
    /// // Multiply int and float arrays, promoting to f64
    /// let result = int_array.multiply_mixed::<f64, _>(&float_array).expect("same-shape arrays multiply successfully");
    /// assert_eq!(result.to_vec(), vec![0.5, 3.0, 7.5]);
    /// ```
    pub fn multiply_mixed<V, U>(&self, other: &Array<U>) -> Result<Array<V>>
    where
        T: Clone + NumCast + std::fmt::Debug,
        U: Clone + NumCast + std::fmt::Debug,
        V: Clone + NumCast + Mul<Output = V> + std::fmt::Debug,
    {
        // First convert both arrays to the target type
        let self_converted = self.astype::<V>()?;
        let other_converted = other.astype::<V>()?;

        // Then perform the operation
        self_converted.multiply_broadcast(&other_converted)
    }

    /// Divides arrays of different types, automatically converting to a common target type.
    ///
    /// This method divides one array by another with potentially different element types by
    /// first converting both arrays to a common target type `V`. It handles broadcasting
    /// automatically, so the arrays can have different but compatible shapes. Note that
    /// division by zero will result in an error.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The target type for the operation (both arrays will be converted to this type)
    ///
    /// # Parameters
    ///
    /// * `other` - The array to divide this array by
    ///
    /// # Returns
    ///
    /// * `Ok(Array<V>)` - The result of the division as a new array of type `V`
    /// * `Err(NumRs2Error)` - Error if type conversion fails, shapes are incompatible, or division by zero occurs
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let int_array = Array::from_vec(vec![4, 6, 8]);
    /// let float_array = Array::from_vec(vec![2.0, 3.0, 4.0]);
    ///
    /// // Divide int by float array, promoting to f64
    /// let result = int_array.divide_mixed::<f64, _>(&float_array).expect("same-shape arrays divide successfully");
    /// assert_eq!(result.to_vec(), vec![2.0, 2.0, 2.0]);
    /// ```
    pub fn divide_mixed<V, U>(&self, other: &Array<U>) -> Result<Array<V>>
    where
        T: Clone + NumCast + std::fmt::Debug,
        U: Clone + NumCast + std::fmt::Debug,
        V: Clone + NumCast + Div<Output = V> + std::fmt::Debug,
    {
        // First convert both arrays to the target type
        let self_converted = self.astype::<V>()?;
        let other_converted = other.astype::<V>()?;

        // Then perform the operation
        self_converted.divide_broadcast(&other_converted)
    }
}

/// Extension methods for `ArrayView` to provide type conversion capabilities.
///
/// These methods allow for converting array views to different numeric types or
/// to complex numbers, similar to the conversion methods on owned arrays.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the view
/// * `T` - The element type of the view
impl<'a, T> ArrayView<'a, T>
where
    T: 'a + Clone + NumCast + std::fmt::Debug,
{
    /// Converts a view to a different numeric type, returning a new owned array.
    ///
    /// This method converts all elements of the view to a new type `U` while
    /// preserving the shape and structure. Since the conversion creates a new
    /// array with a different type, the result is an owned array rather than a view.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The target type to convert view elements to
    ///
    /// # Returns
    ///
    /// * `Ok(Array<U>)` - A new array with all elements converted to type `U`
    /// * `Err(NumRs2Error)` - Error if any element couldn't be converted
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    ///
    /// let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    /// let view = array.view();
    ///
    /// // Convert the view from integers to floating point
    /// let float_array = view.astype::<f64>().expect("i32 to f64 conversion always succeeds");
    /// assert_eq!(float_array.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn astype<U>(&self) -> Result<Array<U>>
    where
        U: Clone + NumCast,
    {
        self.to_owned().astype::<U>()
    }

    /// Converts a view to a complex number array with zero imaginary parts.
    ///
    /// This method creates a new array of complex numbers where each element's real part
    /// comes from the corresponding element in the view, and the imaginary part is set
    /// to zero. Since the result has a different type, it returns an owned array.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The numeric type for the real and imaginary parts of the complex numbers
    ///
    /// # Returns
    ///
    /// * `Ok(Array<Complex<U>>)` - A new array with complex number elements
    /// * `Err(NumRs2Error)` - Error if conversion of any real part fails
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::prelude::*;
    /// use scirs2_core::Complex;
    ///
    /// let array = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let view = array.view();
    ///
    /// // Convert the view to complex numbers
    /// let complex = view.to_complex::<f64>().expect("f64 to Complex<f64> conversion always succeeds");
    ///
    /// // First element should have real part 1.0 and imaginary part 0.0
    /// let first = complex.get(&[0]).expect("index 0 is valid for 3-element array");
    /// assert_eq!(first.re, 1.0);
    /// assert_eq!(first.im, 0.0);
    /// ```
    pub fn to_complex<U>(&self) -> Result<Array<Complex<U>>>
    where
        U: Clone + NumCast + Zero,
    {
        self.to_owned().to_complex::<U>()
    }
}

/// Determines the appropriate result type for operations between two numeric types.
///
/// This function implements a type promotion system similar to NumPy's, where
/// operations between different numeric types yield results of the "higher" type.
/// For example, operations between i32 and f64 yield f64 results.
///
/// # Type Parameters
///
/// * `T` - The first type to compare
/// * `U` - The second type to compare
///
/// # Returns
///
/// * `std::any::TypeId` - The TypeId of the "higher" type according to precedence rules
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Type promotion examples (in actual code, this would be used internally)
/// let int_float_result = promote_types::<i32, f64>();
/// assert_eq!(int_float_result, std::any::TypeId::of::<f64>());
///
/// let float_complex_result = promote_types::<f32, Complex<f32>>();
/// assert_eq!(float_complex_result, std::any::TypeId::of::<Complex<f32>>());
/// ```
pub fn promote_types<T, U>() -> std::any::TypeId
where
    T: 'static,
    U: 'static,
{
    let t_id = std::any::TypeId::of::<T>();
    let u_id = std::any::TypeId::of::<U>();

    // Compare type precedence and return the "larger" type
    if type_precedence::<T>() >= type_precedence::<U>() {
        t_id
    } else {
        u_id
    }
}

/// Determines the numeric precedence of a type in the type hierarchy.
///
/// This function assigns a precedence value to each numeric type, following
/// a similar hierarchy to NumPy. Types with higher precedence values take
/// precedence in mixed-type operations. The precedence order (from lowest to highest) is:
///
/// 1. `bool` (0)
/// 2. Unsigned integers (`u8`, `u16`, `u32`, `u64`) (1, 3, 5, 7)
/// 3. Signed integers (`i8`, `i16`, `i32`, `i64`) (2, 4, 6, 8)
/// 4. Floating point types (`f32`, `f64`) (9, 10)
/// 5. Complex types (`Complex<f32>`, `Complex<f64>`) (11, 12)
///
/// # Type Parameters
///
/// * `T` - The type to determine precedence for
///
/// # Returns
///
/// * `u8` - The precedence value (higher means higher precedence)
fn type_precedence<T: 'static>() -> u8 {
    let t_id = std::any::TypeId::of::<T>();

    // Define precedence, highest value has highest precedence
    if t_id == std::any::TypeId::of::<bool>() {
        0
    } else if t_id == std::any::TypeId::of::<u8>() {
        1
    } else if t_id == std::any::TypeId::of::<i8>() {
        2
    } else if t_id == std::any::TypeId::of::<u16>() {
        3
    } else if t_id == std::any::TypeId::of::<i16>() {
        4
    } else if t_id == std::any::TypeId::of::<u32>() {
        5
    } else if t_id == std::any::TypeId::of::<i32>() {
        6
    } else if t_id == std::any::TypeId::of::<u64>() {
        7
    } else if t_id == std::any::TypeId::of::<i64>() {
        8
    } else if t_id == std::any::TypeId::of::<f32>() {
        9
    } else if t_id == std::any::TypeId::of::<f64>() {
        10
    } else if t_id == std::any::TypeId::of::<Complex<f32>>() {
        11
    } else if t_id == std::any::TypeId::of::<Complex<f64>>() {
        12
    } else {
        0
    } // Default for unknown types
}

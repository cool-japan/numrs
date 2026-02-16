//! Slice specification types and functions
//!
//! This module provides types and functions for array slicing:
//! - `SliceSpec` - Type representing slice specifications
//! - `s_` - Slice object builder function
//! - `NEWAXIS` - Constant for adding new axes

/// Type representing a slice object (equivalent to np.s_)
///
/// This is a simplified version of NumPy's slice objects.
/// In NumPy, s_[...] creates slice objects for advanced indexing.
#[derive(Debug, Clone, PartialEq)]
pub enum SliceSpec {
    /// Simple range slice: start:stop:step
    Range {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    /// Single index
    Index(isize),
    /// Ellipsis (...)
    Ellipsis,
    /// Newaxis (None)
    NewAxis,
}

impl SliceSpec {
    /// Create a new range slice
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::range(Some(1), Some(5), Some(2));
    /// // Equivalent to 1:5:2 in NumPy
    /// ```
    pub fn range(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Self {
        SliceSpec::Range { start, stop, step }
    }

    /// Create a slice from start to end with step 1
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::from_to(1, 5);
    /// // Equivalent to 1:5 in NumPy
    /// ```
    pub fn from_to(start: isize, stop: isize) -> Self {
        SliceSpec::Range {
            start: Some(start),
            stop: Some(stop),
            step: Some(1),
        }
    }

    /// Create a slice from start to end of array
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::from(2);
    /// // Equivalent to 2: in NumPy
    /// ```
    pub fn from(start: isize) -> Self {
        SliceSpec::Range {
            start: Some(start),
            stop: None,
            step: Some(1),
        }
    }

    /// Create a slice from beginning to stop
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::to(5);
    /// // Equivalent to :5 in NumPy
    /// ```
    pub fn to(stop: isize) -> Self {
        SliceSpec::Range {
            start: None,
            stop: Some(stop),
            step: Some(1),
        }
    }

    /// Create a slice with just step (equivalent to ::step)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::step(2);
    /// // Equivalent to ::2 in NumPy
    /// ```
    pub fn step(step: isize) -> Self {
        SliceSpec::Range {
            start: None,
            stop: None,
            step: Some(step),
        }
    }

    /// Create a full slice (equivalent to :)
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::array_ops::creation::SliceSpec;
    ///
    /// let slice = SliceSpec::full();
    /// // Equivalent to : in NumPy
    /// ```
    pub fn full() -> Self {
        SliceSpec::Range {
            start: None,
            stop: None,
            step: Some(1),
        }
    }
}

/// Slice object builder function (equivalent to np.s_[...])
///
/// Creates slice specifications for array indexing. This provides a convenient
/// way to create complex slice objects similar to NumPy's s_ indexing.
///
/// # Parameters
///
/// * `specs` - Vector of slice specifications
///
/// # Returns
///
/// Vector of slice specifications that can be used for array indexing
///
/// # Examples
///
/// ```
/// use numrs2::array_ops::creation::{s_, SliceSpec};
///
/// // Create slice specifications
/// let slices = s_(&[
///     SliceSpec::from_to(1, 5),
///     SliceSpec::step(2),
///     SliceSpec::Index(3),
/// ]);
///
/// // Equivalent to NumPy's s_[1:5, ::2, 3]
/// assert_eq!(slices.len(), 3);
/// ```
pub fn s_(specs: &[SliceSpec]) -> Vec<SliceSpec> {
    specs.to_vec()
}

/// Constant representing newaxis for array indexing (equivalent to np.newaxis)
///
/// This is used to add new axes to arrays during indexing operations.
/// In NumPy, newaxis is just an alias for None.
///
/// # Examples
///
/// ```
/// use numrs2::array_ops::creation::{NEWAXIS, SliceSpec};
///
/// // Using NEWAXIS in slice specifications
/// let slice_with_newaxis = SliceSpec::NewAxis;
/// assert_eq!(slice_with_newaxis, NEWAXIS);
/// ```
pub const NEWAXIS: SliceSpec = SliceSpec::NewAxis;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_spec() {
        // Test range slice
        let slice = SliceSpec::range(Some(1), Some(5), Some(2));
        match slice {
            SliceSpec::Range { start, stop, step } => {
                assert_eq!(start, Some(1));
                assert_eq!(stop, Some(5));
                assert_eq!(step, Some(2));
            }
            _ => panic!("Expected Range slice"),
        }

        // Test convenience constructors
        let from_to = SliceSpec::from_to(1, 5);
        let from = SliceSpec::from(2);
        let to = SliceSpec::to(5);
        let step = SliceSpec::step(2);
        let full = SliceSpec::full();

        // Verify types
        assert!(matches!(from_to, SliceSpec::Range { .. }));
        assert!(matches!(from, SliceSpec::Range { .. }));
        assert!(matches!(to, SliceSpec::Range { .. }));
        assert!(matches!(step, SliceSpec::Range { .. }));
        assert!(matches!(full, SliceSpec::Range { .. }));

        // Test index and special slices
        let index = SliceSpec::Index(3);
        assert!(matches!(index, SliceSpec::Index(3)));

        let ellipsis = SliceSpec::Ellipsis;
        assert!(matches!(ellipsis, SliceSpec::Ellipsis));

        let newaxis = SliceSpec::NewAxis;
        assert!(matches!(newaxis, SliceSpec::NewAxis));
        assert_eq!(newaxis, NEWAXIS);
    }

    #[test]
    fn test_s_() {
        // Test slice object builder
        let slices = s_(&[
            SliceSpec::from_to(1, 5),
            SliceSpec::step(2),
            SliceSpec::Index(3),
        ]);

        assert_eq!(slices.len(), 3);
        assert!(matches!(slices[0], SliceSpec::Range { .. }));
        assert!(matches!(slices[1], SliceSpec::Range { .. }));
        assert!(matches!(slices[2], SliceSpec::Index(3)));

        // Test empty slice list
        let empty_slices = s_(&[]);
        assert_eq!(empty_slices.len(), 0);
    }
}

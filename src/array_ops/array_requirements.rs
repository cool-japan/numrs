use crate::array::Array;
use crate::error::Result;

/// Return an array with the specified requirements
///
/// # Parameters
///
/// * `array` - Input array
/// * `requirements` - Requirements for the array, specified as a combination of flags
///
/// # Returns
///
/// * Array with the specified requirements. If the input array satisfies the requirements,
///   it may be returned as-is.
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create an array
/// let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
///
/// // Require a C-contiguous array (row-major order)
/// let b = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::C_LAYOUT)).expect("operation should succeed");
/// assert!(b.is_c_contiguous());
///
/// // Require a Fortran-contiguous array (column-major order)
/// let c = require(&a, Some(ArrayRequirements::CONTIGUOUS | ArrayRequirements::F_LAYOUT)).expect("operation should succeed");
/// assert!(c.is_f_contiguous());
/// ```
pub fn require<T: Clone>(
    array: &Array<T>,
    requirements: Option<ArrayRequirements>,
) -> Result<Array<T>> {
    // If no requirements are specified, return a copy of the input array
    let requirements = requirements.unwrap_or(ArrayRequirements::empty());

    // If no requirements are specified, return a copy of the input array
    if requirements.is_empty() {
        return Ok(array.clone());
    }

    // Check if we need a specific layout
    let need_c_layout = requirements.contains(ArrayRequirements::C_LAYOUT);
    let need_f_layout = requirements.contains(ArrayRequirements::F_LAYOUT);

    // Check if we need a contiguous array
    let need_contiguous = requirements.contains(ArrayRequirements::CONTIGUOUS);

    // Check if we need to own the data
    let _need_owner = requirements.contains(ArrayRequirements::OWNDATA);

    // Check if we need a writeable array
    let _need_writeable = requirements.contains(ArrayRequirements::WRITEABLE);

    // Check if the input array satisfies the requirements
    let meets_c_layout = if need_c_layout {
        array.is_c_contiguous()
    } else {
        true
    };
    let meets_f_layout = if need_f_layout {
        array.is_f_contiguous()
    } else {
        true
    };
    let meets_contiguous = if need_contiguous {
        array.is_contiguous()
    } else {
        true
    };

    // NumRS arrays always own their data and are writeable, so these are always met
    // If this changes in the future, we should check for these requirements

    // If all requirements are met, return the original array
    if meets_c_layout && meets_f_layout && meets_contiguous {
        return Ok(array.clone());
    }

    // Otherwise, create a new array that meets the requirements
    let mut result = array.clone();

    // If we need a C-contiguous array, convert to C layout
    if need_c_layout && !meets_c_layout {
        result = result.to_c_layout();
    }

    // If we need a Fortran-contiguous array, convert to F layout
    if need_f_layout && !meets_f_layout {
        result = result.to_f_layout();
    }

    // If we need a contiguous array but neither C nor F layout is specified,
    // prefer C layout (row-major order)
    if need_contiguous && !meets_contiguous && !need_c_layout && !need_f_layout {
        result = result.to_c_layout();
    }

    Ok(result)
}

/// Bitflags to specify requirements for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayRequirements(u32);

impl ArrayRequirements {
    // Flag values
    /// Ensure array is contiguous in memory
    pub const CONTIGUOUS: Self = Self(1 << 0);
    /// Ensure array is in C layout (row-major order)
    pub const C_LAYOUT: Self = Self(1 << 1);
    /// Ensure array is in Fortran layout (column-major order)
    pub const F_LAYOUT: Self = Self(1 << 2);
    /// Ensure array owns its data (not a view)
    pub const OWNDATA: Self = Self(1 << 3);
    /// Ensure array is writeable
    pub const WRITEABLE: Self = Self(1 << 4);

    /// Create an empty requirements set
    pub fn empty() -> Self {
        Self(0)
    }

    /// Check if the requirements are empty
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Check if the requirements contain a specific flag
    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ArrayRequirements {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for ArrayRequirements {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

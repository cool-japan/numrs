//! Core types and structures for array operations
//!
//! This module contains fundamental types used across array operations.

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

/// Enum to handle both single axis and multiple axes for concatenation
pub enum AxisArg {
    Single(usize),
    Multiple(Vec<usize>),
}

impl From<usize> for AxisArg {
    fn from(axis: usize) -> Self {
        AxisArg::Single(axis)
    }
}

impl From<&[usize]> for AxisArg {
    fn from(axes: &[usize]) -> Self {
        AxisArg::Multiple(axes.to_vec())
    }
}

impl From<Vec<usize>> for AxisArg {
    fn from(axes: Vec<usize>) -> Self {
        AxisArg::Multiple(axes)
    }
}

/// Enumeration to handle either sections or indices for split functions
pub enum SplitArg {
    Sections(usize),
    Indices(Vec<usize>),
}

impl From<usize> for SplitArg {
    fn from(sections: usize) -> Self {
        SplitArg::Sections(sections)
    }
}

impl From<&[usize]> for SplitArg {
    fn from(indices: &[usize]) -> Self {
        SplitArg::Indices(indices.to_vec())
    }
}

impl From<Vec<usize>> for SplitArg {
    fn from(indices: Vec<usize>) -> Self {
        SplitArg::Indices(indices)
    }
}

/// Extension trait for numeric operations
pub trait NumericExt {
    /// Check if a number is a multiple of another
    fn is_multiple_of(&self, divisor: Self) -> bool;
}

impl NumericExt for usize {
    #[allow(clippy::manual_is_multiple_of)]
    fn is_multiple_of(&self, divisor: Self) -> bool {
        if divisor == 0 {
            false
        } else {
            self % divisor == 0
        }
    }
}

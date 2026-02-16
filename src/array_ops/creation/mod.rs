//! Array creation functions
//!
//! This module provides various functions to create arrays, organized into submodules:
//!
//! - [`core`] - Core creation from data sources (fromfunction, frombuffer, fromiter, frommemmap, fromstring)
//! - [`grids`] - Grid/mesh creation (meshgrid, mgrid, ogrid)
//! - [`ranges`] - Range-based creation on log/geometric scales (logspace, geomspace)
//! - [`special`] - Special array creation (tri, diagflat, vander)
//! - [`contiguous`] - Memory layout functions (asanyarray, ascontiguousarray, asfortranarray, etc.)
//! - [`mod@concat`] - Concatenation helpers (r_concatenate, c_concatenate, ix_)
//! - [`mod@slice`] - Slice specification types (SliceSpec, s_, NEWAXIS)

pub mod concat;
pub mod contiguous;
pub mod core;
pub mod grids;
pub mod ranges;
pub mod slice;
pub mod special;

// Re-export all public items from submodules

// Core creation functions
pub use self::core::{frombuffer, fromfunction, fromiter, frommemmap, fromstring};

// Grid creation functions
pub use self::grids::{meshgrid, mgrid, ogrid};

// Range-based creation functions
pub use self::ranges::{geomspace, logspace};

// Special array creation functions
pub use self::special::{diagflat, tri, vander};

// Memory layout functions
pub use self::contiguous::{
    asanyarray, ascontiguousarray, asfortranarray, iscontiguous, isfortran, may_share_memory,
    shares_memory,
};

// Concatenation helpers
pub use self::concat::{c_concatenate, ix_, r_concatenate};

// Slice specification
pub use self::slice::{s_, SliceSpec, NEWAXIS};

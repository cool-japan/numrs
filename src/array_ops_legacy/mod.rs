//! Legacy array operations module
//!
//! This module provides a comprehensive set of array manipulation, transformation,
//! searching, and creation operations compatible with NumPy-style operations.
//!
//! # Submodules
//!
//! - [`core`] - Core types (ArrayRequirements, AxisArg, SplitArg)
//! - [`manipulation`] - Array manipulation (tile, repeat, concatenate, stack, block)
//! - [`splitting`] - Array splitting (hsplit, vsplit, dsplit, split)
//! - [`search`] - Search and sort operations (partition, searchsorted)
//! - [`transform`] - Shape transformations (diag, diagonal, swapaxes, moveaxis)
//! - [`selection`] - Conditional selection (where_cond, select, interp)
//! - [`creation`] - Array creation (fromfunction, frombuffer, fromiter)

pub mod core;
pub mod creation;
pub mod manipulation;
pub mod search;
pub mod selection;
pub mod splitting;
pub mod transform;

// Re-export all public items for backward compatibility

// Core types
pub use self::core::{ArrayRequirements, AxisArg, NumericExt, SplitArg};

// Manipulation operations
pub use manipulation::{block, c_, concatenate, r_, repeat, require, stack, tile};

// Splitting operations
pub use splitting::{dsplit, hsplit, split, vsplit};

// Search operations
pub use search::{partition, searchsorted};

// Transformation operations
pub use transform::{
    atleast_1d, atleast_2d, atleast_3d, broadcast_arrays, broadcast_to, diag, diagonal, moveaxis,
    rollaxis, swapaxes,
};

// Selection operations
pub use selection::{interp, select, where_cond};

// Creation operations
pub use creation::{frombuffer, fromfunction, fromiter};

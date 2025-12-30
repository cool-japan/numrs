pub mod advanced_indexing;
pub mod array_requirements;
pub mod atleast;
pub mod axis_ops;
pub mod broadcasting;
pub mod conditional;
pub mod creation;
pub mod diagonal;
pub mod joining;
pub mod manipulation;
pub mod sorting;
pub mod splitting;
pub mod string_ops;
pub mod tiling;

// Re-export all public functions
// Import advanced_indexing selectively to avoid conflicts with manipulation
pub use advanced_indexing::{
    apply_along_axis, apply_over_axes, compress as adv_compress, extract as adv_extract,
    place as adv_place, put as adv_put, putmask, take_along_axis,
};
pub use array_requirements::*;
pub use atleast::*;
pub use axis_ops::*;
pub use broadcasting::*;
pub use conditional::*;
pub use creation::*;
pub use diagonal::*;
pub use joining::{
    block, bmat, bmat_from_arrays, bmat_from_string, c_, column_stack, concatenate, dstack, hstack,
    r_, row_stack, stack, vstack,
};
pub use manipulation::*;
pub use sorting::*;
pub use splitting::*;
// Import string_ops selectively to avoid name collisions
pub use string_ops::{
    add as string_add, array_from_strings, capitalize, center, chartype, compare,
    count as string_count, endswith, find as string_find, join as string_join, ljust, lower,
    lstrip, mod_format, multiply as string_multiply, replace, rfind as string_rfind, rjust, rstrip,
    split as string_split, startswith, strip, title, upper, StringArray, StringElement,
};
pub use tiling::*;

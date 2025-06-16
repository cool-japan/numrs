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
pub use array_requirements::*;
pub use atleast::*;
pub use axis_ops::*;
pub use broadcasting::*;
pub use conditional::*;
pub use creation::*;
pub use diagonal::*;
pub use joining::*;
pub use manipulation::*;
pub use sorting::*;
pub use splitting::*;
// Import string_ops selectively to avoid name collisions
pub use string_ops::{
    StringArray, StringElement, array_from_strings,
    add as string_add, multiply as string_multiply, mod_format,
    center, ljust, rjust, strip, lstrip, rstrip,
    upper, lower, title, capitalize, replace,
    split as string_split, join as string_join,
    count as string_count, find as string_find, rfind as string_rfind,
    startswith, endswith, chartype, compare
};
pub use tiling::*;

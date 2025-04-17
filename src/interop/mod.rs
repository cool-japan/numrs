//! Interoperability with other Rust numerical libraries
//!
//! This module provides conversion functions to and from other popular Rust
//! numerical libraries like ndarray and nalgebra.

pub mod ndarray_compat;
pub mod nalgebra_compat;

#[cfg(test)]
mod tests {
    
    #[test]
    fn test_ndarray_conversions() {
        // Tests moved to the ndarray_compat module
    }
    
    #[test]
    fn test_nalgebra_conversions() {
        // Tests moved to the nalgebra_compat module
    }
}
//! Array Printing and Display Module
//!
//! This module provides NumPy-compatible array printing functionality,
//! including `set_printoptions` for controlling array display format.

use crate::array::Array;
use lazy_static::lazy_static;
use std::sync::RwLock;

/// Global print options for array display
pub struct PrintOptions {
    /// Maximum number of array elements to print (total)
    pub threshold: usize,
    /// Number of elements at beginning and end when array is summarized
    pub edgeitems: usize,
    /// Number of characters per line for the purpose of inserting line breaks (default 75)
    pub linewidth: usize,
    /// Whether to suppress printing of small floating point values using scientific notation
    pub suppress: bool,
    /// Number of digits of precision for floating point output (default 8)
    pub precision: usize,
    /// Controls printing of the sign of floats. Options: '-', '+', ' ' (default '-')
    pub sign: char,
    /// String used to separate array elements (default ' ')
    pub separator: String,
    /// String used as prefix (default '')
    pub prefix: String,
    /// String used as suffix (default '')
    pub suffix: String,
}

impl Clone for PrintOptions {
    fn clone(&self) -> Self {
        Self {
            threshold: self.threshold,
            edgeitems: self.edgeitems,
            linewidth: self.linewidth,
            suppress: self.suppress,
            precision: self.precision,
            sign: self.sign,
            separator: self.separator.clone(),
            prefix: self.prefix.clone(),
            suffix: self.suffix.clone(),
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            threshold: 1000,
            edgeitems: 3,
            linewidth: 75,
            suppress: false,
            precision: 8,
            sign: '-',
            separator: " ".to_string(),
            prefix: "".to_string(),
            suffix: "".to_string(),
        }
    }
}

lazy_static! {
    static ref PRINT_OPTIONS: RwLock<PrintOptions> = RwLock::new(PrintOptions::default());
}

/// Set printing options for arrays.
///
/// These options determine the way floating point numbers, arrays and
/// other NumPy objects are displayed.
///
/// # Arguments
///
/// * `precision` - Number of digits of precision for floating point output (default 8)
/// * `threshold` - Total number of array elements which trigger summarization rather than full repr (default 1000)
/// * `edgeitems` - Number of array items in summary at beginning and end of each dimension (default 3)
/// * `linewidth` - Number of characters per line for the purpose of inserting line breaks (default 75)
/// * `suppress` - Whether to suppress printing of small floating point values using scientific notation (default false)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::printing::set_printoptions;
///
/// // Set precision to 2 decimal places
/// set_printoptions(Some(2), None, None, None, None);
///
/// let arr = Array::from_vec(vec![1.0/3.0, 2.0/3.0, 1.0]);
/// println!("{}", arr); // Will display with 2 decimal places
/// ```
pub fn set_printoptions(
    precision: Option<usize>,
    threshold: Option<usize>,
    edgeitems: Option<usize>,
    linewidth: Option<usize>,
    suppress: Option<bool>,
) {
    let mut options = PRINT_OPTIONS
        .write()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire write lock");

    if let Some(p) = precision {
        options.precision = p;
    }
    if let Some(t) = threshold {
        options.threshold = t;
    }
    if let Some(e) = edgeitems {
        options.edgeitems = e;
    }
    if let Some(l) = linewidth {
        options.linewidth = l;
    }
    if let Some(s) = suppress {
        options.suppress = s;
    }
}

/// Get current printing options.
///
/// # Returns
///
/// A clone of the current PrintOptions
///
/// # Examples
///
/// ```
/// use numrs2::printing::{set_printoptions, get_printoptions};
///
/// set_printoptions(Some(4), None, None, None, None);
/// let options = get_printoptions();
/// assert_eq!(options.precision, 4);
/// ```
pub fn get_printoptions() -> PrintOptions {
    PRINT_OPTIONS
        .read()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire read lock")
        .clone()
}

/// Reset printing options to default values.
///
/// # Examples
///
/// ```
/// use numrs2::printing::{set_printoptions, reset_printoptions, get_printoptions};
///
/// set_printoptions(Some(2), Some(100), None, None, None);
/// reset_printoptions();
/// let options = get_printoptions();
/// assert_eq!(options.precision, 8); // Back to default
/// assert_eq!(options.threshold, 1000); // Back to default
/// ```
pub fn reset_printoptions() {
    let mut options = PRINT_OPTIONS
        .write()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire write lock");
    *options = PrintOptions::default();
}

/// Format a single floating point value according to current print options
pub fn format_float(value: f64) -> String {
    let options = PRINT_OPTIONS
        .read()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire read lock");

    if options.suppress && value.abs() < 10_f64.powi(-(options.precision as i32)) {
        return "0".to_string();
    }

    match options.sign {
        '+' => format!("{:+.*}", options.precision, value),
        ' ' => {
            if value >= 0.0 {
                format!(" {:.*}", options.precision, value)
            } else {
                format!("{:.*}", options.precision, value)
            }
        }
        _ => format!("{:.*}", options.precision, value),
    }
}

/// Format a single integer value according to current print options
pub fn format_int<T: std::fmt::Display>(value: T) -> String {
    let options = PRINT_OPTIONS
        .read()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire read lock");

    match options.sign {
        '+' => format!("{:+}", value),
        ' ' => format!(" {}", value),
        _ => format!("{}", value),
    }
}

/// Utility function to print an array with custom options temporarily
///
/// # Arguments
///
/// * `array` - The array to print
/// * `precision` - Temporary precision setting
/// * `threshold` - Temporary threshold setting
/// * `edgeitems` - Temporary edgeitems setting
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::printing::array_str;
///
/// let arr = Array::from_vec(vec![1.0/3.0, 2.0/3.0, 1.0]);
/// let formatted = array_str(&arr, Some(2), None, None);
/// println!("{}", formatted);
/// ```
pub fn array_str<T>(
    array: &Array<T>,
    precision: Option<usize>,
    threshold: Option<usize>,
    edgeitems: Option<usize>,
) -> String
where
    T: Clone + std::fmt::Display + std::fmt::Debug,
{
    // Save current options
    let original_options = get_printoptions();

    // Set temporary options
    set_printoptions(precision, threshold, edgeitems, None, None);

    // Format the array using existing Display implementation
    let result = format!("{}", array);

    // Restore original options
    let mut options = PRINT_OPTIONS
        .write()
        .expect("PRINT_OPTIONS RwLock poisoned: failed to acquire write lock");
    *options = original_options;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_set_and_get_printoptions() {
        // Save original options
        let original = get_printoptions();

        // Set new options
        set_printoptions(Some(2), Some(100), Some(2), Some(50), Some(true));

        let options = get_printoptions();
        assert_eq!(options.precision, 2);
        assert_eq!(options.threshold, 100);
        assert_eq!(options.edgeitems, 2);
        assert_eq!(options.linewidth, 50);
        assert!(options.suppress);

        // Restore original options
        let mut global_options = PRINT_OPTIONS
            .write()
            .expect("PRINT_OPTIONS lock should not be poisoned in test");
        *global_options = original;
    }

    #[test]
    fn test_reset_printoptions() {
        // Change options
        set_printoptions(Some(2), Some(100), None, None, None);

        // Reset to defaults
        reset_printoptions();

        let options = get_printoptions();
        assert_eq!(options.precision, 8);
        assert_eq!(options.threshold, 1000);
        assert_eq!(options.edgeitems, 3);
    }

    #[test]
    fn test_array_display_1d() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let display = format!("{}", arr);
        assert!(display.contains("["));
        assert!(display.contains("]"));
        assert!(display.contains("1"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_array_display_2d() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let display = format!("{}", arr);
        assert!(display.contains("["));
        assert!(display.contains("]"));
        assert!(display.contains("1"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
        assert!(display.contains("4"));
    }

    #[test]
    fn test_array_str_function() {
        let arr = Array::from_vec(vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
        let formatted = array_str(&arr, Some(2), None, None);
        assert!(!formatted.is_empty());
    }

    #[test]
    fn test_format_float() {
        // Save original options
        let original = get_printoptions();

        set_printoptions(Some(2), None, None, None, None);
        let formatted = format_float(1.23456);
        assert!(formatted.contains("1.23"));

        // Restore original options
        let mut global_options = PRINT_OPTIONS
            .write()
            .expect("PRINT_OPTIONS lock should not be poisoned in test");
        *global_options = original;
    }

    #[test]
    fn test_suppress_small_values() {
        // Save original options
        let original = get_printoptions();

        set_printoptions(Some(8), None, None, None, Some(true));
        let formatted = format_float(1e-10);
        assert_eq!(formatted, "0");

        // Restore original options
        let mut global_options = PRINT_OPTIONS
            .write()
            .expect("PRINT_OPTIONS lock should not be poisoned in test");
        *global_options = original;
    }

    #[test]
    fn test_sign_formatting() {
        // Save original options
        let original = get_printoptions();

        // Test positive sign
        {
            let mut options = PRINT_OPTIONS
                .write()
                .expect("PRINT_OPTIONS lock should not be poisoned in test");
            options.sign = '+';
        }
        let formatted = format_float(1.23);
        assert!(formatted.starts_with('+'));

        // Restore original options
        let mut global_options = PRINT_OPTIONS
            .write()
            .expect("PRINT_OPTIONS lock should not be poisoned in test");
        *global_options = original;
    }
}

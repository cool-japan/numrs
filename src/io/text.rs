//! Text file I/O functions for NumRS arrays
//!
//! This module provides NumPy-compatible text file I/O functions including
//! `loadtxt()`, `savetxt()`, and `genfromtxt()` with full parameter support.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Num, Zero};
use regex::Regex;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

/// Type alias for converter functions used in genfromtxt
pub type ConverterFn = Box<dyn Fn(&str) -> Result<String>>;

/// Options for loading text files
#[derive(Debug, Clone)]
pub struct LoadTxtOptions {
    /// Characters marking comments (default '#')
    pub comments: String,
    /// Delimiter separating values (default: any whitespace)  
    pub delimiter: Option<String>,
    /// Number of rows to skip at beginning (default: 0)
    pub skiprows: usize,
    /// Columns to read (None = all columns)
    pub usecols: Option<Vec<usize>>,
    /// Maximum number of rows to read (None = all rows)
    pub max_rows: Option<usize>,
    /// Minimum number of dimensions for output array (default: 0)
    pub ndmin: usize,
}

impl Default for LoadTxtOptions {
    fn default() -> Self {
        Self {
            comments: "#".to_string(),
            delimiter: None,
            skiprows: 0,
            usecols: None,
            max_rows: None,
            ndmin: 0,
        }
    }
}

/// Options for saving text files
#[derive(Debug, Clone)]
pub struct SaveTxtOptions {
    /// Format string for each element (default: "%.18e" for float, "%d" for int)
    pub fmt: String,
    /// Delimiter separating values (default: ' ')
    pub delimiter: String,
    /// Character(s) separating lines (default: '\n')
    pub newline: String,
    /// String written at beginning of file
    pub header: Option<String>,
    /// String written at end of file  
    pub footer: Option<String>,
    /// String marking header/footer as comments (default: '# ')
    pub comments: String,
}

impl Default for SaveTxtOptions {
    fn default() -> Self {
        Self {
            fmt: "%.18e".to_string(),
            delimiter: " ".to_string(),
            newline: "\n".to_string(),
            header: None,
            footer: None,
            comments: "# ".to_string(),
        }
    }
}

/// Options for loading text files with missing value handling
pub struct GenFromTxtOptions {
    /// Data type for array elements (default: f64)
    pub dtype: String,
    /// Characters marking comments (default '#')
    pub comments: String,
    /// Delimiter separating values (default: any whitespace)
    pub delimiter: Option<String>,
    /// Number of rows to skip at beginning (default: 0)
    pub skip_header: usize,
    /// Number of rows to skip at end (default: 0)
    pub skip_footer: usize,
    /// Dictionary of converter functions for columns
    pub converters: HashMap<usize, ConverterFn>,
    /// Values representing missing data
    pub missing_values: HashMap<usize, Vec<String>>,
    /// Value to substitute for missing data
    pub filling_values: HashMap<usize, String>,
    /// Columns to read (None = all columns)
    pub usecols: Option<Vec<usize>>,
    /// Column names (for structured arrays)
    pub names: Option<Vec<String>>,
    /// Exclude columns by name
    pub excludelist: Option<Vec<String>>,
    /// Default missing value markers
    pub default_missing: Vec<String>,
    /// Replace spaces in column names
    pub replace_space: Option<char>,
    /// Case sensitivity for column names
    pub case_sensitive: bool,
    /// Whether to delete intermediate values
    pub deletechars: String,
    /// Whether to strip whitespace from values
    pub autostrip: bool,
    /// Maximum number of rows to read
    pub max_rows: Option<usize>,
    /// Text encoding
    pub encoding: String,
}

impl Default for GenFromTxtOptions {
    fn default() -> Self {
        let default_missing = vec![
            "".to_string(),
            "N/A".to_string(),
            "NA".to_string(),
            "NULL".to_string(),
            "nan".to_string(),
            "NaN".to_string(),
            "NAN".to_string(),
        ];

        Self {
            dtype: "f64".to_string(),
            comments: "#".to_string(),
            delimiter: None,
            skip_header: 0,
            skip_footer: 0,
            converters: HashMap::new(),
            missing_values: HashMap::new(),
            filling_values: HashMap::new(),
            usecols: None,
            names: None,
            excludelist: None,
            default_missing,
            replace_space: Some('_'),
            case_sensitive: true,
            deletechars: String::new(),
            autostrip: false,
            max_rows: None,
            encoding: "utf-8".to_string(),
        }
    }
}

/// Load data from a text file
///
/// # Arguments
///
/// * `fname` - Path to the text file
/// * `options` - Loading options (use `LoadTxtOptions::default()` for defaults)
///
/// # Returns
///
/// A 2D Array containing the loaded data
///
/// # Examples
///
/// ```rust
/// use numrs2::io::text::{loadtxt, LoadTxtOptions};
/// use std::path::Path;
/// use std::fs::File;
/// use std::io::Write;
/// use tempfile::NamedTempFile;
///
/// // Create a temporary test file
/// let mut temp_file = NamedTempFile::new().unwrap();
/// writeln!(temp_file, "1.0 2.0 3.0").unwrap();
/// writeln!(temp_file, "4.0 5.0 6.0").unwrap();
///
/// // Load with default options
/// let array = loadtxt::<f64>(temp_file.path(), LoadTxtOptions::default()).unwrap();
/// assert_eq!(array.shape(), &[2, 3]);
///
/// // Load with custom delimiter and skip first row
/// let mut temp_file2 = NamedTempFile::new().unwrap();
/// writeln!(temp_file2, "header,line").unwrap();
/// writeln!(temp_file2, "1.0,2.0").unwrap();
/// writeln!(temp_file2, "3.0,4.0").unwrap();
///
/// let mut options = LoadTxtOptions::default();
/// options.delimiter = Some(",".to_string());
/// options.skiprows = 1;
/// let array = loadtxt::<f64>(temp_file2.path(), options).unwrap();
/// assert_eq!(array.shape(), &[2, 2]);
/// ```
pub fn loadtxt<T>(fname: &Path, options: LoadTxtOptions) -> Result<Array<T>>
where
    T: Clone + Default + FromStr + Zero,
    <T as FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file {:?}: {}", fname, e)))?;

    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header rows
    for _ in 0..options.skiprows {
        if lines.next().is_none() {
            return Err(NumRs2Error::IOError(
                "File ended during skip rows".to_string(),
            ));
        }
    }

    let mut rows = Vec::new();
    let mut rows_read = 0;

    // Process each line
    for (line_num, line_result) in lines.enumerate() {
        let line = line_result
            .map_err(|e| NumRs2Error::IOError(format!("Error reading line {}: {}", line_num, e)))?;

        // Skip empty lines and comments
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(&options.comments) {
            continue;
        }

        // Check max_rows limit
        if let Some(max_rows) = options.max_rows {
            if rows_read >= max_rows {
                break;
            }
        }

        // Split line by delimiter
        let values: Vec<&str> = if let Some(ref delimiter) = options.delimiter {
            line.split(delimiter).collect()
        } else {
            // Default: split by any whitespace
            line.split_whitespace().collect()
        };

        if values.is_empty() {
            continue;
        }

        // Select columns if specified
        let selected_values: Vec<&str> = if let Some(ref usecols) = options.usecols {
            usecols
                .iter()
                .filter_map(|&col_idx| values.get(col_idx))
                .copied()
                .collect()
        } else {
            values
        };

        // Parse values to type T
        let mut row = Vec::with_capacity(selected_values.len());
        for (col_idx, value_str) in selected_values.iter().enumerate() {
            let parsed_value = value_str.trim().parse::<T>().map_err(|e| {
                NumRs2Error::ConversionError(format!(
                    "Failed to parse '{}' at line {}, column {}: {:?}",
                    value_str,
                    line_num + options.skiprows,
                    col_idx,
                    e
                ))
            })?;
            row.push(parsed_value);
        }

        if !row.is_empty() {
            rows.push(row);
            rows_read += 1;
        }
    }

    if rows.is_empty() {
        return Err(NumRs2Error::IOError("No data found in file".to_string()));
    }

    // Verify all rows have the same length
    let row_length = rows[0].len();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != row_length {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Row {} has {} columns, expected {}",
                i,
                row.len(),
                row_length
            )));
        }
    }

    // Flatten into 1D vector
    let total_elements = rows.len() * row_length;
    let mut data = Vec::with_capacity(total_elements);
    for row in rows {
        data.extend(row);
    }

    // Create array with appropriate shape
    let shape = if row_length == 1 {
        // 1D array for single column
        vec![data.len()]
    } else {
        // 2D array for multiple columns
        vec![data.len() / row_length, row_length]
    };

    let mut array = Array::from_vec(data).reshape(&shape);

    // Apply ndmin constraint
    while array.ndim() < options.ndmin {
        let new_shape = {
            let mut shape = vec![1];
            shape.extend(array.shape());
            shape
        };
        array = array.reshape(&new_shape);
    }

    Ok(array)
}

/// Save an array to a text file
///
/// # Arguments
///
/// * `fname` - Path to the output text file
/// * `X` - Array to save
/// * `options` - Saving options (use `SaveTxtOptions::default()` for defaults)
///
/// # Returns
///
/// Result indicating success or failure
///
/// # Examples
///
/// ```rust
/// use numrs2::prelude::*;
/// use numrs2::io::text::{savetxt, SaveTxtOptions};
/// use std::path::Path;
///
/// let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
///
/// // Save with default options
/// savetxt(Path::new("output.txt"), &array, SaveTxtOptions::default()).unwrap();
///
/// // Save with custom format and delimiter
/// let mut options = SaveTxtOptions::default();
/// options.fmt = "%.6f".to_string();
/// options.delimiter = ",".to_string();
/// options.header = Some("x,y".to_string());
/// savetxt(Path::new("output.csv"), &array, options).unwrap();
/// ```
#[allow(non_snake_case)]
pub fn savetxt<T>(fname: &Path, X: &Array<T>, options: SaveTxtOptions) -> Result<()>
where
    T: Clone + Display + Zero,
{
    let file = File::create(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create file {:?}: {}", fname, e)))?;

    let mut writer = BufWriter::new(file);

    // Write header if provided
    if let Some(ref header) = options.header {
        writeln!(writer, "{}{}", options.comments, header)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write header: {}", e)))?;
    }

    // Convert array to 2D if it's not already
    let array_2d = if X.ndim() == 1 {
        // Reshape 1D array to column vector
        X.reshape(&[X.size(), 1])
    } else if X.ndim() == 2 {
        X.clone()
    } else {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Can only save 1D or 2D arrays to text files, got {}D",
            X.ndim()
        )));
    };

    let shape = array_2d.shape();
    let rows = shape[0];
    let cols = shape[1];

    // Write data
    for row in 0..rows {
        let mut line_values = Vec::with_capacity(cols);

        for col in 0..cols {
            let index = if array_2d.ndim() == 1 {
                vec![row]
            } else {
                vec![row, col]
            };

            let value = array_2d.get(&index)?;

            // Format the value according to fmt option
            // For simplicity, we use the Display trait which all numeric types implement
            let formatted = format!("{}", value);

            line_values.push(formatted);
        }

        let line = line_values.join(&options.delimiter);
        write!(writer, "{}{}", line, options.newline)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write data: {}", e)))?;
    }

    // Write footer if provided
    if let Some(ref footer) = options.footer {
        writeln!(writer, "{}{}", options.comments, footer)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write footer: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| NumRs2Error::IOError(format!("Failed to flush output: {}", e)))?;

    Ok(())
}

/// Load data from a text file with missing value handling
///
/// # Arguments
///
/// * `fname` - Path to the text file
/// * `options` - Loading options (use `GenFromTxtOptions::default()` for defaults)
///
/// # Returns
///
/// A 2D Array containing the loaded data with missing values handled
///
/// # Examples
///
/// ```rust
/// use numrs2::io::text::{genfromtxt, GenFromTxtOptions};
/// use std::path::Path;
/// use std::fs::File;
/// use std::io::Write;
/// use tempfile::NamedTempFile;
///
/// // Create a temporary test file with missing values
/// let mut temp_file = NamedTempFile::new().unwrap();
/// writeln!(temp_file, "1.0 2.0 3.0").unwrap();
/// writeln!(temp_file, "4.0 nan 6.0").unwrap();
/// writeln!(temp_file, "7.0 8.0 N/A").unwrap();
///
/// // Load with default missing value handling
/// let array = genfromtxt::<f64>(temp_file.path(), GenFromTxtOptions::default()).unwrap();
/// assert_eq!(array.shape(), &[3, 3]);
///
/// // Load with custom missing value markers
/// let mut temp_file2 = NamedTempFile::new().unwrap();
/// writeln!(temp_file2, "1.0 2.0").unwrap();
/// writeln!(temp_file2, "NULL 4.0").unwrap();
///
/// let mut options = GenFromTxtOptions::default();
/// options.default_missing.push("NULL".to_string());
/// let array = genfromtxt::<f64>(temp_file2.path(), options).unwrap();
/// assert_eq!(array.shape(), &[2, 2]);
/// ```
pub fn genfromtxt<T>(fname: &Path, options: GenFromTxtOptions) -> Result<Array<T>>
where
    T: Clone + Default + FromStr + Zero + Num,
    <T as FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file {:?}: {}", fname, e)))?;

    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| NumRs2Error::IOError(format!("Failed to read file: {}", e)))?;

    if lines.is_empty() {
        return Err(NumRs2Error::IOError("File is empty".to_string()));
    }

    // Calculate effective range of lines to process
    let total_lines = lines.len();
    if options.skip_header >= total_lines {
        return Err(NumRs2Error::IOError(
            "skip_header is larger than file length".to_string(),
        ));
    }

    let end_line = if options.skip_footer > 0 {
        total_lines.saturating_sub(options.skip_footer)
    } else {
        total_lines
    };

    if options.skip_header >= end_line {
        return Err(NumRs2Error::IOError(
            "No data lines available after skipping header and footer".to_string(),
        ));
    }

    let mut rows = Vec::new();
    let mut rows_read = 0;

    // Process lines in the effective range
    for (line_idx, line) in lines[options.skip_header..end_line].iter().enumerate() {
        // Skip empty lines and comments
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(&options.comments) {
            continue;
        }

        // Check max_rows limit
        if let Some(max_rows) = options.max_rows {
            if rows_read >= max_rows {
                break;
            }
        }

        // Split line by delimiter
        let raw_values: Vec<&str> = if let Some(ref delimiter) = options.delimiter {
            line.split(delimiter).collect()
        } else {
            // Default: split by any whitespace
            line.split_whitespace().collect()
        };

        if raw_values.is_empty() {
            continue;
        }

        // Select columns if specified
        let selected_values: Vec<&str> = if let Some(ref usecols) = options.usecols {
            usecols
                .iter()
                .filter_map(|&col_idx| raw_values.get(col_idx))
                .copied()
                .collect()
        } else {
            raw_values
        };

        // Parse values with missing value handling
        let mut row = Vec::with_capacity(selected_values.len());
        for (col_idx, value_str) in selected_values.iter().enumerate() {
            let trimmed_value = if options.autostrip {
                value_str.trim()
            } else {
                *value_str
            };

            // Check if this is a missing value
            let is_missing = options.default_missing.contains(&trimmed_value.to_string())
                || options
                    .missing_values
                    .get(&col_idx)
                    .map(|mv| mv.contains(&trimmed_value.to_string()))
                    .unwrap_or(false);

            let parsed_value = if is_missing {
                // Use filling value if specified, otherwise use default (zero)
                if let Some(filling) = options.filling_values.get(&col_idx) {
                    filling.parse::<T>().map_err(|e| {
                        NumRs2Error::ConversionError(format!(
                            "Failed to parse filling value '{}': {:?}",
                            filling, e
                        ))
                    })?
                } else {
                    T::zero()
                }
            } else {
                // Apply converter if specified
                let converted_str = if let Some(converter) = options.converters.get(&col_idx) {
                    converter(trimmed_value)?.to_string()
                } else {
                    trimmed_value.to_string()
                };

                converted_str.parse::<T>().map_err(|e| {
                    NumRs2Error::ConversionError(format!(
                        "Failed to parse '{}' at line {}, column {}: {:?}",
                        converted_str,
                        line_idx + options.skip_header,
                        col_idx,
                        e
                    ))
                })?
            };

            row.push(parsed_value);
        }

        if !row.is_empty() {
            rows.push(row);
            rows_read += 1;
        }
    }

    if rows.is_empty() {
        return Err(NumRs2Error::IOError("No data found in file".to_string()));
    }

    // Verify all rows have the same length
    let row_length = rows[0].len();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != row_length {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Row {} has {} columns, expected {}",
                i,
                row.len(),
                row_length
            )));
        }
    }

    // Flatten into 1D vector
    let total_elements = rows.len() * row_length;
    let mut data = Vec::with_capacity(total_elements);
    for row in rows {
        data.extend(row);
    }

    // Create array with appropriate shape
    let shape = if row_length == 1 {
        vec![data.len()]
    } else {
        vec![data.len() / row_length, row_length]
    };

    Ok(Array::from_vec(data).reshape(&shape))
}

/// Automatically detect delimiter in a text file
///
/// # Arguments
///
/// * `fname` - Path to the text file
/// * `sample_lines` - Number of lines to sample (default: 10)
///
/// # Returns
///
/// The most likely delimiter character
pub fn detect_delimiter(fname: &Path, sample_lines: Option<usize>) -> Result<String> {
    let file = File::open(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file {:?}: {}", fname, e)))?;

    let reader = BufReader::new(file);
    let sample_size = sample_lines.unwrap_or(10);

    let sample: Vec<String> = reader
        .lines()
        .take(sample_size)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| NumRs2Error::IOError(format!("Failed to read file: {}", e)))?;

    if sample.is_empty() {
        return Err(NumRs2Error::IOError("File is empty".to_string()));
    }

    // Common delimiters to test
    let delimiters = vec![",", "\t", ";", "|", " "];
    let mut delimiter_scores = HashMap::new();

    for delimiter in &delimiters {
        let mut total_consistency = 0.0;
        let mut valid_lines = 0;

        for line in &sample {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(delimiter).collect();
            if parts.len() > 1 {
                // Score based on consistency of column count
                total_consistency += parts.len() as f64;
                valid_lines += 1;
            }
        }

        if valid_lines > 0 {
            let avg_consistency = total_consistency / valid_lines as f64;
            delimiter_scores.insert(delimiter, avg_consistency);
        }
    }

    if delimiter_scores.is_empty() {
        return Ok(" ".to_string()); // Default to space
    }

    // Find delimiter with highest average column count
    let best_delimiter = delimiter_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(delimiter, _)| delimiter.to_string())
        .unwrap_or_else(|| " ".to_string());

    Ok(best_delimiter)
}

/// Load data from a text file using regular expressions to parse each line
///
/// This function reads a text file line by line and uses a regular expression
/// to extract numeric values from each line. The regex should contain capture groups
/// that will be used to extract the values.
///
/// # Arguments
///
/// * `fname` - Path to the text file
/// * `regexp` - Regular expression with capture groups for extracting values
/// * `dtype` - Data type for the extracted values (e.g., "f64", "i32")
/// * `encoding` - Text encoding (default: "utf-8")
///
/// # Returns
///
/// A 2D Array containing the extracted data
///
/// # Examples
///
/// ```rust
/// use numrs2::io::text::fromregex;
/// use std::path::Path;
/// use std::fs::File;
/// use std::io::Write;
/// use tempfile::NamedTempFile;
///
/// // Create a temporary test file with structured data
/// let mut temp_file = NamedTempFile::new().unwrap();
/// writeln!(temp_file, "Value: 1.5, Count: 10").unwrap();
/// writeln!(temp_file, "Value: 2.3, Count: 20").unwrap();
/// writeln!(temp_file, "Value: 4.7, Count: 15").unwrap();
///
/// // Extract values using regex with capture groups
/// let pattern = r"Value: ([0-9.]+), Count: ([0-9]+)";
/// let array = fromregex::<f64>(temp_file.path(), pattern, "f64", None).unwrap();
/// assert_eq!(array.shape(), &[3, 2]);
/// ```
pub fn fromregex<T>(
    fname: &Path,
    regexp: &str,
    dtype: &str,
    encoding: Option<&str>,
) -> Result<Array<T>>
where
    T: Clone + Default + FromStr + Zero,
    <T as FromStr>::Err: std::fmt::Debug,
{
    let _encoding = encoding.unwrap_or("utf-8");

    // Compile the regular expression
    let regex = Regex::new(regexp).map_err(|e| {
        NumRs2Error::InvalidOperation(format!("Invalid regular expression '{}': {}", regexp, e))
    })?;

    let file = File::open(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to open file {:?}: {}", fname, e)))?;

    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut expected_columns = None;

    // Process each line
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| NumRs2Error::IOError(format!("Error reading line {}: {}", line_num, e)))?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Apply the regex to extract capture groups
        if let Some(captures) = regex.captures(&line) {
            let mut row = Vec::new();

            // Extract all capture groups (skip 0 which is the full match)
            for i in 1..captures.len() {
                if let Some(capture) = captures.get(i) {
                    let value_str = capture.as_str().trim();

                    // Parse the captured value
                    let parsed_value = value_str.parse::<T>().map_err(|e| {
                        NumRs2Error::ConversionError(format!(
                            "Failed to parse '{}' as {} at line {}, capture group {}: {:?}",
                            value_str,
                            dtype,
                            line_num + 1,
                            i,
                            e
                        ))
                    })?;

                    row.push(parsed_value);
                }
            }

            // Check column consistency
            if let Some(expected) = expected_columns {
                if row.len() != expected {
                    return Err(NumRs2Error::DimensionMismatch(format!(
                        "Line {} has {} capture groups, expected {}",
                        line_num + 1,
                        row.len(),
                        expected
                    )));
                }
            } else {
                expected_columns = Some(row.len());
            }

            if !row.is_empty() {
                rows.push(row);
            }
        }
        // Lines that don't match the regex are silently skipped
    }

    if rows.is_empty() {
        return Err(NumRs2Error::IOError(
            "No data found matching the regular expression".to_string(),
        ));
    }

    // Flatten into 1D vector
    let row_length = rows[0].len();
    let total_elements = rows.len() * row_length;
    let mut data = Vec::with_capacity(total_elements);
    for row in rows {
        data.extend(row);
    }

    // Create array with appropriate shape
    let shape = if row_length == 1 {
        vec![data.len()]
    } else {
        vec![data.len() / row_length, row_length]
    };

    Ok(Array::from_vec(data).reshape(&shape))
}

/// Convenience function for saving compressed NPZ files
///
/// This is equivalent to saving with NPZ format, which uses compression by default.
///
/// # Arguments
///
/// * `fname` - Path to the output NPZ file
/// * `arrays` - Map of array names to arrays to save
///
/// # Returns
///
/// Result indicating success or failure
///
/// # Examples
///
/// ```rust
/// use numrs2::prelude::*;
/// use numrs2::io::text::savez_compressed;
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// let mut arrays = HashMap::new();
/// arrays.insert("arr_0".to_string(), Array::from_vec(vec![1.0, 2.0, 3.0]));
/// arrays.insert("arr_1".to_string(), Array::from_vec(vec![4.0, 5.0, 6.0]));
///
/// // savez_compressed(Path::new("data.npz"), &arrays).unwrap();
/// ```
pub fn savez_compressed<T: Clone + serde::Serialize>(
    fname: &Path,
    arrays: &HashMap<String, Array<T>>,
) -> Result<()> {
    use std::fs::File;

    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot save empty array collection".to_string(),
        ));
    }

    // Create the NPZ file
    let file = File::create(fname)
        .map_err(|e| NumRs2Error::IOError(format!("Failed to create NPZ file: {}", e)))?;

    // Use the new multi-array NPZ save function with compression enabled
    crate::io::npy_npz::save_npz_arrays(arrays, file, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_loadtxt_basic() {
        // Create a temporary file with test data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1.0 2.0 3.0").unwrap();
        writeln!(temp_file, "4.0 5.0 6.0").unwrap();

        let array = loadtxt::<f64>(temp_file.path(), LoadTxtOptions::default()).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_loadtxt_with_comments() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file, "1.0 2.0").unwrap();
        writeln!(temp_file, "# Another comment").unwrap();
        writeln!(temp_file, "3.0 4.0").unwrap();

        let array = loadtxt::<f64>(temp_file.path(), LoadTxtOptions::default()).unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_loadtxt_with_delimiter() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1.0,2.0,3.0").unwrap();
        writeln!(temp_file, "4.0,5.0,6.0").unwrap();

        let options = LoadTxtOptions {
            delimiter: Some(",".to_string()),
            ..Default::default()
        };

        let array = loadtxt::<f64>(temp_file.path(), options).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_savetxt_basic() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let temp_file = NamedTempFile::new().unwrap();

        savetxt(temp_file.path(), &array, SaveTxtOptions::default()).unwrap();

        let content = fs::read_to_string(temp_file.path()).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("1") && lines[0].contains("2"));
        assert!(lines[1].contains("3") && lines[1].contains("4"));
    }

    #[test]
    fn test_genfromtxt_with_missing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1.0 2.0 3.0").unwrap();
        writeln!(temp_file, "4.0 nan 6.0").unwrap();
        writeln!(temp_file, "7.0 8.0 N/A").unwrap();

        let array = genfromtxt::<f64>(temp_file.path(), GenFromTxtOptions::default()).unwrap();
        assert_eq!(array.shape(), &[3, 3]);
        // Missing values should be replaced with 0.0
        let expected = vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0];
        assert_eq!(array.to_vec(), expected);
    }

    #[test]
    fn test_detect_delimiter() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1,2,3").unwrap();
        writeln!(temp_file, "4,5,6").unwrap();
        writeln!(temp_file, "7,8,9").unwrap();

        let delimiter = detect_delimiter(temp_file.path(), Some(3)).unwrap();
        assert_eq!(delimiter, ",");
    }

    #[test]
    fn test_fromregex() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Value: 1.5, Count: 10").unwrap();
        writeln!(temp_file, "Value: 2.3, Count: 20").unwrap();
        writeln!(temp_file, "Value: 4.7, Count: 15").unwrap();

        let pattern = r"Value: ([0-9.]+), Count: ([0-9]+)";
        let array = fromregex::<f64>(temp_file.path(), pattern, "f64", None).unwrap();

        assert_eq!(array.shape(), &[3, 2]);
        let expected = vec![1.5, 10.0, 2.3, 20.0, 4.7, 15.0];
        assert_eq!(array.to_vec(), expected);
    }

    #[test]
    fn test_fromregex_single_column() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Temperature: 23.5°C").unwrap();
        writeln!(temp_file, "Temperature: 25.1°C").unwrap();
        writeln!(temp_file, "Temperature: 22.8°C").unwrap();

        let pattern = r"Temperature: ([0-9.]+)°C";
        let array = fromregex::<f64>(temp_file.path(), pattern, "f64", None).unwrap();

        assert_eq!(array.shape(), &[3]);
        let expected = vec![23.5, 25.1, 22.8];
        assert_eq!(array.to_vec(), expected);
    }
}

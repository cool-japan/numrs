//! String array operations for NumRS2
//!
//! This module provides string array functionality similar to NumPy's `char` module,
//! including string creation, manipulation, and comparison operations.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Zero;

/// Represents a string array element that can handle both fixed and variable length strings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StringElement {
    /// Fixed-length string (similar to NumPy's 'S' dtype)
    Fixed { data: Vec<u8>, max_len: usize },
    /// Variable-length Unicode string (similar to NumPy's 'U' dtype)  
    Unicode(String),
}

impl StringElement {
    /// Create a new fixed-length string element
    pub fn fixed(s: &str, max_len: usize) -> Self {
        let mut data = s.as_bytes().to_vec();
        data.resize(max_len, 0); // Pad with null bytes
        Self::Fixed { data, max_len }
    }
    
    /// Create a new Unicode string element
    pub fn unicode<S: Into<String>>(s: S) -> Self {
        Self::Unicode(s.into())
    }
    
    /// Get the string content as a &str
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Fixed { data, .. } => {
                // Find the first null byte or use entire length
                let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
                std::str::from_utf8(&data[..end]).map_err(|_| {
                    NumRs2Error::ValueError("Invalid UTF-8 in fixed string".to_string())
                })
            }
            Self::Unicode(s) => Ok(s.as_str()),
        }
    }
    
    /// Get the string content as an owned String
    pub fn to_string(&self) -> Result<String> {
        self.as_str().map(|s| s.to_string())
    }
    
    /// Get the length of the string content (not the buffer size)
    pub fn len(&self) -> Result<usize> {
        self.as_str().map(|s| s.len())
    }
    
    /// Check if the string is empty
    pub fn is_empty(&self) -> Result<bool> {
        self.as_str().map(|s| s.is_empty())
    }
    
    /// Get the maximum capacity for fixed strings, or current length for Unicode strings
    pub fn capacity(&self) -> usize {
        match self {
            Self::Fixed { max_len, .. } => *max_len,
            Self::Unicode(s) => s.len(),
        }
    }
}

impl Default for StringElement {
    fn default() -> Self {
        Self::Unicode(String::new())
    }
}

impl std::fmt::Display for StringElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid utf-8>"),
        }
    }
}

impl Zero for StringElement {
    fn zero() -> Self {
        Self::Unicode(String::new())
    }

    fn is_zero(&self) -> bool {
        match self.is_empty() {
            Ok(empty) => empty,
            Err(_) => false,
        }
    }
}

impl std::ops::Add for StringElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self.to_string(), other.to_string()) {
            (Ok(s1), Ok(s2)) => Self::Unicode(format!("{}{}", s1, s2)),
            _ => Self::Unicode(String::new()),
        }
    }
}

impl PartialOrd for StringElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self.to_string(), other.to_string()) {
            (Ok(s1), Ok(s2)) => s1.partial_cmp(&s2),
            _ => None,
        }
    }
}

impl Ord for StringElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.to_string(), other.to_string()) {
            (Ok(s1), Ok(s2)) => s1.cmp(&s2),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

/// Type alias for string arrays
pub type StringArray = Array<StringElement>;

/// Create a string array from a vector of strings
///
/// # Parameters
///
/// * `strings` - Vector of string data
/// * `dtype` - String data type specification ("U" for Unicode, "S<len>" for fixed)
/// * `shape` - Optional shape for the resulting array
///
/// # Examples
///
/// ```
/// use numrs2::array_ops::string_ops::*;
///
/// // Create Unicode string array
/// let strings = vec!["hello", "world", "test"];
/// let arr = array_from_strings(&strings, "U", Some(&[3])).unwrap();
/// assert_eq!(arr.shape(), vec![3]);
///
/// // Create fixed-length string array
/// let arr = array_from_strings(&strings, "S10", Some(&[3])).unwrap();
/// assert_eq!(arr.shape(), vec![3]);
/// ```
pub fn array_from_strings<S: AsRef<str>>(
    strings: &[S],
    dtype: &str,
    shape: Option<&[usize]>,
) -> Result<StringArray> {
    let elements: Result<Vec<StringElement>> = if dtype == "U" {
        // Unicode strings
        strings
            .iter()
            .map(|s| Ok(StringElement::unicode(s.as_ref())))
            .collect()
    } else if dtype.starts_with('S') {
        // Fixed-length strings
        let len_str = &dtype[1..];
        let max_len: usize = len_str.parse().map_err(|_| {
            NumRs2Error::ValueError(format!("Invalid string dtype: {}", dtype))
        })?;
        
        strings
            .iter()
            .map(|s| Ok(StringElement::fixed(s.as_ref(), max_len)))
            .collect()
    } else {
        return Err(NumRs2Error::ValueError(format!(
            "Unsupported string dtype: {}. Use 'U' for Unicode or 'S<len>' for fixed-length",
            dtype
        )));
    };
    
    let elements = elements?;
    let array = Array::from_vec(elements);
    
    match shape {
        Some(s) => {
            let expected_size: usize = s.iter().product();
            if array.size() != expected_size {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![expected_size],
                    actual: vec![array.size()],
                });
            }
            Ok(array.reshape(s))
        }
        None => Ok(array),
    }
}

/// Add two string arrays element-wise
///
/// Similar to NumPy's `np.char.add()`
pub fn add(arr1: &StringArray, arr2: &StringArray) -> Result<StringArray> {
    if arr1.shape() != arr2.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: arr1.shape().to_vec(),
            actual: arr2.shape().to_vec(),
        });
    }
    
    let mut result = Vec::with_capacity(arr1.size());
    let arr1_data = arr1.to_vec();
    let arr2_data = arr2.to_vec();
    
    for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
        let combined = format!("{}{}", s1.to_string()?, s2.to_string()?);
        result.push(StringElement::unicode(combined));
    }
    
    Ok(Array::from_vec(result).reshape(&arr1.shape()))
}

/// Multiply string array by integer array element-wise
///
/// Similar to NumPy's `np.char.multiply()`
pub fn multiply(arr: &StringArray, times: &Array<i32>) -> Result<StringArray> {
    if arr.shape() != times.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: arr.shape().to_vec(),
            actual: times.shape().to_vec(),
        });
    }
    
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    let times_data = times.to_vec();
    
    for (s, &n) in arr_data.iter().zip(times_data.iter()) {
        if n < 0 {
            result.push(StringElement::unicode(""));
        } else {
            let repeated = s.to_string()?.repeat(n as usize);
            result.push(StringElement::unicode(repeated));
        }
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Apply modulo formatting to string array
///
/// Similar to NumPy's `np.char.mod()`
pub fn mod_format(arr: &StringArray, values: &Array<f64>) -> Result<StringArray> {
    if arr.shape() != values.shape() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: arr.shape().to_vec(),
            actual: values.shape().to_vec(),
        });
    }
    
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    let values_data = values.to_vec();
    
    for (s, &val) in arr_data.iter().zip(values_data.iter()) {
        let format_str = s.to_string()?;
        let formatted = if format_str.contains("%") {
            // Simple format string replacement - this could be enhanced
            format_str.replace("%f", &val.to_string())
                     .replace("%d", &(val as i64).to_string())
                     .replace("%g", &val.to_string())
        } else {
            format_str
        };
        result.push(StringElement::unicode(formatted));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Center string array elements
///
/// Similar to NumPy's `np.char.center()`
pub fn center(arr: &StringArray, width: usize, fillchar: Option<char>) -> Result<StringArray> {
    let fill = fillchar.unwrap_or(' ');
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let centered = if string.len() >= width {
            string
        } else {
            let padding = width - string.len();
            let left_pad = padding / 2;
            let right_pad = padding - left_pad;
            format!("{}{}{}", 
                    fill.to_string().repeat(left_pad),
                    string,
                    fill.to_string().repeat(right_pad))
        };
        result.push(StringElement::unicode(centered));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Left justify string array elements
///
/// Similar to NumPy's `np.char.ljust()`
pub fn ljust(arr: &StringArray, width: usize, fillchar: Option<char>) -> Result<StringArray> {
    let fill = fillchar.unwrap_or(' ');
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let justified = if string.len() >= width {
            string
        } else {
            let padding = width - string.len();
            format!("{}{}", string, fill.to_string().repeat(padding))
        };
        result.push(StringElement::unicode(justified));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Right justify string array elements
///
/// Similar to NumPy's `np.char.rjust()`
pub fn rjust(arr: &StringArray, width: usize, fillchar: Option<char>) -> Result<StringArray> {
    let fill = fillchar.unwrap_or(' ');
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let justified = if string.len() >= width {
            string
        } else {
            let padding = width - string.len();
            format!("{}{}", fill.to_string().repeat(padding), string)
        };
        result.push(StringElement::unicode(justified));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Strip leading and trailing characters from string array elements
///
/// Similar to NumPy's `np.char.strip()`
pub fn strip(arr: &StringArray, chars: Option<&str>) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let stripped = match chars {
            Some(chars) => {
                let char_set: std::collections::HashSet<char> = chars.chars().collect();
                string.trim_matches(|c| char_set.contains(&c)).to_string()
            }
            None => string.trim().to_string(),
        };
        result.push(StringElement::unicode(stripped));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Strip leading characters from string array elements
///
/// Similar to NumPy's `np.char.lstrip()`
pub fn lstrip(arr: &StringArray, chars: Option<&str>) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let stripped = match chars {
            Some(chars) => {
                let char_set: std::collections::HashSet<char> = chars.chars().collect();
                string.trim_start_matches(|c| char_set.contains(&c)).to_string()
            }
            None => string.trim_start().to_string(),
        };
        result.push(StringElement::unicode(stripped));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Strip trailing characters from string array elements
///
/// Similar to NumPy's `np.char.rstrip()`
pub fn rstrip(arr: &StringArray, chars: Option<&str>) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let stripped = match chars {
            Some(chars) => {
                let char_set: std::collections::HashSet<char> = chars.chars().collect();
                string.trim_end_matches(|c| char_set.contains(&c)).to_string()
            }
            None => string.trim_end().to_string(),
        };
        result.push(StringElement::unicode(stripped));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Convert string array elements to uppercase
///
/// Similar to NumPy's `np.char.upper()`
pub fn upper(arr: &StringArray) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let upper_str = s.to_string()?.to_uppercase();
        result.push(StringElement::unicode(upper_str));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Convert string array elements to lowercase
///
/// Similar to NumPy's `np.char.lower()`
pub fn lower(arr: &StringArray) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let lower_str = s.to_string()?.to_lowercase();
        result.push(StringElement::unicode(lower_str));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Convert string array elements to title case
///
/// Similar to NumPy's `np.char.title()`
pub fn title(arr: &StringArray) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let mut title_str = String::new();
        let mut capitalize_next = true;
        
        for c in string.chars() {
            if c.is_alphabetic() {
                if capitalize_next {
                    title_str.push(c.to_uppercase().next().unwrap_or(c));
                    capitalize_next = false;
                } else {
                    title_str.push(c.to_lowercase().next().unwrap_or(c));
                }
            } else {
                title_str.push(c);
                capitalize_next = true;
            }
        }
        result.push(StringElement::unicode(title_str));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Capitalize first character of string array elements
///
/// Similar to NumPy's `np.char.capitalize()`
pub fn capitalize(arr: &StringArray) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let capitalized = if string.is_empty() {
            string
        } else {
            let mut chars: Vec<char> = string.chars().collect();
            if let Some(first) = chars.get_mut(0) {
                *first = first.to_uppercase().next().unwrap_or(*first);
            }
            for c in chars.iter_mut().skip(1) {
                *c = c.to_lowercase().next().unwrap_or(*c);
            }
            chars.into_iter().collect()
        };
        result.push(StringElement::unicode(capitalized));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Replace occurrences of substring in string array elements
///
/// Similar to NumPy's `np.char.replace()`
pub fn replace(arr: &StringArray, old: &str, new: &str, count: Option<usize>) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let replaced = match count {
            Some(n) => string.replacen(old, new, n),
            None => string.replace(old, new),
        };
        result.push(StringElement::unicode(replaced));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Split string array elements by delimiter
///
/// Similar to NumPy's `np.char.split()`
pub fn split(arr: &StringArray, sep: Option<&str>, maxsplit: Option<usize>) -> Result<Vec<Vec<String>>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let parts: Vec<String> = match (sep, maxsplit) {
            (Some(delimiter), Some(max)) => {
                string.splitn(max + 1, delimiter).map(|s| s.to_string()).collect()
            }
            (Some(delimiter), None) => {
                string.split(delimiter).map(|s| s.to_string()).collect()
            }
            (None, Some(max)) => {
                string.split_whitespace().take(max + 1).map(|s| s.to_string()).collect()
            }
            (None, None) => {
                string.split_whitespace().map(|s| s.to_string()).collect()
            }
        };
        result.push(parts);
    }
    
    Ok(result)
}

/// Join array of strings with separator
///
/// Similar to NumPy's `np.char.join()`
pub fn join(sep: &str, arr: &[Vec<String>]) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.len());
    
    for string_vec in arr.iter() {
        let joined = string_vec.join(sep);
        result.push(StringElement::unicode(joined));
    }
    
    Ok(Array::from_vec(result))
}

/// Count occurrences of substring in string array elements
///
/// Similar to NumPy's `np.char.count()`
pub fn count(arr: &StringArray, sub: &str, start: Option<usize>, end: Option<usize>) -> Result<Array<i32>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let search_string = match (start, end) {
            (Some(start), Some(end)) => {
                if start >= string.len() {
                    ""
                } else {
                    let end = end.min(string.len());
                    &string[start..end]
                }
            }
            (Some(start), None) => {
                if start >= string.len() {
                    ""
                } else {
                    &string[start..]
                }
            }
            (None, Some(end)) => {
                let end = end.min(string.len());
                &string[..end]
            }
            (None, None) => &string,
        };
        
        let count = search_string.matches(sub).count() as i32;
        result.push(count);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Find index of first occurrence of substring
///
/// Similar to NumPy's `np.char.find()`
pub fn find(arr: &StringArray, sub: &str, start: Option<usize>, end: Option<usize>) -> Result<Array<i32>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let search_string = match (start, end) {
            (Some(start), Some(end)) => {
                if start >= string.len() {
                    ""
                } else {
                    let end = end.min(string.len());
                    &string[start..end]
                }
            }
            (Some(start), None) => {
                if start >= string.len() {
                    ""
                } else {
                    &string[start..]
                }
            }
            (None, Some(end)) => {
                let end = end.min(string.len());
                &string[..end]
            }
            (None, None) => &string,
        };
        
        let index = search_string.find(sub)
            .map(|i| (start.unwrap_or(0) + i) as i32)
            .unwrap_or(-1);
        result.push(index);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Find index of last occurrence of substring
///
/// Similar to NumPy's `np.char.rfind()`
pub fn rfind(arr: &StringArray, sub: &str, start: Option<usize>, end: Option<usize>) -> Result<Array<i32>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let search_string = match (start, end) {
            (Some(start), Some(end)) => {
                if start >= string.len() {
                    ""
                } else {
                    let end = end.min(string.len());
                    &string[start..end]
                }
            }
            (Some(start), None) => {
                if start >= string.len() {
                    ""
                } else {
                    &string[start..]
                }
            }
            (None, Some(end)) => {
                let end = end.min(string.len());
                &string[..end]
            }
            (None, None) => &string,
        };
        
        let index = search_string.rfind(sub)
            .map(|i| (start.unwrap_or(0) + i) as i32)
            .unwrap_or(-1);
        result.push(index);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Check if string array elements start with prefix
///
/// Similar to NumPy's `np.char.startswith()`
pub fn startswith(arr: &StringArray, prefix: &str, start: Option<usize>, end: Option<usize>) -> Result<Array<bool>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let search_string = match (start, end) {
            (Some(start), Some(end)) => {
                if start >= string.len() {
                    ""
                } else {
                    let end = end.min(string.len());
                    &string[start..end]
                }
            }
            (Some(start), None) => {
                if start >= string.len() {
                    ""
                } else {
                    &string[start..]
                }
            }
            (None, Some(end)) => {
                let end = end.min(string.len());
                &string[..end]
            }
            (None, None) => &string,
        };
        
        let starts = search_string.starts_with(prefix);
        result.push(starts);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Check if string array elements end with suffix
///
/// Similar to NumPy's `np.char.endswith()`
pub fn endswith(arr: &StringArray, suffix: &str, start: Option<usize>, end: Option<usize>) -> Result<Array<bool>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let search_string = match (start, end) {
            (Some(start), Some(end)) => {
                if start >= string.len() {
                    ""
                } else {
                    let end = end.min(string.len());
                    &string[start..end]
                }
            }
            (Some(start), None) => {
                if start >= string.len() {
                    ""
                } else {
                    &string[start..]
                }
            }
            (None, Some(end)) => {
                let end = end.min(string.len());
                &string[..end]
            }
            (None, None) => &string,
        };
        
        let ends = search_string.ends_with(suffix);
        result.push(ends);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Check character type properties
pub mod chartype {
    use super::*;
    
    /// Check if string array elements are alphabetic
    pub fn isalpha(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let is_alpha = !string.is_empty() && string.chars().all(|c| c.is_alphabetic());
            result.push(is_alpha);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements are alphanumeric
    pub fn isalnum(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let is_alnum = !string.is_empty() && string.chars().all(|c| c.is_alphanumeric());
            result.push(is_alnum);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements are digits
    pub fn isdigit(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let is_digit = !string.is_empty() && string.chars().all(|c| c.is_ascii_digit());
            result.push(is_digit);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements are lowercase
    pub fn islower(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let has_alpha = string.chars().any(|c| c.is_alphabetic());
            let is_lower = has_alpha && string.chars().all(|c| !c.is_alphabetic() || c.is_lowercase());
            result.push(is_lower);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements are uppercase
    pub fn isupper(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let has_alpha = string.chars().any(|c| c.is_alphabetic());
            let is_upper = has_alpha && string.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());
            result.push(is_upper);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements are title case
    pub fn istitle(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let is_title = if string.is_empty() {
                false
            } else {
                let mut word_start = true;
                let mut has_alpha = false;
                let mut is_title_case = true;
                
                for c in string.chars() {
                    if c.is_alphabetic() {
                        has_alpha = true;
                        if word_start {
                            if !c.is_uppercase() {
                                is_title_case = false;
                                break;
                            }
                            word_start = false;
                        } else {
                            if !c.is_lowercase() {
                                is_title_case = false;
                                break;
                            }
                        }
                    } else {
                        word_start = true;
                    }
                }
                
                has_alpha && is_title_case
            };
            result.push(is_title);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Check if string array elements contain only whitespace
    pub fn isspace(arr: &StringArray) -> Result<Array<bool>> {
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let is_space = !string.is_empty() && string.chars().all(|c| c.is_whitespace());
            result.push(is_space);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
}

/// String comparison operations
pub mod compare {
    use super::*;
    
    /// Element-wise string equality comparison
    pub fn equal(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        if arr1.shape() != arr2.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr1.shape().to_vec(),
                actual: arr2.shape().to_vec(),
            });
        }
        
        let mut result = Vec::with_capacity(arr1.size());
        let arr1_data = arr1.to_vec();
        let arr2_data = arr2.to_vec();
        
        for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
            let equal = s1.to_string()? == s2.to_string()?;
            result.push(equal);
        }
        
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
    
    /// Element-wise string inequality comparison
    pub fn not_equal(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        let eq_result = equal(arr1, arr2)?;
        let result: Vec<bool> = eq_result.to_vec().into_iter().map(|x| !x).collect();
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
    
    /// Element-wise string less than comparison
    pub fn less(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        if arr1.shape() != arr2.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr1.shape().to_vec(),
                actual: arr2.shape().to_vec(),
            });
        }
        
        let mut result = Vec::with_capacity(arr1.size());
        let arr1_data = arr1.to_vec();
        let arr2_data = arr2.to_vec();
        
        for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
            let less = s1.to_string()? < s2.to_string()?;
            result.push(less);
        }
        
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
    
    /// Element-wise string less than or equal comparison
    pub fn less_equal(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        if arr1.shape() != arr2.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr1.shape().to_vec(),
                actual: arr2.shape().to_vec(),
            });
        }
        
        let mut result = Vec::with_capacity(arr1.size());
        let arr1_data = arr1.to_vec();
        let arr2_data = arr2.to_vec();
        
        for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
            let less_eq = s1.to_string()? <= s2.to_string()?;
            result.push(less_eq);
        }
        
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
    
    /// Element-wise string greater than comparison
    pub fn greater(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        if arr1.shape() != arr2.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr1.shape().to_vec(),
                actual: arr2.shape().to_vec(),
            });
        }
        
        let mut result = Vec::with_capacity(arr1.size());
        let arr1_data = arr1.to_vec();
        let arr2_data = arr2.to_vec();
        
        for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
            let greater = s1.to_string()? > s2.to_string()?;
            result.push(greater);
        }
        
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
    
    /// Element-wise string greater than or equal comparison
    pub fn greater_equal(arr1: &StringArray, arr2: &StringArray) -> Result<Array<bool>> {
        if arr1.shape() != arr2.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: arr1.shape().to_vec(),
                actual: arr2.shape().to_vec(),
            });
        }
        
        let mut result = Vec::with_capacity(arr1.size());
        let arr1_data = arr1.to_vec();
        let arr2_data = arr2.to_vec();
        
        for (s1, s2) in arr1_data.iter().zip(arr2_data.iter()) {
            let greater_eq = s1.to_string()? >= s2.to_string()?;
            result.push(greater_eq);
        }
        
        Ok(Array::from_vec(result).reshape(&arr1.shape()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_element_creation() {
        // Test Unicode string
        let unicode = StringElement::unicode("hello");
        assert_eq!(unicode.to_string().unwrap(), "hello");
        assert_eq!(unicode.len().unwrap(), 5);
        assert!(!unicode.is_empty().unwrap());
        
        // Test fixed-length string
        let fixed = StringElement::fixed("test", 10);
        assert_eq!(fixed.to_string().unwrap(), "test");
        assert_eq!(fixed.len().unwrap(), 4);
        assert_eq!(fixed.capacity(), 10);
    }
    
    #[test]
    fn test_array_from_strings() {
        let strings = vec!["hello", "world", "test"];
        
        // Test Unicode array
        let unicode_arr = array_from_strings(&strings, "U", Some(&[3])).unwrap();
        assert_eq!(unicode_arr.shape(), vec![3]);
        assert_eq!(unicode_arr.get(&[0]).unwrap().to_string().unwrap(), "hello");
        
        // Test fixed-length array
        let fixed_arr = array_from_strings(&strings, "S10", Some(&[3])).unwrap();
        assert_eq!(fixed_arr.shape(), vec![3]);
        assert_eq!(fixed_arr.get(&[1]).unwrap().to_string().unwrap(), "world");
    }
    
    #[test]
    fn test_string_operations() {
        let strings1 = vec!["hello", "world"];
        let strings2 = vec![" ", "!"];
        let arr1 = array_from_strings(&strings1, "U", None).unwrap();
        let arr2 = array_from_strings(&strings2, "U", None).unwrap();
        
        // Test add operation
        let result = add(&arr1, &arr2).unwrap();
        assert_eq!(result.get(&[0]).unwrap().to_string().unwrap(), "hello ");
        assert_eq!(result.get(&[1]).unwrap().to_string().unwrap(), "world!");
        
        // Test case operations
        let upper_result = upper(&arr1).unwrap();
        assert_eq!(upper_result.get(&[0]).unwrap().to_string().unwrap(), "HELLO");
        
        let lower_result = lower(&upper_result).unwrap();
        assert_eq!(lower_result.get(&[0]).unwrap().to_string().unwrap(), "hello");
    }
    
    #[test]
    fn test_string_comparisons() {
        let strings1 = vec!["apple", "banana"];
        let strings2 = vec!["apple", "cherry"];
        let arr1 = array_from_strings(&strings1, "U", None).unwrap();
        let arr2 = array_from_strings(&strings2, "U", None).unwrap();
        
        let eq_result = compare::equal(&arr1, &arr2).unwrap();
        assert_eq!(eq_result.to_vec(), vec![true, false]);
        
        let less_result = compare::less(&arr1, &arr2).unwrap();
        assert_eq!(less_result.to_vec(), vec![false, true]);
    }
    
    #[test]
    fn test_character_type_checks() {
        let strings = vec!["abc", "123", "ABC", "Hello World"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        let alpha_result = chartype::isalpha(&arr).unwrap();
        assert_eq!(alpha_result.to_vec(), vec![true, false, true, false]);
        
        let digit_result = chartype::isdigit(&arr).unwrap();
        assert_eq!(digit_result.to_vec(), vec![false, true, false, false]);
        
        let upper_result = chartype::isupper(&arr).unwrap();
        assert_eq!(upper_result.to_vec(), vec![false, false, true, false]);
    }
    
    #[test]
    fn test_string_manipulation() {
        let strings = vec!["  hello  ", "WORLD", "Test"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        // Test strip
        let strip_result = strip(&arr, None).unwrap();
        assert_eq!(strip_result.get(&[0]).unwrap().to_string().unwrap(), "hello");
        
        // Test replace
        let replace_result = replace(&arr, "Test", "Example", None).unwrap();
        assert_eq!(replace_result.get(&[2]).unwrap().to_string().unwrap(), "Example");
        
        // Test capitalize
        let cap_result = capitalize(&arr).unwrap();
        assert_eq!(cap_result.get(&[1]).unwrap().to_string().unwrap(), "World");
    }
    
    #[test]
    fn test_find_operations() {
        let strings = vec!["hello world", "python programming", "rust language"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        // Test find
        let find_result = find(&arr, "o", None, None).unwrap();
        assert_eq!(find_result.to_vec(), vec![4, 4, -1]); // First 'o' positions
        
        // Test count
        let count_result = count(&arr, "o", None, None).unwrap();
        assert_eq!(count_result.to_vec(), vec![2, 1, 0]); // Count of 'o' characters
        
        // Test startswith
        let starts_result = startswith(&arr, "hello", None, None).unwrap();
        assert_eq!(starts_result.to_vec(), vec![true, false, false]);
    }
}
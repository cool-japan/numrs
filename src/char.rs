//! String and character array operations (NumPy `char` module equivalent)
//!
//! This module provides character and string array operations similar to NumPy's `char` module.
//! It includes functions for string manipulation, comparison, and character type checking.

pub use crate::array_ops::string_ops::{
    // Core string operations
    StringArray, StringElement, array_from_strings,
    
    // String manipulation functions
    add, multiply, mod_format, center, ljust, rjust,
    strip, lstrip, rstrip, upper, lower, title, capitalize,
    replace, split, join, count, find, rfind,
    startswith, endswith,
};

pub use crate::array_ops::string_ops::chartype::{
    // Character type checking functions
    isalpha, isalnum, isdigit, islower, isupper, istitle, isspace,
};

pub use crate::array_ops::string_ops::compare::{
    // String comparison functions
    equal, not_equal, less, less_equal, greater, greater_equal,
};

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use regex::Regex;

/// Decode string array elements from specified encoding
///
/// Similar to NumPy's `np.char.decode()`
pub fn decode(arr: &StringArray, encoding: &str, errors: &str) -> Result<StringArray> {
    // For now, we only support UTF-8 encoding
    if encoding.to_lowercase() != "utf-8" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Encoding '{}' not supported. Only UTF-8 is currently supported.",
            encoding
        )));
    }
    
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        match s.to_string() {
            Ok(string) => result.push(StringElement::unicode(string)),
            Err(_) => {
                match errors {
                    "strict" => return Err(NumRs2Error::ValueError("Decoding error".to_string())),
                    "ignore" => result.push(StringElement::unicode("")),
                    "replace" => result.push(StringElement::unicode("�")),
                    _ => return Err(NumRs2Error::ValueError(format!("Invalid error handling: {}", errors))),
                }
            }
        }
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Encode string array elements to specified encoding
///
/// Similar to NumPy's `np.char.encode()`
pub fn encode(arr: &StringArray, encoding: &str, errors: &str) -> Result<StringArray> {
    // For now, we only support UTF-8 encoding
    if encoding.to_lowercase() != "utf-8" {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Encoding '{}' not supported. Only UTF-8 is currently supported.",
            encoding
        )));
    }
    
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        match s.to_string() {
            Ok(string) => {
                // For UTF-8, the string is already properly encoded
                result.push(StringElement::unicode(string));
            }
            Err(_) => {
                match errors {
                    "strict" => return Err(NumRs2Error::ValueError("Encoding error".to_string())),
                    "ignore" => result.push(StringElement::unicode("")),
                    "replace" => result.push(StringElement::unicode("�")),
                    _ => return Err(NumRs2Error::ValueError(format!("Invalid error handling: {}", errors))),
                }
            }
        }
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Expand tabs in string array elements
///
/// Similar to NumPy's `np.char.expandtabs()`
pub fn expandtabs(arr: &StringArray, tabsize: Option<usize>) -> Result<StringArray> {
    let tab_size = tabsize.unwrap_or(8);
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let expanded = string.chars()
            .fold((String::new(), 0), |(mut acc, pos), c| {
                if c == '\t' {
                    let spaces_needed = tab_size - (pos % tab_size);
                    acc.push_str(&" ".repeat(spaces_needed));
                    (acc, pos + spaces_needed)
                } else if c == '\n' {
                    acc.push(c);
                    (acc, 0)
                } else {
                    acc.push(c);
                    (acc, pos + 1)
                }
            })
            .0;
        result.push(StringElement::unicode(expanded));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Return string array element lengths
///
/// Similar to NumPy's `np.char.str_len()`
pub fn str_len(arr: &StringArray) -> Result<Array<i32>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let length = string.chars().count() as i32;
        result.push(length);
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Translate characters in string array elements
///
/// Similar to NumPy's `np.char.translate()`
pub fn translate(arr: &StringArray, table: &std::collections::HashMap<char, char>, delete: Option<&str>) -> Result<StringArray> {
    let delete_chars: std::collections::HashSet<char> = delete
        .map(|s| s.chars().collect())
        .unwrap_or_default();
    
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let translated: String = string
            .chars()
            .filter(|c| !delete_chars.contains(c))
            .map(|c| table.get(&c).copied().unwrap_or(c))
            .collect();
        result.push(StringElement::unicode(translated));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Pad string array elements with zeros
///
/// Similar to NumPy's `np.char.zfill()`
pub fn zfill(arr: &StringArray, width: usize) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let filled = if string.len() >= width {
            string
        } else {
            let padding = width - string.len();
            if string.starts_with('-') || string.starts_with('+') {
                format!("{}{}{}", &string[..1], "0".repeat(padding), &string[1..])
            } else {
                format!("{}{}", "0".repeat(padding), string)
            }
        };
        result.push(StringElement::unicode(filled));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Partition string array elements around separator
///
/// Similar to NumPy's `np.char.partition()`
pub fn partition(arr: &StringArray, sep: &str) -> Result<Vec<(String, String, String)>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        if let Some(pos) = string.find(sep) {
            let before = string[..pos].to_string();
            let separator = sep.to_string();
            let after = string[pos + sep.len()..].to_string();
            result.push((before, separator, after));
        } else {
            result.push((string, String::new(), String::new()));
        }
    }
    
    Ok(result)
}

/// Partition string array elements around separator from the right
///
/// Similar to NumPy's `np.char.rpartition()`
pub fn rpartition(arr: &StringArray, sep: &str) -> Result<Vec<(String, String, String)>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        if let Some(pos) = string.rfind(sep) {
            let before = string[..pos].to_string();
            let separator = sep.to_string();
            let after = string[pos + sep.len()..].to_string();
            result.push((before, separator, after));
        } else {
            result.push((String::new(), String::new(), string));
        }
    }
    
    Ok(result)
}

/// Split string array elements from the right
///
/// Similar to NumPy's `np.char.rsplit()`
pub fn rsplit(arr: &StringArray, sep: Option<&str>, maxsplit: Option<usize>) -> Result<Vec<Vec<String>>> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let parts: Vec<String> = match (sep, maxsplit) {
            (Some(delimiter), Some(max)) => {
                let parts: Vec<&str> = string.rsplitn(max + 1, delimiter).collect();
                parts.into_iter().rev().map(|s| s.to_string()).collect()
            }
            (Some(delimiter), None) => {
                string.rsplit(delimiter).map(|s| s.to_string()).collect()
            }
            (None, Some(max)) => {
                // For rsplit with maxsplit, we want at most `max` splits from the right
                // which results in at most `max+1` parts
                let all_words: Vec<&str> = string.split_whitespace().collect();
                if all_words.len() <= max + 1 {
                    // If we don't need to limit, return all words
                    all_words.into_iter().map(|s| s.to_string()).collect()
                } else {
                    // We need to limit splits. Keep the last `max` words separate,
                    // and join the rest as the first part
                    let mut result = Vec::new();
                    let join_count = all_words.len() - max;
                    
                    // Join the first (len - max) words into one part
                    let first_part = all_words[..join_count].join(" ");
                    result.push(first_part);
                    
                    // Add the last `max` words as separate parts
                    for word in &all_words[join_count..] {
                        result.push(word.to_string());
                    }
                    
                    result
                }
            }
            (None, None) => {
                let mut parts: Vec<&str> = string.split_whitespace().collect();
                parts.reverse();
                parts.into_iter().map(|s| s.to_string()).collect()
            }
        };
        result.push(parts);
    }
    
    Ok(result)
}

/// Split string array elements on line boundaries
///
/// Similar to NumPy's `np.char.splitlines()`
pub fn splitlines(arr: &StringArray, keepends: Option<bool>) -> Result<Vec<Vec<String>>> {
    let keep_endings = keepends.unwrap_or(false);
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let lines: Vec<String> = if keep_endings {
            // Split while keeping line endings
            let mut lines = Vec::new();
            let mut current_line = String::new();
            
            for c in string.chars() {
                current_line.push(c);
                if c == '\n' || c == '\r' {
                    lines.push(current_line.clone());
                    current_line.clear();
                }
            }
            
            if !current_line.is_empty() {
                lines.push(current_line);
            }
            
            lines
        } else {
            string.lines().map(|s| s.to_string()).collect()
        };
        
        result.push(lines);
    }
    
    Ok(result)
}

/// Swap case of string array elements
///
/// Similar to NumPy's `np.char.swapcase()`
pub fn swapcase(arr: &StringArray) -> Result<StringArray> {
    let mut result = Vec::with_capacity(arr.size());
    let arr_data = arr.to_vec();
    
    for s in arr_data.iter() {
        let string = s.to_string()?;
        let swapped: String = string.chars()
            .map(|c| {
                if c.is_uppercase() {
                    c.to_lowercase().collect::<String>()
                } else if c.is_lowercase() {
                    c.to_uppercase().collect::<String>()
                } else {
                    c.to_string()
                }
            })
            .collect();
        result.push(StringElement::unicode(swapped));
    }
    
    Ok(Array::from_vec(result).reshape(&arr.shape()))
}

/// Create array of strings with common prefix
///
/// Utility function for creating string arrays with patterns
pub fn array_with_prefix<S: AsRef<str>>(prefix: S, suffixes: &[S], dtype: &str) -> Result<StringArray> {
    let strings: Vec<String> = suffixes.iter()
        .map(|suffix| format!("{}{}", prefix.as_ref(), suffix.as_ref()))
        .collect();
    
    let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
    array_from_strings(&string_refs, dtype, None)
}

/// Create array of strings with common suffix
///
/// Utility function for creating string arrays with patterns
pub fn array_with_suffix<S: AsRef<str>>(prefixes: &[S], suffix: S, dtype: &str) -> Result<StringArray> {
    let strings: Vec<String> = prefixes.iter()
        .map(|prefix| format!("{}{}", prefix.as_ref(), suffix.as_ref()))
        .collect();
    
    let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
    array_from_strings(&string_refs, dtype, None)
}

/// Regular expression operations
pub mod regex_ops {
    use super::*;
    
    /// Find all matches of pattern in string array elements
    pub fn findall(arr: &StringArray, pattern: &str) -> Result<Vec<Vec<String>>> {
        let regex = Regex::new(pattern).map_err(|e| {
            NumRs2Error::ValueError(format!("Invalid regex pattern: {}", e))
        })?;
        
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let matches: Vec<String> = regex.find_iter(&string)
                .map(|m| m.as_str().to_string())
                .collect();
            result.push(matches);
        }
        
        Ok(result)
    }
    
    /// Replace pattern matches in string array elements
    pub fn sub(arr: &StringArray, pattern: &str, replacement: &str, count: Option<usize>) -> Result<StringArray> {
        let regex = Regex::new(pattern).map_err(|e| {
            NumRs2Error::ValueError(format!("Invalid regex pattern: {}", e))
        })?;
        
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let replaced = match count {
                Some(n) => {
                    let mut replaced = string.clone();
                    for _ in 0..n {
                        if let Some(captures) = regex.find(&replaced) {
                            replaced = format!("{}{}{}", 
                                &replaced[..captures.start()],
                                replacement,
                                &replaced[captures.end()..]
                            );
                        } else {
                            break;
                        }
                    }
                    replaced
                }
                None => regex.replace_all(&string, replacement).to_string(),
            };
            result.push(StringElement::unicode(replaced));
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Test if pattern matches string array elements
    pub fn match_pattern(arr: &StringArray, pattern: &str) -> Result<Array<bool>> {
        let regex = Regex::new(pattern).map_err(|e| {
            NumRs2Error::ValueError(format!("Invalid regex pattern: {}", e))
        })?;
        
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let matches = regex.is_match(&string);
            result.push(matches);
        }
        
        Ok(Array::from_vec(result).reshape(&arr.shape()))
    }
    
    /// Split string array elements using regex pattern
    pub fn split_regex(arr: &StringArray, pattern: &str, maxsplit: Option<usize>) -> Result<Vec<Vec<String>>> {
        let regex = Regex::new(pattern).map_err(|e| {
            NumRs2Error::ValueError(format!("Invalid regex pattern: {}", e))
        })?;
        
        let mut result = Vec::with_capacity(arr.size());
        let arr_data = arr.to_vec();
        
        for s in arr_data.iter() {
            let string = s.to_string()?;
            let parts: Vec<String> = match maxsplit {
                Some(max) => regex.splitn(&string, max + 1).map(|s| s.to_string()).collect(),
                None => regex.split(&string).map(|s| s.to_string()).collect(),
            };
            result.push(parts);
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_char_module_functions() {
        let strings = vec!["hello", "world", "test"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        // Test length function
        let lengths = str_len(&arr).unwrap();
        assert_eq!(lengths.to_vec(), vec![5, 5, 4]);
        
        // Test expandtabs
        let tab_strings = vec!["hello\tworld", "test\ttab"];
        let tab_arr = array_from_strings(&tab_strings, "U", None).unwrap();
        let expanded = expandtabs(&tab_arr, Some(4)).unwrap();
        assert_eq!(expanded.get(&[0]).unwrap().to_string().unwrap(), "hello   world");
        
        // Test zfill
        let numbers = vec!["42", "-17", "123"];
        let num_arr = array_from_strings(&numbers, "U", None).unwrap();
        let filled = zfill(&num_arr, 5).unwrap();
        assert_eq!(filled.get(&[0]).unwrap().to_string().unwrap(), "00042");
        assert_eq!(filled.get(&[1]).unwrap().to_string().unwrap(), "-0017");
    }
    
    #[test]
    fn test_translation() {
        let strings = vec!["hello", "world"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        let mut table = HashMap::new();
        table.insert('l', 'L');
        table.insert('o', 'O');
        
        let translated = translate(&arr, &table, None).unwrap();
        assert_eq!(translated.get(&[0]).unwrap().to_string().unwrap(), "heLLO");
        assert_eq!(translated.get(&[1]).unwrap().to_string().unwrap(), "wOrLd");
    }
    
    #[test]
    fn test_partition_operations() {
        let strings = vec!["hello-world-test", "one-two-three"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        let partitions = partition(&arr, "-").unwrap();
        assert_eq!(partitions[0], ("hello".to_string(), "-".to_string(), "world-test".to_string()));
        
        let rpartitions = rpartition(&arr, "-").unwrap();
        assert_eq!(rpartitions[0], ("hello-world".to_string(), "-".to_string(), "test".to_string()));
    }
    
    #[test]
    fn test_regex_operations() {
        let strings = vec!["hello123world", "test456example"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        // Test findall
        let matches = regex_ops::findall(&arr, r"\d+").unwrap();
        assert_eq!(matches[0], vec!["123"]);
        assert_eq!(matches[1], vec!["456"]);
        
        // Test substitution
        let substituted = regex_ops::sub(&arr, r"\d+", "XXX", None).unwrap();
        assert_eq!(substituted.get(&[0]).unwrap().to_string().unwrap(), "helloXXXworld");
        
        // Test pattern matching
        let matches_bool = regex_ops::match_pattern(&arr, r".*\d+.*").unwrap();
        assert_eq!(matches_bool.to_vec(), vec![true, true]);
    }
    
    #[test]
    fn test_utility_functions() {
        // Test array_with_prefix
        let suffixes = vec!["1", "2", "3"];
        let prefixed = array_with_prefix("test_", &suffixes, "U").unwrap();
        assert_eq!(prefixed.get(&[0]).unwrap().to_string().unwrap(), "test_1");
        
        // Test array_with_suffix
        let prefixes = vec!["file", "data", "image"];
        let suffixed = array_with_suffix(&prefixes, ".txt", "U").unwrap();
        assert_eq!(suffixed.get(&[0]).unwrap().to_string().unwrap(), "file.txt");
    }
    
    #[test]
    fn test_swapcase() {
        let strings = vec!["Hello", "WORLD", "tEsT"];
        let arr = array_from_strings(&strings, "U", None).unwrap();
        
        let swapped = swapcase(&arr).unwrap();
        assert_eq!(swapped.get(&[0]).unwrap().to_string().unwrap(), "hELLO");
        assert_eq!(swapped.get(&[1]).unwrap().to_string().unwrap(), "world");
        assert_eq!(swapped.get(&[2]).unwrap().to_string().unwrap(), "TeSt");
    }
}
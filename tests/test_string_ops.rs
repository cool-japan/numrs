//! Comprehensive tests for string array operations
//!
//! Tests for NumPy char module equivalent functionality in NumRS2

use numrs2::char::{self, *};
use numrs2::prelude::*;
use numrs2::prelude::{chartype, compare};
use std::collections::HashMap;

#[test]
fn test_string_element_basic_operations() {
    // Test Unicode string element
    let unicode_elem = StringElement::unicode("hello world");
    assert_eq!(unicode_elem.to_string().unwrap(), "hello world");
    assert_eq!(unicode_elem.len().unwrap(), 11);
    assert!(!unicode_elem.is_empty().unwrap());
    assert_eq!(unicode_elem.capacity(), 11);

    // Test fixed-length string element
    let fixed_elem = StringElement::fixed("test", 10);
    assert_eq!(fixed_elem.to_string().unwrap(), "test");
    assert_eq!(fixed_elem.len().unwrap(), 4);
    assert_eq!(fixed_elem.capacity(), 10);

    // Test empty string
    let empty_elem = StringElement::unicode("");
    assert!(empty_elem.is_empty().unwrap());
    assert_eq!(empty_elem.len().unwrap(), 0);
}

#[test]
fn test_array_creation_functions() {
    // Test Unicode string array creation
    let strings = vec!["hello", "world", "test", "array"];
    let unicode_arr = array_from_strings(&strings, "U", Some(&[4])).unwrap();
    assert_eq!(unicode_arr.shape(), vec![4]);
    assert_eq!(unicode_arr.get(&[0]).unwrap().to_string().unwrap(), "hello");
    assert_eq!(unicode_arr.get(&[3]).unwrap().to_string().unwrap(), "array");

    // Test fixed-length string array creation
    let fixed_arr = array_from_strings(&strings, "S10", Some(&[2, 2])).unwrap();
    assert_eq!(fixed_arr.shape(), vec![2, 2]);
    assert_eq!(
        fixed_arr.get(&[0, 0]).unwrap().to_string().unwrap(),
        "hello"
    );
    assert_eq!(
        fixed_arr.get(&[1, 1]).unwrap().to_string().unwrap(),
        "array"
    );

    // Test invalid dtype
    let result = array_from_strings(&strings, "Invalid", None);
    assert!(result.is_err());

    // Test shape mismatch
    let result = array_from_strings(&strings, "U", Some(&[2]));
    assert!(result.is_err());
}

#[test]
fn test_string_arithmetic_operations() {
    let strings1 = vec!["hello", "good", "test"];
    let strings2 = vec![" world", " morning", " case"];
    let arr1 = array_from_strings(&strings1, "U", None).unwrap();
    let arr2 = array_from_strings(&strings2, "U", None).unwrap();

    // Test string addition
    let result = char::add(&arr1, &arr2).unwrap();
    assert_eq!(
        result.get(&[0]).unwrap().to_string().unwrap(),
        "hello world"
    );
    assert_eq!(
        result.get(&[1]).unwrap().to_string().unwrap(),
        "good morning"
    );
    assert_eq!(result.get(&[2]).unwrap().to_string().unwrap(), "test case");

    // Test string multiplication
    let times = Array::from_vec(vec![2, 3, 0]).reshape(&[3]);
    let result = char::multiply(&arr1, &times).unwrap();
    assert_eq!(result.get(&[0]).unwrap().to_string().unwrap(), "hellohello");
    assert_eq!(
        result.get(&[1]).unwrap().to_string().unwrap(),
        "goodgoodgood"
    );
    assert_eq!(result.get(&[2]).unwrap().to_string().unwrap(), "");

    // Test shape mismatch
    let arr3 = array_from_strings(&["extra"], "U", None).unwrap();
    let result = char::add(&arr1, &arr3);
    assert!(result.is_err());
}

#[test]
fn test_string_formatting_operations() {
    let format_strings = vec!["Value: %f", "Number: %d", "Float: %g"];
    let values = vec![std::f64::consts::PI, 42.0, std::f64::consts::E];
    let arr = array_from_strings(&format_strings, "U", None).unwrap();
    let val_arr = Array::from_vec(values).reshape(&[3]);

    let result = mod_format(&arr, &val_arr).unwrap();
    assert_eq!(
        result.get(&[0]).unwrap().to_string().unwrap(),
        "Value: 3.14159"
    );
    assert_eq!(result.get(&[1]).unwrap().to_string().unwrap(), "Number: 42");
    assert_eq!(
        result.get(&[2]).unwrap().to_string().unwrap(),
        "Float: 2.71828"
    );
}

#[test]
fn test_string_justification_operations() {
    let strings = vec!["hello", "test", "a"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test center justification
    let centered = center(&arr, 10, Some('*')).unwrap();
    assert_eq!(
        centered.get(&[0]).unwrap().to_string().unwrap(),
        "**hello***"
    );
    assert_eq!(
        centered.get(&[1]).unwrap().to_string().unwrap(),
        "***test***"
    );
    assert_eq!(
        centered.get(&[2]).unwrap().to_string().unwrap(),
        "****a*****"
    );

    // Test left justification
    let left_just = ljust(&arr, 8, Some('-')).unwrap();
    assert_eq!(
        left_just.get(&[0]).unwrap().to_string().unwrap(),
        "hello---"
    );
    assert_eq!(
        left_just.get(&[1]).unwrap().to_string().unwrap(),
        "test----"
    );
    assert_eq!(
        left_just.get(&[2]).unwrap().to_string().unwrap(),
        "a-------"
    );

    // Test right justification
    let right_just = rjust(&arr, 8, Some('=')).unwrap();
    assert_eq!(
        right_just.get(&[0]).unwrap().to_string().unwrap(),
        "===hello"
    );
    assert_eq!(
        right_just.get(&[1]).unwrap().to_string().unwrap(),
        "====test"
    );
    assert_eq!(
        right_just.get(&[2]).unwrap().to_string().unwrap(),
        "=======a"
    );
}

#[test]
fn test_string_stripping_operations() {
    let strings = vec!["  hello  ", "***world***", "\t\ntest\n\t"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test general strip
    let stripped = strip(&arr, None).unwrap();
    assert_eq!(stripped.get(&[0]).unwrap().to_string().unwrap(), "hello");
    assert_eq!(stripped.get(&[2]).unwrap().to_string().unwrap(), "test");

    // Test strip with specific characters
    let char_stripped = strip(&arr, Some("* \t\n")).unwrap();
    assert_eq!(
        char_stripped.get(&[0]).unwrap().to_string().unwrap(),
        "hello"
    );
    assert_eq!(
        char_stripped.get(&[1]).unwrap().to_string().unwrap(),
        "world"
    );
    assert_eq!(
        char_stripped.get(&[2]).unwrap().to_string().unwrap(),
        "test"
    );

    // Test left strip
    let left_stripped = lstrip(&arr, None).unwrap();
    assert_eq!(
        left_stripped.get(&[0]).unwrap().to_string().unwrap(),
        "hello  "
    );

    // Test right strip
    let right_stripped = rstrip(&arr, None).unwrap();
    assert_eq!(
        right_stripped.get(&[0]).unwrap().to_string().unwrap(),
        "  hello"
    );
}

#[test]
fn test_string_case_operations() {
    let strings = vec!["hello world", "TEST CASE", "MiXeD cAsE", "already_lower"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test uppercase
    let upper_result = upper(&arr).unwrap();
    assert_eq!(
        upper_result.get(&[0]).unwrap().to_string().unwrap(),
        "HELLO WORLD"
    );
    assert_eq!(
        upper_result.get(&[3]).unwrap().to_string().unwrap(),
        "ALREADY_LOWER"
    );

    // Test lowercase
    let lower_result = lower(&arr).unwrap();
    assert_eq!(
        lower_result.get(&[1]).unwrap().to_string().unwrap(),
        "test case"
    );
    assert_eq!(
        lower_result.get(&[2]).unwrap().to_string().unwrap(),
        "mixed case"
    );

    // Test title case
    let title_result = title(&arr).unwrap();
    assert_eq!(
        title_result.get(&[0]).unwrap().to_string().unwrap(),
        "Hello World"
    );
    assert_eq!(
        title_result.get(&[1]).unwrap().to_string().unwrap(),
        "Test Case"
    );

    // Test capitalize
    let cap_result = capitalize(&arr).unwrap();
    assert_eq!(
        cap_result.get(&[0]).unwrap().to_string().unwrap(),
        "Hello world"
    );
    assert_eq!(
        cap_result.get(&[2]).unwrap().to_string().unwrap(),
        "Mixed case"
    );

    // Test swapcase
    let swap_result = char::swapcase(&arr).unwrap();
    assert_eq!(
        swap_result.get(&[0]).unwrap().to_string().unwrap(),
        "HELLO WORLD"
    );
    assert_eq!(
        swap_result.get(&[1]).unwrap().to_string().unwrap(),
        "test case"
    );
}

#[test]
fn test_string_replacement_operations() {
    let strings = vec!["hello world hello", "test case test", "no match here"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test replace all occurrences
    let replaced = replace(&arr, "test", "example", None).unwrap();
    assert_eq!(
        replaced.get(&[1]).unwrap().to_string().unwrap(),
        "example case example"
    );
    assert_eq!(
        replaced.get(&[2]).unwrap().to_string().unwrap(),
        "no match here"
    );

    // Test replace with count limit
    let limited_replace = replace(&arr, "hello", "hi", Some(1)).unwrap();
    assert_eq!(
        limited_replace.get(&[0]).unwrap().to_string().unwrap(),
        "hi world hello"
    );
}

#[test]
fn test_string_split_join_operations() {
    let strings = vec!["apple,banana,cherry", "one two three", "single"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test split with delimiter
    let split_result = char::split(&arr, Some(","), None).unwrap();
    assert_eq!(split_result[0], vec!["apple", "banana", "cherry"]);
    assert_eq!(split_result[2], vec!["single"]);

    // Test split on whitespace
    let whitespace_split = char::split(&arr, None, None).unwrap();
    assert_eq!(whitespace_split[1], vec!["one", "two", "three"]);

    // Test split with maxsplit
    let limited_split = char::split(&arr, Some(","), Some(1)).unwrap();
    assert_eq!(limited_split[0], vec!["apple", "banana,cherry"]);

    // Test join
    let parts = vec![
        vec!["hello".to_string(), "world".to_string()],
        vec!["test".to_string(), "case".to_string()],
    ];
    let joined = char::join("-", &parts).unwrap();
    assert_eq!(
        joined.get(&[0]).unwrap().to_string().unwrap(),
        "hello-world"
    );
    assert_eq!(joined.get(&[1]).unwrap().to_string().unwrap(), "test-case");
}

#[test]
fn test_string_search_operations() {
    let strings = vec!["hello world hello", "testing test", "no match"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test count
    let count_result = count(&arr, "test", None, None).unwrap();
    assert_eq!(count_result.to_vec(), vec![0, 2, 0]);

    let hello_count = count(&arr, "hello", None, None).unwrap();
    assert_eq!(hello_count.to_vec(), vec![2, 0, 0]);

    // Test find
    let find_result = find(&arr, "world", None, None).unwrap();
    assert_eq!(find_result.to_vec(), vec![6, -1, -1]);

    let test_find = find(&arr, "test", None, None).unwrap();
    assert_eq!(test_find.to_vec(), vec![-1, 0, -1]);

    // Test rfind
    let rfind_result = rfind(&arr, "hello", None, None).unwrap();
    assert_eq!(rfind_result.to_vec(), vec![12, -1, -1]);

    // Test find with range
    let range_find = find(&arr, "hello", Some(10), None).unwrap();
    assert_eq!(range_find.to_vec(), vec![12, -1, -1]);
}

#[test]
fn test_string_prefix_suffix_operations() {
    let strings = vec!["hello world", "test case", "hello test"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test startswith
    let starts_hello = startswith(&arr, "hello", None, None).unwrap();
    assert_eq!(starts_hello.to_vec(), vec![true, false, true]);

    let starts_test = startswith(&arr, "test", None, None).unwrap();
    assert_eq!(starts_test.to_vec(), vec![false, true, false]);

    // Test endswith
    let ends_world = endswith(&arr, "world", None, None).unwrap();
    assert_eq!(ends_world.to_vec(), vec![true, false, false]);

    let ends_test = endswith(&arr, "test", None, None).unwrap();
    assert_eq!(ends_test.to_vec(), vec![false, false, true]);

    // Test with start/end positions
    let range_starts = startswith(&arr, "world", Some(6), None).unwrap();
    assert_eq!(range_starts.to_vec(), vec![true, false, false]);
}

#[test]
fn test_character_type_checks() {
    let strings = vec!["abc", "123", "ABC", "aBc", "Hello World", "   ", "abc123"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test isalpha
    let alpha_result = chartype::isalpha(&arr).unwrap();
    assert_eq!(
        alpha_result.to_vec(),
        vec![true, false, true, true, false, false, false]
    );

    // Test isdigit
    let digit_result = chartype::isdigit(&arr).unwrap();
    assert_eq!(
        digit_result.to_vec(),
        vec![false, true, false, false, false, false, false]
    );

    // Test isalnum
    let alnum_result = chartype::isalnum(&arr).unwrap();
    assert_eq!(
        alnum_result.to_vec(),
        vec![true, true, true, true, false, false, true]
    );

    // Test islower
    let lower_result = chartype::islower(&arr).unwrap();
    assert_eq!(
        lower_result.to_vec(),
        vec![true, false, false, false, false, false, true]
    );

    // Test isupper
    let upper_result = chartype::isupper(&arr).unwrap();
    assert_eq!(
        upper_result.to_vec(),
        vec![false, false, true, false, false, false, false]
    );

    // Test isspace
    let space_result = chartype::isspace(&arr).unwrap();
    assert_eq!(
        space_result.to_vec(),
        vec![false, false, false, false, false, true, false]
    );
}

#[test]
fn test_title_case_checks() {
    let strings = vec!["Title Case", "title case", "TITLE CASE", "Title case", ""];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    let title_result = chartype::istitle(&arr).unwrap();
    assert_eq!(
        title_result.to_vec(),
        vec![true, false, false, false, false]
    );
}

#[test]
fn test_string_comparisons() {
    let strings1 = vec!["apple", "banana", "cherry"];
    let strings2 = vec!["apple", "blueberry", "cherry"];
    let arr1 = array_from_strings(&strings1, "U", None).unwrap();
    let arr2 = array_from_strings(&strings2, "U", None).unwrap();

    // Test equal
    let eq_result = compare::equal(&arr1, &arr2).unwrap();
    assert_eq!(eq_result.to_vec(), vec![true, false, true]);

    // Test not equal
    let neq_result = compare::not_equal(&arr1, &arr2).unwrap();
    assert_eq!(neq_result.to_vec(), vec![false, true, false]);

    // Test less than
    let less_result = compare::less(&arr1, &arr2).unwrap();
    assert_eq!(less_result.to_vec(), vec![false, true, false]);

    // Test less than or equal
    let leq_result = compare::less_equal(&arr1, &arr2).unwrap();
    assert_eq!(leq_result.to_vec(), vec![true, true, true]);

    // Test greater than
    let gt_result = compare::greater(&arr1, &arr2).unwrap();
    assert_eq!(gt_result.to_vec(), vec![false, false, false]);

    // Test greater than or equal
    let geq_result = compare::greater_equal(&arr1, &arr2).unwrap();
    assert_eq!(geq_result.to_vec(), vec![true, false, true]);
}

#[test]
fn test_advanced_char_functions() {
    let strings = vec!["hello\tworld", "test", "-42", "+123"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test str_len
    let lengths = str_len(&arr).unwrap();
    assert_eq!(lengths.to_vec(), vec![11, 4, 3, 4]);

    // Test expandtabs
    let expanded = expandtabs(&arr, Some(4)).unwrap();
    assert_eq!(
        expanded.get(&[0]).unwrap().to_string().unwrap(),
        "hello   world"
    );

    // Test zfill
    let filled = zfill(&arr, 8).unwrap();
    assert_eq!(filled.get(&[1]).unwrap().to_string().unwrap(), "0000test");
    assert_eq!(filled.get(&[2]).unwrap().to_string().unwrap(), "-0000042");
    assert_eq!(filled.get(&[3]).unwrap().to_string().unwrap(), "+0000123");
}

#[test]
fn test_translation() {
    let strings = vec!["hello", "world", "test"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    let mut table = HashMap::new();
    table.insert('l', 'L');
    table.insert('o', 'O');
    table.insert('e', 'E');

    let translated = translate(&arr, &table, Some("t")).unwrap();
    assert_eq!(translated.get(&[0]).unwrap().to_string().unwrap(), "hELLO");
    assert_eq!(translated.get(&[1]).unwrap().to_string().unwrap(), "wOrLd");
    assert_eq!(translated.get(&[2]).unwrap().to_string().unwrap(), "Es"); // 't' deleted, 'e' -> 'E'
}

#[test]
fn test_partition_operations() {
    let strings = vec!["hello-world-test", "one-two-three", "no-separator"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test partition
    let partitions = char::partition(&arr, "-").unwrap();
    assert_eq!(
        partitions[0],
        (
            "hello".to_string(),
            "-".to_string(),
            "world-test".to_string()
        )
    );
    assert_eq!(
        partitions[1],
        ("one".to_string(), "-".to_string(), "two-three".to_string())
    );

    // Test rpartition
    let rpartitions = char::rpartition(&arr, "-").unwrap();
    assert_eq!(
        rpartitions[0],
        (
            "hello-world".to_string(),
            "-".to_string(),
            "test".to_string()
        )
    );
    assert_eq!(
        rpartitions[1],
        ("one-two".to_string(), "-".to_string(), "three".to_string())
    );

    // Test with non-existent separator
    let no_sep_partition = char::partition(&arr, "@").unwrap();
    assert_eq!(
        no_sep_partition[0],
        (
            "hello-world-test".to_string(),
            "".to_string(),
            "".to_string()
        )
    );
}

#[test]
fn test_split_variations() {
    let strings = vec!["one two three four", "split\ton\ttabs"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test rsplit
    let rsplit_result = char::rsplit(&arr, None, Some(2)).unwrap();
    assert_eq!(rsplit_result[0], vec!["one two", "three", "four"]);

    // Test splitlines
    let multiline = vec!["line1\nline2\nline3", "single line"];
    let multiline_arr = array_from_strings(&multiline, "U", None).unwrap();

    let lines = char::splitlines(&multiline_arr, Some(false)).unwrap();
    assert_eq!(lines[0], vec!["line1", "line2", "line3"]);
    assert_eq!(lines[1], vec!["single line"]);

    let lines_with_ends = char::splitlines(&multiline_arr, Some(true)).unwrap();
    assert_eq!(lines_with_ends[0], vec!["line1\n", "line2\n", "line3"]);
}

#[test]
fn test_regex_operations() {
    let strings = vec!["hello123world", "test456example", "no_digits_here"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test findall
    let matches = regex_ops::findall(&arr, r"\d+").unwrap();
    assert_eq!(matches[0], vec!["123"]);
    assert_eq!(matches[1], vec!["456"]);
    assert_eq!(matches[2], Vec::<String>::new());

    // Test substitution
    let substituted = regex_ops::sub(&arr, r"\d+", "XXX", None).unwrap();
    assert_eq!(
        substituted.get(&[0]).unwrap().to_string().unwrap(),
        "helloXXXworld"
    );
    assert_eq!(
        substituted.get(&[1]).unwrap().to_string().unwrap(),
        "testXXXexample"
    );
    assert_eq!(
        substituted.get(&[2]).unwrap().to_string().unwrap(),
        "no_digits_here"
    );

    // Test pattern matching
    let matches_bool = regex_ops::match_pattern(&arr, r".*\d+.*").unwrap();
    assert_eq!(matches_bool.to_vec(), vec![true, true, false]);

    // Test regex split
    let split_result = regex_ops::split_regex(&arr, r"\d+", None).unwrap();
    assert_eq!(split_result[0], vec!["hello", "world"]);
    assert_eq!(split_result[1], vec!["test", "example"]);
    assert_eq!(split_result[2], vec!["no_digits_here"]);
}

#[test]
fn test_utility_functions() {
    // Test array_with_prefix
    let suffixes = vec!["file1", "file2", "file3"];
    let prefixed = array_with_prefix("data_", &suffixes, "U").unwrap();
    assert_eq!(
        prefixed.get(&[0]).unwrap().to_string().unwrap(),
        "data_file1"
    );
    assert_eq!(
        prefixed.get(&[2]).unwrap().to_string().unwrap(),
        "data_file3"
    );

    // Test array_with_suffix
    let prefixes = vec!["image", "document", "archive"];
    let suffixed = array_with_suffix(&prefixes, ".txt", "U").unwrap();
    assert_eq!(
        suffixed.get(&[0]).unwrap().to_string().unwrap(),
        "image.txt"
    );
    assert_eq!(
        suffixed.get(&[2]).unwrap().to_string().unwrap(),
        "archive.txt"
    );
}

#[test]
fn test_encoding_operations() {
    let strings = vec!["hello", "world", "test"];
    let arr = array_from_strings(&strings, "U", None).unwrap();

    // Test encode (basic UTF-8 functionality)
    let encoded = encode(&arr, "utf-8", "strict").unwrap();
    assert_eq!(encoded.get(&[0]).unwrap().to_string().unwrap(), "hello");

    // Test decode (basic UTF-8 functionality)
    let decoded = decode(&arr, "utf-8", "strict").unwrap();
    assert_eq!(decoded.get(&[1]).unwrap().to_string().unwrap(), "world");

    // Test unsupported encoding
    let result = encode(&arr, "latin-1", "strict");
    assert!(result.is_err());
}

#[test]
fn test_edge_cases() {
    // Test empty strings
    let empty_strings = vec!["", "", ""];
    let empty_arr = array_from_strings(&empty_strings, "U", None).unwrap();

    let upper_empty = upper(&empty_arr).unwrap();
    assert_eq!(upper_empty.get(&[0]).unwrap().to_string().unwrap(), "");

    let lengths = str_len(&empty_arr).unwrap();
    assert_eq!(lengths.to_vec(), vec![0, 0, 0]);

    // Test single character strings
    let single_chars = vec!["a", "B", "1"];
    let single_arr = array_from_strings(&single_chars, "U", None).unwrap();

    let upper_single = upper(&single_arr).unwrap();
    assert_eq!(
        upper_single
            .to_vec()
            .iter()
            .map(|s| s.to_string().unwrap())
            .collect::<Vec<_>>(),
        vec!["A", "B", "1"]
    );

    // Test very long strings
    let long_string = "a".repeat(1000);
    let long_arr = array_from_strings(&[&long_string], "U", None).unwrap();
    assert_eq!(long_arr.get(&[0]).unwrap().len().unwrap(), 1000);
}

#[test]
fn test_unicode_handling() {
    // Test Unicode characters
    let unicode_strings = vec!["café", "naïve", "résumé"];
    let unicode_arr = array_from_strings(&unicode_strings, "U", None).unwrap();

    let upper_unicode = upper(&unicode_arr).unwrap();
    assert_eq!(
        upper_unicode.get(&[0]).unwrap().to_string().unwrap(),
        "CAFÉ"
    );

    let lengths = str_len(&unicode_arr).unwrap();
    assert_eq!(lengths.to_vec(), vec![4, 5, 6]);

    // Test emojis and special characters
    let emoji_strings = vec!["hello 👋", "test 🧪", "data 📊"];
    let emoji_arr = array_from_strings(&emoji_strings, "U", None).unwrap();

    let emoji_lengths = str_len(&emoji_arr).unwrap();
    assert_eq!(emoji_lengths.get(&[0]).unwrap(), 7); // 'hello ' (6 chars) + emoji (1 char)
}

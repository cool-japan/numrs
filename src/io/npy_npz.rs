use super::SerializeFormat;
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use oxiarc_archive::zip::{ZipCompressionLevel, ZipReader, ZipWriter};
use std::io::{Read, Seek, Write};

// NPY magic numbers and constants
const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";
const NPY_MAJOR_VERSION: u8 = 1;
const NPY_MINOR_VERSION: u8 = 0;

// Helper function to get NumPy dtype string for Rust type
fn get_numpy_dtype<T>() -> &'static str {
    // This is a simplified mapping for common types
    // NumPy has more detailed type descriptors
    if std::any::type_name::<T>() == "f32" {
        "<f4"
    } else if std::any::type_name::<T>() == "f64" {
        "<f8"
    } else if std::any::type_name::<T>() == "i8" {
        "<i1"
    } else if std::any::type_name::<T>() == "i16" {
        "<i2"
    } else if std::any::type_name::<T>() == "i32" {
        "<i4"
    } else if std::any::type_name::<T>() == "i64" {
        "<i8"
    } else if std::any::type_name::<T>() == "u8" {
        "<u1"
    } else if std::any::type_name::<T>() == "u16" {
        "<u2"
    } else if std::any::type_name::<T>() == "u32" {
        "<u4"
    } else if std::any::type_name::<T>() == "u64" {
        "<u8"
    } else if std::any::type_name::<T>() == "bool" {
        "|b1"
    } else {
        "unknown"
    }
}

// Construct NPY header for the specified array
fn construct_npy_header<T>(shape: &[usize]) -> Result<Vec<u8>> {
    let dtype = get_numpy_dtype::<T>();
    if dtype == "unknown" {
        return Err(NumRs2Error::SerializationError(format!(
            "Unsupported type for NPY format: {}",
            std::any::type_name::<T>()
        )));
    }

    // Construct the Python dictionary that will go in the header
    let mut dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': (", dtype);

    // Add shape information
    for (i, &dim) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        dict.push_str(&dim.to_string());

        // If it's a 1D array, add a trailing comma to make it a tuple in Python
        if shape.len() == 1 && i == shape.len() - 1 {
            dict.push(',');
        }
    }
    dict.push_str("), }");

    // Pad with spaces to make header length + magic string + version a multiple of 16
    // 10 is for magic string (6) + version (2) + header length (2)
    let header_len = dict.len();
    let padding_needed = 16 - ((header_len + 10) % 16);
    dict.push_str(&" ".repeat(padding_needed));

    // Calculate header length (Python dict length)
    let header_len_u16 = dict.len() as u16;

    // Create the full header buffer
    let mut header = Vec::with_capacity(10 + dict.len());

    // Add magic string
    header.extend_from_slice(NPY_MAGIC_STRING);

    // Add version info
    header.push(NPY_MAJOR_VERSION);
    header.push(NPY_MINOR_VERSION);

    // Add header length (little endian)
    let mut header_len_bytes = [0; 2];
    LittleEndian::write_u16(&mut header_len_bytes, header_len_u16);
    header.extend_from_slice(&header_len_bytes);

    // Add the Python dictionary with array metadata
    header.extend_from_slice(dict.as_bytes());

    Ok(header)
}

/// Appends the little-endian NPY payload bytes for `data: &[T]` to `out`,
/// dispatching on `type_name` (the caller's `std::any::type_name::<T>()`).
///
/// # Soundness
///
/// `std::mem::transmute_copy` performs **no size check**: if the
/// destination type is larger than the source, it silently reads past the
/// end of the source value, which is undefined behavior. This module
/// dispatches on `std::any::type_name::<T>()` rather than
/// `std::any::TypeId::of::<T>()` (the pattern used at
/// `src/types/structured.rs`) because the public entry points that call
/// this helper (`serialize_to_file`, `save_npz_arrays`) are themselves
/// called generically from `io::mod` and `io::text` with only a `T: Clone`
/// bound — `TypeId::of` requires `T: 'static`, and adding that bound here
/// would require adding it all the way up through those call sites too,
/// which live outside this module.
///
/// To stay sound despite the weaker `type_name` guard, every arm below
/// re-verifies `size_of::<T>() == size_of::<Target>()` immediately before
/// its `transmute_copy`, turning any mismatch into a clean `Err` instead of
/// an out-of-bounds read. `bool` gets extra care beyond the size check:
/// transmuting *into* `bool` materializes an invalid value (instant
/// undefined behavior, even if never branched on) if the source byte is
/// anything but 0 or 1, so the `bool` arm never constructs a `bool` via
/// transmute — it reads `T`'s raw byte directly instead.
fn encode_npy_elements<T: Clone>(
    format_label: &str,
    type_name: &str,
    data: &[T],
    out: &mut Vec<u8>,
) -> Result<()> {
    macro_rules! encode_numeric {
        ($target:ty) => {{
            if std::mem::size_of::<T>() != std::mem::size_of::<$target>() {
                return Err(NumRs2Error::SerializationError(format!(
                    "Internal size mismatch encoding '{}' as {}",
                    type_name,
                    stringify!($target)
                )));
            }
            for val in data {
                // SAFETY: size checked immediately above.
                let bytes = unsafe { std::mem::transmute_copy::<T, $target>(val) }.to_le_bytes();
                out.extend_from_slice(&bytes);
            }
        }};
    }

    match type_name {
        "f32" => encode_numeric!(f32),
        "f64" => encode_numeric!(f64),
        "i8" => encode_numeric!(i8),
        "i16" => encode_numeric!(i16),
        "i32" => encode_numeric!(i32),
        "i64" => encode_numeric!(i64),
        "u8" => encode_numeric!(u8),
        "u16" => encode_numeric!(u16),
        "u32" => encode_numeric!(u32),
        "u64" => encode_numeric!(u64),
        "bool" => {
            if std::mem::size_of::<T>() != std::mem::size_of::<u8>() {
                return Err(NumRs2Error::SerializationError(format!(
                    "Internal size mismatch encoding '{}' as bool",
                    type_name
                )));
            }
            for val in data {
                // SAFETY: size checked above. Read the raw byte instead of
                // transmuting to `bool`: an arbitrary same-size `T` could
                // carry a bit pattern that is not a valid `bool`, and
                // materializing an invalid `bool` is immediate UB even if
                // it is never branched on.
                let raw_byte: u8 = unsafe { *(val as *const T as *const u8) };
                out.push(if raw_byte != 0 { 1u8 } else { 0u8 });
            }
        }
        other => {
            return Err(NumRs2Error::SerializationError(format!(
                "{} format does not support type: {}",
                format_label, other
            )));
        }
    }
    Ok(())
}

/// Decodes `raw_data` (little-endian NPY payload bytes for the primitive
/// type identified by `type_name`) into `Vec<T>`. See
/// [`encode_npy_elements`] for why dispatch is by `type_name` rather than
/// `TypeId`, and why the size check below is what keeps every
/// `transmute_copy` sound.
fn decode_npy_elements<T: Clone>(type_name: &str, raw_data: &[u8]) -> Result<Vec<T>> {
    macro_rules! decode_numeric {
        ($target:ty, $chunk_size:literal) => {{
            if std::mem::size_of::<T>() != std::mem::size_of::<$target>() {
                return Err(NumRs2Error::DeserializationError(format!(
                    "Internal size mismatch decoding '{}' from {}",
                    type_name,
                    stringify!($target)
                )));
            }
            let mut typed_data = Vec::with_capacity(raw_data.len() / $chunk_size);
            for chunk in raw_data.chunks_exact($chunk_size) {
                let mut buf = [0u8; $chunk_size];
                buf.copy_from_slice(chunk);
                let value = <$target>::from_le_bytes(buf);
                // SAFETY: size checked above.
                typed_data.push(unsafe { std::mem::transmute_copy::<$target, T>(&value) });
            }
            Ok(typed_data)
        }};
    }

    match type_name {
        "f32" => decode_numeric!(f32, 4),
        "f64" => decode_numeric!(f64, 8),
        "i8" => decode_numeric!(i8, 1),
        "i16" => decode_numeric!(i16, 2),
        "i32" => decode_numeric!(i32, 4),
        "i64" => decode_numeric!(i64, 8),
        "u8" => decode_numeric!(u8, 1),
        "u16" => decode_numeric!(u16, 2),
        "u32" => decode_numeric!(u32, 4),
        "u64" => decode_numeric!(u64, 8),
        "bool" => {
            if std::mem::size_of::<T>() != std::mem::size_of::<u8>() {
                return Err(NumRs2Error::DeserializationError(format!(
                    "Internal size mismatch decoding '{}' from bool",
                    type_name
                )));
            }
            let mut typed_data = Vec::with_capacity(raw_data.len());
            for chunk in raw_data.chunks_exact(1) {
                let value: u8 = if chunk[0] != 0 { 1 } else { 0 };
                // SAFETY: size checked above. Transmute from a plain,
                // always-valid `u8` rather than an intermediate `bool`, so
                // no `bool`-typed value with a possibly-invalid bit
                // pattern is ever materialized.
                typed_data.push(unsafe { std::mem::transmute_copy::<u8, T>(&value) });
            }
            Ok(typed_data)
        }
        other => {
            // Unreachable in normal use: every caller of this function
            // already matched `type_name` against this exact literal set
            // (see the `element_size`/`expected_dtype` computation in
            // `read_npy_generic`) before calling here, and `type_name` is
            // never recomputed in between -- so any value that reaches
            // this arm was already proven to be one of the arms above.
            // Kept as a returned `Err` rather than `unreachable!()` so
            // that if a future edit ever breaks that invariant (e.g. the
            // two literal sets drift apart), the library reports a clean
            // error instead of panicking.
            Err(NumRs2Error::DeserializationError(format!(
                "Unsupported type for NPY deserialization: {}",
                other
            )))
        }
    }
}

// Parse NPY header to extract shape and dtype information
fn parse_npy_header(header: &[u8]) -> Result<(Vec<usize>, String)> {
    // Check magic string
    if header.len() < 8 || &header[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string(),
        ));
    }

    // Read version
    let major_version = header[6];
    let minor_version = header[7];

    if major_version != 1 || minor_version != 0 {
        return Err(NumRs2Error::DeserializationError(format!(
            "Unsupported NPY version: {}.{}",
            major_version, minor_version
        )));
    }

    // Read header length
    let header_len = LittleEndian::read_u16(&header[8..10]) as usize;

    if header.len() < 10 + header_len {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: header too short".to_string(),
        ));
    }

    // Extract the Python dictionary string
    let dict_bytes = &header[10..10 + header_len];
    let dict_str = std::str::from_utf8(dict_bytes).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Invalid NPY header encoding: {}", e))
    })?;

    // Parse dtype
    let dtype_start = dict_str.find("'descr': '").ok_or_else(|| {
        NumRs2Error::DeserializationError("Invalid NPY header: missing 'descr'".to_string())
    })?;
    let dtype_start = dtype_start + "'descr': '".len();
    let dtype_end = dict_str[dtype_start..].find("'").ok_or_else(|| {
        NumRs2Error::DeserializationError("Invalid NPY header: malformed 'descr'".to_string())
    })?;
    let dtype = dict_str[dtype_start..dtype_start + dtype_end].to_string();

    // Parse shape
    let shape_start = dict_str.find("'shape': (").ok_or_else(|| {
        NumRs2Error::DeserializationError("Invalid NPY header: missing 'shape'".to_string())
    })?;
    let shape_start = shape_start + "'shape': (".len();
    let shape_end = dict_str[shape_start..].find(")").ok_or_else(|| {
        NumRs2Error::DeserializationError("Invalid NPY header: malformed 'shape'".to_string())
    })?;
    let shape_str = dict_str[shape_start..shape_start + shape_end].trim();

    // Handle empty shape (scalar)
    if shape_str.is_empty() {
        return Ok((vec![], dtype));
    }

    // Parse shape dimensions
    let mut shape = Vec::new();
    for dim_str in shape_str.split(',') {
        let dim_str = dim_str.trim();
        if dim_str.is_empty() {
            continue;
        }
        let dim = dim_str.parse::<usize>().map_err(|e| {
            NumRs2Error::DeserializationError(format!(
                "Invalid shape dimension in NPY header: {}",
                e
            ))
        })?;
        shape.push(dim);
    }

    Ok((shape, dtype))
}

// Public function to serialize an array to a file in NPY or NPZ format
pub fn serialize_to_file<T: Clone, W: Write + Seek>(
    array: &Array<T>,
    writer: &mut W,
    format: SerializeFormat,
) -> Result<()> {
    let type_name = std::any::type_name::<T>();

    // Create a temporary buffer for the NPY file content
    let mut npy_data = Vec::new();

    // Create NPY header
    let header = construct_npy_header::<T>(&array.shape())?;

    // Write header to the buffer
    npy_data.extend_from_slice(&header);

    // Write the data based on its type
    let data = array.to_vec();
    encode_npy_elements("NPY/NPZ", type_name, &data, &mut npy_data)?;

    // If it's just NPY format, write directly to the file
    if matches!(format, SerializeFormat::Npy) {
        writer
            .write_all(&npy_data)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to write NPY data: {}", e)))?;
    } else {
        // For NPZ format, create a ZIP file using OxiARC (takes ownership of writer)
        let mut zip_writer = ZipWriter::new(writer);

        // NOTE: Using Store (no compression) for this simple wrapper
        // For compressed NPZ, use save_npz_arrays() with compressed=true
        zip_writer.set_compression(ZipCompressionLevel::Store);

        // Add the NPY file to the ZIP archive
        // Use "arr_0.npy" as the default name
        zip_writer
            .add_file("arr_0.npy", &npy_data)
            .map_err(|e| NumRs2Error::IOError(format!("Failed to add file to NPZ: {}", e)))?;

        // Finish and consume the ZIP writer (this also flushes the writer)
        zip_writer
            .into_inner()
            .map_err(|e| NumRs2Error::IOError(format!("Failed to finalize NPZ file: {}", e)))?;
    }

    Ok(())
}

// Generic function to read NPY data for any supported type
fn read_npy_generic<T: Clone, R: Read>(mut reader: R) -> Result<Array<T>> {
    // Read NPY header (first 10 bytes contain magic string, version, and header length)
    let mut header_prefix = [0u8; 10];
    reader.read_exact(&mut header_prefix).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY header: {}", e))
    })?;

    // Check magic string
    if &header_prefix[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string(),
        ));
    }

    // Read header length (little endian)
    let header_len = LittleEndian::read_u16(&header_prefix[8..10]) as usize;

    // Read the rest of the header
    let mut header_data = vec![0u8; header_len];
    reader.read_exact(&mut header_data).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY header data: {}", e))
    })?;

    // Combine header prefix and data for parsing
    let mut full_header = Vec::with_capacity(10 + header_len);
    full_header.extend_from_slice(&header_prefix);
    full_header.extend_from_slice(&header_data);

    // Parse header to get shape and dtype
    let (shape, dtype) = parse_npy_header(&full_header)?;

    // Determine element size and read data
    let type_name = std::any::type_name::<T>();
    let (element_size, expected_dtype) = match type_name {
        "f32" => (4, "<f4"),
        "f64" => (8, "<f8"),
        "i8" => (1, "<i1"),
        "i16" => (2, "<i2"),
        "i32" => (4, "<i4"),
        "i64" => (8, "<i8"),
        "u8" => (1, "<u1"),
        "u16" => (2, "<u2"),
        "u32" => (4, "<u4"),
        "u64" => (8, "<u8"),
        "bool" => (1, "|b1"),
        _ => {
            return Err(NumRs2Error::DeserializationError(format!(
                "Unsupported type for NPY deserialization: {}",
                type_name
            )));
        }
    };

    // Verify the dtype is compatible
    if dtype != expected_dtype {
        return Err(NumRs2Error::DeserializationError(format!(
            "Expected {} data (dtype '{}'), but got '{}'",
            type_name, expected_dtype, dtype
        )));
    }

    // Read raw data
    let total_elements: usize = shape.iter().product();
    let mut raw_data = vec![0u8; total_elements * element_size];
    reader.read_exact(&mut raw_data).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY data: {}", e))
    })?;

    // Convert raw bytes to typed values
    let typed_data = decode_npy_elements::<T>(type_name, &raw_data)?;

    // Create the array
    Array::from_vec_shape(typed_data, &shape)
}

// Generic function to read NPZ data for any supported type
fn read_npz_generic<T: Clone, R: Read + Seek>(reader: R) -> Result<Array<T>> {
    // Open a ZIP archive from the reader using OxiARC
    let mut zip_reader = ZipReader::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;

    // Find the first .npy file in the archive and clone it
    let npy_entry = zip_reader
        .entries()
        .iter()
        .find(|e| e.name.ends_with(".npy"))
        .cloned()
        .ok_or_else(|| {
            NumRs2Error::DeserializationError("No .npy files found in NPZ archive".to_string())
        })?;

    // Extract the NPY file data
    let npy_data = zip_reader.extract(&npy_entry).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
    })?;

    // Use the generic NPY reader with a cursor
    read_npy_generic(std::io::Cursor::new(npy_data))
}

/// List all array names in an NPZ archive
///
/// # Arguments
/// * `reader` - A readable and seekable source (e.g., File)
///
/// # Returns
/// A vector of array names (without .npy extension)
///
/// # Examples
/// ```rust,no_run
/// use numrs2::io::list_npz_arrays;
/// use std::fs::File;
///
/// let file = File::open("arrays.npz").expect("Failed to open NPZ file");
/// let array_names = list_npz_arrays(file).expect("Failed to list NPZ arrays");
/// println!("Arrays in NPZ: {:?}", array_names);
/// ```
pub fn list_npz_arrays<R: Read + Seek>(reader: R) -> Result<Vec<String>> {
    let zip_reader = ZipReader::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;

    let array_names: Vec<String> = zip_reader
        .entries()
        .iter()
        .filter(|entry| entry.name.ends_with(".npy"))
        .map(|entry| entry.name.trim_end_matches(".npy").to_string())
        .collect();

    Ok(array_names)
}

/// Load a specific named array from an NPZ archive
///
/// # Arguments
/// * `reader` - A readable and seekable source (e.g., File)
/// * `array_name` - Name of the array to load (without .npy extension)
///
/// # Returns
/// The array with the specified name
///
/// # Examples
/// ```rust,no_run
/// use numrs2::io::load_npz_array;
/// use std::fs::File;
///
/// let file = File::open("arrays.npz").expect("Failed to open NPZ file");
/// let array = load_npz_array::<f32, _>(file, "my_array").expect("Failed to load array from NPZ");
/// ```
pub fn load_npz_array<T: Clone, R: Read + Seek>(reader: R, array_name: &str) -> Result<Array<T>> {
    let mut zip_reader = ZipReader::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;

    // Look for the array with the specified name and clone it
    let npy_filename = format!("{}.npy", array_name);
    let npy_entry = zip_reader
        .entry_by_name(&npy_filename)
        .cloned()
        .ok_or_else(|| {
            NumRs2Error::DeserializationError(format!(
                "Array '{}' not found in NPZ archive",
                array_name
            ))
        })?;

    // Extract the NPY file data
    let npy_data = zip_reader.extract(&npy_entry).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
    })?;

    // Use the generic NPY reader with a cursor
    read_npy_generic(std::io::Cursor::new(npy_data))
}

/// Load all arrays from an NPZ archive
///
/// # Arguments
/// * `reader` - A readable and seekable source (e.g., File)
///
/// # Returns
/// A HashMap mapping array names to their data
///
/// # Examples
/// ```rust,no_run
/// use numrs2::io::load_all_npz_arrays;
/// use std::fs::File;
///
/// let file = File::open("arrays.npz").expect("Failed to open NPZ file");
/// let arrays = load_all_npz_arrays::<f32, _>(file).expect("Failed to load all arrays from NPZ");
/// for (name, array) in &arrays {
///     println!("Array '{}' has shape {:?}", name, array.shape());
/// }
/// ```
pub fn load_all_npz_arrays<T: Clone, R: Read + Seek>(
    reader: R,
) -> Result<std::collections::HashMap<String, Array<T>>> {
    let mut zip_reader = ZipReader::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;

    let mut arrays = std::collections::HashMap::new();

    // Collect all .npy entries
    let npy_entries: Vec<_> = zip_reader
        .entries()
        .iter()
        .filter(|entry| entry.name.ends_with(".npy"))
        .cloned()
        .collect();

    // Load each NPY file
    for entry in npy_entries {
        let array_name = entry.name.trim_end_matches(".npy").to_string();

        // Extract the NPY file data
        let npy_data = zip_reader.extract(&entry).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
        })?;

        // Parse the NPY data
        let array = read_npy_generic(std::io::Cursor::new(npy_data))?;
        arrays.insert(array_name, array);
    }

    Ok(arrays)
}

/// Save multiple arrays to a single NPZ archive
///
/// # Arguments
/// * `arrays` - HashMap mapping array names to arrays
/// * `writer` - A writable and seekable destination (e.g., File)
/// * `compressed` - Whether to use compression (Deflated if true, Stored if false)
///
/// # Returns
/// Ok(()) if successful
///
/// # Examples
/// ```rust,no_run
/// use numrs2::io::save_npz_arrays;
/// use numrs2::prelude::*;
/// use std::collections::HashMap;
/// use std::fs::File;
///
/// let mut arrays = HashMap::new();
/// arrays.insert("data".to_string(), Array::from_vec(vec![1.0, 2.0, 3.0]));
/// arrays.insert("weights".to_string(), Array::from_vec(vec![0.1, 0.5, 0.4]));
///
/// let file = File::create("output.npz").expect("Failed to create NPZ file");
/// save_npz_arrays(&arrays, file, true).expect("Failed to save arrays to NPZ");
/// ```
pub fn save_npz_arrays<T: Clone, W: Write + Seek>(
    arrays: &std::collections::HashMap<String, Array<T>>,
    writer: W,
    compressed: bool,
) -> Result<()> {
    if arrays.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot save empty array collection to NPZ".to_string(),
        ));
    }

    let type_name = std::any::type_name::<T>();

    // Create ZIP writer using OxiARC (takes ownership of writer)
    let mut zip_writer = ZipWriter::new(writer);

    // Use compression based on parameter (OxiARC v0.2.1+ supports multi-file DEFLATE)
    let compression = if compressed {
        ZipCompressionLevel::Normal
    } else {
        ZipCompressionLevel::Store
    };

    // Save each array to the NPZ archive
    for (name, array) in arrays.iter() {
        // Create NPY data for this array
        let mut npy_data = Vec::new();

        // Create NPY header
        let header = construct_npy_header::<T>(&array.shape())?;
        npy_data.extend_from_slice(&header);

        // Write the data based on its type
        let data = array.to_vec();
        encode_npy_elements("NPZ", type_name, &data, &mut npy_data)?;

        // Add this array to the ZIP archive using OxiARC
        let filename = format!("{}.npy", name);
        zip_writer
            .add_file_with_options(&filename, &npy_data, compression)
            .map_err(|e| {
                NumRs2Error::IOError(format!("Failed to add NPZ entry '{}': {}", name, e))
            })?;
    }

    // Finish and consume the ZIP writer (this also flushes the writer)
    zip_writer
        .into_inner()
        .map_err(|e| NumRs2Error::IOError(format!("Failed to finalize NPZ file: {}", e)))?;

    Ok(())
}

// Public function to deserialize an array from a file in NPY or NPZ format
pub fn deserialize_from_file<T: Clone, R: Read + Seek>(
    reader: R,
    format: SerializeFormat,
) -> Result<Array<T>> {
    match format {
        SerializeFormat::Npy => read_npy_generic(reader),
        SerializeFormat::Npz => read_npz_generic(reader),
        _ => Err(NumRs2Error::DeserializationError(
            "Only NPY and NPZ formats are supported".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_header_construction() {
        // Test header construction for a simple 2D array
        let shape = vec![2, 3];
        let header = construct_npy_header::<f32>(&shape).expect("Failed to construct NPY header");

        // Check that the header contains the magic string and correct version
        assert_eq!(&header[0..6], NPY_MAGIC_STRING);
        assert_eq!(header[6], NPY_MAJOR_VERSION);
        assert_eq!(header[7], NPY_MINOR_VERSION);

        // Check that the header contains the correct shape information
        let header_str = std::str::from_utf8(&header[10..]).expect("Invalid UTF-8 in header");
        assert!(header_str.contains("'shape': (2, 3)"));
        assert!(header_str.contains("'descr': '<f4'"));
        assert!(header_str.contains("'fortran_order': False"));
    }

    #[test]
    fn test_npy_header_parsing() {
        // Create a test header
        let shape = vec![2, 3];
        let header = construct_npy_header::<f32>(&shape).expect("Failed to construct NPY header");

        // Parse the header and check the result
        let (parsed_shape, dtype) = parse_npy_header(&header).expect("Failed to parse NPY header");
        assert_eq!(parsed_shape, shape);
        assert_eq!(dtype, "<f4");
    }

    #[test]
    fn test_save_multiple_arrays_npz() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create multiple arrays
        let mut arrays = HashMap::new();
        arrays.insert(
            "data".to_string(),
            Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]),
        );
        arrays.insert(
            "weights".to_string(),
            Array::from_vec(vec![0.1f64, 0.5, 0.4]),
        );
        arrays.insert("labels".to_string(), Array::from_vec(vec![10.0f64, 20.0]));

        // Save to NPZ
        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

        // Load all arrays back
        buffer.set_position(0);
        let loaded_arrays =
            load_all_npz_arrays::<f64, _>(buffer).expect("Failed to load all NPZ arrays");

        // Verify we got all arrays back
        assert_eq!(loaded_arrays.len(), 3);
        assert!(loaded_arrays.contains_key("data"));
        assert!(loaded_arrays.contains_key("weights"));
        assert!(loaded_arrays.contains_key("labels"));

        // Verify the content
        let data = &loaded_arrays["data"];
        assert_eq!(data.shape(), vec![2, 3]);
        assert_eq!(data.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let weights = &loaded_arrays["weights"];
        assert_eq!(weights.shape(), vec![3]);
        assert_eq!(weights.to_vec(), vec![0.1, 0.5, 0.4]);

        let labels = &loaded_arrays["labels"];
        assert_eq!(labels.shape(), vec![2]);
        assert_eq!(labels.to_vec(), vec![10.0, 20.0]);
    }

    #[test]
    fn test_save_multiple_arrays_uncompressed() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create multiple arrays with different types
        let mut arrays = HashMap::new();
        arrays.insert("a".to_string(), Array::from_vec(vec![1i32, 2, 3]));
        arrays.insert("b".to_string(), Array::from_vec(vec![4i32, 5, 6, 7]));

        // Save without compression
        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, false).expect("Failed to save NPZ arrays");

        // Load back
        buffer.set_position(0);
        let loaded = load_all_npz_arrays::<i32, _>(buffer).expect("Failed to load all NPZ arrays");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["a"].to_vec(), vec![1, 2, 3]);
        assert_eq!(loaded["b"].to_vec(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_load_specific_array_from_npz() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create and save multiple arrays
        let mut arrays = HashMap::new();
        arrays.insert("first".to_string(), Array::from_vec(vec![1.0f32, 2.0]));
        arrays.insert(
            "second".to_string(),
            Array::from_vec(vec![3.0f32, 4.0, 5.0]),
        );
        arrays.insert("third".to_string(), Array::from_vec(vec![6.0f32]));

        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

        // Load only the second array
        buffer.set_position(0);
        let second_array = load_npz_array::<f32, _>(buffer, "second")
            .expect("Failed to load second array from NPZ");
        assert_eq!(second_array.to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_list_arrays_in_npz() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create and save multiple arrays
        let mut arrays = HashMap::new();
        arrays.insert("alpha".to_string(), Array::from_vec(vec![1.0f64]));
        arrays.insert("beta".to_string(), Array::from_vec(vec![2.0f64]));
        arrays.insert("gamma".to_string(), Array::from_vec(vec![3.0f64]));

        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

        // List array names
        buffer.set_position(0);
        let mut names = list_npz_arrays(buffer).expect("Failed to list NPZ arrays");
        names.sort(); // Sort for consistent comparison

        let mut expected = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        expected.sort();

        assert_eq!(names, expected);
    }

    #[test]
    fn test_save_empty_arrays_fails() {
        use std::collections::HashMap;
        use std::io::Cursor;

        let arrays: HashMap<String, Array<f64>> = HashMap::new();
        let mut buffer = Cursor::new(Vec::new());

        let result = save_npz_arrays(&arrays, &mut buffer, true);
        assert!(result.is_err());
        assert!(matches!(result, Err(NumRs2Error::InvalidOperation(_))));
    }

    #[test]
    fn test_save_different_shapes_same_npz() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create arrays with different shapes and dimensions
        let mut arrays = HashMap::new();
        arrays.insert("scalar".to_string(), Array::from_vec(vec![42.0f64])); // 1D, size 1
        arrays.insert(
            "vector".to_string(),
            Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]),
        ); // 1D, size 5
        arrays.insert(
            "matrix".to_string(),
            Array::from_vec(vec![
                1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ])
            .reshape(&[3, 4]),
        ); // 2D, 3x4
        arrays.insert(
            "tensor".to_string(),
            Array::from_vec(vec![1.0f64; 24]).reshape(&[2, 3, 4]),
        ); // 3D, 2x3x4

        // Save all to NPZ
        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

        // Load all back
        buffer.set_position(0);
        let loaded = load_all_npz_arrays::<f64, _>(buffer).expect("Failed to load all NPZ arrays");

        // Verify all arrays
        assert_eq!(loaded["scalar"].shape(), vec![1]);
        assert_eq!(loaded["vector"].shape(), vec![5]);
        assert_eq!(loaded["matrix"].shape(), vec![3, 4]);
        assert_eq!(loaded["tensor"].shape(), vec![2, 3, 4]);
    }

    #[test]
    fn test_save_different_types() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Test with different numeric types
        macro_rules! test_type {
            ($t:ty, $values:expr) => {{
                let mut arrays = HashMap::new();
                arrays.insert("test".to_string(), Array::from_vec($values));

                let mut buffer = Cursor::new(Vec::new());
                save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

                buffer.set_position(0);
                let loaded =
                    load_all_npz_arrays::<$t, _>(buffer).expect("Failed to load all NPZ arrays");
                assert_eq!(loaded["test"].to_vec(), $values);
            }};
        }

        test_type!(f32, vec![1.0f32, 2.0, 3.0]);
        test_type!(f64, vec![1.0f64, 2.0, 3.0]);
        test_type!(i32, vec![1i32, 2, 3]);
        test_type!(i64, vec![1i64, 2, 3]);
        test_type!(u32, vec![1u32, 2, 3]);
        test_type!(u64, vec![1u64, 2, 3]);
    }

    #[test]
    fn test_load_nonexistent_array() {
        use std::collections::HashMap;
        use std::io::Cursor;

        // Create and save an array
        let mut arrays = HashMap::new();
        arrays.insert("exists".to_string(), Array::from_vec(vec![1.0f64]));

        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save NPZ arrays");

        // Try to load an array that doesn't exist
        buffer.set_position(0);
        let result = load_npz_array::<f64, _>(buffer, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_bool_roundtrip_npy() {
        use std::io::Cursor;

        let array = Array::from_vec(vec![true, false, true, true, false]);
        let mut buffer = Cursor::new(Vec::new());
        serialize_to_file(&array, &mut buffer, SerializeFormat::Npy)
            .expect("Failed to serialize bool array to NPY");

        buffer.set_position(0);
        let loaded: Array<bool> = deserialize_from_file(buffer, SerializeFormat::Npy)
            .expect("Failed to deserialize bool array from NPY");

        assert_eq!(loaded.to_vec(), vec![true, false, true, true, false]);
    }

    #[test]
    fn test_bool_roundtrip_npz() {
        use std::collections::HashMap;
        use std::io::Cursor;

        let mut arrays = HashMap::new();
        arrays.insert(
            "flags".to_string(),
            Array::from_vec(vec![false, true, false, false, true, true]),
        );

        let mut buffer = Cursor::new(Vec::new());
        save_npz_arrays(&arrays, &mut buffer, true).expect("Failed to save bool array to NPZ");

        buffer.set_position(0);
        let loaded =
            load_npz_array::<bool, _>(buffer, "flags").expect("Failed to load bool array from NPZ");

        assert_eq!(loaded.to_vec(), vec![false, true, false, false, true, true]);
    }

    #[test]
    fn test_type_name_dispatch_not_confused_by_custom_type_named_f32() {
        // A zero-sized type deliberately named `f32`, defined in a nested
        // module, to prove that the `type_name`-based dispatch above
        // distinguishes it from the real primitive `f32`. `std::any::type_name`
        // always includes the defining module's path for non-primitive
        // types -- only the actual primitives print as a bare keyword like
        // "f32" -- so this type's `type_name()` is `"...::fake::f32"`, not
        // `"f32"`, and dispatch must fall through to the "unsupported
        // type" error path rather than (mis)treating it as the primitive.
        #[allow(non_camel_case_types)]
        mod fake {
            #[derive(Clone)]
            pub struct f32;
        }

        use std::io::Cursor;

        let array: Array<fake::f32> = Array::from_vec(vec![fake::f32, fake::f32, fake::f32]);
        let mut buffer = Cursor::new(Vec::new());
        let result = serialize_to_file(&array, &mut buffer, SerializeFormat::Npy);

        assert!(
            result.is_err(),
            "a custom type literally named `f32` must not be silently treated as the primitive f32"
        );
    }

    #[test]
    fn test_encode_size_mismatch_is_caught_not_ub() {
        // Simulates what would happen if the `type_name`-based
        // classification were ever wrong: force a mismatch between the
        // dispatch string ("f64", 8 bytes) and T's real size (i32, 4
        // bytes) and confirm the explicit size guard turns it into a
        // clean error instead of an out-of-bounds `transmute_copy` read.
        let data: Vec<i32> = vec![1, 2, 3];
        let mut out = Vec::new();
        let result = encode_npy_elements("test", "f64", &data, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_size_mismatch_is_caught_not_ub() {
        let raw = vec![0u8; 32];
        let result: Result<Vec<i32>> = decode_npy_elements("f64", &raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_unsupported_type_name_returns_error() {
        // Exercises the (should-be-unreachable-in-practice) fallback arm
        // directly: an unknown dispatch string must return a clean `Err`,
        // never panic.
        let raw = vec![0u8; 16];
        let result: Result<Vec<f64>> = decode_npy_elements("totally_unknown_type", &raw);
        assert!(result.is_err());
    }
}

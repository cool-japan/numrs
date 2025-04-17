use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use std::io::{Read, Seek, Write};
use zip::{ZipArchive, ZipWriter, write::FileOptions};
use super::SerializeFormat;

// NPY magic numbers and constants
const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";
const NPY_MAJOR_VERSION: u8 = 1;
const NPY_MINOR_VERSION: u8 = 0;

// Helper function to get NumPy dtype string for Rust type
fn get_numpy_dtype<T>() -> &'static str {
    // This is a simplified mapping for common types
    // NumPy has more detailed type descriptors
    if std::any::type_name::<T>() == "f32" { "<f4" }
    else if std::any::type_name::<T>() == "f64" { "<f8" }
    else if std::any::type_name::<T>() == "i8" { "<i1" }
    else if std::any::type_name::<T>() == "i16" { "<i2" }
    else if std::any::type_name::<T>() == "i32" { "<i4" }
    else if std::any::type_name::<T>() == "i64" { "<i8" }
    else if std::any::type_name::<T>() == "u8" { "<u1" }
    else if std::any::type_name::<T>() == "u16" { "<u2" }
    else if std::any::type_name::<T>() == "u32" { "<u4" }
    else if std::any::type_name::<T>() == "u64" { "<u8" }
    else if std::any::type_name::<T>() == "bool" { "|b1" }
    else { "unknown" }
}

// Construct NPY header for the specified array
fn construct_npy_header<T>(shape: &[usize]) -> Result<Vec<u8>> {
    let dtype = get_numpy_dtype::<T>();
    if dtype == "unknown" {
        return Err(NumRs2Error::SerializationError(
            format!("Unsupported type for NPY format: {}", std::any::type_name::<T>())
        ));
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

// Parse NPY header to extract shape and dtype information
fn parse_npy_header(header: &[u8]) -> Result<(Vec<usize>, String)> {
    // Check magic string
    if header.len() < 8 || &header[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string()
        ));
    }
    
    // Read version
    let major_version = header[6];
    let minor_version = header[7];
    
    if major_version != 1 || minor_version != 0 {
        return Err(NumRs2Error::DeserializationError(
            format!("Unsupported NPY version: {}.{}", major_version, minor_version)
        ));
    }
    
    // Read header length
    let header_len = LittleEndian::read_u16(&header[8..10]) as usize;
    
    if header.len() < 10 + header_len {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: header too short".to_string()
        ));
    }
    
    // Extract the Python dictionary string
    let dict_bytes = &header[10..10 + header_len];
    let dict_str = std::str::from_utf8(dict_bytes)
        .map_err(|e| NumRs2Error::DeserializationError(
            format!("Invalid NPY header encoding: {}", e)
        ))?;
    
    // Parse dtype
    let dtype_start = dict_str.find("'descr': '")
        .ok_or_else(|| NumRs2Error::DeserializationError(
            "Invalid NPY header: missing 'descr'".to_string()
        ))?;
    let dtype_start = dtype_start + "'descr': '".len();
    let dtype_end = dict_str[dtype_start..].find("'")
        .ok_or_else(|| NumRs2Error::DeserializationError(
            "Invalid NPY header: malformed 'descr'".to_string()
        ))?;
    let dtype = dict_str[dtype_start..dtype_start + dtype_end].to_string();
    
    // Parse shape
    let shape_start = dict_str.find("'shape': (")
        .ok_or_else(|| NumRs2Error::DeserializationError(
            "Invalid NPY header: missing 'shape'".to_string()
        ))?;
    let shape_start = shape_start + "'shape': (".len();
    let shape_end = dict_str[shape_start..].find(")")
        .ok_or_else(|| NumRs2Error::DeserializationError(
            "Invalid NPY header: malformed 'shape'".to_string()
        ))?;
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
        let dim = dim_str.parse::<usize>()
            .map_err(|e| NumRs2Error::DeserializationError(
                format!("Invalid shape dimension in NPY header: {}", e)
            ))?;
        shape.push(dim);
    }
    
    Ok((shape, dtype))
}

// Public function to serialize an array to a file in NPY or NPZ format
pub fn serialize_to_file<T: Clone, W: Write + Seek>(array: &Array<T>, writer: &mut W, format: SerializeFormat) -> Result<()> {
    let type_name = std::any::type_name::<T>();
    
    // Create a temporary buffer for the NPY file content
    let mut npy_data = Vec::new();
    
    // Create NPY header
    let header = construct_npy_header::<T>(&array.shape())?;
    
    // Write header to the buffer
    npy_data.extend_from_slice(&header);
    
    // Write the data based on its type
    if type_name == "f32" {
        // Convert data to raw bytes and write
        let data = array.to_vec();
        for val in data.iter() {
            // Safety: This is only allowed because we've checked that T is f32
            let val_bytes = unsafe { std::mem::transmute_copy::<T, f32>(val) }.to_le_bytes();
            npy_data.extend_from_slice(&val_bytes);
        }
    } else if type_name == "f64" {
        // Convert data to raw bytes and write
        let data = array.to_vec();
        for val in data.iter() {
            // Safety: This is only allowed because we've checked that T is f64
            let val_bytes = unsafe { std::mem::transmute_copy::<T, f64>(val) }.to_le_bytes();
            npy_data.extend_from_slice(&val_bytes);
        }
    } else if type_name == "i32" {
        // Convert data to raw bytes and write
        let data = array.to_vec();
        for val in data.iter() {
            // Safety: This is only allowed because we've checked that T is i32
            let val_bytes = unsafe { std::mem::transmute_copy::<T, i32>(val) }.to_le_bytes();
            npy_data.extend_from_slice(&val_bytes);
        }
    } else {
        return Err(NumRs2Error::SerializationError(
            format!("NPY/NPZ format only supports f32, f64, and i32 types. Got: {}", type_name)
        ));
    }
    
    // If it's just NPY format, write directly to the file
    if matches!(format, SerializeFormat::Npy) {
        writer.write_all(&npy_data).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to write NPY data: {}", e))
        })?;
    } else {
        // For NPZ format, create a ZIP file
        let mut zip_writer = ZipWriter::new(writer);
        
        // Add the NPY file to the ZIP archive
        let options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        
        // Use "arr_0.npy" as the default name
        zip_writer.start_file("arr_0.npy", options).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to create NPZ file: {}", e))
        })?;
        
        // Write the NPY content to the ZIP file
        zip_writer.write_all(&npy_data).map_err(|e| {
            NumRs2Error::IOError(format!("Failed to write NPY data to NPZ: {}", e))
        })?;
        
        // Finish the ZIP file
        zip_writer.finish().map_err(|e| {
            NumRs2Error::IOError(format!("Failed to finalize NPZ file: {}", e))
        })?;
    }
    
    Ok(())
}

// Helper function for NPY format deserialization for f32 arrays
fn read_npy_f32<R: Read>(mut reader: R) -> Result<Array<f32>> {
    // Read NPY header (first 10 bytes contain magic string, version, and header length)
    let mut header_prefix = [0u8; 10];
    reader.read_exact(&mut header_prefix).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY header: {}", e))
    })?;
    
    // Check magic string
    if &header_prefix[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string()
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
    
    // Verify the dtype is compatible with f32
    if dtype != "<f4" {
        return Err(NumRs2Error::DeserializationError(
            format!("Expected f32 data (dtype '<f4'), but got '{}'", dtype)
        ));
    }
    
    // Read data based on dtype and shape
    let total_elements: usize = shape.iter().product();
    let mut raw_data = vec![0u8; total_elements * 4]; // 4 bytes per f32
    reader.read_exact(&mut raw_data).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY data: {}", e))
    })?;
    
    // Convert raw bytes to f32 values
    let mut f32_data = Vec::with_capacity(total_elements);
    for chunk in raw_data.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        f32_data.push(value);
    }
    
    // Create the array
    Ok(Array::from_vec(f32_data).reshape(&shape))
}

// Helper function for NPY format deserialization for f64 arrays
fn read_npy_f64<R: Read>(mut reader: R) -> Result<Array<f64>> {
    // Read NPY header (first 10 bytes contain magic string, version, and header length)
    let mut header_prefix = [0u8; 10];
    reader.read_exact(&mut header_prefix).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY header: {}", e))
    })?;
    
    // Check magic string
    if &header_prefix[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string()
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
    
    // Verify the dtype is compatible with f64
    if dtype != "<f8" {
        return Err(NumRs2Error::DeserializationError(
            format!("Expected f64 data (dtype '<f8'), but got '{}'", dtype)
        ));
    }
    
    // Read data based on dtype and shape
    let total_elements: usize = shape.iter().product();
    let mut raw_data = vec![0u8; total_elements * 8]; // 8 bytes per f64
    reader.read_exact(&mut raw_data).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY data: {}", e))
    })?;
    
    // Convert raw bytes to f64 values
    let mut f64_data = Vec::with_capacity(total_elements);
    for chunk in raw_data.chunks_exact(8) {
        let value = f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7]
        ]);
        f64_data.push(value);
    }
    
    // Create the array
    Ok(Array::from_vec(f64_data).reshape(&shape))
}

// Helper function for NPY format deserialization for i32 arrays
fn read_npy_i32<R: Read>(mut reader: R) -> Result<Array<i32>> {
    // Read NPY header (first 10 bytes contain magic string, version, and header length)
    let mut header_prefix = [0u8; 10];
    reader.read_exact(&mut header_prefix).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY header: {}", e))
    })?;
    
    // Check magic string
    if &header_prefix[0..6] != NPY_MAGIC_STRING {
        return Err(NumRs2Error::DeserializationError(
            "Invalid NPY file: missing magic string".to_string()
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
    
    // Verify the dtype is compatible with i32
    if dtype != "<i4" {
        return Err(NumRs2Error::DeserializationError(
            format!("Expected i32 data (dtype '<i4'), but got '{}'", dtype)
        ));
    }
    
    // Read data based on dtype and shape
    let total_elements: usize = shape.iter().product();
    let mut raw_data = vec![0u8; total_elements * 4]; // 4 bytes per i32
    reader.read_exact(&mut raw_data).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to read NPY data: {}", e))
    })?;
    
    // Convert raw bytes to i32 values
    let mut i32_data = Vec::with_capacity(total_elements);
    for chunk in raw_data.chunks_exact(4) {
        let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        i32_data.push(value);
    }
    
    // Create the array
    Ok(Array::from_vec(i32_data).reshape(&shape))
}

// Helper function for NPZ format deserialization for f32 arrays
fn read_npz_f32<R: Read + Seek>(reader: R) -> Result<Array<f32>> {
    // Open a ZIP archive from the reader
    let mut archive = ZipArchive::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;
    
    // Find the first .npy file in the archive
    let mut npy_index = None;
    for i in 0..archive.len() {
        let name = archive.by_index(i).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to access file in NPZ archive: {}", e))
        })?.name().to_string();
        
        if name.ends_with(".npy") {
            npy_index = Some(i);
            break;
        }
    }
    
    let npy_idx = npy_index.ok_or_else(|| NumRs2Error::DeserializationError(
        "No .npy files found in NPZ archive".to_string()
    ))?;
    
    // Extract the NPY file and read it
    let npy_file = archive.by_index(npy_idx).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
    })?;
    
    // Use the same logic as read_npy_f32 but with the npy_file reader
    read_npy_f32(npy_file)
}

// Helper function for NPZ format deserialization for f64 arrays
fn read_npz_f64<R: Read + Seek>(reader: R) -> Result<Array<f64>> {
    // Open a ZIP archive from the reader
    let mut archive = ZipArchive::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;
    
    // Find the first .npy file in the archive
    let mut npy_index = None;
    for i in 0..archive.len() {
        let name = archive.by_index(i).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to access file in NPZ archive: {}", e))
        })?.name().to_string();
        
        if name.ends_with(".npy") {
            npy_index = Some(i);
            break;
        }
    }
    
    let npy_idx = npy_index.ok_or_else(|| NumRs2Error::DeserializationError(
        "No .npy files found in NPZ archive".to_string()
    ))?;
    
    // Extract the NPY file and read it
    let npy_file = archive.by_index(npy_idx).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
    })?;
    
    // Use the same logic as read_npy_f64 but with the npy_file reader
    read_npy_f64(npy_file)
}

// Helper function for NPZ format deserialization for i32 arrays
fn read_npz_i32<R: Read + Seek>(reader: R) -> Result<Array<i32>> {
    // Open a ZIP archive from the reader
    let mut archive = ZipArchive::new(reader).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to open NPZ file: {}", e))
    })?;
    
    // Find the first .npy file in the archive
    let mut npy_index = None;
    for i in 0..archive.len() {
        let name = archive.by_index(i).map_err(|e| {
            NumRs2Error::DeserializationError(format!("Failed to access file in NPZ archive: {}", e))
        })?.name().to_string();
        
        if name.ends_with(".npy") {
            npy_index = Some(i);
            break;
        }
    }
    
    let npy_idx = npy_index.ok_or_else(|| NumRs2Error::DeserializationError(
        "No .npy files found in NPZ archive".to_string()
    ))?;
    
    // Extract the NPY file and read it
    let npy_file = archive.by_index(npy_idx).map_err(|e| {
        NumRs2Error::DeserializationError(format!("Failed to extract NPY file from NPZ: {}", e))
    })?;
    
    // Use the same logic as read_npy_i32 but with the npy_file reader
    read_npy_i32(npy_file)
}

// Public function to deserialize an array from a file in NPY or NPZ format
pub fn deserialize_from_file<T: Clone, R: Read + Seek>(reader: R, format: SerializeFormat) -> Result<Array<T>> {
    // For NPY and NPZ formats, we need to simplify in the MVP implementation
    // We'll limit to specific types and have separate implementations for each type
    let type_name = std::any::type_name::<T>();
    
    // Call the appropriate type-specific helper function
    // Unfortunately, we can't directly return Array<T> from a function that produces Array<specific_type>
    // So we have to use unsafe transmute, but we carefully check the types match first
    if type_name == "f32" {
        // Safe because we've checked that T is f32
        unsafe {
            let result = if matches!(format, SerializeFormat::Npy) {
                read_npy_f32(reader)
            } else {
                read_npz_f32(reader)
            };
            
            // We need to reinterpret the Result<Array<f32>> as Result<Array<T>>
            // This is safe because we've verified that T is f32
            std::mem::transmute(result)
        }
    } else if type_name == "f64" {
        // Safe because we've checked that T is f64
        unsafe {
            let result = if matches!(format, SerializeFormat::Npy) {
                read_npy_f64(reader)
            } else {
                read_npz_f64(reader)
            };
            
            // We need to reinterpret the Result<Array<f64>> as Result<Array<T>>
            // This is safe because we've verified that T is f64
            std::mem::transmute(result)
        }
    } else if type_name == "i32" {
        // Safe because we've checked that T is i32
        unsafe {
            let result = if matches!(format, SerializeFormat::Npy) {
                read_npy_i32(reader)
            } else {
                read_npz_i32(reader)
            };
            
            // We need to reinterpret the Result<Array<i32>> as Result<Array<T>>
            // This is safe because we've verified that T is i32
            std::mem::transmute(result)
        }
    } else {
        // For simplicity, we're only supporting f32, f64, and i32 in this initial implementation
        Err(NumRs2Error::DeserializationError(
            format!("NPY/NPZ format currently only supports f32, f64, and i32 types, got: {}", type_name)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_npy_header_construction() {
        // Test header construction for a simple 2D array
        let shape = vec![2, 3];
        let header = construct_npy_header::<f32>(&shape).unwrap();
        
        // Check that the header contains the magic string and correct version
        assert_eq!(&header[0..6], NPY_MAGIC_STRING);
        assert_eq!(header[6], NPY_MAJOR_VERSION);
        assert_eq!(header[7], NPY_MINOR_VERSION);
        
        // Check that the header contains the correct shape information
        let header_str = std::str::from_utf8(&header[10..]).unwrap();
        assert!(header_str.contains("'shape': (2, 3)"));
        assert!(header_str.contains("'descr': '<f4'"));
        assert!(header_str.contains("'fortran_order': False"));
    }
    
    #[test]
    fn test_npy_header_parsing() {
        // Create a test header
        let shape = vec![2, 3];
        let header = construct_npy_header::<f32>(&shape).unwrap();
        
        // Parse the header and check the result
        let (parsed_shape, dtype) = parse_npy_header(&header).unwrap();
        assert_eq!(parsed_shape, shape);
        assert_eq!(dtype, "<f4");
    }
}
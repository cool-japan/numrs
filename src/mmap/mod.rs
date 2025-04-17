// Memory-mapped array module for NumRS2
// Provides memory-mapped array functionality for efficient file-backed arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::io::{Read, Write};
use std::marker::PhantomData;
use memmap2::{MmapMut, MmapOptions};
use std::fmt;
use serde::{Serialize, Deserialize};
use std::mem;

/// Memory-mapped array with file backing
///
/// A memory-mapped array allows you to work with data that is stored in a file
/// as if it were in memory. This is particularly useful for large arrays that
/// might not fit in RAM.
#[derive(Debug)]
pub struct MmapArray<T: Copy> {
    /// The memory-mapped file
    mmap: MmapMut,
    /// The shape of the array
    shape: Vec<usize>,
    /// The total number of elements
    size: usize,
    /// Path to the backing file
    path: PathBuf,
    /// Phantom data for type T
    _phantom: PhantomData<T>,
}

/// Metadata for memory-mapped arrays
#[derive(Serialize, Deserialize, Debug)]
pub struct MmapArrayMeta {
    /// Element type name
    pub type_name: String,
    /// Element size in bytes
    pub type_size: usize,
    /// Array shape
    pub shape: Vec<usize>,
    /// Total number of elements
    pub size: usize,
    /// Version information
    pub version: u8,
}

impl<T: Copy> MmapArray<T> {
    /// Create a new memory-mapped array
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to back the array
    /// * `shape` - Shape of the array
    /// * `create` - Whether to create a new file (true) or open an existing one (false)
    ///
    /// # Returns
    ///
    /// A new MmapArray instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or opened, or if the file size
    /// does not match the expected size for the given shape.
    pub fn new<P: AsRef<Path>>(path: &P, shape: &[usize], create: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let size: usize = shape.iter().product();
        let data_size = size * mem::size_of::<T>();
        let meta_size = calculate_meta_size(shape);
        let total_size = meta_size + data_size;

        let file = if create {
            // Create a new file or truncate existing one
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            
            // Set the file size to accommodate metadata and data
            file.set_len(total_size as u64)?;
            
            // Write metadata
            let meta = MmapArrayMeta {
                type_name: std::any::type_name::<T>().to_string(),
                type_size: mem::size_of::<T>(),
                shape: shape.to_vec(),
                size,
                version: 1,
            };
            
            let meta_bytes = bincode::serialize(&meta)?;
            let mut file = file;
            file.write_all(&meta_bytes)?;
            
            file
        } else {
            // Open existing file
            let mut file = File::open(&path)?;
            
            // Read metadata
            let mut meta_bytes = vec![0u8; meta_size];
            file.read_exact(&mut meta_bytes)?;
            
            let meta: MmapArrayMeta = bincode::deserialize(&meta_bytes)?;
            
            // Verify metadata
            if meta.type_name != std::any::type_name::<T>() {
                return Err(NumRs2Error::InvalidOperation(
                    format!("Type mismatch: file contains '{}', but requested '{}'",
                            meta.type_name, std::any::type_name::<T>())
                ));
            }
            
            if meta.shape != shape {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: shape.to_vec(),
                    actual: meta.shape,
                });
            }
            
            file
        };
        
        // Create memory mapping
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        
        if mmap.len() != total_size {
            return Err(NumRs2Error::InvalidOperation(
                format!("File size mismatch: expected {} bytes, got {} bytes",
                        total_size, mmap.len())
            ));
        }
        
        Ok(Self {
            mmap,
            shape: shape.to_vec(),
            size,
            path,
            _phantom: PhantomData,
        })
    }
    
    /// Get the value at the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices to access
    ///
    /// # Returns
    ///
    /// The value at the specified indices
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are out of bounds or if the number of indices
    /// doesn't match the number of dimensions.
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.shape.len() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Expected {} indices, got {}", self.shape.len(), indices.len())
            ));
        }
        
        let offset = self.calculate_offset(indices)?;
        let meta_size = calculate_meta_size(&self.shape);
        let byte_offset = meta_size + offset * mem::size_of::<T>();
        
        if byte_offset + mem::size_of::<T>() > self.mmap.len() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Index out of bounds: offset {} exceeds mmap size {}", 
                        byte_offset, self.mmap.len())
            ));
        }
        
        // Read value from memory map
        let bytes = &self.mmap[byte_offset..byte_offset + mem::size_of::<T>()];
        let value = unsafe { *(bytes.as_ptr() as *const T) };
        
        Ok(value)
    }
    
    /// Set the value at the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices to access
    /// * `value` - The value to set
    ///
    /// # Returns
    ///
    /// () if successful
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are out of bounds or if the number of indices
    /// doesn't match the number of dimensions.
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(NumRs2Error::DimensionMismatch(
                format!("Expected {} indices, got {}", self.shape.len(), indices.len())
            ));
        }
        
        let offset = self.calculate_offset(indices)?;
        let meta_size = calculate_meta_size(&self.shape);
        let byte_offset = meta_size + offset * mem::size_of::<T>();
        
        if byte_offset + mem::size_of::<T>() > self.mmap.len() {
            return Err(NumRs2Error::IndexOutOfBounds(
                format!("Index out of bounds: offset {} exceeds mmap size {}", 
                        byte_offset, self.mmap.len())
            ));
        }
        
        // Write value to memory map
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &value as *const T as *const u8,
                mem::size_of::<T>()
            )
        };
        
        self.mmap[byte_offset..byte_offset + mem::size_of::<T>()].copy_from_slice(bytes);
        
        Ok(())
    }
    
    /// Calculate the linear offset for the given indices
    fn calculate_offset(&self, indices: &[usize]) -> Result<usize> {
        // Check if indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(NumRs2Error::IndexOutOfBounds(
                    format!("Index {} out of bounds for dimension {}: {}",
                            idx, i, self.shape[i])
                ));
            }
        }
        
        // Calculate linear offset
        let mut offset = 0;
        let mut stride = 1;
        
        for i in (0..indices.len()).rev() {
            offset += indices[i] * stride;
            stride *= self.shape[i];
        }
        
        Ok(offset)
    }
    
    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the path to the backing file
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }
    
    /// Convert to a regular Array
    ///
    /// This copies all data from the memory map into a new Array.
    pub fn to_array(&self) -> Result<Array<T>> {
        let mut data = Vec::with_capacity(self.size);
        
        // Iterate through all elements and copy them
        let meta_size = calculate_meta_size(&self.shape);
        let data_start = meta_size;
        let _data_end = meta_size + self.size * mem::size_of::<T>();
        
        // Read all elements from the memory map
        for i in 0..self.size {
            let byte_offset = data_start + i * mem::size_of::<T>();
            let bytes = &self.mmap[byte_offset..byte_offset + mem::size_of::<T>()];
            let value = unsafe { *(bytes.as_ptr() as *const T) };
            data.push(value);
        }
        
        // Create a new Array from the data
        let array = Array::from_vec(data).reshape(&self.shape);
        Ok(array)
    }
    
    /// Create a memory-mapped array from a regular Array
    ///
    /// # Arguments
    ///
    /// * `array` - The array to convert
    /// * `path` - Path to the file to back the array
    ///
    /// # Returns
    ///
    /// A new MmapArray instance
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or if the conversion fails.
    pub fn from_array<P: AsRef<Path>>(array: &Array<T>, path: &P) -> Result<Self> {
        let shape = array.shape();
        let mut mmap_array = Self::new(path, &shape, true)?;
        
        // Get the data from the array
        let data = array.to_vec();
        
        // Copy data to the memory map
        let meta_size = calculate_meta_size(&shape);
        let data_start = meta_size;
        
        for (i, &value) in data.iter().enumerate() {
            let byte_offset = data_start + i * mem::size_of::<T>();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &value as *const T as *const u8,
                    mem::size_of::<T>()
                )
            };
            
            mmap_array.mmap[byte_offset..byte_offset + mem::size_of::<T>()].copy_from_slice(bytes);
        }
        
        // Flush changes to disk
        mmap_array.flush()?;
        
        Ok(mmap_array)
    }
}

impl<T: Copy + fmt::Debug> fmt::Display for MmapArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "MmapArray(shape={:?}, file={})", 
                self.shape, self.path.display())?;
        
        // Display a small preview of the data
        const MAX_ELEMENTS: usize = 10;
        let preview_size = std::cmp::min(self.size, MAX_ELEMENTS);
        
        write!(f, "Data preview: [")?;
        
        for i in 0..preview_size {
            // Create indices for the i-th element
            let mut indices = Vec::with_capacity(self.ndim());
            let mut remaining = i;
            
            for &dim in self.shape.iter().rev() {
                indices.insert(0, remaining % dim);
                remaining /= dim;
            }
            
            if i > 0 {
                write!(f, ", ")?;
            }
            
            match self.get(&indices) {
                Ok(value) => write!(f, "{:?}", value)?,
                Err(_) => write!(f, "<?>")?
            }
        }
        
        if self.size > MAX_ELEMENTS {
            write!(f, ", ...]")?;
        } else {
            write!(f, "]")?;
        }
        
        Ok(())
    }
}

// Helper function to calculate metadata size
fn calculate_meta_size(_shape: &[usize]) -> usize {
    // Use a fixed size for metadata to make it simpler
    // In a real implementation, you might want to use a more sophisticated approach
    1024 // 1KB for metadata
}

/// Open an existing memory-mapped array file
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// The metadata for the array
///
/// # Errors
///
/// Returns an error if the file cannot be opened or if the metadata cannot be read.
pub fn open_mmap_info<P: AsRef<Path>>(path: &P) -> Result<MmapArrayMeta> {
    let mut file = File::open(path)?;
    
    // Read metadata (fixed size)
    let meta_size = 1024; // Same as calculate_meta_size
    let mut meta_bytes = vec![0u8; meta_size];
    file.read_exact(&mut meta_bytes)?;
    
    let meta: MmapArrayMeta = bincode::deserialize(&meta_bytes)?;
    
    Ok(meta)
}
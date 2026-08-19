// Memory-mapped array module for NumRS2
// Provides memory-mapped array functionality for efficient file-backed arrays

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
// Alignment utilities available if needed
use crate::memory_optimize::cache_layout::{optimize_layout, LayoutStrategy};
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Configuration for memory-mapped arrays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapConfig {
    /// Cache optimization strategy
    pub layout_strategy: LayoutStrategy,
    /// Whether to use write-back caching
    pub write_back: bool,
    /// Prefetch strategy
    pub prefetch: PrefetchStrategy,
    /// Alignment requirements
    pub alignment: usize,
    /// Page size hint
    pub page_size_hint: Option<usize>,
}

/// Prefetch strategy for memory-mapped arrays
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    None,
    Sequential,
    Random,
    Adaptive,
}

/// Access pattern information for optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AccessPattern {
    /// Recent access indices
    recent_accesses: Vec<Vec<usize>>,
    /// Access frequency counter
    access_count: HashMap<Vec<usize>, u64>,
    /// Last access time
    last_access: SystemTime,
    /// Detected pattern type
    pattern_type: AccessPatternType,
}

/// Types of detected access patterns
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum AccessPatternType {
    Unknown,
    Sequential,
    Strided,
    Random,
    Blocked,
}

lazy_static::lazy_static! {
    static ref GLOBAL_MMAP_CACHE: Mutex<HashMap<PathBuf, Arc<Mutex<AccessPattern>>>> =
        Mutex::new(HashMap::new());
}

/// Memory-mapped array with file backing
///
/// A memory-mapped array allows you to work with data that is stored in a file
/// as if it were in memory. This is particularly useful for large arrays that
/// might not fit in RAM. This implementation includes advanced optimizations
/// for cache efficiency and access pattern detection.
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
    /// Configuration for optimization
    config: MmapConfig,
    /// Access pattern tracking
    access_pattern: Arc<Mutex<AccessPattern>>,
    /// Data offset in the file (after metadata)
    data_offset: usize,
    /// Page size for optimal I/O
    #[allow(dead_code)]
    page_size: usize,
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
    /// Configuration used when creating the array
    pub config: Option<MmapConfig>,
    /// Checksum for data integrity
    pub checksum: Option<u64>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
}

/// Verifies that `meta` (persisted metadata read back from an mmap-backed
/// array file) describes exactly the type `T` the caller is about to
/// reinterpret the file's raw bytes as.
///
/// Comparing `type_name` alone is not sufficient: `std::any::type_name`'s
/// documentation explicitly disclaims being a stable or unique identifier
/// of a type. Also comparing `type_size` closes the "no size check" gap
/// that would otherwise leave: a metadata record whose name happened to
/// (mis)match but whose size did not would let the raw-byte reads in
/// [`MmapArray::get`]/[`MmapArray::set`] read past the end of a value of a
/// different, mismatched type. `TypeId` is not used here (nor could it
/// be persisted): it requires `T: 'static`, which would ripple out through
/// every generic caller of [`MmapArray`], and its value is not guaranteed
/// stable across separate compilations anyway -- unlike `type_name`, which
/// is exactly the kind of "same process family, self-consistency" check
/// this on-disk format needs.
pub(crate) fn check_meta_type<T>(meta: &MmapArrayMeta) -> Result<()> {
    let expected_name = std::any::type_name::<T>();
    let expected_size = mem::size_of::<T>();

    if meta.type_name != expected_name {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Type mismatch: file contains '{}', but requested '{}'",
            meta.type_name, expected_name
        )));
    }

    if meta.type_size != expected_size {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Type size mismatch: file contains type '{}' ({} bytes), but requested '{}' ({} bytes)",
            meta.type_name, meta.type_size, expected_name, expected_size
        )));
    }

    Ok(())
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            layout_strategy: LayoutStrategy::RowMajor,
            write_back: true,
            prefetch: PrefetchStrategy::Adaptive,
            alignment: 0, // Will be calculated based on type
            page_size_hint: None,
        }
    }
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self {
            recent_accesses: Vec::with_capacity(100),
            access_count: HashMap::new(),
            last_access: SystemTime::now(),
            pattern_type: AccessPatternType::Unknown,
        }
    }
}

#[allow(dead_code)]
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
                config: None,
                checksum: None,
                created_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                modified_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let config = oxicode::config::standard();
            let meta_bytes = oxicode::serde::encode_to_vec(&meta, config)
                .map_err(|e| NumRs2Error::SerializationError(e.to_string()))?;
            let mut file = file;
            file.write_all(&meta_bytes)?;

            file
        } else {
            // Open existing file
            let mut file = File::open(&path)?;

            // Read metadata
            let mut meta_bytes = vec![0u8; meta_size];
            file.read_exact(&mut meta_bytes)?;

            let config = oxicode::config::standard();
            let (meta, _): (MmapArrayMeta, usize) =
                oxicode::serde::decode_from_slice(&meta_bytes, config)
                    .map_err(|e| NumRs2Error::DeserializationError(e.to_string()))?;

            // Verify metadata
            check_meta_type::<T>(&meta)?;

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
            return Err(NumRs2Error::InvalidOperation(format!(
                "File size mismatch: expected {} bytes, got {} bytes",
                total_size,
                mmap.len()
            )));
        }

        let access_pattern = get_or_create_access_pattern(&path);
        let config = MmapConfig::default();
        let page_size = get_page_size();
        let data_offset = meta_size;

        Ok(Self {
            mmap,
            shape: shape.to_vec(),
            size,
            path,
            config,
            access_pattern,
            data_offset,
            page_size,
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
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }

        let offset = self.calculate_offset(indices)?;
        let byte_offset = self.data_offset + offset * mem::size_of::<T>();

        if byte_offset + mem::size_of::<T>() > self.mmap.len() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index out of bounds: offset {} exceeds mmap size {}",
                byte_offset,
                self.mmap.len()
            )));
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
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }

        let offset = self.calculate_offset(indices)?;
        let byte_offset = self.data_offset + offset * mem::size_of::<T>();

        if byte_offset + mem::size_of::<T>() > self.mmap.len() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Index out of bounds: offset {} exceeds mmap size {}",
                byte_offset,
                self.mmap.len()
            )));
        }

        // Write value to memory map
        let bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, mem::size_of::<T>())
        };

        self.mmap[byte_offset..byte_offset + mem::size_of::<T>()].copy_from_slice(bytes);

        Ok(())
    }

    /// Calculate the linear offset for the given indices
    fn calculate_offset(&self, indices: &[usize]) -> Result<usize> {
        // Check if indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} out of bounds for dimension {}: {}",
                    idx, i, self.shape[i]
                )));
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
                std::slice::from_raw_parts(&value as *const T as *const u8, mem::size_of::<T>())
            };

            mmap_array.mmap[byte_offset..byte_offset + mem::size_of::<T>()].copy_from_slice(bytes);
        }

        // Flush changes to disk
        mmap_array.flush()?;

        Ok(mmap_array)
    }

    /// Track access patterns for optimization
    fn track_access(&self, indices: &[usize]) {
        if let Ok(mut pattern) = self.access_pattern.lock() {
            // Update access pattern
            pattern.last_access = SystemTime::now();

            // Store recent access
            if pattern.recent_accesses.len() >= 100 {
                pattern.recent_accesses.remove(0);
            }
            pattern.recent_accesses.push(indices.to_vec());

            // Update access count
            *pattern.access_count.entry(indices.to_vec()).or_insert(0) += 1;

            // Detect access pattern type
            pattern.pattern_type = self.detect_access_pattern(&pattern.recent_accesses);
        }
    }

    /// Detect the type of access pattern
    fn detect_access_pattern(&self, accesses: &[Vec<usize>]) -> AccessPatternType {
        if accesses.len() < 3 {
            return AccessPatternType::Unknown;
        }

        // Check for sequential access
        let mut sequential = true;
        let mut stride = None;

        for i in 1..accesses.len() {
            if accesses[i].len() != accesses[i - 1].len() {
                sequential = false;
                break;
            }

            // Calculate stride for last dimension
            let last_dim = accesses[i].len() - 1;
            let current_stride = accesses[i][last_dim] as i64 - accesses[i - 1][last_dim] as i64;

            if let Some(expected_stride) = stride {
                if current_stride != expected_stride {
                    sequential = false;
                    break;
                }
            } else {
                stride = Some(current_stride);
            }
        }

        if sequential {
            if stride == Some(1) {
                return AccessPatternType::Sequential;
            } else if stride.is_some() {
                return AccessPatternType::Strided;
            }
        }

        // Check for blocked access pattern
        // (Implementation simplified for brevity)

        AccessPatternType::Random
    }

    /// Apply prefetching based on detected access pattern
    fn prefetch_if_needed(&self, indices: &[usize]) -> Result<()> {
        if self.config.prefetch == PrefetchStrategy::None {
            return Ok(());
        }

        let pattern = self
            .access_pattern
            .lock()
            .expect("Access pattern mutex poisoned");

        match (self.config.prefetch, pattern.pattern_type) {
            (PrefetchStrategy::Sequential, AccessPatternType::Sequential)
            | (PrefetchStrategy::Adaptive, AccessPatternType::Sequential) => {
                self.prefetch_sequential(indices)?;
            }
            (PrefetchStrategy::Adaptive, AccessPatternType::Strided) => {
                self.prefetch_strided(indices)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Prefetch data for sequential access pattern
    fn prefetch_sequential(&self, indices: &[usize]) -> Result<()> {
        const PREFETCH_SIZE: usize = 8; // Prefetch 8 elements ahead

        // Calculate next indices for prefetching
        let mut next_indices = indices.to_vec();
        let last_dim = next_indices.len() - 1;

        for i in 1..=PREFETCH_SIZE {
            if next_indices[last_dim] + i < self.shape[last_dim] {
                next_indices[last_dim] += 1;
                let offset = self.calculate_offset(&next_indices)?;
                let byte_offset = self.data_offset + offset * mem::size_of::<T>();

                // Touch the memory to trigger prefetch
                if byte_offset + mem::size_of::<T>() <= self.mmap.len() {
                    let _ = self.mmap[byte_offset];
                }
            }
        }

        Ok(())
    }

    /// Prefetch data for strided access pattern
    fn prefetch_strided(&self, _indices: &[usize]) -> Result<()> {
        // Implementation for strided prefetching
        // (Simplified for brevity)
        Ok(())
    }

    /// Optimize the memory layout of the data
    fn optimize_layout(&mut self) -> Result<()> {
        if self.config.layout_strategy == LayoutStrategy::RowMajor {
            return Ok(()); // Already in optimal layout
        }

        // Get current data
        let data = self.get_all_data()?;

        // Apply layout optimization
        let mut optimized_data = data;
        optimize_layout(&mut optimized_data, self.config.layout_strategy);

        // Write back optimized data
        self.set_all_data(&optimized_data)?;

        Ok(())
    }

    /// Get all data as a flat vector
    fn get_all_data(&self) -> Result<Vec<T>> {
        let mut data = Vec::with_capacity(self.size);

        let data_start = self.data_offset;
        let element_size = mem::size_of::<T>();

        for i in 0..self.size {
            let byte_offset = data_start + i * element_size;
            if byte_offset + element_size <= self.mmap.len() {
                let bytes = &self.mmap[byte_offset..byte_offset + element_size];
                let value = unsafe { *(bytes.as_ptr() as *const T) };
                data.push(value);
            }
        }

        Ok(data)
    }

    /// Set all data from a flat vector
    fn set_all_data(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.size {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Data size mismatch: expected {}, got {}",
                self.size,
                data.len()
            )));
        }

        let data_start = self.data_offset;
        let element_size = mem::size_of::<T>();

        for (i, &value) in data.iter().enumerate() {
            let byte_offset = data_start + i * element_size;
            if byte_offset + element_size <= self.mmap.len() {
                let bytes = unsafe {
                    std::slice::from_raw_parts(&value as *const T as *const u8, element_size)
                };
                self.mmap[byte_offset..byte_offset + element_size].copy_from_slice(bytes);
            }
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &MmapConfig {
        &self.config
    }

    /// Update configuration (affects future operations)
    pub fn update_config(&mut self, config: MmapConfig) {
        self.config = config;
    }

    /// Get access pattern statistics
    pub fn access_stats(&self) -> Option<(AccessPatternType, usize)> {
        if let Ok(pattern) = self.access_pattern.lock() {
            Some((pattern.pattern_type, pattern.recent_accesses.len()))
        } else {
            None
        }
    }
}

impl<T: Copy + fmt::Debug> fmt::Display for MmapArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "MmapArray(shape={:?}, file={})",
            self.shape,
            self.path.display()
        )?;

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
                Err(_) => write!(f, "<?>")?,
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

/// Get system page size
fn get_page_size() -> usize {
    // Default page size on most systems
    4096
}

/// Align size to page boundary
#[allow(dead_code)]
fn align_to_page(size: usize, page_size: usize) -> usize {
    (size + page_size - 1) & !(page_size - 1)
}

/// Apply memory advice for optimization
#[allow(dead_code)]
fn apply_memory_advice(_mmap: &mut MmapMut, _config: &MmapConfig) {
    // Memory advice is platform-specific and not always available
    // For now, we'll skip this optimization
    // In a full implementation, you would use platform-specific calls
    #[cfg(unix)]
    {
        // On Unix systems, we could use madvise() directly
        // let _ = mmap.advise(advice);
    }
}

/// Get or create access pattern tracking for a file
fn get_or_create_access_pattern(path: &Path) -> Arc<Mutex<AccessPattern>> {
    let mut cache = GLOBAL_MMAP_CACHE
        .lock()
        .expect("Global mmap cache mutex poisoned");

    cache
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(AccessPattern::default())))
        .clone()
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

    let config = oxicode::config::standard();
    let (meta, _): (MmapArrayMeta, usize) = oxicode::serde::decode_from_slice(&meta_bytes, config)
        .map_err(|e| NumRs2Error::DeserializationError(e.to_string()))?;

    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta(type_name: &str, type_size: usize) -> MmapArrayMeta {
        MmapArrayMeta {
            type_name: type_name.to_string(),
            type_size,
            shape: vec![4],
            size: 4,
            version: 1,
            config: None,
            checksum: None,
            created_at: 0,
            modified_at: 0,
        }
    }

    #[test]
    fn test_check_meta_type_accepts_matching_type() {
        let meta = sample_meta(std::any::type_name::<f64>(), mem::size_of::<f64>());
        assert!(check_meta_type::<f64>(&meta).is_ok());
    }

    #[test]
    fn test_check_meta_type_rejects_name_mismatch() {
        let meta = sample_meta(std::any::type_name::<f32>(), mem::size_of::<f32>());
        assert!(check_meta_type::<f64>(&meta).is_err());
    }

    #[test]
    fn test_check_meta_type_rejects_size_mismatch_with_matching_name() {
        // A metadata record whose *name* matches the requested type but
        // whose *size* was corrupted to disagree (as could happen with a
        // foreign/legacy file, or if two distinct types were ever to
        // produce the same `type_name`). This is the "no size check" gap
        // the name-only comparison used to leave open: without checking
        // `type_size` too, this would be accepted and the raw-byte reads
        // in `get`/`set` would read past the end of an 4-byte value while
        // believing it is 8 bytes.
        let meta = sample_meta(std::any::type_name::<f64>(), 4);
        assert!(check_meta_type::<f64>(&meta).is_err());
    }

    #[test]
    fn test_mmap_array_reopen_with_wrong_type_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "numrs2_mmap_type_mismatch_{}.tmp",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        {
            let _created =
                MmapArray::<f64>::new(&path, &[3], true).expect("failed to create mmap file");
        }

        let reopened = MmapArray::<i32>::new(&path, &[3], false);
        assert!(
            reopened.is_err(),
            "reopening an f64-typed mmap file as i32 must fail"
        );

        let _ = std::fs::remove_file(&path);
    }
}

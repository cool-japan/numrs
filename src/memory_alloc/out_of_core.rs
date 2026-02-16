//! Out-of-core array implementation for very large datasets
//!
//! This module provides array types that can handle datasets larger than available
//! memory by automatically managing data spilling to disk and loading chunks on demand.

#[allow(unused_imports)]
use super::large_scale::{LargeScaleConfig, LargeScaleManager, MemoryTracker};
use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[allow(unused_imports)]
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// Configuration for out-of-core arrays
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum number of chunks to keep in memory
    pub max_chunks_in_memory: usize,
    /// Size of each chunk in elements
    pub chunk_size: usize,
    /// Base path for storing chunks on disk
    pub storage_path: PathBuf,
    /// Whether to use compression for stored chunks
    pub use_compression: bool,
    /// Cache replacement strategy
    pub cache_strategy: CacheStrategy,
    /// Enable prefetching of adjacent chunks
    pub enable_prefetch: bool,
    /// Number of chunks to prefetch
    pub prefetch_count: usize,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_chunks_in_memory: 16, // Keep up to 16 chunks in memory
            chunk_size: 1024 * 1024,  // 1M elements per chunk
            storage_path: std::env::temp_dir().join("numrs_ooc"),
            use_compression: false, // Disabled for now for simplicity
            cache_strategy: CacheStrategy::LRU,
            enable_prefetch: true,
            prefetch_count: 2,
        }
    }
}

/// Cache replacement strategies for in-memory chunks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
}

/// Metadata for a chunk stored on disk or in memory
#[derive(Debug, Clone)]
struct ChunkMetadata {
    /// Index of this chunk
    chunk_index: usize,
    /// Number of elements in this chunk
    element_count: usize,
    /// Path to the chunk file on disk (if spilled)
    disk_path: Option<PathBuf>,
    /// Whether the chunk is currently in memory
    in_memory: bool,
    /// Access count for LFU strategy
    access_count: u64,
    /// Last access timestamp for LRU strategy
    last_access: u64,
    /// Whether the chunk has been modified since last disk sync
    dirty: bool,
}

/// A chunk of data that can be either in memory or on disk
#[derive(Debug)]
struct DataChunk<T: Copy> {
    /// The actual data (None if spilled to disk)
    data: Option<Vec<T>>,
    /// Metadata for this chunk
    metadata: ChunkMetadata,
}

#[allow(dead_code)]
impl<T: Copy> DataChunk<T> {
    /// Create a new in-memory chunk
    fn new_in_memory(chunk_index: usize, data: Vec<T>) -> Self {
        let element_count = data.len();
        Self {
            data: Some(data),
            metadata: ChunkMetadata {
                chunk_index,
                element_count,
                disk_path: None,
                in_memory: true,
                access_count: 0,
                last_access: current_timestamp(),
                dirty: true,
            },
        }
    }

    /// Create a new chunk from disk
    fn new_from_disk(chunk_index: usize, element_count: usize, disk_path: PathBuf) -> Self {
        Self {
            data: None,
            metadata: ChunkMetadata {
                chunk_index,
                element_count,
                disk_path: Some(disk_path),
                in_memory: false,
                access_count: 0,
                last_access: current_timestamp(),
                dirty: false,
            },
        }
    }

    /// Load chunk data into memory from disk
    fn load_from_disk(&mut self) -> Result<()> {
        if self.data.is_some() {
            return Ok(()); // Already in memory
        }

        let disk_path =
            self.metadata.disk_path.as_ref().ok_or_else(|| {
                NumRs2Error::InvalidOperation("No disk path for chunk".to_string())
            })?;

        let mut file = File::open(disk_path)?;
        let mut buffer = vec![0u8; self.metadata.element_count * std::mem::size_of::<T>()];
        file.read_exact(&mut buffer)?;

        // Convert bytes back to T
        let data = unsafe {
            let ptr = buffer.as_ptr() as *const T;
            std::slice::from_raw_parts(ptr, self.metadata.element_count).to_vec()
        };

        self.data = Some(data);
        self.metadata.in_memory = true;
        self.update_access();

        Ok(())
    }

    /// Spill chunk data to disk
    fn spill_to_disk(&mut self, storage_path: &Path) -> Result<()> {
        if self.data.is_none() {
            return Ok(()); // Already spilled
        }

        let chunk_path = storage_path.join(format!("chunk_{}.bin", self.metadata.chunk_index));

        // Ensure directory exists
        if let Some(parent) = chunk_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&chunk_path)?;

        if let Some(ref data) = self.data {
            // Convert T to bytes
            let bytes = unsafe {
                let ptr = data.as_ptr() as *const u8;
                std::slice::from_raw_parts(ptr, data.len() * std::mem::size_of::<T>())
            };
            file.write_all(bytes)?;
            file.sync_all()?;
        }

        self.data = None;
        self.metadata.disk_path = Some(chunk_path);
        self.metadata.in_memory = false;
        self.metadata.dirty = false;

        Ok(())
    }

    /// Get a reference to the data, loading from disk if necessary
    fn get_data(&mut self) -> Result<&Vec<T>> {
        if self.data.is_none() {
            self.load_from_disk()?;
        }
        self.update_access();
        Ok(self
            .data
            .as_ref()
            .expect("data should be loaded at this point"))
    }

    /// Get a mutable reference to the data, loading from disk if necessary
    fn get_data_mut(&mut self) -> Result<&mut Vec<T>> {
        if self.data.is_none() {
            self.load_from_disk()?;
        }
        self.update_access();
        self.metadata.dirty = true;
        Ok(self
            .data
            .as_mut()
            .expect("data should be loaded at this point"))
    }

    /// Update access statistics
    fn update_access(&mut self) {
        self.metadata.access_count += 1;
        self.metadata.last_access = current_timestamp();
    }

    /// Check if chunk needs to be written to disk
    fn is_dirty(&self) -> bool {
        self.metadata.dirty
    }

    /// Sync chunk to disk if dirty
    fn sync_if_dirty(&mut self, storage_path: &Path) -> Result<()> {
        if self.is_dirty() && self.data.is_some() {
            self.spill_to_disk(storage_path)?;
        }
        Ok(())
    }
}

/// Out-of-core array that can handle datasets larger than memory
pub struct OutOfCoreArray<T: Copy + Send + Sync + Default + 'static> {
    /// Configuration
    config: OutOfCoreConfig,
    /// Array shape
    shape: Vec<usize>,
    /// Total number of elements
    total_elements: usize,
    /// Chunks of data
    chunks: Arc<RwLock<HashMap<usize, DataChunk<T>>>>,
    /// Cache management
    cache_manager: Arc<Mutex<CacheManager>>,
    /// Memory tracker for monitoring usage
    memory_tracker: Arc<MemoryTracker>,
    /// Unique identifier for this array
    #[allow(dead_code)]
    array_id: String,
    /// Phantom data for T
    _phantom: PhantomData<T>,
}

/// Cache manager for tracking which chunks are in memory
#[derive(Debug)]
struct CacheManager {
    /// LRU queue for cache eviction
    lru_queue: std::collections::VecDeque<usize>,
    /// Access frequency for LFU strategy
    access_frequency: HashMap<usize, u64>,
    /// FIFO queue for FIFO strategy
    fifo_queue: std::collections::VecDeque<usize>,
    /// Number of chunks currently in memory
    chunks_in_memory: usize,
}

impl CacheManager {
    fn new() -> Self {
        Self {
            lru_queue: std::collections::VecDeque::new(),
            access_frequency: HashMap::new(),
            fifo_queue: std::collections::VecDeque::new(),
            chunks_in_memory: 0,
        }
    }

    /// Record access to a chunk
    fn record_access(&mut self, chunk_index: usize, strategy: CacheStrategy) {
        match strategy {
            CacheStrategy::LRU => {
                // Move to front of LRU queue
                self.lru_queue.retain(|&x| x != chunk_index);
                self.lru_queue.push_front(chunk_index);
            }
            CacheStrategy::LFU => {
                *self.access_frequency.entry(chunk_index).or_insert(0) += 1;
            }
            CacheStrategy::FIFO => {
                if !self.fifo_queue.contains(&chunk_index) {
                    self.fifo_queue.push_back(chunk_index);
                }
            }
        }
    }

    /// Get chunk to evict based on strategy
    fn get_eviction_candidate(&mut self, strategy: CacheStrategy) -> Option<usize> {
        match strategy {
            CacheStrategy::LRU => self.lru_queue.pop_back(),
            CacheStrategy::LFU => {
                let min_chunk = self
                    .access_frequency
                    .iter()
                    .min_by_key(|(_, &freq)| freq)
                    .map(|(&chunk, _)| chunk);
                if let Some(chunk) = min_chunk {
                    self.access_frequency.remove(&chunk);
                }
                min_chunk
            }
            CacheStrategy::FIFO => self.fifo_queue.pop_front(),
        }
    }

    /// Record that a chunk is now in memory
    fn chunk_loaded(&mut self, chunk_index: usize, strategy: CacheStrategy) {
        self.chunks_in_memory += 1;

        // Track the chunk for eviction according to the cache strategy
        match strategy {
            CacheStrategy::LRU => {
                // Remove if already exists, then add to front
                self.lru_queue.retain(|&x| x != chunk_index);
                self.lru_queue.push_front(chunk_index);
            }
            CacheStrategy::LFU => {
                // Increment access frequency
                *self.access_frequency.entry(chunk_index).or_insert(0) += 1;
            }
            CacheStrategy::FIFO => {
                // Add to the back of the queue if not already present
                if !self.fifo_queue.contains(&chunk_index) {
                    self.fifo_queue.push_back(chunk_index);
                }
            }
        }
    }

    /// Record that a chunk is no longer in memory
    fn chunk_evicted(&mut self, chunk_index: usize, strategy: CacheStrategy) {
        self.chunks_in_memory = self.chunks_in_memory.saturating_sub(1);

        match strategy {
            CacheStrategy::LRU => {
                self.lru_queue.retain(|&x| x != chunk_index);
            }
            CacheStrategy::LFU => {
                self.access_frequency.remove(&chunk_index);
            }
            CacheStrategy::FIFO => {
                self.fifo_queue.retain(|&x| x != chunk_index);
            }
        }
    }
}

impl<T: Copy + Send + Sync + Default + 'static> OutOfCoreArray<T> {
    /// Create a new out-of-core array
    pub fn new(shape: Vec<usize>, config: OutOfCoreConfig) -> Result<Self> {
        let total_elements: usize = shape.iter().product();

        // Create storage directory
        std::fs::create_dir_all(&config.storage_path)?;

        let array_id = format!("ooc_array_{}", current_timestamp());

        Ok(Self {
            config,
            shape,
            total_elements,
            chunks: Arc::new(RwLock::new(HashMap::new())),
            cache_manager: Arc::new(Mutex::new(CacheManager::new())),
            memory_tracker: Arc::new(MemoryTracker::new()),
            array_id,
            _phantom: PhantomData,
        })
    }

    /// Create an out-of-core array from existing data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>, config: OutOfCoreConfig) -> Result<Self> {
        let array = Self::new(shape, config)?;

        // Split data into chunks
        let chunk_size = array.config.chunk_size;
        for (chunk_index, chunk_data) in data.chunks(chunk_size).enumerate() {
            // Check if we need to evict chunks first
            array.evict_chunks_if_needed()?;

            let chunk = DataChunk::new_in_memory(chunk_index, chunk_data.to_vec());
            let memory_usage = chunk.metadata.element_count * std::mem::size_of::<T>();
            array
                .chunks
                .write()
                .expect("chunks RwLock should not be poisoned")
                .insert(chunk_index, chunk);
            array
                .cache_manager
                .lock()
                .expect("cache_manager mutex should not be poisoned")
                .chunk_loaded(chunk_index, array.config.cache_strategy);
            array.memory_tracker.record_allocation(memory_usage);
        }

        Ok(array)
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.total_elements
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.total_elements == 0
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        self.total_elements.div_ceil(self.config.chunk_size)
    }

    /// Calculate which chunk contains the given linear index
    fn get_chunk_index(&self, linear_index: usize) -> usize {
        linear_index / self.config.chunk_size
    }

    /// Calculate the offset within a chunk for the given linear index
    fn get_chunk_offset(&self, linear_index: usize) -> usize {
        linear_index % self.config.chunk_size
    }

    /// Convert multidimensional indices to linear index
    fn indices_to_linear(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }

        let mut linear_index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            if indices[i] >= self.shape[i] {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Index {} out of bounds for dimension {} (size {})",
                    indices[i], i, self.shape[i]
                )));
            }
            linear_index += indices[i] * stride;
            stride *= self.shape[i];
        }

        Ok(linear_index)
    }

    /// Get element at the specified indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        let linear_index = self.indices_to_linear(indices)?;
        self.get_linear(linear_index)
    }

    /// Get element at the specified linear index
    pub fn get_linear(&self, linear_index: usize) -> Result<T> {
        if linear_index >= self.total_elements {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Linear index {} out of bounds (size {})",
                linear_index, self.total_elements
            )));
        }

        let chunk_index = self.get_chunk_index(linear_index);
        let chunk_offset = self.get_chunk_offset(linear_index);

        // Ensure chunk is loaded
        self.ensure_chunk_loaded(chunk_index)?;

        // Access the data
        let chunks = self
            .chunks
            .read()
            .expect("chunks RwLock should not be poisoned");
        let chunk = chunks
            .get(&chunk_index)
            .ok_or_else(|| NumRs2Error::InvalidOperation("Chunk not found".to_string()))?;

        let data = chunk
            .data
            .as_ref()
            .ok_or_else(|| NumRs2Error::InvalidOperation("Chunk data not in memory".to_string()))?;

        if chunk_offset >= data.len() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Chunk offset {} out of bounds (chunk size {})",
                chunk_offset,
                data.len()
            )));
        }

        // Record access
        let mut cache_manager = self
            .cache_manager
            .lock()
            .expect("cache_manager mutex should not be poisoned");
        cache_manager.record_access(chunk_index, self.config.cache_strategy);

        Ok(data[chunk_offset])
    }

    /// Set element at the specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        let linear_index = self.indices_to_linear(indices)?;
        self.set_linear(linear_index, value)
    }

    /// Set element at the specified linear index
    pub fn set_linear(&mut self, linear_index: usize, value: T) -> Result<()> {
        if linear_index >= self.total_elements {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Linear index {} out of bounds (size {})",
                linear_index, self.total_elements
            )));
        }

        let chunk_index = self.get_chunk_index(linear_index);
        let chunk_offset = self.get_chunk_offset(linear_index);

        // Ensure chunk is loaded
        self.ensure_chunk_loaded(chunk_index)?;

        // Modify the data
        {
            let mut chunks = self
                .chunks
                .write()
                .expect("chunks RwLock should not be poisoned");
            let chunk = chunks
                .get_mut(&chunk_index)
                .ok_or_else(|| NumRs2Error::InvalidOperation("Chunk not found".to_string()))?;

            let data = chunk.get_data_mut()?;

            if chunk_offset >= data.len() {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Chunk offset {} out of bounds (chunk size {})",
                    chunk_offset,
                    data.len()
                )));
            }

            data[chunk_offset] = value;
        }

        // Record access
        let mut cache_manager = self
            .cache_manager
            .lock()
            .expect("cache_manager mutex should not be poisoned");
        cache_manager.record_access(chunk_index, self.config.cache_strategy);

        Ok(())
    }

    /// Ensure a chunk is loaded in memory, evicting others if necessary
    fn ensure_chunk_loaded(&self, chunk_index: usize) -> Result<()> {
        // Check if chunk is already in memory
        {
            let chunks = self
                .chunks
                .read()
                .expect("chunks RwLock should not be poisoned");
            if let Some(chunk) = chunks.get(&chunk_index) {
                if chunk.metadata.in_memory {
                    // Mark as recently accessed to prevent immediate eviction
                    let mut cache_manager = self
                        .cache_manager
                        .lock()
                        .expect("cache_manager mutex should not be poisoned");
                    cache_manager.record_access(chunk_index, self.config.cache_strategy);
                    return Ok(());
                }
            }
        }

        // Check if we need to evict chunks first, but protect the chunk we're loading
        self.evict_chunks_if_needed_protected(Some(chunk_index))?;

        // Load the chunk
        {
            let mut chunks = self
                .chunks
                .write()
                .expect("chunks RwLock should not be poisoned");
            if let Some(chunk) = chunks.get_mut(&chunk_index) {
                if !chunk.metadata.in_memory {
                    chunk.load_from_disk()?;
                    self.cache_manager
                        .lock()
                        .expect("cache_manager mutex should not be poisoned")
                        .chunk_loaded(chunk_index, self.config.cache_strategy);

                    // Track memory usage
                    let memory_usage = chunk.metadata.element_count * std::mem::size_of::<T>();
                    self.memory_tracker.record_allocation(memory_usage);
                }
            } else {
                // Create a new empty chunk if it doesn't exist
                let chunk_data = vec![
                    T::default();
                    std::cmp::min(
                        self.config.chunk_size,
                        self.total_elements - chunk_index * self.config.chunk_size
                    )
                ];
                let chunk = DataChunk::new_in_memory(chunk_index, chunk_data);
                let memory_usage = chunk.metadata.element_count * std::mem::size_of::<T>();
                chunks.insert(chunk_index, chunk);
                self.cache_manager
                    .lock()
                    .expect("cache_manager mutex should not be poisoned")
                    .chunk_loaded(chunk_index, self.config.cache_strategy);
                self.memory_tracker.record_allocation(memory_usage);
            }
        }

        // Mark the chunk as recently accessed to prevent immediate eviction
        {
            let mut cache_manager = self
                .cache_manager
                .lock()
                .expect("cache_manager mutex should not be poisoned");
            cache_manager.record_access(chunk_index, self.config.cache_strategy);
        }

        // Prefetch adjacent chunks if enabled
        if self.config.enable_prefetch {
            self.prefetch_adjacent_chunks(chunk_index);
        }

        Ok(())
    }

    /// Evict chunks from memory if we exceed the cache limit
    fn evict_chunks_if_needed(&self) -> Result<()> {
        self.evict_chunks_if_needed_protected(None)
    }

    /// Evict chunks from memory if we exceed the cache limit, protecting a specific chunk
    fn evict_chunks_if_needed_protected(&self, protected_chunk: Option<usize>) -> Result<()> {
        let mut cache_manager = self
            .cache_manager
            .lock()
            .expect("cache_manager mutex should not be poisoned");

        while cache_manager.chunks_in_memory >= self.config.max_chunks_in_memory {
            if let Some(evict_chunk_index) =
                cache_manager.get_eviction_candidate(self.config.cache_strategy)
            {
                // Skip protected chunk
                if let Some(protected) = protected_chunk {
                    if evict_chunk_index == protected {
                        break; // Can't evict the protected chunk, stop trying
                    }
                }
                // Spill the chunk to disk
                {
                    let mut chunks = self
                        .chunks
                        .write()
                        .expect("chunks RwLock should not be poisoned");
                    if let Some(chunk) = chunks.get_mut(&evict_chunk_index) {
                        if chunk.metadata.in_memory {
                            chunk.sync_if_dirty(&self.config.storage_path)?;

                            if chunk.data.is_some() {
                                let memory_usage =
                                    chunk.metadata.element_count * std::mem::size_of::<T>();
                                self.memory_tracker.record_deallocation(memory_usage);
                            }

                            chunk.data = None;
                            chunk.metadata.in_memory = false;
                        }
                    }
                }

                cache_manager.chunk_evicted(evict_chunk_index, self.config.cache_strategy);
            } else {
                break; // No chunks to evict
            }
        }

        Ok(())
    }

    /// Prefetch adjacent chunks in the background
    fn prefetch_adjacent_chunks(&self, current_chunk: usize) {
        if !self.config.enable_prefetch {
            return;
        }

        let num_chunks = self.num_chunks();
        let prefetch_count = self.config.prefetch_count;

        // Prefetch next chunks, but only 1 to avoid aggressive eviction
        let safe_prefetch_count = std::cmp::min(prefetch_count, 1);
        for i in 1..=safe_prefetch_count {
            let next_chunk = current_chunk + i;
            if next_chunk < num_chunks {
                // Only prefetch if we have room in cache
                let cache_manager = self
                    .cache_manager
                    .lock()
                    .expect("cache_manager mutex should not be poisoned");
                if cache_manager.chunks_in_memory < self.config.max_chunks_in_memory {
                    drop(cache_manager);
                    let _ = self.ensure_chunk_loaded(next_chunk);
                }
            }
        }
    }

    /// Convert to a regular Array (loads all data into memory)
    pub fn to_array(&self) -> Result<Array<T>> {
        let mut all_data = Vec::with_capacity(self.total_elements);

        // Load all chunks and collect data
        let num_chunks = self.num_chunks();
        for chunk_index in 0..num_chunks {
            // Ensure chunk is loaded
            self.ensure_chunk_loaded(chunk_index)?;

            let chunks = self
                .chunks
                .read()
                .expect("chunks RwLock should not be poisoned");
            if let Some(chunk) = chunks.get(&chunk_index) {
                if let Some(ref data) = chunk.data {
                    all_data.extend_from_slice(data);
                }
            }
        }

        // Truncate to exact size (last chunk might be smaller)
        all_data.truncate(self.total_elements);

        Ok(Array::from_vec(all_data).reshape(&self.shape))
    }

    /// Sync all dirty chunks to disk
    pub fn sync_all(&self) -> Result<()> {
        let mut chunks = self
            .chunks
            .write()
            .expect("chunks RwLock should not be poisoned");
        for chunk in chunks.values_mut() {
            chunk.sync_if_dirty(&self.config.storage_path)?;
        }
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> super::large_scale::MemoryStats {
        self.memory_tracker.get_stats()
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let _cache_manager = self
            .cache_manager
            .lock()
            .expect("cache_manager mutex should not be poisoned");
        let chunks = self
            .chunks
            .read()
            .expect("chunks RwLock should not be poisoned");

        let chunks_in_memory = chunks.values().filter(|c| c.metadata.in_memory).count();
        let chunks_on_disk = chunks
            .values()
            .filter(|c| c.metadata.disk_path.is_some())
            .count();
        let dirty_chunks = chunks.values().filter(|c| c.is_dirty()).count();

        CacheStats {
            chunks_in_memory,
            chunks_on_disk,
            dirty_chunks,
            total_chunks: chunks.len(),
            cache_limit: self.config.max_chunks_in_memory,
        }
    }
}

impl<T: Copy + Send + Sync + Default + 'static> Drop for OutOfCoreArray<T> {
    fn drop(&mut self) {
        // Sync all dirty chunks before dropping
        let _ = self.sync_all();

        // Clean up chunk files
        let chunks = self
            .chunks
            .read()
            .expect("chunks RwLock should not be poisoned");
        for chunk in chunks.values() {
            if let Some(ref path) = chunk.metadata.disk_path {
                let _ = std::fs::remove_file(path);
            }
        }

        // Remove storage directory if empty
        let _ = std::fs::remove_dir(&self.config.storage_path);
    }
}

/// Statistics for cache performance
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub chunks_in_memory: usize,
    pub chunks_on_disk: usize,
    pub dirty_chunks: usize,
    pub total_chunks: usize,
    pub cache_limit: usize,
}

// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_out_of_core_array_creation() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("test_ooc_creation");
        let config = OutOfCoreConfig {
            storage_path: temp_dir.clone(),
            max_chunks_in_memory: 2,
            chunk_size: 10,
            ..Default::default()
        };

        let shape = vec![5, 4]; // 20 elements total
        let array: OutOfCoreArray<f64> = OutOfCoreArray::new(shape.clone(), config)?;

        assert_eq!(array.shape(), &[5, 4]);
        assert_eq!(array.len(), 20);
        assert_eq!(array.num_chunks(), 2); // 20 elements / 10 per chunk = 2 chunks

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(())
    }

    #[test]
    fn test_out_of_core_array_from_data() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("test_ooc_from_data");
        let config = OutOfCoreConfig {
            storage_path: temp_dir.clone(),
            max_chunks_in_memory: 2,
            chunk_size: 5,
            enable_prefetch: true,
            ..Default::default()
        };

        let data: Vec<i32> = (0..15).collect();
        let shape = vec![3, 5];
        let array = OutOfCoreArray::from_data(data.clone(), shape, config)?;

        // Test accessing elements
        assert_eq!(array.get(&[0, 0])?, 0);
        assert_eq!(array.get(&[1, 2])?, 7); // 1*5 + 2 = 7
        assert_eq!(array.get(&[2, 4])?, 14); // 2*5 + 4 = 14

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(())
    }

    #[test]
    fn test_out_of_core_array_chunking() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("test_ooc_chunking");
        let config = OutOfCoreConfig {
            storage_path: temp_dir.clone(),
            max_chunks_in_memory: 1, // Force spilling
            chunk_size: 3,
            ..Default::default()
        };

        let data: Vec<i32> = (0..9).collect();
        let shape = vec![9];
        let mut array = OutOfCoreArray::from_data(data, shape, config)?;

        // Access different chunks to trigger spilling
        assert_eq!(array.get(&[0])?, 0); // Chunk 0
        assert_eq!(array.get(&[5])?, 5); // Chunk 1 (should spill chunk 0)
        assert_eq!(array.get(&[8])?, 8); // Chunk 2 (should spill chunk 1)

        // Modify an element
        array.set(&[2], 42)?;
        assert_eq!(array.get(&[2])?, 42);

        // Convert back to regular array
        let regular_array = array.to_array()?;
        let data = regular_array.to_vec();

        assert_eq!(data[0], 0);
        assert_eq!(data[2], 42);
        assert_eq!(data[8], 8);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(())
    }

    #[test]
    fn test_cache_stats() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("test_ooc_cache_stats");
        let config = OutOfCoreConfig {
            storage_path: temp_dir.clone(),
            max_chunks_in_memory: 2,
            chunk_size: 5,
            ..Default::default()
        };

        let data: Vec<i32> = (0..15).collect();
        let shape = vec![15];
        let array = OutOfCoreArray::from_data(data, shape, config)?;

        let stats = array.get_cache_stats();
        assert_eq!(stats.total_chunks, 3); // 15 elements / 5 per chunk = 3 chunks
        assert_eq!(stats.cache_limit, 2);
        assert!(stats.chunks_in_memory <= 2);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(())
    }
}

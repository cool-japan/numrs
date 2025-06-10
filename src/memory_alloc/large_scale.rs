//! Large-scale data management strategies for NumRS2
//!
//! This module provides memory management strategies specifically designed
//! for handling datasets that are too large to fit entirely in memory.

#[allow(unused_imports)]
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
#[allow(unused_imports)]
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::fs::{File, OpenOptions};
#[allow(unused_imports)]
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::thread;
use crate::error::{NumRs2Error, Result};

/// Configuration for large-scale data management
#[derive(Debug, Clone)]
pub struct LargeScaleConfig {
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_usage: usize,
    /// Memory usage threshold before triggering spilling (0.0-1.0)
    pub spill_threshold: f64,
    /// Chunk size for streaming operations
    pub chunk_size: usize,
    /// Temporary directory for spilled data
    pub temp_dir: PathBuf,
    /// Enable background memory cleanup
    pub background_cleanup: bool,
    /// Memory monitoring interval in milliseconds
    pub monitor_interval_ms: u64,
    /// Enable memory usage statistics
    pub enable_stats: bool,
}

impl Default for LargeScaleConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 8 * 1024 * 1024 * 1024, // 8GB default
            spill_threshold: 0.8, // Spill when 80% of max memory is used
            chunk_size: 64 * 1024 * 1024, // 64MB chunks
            temp_dir: std::env::temp_dir().join("numrs_temp"),
            background_cleanup: true,
            monitor_interval_ms: 1000, // 1 second
            enable_stats: true,
        }
    }
}

/// Memory allocation tracking for large-scale operations
#[derive(Debug)]
pub struct MemoryTracker {
    /// Current memory usage in bytes
    current_usage: AtomicUsize,
    /// Peak memory usage in bytes
    peak_usage: AtomicUsize,
    /// Number of active allocations
    active_allocations: AtomicUsize,
    /// Total allocations made
    total_allocations: AtomicUsize,
    /// Total deallocations made
    total_deallocations: AtomicUsize,
    /// Whether tracking is enabled
    enabled: AtomicBool,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            active_allocations: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let new_usage = self.current_usage.fetch_add(size, Ordering::SeqCst) + size;
        let _ = self.peak_usage.fetch_max(new_usage, Ordering::SeqCst);
        self.active_allocations.fetch_add(1, Ordering::SeqCst);
        self.total_allocations.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        
        self.current_usage.fetch_sub(size, Ordering::SeqCst);
        self.active_allocations.fetch_sub(1, Ordering::SeqCst);
        self.total_deallocations.fetch_add(1, Ordering::SeqCst);
    }
    
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::SeqCst)
    }
    
    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::SeqCst)
    }
    
    /// Get number of active allocations
    pub fn active_allocations(&self) -> usize {
        self.active_allocations.load(Ordering::SeqCst)
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            active_allocations: self.active_allocations(),
            total_allocations: self.total_allocations.load(Ordering::SeqCst),
            total_deallocations: self.total_deallocations.load(Ordering::SeqCst),
        }
    }
    
    /// Reset all statistics
    pub fn reset_stats(&self) {
        self.current_usage.store(0, Ordering::SeqCst);
        self.peak_usage.store(0, Ordering::SeqCst);
        self.active_allocations.store(0, Ordering::SeqCst);
        self.total_allocations.store(0, Ordering::SeqCst);
        self.total_deallocations.store(0, Ordering::SeqCst);
    }
    
    /// Enable or disable tracking
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub active_allocations: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
}

/// Handle for spilled data on disk
#[derive(Debug)]
pub struct SpilledData {
    /// Path to the spilled file
    path: PathBuf,
    /// Size of the data in bytes
    size: usize,
    /// Creation timestamp
    #[allow(dead_code)]
    created_at: SystemTime,
    /// Last access timestamp
    last_accessed: SystemTime,
    /// Whether the data has been marked for cleanup
    marked_for_cleanup: bool,
}

impl SpilledData {
    /// Create a new spilled data handle
    pub fn new(path: PathBuf, size: usize) -> Self {
        let now = SystemTime::now();
        Self {
            path,
            size,
            created_at: now,
            last_accessed: now,
            marked_for_cleanup: false,
        }
    }
    
    /// Get the path to the spilled file
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// Get the size of the spilled data
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Update the last accessed timestamp
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now();
    }
    
    /// Check if the data is eligible for cleanup based on age
    pub fn is_eligible_for_cleanup(&self, max_age_seconds: u64) -> bool {
        if self.marked_for_cleanup {
            return true;
        }
        
        if let Ok(duration) = self.last_accessed.duration_since(UNIX_EPOCH) {
            let age_seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() - duration.as_secs();
            age_seconds > max_age_seconds
        } else {
            false
        }
    }
    
    /// Mark this data for cleanup
    pub fn mark_for_cleanup(&mut self) {
        self.marked_for_cleanup = true;
    }
    
    /// Load the spilled data back into memory
    pub fn load(&mut self) -> Result<Vec<u8>> {
        self.touch();
        let mut file = File::open(&self.path)?;
        let mut buffer = Vec::with_capacity(self.size);
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }
}

impl Drop for SpilledData {
    fn drop(&mut self) {
        // Clean up the spilled file
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Large-scale memory manager for out-of-core operations
pub struct LargeScaleManager {
    /// Configuration
    config: LargeScaleConfig,
    /// Memory usage tracker
    tracker: Arc<MemoryTracker>,
    /// Spilled data registry
    spilled_data: Arc<RwLock<HashMap<String, SpilledData>>>,
    /// LRU queue for spilled data cleanup
    cleanup_queue: Arc<Mutex<VecDeque<String>>>,
    /// Background cleanup thread handle
    cleanup_thread: Option<thread::JoinHandle<()>>,
    /// Shutdown signal for background thread
    shutdown_signal: Arc<AtomicBool>,
    /// Next spill file ID
    next_spill_id: AtomicUsize,
}

impl LargeScaleManager {
    /// Create a new large-scale memory manager
    pub fn new(config: LargeScaleConfig) -> Result<Self> {
        // Create temp directory if it doesn't exist
        std::fs::create_dir_all(&config.temp_dir)?;
        
        let tracker = Arc::new(MemoryTracker::new());
        let spilled_data = Arc::new(RwLock::new(HashMap::new()));
        let cleanup_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        
        let mut manager = Self {
            config,
            tracker,
            spilled_data,
            cleanup_queue,
            cleanup_thread: None,
            shutdown_signal,
            next_spill_id: AtomicUsize::new(0),
        };
        
        // Start background cleanup thread if enabled
        if manager.config.background_cleanup {
            manager.start_background_cleanup();
        }
        
        Ok(manager)
    }
    
    /// Check if memory usage is above the spill threshold
    pub fn should_spill(&self) -> bool {
        if self.config.max_memory_usage == 0 {
            return false; // Unlimited memory
        }
        
        let current_usage = self.tracker.current_usage();
        let threshold = (self.config.max_memory_usage as f64 * self.config.spill_threshold) as usize;
        current_usage > threshold
    }
    
    /// Spill data to disk
    pub fn spill_data(&self, data: &[u8], id: Option<String>) -> Result<String> {
        let spill_id = id.unwrap_or_else(|| {
            format!("spill_{}", self.next_spill_id.fetch_add(1, Ordering::SeqCst))
        });
        
        let spill_path = self.config.temp_dir.join(format!("{}.tmp", spill_id));
        
        // Write data to disk
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&spill_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        
        // Register the spilled data
        let spilled = SpilledData::new(spill_path, data.len());
        {
            let mut registry = self.spilled_data.write().unwrap();
            registry.insert(spill_id.clone(), spilled);
        }
        
        // Add to cleanup queue
        {
            let mut queue = self.cleanup_queue.lock().unwrap();
            queue.push_back(spill_id.clone());
        }
        
        Ok(spill_id)
    }
    
    /// Load spilled data back into memory
    pub fn load_spilled_data(&self, spill_id: &str) -> Result<Vec<u8>> {
        let mut registry = self.spilled_data.write().unwrap();
        if let Some(spilled) = registry.get_mut(spill_id) {
            spilled.load()
        } else {
            Err(NumRs2Error::InvalidOperation(
                format!("Spilled data '{}' not found", spill_id)
            ))
        }
    }
    
    /// Remove spilled data
    pub fn remove_spilled_data(&self, spill_id: &str) -> Result<()> {
        let mut registry = self.spilled_data.write().unwrap();
        if registry.remove(spill_id).is_some() {
            // Remove from cleanup queue as well
            let mut queue = self.cleanup_queue.lock().unwrap();
            queue.retain(|id| id != spill_id);
            Ok(())
        } else {
            Err(NumRs2Error::InvalidOperation(
                format!("Spilled data '{}' not found", spill_id)
            ))
        }
    }
    
    /// Get list of all spilled data IDs
    pub fn list_spilled_data(&self) -> Vec<String> {
        let registry = self.spilled_data.read().unwrap();
        registry.keys().cloned().collect()
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.tracker.get_stats()
    }
    
    /// Get spilled data statistics
    pub fn get_spill_stats(&self) -> SpillStats {
        let registry = self.spilled_data.read().unwrap();
        let total_spilled_size: usize = registry.values().map(|s| s.size()).sum();
        let spilled_count = registry.len();
        
        SpillStats {
            total_spilled_size,
            spilled_count,
            temp_dir: self.config.temp_dir.clone(),
        }
    }
    
    /// Perform manual cleanup of old spilled data
    pub fn cleanup_spilled_data(&self, max_age_seconds: u64) -> usize {
        let mut registry = self.spilled_data.write().unwrap();
        let mut queue = self.cleanup_queue.lock().unwrap();
        
        let mut cleaned_up = 0;
        let mut to_remove = Vec::new();
        
        for (id, spilled) in registry.iter_mut() {
            if spilled.is_eligible_for_cleanup(max_age_seconds) {
                spilled.mark_for_cleanup();
                to_remove.push(id.clone());
                cleaned_up += 1;
            }
        }
        
        // Remove from registry and cleanup queue
        for id in &to_remove {
            registry.remove(id);
            queue.retain(|queue_id| queue_id != id);
        }
        
        cleaned_up
    }
    
    /// Force cleanup of all spilled data
    pub fn force_cleanup_all(&self) {
        let mut registry = self.spilled_data.write().unwrap();
        let mut queue = self.cleanup_queue.lock().unwrap();
        
        registry.clear();
        queue.clear();
    }
    
    /// Start background cleanup thread
    fn start_background_cleanup(&mut self) {
        let spilled_data = Arc::clone(&self.spilled_data);
        let cleanup_queue = Arc::clone(&self.cleanup_queue);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let monitor_interval = self.config.monitor_interval_ms;
        
        let handle = thread::spawn(move || {
            let cleanup_interval = std::time::Duration::from_millis(monitor_interval * 10); // Cleanup less frequently
            let max_age_seconds = 3600; // 1 hour default
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                thread::sleep(cleanup_interval);
                
                // Perform cleanup
                let mut registry = spilled_data.write().unwrap();
                let mut queue = cleanup_queue.lock().unwrap();
                
                let mut to_remove = Vec::new();
                
                for (id, spilled) in registry.iter_mut() {
                    if spilled.is_eligible_for_cleanup(max_age_seconds) {
                        spilled.mark_for_cleanup();
                        to_remove.push(id.clone());
                    }
                }
                
                for id in &to_remove {
                    registry.remove(id);
                    queue.retain(|queue_id| queue_id != id);
                }
                
                drop(registry);
                drop(queue);
            }
        });
        
        self.cleanup_thread = Some(handle);
    }
    
    /// Get memory tracker
    pub fn tracker(&self) -> &Arc<MemoryTracker> {
        &self.tracker
    }
    
    /// Get configuration
    pub fn config(&self) -> &LargeScaleConfig {
        &self.config
    }
    
    /// Create a chunked iterator for processing large data
    pub fn chunk_iterator<'a, T>(&self, data: &'a [T], chunk_size: Option<usize>) -> ChunkIterator<'a, T> {
        let chunk_size = chunk_size.unwrap_or(self.config.chunk_size / std::mem::size_of::<T>());
        ChunkIterator::new(data, chunk_size)
    }
}

impl Drop for LargeScaleManager {
    fn drop(&mut self) {
        // Signal shutdown and wait for background thread
        self.shutdown_signal.store(true, Ordering::SeqCst);
        
        if let Some(handle) = self.cleanup_thread.take() {
            let _ = handle.join();
        }
        
        // Clean up all spilled data
        self.force_cleanup_all();
    }
}

/// Statistics for spilled data
#[derive(Debug, Clone)]
pub struct SpillStats {
    pub total_spilled_size: usize,
    pub spilled_count: usize,
    pub temp_dir: PathBuf,
}

/// Iterator for processing data in chunks
pub struct ChunkIterator<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    current_index: usize,
}

impl<'a, T> ChunkIterator<'a, T> {
    pub fn new(data: &'a [T], chunk_size: usize) -> Self {
        Self {
            data,
            chunk_size: chunk_size.max(1), // Ensure chunk size is at least 1
            current_index: 0,
        }
    }
}

impl<'a, T> Iterator for ChunkIterator<'a, T> {
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.data.len() {
            return None;
        }
        
        let end_index = std::cmp::min(self.current_index + self.chunk_size, self.data.len());
        let chunk = &self.data[self.current_index..end_index];
        self.current_index = end_index;
        
        Some(chunk)
    }
}

lazy_static::lazy_static! {
    /// Global large-scale manager instance
    static ref GLOBAL_MANAGER: std::sync::RwLock<Option<LargeScaleManager>> = std::sync::RwLock::new(None);
}

/// Initialize the global large-scale manager
pub fn init_global_manager(config: LargeScaleConfig) -> Result<()> {
    let mut manager = GLOBAL_MANAGER.write().unwrap();
    if manager.is_some() {
        return Err(NumRs2Error::InvalidOperation(
            "Global large-scale manager already initialized".to_string()
        ));
    }
    
    *manager = Some(LargeScaleManager::new(config)?);
    Ok(())
}

/// Execute a function with access to the global large-scale manager
pub fn with_global_manager<F, R>(f: F) -> Result<R> 
where
    F: FnOnce(&LargeScaleManager) -> R,
{
    let manager = GLOBAL_MANAGER.read().unwrap();
    let manager_ref = manager.as_ref().ok_or_else(|| {
        NumRs2Error::InvalidOperation(
            "Global large-scale manager not initialized. Call init_global_manager() first.".to_string()
        )
    })?;
    Ok(f(manager_ref))
}

/// Convenience function to check if we should spill based on global manager
pub fn should_spill_globally() -> bool {
    with_global_manager(|manager| manager.should_spill())
        .unwrap_or(false)
}

/// Execute a function with mutable access to the global large-scale manager
pub fn with_global_manager_mut<F, R>(f: F) -> Result<R> 
where
    F: FnOnce(&mut LargeScaleManager) -> Result<R>,
{
    let mut manager = GLOBAL_MANAGER.write().unwrap();
    let manager_ref = manager.as_mut().ok_or_else(|| {
        NumRs2Error::InvalidOperation(
            "Global large-scale manager not initialized. Call init_global_manager() first.".to_string()
        )
    })?;
    f(manager_ref)
}

/// Convenience function to spill data using global manager
pub fn spill_data_globally(data: &[u8], id: Option<String>) -> Result<String> {
    with_global_manager_mut(|manager| manager.spill_data(data, id))
}

/// Convenience function to load spilled data using global manager
pub fn load_spilled_data_globally(spill_id: &str) -> Result<Vec<u8>> {
    with_global_manager_mut(|manager| manager.load_spilled_data(spill_id))
}

/// Convenience function to get memory stats from global manager
pub fn get_global_memory_stats() -> Result<MemoryStats> {
    with_global_manager(|manager| manager.get_memory_stats())
}

/// Convenience function to get spill stats from global manager
pub fn get_global_spill_stats() -> Result<SpillStats> {
    with_global_manager(|manager| manager.get_spill_stats())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        
        // Test allocation tracking
        tracker.record_allocation(1000);
        assert_eq!(tracker.current_usage(), 1000);
        assert_eq!(tracker.active_allocations(), 1);
        
        tracker.record_allocation(500);
        assert_eq!(tracker.current_usage(), 1500);
        assert_eq!(tracker.active_allocations(), 2);
        assert_eq!(tracker.peak_usage(), 1500);
        
        // Test deallocation tracking
        tracker.record_deallocation(500);
        assert_eq!(tracker.current_usage(), 1000);
        assert_eq!(tracker.active_allocations(), 1);
        assert_eq!(tracker.peak_usage(), 1500); // Peak should remain
        
        // Test statistics
        let stats = tracker.get_stats();
        assert_eq!(stats.current_usage, 1000);
        assert_eq!(stats.peak_usage, 1500);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        
        // Test reset
        tracker.reset_stats();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
    }
    
    #[test]
    fn test_large_scale_manager() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("test_large_scale");
        let config = LargeScaleConfig {
            max_memory_usage: 1000,
            spill_threshold: 0.5,
            temp_dir: temp_dir.clone(),
            background_cleanup: false,
            ..Default::default()
        };
        
        let manager = LargeScaleManager::new(config)?;
        
        // Test spill threshold
        manager.tracker().record_allocation(600); // Above 50% of 1000
        assert!(manager.should_spill());
        
        manager.tracker().record_deallocation(200); // Now at 400, below threshold
        assert!(!manager.should_spill());
        
        // Test data spilling
        let test_data = b"Hello, world! This is test data for spilling.";
        let spill_id = manager.spill_data(test_data, Some("test_spill".to_string()))?;
        assert_eq!(spill_id, "test_spill");
        
        // Test loading spilled data
        let loaded_data = manager.load_spilled_data(&spill_id)?;
        assert_eq!(loaded_data, test_data);
        
        // Test spill stats
        let spill_stats = manager.get_spill_stats();
        assert_eq!(spill_stats.spilled_count, 1);
        assert_eq!(spill_stats.total_spilled_size, test_data.len());
        
        // Test cleanup
        manager.remove_spilled_data(&spill_id)?;
        let spill_stats = manager.get_spill_stats();
        assert_eq!(spill_stats.spilled_count, 0);
        
        // Clean up temp directory
        let _ = fs::remove_dir_all(&temp_dir);
        
        Ok(())
    }
    
    #[test]
    fn test_chunk_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunk_iter = ChunkIterator::new(&data, 3);
        
        let chunks: Vec<&[i32]> = chunk_iter.collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], &[1, 2, 3]);
        assert_eq!(chunks[1], &[4, 5, 6]);
        assert_eq!(chunks[2], &[7, 8, 9]);
        assert_eq!(chunks[3], &[10]);
    }
    
    #[test]
    fn test_spilled_data() {
        let temp_dir = std::env::temp_dir();
        let spill_path = temp_dir.join("test_spill.tmp");
        
        // Create test file
        let test_data = b"Test spilled data";
        std::fs::write(&spill_path, test_data).unwrap();
        
        let mut spilled = SpilledData::new(spill_path.clone(), test_data.len());
        
        // Test loading
        let loaded = spilled.load().unwrap();
        assert_eq!(loaded, test_data);
        
        // Test cleanup eligibility (should not be eligible for cleanup immediately)
        assert!(!spilled.is_eligible_for_cleanup(3600));
        
        // Mark for cleanup
        spilled.mark_for_cleanup();
        assert!(spilled.is_eligible_for_cleanup(0));
        
        // Clean up
        drop(spilled);
        assert!(!spill_path.exists());
    }
}
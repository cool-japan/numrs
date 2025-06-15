//! Cache-aware memory optimization and data layout strategies
//!
//! This module provides cache-optimized data structures and algorithms that are aware
//! of modern CPU cache hierarchies and memory access patterns to maximize performance.

use crate::error::{NumRs2Error, Result};
use crate::traits::{MemoryAllocator, SpecializedAllocator};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// Cache line size constants for different architectures
pub mod cache_constants {
    /// Common cache line size (64 bytes) for x86/x64
    pub const CACHE_LINE_SIZE: usize = 64;
    /// L1 cache size estimate (32KB typical)
    pub const L1_CACHE_SIZE: usize = 32 * 1024;
    /// L2 cache size estimate (256KB typical)
    pub const L2_CACHE_SIZE: usize = 256 * 1024;
    /// L3 cache size estimate (8MB typical)
    pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;
    /// Memory page size (4KB typical)
    pub const PAGE_SIZE: usize = 4096;
    /// Prefetch distance for linear access patterns
    pub const PREFETCH_DISTANCE: usize = 3;
}

/// Cache-aware memory layout configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache line size for this system
    pub cache_line_size: usize,
    /// L1 cache size
    pub l1_cache_size: usize,
    /// L2 cache size
    pub l2_cache_size: usize,
    /// L3 cache size
    pub l3_cache_size: usize,
    /// Enable cache line padding for data structures
    pub enable_cache_padding: bool,
    /// Enable block-based algorithms
    pub enable_blocking: bool,
    /// Enable data prefetching
    pub enable_prefetch: bool,
    /// Memory access pattern optimization
    pub optimize_access_patterns: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_line_size: cache_constants::CACHE_LINE_SIZE,
            l1_cache_size: cache_constants::L1_CACHE_SIZE,
            l2_cache_size: cache_constants::L2_CACHE_SIZE,
            l3_cache_size: cache_constants::L3_CACHE_SIZE,
            enable_cache_padding: true,
            enable_blocking: true,
            enable_prefetch: true,
            optimize_access_patterns: true,
        }
    }
}

/// Cache performance metrics for monitoring and tuning
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// Estimated cache hits (L1)
    pub l1_hits: u64,
    /// Estimated cache misses (L1)
    pub l1_misses: u64,
    /// Estimated cache hits (L2)
    pub l2_hits: u64,
    /// Estimated cache misses (L2)
    pub l2_misses: u64,
    /// Total memory accesses tracked
    pub total_accesses: u64,
    /// Average access latency
    pub avg_latency_ns: f64,
    /// Cache efficiency ratio (0.0-1.0)
    pub cache_efficiency: f64,
    /// Last measurement timestamp
    pub last_updated: Option<Instant>,
}

impl CacheMetrics {
    /// Calculate cache hit ratio for L1
    pub fn l1_hit_ratio(&self) -> f64 {
        if self.l1_hits + self.l1_misses == 0 {
            0.0
        } else {
            self.l1_hits as f64 / (self.l1_hits + self.l1_misses) as f64
        }
    }

    /// Calculate cache hit ratio for L2
    pub fn l2_hit_ratio(&self) -> f64 {
        if self.l2_hits + self.l2_misses == 0 {
            0.0
        } else {
            self.l2_hits as f64 / (self.l2_hits + self.l2_misses) as f64
        }
    }

    /// Calculate overall cache efficiency
    pub fn overall_efficiency(&self) -> f64 {
        let l1_ratio = self.l1_hit_ratio();
        let l2_ratio = self.l2_hit_ratio();
        // Weighted efficiency: L1 hits are much faster than L2 hits
        (l1_ratio * 0.9) + (l2_ratio * 0.1)
    }
}

/// Cache-optimized memory allocator that aligns data to cache boundaries
pub struct CacheOptimizedAllocator {
    config: CacheConfig,
    metrics: std::sync::Mutex<CacheMetrics>,
    access_tracker: std::sync::Mutex<AccessTracker>,
}

impl CacheOptimizedAllocator {
    /// Create a new cache-optimized allocator
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            metrics: std::sync::Mutex::new(CacheMetrics::default()),
            access_tracker: std::sync::Mutex::new(AccessTracker::new()),
        }
    }

    /// Create with default cache configuration
    pub fn default() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get current cache performance metrics
    pub fn get_cache_metrics(&self) -> CacheMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Calculate optimal block size for cache-blocking algorithms
    pub fn optimal_block_size(&self, element_size: usize, cache_level: CacheLevel) -> usize {
        let cache_size = match cache_level {
            CacheLevel::L1 => self.config.l1_cache_size,
            CacheLevel::L2 => self.config.l2_cache_size,
            CacheLevel::L3 => self.config.l3_cache_size,
        };

        // Use approximately 1/4 of cache for blocking to leave room for other data
        let usable_cache = cache_size / 4;
        let elements_per_block = usable_cache / element_size;

        // Round down to nearest power of 2 for efficient indexing
        elements_per_block.next_power_of_two() / 2
    }

    /// Check if data size fits efficiently in a specific cache level
    pub fn fits_in_cache(&self, size: usize, cache_level: CacheLevel) -> bool {
        let cache_size = match cache_level {
            CacheLevel::L1 => self.config.l1_cache_size,
            CacheLevel::L2 => self.config.l2_cache_size,
            CacheLevel::L3 => self.config.l3_cache_size,
        };

        // Data should fit comfortably (use 80% of cache)
        size <= (cache_size * 4) / 5
    }

    /// Record a memory access for performance tracking
    fn record_access(&self, address: usize, size: usize, access_type: AccessType) {
        if let Ok(mut tracker) = self.access_tracker.try_lock() {
            tracker.record_access(address, size, access_type);

            // Update cache metrics periodically
            if tracker.should_update_metrics() {
                if let Ok(mut metrics) = self.metrics.try_lock() {
                    tracker.update_cache_metrics(&mut metrics, &self.config);
                }
            }
        }
    }

    /// Generate cache optimization recommendations
    pub fn analyze_cache_performance(&self) -> Vec<CacheOptimizationRecommendation> {
        let metrics = self.get_cache_metrics();
        let mut recommendations = Vec::new();

        // Analyze L1 cache performance
        if metrics.l1_hit_ratio() < 0.9 {
            recommendations.push(CacheOptimizationRecommendation {
                optimization_type: CacheOptimizationType::ImproveLocality,
                description: "L1 cache hit ratio is low. Consider improving data locality through blocking or tiling.".to_string(),
                estimated_improvement: 0.15,
                complexity: 3,
                target_cache_level: CacheLevel::L1,
            });
        }

        // Analyze L2 cache performance
        if metrics.l2_hit_ratio() < 0.95 {
            recommendations.push(CacheOptimizationRecommendation {
                optimization_type: CacheOptimizationType::ReduceWorkingSet,
                description: "L2 cache hit ratio indicates large working set. Consider data structure reorganization.".to_string(),
                estimated_improvement: 0.10,
                complexity: 4,
                target_cache_level: CacheLevel::L2,
            });
        }

        // Check for false sharing potential
        if self.has_potential_false_sharing() {
            recommendations.push(CacheOptimizationRecommendation {
                optimization_type: CacheOptimizationType::EliminateFalseSharing,
                description: "Potential false sharing detected. Consider cache line padding for frequently accessed data.".to_string(),
                estimated_improvement: 0.25,
                complexity: 2,
                target_cache_level: CacheLevel::L1,
            });
        }

        recommendations
    }

    /// Check for potential false sharing issues
    fn has_potential_false_sharing(&self) -> bool {
        // This is a simplified heuristic - in a real implementation,
        // this would analyze access patterns for concurrent writes to the same cache line
        if let Ok(tracker) = self.access_tracker.try_lock() {
            tracker.concurrent_writes_in_cache_lines() > 10
        } else {
            false
        }
    }
}

impl std::fmt::Debug for CacheOptimizedAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheOptimizedAllocator")
            .field("config", &self.config)
            .field("metrics", &"Mutex<CacheMetrics>")
            .finish()
    }
}

impl MemoryAllocator for CacheOptimizedAllocator {
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        // Align allocation to cache line boundaries for better performance
        let aligned_size = if self.config.enable_cache_padding {
            // Round up to next cache line boundary
            (layout.size() + self.config.cache_line_size - 1) & !(self.config.cache_line_size - 1)
        } else {
            layout.size()
        };

        let aligned_layout = Layout::from_size_align(
            aligned_size,
            layout.align().max(self.config.cache_line_size),
        )
        .map_err(|_| {
            NumRs2Error::Memory(crate::error::memory::MemoryError::alignment_error(
                "Invalid layout",
                self.config.cache_line_size,
            ))
        })?;

        unsafe {
            let ptr = std::alloc::alloc(aligned_layout);
            if ptr.is_null() {
                return Err(NumRs2Error::Memory(
                    crate::error::memory::MemoryError::allocation_failed(
                        "Allocation failed",
                        aligned_size,
                    ),
                ));
            }

            let non_null_ptr = NonNull::new_unchecked(ptr);

            // Record the allocation for cache analysis
            self.record_access(ptr as usize, aligned_size, AccessType::Write);

            Ok(non_null_ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        let aligned_size = if self.config.enable_cache_padding {
            (layout.size() + self.config.cache_line_size - 1) & !(self.config.cache_line_size - 1)
        } else {
            layout.size()
        };

        let aligned_layout = Layout::from_size_align(
            aligned_size,
            layout.align().max(self.config.cache_line_size),
        )
        .map_err(|_| {
            NumRs2Error::Memory(crate::error::memory::MemoryError::alignment_error(
                "Invalid layout",
                self.config.cache_line_size,
            ))
        })?;

        std::alloc::dealloc(ptr.as_ptr(), aligned_layout);
        Ok(())
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        // For cache optimization, we prefer to allocate new memory and copy
        // to maintain alignment and avoid fragmentation
        let new_ptr = self.allocate(new_layout)?;

        let copy_size = old_layout.size().min(new_layout.size());
        std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);

        self.deallocate(ptr, old_layout)?;

        Ok(new_ptr)
    }

    fn supports_layout(&self, layout: Layout) -> bool {
        // Support any layout that can be aligned to cache boundaries
        layout.size() > 0 && layout.align() <= self.config.cache_line_size
    }

    fn preferred_alignment(&self) -> usize {
        self.config.cache_line_size
    }

    fn statistics(&self) -> Option<crate::traits::AllocationStats> {
        let metrics = self.get_cache_metrics();
        Some(crate::traits::AllocationStats {
            bytes_allocated: 0, // Would need tracking
            bytes_deallocated: 0,
            active_allocations: 0,
            peak_usage: 0,
            allocation_count: metrics.total_accesses as usize,
            deallocation_count: 0,
        })
    }
}

impl SpecializedAllocator for CacheOptimizedAllocator {
    fn allocation_error(&self, msg: &str) -> Self::Error {
        NumRs2Error::Memory(crate::error::memory::MemoryError::allocation_failed(msg, 0))
    }
}

/// Cache level enumeration for optimization targeting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// Memory access type for cache analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Cache optimization recommendation
#[derive(Debug, Clone)]
pub struct CacheOptimizationRecommendation {
    pub optimization_type: CacheOptimizationType,
    pub description: String,
    pub estimated_improvement: f64,
    pub complexity: u8,
    pub target_cache_level: CacheLevel,
}

/// Types of cache optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOptimizationType {
    ImproveLocality,
    ReduceWorkingSet,
    EliminateFalseSharing,
    OptimizeStride,
    EnablePrefetch,
    UseBlocking,
    AlignDataStructures,
}

/// Access pattern tracker for cache analysis
#[derive(Debug)]
struct AccessTracker {
    recent_accesses: Vec<MemoryAccess>,
    access_histogram: HashMap<usize, usize>, // cache line -> access count
    total_accesses: u64,
    concurrent_writes: u64,
    last_analysis: Option<Instant>,
}

#[derive(Debug, Clone)]
struct MemoryAccess {
    address: usize,
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    access_type: AccessType,
    #[allow(dead_code)]
    timestamp: Instant,
}

impl AccessTracker {
    fn new() -> Self {
        Self {
            recent_accesses: Vec::new(),
            access_histogram: HashMap::new(),
            total_accesses: 0,
            concurrent_writes: 0,
            last_analysis: None,
        }
    }

    fn record_access(&mut self, address: usize, size: usize, access_type: AccessType) {
        let access = MemoryAccess {
            address,
            size,
            access_type,
            timestamp: Instant::now(),
        };

        // Track cache line accesses
        let cache_line = address / cache_constants::CACHE_LINE_SIZE;
        *self.access_histogram.entry(cache_line).or_insert(0) += 1;

        // Track concurrent writes for false sharing detection
        if access_type == AccessType::Write || access_type == AccessType::ReadWrite {
            self.concurrent_writes += 1;
        }

        self.recent_accesses.push(access);
        self.total_accesses += 1;

        // Keep only recent accesses (last 1000) to bound memory usage
        if self.recent_accesses.len() > 1000 {
            self.recent_accesses.remove(0);
        }
    }

    fn should_update_metrics(&self) -> bool {
        match self.last_analysis {
            None => true,
            Some(last) => last.elapsed() > Duration::from_secs(1),
        }
    }

    fn update_cache_metrics(&mut self, metrics: &mut CacheMetrics, config: &CacheConfig) {
        // Simplified cache simulation based on access patterns
        let mut l1_hits = 0u64;
        let mut l1_misses = 0u64;
        let mut l2_hits = 0u64;
        let mut l2_misses = 0u64;

        // Simulate cache behavior based on recent access patterns
        let mut simulated_l1_cache = std::collections::HashSet::new();
        let mut simulated_l2_cache = std::collections::HashSet::new();

        let l1_cache_lines = config.l1_cache_size / config.cache_line_size;
        let l2_cache_lines = config.l2_cache_size / config.cache_line_size;

        for access in &self.recent_accesses {
            let cache_line = access.address / config.cache_line_size;

            if simulated_l1_cache.contains(&cache_line) {
                l1_hits += 1;
            } else if simulated_l2_cache.contains(&cache_line) {
                l1_misses += 1;
                l2_hits += 1;
                // Move to L1 cache
                if simulated_l1_cache.len() >= l1_cache_lines {
                    // Evict random entry (simplified LRU)
                    simulated_l1_cache.clear();
                }
                simulated_l1_cache.insert(cache_line);
            } else {
                l1_misses += 1;
                l2_misses += 1;
                // Load into both caches
                if simulated_l2_cache.len() >= l2_cache_lines {
                    simulated_l2_cache.clear();
                }
                simulated_l2_cache.insert(cache_line);
                if simulated_l1_cache.len() >= l1_cache_lines {
                    simulated_l1_cache.clear();
                }
                simulated_l1_cache.insert(cache_line);
            }
        }

        metrics.l1_hits = l1_hits;
        metrics.l1_misses = l1_misses;
        metrics.l2_hits = l2_hits;
        metrics.l2_misses = l2_misses;
        metrics.total_accesses = self.total_accesses;
        metrics.cache_efficiency = metrics.overall_efficiency();
        metrics.last_updated = Some(Instant::now());

        self.last_analysis = Some(Instant::now());
    }

    fn concurrent_writes_in_cache_lines(&self) -> u64 {
        self.concurrent_writes
    }
}

/// Cache-blocked matrix operations for improved cache performance
pub struct CacheBlockedMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    block_size: usize,
    cache_config: CacheConfig,
}

impl<T: Clone> CacheBlockedMatrix<T> {
    /// Create a new cache-blocked matrix
    pub fn new(rows: usize, cols: usize, cache_config: CacheConfig) -> Self {
        let element_size = std::mem::size_of::<T>();
        let block_size = if cache_config.enable_blocking {
            // Calculate optimal block size for L1 cache
            let cache_size = cache_config.l1_cache_size / 4; // Use 1/4 of L1 cache
            let elements_per_block = cache_size / element_size;
            (elements_per_block as f64).sqrt() as usize
        } else {
            64 // Default block size
        };

        Self {
            data: vec![unsafe { std::mem::zeroed() }; rows * cols],
            rows,
            cols,
            block_size,
            cache_config,
        }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> bool {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            true
        } else {
            false
        }
    }

    /// Cache-blocked matrix multiplication
    pub fn multiply_blocked(&self, other: &Self) -> Option<Self>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default,
    {
        if self.cols != other.rows {
            return None;
        }

        let mut result = Self {
            data: vec![T::default(); self.rows * other.cols],
            rows: self.rows,
            cols: other.cols,
            block_size: self.block_size,
            cache_config: self.cache_config.clone(),
        };

        // Cache-blocked matrix multiplication
        for i_block in (0..self.rows).step_by(self.block_size) {
            for j_block in (0..other.cols).step_by(self.block_size) {
                for k_block in (0..self.cols).step_by(self.block_size) {
                    // Process this block
                    let i_end = (i_block + self.block_size).min(self.rows);
                    let j_end = (j_block + self.block_size).min(other.cols);
                    let k_end = (k_block + self.block_size).min(self.cols);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = result.data[i * result.cols + j];
                            for k in k_block..k_end {
                                sum = sum
                                    + self.data[i * self.cols + k] * other.data[k * other.cols + j];
                            }
                            result.data[i * result.cols + j] = sum;
                        }
                    }
                }
            }
        }

        Some(result)
    }

    /// Transpose with cache-blocking
    pub fn transpose_blocked(&self) -> Self
    where
        T: Copy + Default,
    {
        let mut result = Self {
            data: vec![T::default(); self.rows * self.cols],
            rows: self.cols,
            cols: self.rows,
            block_size: self.block_size,
            cache_config: self.cache_config.clone(),
        };

        // Cache-blocked transpose
        for i_block in (0..self.rows).step_by(self.block_size) {
            for j_block in (0..self.cols).step_by(self.block_size) {
                let i_end = (i_block + self.block_size).min(self.rows);
                let j_end = (j_block + self.block_size).min(self.cols);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        result.data[j * result.cols + i] = self.data[i * self.cols + j];
                    }
                }
            }
        }

        result
    }
}

/// Cache-friendly data structure builder
pub struct CacheOptimizedBuilder {
    config: CacheConfig,
}

impl CacheOptimizedBuilder {
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Create a cache-optimized allocator
    pub fn build_allocator(&self) -> CacheOptimizedAllocator {
        CacheOptimizedAllocator::new(self.config.clone())
    }

    /// Create a cache-blocked matrix
    pub fn build_matrix<T: Clone>(&self, rows: usize, cols: usize) -> CacheBlockedMatrix<T> {
        CacheBlockedMatrix::new(rows, cols, self.config.clone())
    }

    /// Calculate optimal parameters for a given data size
    pub fn optimize_for_size(
        &self,
        total_elements: usize,
        element_size: usize,
    ) -> CacheOptimizationParams {
        let total_size = total_elements * element_size;

        let recommended_block_size = if total_size <= self.config.l1_cache_size {
            // Fits in L1, use small blocks
            (self.config.l1_cache_size / 4 / element_size) as usize
        } else if total_size <= self.config.l2_cache_size {
            // Fits in L2, use medium blocks
            (self.config.l2_cache_size / 8 / element_size) as usize
        } else {
            // Large data, use L3-optimized blocks
            (self.config.l3_cache_size / 16 / element_size) as usize
        };

        CacheOptimizationParams {
            block_size: recommended_block_size.max(1),
            enable_prefetch: total_size > self.config.l2_cache_size,
            enable_blocking: total_size > self.config.l1_cache_size,
            target_cache_level: if total_size <= self.config.l1_cache_size {
                CacheLevel::L1
            } else if total_size <= self.config.l2_cache_size {
                CacheLevel::L2
            } else {
                CacheLevel::L3
            },
        }
    }
}

/// Cache optimization parameters for algorithms
#[derive(Debug, Clone)]
pub struct CacheOptimizationParams {
    pub block_size: usize,
    pub enable_prefetch: bool,
    pub enable_blocking: bool,
    pub target_cache_level: CacheLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_optimized_allocator() {
        let config = CacheConfig::default();
        let allocator = CacheOptimizedAllocator::new(config);

        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = allocator.allocate(layout).unwrap();

        // Check that allocation is cache-line aligned
        assert_eq!(ptr.as_ptr() as usize % cache_constants::CACHE_LINE_SIZE, 0);

        unsafe {
            allocator.deallocate(ptr, layout).unwrap();
        }
    }

    #[test]
    fn test_cache_metrics() {
        let mut metrics = CacheMetrics::default();
        metrics.l1_hits = 900;
        metrics.l1_misses = 100;
        metrics.l2_hits = 80;
        metrics.l2_misses = 20;

        assert_eq!(metrics.l1_hit_ratio(), 0.9);
        assert_eq!(metrics.l2_hit_ratio(), 0.8);
        assert!(metrics.overall_efficiency() > 0.8);
    }

    #[test]
    fn test_cache_blocked_matrix() {
        let config = CacheConfig::default();
        let mut matrix = CacheBlockedMatrix::<f32>::new(4, 4, config);

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);
        matrix.set(3, 3, 4.0);

        // Test transpose
        let transposed = matrix.transpose_blocked();
        assert_eq!(transposed.get(0, 0), Some(&1.0));
        assert_eq!(transposed.get(1, 1), Some(&2.0));
    }

    #[test]
    fn test_cache_optimization_params() {
        let config = CacheConfig::default();
        let builder = CacheOptimizedBuilder::new(config);

        // Small data should target L1
        let params = builder.optimize_for_size(1000, 4);
        assert_eq!(params.target_cache_level, CacheLevel::L1);
        assert!(!params.enable_blocking);

        // Large data should target L3
        let params = builder.optimize_for_size(1_000_000, 8);
        assert_eq!(params.target_cache_level, CacheLevel::L3);
        assert!(params.enable_blocking);
        assert!(params.enable_prefetch);
    }

    #[test]
    fn test_optimal_block_size_calculation() {
        let config = CacheConfig::default();
        let allocator = CacheOptimizedAllocator::new(config);

        let block_size_l1 = allocator.optimal_block_size(8, CacheLevel::L1);
        let block_size_l2 = allocator.optimal_block_size(8, CacheLevel::L2);
        let block_size_l3 = allocator.optimal_block_size(8, CacheLevel::L3);

        assert!(block_size_l1 < block_size_l2);
        assert!(block_size_l2 < block_size_l3);
        assert!(block_size_l1.is_power_of_two());
    }

    #[test]
    fn test_cache_fits_analysis() {
        let config = CacheConfig::default();
        let allocator = CacheOptimizedAllocator::new(config);

        // Small data should fit in L1
        assert!(allocator.fits_in_cache(16 * 1024, CacheLevel::L1));

        // Medium data should fit in L2 but not L1
        assert!(!allocator.fits_in_cache(100 * 1024, CacheLevel::L1));
        assert!(allocator.fits_in_cache(100 * 1024, CacheLevel::L2));

        // Large data should only fit in L3
        assert!(!allocator.fits_in_cache(4 * 1024 * 1024, CacheLevel::L2));
        assert!(allocator.fits_in_cache(4 * 1024 * 1024, CacheLevel::L3));
    }

    #[test]
    fn test_matrix_blocked_multiplication() {
        let config = CacheConfig::default();
        let mut a = CacheBlockedMatrix::<f32>::new(2, 2, config.clone());
        let mut b = CacheBlockedMatrix::<f32>::new(2, 2, config);

        // Set up identity matrices for simple test
        a.set(0, 0, 1.0);
        a.set(1, 1, 1.0);
        b.set(0, 0, 2.0);
        b.set(1, 1, 2.0);

        let result = a.multiply_blocked(&b).unwrap();
        assert_eq!(result.get(0, 0), Some(&2.0));
        assert_eq!(result.get(1, 1), Some(&2.0));
        assert_eq!(result.get(0, 1), Some(&0.0));
        assert_eq!(result.get(1, 0), Some(&0.0));
    }

    #[test]
    fn test_cache_performance_analysis() {
        let config = CacheConfig::default();
        let allocator = CacheOptimizedAllocator::new(config);

        // Simulate some allocations to generate metrics
        for _ in 0..100 {
            let layout = Layout::from_size_align(64, 8).unwrap();
            let _ptr = allocator.allocate(layout).unwrap();
        }

        let recommendations = allocator.analyze_cache_performance();
        // Should have some recommendations for a new allocator
        assert!(!recommendations.is_empty());
    }
}

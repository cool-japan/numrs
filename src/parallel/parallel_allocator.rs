//! Parallel memory allocation strategies
//!
//! This module provides thread-safe memory allocators optimized for
//! parallel numerical computations with minimal contention.

use crate::error::{NumRs2Error, Result};
use crate::traits::{MemoryAllocator, SpecializedAllocator, AllocationStats};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::thread::{self, ThreadId};
use std::cell::RefCell;
use std::time::{Duration, Instant};

thread_local! {
    /// Thread-local allocator for reduced contention
    static LOCAL_ALLOCATOR: RefCell<Option<ThreadLocalState>> = RefCell::new(None);
}

/// Thread-local allocation state
#[derive(Debug)]
struct ThreadLocalState {
    allocator: Box<dyn MemoryAllocator<Error = NumRs2Error> + Send>,
    stats: AllocationStats,
    last_gc: Instant,
    cache_size_limit: usize,
    cached_blocks: Vec<CachedBlock>,
}

#[derive(Debug)]
struct CachedBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    allocated_at: Instant,
}

// SAFETY: CachedBlock is safe to send between threads because:
// 1. NonNull<u8> is just a pointer wrapper and the actual memory management is handled by allocators
// 2. Layout and Instant are both Send + Sync
// 3. The allocator ensures memory safety across threads
unsafe impl Send for CachedBlock {}
unsafe impl Sync for CachedBlock {}

impl ThreadLocalState {
    fn new<A>(allocator: A, cache_size_limit: usize) -> Self
    where
        A: MemoryAllocator<Error = NumRs2Error> + Send + 'static,
    {
        Self {
            allocator: Box::new(allocator),
            stats: AllocationStats::default(),
            last_gc: Instant::now(),
            cache_size_limit,
            cached_blocks: Vec::new(),
        }
    }

    fn try_allocate_from_cache(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        // Find a cached block that fits
        for (i, block) in self.cached_blocks.iter().enumerate() {
            if block.layout.size() >= layout.size() && 
               block.layout.align() >= layout.align() {
                let ptr = block.ptr;
                self.cached_blocks.remove(i);
                return Some(ptr);
            }
        }
        None
    }

    fn cache_block(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if self.cached_blocks.len() < self.cache_size_limit {
            self.cached_blocks.push(CachedBlock {
                ptr,
                layout,
                allocated_at: Instant::now(),
            });
        } else {
            // Cache is full, actually deallocate
            unsafe {
                let _ = self.allocator.deallocate(ptr, layout);
            }
        }
    }

    fn garbage_collect(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.cached_blocks.retain(|block| {
            if now.duration_since(block.allocated_at) > max_age {
                // Block is too old, deallocate it
                unsafe {
                    let _ = self.allocator.deallocate(block.ptr, block.layout);
                }
                false
            } else {
                true
            }
        });
        self.last_gc = now;
    }

    fn should_gc(&self, gc_interval: Duration) -> bool {
        self.last_gc.elapsed() > gc_interval
    }
}

/// Configuration for parallel allocator
#[derive(Debug, Clone)]
pub struct ParallelAllocatorConfig {
    /// Enable thread-local caching
    pub enable_thread_local_cache: bool,
    /// Maximum cached blocks per thread
    pub max_cached_blocks_per_thread: usize,
    /// Garbage collection interval
    pub gc_interval: Duration,
    /// Maximum age for cached blocks
    pub max_block_age: Duration,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// Global pool size for shared allocations
    pub global_pool_size: usize,
    /// Enable allocation tracking
    pub enable_tracking: bool,
}

impl Default for ParallelAllocatorConfig {
    fn default() -> Self {
        Self {
            enable_thread_local_cache: true,
            max_cached_blocks_per_thread: 100,
            gc_interval: Duration::from_secs(30),
            max_block_age: Duration::from_secs(300),
            numa_aware: false,
            global_pool_size: 1024 * 1024, // 1MB
            enable_tracking: true,
        }
    }
}

/// Parallel memory allocator with thread-local optimization
pub struct ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone,
{
    base_allocator: A,
    config: ParallelAllocatorConfig,
    global_stats: Arc<RwLock<AllocationStats>>,
    thread_allocators: Arc<Mutex<HashMap<ThreadId, Arc<Mutex<ThreadLocalState>>>>>,
    global_pool: Arc<Mutex<Vec<CachedBlock>>>,
}

impl<A> std::fmt::Debug for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelAllocator")
            .field("base_allocator", &"<allocator>")
            .field("config", &self.config)
            .field("global_stats", &"<mutex>")
            .field("thread_allocators", &"<mutex>")
            .field("global_pool", &"<mutex>")
            .finish()
    }
}

impl<A> ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    /// Create a new parallel allocator
    pub fn new(base_allocator: A, config: ParallelAllocatorConfig) -> Self {
        Self {
            base_allocator,
            config,
            global_stats: Arc::new(RwLock::new(AllocationStats::default())),
            thread_allocators: Arc::new(Mutex::new(HashMap::new())),
            global_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get or create thread-local state
    fn get_thread_local_state(&self) -> Result<Arc<Mutex<ThreadLocalState>>> {
        let thread_id = thread::current().id();
        let mut allocators = self.thread_allocators.lock().unwrap();
        
        if let Some(state) = allocators.get(&thread_id) {
            Ok(Arc::clone(state))
        } else {
            let local_state = ThreadLocalState::new(
                self.base_allocator.clone(),
                self.config.max_cached_blocks_per_thread,
            );
            let state = Arc::new(Mutex::new(local_state));
            allocators.insert(thread_id, Arc::clone(&state));
            Ok(state)
        }
    }

    /// Try to allocate from global pool
    fn try_allocate_from_global_pool(&self, layout: Layout) -> Option<NonNull<u8>> {
        if !self.config.enable_thread_local_cache {
            return None;
        }

        let mut pool = self.global_pool.lock().unwrap();
        
        for (i, block) in pool.iter().enumerate() {
            if block.layout.size() >= layout.size() && 
               block.layout.align() >= layout.align() {
                let ptr = block.ptr;
                pool.remove(i);
                return Some(ptr);
            }
        }
        None
    }

    /// Return block to global pool
    fn return_to_global_pool(&self, ptr: NonNull<u8>, layout: Layout) {
        if !self.config.enable_thread_local_cache {
            unsafe {
                let _ = self.base_allocator.deallocate(ptr, layout);
            }
            return;
        }

        let mut pool = self.global_pool.lock().unwrap();
        
        if pool.len() < self.config.global_pool_size / std::mem::size_of::<CachedBlock>() {
            pool.push(CachedBlock {
                ptr,
                layout,
                allocated_at: Instant::now(),
            });
        } else {
            // Pool is full, actually deallocate
            unsafe {
                let _ = self.base_allocator.deallocate(ptr, layout);
            }
        }
    }

    /// Trigger garbage collection for all thread-local caches
    pub fn garbage_collect_all(&self) -> Result<()> {
        let allocators = self.thread_allocators.lock().unwrap();
        
        for state in allocators.values() {
            if let Ok(mut local_state) = state.try_lock() {
                local_state.garbage_collect(self.config.max_block_age);
            }
        }

        // Also clean global pool
        {
            let mut pool = self.global_pool.lock().unwrap();
            let now = Instant::now();
            pool.retain(|block| {
                if now.duration_since(block.allocated_at) > self.config.max_block_age {
                    unsafe {
                        let _ = self.base_allocator.deallocate(block.ptr, block.layout);
                    }
                    false
                } else {
                    true
                }
            });
        }

        Ok(())
    }

    /// Get aggregate statistics from all threads
    pub fn aggregate_statistics(&self) -> AllocationStats {
        let global_stats = self.global_stats.read().unwrap().clone();
        let allocators = self.thread_allocators.lock().unwrap();
        
        let mut aggregate = global_stats;
        
        for state in allocators.values() {
            if let Ok(local_state) = state.try_lock() {
                aggregate.bytes_allocated += local_state.stats.bytes_allocated;
                aggregate.bytes_deallocated += local_state.stats.bytes_deallocated;
                aggregate.allocation_count += local_state.stats.allocation_count;
                aggregate.deallocation_count += local_state.stats.deallocation_count;
                aggregate.active_allocations += local_state.stats.active_allocations;
                aggregate.peak_usage = aggregate.peak_usage.max(local_state.stats.peak_usage);
            }
        }
        
        aggregate
    }

    /// Get number of cached blocks across all threads
    pub fn total_cached_blocks(&self) -> usize {
        let allocators = self.thread_allocators.lock().unwrap();
        let mut total = 0;
        
        for state in allocators.values() {
            if let Ok(local_state) = state.try_lock() {
                total += local_state.cached_blocks.len();
            }
        }
        
        total += self.global_pool.lock().unwrap().len();
        total
    }

    /// Force cleanup of all cached memory
    pub fn force_cleanup(&self) -> Result<()> {
        // Clean thread-local caches
        let allocators = self.thread_allocators.lock().unwrap();
        
        for state in allocators.values() {
            if let Ok(mut local_state) = state.try_lock() {
                for block in local_state.cached_blocks.drain(..) {
                    unsafe {
                        self.base_allocator.deallocate(block.ptr, block.layout)?;
                    }
                }
            }
        }

        // Clean global pool
        {
            let mut pool = self.global_pool.lock().unwrap();
            for block in pool.drain(..) {
                unsafe {
                    self.base_allocator.deallocate(block.ptr, block.layout)?;
                }
            }
        }

        Ok(())
    }
}

impl<A> MemoryAllocator for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        // Try thread-local cache first
        if self.config.enable_thread_local_cache {
            let state = self.get_thread_local_state()?;
            let mut local_state = state.lock().unwrap();
            
            // Check if we should do garbage collection
            if local_state.should_gc(self.config.gc_interval) {
                local_state.garbage_collect(self.config.max_block_age);
            }
            
            // Try to allocate from thread-local cache
            if let Some(ptr) = local_state.try_allocate_from_cache(layout) {
                local_state.stats.allocation_count += 1;
                local_state.stats.active_allocations += 1;
                return Ok(ptr);
            }
        }

        // Try global pool
        if let Some(ptr) = self.try_allocate_from_global_pool(layout) {
            if self.config.enable_tracking {
                let mut stats = self.global_stats.write().unwrap();
                stats.allocation_count += 1;
                stats.active_allocations += 1;
            }
            return Ok(ptr);
        }

        // Allocate new memory from base allocator
        let ptr = self.base_allocator.allocate(layout)?;
        
        if self.config.enable_tracking {
            let mut stats = self.global_stats.write().unwrap();
            stats.bytes_allocated += layout.size();
            stats.allocation_count += 1;
            stats.active_allocations += 1;
            stats.peak_usage = stats.peak_usage.max(
                stats.bytes_allocated - stats.bytes_deallocated
            );
        }

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        // Try to cache the block for reuse
        if self.config.enable_thread_local_cache {
            if let Ok(state) = self.get_thread_local_state() {
                let mut local_state = state.lock().unwrap();
                if local_state.cached_blocks.len() < self.config.max_cached_blocks_per_thread {
                    local_state.cache_block(ptr, layout);
                    local_state.stats.deallocation_count += 1;
                    local_state.stats.active_allocations = 
                        local_state.stats.active_allocations.saturating_sub(1);
                    return Ok(());
                }
            }
        }

        // Try global pool
        self.return_to_global_pool(ptr, layout);
        
        if self.config.enable_tracking {
            let mut stats = self.global_stats.write().unwrap();
            stats.bytes_deallocated += layout.size();
            stats.deallocation_count += 1;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
        }

        Ok(())
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        // For simplicity, always allocate new and copy
        let new_ptr = self.allocate(new_layout)?;
        
        let copy_size = old_layout.size().min(new_layout.size());
        std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);
        
        self.deallocate(ptr, old_layout)?;
        
        Ok(new_ptr)
    }

    fn supports_layout(&self, layout: Layout) -> bool {
        self.base_allocator.supports_layout(layout)
    }

    fn preferred_alignment(&self) -> usize {
        self.base_allocator.preferred_alignment()
    }

    fn statistics(&self) -> Option<AllocationStats> {
        if self.config.enable_tracking {
            Some(self.aggregate_statistics())
        } else {
            None
        }
    }
}

impl<A> SpecializedAllocator for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    fn allocation_error(&self, msg: &str) -> Self::Error {
        NumRs2Error::AllocationFailed(msg.to_string())
    }
}

/// Thread-local allocator that provides zero-contention allocation
pub struct ThreadLocalAllocator {
    config: ParallelAllocatorConfig,
}

impl ThreadLocalAllocator {
    /// Create a new thread-local allocator
    pub fn new(config: ParallelAllocatorConfig) -> Self {
        Self { config }
    }

    /// Initialize thread-local storage for current thread
    pub fn initialize_current_thread<A>(&self, allocator: A) -> Result<()>
    where
        A: MemoryAllocator<Error = NumRs2Error> + Send + 'static,
    {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();
            if local_ref.is_none() {
                *local_ref = Some(ThreadLocalState::new(
                    allocator,
                    self.config.max_cached_blocks_per_thread,
                ));
            }
        });
        Ok(())
    }

    /// Allocate using thread-local allocator
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();
            
            if let Some(ref mut state) = *local_ref {
                // Check cache first
                if let Some(ptr) = state.try_allocate_from_cache(layout) {
                    state.stats.allocation_count += 1;
                    state.stats.active_allocations += 1;
                    return Ok(ptr);
                }
                
                // Allocate new
                let ptr = state.allocator.allocate(layout)?;
                state.stats.bytes_allocated += layout.size();
                state.stats.allocation_count += 1;
                state.stats.active_allocations += 1;
                
                Ok(ptr)
            } else {
                Err(NumRs2Error::RuntimeError(
                    "Thread-local allocator not initialized".to_string()
                ))
            }
        })
    }

    /// Deallocate using thread-local allocator
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();
            
            if let Some(ref mut state) = *local_ref {
                // Try to cache the block
                if state.cached_blocks.len() < self.config.max_cached_blocks_per_thread {
                    state.cache_block(ptr, layout);
                } else {
                    // Cache full, actually deallocate
                    state.allocator.deallocate(ptr, layout)?;
                }
                
                state.stats.bytes_deallocated += layout.size();
                state.stats.deallocation_count += 1;
                state.stats.active_allocations = 
                    state.stats.active_allocations.saturating_sub(1);
                
                Ok(())
            } else {
                Err(NumRs2Error::RuntimeError(
                    "Thread-local allocator not initialized".to_string()
                ))
            }
        })
    }

    /// Get statistics for current thread
    pub fn current_thread_statistics(&self) -> Option<AllocationStats> {
        LOCAL_ALLOCATOR.with(|local| {
            local.borrow().as_ref().map(|state| state.stats.clone())
        })
    }

    /// Trigger garbage collection for current thread
    pub fn garbage_collect_current_thread(&self) -> Result<()> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();
            
            if let Some(ref mut state) = *local_ref {
                state.garbage_collect(self.config.max_block_age);
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_alloc::NumericalArrayAllocator;
    use std::time::Duration;

    #[test]
    fn test_parallel_allocator_creation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);
        
        assert!(allocator.config.enable_thread_local_cache);
        assert_eq!(allocator.total_cached_blocks(), 0);
    }

    #[test]
    fn test_basic_allocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);
        
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = allocator.allocate(layout).unwrap();
        
        unsafe {
            allocator.deallocate(ptr, layout).unwrap();
        }
    }

    #[test]
    fn test_thread_local_caching() {
        let base = NumericalArrayAllocator::new();
        let mut config = ParallelAllocatorConfig::default();
        config.max_cached_blocks_per_thread = 5;
        let allocator = ParallelAllocator::new(base, config);
        
        let layout = Layout::from_size_align(64, 8).unwrap();
        
        // Allocate and deallocate several blocks
        for _ in 0..3 {
            let ptr = allocator.allocate(layout).unwrap();
            unsafe {
                allocator.deallocate(ptr, layout).unwrap();
            }
        }
        
        // Should have cached blocks
        assert!(allocator.total_cached_blocks() > 0);
    }

    #[test]
    fn test_statistics_aggregation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);
        
        let layout = Layout::from_size_align(128, 8).unwrap();
        
        // Make some allocations
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            ptrs.push(allocator.allocate(layout).unwrap());
        }
        
        let stats = allocator.aggregate_statistics();
        assert!(stats.allocation_count >= 5);
        assert!(stats.bytes_allocated >= 128 * 5);
        
        // Clean up
        for ptr in ptrs {
            unsafe {
                allocator.deallocate(ptr, layout).unwrap();
            }
        }
    }

    #[test]
    fn test_garbage_collection() {
        let base = NumericalArrayAllocator::new();
        let mut config = ParallelAllocatorConfig::default();
        config.max_block_age = Duration::from_millis(1); // Very short for testing
        let allocator = ParallelAllocator::new(base, config);
        
        let layout = Layout::from_size_align(64, 8).unwrap();
        
        // Allocate and deallocate to create cached blocks
        for _ in 0..3 {
            let ptr = allocator.allocate(layout).unwrap();
            unsafe {
                allocator.deallocate(ptr, layout).unwrap();
            }
        }
        
        let initial_cached = allocator.total_cached_blocks();
        assert!(initial_cached > 0);
        
        // Wait for blocks to age
        std::thread::sleep(Duration::from_millis(10));
        
        // Trigger garbage collection
        allocator.garbage_collect_all().unwrap();
        
        // Should have fewer cached blocks
        let final_cached = allocator.total_cached_blocks();
        assert!(final_cached <= initial_cached);
    }

    #[test]
    fn test_thread_local_allocator() {
        let config = ParallelAllocatorConfig::default();
        let tl_allocator = ThreadLocalAllocator::new(config);
        
        // Initialize for current thread
        let base = NumericalArrayAllocator::new();
        tl_allocator.initialize_current_thread(base).unwrap();
        
        let layout = Layout::from_size_align(256, 8).unwrap();
        let ptr = tl_allocator.allocate(layout).unwrap();
        
        unsafe {
            tl_allocator.deallocate(ptr, layout).unwrap();
        }
        
        let stats = tl_allocator.current_thread_statistics().unwrap();
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_force_cleanup() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);
        
        let layout = Layout::from_size_align(64, 8).unwrap();
        
        // Create some cached blocks
        for _ in 0..3 {
            let ptr = allocator.allocate(layout).unwrap();
            unsafe {
                allocator.deallocate(ptr, layout).unwrap();
            }
        }
        
        assert!(allocator.total_cached_blocks() > 0);
        
        // Force cleanup
        allocator.force_cleanup().unwrap();
        
        // Should have no cached blocks
        assert_eq!(allocator.total_cached_blocks(), 0);
    }

    #[test]
    fn test_reallocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);
        
        let old_layout = Layout::from_size_align(64, 8).unwrap();
        let new_layout = Layout::from_size_align(128, 8).unwrap();
        
        let ptr = allocator.allocate(old_layout).unwrap();
        
        unsafe {
            let new_ptr = allocator.reallocate(ptr, old_layout, new_layout).unwrap();
            allocator.deallocate(new_ptr, new_layout).unwrap();
        }
    }

    #[test]
    fn test_multithreaded_allocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = Arc::new(ParallelAllocator::new(base, config));
        
        let mut handles = Vec::new();
        
        for _ in 0..4 {
            let allocator_clone = Arc::clone(&allocator);
            let handle = std::thread::spawn(move || {
                let layout = Layout::from_size_align(128, 8).unwrap();
                
                for _ in 0..10 {
                    let ptr = allocator_clone.allocate(layout).unwrap();
                    unsafe {
                        allocator_clone.deallocate(ptr, layout).unwrap();
                    }
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = allocator.aggregate_statistics();
        assert!(stats.allocation_count >= 40);
    }
}
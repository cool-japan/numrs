//! Memory pool allocator for fast allocation of fixed-size blocks
//!
//! The pool allocator pre-allocates memory blocks of a fixed size and
//! provides fast allocation/deallocation with minimal overhead.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Configuration for the memory pool allocator
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Size of each memory block in bytes
    pub block_size: usize,
    /// Number of blocks to pre-allocate
    pub initial_blocks: usize,
    /// Maximum number of blocks the pool can grow to (None = unlimited)
    pub max_blocks: Option<usize>,
    /// Whether to automatically resize the pool when out of blocks
    pub auto_resize: bool,
    /// Growth factor when auto-resizing (e.g., 2.0 = double the size)
    pub growth_factor: f64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            block_size: 4096,       // 4KB blocks by default
            initial_blocks: 16,     // Start with 16 blocks (64KB total)
            max_blocks: Some(1024), // Limit to 1024 blocks (4MB total)
            auto_resize: true,      // Automatically resize when needed
            growth_factor: 2.0,     // Double the size when resizing
        }
    }
}

/// A memory block in the pool
#[derive(Debug)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    layout: Layout,
}

// Mark MemoryBlock as Send and Sync
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    /// Create a new memory block with the given layout
    fn new(layout: Layout) -> Option<Self> {
        unsafe {
            let ptr = NonNull::new(alloc(layout))?;
            Some(Self { ptr, layout })
        }
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Internal pool state
#[derive(Debug)]
struct PoolState {
    /// Configuration for this pool
    config: PoolConfig,
    /// Pre-allocated memory chunks
    memory: Vec<MemoryBlock>,
    /// Free list of available block indices
    free_list: Vec<usize>,
    /// Total number of blocks currently allocated
    total_blocks: usize,
    /// Number of blocks currently in use
    used_blocks: usize,
}

/// A memory pool allocator for numerical computing
///
/// Provides efficient allocation and deallocation of fixed-size memory blocks,
/// optimized for the frequent allocation patterns found in numerical computing.
#[derive(Debug)]
pub struct PoolAllocator {
    /// Internal state wrapped in thread-safe containers
    state: Arc<Mutex<UnsafeCell<PoolState>>>,
}

impl PoolAllocator {
    /// Create a new memory pool with the given configuration
    pub fn new(config: PoolConfig) -> Self {
        let mut state = PoolState {
            config: config.clone(),
            memory: Vec::with_capacity(config.initial_blocks),
            free_list: Vec::with_capacity(config.initial_blocks),
            total_blocks: 0,
            used_blocks: 0,
        };

        // Pre-allocate the initial blocks
        let layout = Layout::from_size_align(config.block_size, 8).unwrap_or_else(|_| {
            // Fallback to a safe default layout if the provided parameters are invalid
            Layout::from_size_align(1024, 8).unwrap()
        });

        for _ in 0..config.initial_blocks {
            if let Some(block) = MemoryBlock::new(layout) {
                state.free_list.push(state.memory.len());
                state.memory.push(block);
                state.total_blocks += 1;
            }
        }

        Self {
            state: Arc::new(Mutex::new(UnsafeCell::new(state))),
        }
    }

    /// Allocate a block of memory from the pool
    ///
    /// Returns a pointer to the allocated memory, or None if allocation fails
    pub fn allocate(&self) -> Option<NonNull<u8>> {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &mut *state_mutex.get() };

        // Try to get a block from the free list
        if let Some(index) = state.free_list.pop() {
            state.used_blocks += 1;
            return Some(state.memory[index].ptr);
        }

        // No free blocks - check if we can grow the pool
        if state.config.auto_resize {
            let can_grow = match state.config.max_blocks {
                Some(max) => state.total_blocks < max,
                None => true,
            };

            if can_grow {
                // Calculate how many new blocks to add
                let growth = (state.total_blocks as f64 * (state.config.growth_factor - 1.0)).ceil()
                    as usize;
                let new_blocks = growth.max(1); // At least 1 new block
                let max_new = match state.config.max_blocks {
                    Some(max) => (max - state.total_blocks).min(new_blocks),
                    None => new_blocks,
                };

                // Allocate new blocks
                let layout = Layout::from_size_align(state.config.block_size, 8)
                    .unwrap_or_else(|_| Layout::from_size_align(1024, 8).unwrap());

                for _ in 0..max_new {
                    if let Some(block) = MemoryBlock::new(layout) {
                        state.free_list.push(state.memory.len());
                        state.memory.push(block);
                        state.total_blocks += 1;
                    }
                }

                // Try again to get a block from the free list
                if let Some(index) = state.free_list.pop() {
                    state.used_blocks += 1;
                    return Some(state.memory[index].ptr);
                }
            }
        }

        // Couldn't allocate a new block
        None
    }

    /// Return a block of memory to the pool
    ///
    /// # Safety
    ///
    /// The pointer must have been previously returned by `allocate` and not already freed.
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>) {
        let state_mutex = self.state.lock().unwrap();
        let state = &mut *state_mutex.get();

        // Find the block index
        for (i, block) in state.memory.iter().enumerate() {
            if block.ptr.as_ptr() == ptr.as_ptr() {
                state.free_list.push(i);
                state.used_blocks -= 1;
                return;
            }
        }

        // If we get here, it's an error - the pointer wasn't from this pool
        panic!("Attempted to free a pointer that wasn't allocated from this pool");
    }

    /// Get the number of available blocks in the pool
    pub fn available_blocks(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.free_list.len()
    }

    /// Get the total number of blocks in the pool
    pub fn total_blocks(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.total_blocks
    }

    /// Get the number of blocks currently in use
    pub fn used_blocks(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.used_blocks
    }

    /// Get the block size in bytes
    pub fn block_size(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.config.block_size
    }

    /// Reset the pool, returning all blocks to the free list
    pub fn reset(&self) {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &mut *state_mutex.get() };

        // Clear the free list and rebuild it
        state.free_list.clear();
        for i in 0..state.memory.len() {
            state.free_list.push(i);
        }
        state.used_blocks = 0;
    }
}

// For testing - don't export
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocator_basic() {
        let config = PoolConfig {
            block_size: 1024,
            initial_blocks: 4,
            max_blocks: Some(8),
            auto_resize: true,
            growth_factor: 2.0,
        };

        let pool = PoolAllocator::new(config);

        // Should have 4 available blocks initially
        assert_eq!(pool.available_blocks(), 4);
        assert_eq!(pool.total_blocks(), 4);
        assert_eq!(pool.used_blocks(), 0);

        // Allocate all 4 blocks
        let mut ptrs = Vec::new();
        for _ in 0..4 {
            let ptr = pool.allocate().expect("Allocation should succeed");
            ptrs.push(ptr);
        }

        // Should have 0 available blocks now
        assert_eq!(pool.available_blocks(), 0);
        assert_eq!(pool.used_blocks(), 4);

        // Allocation should trigger auto-resize
        let ptr5 = pool.allocate().expect("Allocation should trigger resize");
        ptrs.push(ptr5);

        // Should have grown to 8 blocks total, with 3 available
        assert_eq!(pool.total_blocks(), 8);
        assert_eq!(pool.available_blocks(), 3);
        assert_eq!(pool.used_blocks(), 5);

        // Free a block
        unsafe {
            pool.deallocate(ptrs[0]);
        }

        // Should have 4 available blocks now
        assert_eq!(pool.available_blocks(), 4);
        assert_eq!(pool.used_blocks(), 4);

        // Reset the pool
        pool.reset();

        // All blocks should be available
        assert_eq!(pool.available_blocks(), 8);
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    fn test_pool_allocator_max_size() {
        // Create a pool with max_blocks = initial_blocks (can't grow)
        let config = PoolConfig {
            block_size: 64,
            initial_blocks: 2,
            max_blocks: Some(2),
            auto_resize: true, // Should be ignored due to max_blocks
            growth_factor: 2.0,
        };

        let pool = PoolAllocator::new(config);

        // Allocate all blocks
        let ptr1 = pool.allocate().expect("First allocation should succeed");
        let ptr2 = pool.allocate().expect("Second allocation should succeed");

        // Third allocation should fail because we can't grow
        assert!(pool.allocate().is_none());

        // Free blocks and verify they're returned to the pool
        unsafe {
            pool.deallocate(ptr1);
            pool.deallocate(ptr2);
        }

        assert_eq!(pool.available_blocks(), 2);
    }
}

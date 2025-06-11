//! Arena allocator for efficient bulk memory allocation
//!
//! The arena allocator provides fast allocation from a contiguous memory region
//! with minimal overhead, ideal for temporary allocations in numerical algorithms.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Configuration for the arena allocator
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Initial capacity of the arena in bytes
    pub initial_size: usize,
    /// Whether to grow the arena when it's full
    pub allow_growth: bool,
    /// Growth factor when resizing (e.g., 2.0 = double the size)
    pub growth_factor: f64,
    /// Alignment for allocations from the arena
    pub alignment: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024, // 1MB default size
            allow_growth: true,        // Allow growing by default
            growth_factor: 2.0,        // Double size when growing
            alignment: 8,              // 8-byte alignment by default
        }
    }
}

/// A memory chunk in the arena
struct ArenaChunk {
    /// Pointer to the memory
    ptr: NonNull<u8>,
    /// Total size of the chunk
    size: usize,
    /// Current offset within the chunk
    offset: usize,
    /// Layout (for deallocation)
    layout: Layout,
}

// Mark ArenaChunk as Send and Sync
unsafe impl Send for ArenaChunk {}
unsafe impl Sync for ArenaChunk {}

impl ArenaChunk {
    /// Create a new memory chunk with the specified size and alignment
    fn new(size: usize, alignment: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, alignment).ok()?;
        unsafe {
            let ptr = NonNull::new(alloc(layout))?;
            Some(Self {
                ptr,
                size,
                offset: 0,
                layout,
            })
        }
    }

    /// Try to allocate `size` bytes from this chunk
    fn allocate(&mut self, size: usize, alignment: usize) -> Option<NonNull<u8>> {
        // Calculate aligned offset
        let align_mask = alignment - 1;
        let aligned_offset = (self.offset + align_mask) & !align_mask;

        // Check if we have enough space
        if aligned_offset + size <= self.size {
            // We have enough space
            let result = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(aligned_offset)) };
            self.offset = aligned_offset + size;
            Some(result)
        } else {
            // Not enough space
            None
        }
    }

    /// Get the amount of free space in this chunk
    fn available_space(&self) -> usize {
        self.size - self.offset
    }

    /// Reset the chunk, making all memory available again
    fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Internal state for the arena allocator
struct ArenaState {
    /// Configuration for this arena
    config: ArenaConfig,
    /// Memory chunks that make up the arena
    chunks: Vec<ArenaChunk>,
    /// Total bytes allocated
    total_allocated: usize,
}

/// A memory arena allocator for numerical computing
///
/// Provides efficient allocation from pre-allocated memory chunks,
/// with minimal overhead. Ideal for algorithms that need many small,
/// temporary allocations.
pub struct ArenaAllocator {
    /// Internal state wrapped in thread-safe containers
    state: Arc<Mutex<UnsafeCell<ArenaState>>>,
}

impl ArenaAllocator {
    /// Create a new arena allocator with the given configuration
    pub fn new(config: ArenaConfig) -> Self {
        let mut state = ArenaState {
            config: config.clone(),
            chunks: Vec::new(),
            total_allocated: 0,
        };

        // Allocate the initial chunk
        if let Some(chunk) = ArenaChunk::new(config.initial_size, config.alignment) {
            state.total_allocated += config.initial_size;
            state.chunks.push(chunk);
        } else {
            panic!("Failed to allocate initial arena chunk");
        }

        Self {
            state: Arc::new(Mutex::new(UnsafeCell::new(state))),
        }
    }

    /// Allocate memory from the arena
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        self.allocate_aligned(size, self.get_alignment())
    }

    /// Allocate aligned memory from the arena
    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }

        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &mut *state_mutex.get() };

        // Try to allocate from existing chunks
        for chunk in &mut state.chunks {
            if let Some(ptr) = chunk.allocate(size, alignment) {
                return Some(ptr);
            }
        }

        // All chunks are full, try to add a new one if allowed
        if state.config.allow_growth {
            // Calculate the size for the new chunk
            let current_total = state.total_allocated;
            let growth =
                (current_total as f64 * (state.config.growth_factor - 1.0)).ceil() as usize;
            let new_size = growth.max(size.max(state.config.initial_size));

            // Allocate new chunk
            if let Some(mut new_chunk) =
                ArenaChunk::new(new_size, alignment.max(state.config.alignment))
            {
                // Allocate from the new chunk
                let result = new_chunk.allocate(size, alignment);
                state.total_allocated += new_size;
                state.chunks.push(new_chunk);
                return result;
            }
        }

        // Couldn't allocate
        None
    }

    /// Reset the arena, reclaiming all allocated memory for reuse
    pub fn reset(&self) {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &mut *state_mutex.get() };

        // Reset all chunks
        for chunk in &mut state.chunks {
            chunk.reset();
        }
    }

    /// Get the total amount of memory allocated by the arena
    pub fn total_size(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.total_allocated
    }

    /// Get the amount of memory currently available in the arena
    pub fn available_size(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state
            .chunks
            .iter()
            .map(|chunk| chunk.available_space())
            .sum()
    }

    /// Get the default alignment used by this arena
    pub fn get_alignment(&self) -> usize {
        let state_mutex = self.state.lock().unwrap();
        let state = unsafe { &*state_mutex.get() };
        state.config.alignment
    }

    /// Allocate a typed object from the arena
    pub fn allocate_object<T>(&self) -> Option<NonNull<T>> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        if size == 0 {
            // Zero-sized types don't need allocation
            return None;
        }

        self.allocate_aligned(size, align)
            .map(|ptr| unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) })
    }

    /// Create a scoped arena allocator that will reset when dropped
    pub fn scoped(&self) -> ScopedArena {
        ScopedArena {
            arena: self.clone(),
        }
    }
}

impl Clone for ArenaAllocator {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

/// A scoped arena that automatically resets when dropped
pub struct ScopedArena {
    arena: ArenaAllocator,
}

impl ScopedArena {
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        self.arena.allocate(size)
    }

    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Option<NonNull<u8>> {
        self.arena.allocate_aligned(size, alignment)
    }

    pub fn allocate_object<T>(&self) -> Option<NonNull<T>> {
        self.arena.allocate_object::<T>()
    }
}

impl Drop for ScopedArena {
    fn drop(&mut self) {
        self.arena.reset();
    }
}

/// An arena-allocated vector
pub struct ArenaVec<'a, T> {
    /// Pointer to the arena-allocated buffer
    ptr: NonNull<T>,
    /// Current length of the vector
    len: usize,
    /// Capacity of the vector
    capacity: usize,
    /// Reference to the parent arena
    arena: &'a ArenaAllocator,
    /// Phantom data for variance over the lifetime
    phantom: PhantomData<&'a mut [T]>,
}

impl<'a, T> ArenaVec<'a, T> {
    /// Create a new empty vector with the given capacity
    pub fn with_capacity(capacity: usize, arena: &'a ArenaAllocator) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                arena,
                phantom: PhantomData,
            };
        }

        // Allocate memory from the arena
        let size = capacity * mem::size_of::<T>();
        let ptr = arena
            .allocate_aligned(size, mem::align_of::<T>())
            .expect("Arena allocation failed");

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) },
            len: 0,
            capacity,
            arena,
            phantom: PhantomData,
        }
    }

    /// Add an element to the end of the vector
    pub fn push(&mut self, value: T) -> bool {
        if self.len >= self.capacity {
            // No more space - try to grow
            let new_capacity = self.capacity.saturating_mul(2).max(4);
            if !self.reserve(new_capacity - self.capacity) {
                return false;
            }
        }

        unsafe {
            // Write value to the end of the buffer
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
        }

        self.len += 1;
        true
    }

    /// Extend capacity of the vector
    pub fn reserve(&mut self, additional: usize) -> bool {
        if additional == 0 {
            return true;
        }

        let new_capacity = self.capacity + additional;
        let new_size = new_capacity * mem::size_of::<T>();

        if let Some(new_ptr) = self.arena.allocate_aligned(new_size, mem::align_of::<T>()) {
            // Copy existing elements to the new buffer
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    new_ptr.as_ptr() as *mut T,
                    self.len,
                );
            }

            // Update pointer and capacity
            self.ptr = unsafe { NonNull::new_unchecked(new_ptr.as_ptr() as *mut T) };
            self.capacity = new_capacity;
            true
        } else {
            false
        }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice of the vector's contents
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the vector's contents
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an item at a specific index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Get a mutable reference to an item at a specific index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Clear the vector without deallocating memory
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = 0;
    }
}

impl<'a, T: Clone> ArenaVec<'a, T> {
    /// Create a vector from a slice
    pub fn from_slice(slice: &[T], arena: &'a ArenaAllocator) -> Self {
        let mut vec = Self::with_capacity(slice.len(), arena);
        for item in slice {
            vec.push(item.clone());
        }
        vec
    }
}

impl<T> Drop for ArenaVec<'_, T> {
    fn drop(&mut self) {
        // Only need to drop the elements, not deallocate the memory
        // as that's handled by the arena
        self.clear();
    }
}

// For testing - don't export
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocator_basic() {
        let config = ArenaConfig {
            initial_size: 1024,
            allow_growth: true,
            growth_factor: 2.0,
            alignment: 8,
        };

        let arena = ArenaAllocator::new(config);

        // Should have 1024 bytes initially
        assert_eq!(arena.total_size(), 1024);
        assert_eq!(arena.available_size(), 1024);

        // Allocate 100 bytes
        let ptr1 = arena.allocate(100).expect("Allocation should succeed");
        assert!(arena.available_size() < 1024);

        // Allocate another 100 bytes
        let ptr2 = arena.allocate(100).expect("Allocation should succeed");
        assert!(ptr1 != ptr2);

        // Allocate a large block that exceeds current capacity
        let ptr3 = arena
            .allocate(1000)
            .expect("Allocation should trigger growth");
        assert!(ptr1 != ptr3 && ptr2 != ptr3);

        // Check that we've grown
        assert!(arena.total_size() > 1024);

        // Reset and check that all space is available again
        arena.reset();
        assert_eq!(arena.available_size(), arena.total_size());
    }

    #[test]
    fn test_arena_allocator_alignment() {
        let config = ArenaConfig {
            initial_size: 1024,
            allow_growth: true,
            growth_factor: 2.0,
            alignment: 8,
        };

        let arena = ArenaAllocator::new(config);

        // Allocate with different alignments
        let ptr1 = arena
            .allocate_aligned(10, 1)
            .expect("Allocation should succeed");
        let ptr2 = arena
            .allocate_aligned(10, 4)
            .expect("Allocation should succeed");
        let ptr3 = arena
            .allocate_aligned(10, 8)
            .expect("Allocation should succeed");
        let ptr4 = arena
            .allocate_aligned(10, 16)
            .expect("Allocation should succeed");

        // Check alignments
        assert_eq!(ptr1.as_ptr() as usize % 1, 0);
        assert_eq!(ptr2.as_ptr() as usize % 4, 0);
        assert_eq!(ptr3.as_ptr() as usize % 8, 0);
        assert_eq!(ptr4.as_ptr() as usize % 16, 0);
    }

    #[test]
    fn test_arena_vec() {
        let config = ArenaConfig::default();
        let arena = ArenaAllocator::new(config);

        // Create a vector with initial capacity
        let mut vec: ArenaVec<i32> = ArenaVec::with_capacity(10, &arena);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);

        // Push elements
        for i in 0..15 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 15);
        assert!(vec.capacity() >= 15);

        // Check contents
        for i in 0..15 {
            assert_eq!(vec.get(i), Some(&(i as i32)));
        }

        // Modify elements
        for i in 0..vec.len() {
            if let Some(value) = vec.get_mut(i) {
                *value *= 2;
            }
        }

        // Check modified contents
        for i in 0..15 {
            assert_eq!(vec.get(i), Some(&((i as i32) * 2)));
        }

        // Create vector from slice
        let source = [1, 2, 3, 4, 5];
        let vec2 = ArenaVec::from_slice(&source, &arena);
        assert_eq!(vec2.len(), 5);
        assert_eq!(vec2.as_slice(), &source[..]);
    }

    #[test]
    fn test_scoped_arena() {
        let config = ArenaConfig::default();
        let arena = ArenaAllocator::new(config);

        // Initial state
        let initial_available = arena.available_size();

        // Create a scope and allocate within it
        {
            let scoped = arena.scoped();
            let _ptr1 = scoped.allocate(100).expect("Allocation should succeed");
            let _ptr2 = scoped.allocate(100).expect("Allocation should succeed");

            // Should have less available space
            assert!(arena.available_size() < initial_available);
        }

        // After the scope, should have reset
        assert_eq!(arena.available_size(), initial_available);
    }
}

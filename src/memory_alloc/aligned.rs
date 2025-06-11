//! Aligned memory allocator for SIMD and cache-efficient operations
//!
//! This module provides allocators that ensure memory is aligned to specific
//! boundaries for optimal performance with SIMD and cache operations.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::mem;
use std::ptr::NonNull;

/// Alignment configuration for memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentConfig {
    /// Alignment in bytes (must be a power of 2)
    pub alignment: usize,
    /// Whether to zero-initialize allocated memory
    pub zero_init: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            alignment: 64, // Default to cache line size on most CPUs
            zero_init: false,
        }
    }
}

/// Common alignment values
impl AlignmentConfig {
    /// Create a new alignment configuration
    pub fn new(alignment: usize, zero_init: bool) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of 2"
        );
        Self {
            alignment,
            zero_init,
        }
    }

    /// Default cache line alignment (64 bytes)
    pub fn cache_line() -> Self {
        Self {
            alignment: 64,
            zero_init: false,
        }
    }

    /// SIMD-friendly alignment (16 bytes for 128-bit SIMD)
    pub fn simd_128() -> Self {
        Self {
            alignment: 16,
            zero_init: false,
        }
    }

    /// SIMD-friendly alignment (32 bytes for 256-bit SIMD)
    pub fn simd_256() -> Self {
        Self {
            alignment: 32,
            zero_init: false,
        }
    }

    /// SIMD-friendly alignment (64 bytes for 512-bit SIMD)
    pub fn simd_512() -> Self {
        Self {
            alignment: 64,
            zero_init: false,
        }
    }

    /// Page-aligned memory (4KB on most systems)
    pub fn page() -> Self {
        Self {
            alignment: 4096,
            zero_init: false,
        }
    }

    /// Zero-initialized variant
    pub fn zeroed(mut self) -> Self {
        self.zero_init = true;
        self
    }
}

/// An allocator for aligned memory
pub struct AlignedAllocator {
    config: AlignmentConfig,
}

impl AlignedAllocator {
    /// Create a new aligned allocator with the given configuration
    pub fn new(config: AlignmentConfig) -> Self {
        Self { config }
    }

    /// Allocate memory with the configured alignment
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }

        let layout = Layout::from_size_align(size, self.config.alignment).ok()?;

        unsafe {
            let ptr = if self.config.zero_init {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            };

            NonNull::new(ptr)
        }
    }

    /// Allocate memory for the given type with appropriate alignment
    pub fn allocate_for_type<T>(&self) -> Option<NonNull<T>> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>().max(self.config.alignment);

        if size == 0 {
            return None;
        }

        let layout = Layout::from_size_align(size, align).ok()?;

        unsafe {
            let ptr = if self.config.zero_init {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            };

            NonNull::new(ptr as *mut T)
        }
    }

    /// Allocate an array of elements with the configured alignment
    pub fn allocate_array<T>(&self, count: usize) -> Option<NonNull<T>> {
        let size = mem::size_of::<T>().checked_mul(count)?;
        let align = mem::align_of::<T>().max(self.config.alignment);

        if size == 0 {
            return None;
        }

        let layout = Layout::from_size_align(size, align).ok()?;

        unsafe {
            let ptr = if self.config.zero_init {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            };

            NonNull::new(ptr as *mut T)
        }
    }

    /// Deallocate memory that was allocated with this allocator
    ///
    /// # Safety
    ///
    /// - The pointer must have been allocated by this allocator
    /// - The size and alignment must match the original allocation
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let layout = Layout::from_size_align_unchecked(size, self.config.alignment);
        dealloc(ptr.as_ptr(), layout);
    }

    /// Deallocate an array that was allocated with this allocator
    ///
    /// # Safety
    ///
    /// - The pointer must have been allocated by this allocator
    /// - The type and count must match the original allocation
    pub unsafe fn deallocate_array<T>(&self, ptr: NonNull<T>, count: usize) {
        let size = mem::size_of::<T>() * count;
        let align = mem::align_of::<T>().max(self.config.alignment);
        let layout = Layout::from_size_align_unchecked(size, align);
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }

    /// Get the alignment used by this allocator
    pub fn alignment(&self) -> usize {
        self.config.alignment
    }

    /// Check if allocations are zero-initialized
    pub fn is_zero_initialized(&self) -> bool {
        self.config.zero_init
    }

    /// Create a new aligned array and initialize it with the given values
    pub fn create_array<T: Copy>(&self, values: &[T]) -> Option<NonNull<T>> {
        let ptr = self.allocate_array::<T>(values.len())?;

        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.as_ptr(), values.len());
        }

        Some(ptr)
    }

    /// Create a reference to the aligned array
    ///
    /// # Safety
    ///
    /// - The pointer must have been allocated by this allocator
    /// - The count must match the original allocation
    /// - The memory must be properly initialized
    pub unsafe fn as_slice<T>(&self, ptr: NonNull<T>, count: usize) -> &[T] {
        std::slice::from_raw_parts(ptr.as_ptr(), count)
    }

    /// Create a mutable reference to the aligned array
    ///
    /// # Safety
    ///
    /// - The pointer must have been allocated by this allocator
    /// - The count must match the original allocation
    /// - The memory must be properly initialized
    pub unsafe fn as_mut_slice<T>(&mut self, ptr: NonNull<T>, count: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(ptr.as_ptr(), count)
    }
}

/// Safe wrapper for aligned memory allocation
pub struct AlignedBox<T> {
    ptr: NonNull<T>,
    allocator: AlignedAllocator,
}

impl<T> AlignedBox<T> {
    /// Create a new aligned box with the given value
    pub fn new(value: T, alignment: usize) -> Option<Self> {
        let config = AlignmentConfig::new(alignment, false);
        let allocator = AlignedAllocator::new(config);

        let ptr = allocator.allocate_for_type::<T>()?;

        unsafe {
            std::ptr::write(ptr.as_ptr(), value);
        }

        Some(Self { ptr, allocator })
    }

    /// Get a reference to the contained value
    pub fn get(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    /// Get a mutable reference to the contained value
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }

    /// Convert into the contained value
    pub fn into_inner(self) -> T {
        // Read the value
        let value = unsafe { std::ptr::read(self.ptr.as_ptr()) };

        // Prevent the destructor from running
        std::mem::forget(self);

        value
    }

    /// Get the alignment of this allocation
    pub fn alignment(&self) -> usize {
        self.allocator.alignment()
    }
}

impl<T> Drop for AlignedBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Drop the contained value
            std::ptr::drop_in_place(self.ptr.as_ptr());

            // Deallocate the memory
            self.allocator.deallocate(
                NonNull::new_unchecked(self.ptr.as_ptr() as *mut u8),
                mem::size_of::<T>(),
            );
        }
    }
}

impl<T: Clone> Clone for AlignedBox<T> {
    fn clone(&self) -> Self {
        let value = self.get().clone();
        Self::new(value, self.allocator.alignment()).unwrap()
    }
}

/// Safe wrapper for aligned arrays
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    allocator: AlignedAllocator,
}

impl<T> AlignedVec<T> {
    /// Create a new aligned vector with the given capacity
    pub fn with_capacity(capacity: usize, alignment: usize) -> Option<Self> {
        if capacity == 0 {
            return None;
        }

        let config = AlignmentConfig::new(alignment, false);
        let allocator = AlignedAllocator::new(config);

        let ptr = allocator.allocate_array::<T>(capacity)?;

        Some(Self {
            ptr,
            len: 0,
            allocator,
        })
    }

    /// Create a new zero-initialized aligned vector with the given capacity
    pub fn with_capacity_zeroed(capacity: usize, alignment: usize) -> Option<Self>
    where
        T: Copy + Default,
    {
        if capacity == 0 {
            return None;
        }

        let config = AlignmentConfig::new(alignment, true);
        let allocator = AlignedAllocator::new(config);

        let ptr = allocator.allocate_array::<T>(capacity)?;

        Some(Self {
            ptr,
            len: capacity,
            allocator,
        })
    }

    /// Push a value to the end of the vector
    ///
    /// Returns false if there's no more capacity
    pub fn push(&mut self, value: T) -> bool {
        if self.len >= self.capacity() {
            return false;
        }

        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
        }

        self.len += 1;
        true
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the current capacity of the vector
    pub fn capacity(&self) -> usize {
        // For simplicity, we just use length as capacity
        // A real implementation would track capacity separately
        self.len
    }

    /// Get a reference to the vector as a slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable reference to the vector as a slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get the alignment of this vector
    pub fn alignment(&self) -> usize {
        self.allocator.alignment()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        unsafe {
            // Drop all elements
            for i in 0..self.len {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }

            // Deallocate the memory
            self.allocator.deallocate_array(self.ptr, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_allocator_basic() {
        // Test with 16-byte alignment
        let config = AlignmentConfig::new(16, false);
        let allocator = AlignedAllocator::new(config);

        // Allocate some memory
        let ptr = allocator.allocate(100).expect("Allocation should succeed");

        // Check alignment
        assert_eq!(
            ptr.as_ptr() as usize % 16,
            0,
            "Pointer should be 16-byte aligned"
        );

        // Deallocate
        unsafe {
            allocator.deallocate(ptr, 100);
        }

        // Test with larger alignment
        let config = AlignmentConfig::new(4096, false);
        let allocator = AlignedAllocator::new(config);

        let ptr = allocator.allocate(100).expect("Allocation should succeed");
        assert_eq!(
            ptr.as_ptr() as usize % 4096,
            0,
            "Pointer should be 4096-byte aligned"
        );

        unsafe {
            allocator.deallocate(ptr, 100);
        }
    }

    #[test]
    fn test_aligned_allocator_zero_init() {
        // Create a zero-initialized allocator
        let config = AlignmentConfig::new(64, true);
        let allocator = AlignedAllocator::new(config);

        // Allocate memory for an array of 10 integers
        let ptr = allocator
            .allocate_array::<i32>(10)
            .expect("Allocation should succeed");

        // Check that it's properly zero-initialized
        unsafe {
            let slice = std::slice::from_raw_parts(ptr.as_ptr(), 10);
            for &value in slice {
                assert_eq!(value, 0, "Value should be zero-initialized");
            }

            // Deallocate
            allocator.deallocate_array(ptr, 10);
        }
    }

    #[test]
    fn test_aligned_box() {
        // Create an aligned box with a value
        let mut aligned_box = AlignedBox::new(42i32, 16).expect("Allocation should succeed");

        // Check alignment
        let ptr_addr = aligned_box.get() as *const i32 as usize;
        assert_eq!(ptr_addr % 16, 0, "AlignedBox should be 16-byte aligned");

        // Access the value
        assert_eq!(*aligned_box.get(), 42);

        // Modify the value
        *aligned_box.get_mut() = 84;
        assert_eq!(*aligned_box.get(), 84);

        // Extract the value
        let value = aligned_box.into_inner();
        assert_eq!(value, 84);
    }

    #[test]
    fn test_aligned_vec() {
        // Create an aligned vector with capacity
        let mut vec =
            AlignedVec::<i32>::with_capacity_zeroed(5, 64).expect("Allocation should succeed");

        // Check initial state
        assert_eq!(vec.len(), 5);
        assert!(!vec.is_empty());

        // Initialize values
        for i in 0..5 {
            vec.as_mut_slice()[i] = i as i32;
        }

        // Check state after initialization
        assert_eq!(vec.len(), 5);
        assert!(!vec.is_empty());

        // Check contents
        let slice = vec.as_slice();
        assert_eq!(slice, &[0, 1, 2, 3, 4]);

        // Modify contents
        let mut_slice = vec.as_mut_slice();
        for item in mut_slice.iter_mut() {
            *item *= 2;
        }

        // Check modified contents
        assert_eq!(vec.as_slice(), &[0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_aligned_vec_zeroed() {
        // Create a zero-initialized vector
        let vec =
            AlignedVec::<i32>::with_capacity_zeroed(10, 32).expect("Allocation should succeed");

        // Check that it's properly zero-initialized
        assert_eq!(vec.len(), 10);
        for &value in vec.as_slice() {
            assert_eq!(value, 0, "Value should be zero-initialized");
        }

        // Check alignment
        assert_eq!(vec.alignment(), 32);
    }

    #[test]
    fn test_alignment_configs() {
        let cache_config = AlignmentConfig::cache_line();
        assert_eq!(cache_config.alignment, 64);
        assert!(!cache_config.zero_init);

        let simd_config = AlignmentConfig::simd_256();
        assert_eq!(simd_config.alignment, 32);
        assert!(!simd_config.zero_init);

        let page_config = AlignmentConfig::page();
        assert_eq!(page_config.alignment, 4096);
        assert!(!page_config.zero_init);

        let zeroed_config = AlignmentConfig::simd_128().zeroed();
        assert_eq!(zeroed_config.alignment, 16);
        assert!(zeroed_config.zero_init);
    }
}

//! Convenient aligned allocation helpers for cache-aligned data structures
//!
//! This module provides high-level wrapper types that make it easy to allocate
//! cache-aligned data for optimal performance in SIMD and parallel operations.

use super::aligned::{AlignedAllocator, AlignmentConfig};
use crate::error::{MemoryError, NumRs2Error, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::{fmt, mem, ptr, slice};

/// A cache-aligned heap-allocated value
///
/// `AlignedBox<T>` is similar to `Box<T>` but ensures the allocated memory
/// is aligned to a cache line boundary (64 bytes by default) for optimal
/// performance in SIMD and parallel operations.
///
/// # Examples
///
/// ```
/// use numrs2::memory_alloc::aligned_helpers::AlignedBox;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a cache-aligned box with default 64-byte alignment
/// let boxed = AlignedBox::new(42u64)?;
/// assert_eq!(*boxed, 42);
///
/// // Verify the address is aligned to 64 bytes
/// let addr = &*boxed as *const _ as usize;
/// assert_eq!(addr % 64, 0);
/// assert!(boxed.verify_alignment());
/// assert_eq!(boxed.alignment(), 64);
/// # Ok(())
/// # }
/// ```
pub struct AlignedBox<T> {
    ptr: NonNull<T>,
    alignment: usize,
}

impl<T> AlignedBox<T> {
    /// Create a new cache-aligned box with default alignment (64 bytes)
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn new(value: T) -> Result<Self> {
        Self::with_alignment(value, 64)
    }

    /// Create a new aligned box with custom alignment
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store
    /// * `alignment` - The alignment in bytes (must be a power of 2)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Allocation fails
    /// - Alignment is not a power of 2
    /// - Alignment is less than the type's natural alignment
    pub fn with_alignment(value: T, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(NumRs2Error::from(MemoryError::allocation_failed(
                "Alignment must be a power of 2",
                alignment,
            )));
        }

        let type_align = mem::align_of::<T>();
        let final_align = alignment.max(type_align);
        let size = mem::size_of::<T>();

        if size == 0 {
            // Zero-sized types don't need allocation
            return Ok(Self {
                ptr: NonNull::dangling(),
                alignment: final_align,
            });
        }

        // Create layout for aligned allocation
        let layout = Layout::from_size_align(size, final_align).map_err(|e| {
            NumRs2Error::from(MemoryError::allocation_failed(
                &format!("Invalid layout: {}", e),
                size,
            ))
        })?;

        // Allocate aligned memory
        unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(NumRs2Error::from(MemoryError::allocation_failed(
                    "Memory allocation failed",
                    size,
                )));
            }

            let ptr = NonNull::new_unchecked(raw_ptr as *mut T);

            // Write the value to the allocated memory
            ptr::write(ptr.as_ptr(), value);

            Ok(Self {
                ptr,
                alignment: final_align,
            })
        }
    }

    /// Create a new cache-aligned box for SIMD operations (AVX-512)
    pub fn new_simd_512(value: T) -> Result<Self> {
        Self::with_alignment(value, 64)
    }

    /// Create a new aligned box for SIMD operations (AVX2)
    pub fn new_simd_256(value: T) -> Result<Self> {
        Self::with_alignment(value, 32)
    }

    /// Create a new aligned box for SIMD operations (SSE)
    pub fn new_simd_128(value: T) -> Result<Self> {
        Self::with_alignment(value, 16)
    }

    /// Get the alignment of this box
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get the address of the contained value
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the contained value
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Verify that the pointer is properly aligned
    ///
    /// This is primarily for testing and debugging.
    pub fn verify_alignment(&self) -> bool {
        let addr = self.ptr.as_ptr() as usize;
        addr.is_multiple_of(self.alignment)
    }
}

impl<T> Deref for AlignedBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> DerefMut for AlignedBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Drop for AlignedBox<T> {
    fn drop(&mut self) {
        let size = mem::size_of::<T>();
        if size == 0 {
            return;
        }

        unsafe {
            // Drop the value
            ptr::drop_in_place(self.ptr.as_ptr());

            // Deallocate the memory
            let layout = Layout::from_size_align_unchecked(size, self.alignment);
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBox")
            .field("value", &**self)
            .field("alignment", &self.alignment)
            .field("address", &(self.ptr.as_ptr() as usize))
            .finish()
    }
}

impl<T: Clone> Clone for AlignedBox<T> {
    fn clone(&self) -> Self {
        Self::with_alignment((**self).clone(), self.alignment)
            .expect("Clone should not fail if original allocation succeeded")
    }
}

unsafe impl<T: Send> Send for AlignedBox<T> {}
unsafe impl<T: Sync> Sync for AlignedBox<T> {}

// ============================================================================
// AlignedVec
// ============================================================================

/// A cache-aligned vector
///
/// `AlignedVec<T>` is similar to `Vec<T>` but ensures the allocated memory
/// is aligned to a cache line boundary for optimal performance.
///
/// # Examples
///
/// ```
/// use numrs2::memory_alloc::aligned_helpers::AlignedVec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a cache-aligned vector with default 64-byte alignment
/// let mut vec = AlignedVec::with_capacity(100)?;
/// vec.push(1.0f64)?;
/// vec.push(2.0f64)?;
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec.capacity(), 100);
///
/// // Verify the data pointer is aligned to 64-byte boundary
/// let addr = vec.as_ptr() as usize;
/// assert_eq!(addr % 64, 0);
/// assert!(vec.verify_alignment());
/// assert_eq!(vec.alignment(), 64);
/// # Ok(())
/// # }
/// ```
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl<T> AlignedVec<T> {
    /// Create a new empty aligned vector with default alignment (64 bytes)
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment: 64,
        }
    }

    /// Create a new aligned vector with the specified capacity and default alignment
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_alignment(capacity, 64)
    }

    /// Create a new aligned vector with specified capacity and alignment
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity
    /// * `alignment` - Alignment in bytes (must be a power of 2)
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or alignment is invalid.
    pub fn with_capacity_and_alignment(capacity: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(NumRs2Error::from(MemoryError::allocation_failed(
                "Alignment must be a power of 2",
                alignment,
            )));
        }

        if capacity == 0 || mem::size_of::<T>() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                alignment,
            });
        }

        let type_align = mem::align_of::<T>();
        let final_align = alignment.max(type_align);
        let size = mem::size_of::<T>().checked_mul(capacity).ok_or_else(|| {
            NumRs2Error::from(MemoryError::allocation_failed(
                "Capacity overflow",
                capacity,
            ))
        })?;

        let layout = Layout::from_size_align(size, final_align).map_err(|e| {
            NumRs2Error::from(MemoryError::allocation_failed(
                &format!("Invalid layout: {}", e),
                size,
            ))
        })?;

        unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(NumRs2Error::from(MemoryError::allocation_failed(
                    "Memory allocation failed",
                    size,
                )));
            }

            Ok(Self {
                ptr: NonNull::new_unchecked(raw_ptr as *mut T),
                len: 0,
                capacity,
                alignment: final_align,
            })
        }
    }

    /// Create an aligned vector for SIMD operations (AVX-512)
    pub fn simd_512_with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_alignment(capacity, 64)
    }

    /// Create an aligned vector for SIMD operations (AVX2)
    pub fn simd_256_with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_alignment(capacity, 32)
    }

    /// Create an aligned vector for SIMD operations (SSE)
    pub fn simd_128_with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_alignment(capacity, 16)
    }

    /// Push a value onto the end of the vector
    ///
    /// # Errors
    ///
    /// Returns an error if reallocation is needed and fails.
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len == self.capacity {
            self.grow()?;
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop a value from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.len))) }
    }

    /// Get the number of elements in the vector
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

    /// Get the alignment of the vector
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get a pointer to the vector's buffer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the vector's buffer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get a slice of the vector's contents
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the vector's contents
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Clear the vector, removing all values
    pub fn clear(&mut self) {
        unsafe {
            ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ));
        }
        self.len = 0;
    }

    /// Reserve capacity for at least `additional` more elements
    ///
    /// # Errors
    ///
    /// Returns an error if reallocation fails.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let required_cap = self.len.checked_add(additional).ok_or_else(|| {
            NumRs2Error::from(MemoryError::allocation_failed(
                "Capacity overflow",
                additional,
            ))
        })?;

        if required_cap <= self.capacity {
            return Ok(());
        }

        self.grow_to(required_cap)
    }

    /// Verify that the pointer is properly aligned
    pub fn verify_alignment(&self) -> bool {
        let addr = self.ptr.as_ptr() as usize;
        addr.is_multiple_of(self.alignment)
    }

    /// Grow the vector's capacity
    fn grow(&mut self) -> Result<()> {
        let new_capacity = if self.capacity == 0 {
            8
        } else {
            self.capacity.saturating_mul(2)
        };
        self.grow_to(new_capacity)
    }

    /// Grow the vector to at least the specified capacity
    fn grow_to(&mut self, min_capacity: usize) -> Result<()> {
        if mem::size_of::<T>() == 0 {
            // Zero-sized types don't need allocation
            self.capacity = usize::MAX;
            return Ok(());
        }

        let new_capacity = min_capacity.max(self.capacity);
        let new_size = mem::size_of::<T>()
            .checked_mul(new_capacity)
            .ok_or_else(|| {
                NumRs2Error::from(MemoryError::allocation_failed(
                    "Capacity overflow in grow",
                    new_capacity,
                ))
            })?;

        let new_layout = Layout::from_size_align(new_size, self.alignment).map_err(|e| {
            NumRs2Error::from(MemoryError::allocation_failed(
                &format!("Invalid layout in grow: {}", e),
                new_size,
            ))
        })?;

        unsafe {
            let new_ptr = alloc(new_layout);
            if new_ptr.is_null() {
                return Err(NumRs2Error::from(MemoryError::allocation_failed(
                    "Memory allocation failed in grow",
                    new_size,
                )));
            }

            // Copy existing elements
            if self.capacity > 0 {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr as *mut T, self.len);

                // Deallocate old memory
                let old_size = mem::size_of::<T>() * self.capacity;
                let old_layout = Layout::from_size_align_unchecked(old_size, self.alignment);
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }

            self.ptr = NonNull::new_unchecked(new_ptr as *mut T);
            self.capacity = new_capacity;
        }

        Ok(())
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity == 0 || mem::size_of::<T>() == 0 {
            return;
        }

        unsafe {
            // Drop all elements
            ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ));

            // Deallocate memory
            let size = mem::size_of::<T>() * self.capacity;
            let layout = Layout::from_size_align_unchecked(size, self.alignment);
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("data", &self.as_slice())
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity_and_alignment(self.len, self.alignment)
            .expect("Clone should not fail if original allocation succeeded");

        for item in self.as_slice() {
            new_vec
                .push(item.clone())
                .expect("Push should not fail with pre-allocated capacity");
        }

        new_vec
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

// ============================================================================
// Raw allocation functions
// ============================================================================

/// Allocate aligned memory
///
/// # Safety
///
/// Caller must ensure:
/// - `size` is non-zero
/// - `alignment` is a power of 2
/// - The returned pointer is properly deallocated with `aligned_dealloc`
///
/// # Arguments
///
/// * `size` - Size in bytes to allocate
/// * `alignment` - Alignment in bytes
///
/// # Returns
///
/// A pointer to the allocated memory, or null if allocation fails.
pub unsafe fn aligned_alloc(size: usize, alignment: usize) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }

    let layout = match Layout::from_size_align(size, alignment) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(),
    };

    alloc(layout)
}

/// Allocate zeroed aligned memory
///
/// # Safety
///
/// Same safety requirements as `aligned_alloc`.
pub unsafe fn aligned_alloc_zeroed(size: usize, alignment: usize) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }

    let layout = match Layout::from_size_align(size, alignment) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(),
    };

    std::alloc::alloc_zeroed(layout)
}

/// Deallocate aligned memory
///
/// # Safety
///
/// Caller must ensure:
/// - `ptr` was allocated by `aligned_alloc` or `aligned_alloc_zeroed`
/// - `size` and `alignment` match the original allocation
/// - `ptr` is not used after this call
pub unsafe fn aligned_dealloc(ptr: *mut u8, size: usize, alignment: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let layout = Layout::from_size_align_unchecked(size, alignment);
    dealloc(ptr, layout);
}

// ============================================================================
// Utility functions
// ============================================================================

/// Check if a pointer is aligned to the specified alignment
pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize).is_multiple_of(alignment)
}

/// Get the cache line size for the current CPU
///
/// Returns 64 bytes on most modern CPUs.
pub fn cache_line_size() -> usize {
    // Use the cache detection from cache_layout module if available
    // For now, return the common default
    64
}

/// Get the optimal SIMD alignment for the current CPU
///
/// Returns:
/// - 64 bytes for AVX-512
/// - 32 bytes for AVX2
/// - 16 bytes for SSE/NEON
pub fn optimal_simd_alignment() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            64
        } else if is_x86_feature_detected!("avx2") || is_x86_feature_detected!("avx") {
            32
        } else {
            16
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        16 // NEON
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        16 // Conservative default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_box_basic() {
        let boxed = AlignedBox::new(42u64).expect("test: valid AlignedBox creation");
        assert_eq!(*boxed, 42);
        assert!(boxed.verify_alignment());
    }

    #[test]
    fn test_aligned_box_alignment() {
        let boxed =
            AlignedBox::with_alignment(123u64, 64).expect("test: valid AlignedBox with alignment");
        let addr = boxed.as_ptr() as usize;
        assert_eq!(addr % 64, 0);
        assert_eq!(boxed.alignment(), 64);
    }

    #[test]
    fn test_aligned_vec_basic() {
        let mut vec = AlignedVec::with_capacity(10).expect("test: valid AlignedVec creation");
        vec.push(1.0f64).expect("test: valid push");
        vec.push(2.0f64).expect("test: valid push");
        vec.push(3.0f64).expect("test: valid push");

        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 2.0);
        assert_eq!(vec[2], 3.0);
        assert!(vec.verify_alignment());
    }

    #[test]
    fn test_aligned_vec_grow() {
        let mut vec = AlignedVec::with_capacity(2).expect("test: valid AlignedVec creation");
        for i in 0..10 {
            vec.push(i).expect("test: valid push");
        }

        assert_eq!(vec.len(), 10);
        assert!(vec.capacity() >= 10);
        assert!(vec.verify_alignment());

        for i in 0..10 {
            assert_eq!(vec[i], i);
        }
    }

    #[test]
    fn test_aligned_vec_pop() {
        let mut vec = AlignedVec::with_capacity(5).expect("test: valid AlignedVec creation");
        vec.push(1).expect("test: valid push");
        vec.push(2).expect("test: valid push");
        vec.push(3).expect("test: valid push");

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_aligned_vec_clear() {
        let mut vec = AlignedVec::with_capacity(5).expect("test: valid AlignedVec creation");
        vec.push(1).expect("test: valid push");
        vec.push(2).expect("test: valid push");
        vec.clear();

        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_raw_alloc() {
        unsafe {
            let ptr = aligned_alloc(1024, 64);
            assert!(!ptr.is_null());
            assert!(is_aligned(ptr, 64));

            aligned_dealloc(ptr, 1024, 64);
        }
    }

    #[test]
    fn test_simd_alignment_detection() {
        let alignment = optimal_simd_alignment();
        assert!(alignment >= 16);
        assert!(alignment.is_power_of_two());
    }
}

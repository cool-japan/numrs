//! Memory alignment optimization
//!
//! This module provides functions for optimizing memory alignment to improve
//! performance of numerical operations, especially those using SIMD instructions.

use std::alloc::{self, Layout};
use std::mem;
use std::ptr;

/// Strategy for optimizing memory alignment
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AlignmentStrategy {
    /// Default alignment (usually the size of the type)
    Default,
    /// SIMD-friendly alignment (16, 32, or 64 bytes depending on CPU)
    Simd,
    /// Cache line alignment (typically 64 bytes)
    CacheLine,
    /// Custom alignment
    Custom(usize),
}

/// Align data for optimal memory access
///
/// # Arguments
///
/// * `data` - The data to align
/// * `strategy` - The alignment strategy to use
///
/// This function creates a new allocation with the specified alignment
/// and copies the data into it. It returns a new slice with the aligned data.
pub fn align_data<T: Copy>(data: &mut [T], strategy: AlignmentStrategy) {
    let alignment = match strategy {
        AlignmentStrategy::Default => mem::align_of::<T>(),
        AlignmentStrategy::Simd => get_simd_alignment::<T>(),
        AlignmentStrategy::CacheLine => get_cache_line_size(),
        AlignmentStrategy::Custom(align) => align,
    };

    // Check if data is already properly aligned
    let data_ptr = data.as_ptr() as usize;
    if data_ptr % alignment == 0 {
        // Already aligned
        return;
    }

    // Create a new aligned allocation
    let size = std::mem::size_of_val(data);
    let layout = Layout::from_size_align(size, alignment).unwrap_or_else(|_| Layout::new::<T>());

    unsafe {
        let new_ptr = alloc::alloc(layout) as *mut T;
        if new_ptr.is_null() {
            // Allocation failed, just return and leave data unaligned
            return;
        }

        // Copy data to the new aligned memory
        ptr::copy_nonoverlapping(data.as_ptr(), new_ptr, data.len());

        // Copy aligned data back to the original slice
        ptr::copy_nonoverlapping(new_ptr, data.as_mut_ptr(), data.len());

        // Free the temporary allocation
        alloc::dealloc(new_ptr as *mut u8, layout);
    }
}

/// Get the appropriate alignment for SIMD operations based on runtime CPU detection
fn get_simd_alignment<T>() -> usize {
    let type_size = mem::size_of::<T>();

    // Determine SIMD alignment based on runtime CPU feature detection
    let base_alignment = if cfg!(target_arch = "x86_64") {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                64 // AVX-512 uses 512-bit registers (64 bytes)
            } else if is_x86_feature_detected!("avx2") || is_x86_feature_detected!("avx") {
                32 // AVX/AVX2 uses 256-bit registers (32 bytes)
            } else if is_x86_feature_detected!("sse2") {
                16 // SSE2 uses 128-bit registers (16 bytes)
            } else {
                8 // Fallback for very old CPUs
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            16 // Default for non-x86_64
        }
    } else if cfg!(target_arch = "aarch64") {
        // For aarch64, NEON requires 16-byte alignment
        16
    } else {
        // For other architectures, use a reasonable default
        8
    };

    // Alignment should be at least as large as the type
    base_alignment.max(type_size)
}

/// Get the CPU's cache line size
fn get_cache_line_size() -> usize {
    // In a real implementation, this would query the CPU for its actual cache line size
    // For now, return a common value
    64 // 64 bytes is a common cache line size
}

/// Create an aligned slice of data
///
/// This function allocates a new aligned buffer and copies the data into it.
/// It returns a new Vec with the aligned data, appropriately sized and aligned.
pub fn create_aligned_vec<T: Copy>(data: &[T], alignment: usize) -> Vec<T> {
    let size = std::mem::size_of_val(data);
    let layout = Layout::from_size_align(size, alignment).unwrap_or_else(|_| Layout::new::<T>());

    unsafe {
        let new_ptr = alloc::alloc(layout) as *mut T;
        if new_ptr.is_null() {
            // Allocation failed, return unaligned data
            return data.to_vec();
        }

        // Copy data to the new aligned memory
        ptr::copy_nonoverlapping(data.as_ptr(), new_ptr, data.len());

        // Create a Vec from the raw parts
        Vec::from_raw_parts(new_ptr, data.len(), data.len())
    }
}

/// Create an aligned buffer with specified size and alignment
///
/// This function allocates memory with the specified alignment and returns
/// a Vec that owns the aligned memory.
pub fn create_aligned_buffer<T: Copy + Default>(size: usize, alignment: usize) -> Vec<T> {
    let byte_size = size * mem::size_of::<T>();
    let layout = Layout::from_size_align(byte_size, alignment)
        .unwrap_or_else(|_| Layout::array::<T>(size).unwrap());

    unsafe {
        let ptr = alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            // Allocation failed, return default Vec
            return vec![T::default(); size];
        }

        Vec::from_raw_parts(ptr, size, size)
    }
}

/// Realign an existing vector to a new alignment
///
/// This function creates a new aligned allocation and moves the data.
pub fn realign_vec<T: Copy>(mut vec: Vec<T>, new_alignment: usize) -> Vec<T> {
    // Check if already properly aligned
    if is_aligned(vec.as_ptr(), new_alignment) {
        return vec;
    }

    // Create new aligned allocation
    let aligned_vec = create_aligned_vec(&vec, new_alignment);

    // Clear the original vector without deallocating (if it was aligned differently)
    vec.clear();
    vec.shrink_to_fit();

    aligned_vec
}

/// Check if a pointer is aligned to a specific boundary
pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Calculate the padding needed to align a given offset
pub fn alignment_padding(offset: usize, alignment: usize) -> usize {
    if offset % alignment == 0 {
        0
    } else {
        alignment - (offset % alignment)
    }
}

/// Get the best alignment for a given data type based on CPU capabilities
pub fn get_optimal_alignment_for_type<T>() -> usize {
    let type_size = mem::size_of::<T>();

    // For floating point types, prefer SIMD alignment
    if mem::size_of::<T>() == mem::size_of::<f32>() || mem::size_of::<T>() == mem::size_of::<f64>()
    {
        return get_simd_alignment::<T>();
    }

    // For integer types, also prefer SIMD alignment if beneficial
    if type_size >= 4 {
        return get_simd_alignment::<T>();
    }

    // For small types, use cache line alignment
    get_cache_line_size().max(type_size)
}

/// Align a memory address to the nearest aligned boundary
pub fn align_address(addr: usize, alignment: usize) -> usize {
    (addr + alignment - 1) & !(alignment - 1)
}

/// Check if a memory range is properly aligned
pub fn is_range_aligned<T>(slice: &[T], alignment: usize) -> bool {
    let ptr = slice.as_ptr() as usize;
    let size = std::mem::size_of_val(slice);

    // Check if start is aligned
    if ptr % alignment != 0 {
        return false;
    }

    // Check if size is a multiple of alignment (for some use cases)
    // This is optional but can be useful for certain algorithms
    size % alignment == 0
}

/// Get alignment information for debugging
pub fn get_alignment_info<T>(data: &[T]) -> AlignmentInfo {
    let ptr = data.as_ptr() as usize;
    let cache_line_size = get_cache_line_size();
    let simd_alignment = get_simd_alignment::<T>();

    AlignmentInfo {
        address: ptr,
        cache_line_aligned: ptr % cache_line_size == 0,
        simd_aligned: ptr % simd_alignment == 0,
        natural_aligned: ptr % mem::align_of::<T>() == 0,
        cache_line_size,
        simd_alignment,
        type_alignment: mem::align_of::<T>(),
    }
}

/// Alignment information for debugging and analysis
#[derive(Debug, Clone)]
pub struct AlignmentInfo {
    pub address: usize,
    pub cache_line_aligned: bool,
    pub simd_aligned: bool,
    pub natural_aligned: bool,
    pub cache_line_size: usize,
    pub simd_alignment: usize,
    pub type_alignment: usize,
}

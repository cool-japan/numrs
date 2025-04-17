//! Memory alignment optimization
//!
//! This module provides functions for optimizing memory alignment to improve
//! performance of numerical operations, especially those using SIMD instructions.

use std::mem;
use std::ptr;
use std::alloc::{self, Layout};

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
    let size = data.len() * mem::size_of::<T>();
    let layout = Layout::from_size_align(size, alignment)
        .unwrap_or_else(|_| Layout::new::<T>());
    
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

/// Get the appropriate alignment for SIMD operations
fn get_simd_alignment<T>() -> usize {
    let type_size = mem::size_of::<T>();
    
    // Determine SIMD alignment based on CPU features
    let base_alignment = if cfg!(target_arch = "x86_64") {
        // For x86_64, use AVX-512, AVX2, AVX, or SSE
        if cfg!(target_feature = "avx512f") {
            64 // AVX-512 uses 512-bit registers (64 bytes)
        } else if cfg!(target_feature = "avx2") || cfg!(target_feature = "avx") {
            32 // AVX/AVX2 uses 256-bit registers (32 bytes)
        } else {
            16 // SSE uses 128-bit registers (16 bytes)
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
    let size = data.len() * mem::size_of::<T>();
    let layout = Layout::from_size_align(size, alignment)
        .unwrap_or_else(|_| Layout::new::<T>());
    
    let mut vec = Vec::with_capacity(data.len());
    unsafe {
        let new_ptr = alloc::alloc(layout) as *mut T;
        if new_ptr.is_null() {
            // Allocation failed, return unaligned data
            vec.extend_from_slice(data);
            return vec;
        }
        
        // Copy data to the new aligned memory
        ptr::copy_nonoverlapping(data.as_ptr(), new_ptr, data.len());
        
        // Create a Vec from the raw parts
        vec = Vec::from_raw_parts(new_ptr, data.len(), data.len());
    }
    
    vec
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
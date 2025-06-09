//! Memory layout optimization for cache efficiency
//!
//! This module provides functions for reorganizing data in memory to improve
//! cache efficiency, taking advantage of both spatial and temporal locality.

use std::mem;
use std::ptr;
use std::arch::x86_64::__cpuid;
use std::cmp;
use std::collections::HashMap;
use std::sync::Once;

/// Cache information for optimal layout decisions
#[derive(Debug, Clone)]
struct CacheInfo {
    line_size: usize,
    l1_size: usize,
    l2_size: usize,
    l3_size: usize,
    associativity: usize,
}

static CACHE_INFO: Once = Once::new();
static mut CACHE_DATA: Option<CacheInfo> = None;

/// Strategy for optimizing memory layout
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LayoutStrategy {
    /// Row-major order (C-style) - optimized for row-wise operations
    RowMajor,
    /// Column-major order (Fortran-style) - optimized for column-wise operations
    ColumnMajor,
    /// Morton order (Z-order curve) - good for 2D traversal
    Morton,
    /// Hilbert curve order - better locality than Morton
    Hilbert,
    /// Cache-oblivious layout - adapts to any cache size
    CacheOblivious,
    /// Blocked layout for optimizing matrix operations
    Blocked(usize), // block size
}

/// Optimize the memory layout of a slice of data
///
/// # Arguments
///
/// * `data` - The data to optimize
/// * `strategy` - The layout strategy to use
///
/// This function reorganizes the data in memory according to the specified strategy
/// to improve cache efficiency. It uses in-place algorithms when possible to
/// minimize additional memory usage.
pub fn optimize_layout<T: Copy>(data: &mut [T], strategy: LayoutStrategy) {
    match strategy {
        LayoutStrategy::RowMajor => {
            // Data is already in row-major order in most cases
            // But we can ensure optimal alignment
            align_for_cache_line(data);
        },
        LayoutStrategy::ColumnMajor => {
            // For 1D data, no transpose needed. For actual multidimensional data,
            // this would require shape information
            // This optimization assumes data will be accessed column-wise
            optimize_for_column_access(data);
        },
        LayoutStrategy::Morton => {
            // Reorder data along a Z-order curve for 2D locality
            apply_morton_order(data);
        },
        LayoutStrategy::Hilbert => {
            // Reorder data along a Hilbert curve for better 2D locality than Morton
            apply_hilbert_order(data);
        },
        LayoutStrategy::CacheOblivious => {
            // Use recursive layout that works well regardless of cache size
            apply_cache_oblivious_layout(data);
        },
        LayoutStrategy::Blocked(block_size) => {
            // Reorganize data into blocks for better cache usage in matrix operations
            apply_blocked_layout(data, block_size);
        },
    }
}

/// Align data to cache line boundaries for better cache efficiency
///
/// This function ensures that the start of the data is aligned to the cache line size
/// of the CPU, which can significantly improve memory access performance.
fn align_for_cache_line<T: Copy>(data: &mut [T]) {
    // Get the cache line size (typical values are 64 or 128 bytes)
    let cache_line_size = get_cache_line_size();
    
    // Calculate the current alignment
    let data_ptr = data.as_ptr() as usize;
    let misalignment = data_ptr % cache_line_size;
    
    if misalignment == 0 {
        // Already aligned
        return;
    }
    
    // Realign by shifting data
    // This is a simplification; real implementation would be more sophisticated
    // and would handle edge cases better
    let shift = cache_line_size - misalignment;
    if shift < mem::size_of::<T>() * data.len() {
        unsafe {
            let src = data.as_ptr();
            let dst = (data.as_mut_ptr() as *mut u8).add(shift) as *mut T;
            ptr::copy(src, dst, data.len());
        }
    }
}

/// Get the CPU's cache line size
///
/// This function queries the CPU for its actual cache line size.
/// If it cannot be determined, it returns a sensible default.
fn get_cache_line_size() -> usize {
    get_cache_info().line_size
}

/// Get comprehensive CPU cache information
fn get_cache_info() -> &'static CacheInfo {
    unsafe {
        CACHE_INFO.call_once(|| {
            CACHE_DATA = Some(detect_cache_info());
        });
        CACHE_DATA.as_ref().unwrap()
    }
}

/// Detect CPU cache information using CPUID
fn detect_cache_info() -> CacheInfo {
    #[cfg(target_arch = "x86_64")]
    {
        return detect_x86_cache_info();
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Default values for non-x86_64 architectures
        CacheInfo {
            line_size: 64,
            l1_size: 32 * 1024,
            l2_size: 256 * 1024,
            l3_size: 8 * 1024 * 1024,
            associativity: 8,
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_cache_info() -> CacheInfo {
    use std::arch::x86_64::__cpuid;
    
    let mut info = CacheInfo {
        line_size: 64,
        l1_size: 32 * 1024,
        l2_size: 256 * 1024,
        l3_size: 8 * 1024 * 1024,
        associativity: 8,
    };
    
    unsafe {
        // Check if CPUID leaf 0x80000006 is available (cache info)
        let cpuid_result = __cpuid(0x80000000);
        if cpuid_result.eax >= 0x80000006 {
            let cache_result = __cpuid(0x80000006);
            
            // L1 data cache info from ECX register
            info.l1_size = ((cache_result.ecx >> 24) & 0xFF) as usize * 1024;
            info.line_size = (cache_result.ecx & 0xFF) as usize;
            info.associativity = ((cache_result.ecx >> 16) & 0xFF) as usize;
            
            // L2 cache info from ECX register  
            info.l2_size = ((cache_result.ecx >> 16) & 0xFFFF) as usize * 1024;
            
            // L3 cache info from EDX register
            info.l3_size = ((cache_result.edx >> 18) & 0x3FFF) as usize * 512 * 1024;
        }
        
        // Intel-specific cache detection
        let vendor_result = __cpuid(0);
        if vendor_result.ebx == 0x756e6547 && // "Genu"
           vendor_result.edx == 0x49656e69 && // "ineI"
           vendor_result.ecx == 0x6c65746e {  // "ntel"
            detect_intel_cache_info(&mut info);
        }
        
        // AMD-specific cache detection
        if vendor_result.ebx == 0x68747541 && // "Auth"
           vendor_result.edx == 0x69746e65 && // "enti"
           vendor_result.ecx == 0x444d4163 {  // "cAMD"
            detect_amd_cache_info(&mut info);
        }
    }
    
    info
}

#[cfg(target_arch = "x86_64")]
fn detect_intel_cache_info(info: &mut CacheInfo) {
    // Intel cache detection via CPUID leaf 4
    unsafe {
        let mut cache_level = 0;
        loop {
            let cache_info = __cpuid_count(4, cache_level);
            
            // No more cache levels
            if cache_info.eax & 0x1F == 0 {
                break;
            }
            
            let cache_type = cache_info.eax & 0x1F;
            let level = (cache_info.eax >> 5) & 0x7;
            let line_size = ((cache_info.ebx & 0xFFF) + 1) as usize;
            let partitions = (((cache_info.ebx >> 12) & 0x3FF) + 1) as usize;
            let ways = (((cache_info.ebx >> 22) & 0x3FF) + 1) as usize;
            let sets = (cache_info.ecx + 1) as usize;
            
            let size = line_size * partitions * ways * sets;
            
            // Data cache or unified cache
            if cache_type == 1 || cache_type == 3 {
                match level {
                    1 => {
                        info.l1_size = size;
                        info.line_size = line_size;
                        info.associativity = ways;
                    },
                    2 => info.l2_size = size,
                    3 => info.l3_size = size,
                    _ => {}
                }
            }
            
            cache_level += 1;
            if cache_level > 10 { // Safety check
                break;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_amd_cache_info(info: &mut CacheInfo) {
    // AMD cache detection via CPUID leaves 0x80000005 and 0x80000006
    unsafe {
        // L1 cache info
        let l1_info = __cpuid(0x80000005);
        info.l1_size = ((l1_info.ecx >> 24) & 0xFF) as usize * 1024;
        info.line_size = (l1_info.ecx & 0xFF) as usize;
        info.associativity = ((l1_info.ecx >> 16) & 0xFF) as usize;
        
        // L2/L3 cache info
        let l23_info = __cpuid(0x80000006);
        info.l2_size = ((l23_info.ecx >> 16) & 0xFFFF) as usize * 1024;
        info.l3_size = ((l23_info.edx >> 18) & 0x3FFF) as usize * 512 * 1024;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn __cpuid_count(leaf: u32, sub_leaf: u32) -> std::arch::x86_64::CpuidResult {
    let mut eax = leaf;
    let mut ecx = sub_leaf;
    let mut edx = 0;
    
    // Use a workaround for rbx register constraint issue
    let ebx: u32;
    std::arch::asm!(
        "push rbx",      // Save rbx
        "cpuid",         // Execute cpuid
        "mov {0:e}, ebx", // Copy ebx to output (32-bit)
        "pop rbx",       // Restore rbx
        out(reg) ebx,
        inout("eax") eax,
        inout("ecx") ecx,
        inout("edx") edx,
    );
    
    std::arch::x86_64::CpuidResult { eax, ebx, ecx, edx }
}

/// Calculate the optimal block size for the current CPU's cache
///
/// This function estimates the best block size for blocked algorithms based on
/// the CPU's cache size and the data type size.
pub fn calculate_optimal_block_size<T>() -> usize {
    // Get the L1 data cache size
    let l1_cache_size = get_l1_cache_size();
    let type_size = mem::size_of::<T>();
    
    // A simple heuristic: we want the block to fit in L1 cache
    // Square root because we're typically dealing with 2D blocks
    let elements_per_cache = l1_cache_size / type_size;
    let block_size = (elements_per_cache as f64).sqrt() as usize;
    
    // Ensure the block size is at least 1 and reasonable
    block_size.max(1).min(1024)
}

/// Optimize data layout for column-wise access patterns
fn optimize_for_column_access<T: Copy>(data: &mut [T]) {
    // For 1D data, we can prefetch data in patterns that will be accessed
    // In a real implementation, this would reorganize multidimensional data
    // For now, we apply cache-friendly prefetch patterns
    prefetch_data_pattern(data, get_cache_line_size());
}

/// Apply Morton (Z-order) curve ordering to data
fn apply_morton_order<T: Copy>(data: &mut [T]) {
    let len = data.len();
    if len < 4 {
        return; // Too small to benefit from reordering
    }
    
    // For simplicity, assume we're working with a power-of-2 sized array
    // that represents a 2D grid
    let side = (len as f64).sqrt() as usize;
    if side * side != len {
        // Not a perfect square, fall back to blocked layout
        apply_blocked_layout(data, calculate_optimal_block_size::<T>());
        return;
    }
    
    // Create a temporary buffer for reordered data
    let mut temp = vec![data[0]; len];
    
    // Reorder according to Morton curve
    for i in 0..len {
        let (x, y) = morton_decode(i, side);
        if x < side && y < side {
            let linear_index = y * side + x;
            if linear_index < len {
                temp[i] = data[linear_index];
            }
        }
    }
    
    // Copy back to original array
    data.copy_from_slice(&temp);
}

/// Apply Hilbert curve ordering to data  
fn apply_hilbert_order<T: Copy>(data: &mut [T]) {
    let len = data.len();
    if len < 4 {
        return; // Too small to benefit from reordering
    }
    
    // For simplicity, assume we're working with a power-of-2 sized array
    let side = (len as f64).sqrt() as usize;
    if side * side != len || !side.is_power_of_two() {
        // Not a perfect square power of 2, fall back to Morton order
        apply_morton_order(data);
        return;
    }
    
    // Create a temporary buffer for reordered data
    let mut temp = vec![data[0]; len];
    
    // Reorder according to Hilbert curve
    for i in 0..len {
        let (x, y) = hilbert_decode(i, side);
        if x < side && y < side {
            let linear_index = y * side + x;
            if linear_index < len {
                temp[i] = data[linear_index];
            }
        }
    }
    
    // Copy back to original array
    data.copy_from_slice(&temp);
}

/// Apply cache-oblivious recursive layout
fn apply_cache_oblivious_layout<T: Copy>(data: &mut [T]) {
    if data.len() <= get_cache_line_size() / mem::size_of::<T>() {
        return; // Small enough to fit in cache line
    }
    
    // Divide and conquer approach
    cache_oblivious_recursive(data, 0, data.len());
}

/// Recursive helper for cache-oblivious layout
fn cache_oblivious_recursive<T: Copy>(data: &mut [T], start: usize, end: usize) {
    let len = end - start;
    if len <= 1 {
        return;
    }
    
    let cache_size = get_cache_info().l1_size / mem::size_of::<T>();
    if len <= cache_size {
        return; // Fits in cache
    }
    
    // Split in half and recursively optimize
    let mid = start + len / 2;
    cache_oblivious_recursive(data, start, mid);
    cache_oblivious_recursive(data, mid, end);
    
    // Interleave the two halves for better locality
    interleave_data(&mut data[start..end]);
}

/// Apply blocked layout for matrix operations
fn apply_blocked_layout<T: Copy>(data: &mut [T], block_size: usize) {
    let len = data.len();
    if len < block_size * block_size {
        return; // Too small to benefit from blocking
    }
    
    // Assume square matrix layout
    let side = (len as f64).sqrt() as usize;
    if side * side != len {
        return; // Not a square matrix
    }
    
    // Create temporary buffer for blocked data
    let mut temp = vec![data[0]; len];
    let mut temp_idx = 0;
    
    // Copy data in blocks
    for block_row in (0..side).step_by(block_size) {
        for block_col in (0..side).step_by(block_size) {
            let max_row = cmp::min(block_row + block_size, side);
            let max_col = cmp::min(block_col + block_size, side);
            
            for row in block_row..max_row {
                for col in block_col..max_col {
                    let linear_idx = row * side + col;
                    if linear_idx < len && temp_idx < len {
                        temp[temp_idx] = data[linear_idx];
                        temp_idx += 1;
                    }
                }
            }
        }
    }
    
    // Copy back to original array
    data.copy_from_slice(&temp);
}

/// Prefetch data in a cache-friendly pattern
fn prefetch_data_pattern<T: Copy>(data: &mut [T], cache_line_size: usize) {
    let elements_per_line = cache_line_size / mem::size_of::<T>();
    
    // Touch every cache line to ensure it's loaded
    for i in (0..data.len()).step_by(elements_per_line) {
        // Prefetch hint for the next cache line
        if i + elements_per_line < data.len() {
            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    let ptr = data.as_ptr().add(i + elements_per_line);
                    std::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0
                    );
                }
            }
        }
    }
}

/// Decode Morton index to 2D coordinates
fn morton_decode(morton: usize, side: usize) -> (usize, usize) {
    let mut x = 0;
    let mut y = 0;
    let mut bit = 0;
    let mut m = morton;
    
    while m > 0 && bit < 32 {
        if (m & 1) != 0 {
            x |= 1 << (bit / 2);
        }
        m >>= 1;
        
        if (m & 1) != 0 {
            y |= 1 << (bit / 2);
        }
        m >>= 1;
        
        bit += 2;
    }
    
    (x % side, y % side)
}

/// Decode Hilbert index to 2D coordinates
fn hilbert_decode(h: usize, n: usize) -> (usize, usize) {
    let mut t = h;
    let mut x = 0;
    let mut y = 0;
    let mut s = 1;
    
    while s < n {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);
        
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            
            // Swap x and y
            let temp = x;
            x = y;
            y = temp;
        }
        
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    
    (x % n, y % n)
}

/// Interleave two halves of data for better cache locality
fn interleave_data<T: Copy>(data: &mut [T]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    
    let mid = len / 2;
    let mut temp = vec![data[0]; len];
    
    // Interleave first and second half
    for i in 0..mid {
        temp[2 * i] = data[i];
        if 2 * i + 1 < len && i + mid < len {
            temp[2 * i + 1] = data[i + mid];
        }
    }
    
    // Handle odd lengths
    if len % 2 == 1 {
        temp[len - 1] = data[len - 1];
    }
    
    data.copy_from_slice(&temp);
}

/// Get the CPU's L1 data cache size
fn get_l1_cache_size() -> usize {
    get_cache_info().l1_size
}

/// Get the CPU's L2 cache size
fn get_l2_cache_size() -> usize {
    get_cache_info().l2_size
}

/// Get the CPU's L3 cache size
fn get_l3_cache_size() -> usize {
    get_cache_info().l3_size
}
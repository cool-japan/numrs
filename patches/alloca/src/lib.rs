//! Pure-Rust implementation of alloca API (no C, no FFI).
//! Uses heap allocation as a fallback since we cannot use VLAs.

use core::mem::{self, MaybeUninit};

/// Allocates `size` bytes and invokes `f` with a mutable slice of uninitialized bytes.
pub fn with_alloca<R>(size: usize, f: impl FnOnce(&mut [MaybeUninit<u8>]) -> R) -> R {
    let mut buf: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); size];
    f(&mut buf)
}

/// Allocates `size` zeroed bytes and invokes `f` with a mutable byte slice.
pub fn with_alloca_zeroed<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let mut buf: Vec<u8> = vec![0u8; size];
    f(&mut buf)
}

/// Allocates space for one `T` on the "stack" (actually heap) and invokes `f`.
pub fn alloca<T, R>(f: impl FnOnce(&mut MaybeUninit<T>) -> R) -> R {
    let align = mem::align_of::<T>();
    let size = mem::size_of::<T>();
    // Allocate extra bytes to guarantee alignment
    let extra = if align > 1 { align - 1 } else { 0 };
    let mut buf: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); size + extra];
    let ptr = buf.as_mut_ptr();
    // SAFETY: `buf` has `size + extra` bytes, and `aligned_addr` is within that range
    // because `(addr + extra) & !(align - 1) <= addr + extra`. The aligned pointer
    // is valid for the lifetime of `buf` which outlives the closure call.
    unsafe {
        let addr = ptr as usize;
        let aligned_addr = (addr + extra) & !(align - 1);
        let aligned_ptr = aligned_addr as *mut MaybeUninit<T>;
        f(&mut *aligned_ptr)
    }
}

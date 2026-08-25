//! N-dimensional indexing kernels: permutation, slicing and broadcasting.
//!
//! All three operations share one idea: the output is dense and every input is
//! addressed through a per-output-axis stride table. A transpose permutes the
//! stride table, a slice scales it and adds a base offset, and a broadcast
//! zeroes the strides of the stretched axes. The GPU therefore needs only two
//! kernels - a type-agnostic gather (`shaders/gather_words.wgsl`) and a typed
//! binary operation (`shaders/broadcast_template.wgsl`).
//!
//! # Layout assumption
//!
//! A [`GpuArray`] buffer always holds its elements in logical row-major order,
//! because [`GpuArray::from_array`] materialises the CPU array with
//! `Array::to_vec()` - which walks a non-contiguous array in logical order -
//! before uploading. Source strides are therefore always derived from the
//! shape rather than read back from the CPU-side array.

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::kernel::{
    contiguous_strides, dispatch, linear_dispatch, meta_buffer, to_u32, words_per_element, Binding,
};
use crate::gpu::ops::ElementWiseOp;

/// A half-open slice range with an optional step.
///
/// Mirrors NumPy's basic slicing for a single axis: elements
/// `start, start + step, start + 2 * step, ...` strictly below `end` are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceRange {
    /// First index of the axis that is kept.
    pub start: usize,
    /// One past the last index that may be kept.
    pub end: usize,
    /// Distance between two kept indices; must be non-zero.
    pub step: usize,
}

impl SliceRange {
    /// Creates a contiguous range `start..end`.
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            step: 1,
        }
    }

    /// Creates a strided range `start..end` keeping every `step`-th element.
    pub fn with_step(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }

    /// Number of elements this range selects.
    fn len(&self) -> usize {
        if self.end <= self.start || self.step == 0 {
            0
        } else {
            (self.end - self.start).div_ceil(self.step)
        }
    }
}

impl From<(usize, usize)> for SliceRange {
    fn from((start, end): (usize, usize)) -> Self {
        Self::new(start, end)
    }
}

/// Runs the strided gather kernel.
///
/// Produces a dense array of shape `out_shape` whose element at multi-index
/// `i` is `source[base + sum_d(i[d] * src_strides[d])]`. This is the single
/// primitive behind [`permute_axes`] and [`slice`].
///
/// The kernel copies raw 32-bit words, so it is exact for every element type
/// and needs no per-precision variant.
pub fn gather_strided<T: bytemuck::Pod + bytemuck::Zeroable>(
    source: &GpuArray<T>,
    out_shape: &[usize],
    src_strides: &[usize],
    base: usize,
    label: &str,
) -> Result<GpuArray<T>> {
    if out_shape.len() != src_strides.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Gather needs one source stride per output axis, got {} strides for {} axes",
            src_strides.len(),
            out_shape.len()
        )));
    }

    let n_elements: usize = out_shape.iter().product();
    if n_elements == 0 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Cannot run a GPU gather producing an empty array of shape {:?}",
            out_shape
        )));
    }

    // Reject any addressing that would read past the end of the source buffer
    // before the kernel does it, since a shader read out of bounds is silently
    // clamped rather than reported.
    let mut max_offset = base;
    for (axis, &dim) in out_shape.iter().enumerate() {
        max_offset += (dim - 1) * src_strides[axis];
    }
    if max_offset >= source.size() {
        return Err(NumRs2Error::IndexError(format!(
            "Gather would read element {} of a source holding {} elements",
            max_offset,
            source.size()
        )));
    }

    let context = source.context().clone();
    let result = GpuArray::<T>::new_with_shape(out_shape, context.clone())?;
    let (groups_x, groups_y) = linear_dispatch(&context, n_elements)?;

    let mut meta = Vec::with_capacity(5 + 2 * out_shape.len());
    meta.push(to_u32(out_shape.len(), "array rank")?);
    meta.push(to_u32(n_elements, "element count")?);
    meta.push(words_per_element::<T>()?);
    meta.push(groups_x);
    meta.push(to_u32(base, "gather base offset")?);
    for &dim in out_shape {
        meta.push(to_u32(dim, "axis length")?);
    }
    for &stride in src_strides {
        meta.push(to_u32(stride, "axis stride")?);
    }
    let meta = meta_buffer(&context, "Gather Metadata", &meta);

    dispatch(
        &context,
        context.gather_shader(),
        "gather",
        label,
        &[
            Binding::Storage(source.buffer()),
            Binding::StorageMut(result.buffer()),
            Binding::Storage(&meta),
        ],
        (groups_x, groups_y, 1),
    );

    Ok(result)
}

/// Permutes the axes of a GPU array (NumPy's `transpose(a, axes)`).
///
/// `axes` must be a permutation of `0..ndim`; the output axis `d` takes its
/// data from the input axis `axes[d]`.
pub fn permute_axes<T: bytemuck::Pod + bytemuck::Zeroable>(
    array: &GpuArray<T>,
    axes: &[usize],
) -> Result<GpuArray<T>> {
    let ndim = array.shape().len();
    if axes.len() != ndim {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Permutation has {} axes but the array has {} dimensions",
            axes.len(),
            ndim
        )));
    }

    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim {
            return Err(NumRs2Error::IndexError(format!(
                "Axis {} is out of range for an array with {} dimensions",
                axis, ndim
            )));
        }
        if seen[axis] {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Axis {} appears more than once in permutation {:?}",
                axis, axes
            )));
        }
        seen[axis] = true;
    }

    let in_strides = contiguous_strides(array.shape());
    let out_shape: Vec<usize> = axes.iter().map(|&axis| array.shape()[axis]).collect();
    let src_strides: Vec<usize> = axes.iter().map(|&axis| in_strides[axis]).collect();

    gather_strided(array, &out_shape, &src_strides, 0, "NumRS2 Permute Axes")
}

/// Reverses the axis order of a GPU array (NumPy's `a.T`).
pub fn reverse_axes<T: bytemuck::Pod + bytemuck::Zeroable>(
    array: &GpuArray<T>,
) -> Result<GpuArray<T>> {
    let axes: Vec<usize> = (0..array.shape().len()).rev().collect();
    permute_axes(array, &axes)
}

/// Extracts a strided slice of a GPU array.
///
/// One [`SliceRange`] per axis is required. The result is a new, dense array
/// holding a copy of the selected elements.
pub fn slice_strided<T: bytemuck::Pod + bytemuck::Zeroable>(
    array: &GpuArray<T>,
    ranges: &[SliceRange],
) -> Result<GpuArray<T>> {
    let shape = array.shape();
    if ranges.len() != shape.len() {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Number of slice ranges ({}) does not match array dimensions ({})",
            ranges.len(),
            shape.len()
        )));
    }

    let in_strides = contiguous_strides(shape);
    let mut out_shape = Vec::with_capacity(ranges.len());
    let mut src_strides = Vec::with_capacity(ranges.len());
    let mut base = 0usize;

    for (axis, range) in ranges.iter().enumerate() {
        if range.step == 0 {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Slice step for dimension {} must be non-zero",
                axis
            )));
        }
        if range.start >= range.end || range.end > shape[axis] {
            return Err(NumRs2Error::IndexError(format!(
                "Invalid range [{}..{}] for dimension {} with size {}",
                range.start, range.end, axis, shape[axis]
            )));
        }

        out_shape.push(range.len());
        src_strides.push(in_strides[axis] * range.step);
        base += range.start * in_strides[axis];
    }

    gather_strided(array, &out_shape, &src_strides, base, "NumRS2 Slice")
}

/// Computes the per-output-axis strides that broadcast `shape` to `out_shape`.
///
/// Follows the NumPy rule: shapes are right-aligned, missing leading axes and
/// axes of length one are stretched, which is expressed as a stride of zero.
pub(crate) fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Result<Vec<usize>> {
    if shape.len() > out_shape.len() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: out_shape.to_vec(),
            actual: shape.to_vec(),
        });
    }

    let strides = contiguous_strides(shape);
    let offset = out_shape.len() - shape.len();
    let mut result = vec![0usize; out_shape.len()];

    for (axis, &dim) in shape.iter().enumerate() {
        let out_axis = axis + offset;
        if dim == out_shape[out_axis] {
            result[out_axis] = strides[axis];
        } else if dim == 1 {
            result[out_axis] = 0;
        } else {
            return Err(NumRs2Error::ShapeMismatch {
                expected: out_shape.to_vec(),
                actual: shape.to_vec(),
            });
        }
    }

    Ok(result)
}

/// Runs a broadcasting element-wise binary operation on the GPU.
///
/// Implements the full NumPy broadcasting rule for both operands at once; no
/// operand is materialised in its broadcast form, the stretched axes simply
/// get a stride of zero.
pub(crate) fn broadcast_binary<T: bytemuck::Pod + bytemuck::Zeroable>(
    a: &GpuArray<T>,
    b: &GpuArray<T>,
    out_shape: &[usize],
    op: ElementWiseOp,
) -> Result<GpuArray<T>> {
    let n_elements: usize = out_shape.iter().product();
    if n_elements == 0 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Cannot run a GPU broadcast producing an empty array of shape {:?}",
            out_shape
        )));
    }

    let a_strides = broadcast_strides(a.shape(), out_shape)?;
    let b_strides = broadcast_strides(b.shape(), out_shape)?;

    let context = a.context().clone();
    let shader = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        context.broadcast_f32_shader()
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        context.broadcast_f64_shader()?
    } else {
        return Err(NumRs2Error::TypeCastError(
            "Broadcasting GPU operations only support f32 and f64 element types".to_string(),
        ));
    };

    let result = GpuArray::<T>::new_with_shape(out_shape, context.clone())?;
    let (groups_x, groups_y) = linear_dispatch(&context, n_elements)?;

    let mut meta = Vec::with_capacity(4 + 3 * out_shape.len());
    meta.push(op as u32);
    meta.push(to_u32(n_elements, "element count")?);
    meta.push(to_u32(out_shape.len(), "array rank")?);
    meta.push(groups_x);
    for &dim in out_shape {
        meta.push(to_u32(dim, "axis length")?);
    }
    for &stride in &a_strides {
        meta.push(to_u32(stride, "axis stride")?);
    }
    for &stride in &b_strides {
        meta.push(to_u32(stride, "axis stride")?);
    }
    let meta = meta_buffer(&context, "Broadcast Metadata", &meta);

    dispatch(
        &context,
        shader,
        "broadcast_binary",
        "NumRS2 Broadcast Binary Op",
        &[
            Binding::Storage(a.buffer()),
            Binding::Storage(b.buffer()),
            Binding::StorageMut(result.buffer()),
            Binding::Storage(&meta),
        ],
        (groups_x, groups_y, 1),
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_range_len() {
        assert_eq!(SliceRange::new(0, 5).len(), 5);
        assert_eq!(SliceRange::with_step(0, 5, 2).len(), 3);
        assert_eq!(SliceRange::with_step(1, 6, 2).len(), 3);
        assert_eq!(SliceRange::with_step(0, 6, 3).len(), 2);
        assert_eq!(SliceRange::new(3, 3).len(), 0);
    }

    #[test]
    fn test_broadcast_strides_matches_numpy_rule() {
        // (3, 1) broadcast against (2, 3, 4): leading axis and the size-1 axis
        // are stretched, the matching axis keeps its contiguous stride.
        let strides =
            broadcast_strides(&[3, 1], &[2, 3, 4]).expect("shapes broadcast together cleanly");
        assert_eq!(strides, vec![0, 1, 0]);

        // A fully matching shape keeps every contiguous stride.
        let strides = broadcast_strides(&[2, 3, 4], &[2, 3, 4]).expect("identical shapes");
        assert_eq!(strides, vec![12, 4, 1]);

        // A scalar-like shape is stretched along every axis.
        let strides = broadcast_strides(&[1], &[2, 3]).expect("scalar broadcast");
        assert_eq!(strides, vec![0, 0]);
    }

    #[test]
    fn test_broadcast_strides_rejects_incompatible() {
        assert!(broadcast_strides(&[3], &[2, 4]).is_err());
        assert!(broadcast_strides(&[2, 3, 4], &[3, 4]).is_err());
    }
}

// Generic strided gather over raw 32-bit words.
//
// One thread handles one *output* element. The output is dense (C-contiguous)
// and the source is addressed through a per-axis stride table, so a single
// kernel covers
//   * arbitrary N-D axis permutation (transpose / permute_axes),
//   * strided slice extraction (base offset + per-axis step),
//   * any other pure gather expressible as
//         src_index = base + sum_d(out_index_d * src_stride_d)
//
// The kernel is element-type agnostic: values are copied as
// `words_per_element` consecutive 32-bit words, which covers f32 (1 word),
// f64 (2 words) and every other `Pod` element whose size is a multiple of
// four bytes. No arithmetic is performed on the payload, so no precision is
// lost and no per-type shader variant is required.
//
// kernel_meta layout (u32 words):
//   0                  : ndim
//   1                  : number of output elements
//   2                  : words per element
//   3                  : number of workgroups dispatched along x
//   4                  : base offset into the source, in elements
//   5 .. 5+ndim        : output shape
//   5+ndim .. 5+2*ndim : source stride per output axis, in elements

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> kernel_meta: array<u32>;

@compute @workgroup_size(256)
fn gather(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ndim = kernel_meta[0];
    let n_elements = kernel_meta[1];
    let words_per_element = kernel_meta[2];
    let groups_x = kernel_meta[3];

    // Flatten a possibly two-dimensional dispatch back into a linear index so
    // that arrays larger than `max_compute_workgroups_per_dimension * 256`
    // elements are still addressable.
    let idx = (workgroup_id.y * groups_x + workgroup_id.x) * 256u + local_id.x;
    if (idx >= n_elements) {
        return;
    }

    // Unravel the linear output index into per-axis coordinates (row-major,
    // last axis fastest) and accumulate the source offset on the way.
    var remainder = idx;
    var src_element = kernel_meta[4];
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let axis = ndim - 1u - d;
        let dim = kernel_meta[5u + axis];
        let coord = remainder % dim;
        remainder = remainder / dim;
        src_element = src_element + coord * kernel_meta[5u + ndim + axis];
    }

    let dst_word = idx * words_per_element;
    let src_word = src_element * words_per_element;
    for (var w: u32 = 0u; w < words_per_element; w = w + 1u) {
        dst[dst_word + w] = src[src_word + w];
    }
}

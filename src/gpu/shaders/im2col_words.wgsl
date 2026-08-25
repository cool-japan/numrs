// im2col patch-matrix materialisation over raw 32-bit words.
//
// Builds the column matrix used by the GEMM formulation of a 2-D convolution.
// Input is NCHW `[batch, channels, in_h, in_w]`, the produced matrix has shape
//   [channels * kernel_h * kernel_w, batch * out_h * out_w]
// so that `weights_2d [out_channels, k] * col [k, batch*out_h*out_w]` yields
// the convolution result laid out as `[out_channels, batch, out_h*out_w]`.
//
// Positions outside the padded input are written as all-zero words, which is
// the zero element for every float and integer type, keeping this kernel
// element-type agnostic (values are copied as `words_per_element` consecutive
// 32-bit words, exactly like `gather_words.wgsl`).
//
// kernel_meta layout (u32 words):
//   0 : number of column-matrix elements (rows * cols)
//   1 : words per element
//   2 : number of workgroups dispatched along x
//   3 : batch
//   4 : channels
//   5 : in_h
//   6 : in_w
//   7 : kernel_h
//   8 : kernel_w
//   9 : out_h
//  10 : out_w
//  11 : stride_h
//  12 : stride_w
//  13 : pad_h
//  14 : pad_w
//  15 : dilation_h
//  16 : dilation_w

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> col: array<u32>;
@group(0) @binding(2) var<storage, read> kernel_meta: array<u32>;

@compute @workgroup_size(256)
fn im2col(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let n_elements = kernel_meta[0];
    let words_per_element = kernel_meta[1];
    let groups_x = kernel_meta[2];

    let idx = (workgroup_id.y * groups_x + workgroup_id.x) * 256u + local_id.x;
    if (idx >= n_elements) {
        return;
    }

    let batch = kernel_meta[3];
    let channels = kernel_meta[4];
    let in_h = kernel_meta[5];
    let in_w = kernel_meta[6];
    let kernel_h = kernel_meta[7];
    let kernel_w = kernel_meta[8];
    let out_h = kernel_meta[9];
    let out_w = kernel_meta[10];
    let stride_h = kernel_meta[11];
    let stride_w = kernel_meta[12];
    let pad_h = kernel_meta[13];
    let pad_w = kernel_meta[14];
    let dilation_h = kernel_meta[15];
    let dilation_w = kernel_meta[16];

    let n_cols = batch * out_h * out_w;

    // Row index -> (channel, kernel_y, kernel_x); column index -> (n, oh, ow).
    let row = idx / n_cols;
    let col_idx = idx % n_cols;

    let kw = row % kernel_w;
    let row_rest = row / kernel_w;
    let kh = row_rest % kernel_h;
    let channel = row_rest / kernel_h;

    let ow = col_idx % out_w;
    let col_rest = col_idx / out_w;
    let oh = col_rest % out_h;
    let n = col_rest / out_h;

    let in_y = i32(oh * stride_h + kh * dilation_h) - i32(pad_h);
    let in_x = i32(ow * stride_w + kw * dilation_w) - i32(pad_w);

    let dst_word = idx * words_per_element;

    if (in_y < 0 || in_y >= i32(in_h) || in_x < 0 || in_x >= i32(in_w)) {
        for (var w: u32 = 0u; w < words_per_element; w = w + 1u) {
            col[dst_word + w] = 0u;
        }
        return;
    }

    let src_element = ((n * channels + channel) * in_h + u32(in_y)) * in_w + u32(in_x);
    let src_word = src_element * words_per_element;
    for (var w: u32 = 0u; w < words_per_element; w = w + 1u) {
        col[dst_word + w] = src[src_word + w];
    }
}

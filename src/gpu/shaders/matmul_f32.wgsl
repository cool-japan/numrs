// Matrix multiplication shader for f32

// Matrix multiplication parameters
struct MatMulParams {
    a_rows: u32,
    a_cols: u32, // same as b_rows
    b_cols: u32,
    _padding: u32,
}

// Bindings
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

// Tiled matrix multiplication for better cache usage
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    // Bounds check
    if (row >= params.a_rows || col >= params.b_cols) {
        return;
    }

    var sum: f32 = 0.0;
    let a_row_offset = row * params.a_cols;

    // Accumulate dot product
    for (var k: u32 = 0u; k < params.a_cols; k = k + 1u) {
        let a_idx = a_row_offset + k;
        let b_idx = k * params.b_cols + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    // Write result
    let c_idx = row * params.b_cols + col;
    c[c_idx] = sum;
}
// Broadcasting element-wise binary operations.
//
// This file is a template: `SCALAR` is replaced with the concrete WGSL scalar
// type before the module is compiled (see
// `GpuContext::create_shader_modules`).
//
// One thread per *output* element. Both operands are addressed through their
// own per-output-axis stride table; an axis that is broadcast carries stride
// zero, so the full NumPy broadcasting rule (right-aligned shapes, size-1 and
// missing axes stretched) is expressed entirely by the stride tables built on
// the host.
//
// kernel_meta layout (u32 words):
//   0                     : op type (matches `ElementWiseOp`)
//   1                     : number of output elements
//   2                     : ndim of the broadcast output shape
//   3                     : number of workgroups dispatched along x
//   4 .. 4+ndim           : output shape
//   4+ndim .. 4+2*ndim    : strides into `a`, in elements (0 where broadcast)
//   4+2*ndim .. 4+3*ndim  : strides into `b`, in elements (0 where broadcast)

@group(0) @binding(0) var<storage, read> input_a: array<SCALAR>;
@group(0) @binding(1) var<storage, read> input_b: array<SCALAR>;
@group(0) @binding(2) var<storage, read_write> output: array<SCALAR>;
@group(0) @binding(3) var<storage, read> kernel_meta: array<u32>;

@compute @workgroup_size(256)
fn broadcast_binary(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let op_type = kernel_meta[0];
    let n_elements = kernel_meta[1];
    let ndim = kernel_meta[2];
    let groups_x = kernel_meta[3];

    let idx = (workgroup_id.y * groups_x + workgroup_id.x) * 256u + local_id.x;
    if (idx >= n_elements) {
        return;
    }

    var remainder = idx;
    var a_index = 0u;
    var b_index = 0u;
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let axis = ndim - 1u - d;
        let dim = kernel_meta[4u + axis];
        let coord = remainder % dim;
        remainder = remainder / dim;
        a_index = a_index + coord * kernel_meta[4u + ndim + axis];
        b_index = b_index + coord * kernel_meta[4u + 2u * ndim + axis];
    }

    let a = input_a[a_index];
    let b = input_b[b_index];
    var result: SCALAR;

    switch op_type {
        case 0u: { // Add
            result = a + b;
        }
        case 1u: { // Subtract
            result = a - b;
        }
        case 2u: { // Multiply
            result = a * b;
        }
        case 3u: { // Divide
            result = a / b;
        }
        case 12u: { // Pow
            result = pow(a, b);
        }
        default: {
            result = SCALAR(0);
        }
    }

    output[idx] = result;
}

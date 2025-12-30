// Element-wise operations for f32 arrays

// Binary operation parameters
struct BinaryOpParams {
    op_type: u32,
    array_size: u32,
    _padding1: u32,
    _padding2: u32,
}

// Transpose parameters
struct TransposeParams {
    width: u32,
    height: u32,
    _padding1: u32,
    _padding2: u32,
}

// Bindings for binary operations
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryOpParams;

// Bindings for unary operations
@group(0) @binding(0) var<storage, read> unary_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> unary_output: array<f32>;
@group(0) @binding(2) var<uniform> unary_params: BinaryOpParams;

// Bindings for transpose operations
@group(0) @binding(0) var<storage, read> transpose_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> transpose_output: array<f32>;
@group(0) @binding(2) var<uniform> transpose_params: TransposeParams;

@compute @workgroup_size(256)
fn binary_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.array_size) {
        return;
    }

    let a = input_a[idx];
    let b = input_b[idx];
    var result: f32;

    switch params.op_type {
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
            result = 0.0;
        }
    }

    output[idx] = result;
}

@compute @workgroup_size(256)
fn unary_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= unary_params.array_size) {
        return;
    }

    let a = unary_input[idx];
    var result: f32;

    switch unary_params.op_type {
        case 4u: { // Exp
            result = exp(a);
        }
        case 5u: { // Log
            result = log(a);
        }
        case 6u: { // Sin
            result = sin(a);
        }
        case 7u: { // Cos
            result = cos(a);
        }
        case 8u: { // Tan
            result = tan(a);
        }
        case 9u: { // Sqrt
            result = sqrt(a);
        }
        case 10u: { // Abs
            result = abs(a);
        }
        case 11u: { // Neg
            result = -a;
        }
        default: {
            result = a;
        }
    }

    unary_output[idx] = result;
}

@compute @workgroup_size(16, 16)
fn transpose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= transpose_params.width || y >= transpose_params.height) {
        return;
    }

    let input_idx = y * transpose_params.width + x;
    let output_idx = x * transpose_params.height + y;

    transpose_output[output_idx] = transpose_input[input_idx];
}
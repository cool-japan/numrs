// Full-array reduction, shared by every scalar type.
//
// This file is a template: `SCALAR` is replaced with the concrete WGSL scalar
// type and `NEG_LIMIT` / `POS_LIMIT` with the corresponding finite extrema
// before the module is compiled (see `GpuContext::create_shader_modules`).
//
// A single entry point performs one pass of a classic workgroup tree
// reduction: every workgroup folds up to 256 input values into one partial
// result. Running the same kernel repeatedly over its own output (with
// `apply_abs` cleared after the first pass) reduces an array of any length to
// a single value entirely on the GPU.
//
// `apply_abs` folds the absolute value into the first pass, which is what the
// L1 norm needs, and keeps it free for every other reduction.
//
// NaN handling follows NumPy: `max`/`min` propagate NaN rather than skipping
// it, so the comparisons below are written out explicitly instead of relying
// on the backend's `max`/`min` NaN behaviour, which is implementation
// defined. The NaN test itself is `IS_NAN_EXPR`, substituted per type: for
// f32 it inspects the bit pattern, because a floating point `x != x` is
// folded away by backends that compile with fast-math relaxations (Metal
// does exactly that and returned the largest non-NaN element instead).

struct ReduceParams {
    // 0 = sum, 1 = mean (summed here, scaled on the host), 2 = max, 3 = min.
    op_type: u32,
    // Number of *input* elements consumed by this pass.
    n_elements: u32,
    // Number of workgroups dispatched along x, used to linearise a 2-D grid.
    groups_x: u32,
    // Non-zero to reduce |x| instead of x (first pass of an L1 norm).
    apply_abs: u32,
}

@group(0) @binding(0) var<storage, read> input: array<SCALAR>;
@group(0) @binding(1) var<storage, read_write> output: array<SCALAR>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_data: array<SCALAR, 256>;

fn is_nan_value(x: SCALAR) -> bool {
    return IS_NAN_EXPR;
}

fn combine(a: SCALAR, b: SCALAR) -> SCALAR {
    switch params.op_type {
        case 2u: { // max, NaN propagating
            if (is_nan_value(a)) { return a; }
            if (is_nan_value(b)) { return b; }
            if (a > b) { return a; }
            return b;
        }
        case 3u: { // min, NaN propagating
            if (is_nan_value(a)) { return a; }
            if (is_nan_value(b)) { return b; }
            if (a < b) { return a; }
            return b;
        }
        default: { // sum / mean
            return a + b;
        }
    }
}

fn identity() -> SCALAR {
    switch params.op_type {
        case 2u: { return NEG_LIMIT; }
        case 3u: { return POS_LIMIT; }
        default: { return SCALAR(0); }
    }
}

@compute @workgroup_size(256)
fn reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let group = workgroup_id.y * params.groups_x + workgroup_id.x;
    let local_idx = local_id.x;
    let global_idx = group * 256u + local_idx;

    // Threads past the end contribute the identity element instead of
    // returning early: `workgroupBarrier` requires uniform control flow.
    var value = identity();
    if (global_idx < params.n_elements) {
        value = input[global_idx];
        if (params.apply_abs != 0u) {
            value = abs(value);
        }
    }
    shared_data[local_idx] = value;

    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = combine(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        output[group] = shared_data[0];
    }
}

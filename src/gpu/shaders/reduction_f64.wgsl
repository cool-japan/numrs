// Reduction operations for f64 arrays

// Reduction parameters
struct ReductionParams {
    op_type: u32,
    array_size: u32,
    workgroup_size: u32,
    _padding: u32,
}

// Bindings
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReductionParams;

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn sum(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    // Load data into shared memory
    var value: f64 = 0.0;
    if (global_idx < params.array_size) {
        value = input[global_idx];
    }
    shared_data[local_idx] = value;

    workgroupBarrier();

    // Reduction in shared memory
    var stride = params.workgroup_size / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write the result to the output buffer
    if (local_idx == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn mean(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    // First compute the sum using the same algorithm as above
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    // Load data into shared memory
    var value: f64 = 0.0;
    if (global_idx < params.array_size) {
        value = input[global_idx];
    }
    shared_data[local_idx] = value;

    workgroupBarrier();

    // Reduction in shared memory
    var stride = params.workgroup_size / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write the result to the output buffer
    if (local_idx == 0u) {
        // Calculate the mean
        let sum = shared_data[0];
        let count = f64(min(params.workgroup_size, params.array_size - workgroup_id.x * params.workgroup_size));
        output[workgroup_id.x] = sum / count;
    }
}

@compute @workgroup_size(256)
fn max(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    // Load data into shared memory
    var value: f64 = -1.7976931348623157e+308; // -DBL_MAX
    if (global_idx < params.array_size) {
        value = input[global_idx];
    }
    shared_data[local_idx] = value;

    workgroupBarrier();

    // Reduction in shared memory
    var stride = params.workgroup_size / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write the result to the output buffer
    if (local_idx == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn min(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    // Load data into shared memory
    var value: f64 = 1.7976931348623157e+308; // DBL_MAX
    if (global_idx < params.array_size) {
        value = input[global_idx];
    }
    shared_data[local_idx] = value;

    workgroupBarrier();

    // Reduction in shared memory
    var stride = params.workgroup_size / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = min(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write the result to the output buffer
    if (local_idx == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
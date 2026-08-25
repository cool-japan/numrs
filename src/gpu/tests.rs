//! GPU correctness tests.
//!
//! Every test compares a GPU kernel against a CPU reference computed with the
//! ordinary [`Array`] API or an explicit loop, and several also pin values
//! produced by NumPy 2.4.2 so that a change of convention (NCHW layout,
//! cross-correlation rather than convolution, NumPy's broadcasting rule) is
//! caught, not just a change of arithmetic.
//!
//! # How these tests are gated
//!
//! They run against whatever adapter the machine offers. When no adapter can
//! be created the test prints a note and returns instead of failing, so the
//! suite is runnable on machines without a GPU. Two environment variables
//! change that:
//!
//! * `NUMRS2_GPU_FALLBACK=1` requests a software adapter (lavapipe, WARP)
//!   rather than a physical GPU. macOS/Metal ships no software adapter, so
//!   there the variable makes context creation fail instead.
//! * `NUMRS2_GPU_REQUIRE=1` turns "no adapter" into a test failure, which is
//!   what a CI job with a known-good GPU should set so a silently skipped
//!   suite cannot be mistaken for a passing one.
//!
//! Tests that need f64 in shaders (`SHADER_F64`) skip themselves on devices
//! without that feature - Metal has none, so on Apple hardware the f64 paths
//! remain hardware-gated and only their error handling is exercised.

use crate::array::Array;
use crate::gpu::context::GpuContextRef;
use crate::gpu::conv::{conv2d, im2col, Conv2dParams};
use crate::gpu::nd::SliceRange;
use crate::gpu::{self, GpuArray};
use scirs2_core::ndarray::{Array as NdArray, IxDyn};

/// Environment variable that turns a missing adapter into a failure.
const REQUIRE_ENV_VAR: &str = "NUMRS2_GPU_REQUIRE";

/// Returns the shared GPU context, or `None` when the machine has none.
fn test_context() -> Option<GpuContextRef> {
    match gpu::util::get_default_context() {
        Ok(context) => Some(context),
        Err(error) => {
            assert!(
                std::env::var(REQUIRE_ENV_VAR).is_err(),
                "{} is set but no GPU adapter could be created: {}",
                REQUIRE_ENV_VAR,
                error
            );
            eprintln!("skipping GPU test: no usable adapter ({})", error);
            None
        }
    }
}

/// Binds a GPU context or returns from the test when there is none.
macro_rules! gpu_context_or_skip {
    () => {
        match test_context() {
            Some(context) => context,
            None => return,
        }
    };
}

/// Asserts that two f32 slices agree to `tolerance`, reporting the worst pair.
#[track_caller]
fn assert_all_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch ({} vs {})",
        what,
        actual.len(),
        expected.len()
    );

    let mut worst = (0usize, 0.0f32);
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let error = (got - want).abs();
        if error > worst.1 {
            worst = (index, error);
        }
    }

    assert!(
        worst.1 <= tolerance,
        "{}: element {} differs by {} (got {}, expected {}), tolerance {}",
        what,
        worst.0,
        worst.1,
        actual[worst.0],
        expected[worst.0],
        tolerance
    );
}

/// Deterministic, sign-varying test data.
fn ramp(len: usize, modulus: usize, offset: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - offset) * scale)
        .collect()
}

/// Linear index of `index` in a C-contiguous array of shape `shape`.
fn linear_index(shape: &[usize], index: &[usize]) -> usize {
    let mut linear = 0;
    for (axis, &coord) in index.iter().enumerate() {
        linear = linear * shape[axis] + coord;
    }
    linear
}

/// Builds a *non-contiguous* CPU array by permuting an owned buffer's axes.
///
/// The resulting array holds its elements in a different memory order than
/// its logical order, which is what the GPU upload path has to normalise.
fn non_contiguous(values: Vec<f32>, shape: &[usize], axes: &[usize]) -> Array<f32> {
    let nd = NdArray::from_shape_vec(IxDyn(shape), values)
        .expect("test data matches the requested shape");
    let permuted = nd.permuted_axes(IxDyn(axes));
    let array = Array::from_ndarray(permuted);
    assert!(
        !array.is_c_contiguous(),
        "the test fixture is meant to be non-contiguous"
    );
    array
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[test]
fn test_reductions_match_cpu_over_multiple_passes() {
    let _context = gpu_context_or_skip!();

    // 100_000 elements need three passes (100_000 -> 391 -> 2 -> 1), which is
    // what exercises the recursive partial reduction.
    let data = ramp(100_000, 101, 50.0, 0.25);
    let cpu = Array::from_vec(data.clone());
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let expected_sum: f64 = data.iter().map(|&v| v as f64).sum();
    let sum = gpu::sum_f32(&gpu_array).expect("GPU sum succeeds");
    let relative = ((sum as f64 - expected_sum) / expected_sum.abs().max(1.0)).abs();
    assert!(
        relative < 1e-5,
        "GPU sum {} differs from CPU sum {} (relative error {})",
        sum,
        expected_sum,
        relative
    );

    let mean = gpu::mean_f32(&gpu_array).expect("GPU mean succeeds");
    let expected_mean = expected_sum / data.len() as f64;
    assert!(
        ((mean as f64 - expected_mean) / expected_mean.abs().max(1.0)).abs() < 1e-5,
        "GPU mean {} differs from CPU mean {}",
        mean,
        expected_mean
    );

    // max/min select an existing element, so they must match exactly.
    let expected_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let expected_min = data.iter().copied().fold(f32::INFINITY, f32::min);
    assert_eq!(
        gpu::max_f32(&gpu_array).expect("GPU max succeeds"),
        expected_max
    );
    assert_eq!(
        gpu::min_f32(&gpu_array).expect("GPU min succeeds"),
        expected_min
    );
}

#[test]
fn test_reductions_on_single_element_and_partial_workgroup() {
    let _context = gpu_context_or_skip!();

    for len in [1usize, 7, 255, 256, 257] {
        let data = ramp(len, 17, 8.0, 0.5);
        let cpu = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

        let expected: f32 = data.iter().sum();
        let sum = gpu::sum_f32(&gpu_array).expect("GPU sum succeeds");
        assert!(
            (sum - expected).abs() < 1e-3,
            "sum of {} elements: got {}, expected {}",
            len,
            sum,
            expected
        );

        assert_eq!(
            gpu::max_f32(&gpu_array).expect("GPU max succeeds"),
            data.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            "max of {} elements",
            len
        );
    }
}

#[test]
fn test_min_max_propagate_nan_like_numpy() {
    let _context = gpu_context_or_skip!();

    // NumPy's max/min propagate NaN (only nanmax/nanmin skip it), and the
    // reduction kernel compares explicitly rather than trusting the backend's
    // max/min NaN behaviour, which WGSL leaves implementation defined.
    let data = vec![1.0f32, -2.0, f32::NAN, 3.0, 0.5];
    let cpu = Array::from_vec(data);
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let max = gpu::max_f32(&gpu_array).expect("GPU max succeeds");
    let min = gpu::min_f32(&gpu_array).expect("GPU min succeeds");
    assert!(
        max.is_nan(),
        "max over data containing NaN returned {}",
        max
    );
    assert!(
        min.is_nan(),
        "min over data containing NaN returned {}",
        min
    );
}

#[test]
fn test_norm_l1_matches_cpu() {
    let _context = gpu_context_or_skip!();

    let data = ramp(50_000, 97, 48.0, 0.125);
    let cpu = Array::from_vec(data.clone());
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let expected: f64 = data.iter().map(|&v| (v as f64).abs()).sum();
    let norm = gpu::linalg::norm_l1(&gpu_array).expect("GPU L1 norm succeeds");
    let relative = ((norm as f64 - expected) / expected).abs();
    assert!(
        relative < 1e-5,
        "GPU L1 norm {} differs from CPU {} (relative error {})",
        norm,
        expected,
        relative
    );
}

#[test]
fn test_norm_l1_small_and_signed() {
    let _context = gpu_context_or_skip!();

    let data = vec![-1.5f32, 2.25, -3.0, 0.0, 4.5];
    let cpu = Array::from_vec(data.clone());
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let norm = gpu::linalg::norm_l1(&gpu_array).expect("GPU L1 norm succeeds");
    assert!((norm - 11.25).abs() < 1e-6, "expected 11.25, got {}", norm);

    // L2 for the same vector, as a cross-check of the GEMM based path.
    let norm2 = gpu::linalg::norm_l2(&gpu_array).expect("GPU L2 norm succeeds");
    let expected2 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm2 - expected2).abs() < 1e-5,
        "expected {}, got {}",
        expected2,
        norm2
    );
}

#[test]
fn test_norm_l1_rejects_non_vector() {
    let _context = gpu_context_or_skip!();

    let cpu = Array::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("2x2 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");
    assert!(gpu::linalg::norm_l1(&gpu_array).is_err());
}

// ---------------------------------------------------------------------------
// Transpose and permutation
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_2d_non_square() {
    let _context = gpu_context_or_skip!();

    let (rows, cols) = (3usize, 5usize);
    let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
    let cpu = Array::from_vec_shape(data.clone(), &[rows, cols]).expect("3x5 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let transposed = gpu::transpose(&gpu_array).expect("GPU transpose succeeds");
    assert_eq!(transposed.shape(), &[cols, rows]);

    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            expected[c * rows + r] = data[r * cols + c];
        }
    }
    assert_all_close(
        &transposed.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "2-D transpose",
    );
}

#[test]
fn test_transpose_3d_reverses_axes() {
    let _context = gpu_context_or_skip!();

    let shape = [2usize, 3, 4];
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let cpu = Array::from_vec_shape(data.clone(), &shape).expect("2x3x4 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let transposed = gpu::transpose(&gpu_array).expect("GPU N-D transpose succeeds");
    assert_eq!(transposed.shape(), &[4, 3, 2]);

    let out_shape = [4usize, 3, 2];
    let mut expected = vec![0.0f32; len];
    for i in 0..out_shape[0] {
        for j in 0..out_shape[1] {
            for k in 0..out_shape[2] {
                expected[linear_index(&out_shape, &[i, j, k])] =
                    data[linear_index(&shape, &[k, j, i])];
            }
        }
    }
    assert_all_close(
        &transposed.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "3-D transpose",
    );
}

#[test]
fn test_permute_axes_4d() {
    let _context = gpu_context_or_skip!();

    let shape = [2usize, 3, 4, 5];
    let axes = [2usize, 0, 3, 1];
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 30.0).collect();
    let cpu = Array::from_vec_shape(data.clone(), &shape).expect("2x3x4x5 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let permuted = gpu::permute_axes(&gpu_array, &axes).expect("GPU permute succeeds");
    let out_shape: Vec<usize> = axes.iter().map(|&axis| shape[axis]).collect();
    assert_eq!(permuted.shape(), out_shape.as_slice());

    let mut expected = vec![0.0f32; len];
    let mut index = vec![0usize; 4];
    for a in 0..out_shape[0] {
        for b in 0..out_shape[1] {
            for c in 0..out_shape[2] {
                for d in 0..out_shape[3] {
                    let out_index = [a, b, c, d];
                    for (out_axis, &in_axis) in axes.iter().enumerate() {
                        index[in_axis] = out_index[out_axis];
                    }
                    expected[linear_index(&out_shape, &out_index)] =
                        data[linear_index(&shape, &index)];
                }
            }
        }
    }
    assert_all_close(
        &permuted.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "4-D permutation",
    );
}

#[test]
fn test_transpose_of_non_contiguous_input() {
    let _context = gpu_context_or_skip!();

    // Logical shape [4, 3] whose memory order is that of a [3, 4] buffer.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = non_contiguous(data.clone(), &[3, 4], &[1, 0]);
    assert_eq!(cpu.shape(), &[4, 3]);

    let logical = cpu.to_vec();
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");
    assert_eq!(gpu_array.shape(), &[4, 3]);

    // The upload must normalise to logical order ...
    assert_all_close(
        &gpu_array.to_array().expect("download succeeds").to_vec(),
        &logical,
        0.0,
        "round trip of a non-contiguous array",
    );

    // ... and transposing it must undo the permutation.
    let transposed = gpu::transpose(&gpu_array).expect("GPU transpose succeeds");
    assert_eq!(transposed.shape(), &[3, 4]);
    assert_all_close(
        &transposed.to_array().expect("download succeeds").to_vec(),
        &data,
        0.0,
        "transpose of a non-contiguous array",
    );
}

#[test]
fn test_permute_axes_rejects_invalid_permutation() {
    let _context = gpu_context_or_skip!();

    let cpu = Array::from_vec_shape((0..24).map(|i| i as f32).collect(), &[2, 3, 4])
        .expect("2x3x4 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    assert!(
        gpu::permute_axes(&gpu_array, &[0, 1]).is_err(),
        "wrong rank"
    );
    assert!(
        gpu::permute_axes(&gpu_array, &[0, 1, 3]).is_err(),
        "axis out of range"
    );
    assert!(
        gpu::permute_axes(&gpu_array, &[0, 1, 1]).is_err(),
        "repeated axis"
    );
}

// ---------------------------------------------------------------------------
// Slicing
// ---------------------------------------------------------------------------

#[test]
fn test_slice_contiguous_ranges() {
    let _context = gpu_context_or_skip!();

    let shape = [4usize, 5, 6];
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let cpu = Array::from_vec_shape(data.clone(), &shape).expect("4x5x6 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let sliced = gpu::slice(&gpu_array, &[(1, 3), (2, 5), (0, 4)]).expect("GPU slice succeeds");
    assert_eq!(sliced.shape(), &[2, 3, 4]);

    let mut expected = Vec::with_capacity(2 * 3 * 4);
    for i in 1..3 {
        for j in 2..5 {
            for k in 0..4 {
                expected.push(data[linear_index(&shape, &[i, j, k])]);
            }
        }
    }
    assert_all_close(
        &sliced.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "contiguous slice",
    );
}

#[test]
fn test_slice_with_steps() {
    let _context = gpu_context_or_skip!();

    let shape = [4usize, 5, 6];
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25).collect();
    let cpu = Array::from_vec_shape(data.clone(), &shape).expect("4x5x6 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let ranges = [
        SliceRange::new(1, 4),
        SliceRange::with_step(0, 5, 2),
        SliceRange::with_step(2, 6, 3),
    ];
    let sliced = gpu::slice_with_steps(&gpu_array, &ranges).expect("GPU strided slice succeeds");
    assert_eq!(sliced.shape(), &[3, 3, 2]);

    let mut expected = Vec::new();
    for i in (1..4).step_by(1) {
        for j in (0..5).step_by(2) {
            for k in (2..6).step_by(3) {
                expected.push(data[linear_index(&shape, &[i, j, k])]);
            }
        }
    }
    assert_all_close(
        &sliced.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "strided slice",
    );
}

#[test]
fn test_slice_of_non_contiguous_input() {
    let _context = gpu_context_or_skip!();

    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu = non_contiguous(data, &[4, 5], &[1, 0]);
    assert_eq!(cpu.shape(), &[5, 4]);
    let logical = cpu.to_vec();

    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");
    let sliced = gpu::slice(&gpu_array, &[(1, 4), (0, 2)]).expect("GPU slice succeeds");
    assert_eq!(sliced.shape(), &[3, 2]);

    let mut expected = Vec::new();
    for i in 1..4 {
        for j in 0..2 {
            expected.push(logical[linear_index(&[5, 4], &[i, j])]);
        }
    }
    assert_all_close(
        &sliced.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "slice of a non-contiguous array",
    );
}

#[test]
fn test_slice_rejects_invalid_ranges() {
    let _context = gpu_context_or_skip!();

    let cpu =
        Array::from_vec_shape((0..12).map(|i| i as f32).collect(), &[3, 4]).expect("3x4 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    assert!(gpu::slice(&gpu_array, &[(0, 2)]).is_err(), "rank mismatch");
    assert!(
        gpu::slice(&gpu_array, &[(0, 2), (2, 9)]).is_err(),
        "out of bounds"
    );
    assert!(
        gpu::slice(&gpu_array, &[(2, 2), (0, 2)]).is_err(),
        "empty range"
    );
    assert!(
        gpu::slice_with_steps(
            &gpu_array,
            &[SliceRange::new(0, 2), SliceRange::with_step(0, 4, 0)]
        )
        .is_err(),
        "zero step"
    );
}

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// CPU reference for a broadcast binary operation over f32 arrays.
fn broadcast_reference(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let total: usize = out_shape.iter().product();
    let mut result = Vec::with_capacity(total);

    for linear in 0..total {
        // Unravel the output index.
        let mut index = vec![0usize; out_shape.len()];
        let mut rest = linear;
        for axis in (0..out_shape.len()).rev() {
            index[axis] = rest % out_shape[axis];
            rest /= out_shape[axis];
        }

        let pick = |shape: &[usize]| -> usize {
            let offset = out_shape.len() - shape.len();
            let mut linear = 0usize;
            for (axis, &dim) in shape.iter().enumerate() {
                let coord = if dim == 1 { 0 } else { index[axis + offset] };
                linear = linear * dim + coord;
            }
            linear
        };

        result.push(op(a[pick(a_shape)], b[pick(b_shape)]));
    }

    result
}

#[test]
fn test_broadcast_add_leading_and_size_one_axes() {
    let _context = gpu_context_or_skip!();

    let a_shape = [2usize, 3, 4];
    let b_shape = [3usize, 1];
    let a: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![10.0, 20.0, 30.0];

    let gpu_a =
        GpuArray::from_array(&Array::from_vec_shape(a.clone(), &a_shape).expect("2x3x4 array"))
            .expect("upload succeeds");
    let gpu_b =
        GpuArray::from_array(&Array::from_vec_shape(b.clone(), &b_shape).expect("3x1 array"))
            .expect("upload succeeds");

    let result = gpu::broadcast_add(&gpu_a, &gpu_b).expect("GPU broadcast add succeeds");
    assert_eq!(result.shape(), &[2, 3, 4]);

    let expected = broadcast_reference(&a, &a_shape, &b, &b_shape, &a_shape, |x, y| x + y);
    assert_all_close(
        &result.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-6,
        "broadcast add",
    );
}

#[test]
fn test_broadcast_outer_product_shapes() {
    let _context = gpu_context_or_skip!();

    let a_shape = [5usize, 1];
    let b_shape = [1usize, 4];
    let a: Vec<f32> = (0..5).map(|i| (i as f32) + 1.0).collect();
    let b: Vec<f32> = (0..4).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let out_shape = [5usize, 4];

    let gpu_a =
        GpuArray::from_array(&Array::from_vec_shape(a.clone(), &a_shape).expect("5x1 array"))
            .expect("upload succeeds");
    let gpu_b =
        GpuArray::from_array(&Array::from_vec_shape(b.clone(), &b_shape).expect("1x4 array"))
            .expect("upload succeeds");

    let product = gpu::broadcast_multiply(&gpu_a, &gpu_b).expect("GPU broadcast multiply");
    assert_eq!(product.shape(), &[5, 4]);
    assert_all_close(
        &product.to_array().expect("download succeeds").to_vec(),
        &broadcast_reference(&a, &a_shape, &b, &b_shape, &out_shape, |x, y| x * y),
        1e-6,
        "broadcast multiply",
    );

    let difference = gpu::broadcast_subtract(&gpu_a, &gpu_b).expect("GPU broadcast subtract");
    assert_all_close(
        &difference.to_array().expect("download succeeds").to_vec(),
        &broadcast_reference(&a, &a_shape, &b, &b_shape, &out_shape, |x, y| x - y),
        1e-6,
        "broadcast subtract",
    );
}

#[test]
fn test_broadcast_scalar_like_operand() {
    let _context = gpu_context_or_skip!();

    let a_shape = [2usize, 3];
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![4.0];

    let gpu_a =
        GpuArray::from_array(&Array::from_vec_shape(a.clone(), &a_shape).expect("2x3 array"))
            .expect("upload succeeds");
    let gpu_b = GpuArray::from_array(&Array::from_vec(b.clone())).expect("upload succeeds");

    let quotient = gpu::broadcast_divide(&gpu_a, &gpu_b).expect("GPU broadcast divide");
    assert_eq!(quotient.shape(), &[2, 3]);
    let expected: Vec<f32> = a.iter().map(|v| v / 4.0).collect();
    assert_all_close(
        &quotient.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-6,
        "broadcast divide",
    );
}

#[test]
fn test_broadcast_equal_shapes_take_the_dense_path() {
    let _context = gpu_context_or_skip!();

    let shape = [3usize, 3];
    let a: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..9).map(|i| (i as f32) * 2.0).collect();

    let gpu_a = GpuArray::from_array(&Array::from_vec_shape(a.clone(), &shape).expect("3x3"))
        .expect("upload succeeds");
    let gpu_b = GpuArray::from_array(&Array::from_vec_shape(b.clone(), &shape).expect("3x3"))
        .expect("upload succeeds");

    let sum = gpu::broadcast_add(&gpu_a, &gpu_b).expect("GPU broadcast add");
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_all_close(
        &sum.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-6,
        "equal-shape broadcast",
    );
}

#[test]
fn test_broadcast_of_non_contiguous_input() {
    let _context = gpu_context_or_skip!();

    let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu_a = non_contiguous(values, &[3, 4], &[1, 0]);
    assert_eq!(cpu_a.shape(), &[4, 3]);
    let logical_a = cpu_a.to_vec();

    let b = vec![1.0f32, 10.0, 100.0];
    let gpu_a = GpuArray::from_array(&cpu_a).expect("upload succeeds");
    let gpu_b = GpuArray::from_array(&Array::from_vec(b.clone())).expect("upload succeeds");

    let sum = gpu::broadcast_add(&gpu_a, &gpu_b).expect("GPU broadcast add");
    assert_eq!(sum.shape(), &[4, 3]);
    let expected = broadcast_reference(&logical_a, &[4, 3], &b, &[3], &[4, 3], |x, y| x + y);
    assert_all_close(
        &sum.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-6,
        "broadcast over a non-contiguous operand",
    );
}

#[test]
fn test_broadcast_pow() {
    let _context = gpu_context_or_skip!();

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![2.0f32];

    let gpu_a = GpuArray::from_array(&Array::from_vec_shape(a.clone(), &[2, 2]).expect("2x2"))
        .expect("upload succeeds");
    let gpu_b = GpuArray::from_array(&Array::from_vec(b)).expect("upload succeeds");

    let powered = gpu::broadcast_pow(&gpu_a, &gpu_b).expect("GPU broadcast pow");
    assert_eq!(powered.shape(), &[2, 2]);
    let expected: Vec<f32> = a.iter().map(|v| v * v).collect();
    assert_all_close(
        &powered.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-4,
        "broadcast pow",
    );
}

#[test]
fn test_broadcast_rejects_incompatible_shapes() {
    let _context = gpu_context_or_skip!();

    let gpu_a = GpuArray::from_array(
        &Array::from_vec_shape((0..6).map(|i| i as f32).collect(), &[2, 3]).expect("2x3"),
    )
    .expect("upload succeeds");
    let gpu_b = GpuArray::from_array(
        &Array::from_vec_shape((0..12).map(|i| i as f32).collect(), &[4, 3]).expect("4x3"),
    )
    .expect("upload succeeds");

    assert!(gpu::broadcast_add(&gpu_a, &gpu_b).is_err());
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

/// CPU reference convolution (cross-correlation) over NCHW data.
fn conv2d_reference(
    input: &[f32],
    input_shape: [usize; 4],
    weights: &[f32],
    weight_shape: [usize; 4],
    params: &Conv2dParams,
) -> (Vec<f32>, [usize; 4]) {
    let [batch, channels, in_h, in_w] = input_shape;
    let [out_channels, _, kernel_h, kernel_w] = weight_shape;
    let (out_h, out_w) = params
        .output_size((in_h, in_w), (kernel_h, kernel_w))
        .expect("reference geometry is valid");

    let mut output = vec![0.0f32; batch * out_channels * out_h * out_w];
    for n in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f32;
                    for c in 0..channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let y = (oh * params.stride.0 + kh * params.dilation.0) as isize
                                    - params.padding.0 as isize;
                                let x = (ow * params.stride.1 + kw * params.dilation.1) as isize
                                    - params.padding.1 as isize;
                                if y < 0 || x < 0 || y >= in_h as isize || x >= in_w as isize {
                                    continue;
                                }
                                let input_index =
                                    linear_index(&input_shape, &[n, c, y as usize, x as usize]);
                                let weight_index = linear_index(&weight_shape, &[oc, c, kh, kw]);
                                acc += input[input_index] * weights[weight_index];
                            }
                        }
                    }
                    output[linear_index(&[batch, out_channels, out_h, out_w], &[n, oc, oh, ow])] =
                        acc;
                }
            }
        }
    }

    (output, [batch, out_channels, out_h, out_w])
}

#[test]
fn test_im2col_matches_numpy_reference() {
    let _context = gpu_context_or_skip!();

    // 1x1x3x3 input, 2x2 kernel, padding 1: NumPy 2.4.2 reference.
    let input: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let cpu = Array::from_vec_shape(input, &[1, 1, 3, 3]).expect("1x1x3x3 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let params = Conv2dParams::new((1, 1), (1, 1));
    let col = im2col(&gpu_array, (2, 2), &params).expect("GPU im2col succeeds");
    assert_eq!(col.shape(), &[4, 16]);

    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, //
        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, //
        0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_all_close(
        &col.to_array().expect("download succeeds").to_vec(),
        &expected,
        0.0,
        "im2col patch matrix",
    );
}

#[test]
fn test_conv2d_matches_numpy_small_case() {
    let _context = gpu_context_or_skip!();

    let input: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let weights = vec![1.0f32, 2.0, 0.0, 0.0, -1.0, 3.0, 2.0, 1.0, -2.0];

    let gpu_input = GpuArray::from_array(
        &Array::from_vec_shape(input.clone(), &[1, 1, 4, 4]).expect("1x1x4x4"),
    )
    .expect("upload succeeds");
    let gpu_weights = GpuArray::from_array(
        &Array::from_vec_shape(weights.clone(), &[1, 1, 3, 3]).expect("1x1x3x3"),
    )
    .expect("upload succeeds");

    // Valid convolution: NumPy 2.4.2 gives [26, 32, 50, 56].
    let valid =
        conv2d(&gpu_input, &gpu_weights, &Conv2dParams::default()).expect("GPU conv2d succeeds");
    assert_eq!(valid.shape(), &[1, 1, 2, 2]);
    assert_all_close(
        &valid.to_array().expect("download succeeds").to_vec(),
        &[26.0, 32.0, 50.0, 56.0],
        1e-4,
        "valid conv2d against NumPy",
    );

    // Padded and strided: NumPy 2.4.2 gives [-2, 12, 16, 56].
    let padded = conv2d(&gpu_input, &gpu_weights, &Conv2dParams::new((2, 2), (1, 1)))
        .expect("GPU conv2d succeeds");
    assert_eq!(padded.shape(), &[1, 1, 2, 2]);
    assert_all_close(
        &padded.to_array().expect("download succeeds").to_vec(),
        &[-2.0, 12.0, 16.0, 56.0],
        1e-4,
        "padded strided conv2d against NumPy",
    );
}

#[test]
fn test_conv2d_batched_multichannel_strided_padded() {
    let _context = gpu_context_or_skip!();

    let input_shape = [2usize, 3, 7, 6];
    let weight_shape = [4usize, 3, 3, 3];
    let input = ramp(input_shape.iter().product(), 13, 6.0, 0.25);
    let weights = ramp(weight_shape.iter().product(), 7, 3.0, 0.125);
    let params = Conv2dParams::new((2, 1), (1, 2));

    let gpu_input = GpuArray::from_array(
        &Array::from_vec_shape(input.clone(), &input_shape).expect("input array"),
    )
    .expect("upload succeeds");
    let gpu_weights = GpuArray::from_array(
        &Array::from_vec_shape(weights.clone(), &weight_shape).expect("weight array"),
    )
    .expect("upload succeeds");

    let result = conv2d(&gpu_input, &gpu_weights, &params).expect("GPU conv2d succeeds");
    assert_eq!(result.shape(), &[2, 4, 4, 8]);

    let (expected, expected_shape) =
        conv2d_reference(&input, input_shape, &weights, weight_shape, &params);
    assert_eq!(result.shape(), expected_shape.as_slice());

    let actual = result.to_array().expect("download succeeds").to_vec();
    assert_all_close(&actual, &expected, 1e-4, "batched conv2d");

    // Pinned NumPy 2.4.2 values for the same geometry, guarding the layout
    // convention rather than just the arithmetic.
    let total: f32 = actual.iter().sum();
    assert!(
        (total - (-61.0625)).abs() < 1e-2,
        "conv2d output sum {} differs from the NumPy reference -61.0625",
        total
    );
    let sample = |n: usize, oc: usize, oh: usize, ow: usize| -> f32 {
        actual[linear_index(&[2, 4, 4, 8], &[n, oc, oh, ow])]
    };
    assert!((sample(0, 0, 0, 0) - 0.28125).abs() < 1e-4);
    assert!((sample(1, 3, 3, 7) - (-0.53125)).abs() < 1e-4);
    assert!((sample(0, 2, 1, 4) - 0.3125).abs() < 1e-4);
}

#[test]
fn test_conv2d_with_dilation_matches_numpy() {
    let _context = gpu_context_or_skip!();

    let input_shape = [1usize, 2, 5, 5];
    let weight_shape = [2usize, 2, 2, 2];
    let input = ramp(input_shape.iter().product(), 11, 5.0, 0.5);
    let weights = ramp(weight_shape.iter().product(), 5, 2.0, 1.0);
    let params = Conv2dParams::default().with_dilation((2, 2));

    let gpu_input = GpuArray::from_array(
        &Array::from_vec_shape(input.clone(), &input_shape).expect("input array"),
    )
    .expect("upload succeeds");
    let gpu_weights = GpuArray::from_array(
        &Array::from_vec_shape(weights.clone(), &weight_shape).expect("weight array"),
    )
    .expect("upload succeeds");

    let result = conv2d(&gpu_input, &gpu_weights, &params).expect("GPU conv2d succeeds");
    assert_eq!(result.shape(), &[1, 2, 3, 3]);

    // NumPy 2.4.2 reference for the dilated convolution.
    let expected = [
        4.0, 2.5, 1.0, -3.5, 6.0, 4.5, -5.5, 4.0, 2.5, //
        -10.5, 1.0, 1.5, 3.0, -2.0, 9.5, -5.5, -10.5, 1.0,
    ];
    assert_all_close(
        &result.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-4,
        "dilated conv2d against NumPy",
    );

    let (reference, _) = conv2d_reference(&input, input_shape, &weights, weight_shape, &params);
    assert_all_close(&expected, &reference, 1e-4, "reference cross-check");
}

#[test]
fn test_conv2d_of_non_contiguous_input() {
    let _context = gpu_context_or_skip!();

    // Logical NCHW shape [2, 3, 6, 7] stored as a [2, 3, 7, 6] buffer: the
    // upload has to normalise to logical order before im2col indexes it.
    let values = ramp(2 * 3 * 7 * 6, 13, 6.0, 0.25);
    let cpu = non_contiguous(values, &[2, 3, 7, 6], &[0, 1, 3, 2]);
    assert_eq!(cpu.shape(), vec![2, 3, 6, 7]);
    let logical = cpu.to_vec();

    let input_shape = [2usize, 3, 6, 7];
    let weight_shape = [2usize, 3, 3, 3];
    let weights = ramp(weight_shape.iter().product(), 7, 3.0, 0.125);
    let params = Conv2dParams::new((2, 1), (1, 1));

    let gpu_input = GpuArray::from_array(&cpu).expect("upload succeeds");
    let gpu_weights = GpuArray::from_array(
        &Array::from_vec_shape(weights.clone(), &weight_shape).expect("weight array"),
    )
    .expect("upload succeeds");

    let result = conv2d(&gpu_input, &gpu_weights, &params).expect("GPU conv2d succeeds");
    let (expected, expected_shape) =
        conv2d_reference(&logical, input_shape, &weights, weight_shape, &params);
    assert_eq!(result.shape(), expected_shape.as_slice());
    assert_all_close(
        &result.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-4,
        "conv2d over a non-contiguous input",
    );
}

#[test]
fn test_conv2d_rejects_bad_shapes() {
    let _context = gpu_context_or_skip!();

    let input = GpuArray::from_array(
        &Array::from_vec_shape((0..16).map(|i| i as f32).collect(), &[1, 1, 4, 4])
            .expect("1x1x4x4"),
    )
    .expect("upload succeeds");
    let wrong_channels = GpuArray::from_array(
        &Array::from_vec_shape((0..18).map(|i| i as f32).collect(), &[1, 2, 3, 3])
            .expect("1x2x3x3"),
    )
    .expect("upload succeeds");
    assert!(conv2d(&input, &wrong_channels, &Conv2dParams::default()).is_err());

    let too_large = GpuArray::from_array(
        &Array::from_vec_shape((0..25).map(|i| i as f32).collect(), &[1, 1, 5, 5])
            .expect("1x1x5x5"),
    )
    .expect("upload succeeds");
    assert!(conv2d(&input, &too_large, &Conv2dParams::default()).is_err());

    let not_4d = GpuArray::from_array(
        &Array::from_vec_shape((0..9).map(|i| i as f32).collect(), &[3, 3]).expect("3x3"),
    )
    .expect("upload succeeds");
    assert!(conv2d(&not_4d, &input, &Conv2dParams::default()).is_err());
}

#[test]
fn test_batch_queue_executes_conv2d() {
    use crate::gpu::batching::{BatchConfig, BatchQueue, OperationType};
    use std::sync::Arc;

    let context = gpu_context_or_skip!();

    let input_shape = [1usize, 2, 5, 5];
    let weight_shape = [3usize, 2, 3, 3];
    let input = ramp(input_shape.iter().product(), 9, 4.0, 0.5);
    let weights = ramp(weight_shape.iter().product(), 5, 2.0, 0.25);
    let params = Conv2dParams::new((1, 1), (1, 1));

    let gpu_input = Arc::new(
        GpuArray::from_array(
            &Array::from_vec_shape(input.clone(), &input_shape).expect("input array"),
        )
        .expect("upload succeeds"),
    );
    let gpu_weights = Arc::new(
        GpuArray::from_array(
            &Array::from_vec_shape(weights.clone(), &weight_shape).expect("weight array"),
        )
        .expect("upload succeeds"),
    );

    let config = BatchConfig {
        enable_auto_flush: false,
        ..BatchConfig::default()
    };
    let mut queue = BatchQueue::new(context, config);
    queue
        .queue_conv2d(gpu_input, gpu_weights, params)
        .expect("queueing a convolution succeeds");

    let results = queue.flush().expect("flush succeeds");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].op_type, OperationType::Conv2D);
    assert_eq!(results[0].result.shape(), &[1, 3, 5, 5]);

    let (expected, _) = conv2d_reference(&input, input_shape, &weights, weight_shape, &params);
    assert_all_close(
        &results[0]
            .result
            .to_array()
            .expect("download succeeds")
            .to_vec(),
        &expected,
        1e-4,
        "batched conv2d",
    );
}

// ---------------------------------------------------------------------------
// Supporting kernels the new code depends on
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_non_square() {
    let _context = gpu_context_or_skip!();

    let a: Vec<f32> = (1..=6).map(|i| i as f32).collect(); // 2x3
    let b: Vec<f32> = (1..=12).map(|i| i as f32).collect(); // 3x4

    let gpu_a = GpuArray::from_array(&Array::from_vec_shape(a.clone(), &[2, 3]).expect("2x3"))
        .expect("upload succeeds");
    let gpu_b = GpuArray::from_array(&Array::from_vec_shape(b.clone(), &[3, 4]).expect("3x4"))
        .expect("upload succeeds");

    let product = gpu::matmul(&gpu_a, &gpu_b).expect("GPU matmul succeeds");
    assert_eq!(product.shape(), &[2, 4]);

    let mut expected = vec![0.0f32; 8];
    for i in 0..2 {
        for j in 0..4 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += a[i * 3 + k] * b[k * 4 + j];
            }
            expected[i * 4 + j] = acc;
        }
    }
    assert_all_close(
        &product.to_array().expect("download succeeds").to_vec(),
        &expected,
        1e-4,
        "non-square matmul",
    );
}

#[test]
fn test_f64_operations_report_unsupported_devices_clearly() {
    let context = gpu_context_or_skip!();

    let cpu = Array::from_vec(vec![1.0f64, -2.0, 3.0, -4.0]);
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let result = gpu::sum_f64(&gpu_array);
    if context.f64_supported() {
        let sum = result.expect("f64 reduction succeeds on an f64-capable device");
        assert!((sum - (-2.0)).abs() < 1e-12);
    } else {
        // Metal and most other backends have no SHADER_F64: the operation must
        // fail loudly instead of silently running an f32 kernel.
        assert!(
            result.is_err(),
            "f64 reduction must be rejected without SHADER_F64"
        );
    }
}

#[test]
fn test_word_based_kernels_are_type_agnostic() {
    let context = gpu_context_or_skip!();

    // The gather kernel copies raw words, so it works for f64 even on devices
    // that cannot do f64 *arithmetic* in a shader.
    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    let cpu = Array::from_vec_shape(data.clone(), &[3, 4]).expect("3x4 array");
    let gpu_array = GpuArray::from_array(&cpu).expect("upload succeeds");

    let permuted = gpu::permute_axes(&gpu_array, &[1, 0]).expect("GPU permute succeeds");
    assert_eq!(permuted.shape(), &[4, 3]);

    let mut expected = vec![0.0f64; 12];
    for r in 0..3 {
        for c in 0..4 {
            expected[c * 3 + r] = data[r * 4 + c];
        }
    }
    let actual = permuted.to_array().expect("download succeeds").to_vec();
    assert_eq!(
        actual,
        expected,
        "f64 permutation on {:?}",
        context.device()
    );
}

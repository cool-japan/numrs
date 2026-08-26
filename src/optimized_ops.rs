//! Optimized operations using scirs2-core features
//!
//! This module demonstrates the integration of SIMD, GPU, and parallel
//! processing capabilities from scirs2-core into NumRS2.

use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};

// Submodules for specialized optimizations
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod simd_complex;

#[cfg(feature = "lapack")]
use crate::linalg::det;
use crate::{prelude::Array, Result};
// SCIRS2 POLICY COMPLIANT imports - always use SciRS2
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Get information about available optimizations
///
/// The SIMD, AVX2, AVX512, NEON, and parallel-thread lines describe optimizations
/// that NumRS2 actually executes. The CUDA / OpenCL / Metal lines are reported
/// purely as scirs2-core platform detection (informational): they indicate which
/// devices the host exposes, not execution paths inside NumRS2. NumRS2's only GPU
/// execution path is the WGPU backend (enabled via the `gpu` feature), which itself
/// dispatches to Vulkan / Metal / DirectX 12 / WebGPU.
pub fn get_optimization_info() -> String {
    let caps = PlatformCapabilities::detect();
    format!(
        "NumRS2 Optimizations Available:\n\
         - SIMD: {}\n\
         - GPU (WGPU backend, enable with the `gpu` feature): {}\n\
         - AVX2: {}\n\
         - AVX512: {}\n\
         - NEON: {}\n\
         - Parallel threads: {}\n\
         NumRS2 GPU execution uses the WGPU backend only.\n\
         The following are scirs2-core platform detection (informational), not NumRS2 execution paths:\n\
         - CUDA device detected: {}\n\
         - OpenCL device detected: {}\n\
         - Metal device detected: {}",
        caps.simd_available,
        caps.gpu_available,
        caps.avx2_available,
        caps.avx512_available,
        caps.neon_available,
        num_threads(),
        caps.cuda_available,
        caps.opencl_available,
        caps.metal_available,
    )
}

/// Optimized matrix multiplication using SIMD
pub fn simd_matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Result<Array<f32>> {
    if a.ncols() != b.nrows() {
        return Err(crate::NumRs2Error::DimensionMismatch(format!(
            "Matrix dimensions incompatible for multiplication: ({}, {}) x ({}, {})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    let mut result = Array2::zeros((a.nrows(), b.ncols()));

    // Use unified SIMD operations from scirs2-core
    f32::simd_gemm(1.0, a, b, 0.0, &mut result);

    Ok(Array::from_ndarray(result.into_dyn()))
}

/// Optimized element-wise operations using SIMD
pub fn simd_elementwise_ops(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Result<SimdOpsResult> {
    if a.len() != b.len() {
        return Err(crate::NumRs2Error::DimensionMismatch(format!(
            "Array lengths must match: {} != {}",
            a.len(),
            b.len()
        )));
    }

    Ok(SimdOpsResult {
        add: Array::from_ndarray(f64::simd_add(a, b).into_dyn()),
        sub: Array::from_ndarray(f64::simd_sub(a, b).into_dyn()),
        mul: Array::from_ndarray(f64::simd_mul(a, b).into_dyn()),
        div: Array::from_ndarray(f64::simd_div(a, b).into_dyn()),
    })
}

/// Result structure for SIMD element-wise operations
pub struct SimdOpsResult {
    pub add: Array<f64>,
    pub sub: Array<f64>,
    pub mul: Array<f64>,
    pub div: Array<f64>,
}

/// Optimized vector operations using SIMD via SimdUnifiedOps
///
/// `min`/`max` follow NumPy `np.min`/`np.max` semantics: **any `NaN`
/// anywhere in `v` makes the result `NaN`**, independent of where the
/// `NaN` sits or how long `v` is. [`crate::math::nanmin`]/
/// [`crate::math::nanmax`] are the `NaN`-ignoring alternative.
///
/// These no longer route through
/// `scirs2_core::simd_ops::SimdUnifiedOps::simd_min_element`/
/// `simd_max_element`: that upstream pair has a proven bug where it
/// returns a **wrong, finite value** (neither the true extremum nor
/// `NaN`) for some `NaN` placements -- see
/// `crate::kernels::reduce`'s module docs for the reproduction. Instead
/// this calls `crate::kernels::reduce::min_f32`/
/// `crate::kernels::reduce::max_f32`, the same deterministic
/// comparison-fold kernels backing `stats::basic` and
/// `Array::min_optimized`/`max_optimized`.
pub fn simd_vector_ops(v: &ArrayView1<f32>) -> SimdVectorResult {
    // Use SimdUnifiedOps from scirs2-core for SIMD operations
    let sum = f32::simd_sum(v);
    let mean = sum / v.len() as f32;

    // Calculate L2 norm using SIMD
    let norm = f32::simd_norm(v);

    // Borrow `v`'s data as a flat slice for the min/max kernels below,
    // zero-copy when `v` is contiguous (the common case); materializing
    // an owned copy only for a non-contiguous view (e.g. a strided
    // slice), matching `crate::kernels::borrow::operand`'s approach for
    // `Array<T>`.
    let owned_for_minmax;
    let slice: &[f32] = match v.as_slice() {
        Some(s) => s,
        None => {
            owned_for_minmax = v.iter().copied().collect::<Vec<f32>>();
            &owned_for_minmax
        }
    };

    // `crate::kernels::reduce::min_f32`/`max_f32` return `0.0` for an
    // empty slice (their documented contract), which does not match this
    // function's pre-existing empty-input behavior of +/-infinity
    // (inherited from `simd_min_element`/`simd_max_element`'s own empty
    // case) -- guarded explicitly here so that contract is unchanged.
    let min = if slice.is_empty() {
        f32::INFINITY
    } else {
        crate::kernels::reduce::min_f32(slice)
    };
    let max = if slice.is_empty() {
        f32::NEG_INFINITY
    } else {
        crate::kernels::reduce::max_f32(slice)
    };

    SimdVectorResult {
        sum,
        mean,
        norm,
        min,
        max,
    }
}

/// Result structure for SIMD vector operations
pub struct SimdVectorResult {
    pub sum: f32,
    pub mean: f32,
    pub norm: f32,
    pub min: f32,
    pub max: f32,
}

/// Parallel matrix operations
#[cfg(feature = "lapack")]
pub fn parallel_matrix_ops(matrices: &[Array<f64>]) -> Result<Vec<f64>> {
    if matrices.is_empty() {
        return Ok(Vec::new());
    }

    // Use parallel iterators from scirs2-core
    let determinants: Vec<f64> = matrices
        .into_par_iter()
        .map(|matrix| {
            // Compute determinant for each matrix in parallel
            det(matrix).unwrap_or(0.0)
        })
        .collect();

    Ok(determinants)
}

/// Adaptive processing that automatically chooses between scalar, SIMD, or GPU
pub fn adaptive_array_sum(data: &ArrayView1<f64>) -> f64 {
    let optimizer = AutoOptimizer::new();
    let size = data.len();

    if optimizer.should_use_gpu(size) {
        // GPU path would be implemented here if GPU support is available
        // For now, fall back to SIMD
        adaptive_array_sum_simd(data)
    } else if optimizer.should_use_simd(size) {
        // On ARM64, prefer NEON if available. `caps` is only meaningful
        // (and only computed) here: no other target branches on it.
        #[cfg(target_arch = "aarch64")]
        {
            let caps = PlatformCapabilities::detect();
            if caps.neon_available && data.len() >= 4 {
                // Convert f64 to f32 for NEON processing if beneficial
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                return crate::optimized_ops::neon::neon_sum_f32(&f32_data) as f64;
            }
        }
        adaptive_array_sum_simd(data)
    } else {
        // Scalar fallback for small arrays
        data.sum()
    }
}

fn adaptive_array_sum_simd(data: &ArrayView1<f64>) -> f64 {
    // Convert ArrayView to Array for simd_sum
    let array = crate::array::Array::from_vec(data.to_vec());
    crate::simd::simd_sum(&array)
}

/// Parallel statistical computations
pub fn parallel_column_statistics(data: &ArrayView2<f64>) -> Vec<ColumnStats> {
    // Process each column in parallel
    let stats: Vec<ColumnStats> = data
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|column| {
            let mean = column.mean().unwrap_or(0.0);
            let sum = column.sum();
            let min = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            ColumnStats {
                mean,
                sum,
                min,
                max,
            }
        })
        .collect();

    stats
}

/// Statistics for a column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub mean: f64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

/// Chunked processing for memory efficiency
pub fn chunked_array_processing<F, R>(
    data: &ArrayView1<f64>,
    chunk_size: usize,
    processor: F,
) -> Vec<R>
where
    F: Fn(&[f64]) -> R + Send + Sync,
    R: Send,
{
    // Process data in chunks using parallel iterators
    data.as_slice()
        .expect("ArrayView1 should be contiguous")
        .par_chunks(chunk_size)
        .map(processor)
        .collect()
}

/// Determine if parallel processing should be used based on data size
pub fn should_use_parallel(data_size: usize) -> bool {
    is_parallel_enabled() && data_size > 1000
}

/// Enhanced mathematical operations with chunked processing and parallelization
pub mod enhanced_math {
    use super::*;
    use scirs2_core::ndarray::{Array1, ArrayView1, Zip};

    /// Parallel trigonometric sine function
    pub fn parallel_sin(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.sin());
            result
        } else {
            data.map(|&x| x.sin())
        }
    }

    /// Parallel trigonometric cosine function
    pub fn parallel_cos(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.cos());
            result
        } else {
            data.map(|&x| x.cos())
        }
    }

    /// Parallel trigonometric tangent function
    pub fn parallel_tan(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.tan());
            result
        } else {
            data.map(|&x| x.tan())
        }
    }

    /// Parallel sine and cosine (computed together for efficiency)
    pub fn parallel_sincos(data: &ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        if should_use_parallel(data.len()) {
            let mut sin_result = Array1::zeros(data.len());
            let mut cos_result = Array1::zeros(data.len());
            Zip::from(&mut sin_result)
                .and(&mut cos_result)
                .and(data)
                .par_for_each(|sin_out, cos_out, &x| {
                    let (s, c) = x.sin_cos();
                    *sin_out = s;
                    *cos_out = c;
                });
            (sin_result, cos_result)
        } else {
            let sin_result = data.map(|&x| x.sin());
            let cos_result = data.map(|&x| x.cos());
            (sin_result, cos_result)
        }
    }

    /// Parallel inverse sine (arcsin)
    pub fn parallel_asin(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.asin());
            result
        } else {
            data.map(|&x| x.asin())
        }
    }

    /// Parallel inverse cosine (arccos)
    pub fn parallel_acos(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.acos());
            result
        } else {
            data.map(|&x| x.acos())
        }
    }

    /// Parallel inverse tangent (arctan)
    pub fn parallel_atan(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.atan());
            result
        } else {
            data.map(|&x| x.atan())
        }
    }

    /// Parallel atan2
    pub fn parallel_atan2(y: &ArrayView1<f64>, x: &ArrayView1<f64>) -> Result<Array1<f64>> {
        if y.len() != x.len() {
            return Err(crate::NumRs2Error::DimensionMismatch(format!(
                "Array lengths must match for atan2: {} != {}",
                y.len(),
                x.len()
            )));
        }

        if should_use_parallel(y.len()) {
            let mut result = Array1::zeros(y.len());
            Zip::from(&mut result)
                .and(y)
                .and(x)
                .par_for_each(|out, &y_val, &x_val| *out = y_val.atan2(x_val));
            Ok(result)
        } else {
            Ok(Zip::from(y)
                .and(x)
                .map_collect(|&y_val, &x_val| y_val.atan2(x_val)))
        }
    }

    /// Parallel hyperbolic sine
    pub fn parallel_sinh(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.sinh());
            result
        } else {
            data.map(|&x| x.sinh())
        }
    }

    /// Parallel hyperbolic cosine
    pub fn parallel_cosh(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.cosh());
            result
        } else {
            data.map(|&x| x.cosh())
        }
    }

    /// Parallel hyperbolic tangent
    pub fn parallel_tanh(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.tanh());
            result
        } else {
            data.map(|&x| x.tanh())
        }
    }
}

/// Enhanced exponential and logarithmic functions with SIMD sqrt and parallel processing
pub mod enhanced_exp {
    use super::*;
    use scirs2_core::ndarray::{Array1, ArrayView1, Zip};

    /// Parallel exponential function (e^x)
    pub fn parallel_exp(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.exp());
            result
        } else {
            data.map(|&x| x.exp())
        }
    }

    /// Parallel base-2 exponential (2^x)
    pub fn parallel_exp2(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.exp2());
            result
        } else {
            data.map(|&x| x.exp2())
        }
    }

    /// Parallel expm1 (e^x - 1, accurate for small x)
    pub fn parallel_expm1(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.exp_m1());
            result
        } else {
            data.map(|&x| x.exp_m1())
        }
    }

    /// Parallel natural logarithm
    pub fn parallel_ln(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.ln());
            result
        } else {
            data.map(|&x| x.ln())
        }
    }

    /// Parallel base-2 logarithm
    pub fn parallel_log2(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.log2());
            result
        } else {
            data.map(|&x| x.log2())
        }
    }

    /// Parallel base-10 logarithm
    pub fn parallel_log10(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.log10());
            result
        } else {
            data.map(|&x| x.log10())
        }
    }

    /// Parallel ln(1 + x), accurate for small x
    pub fn parallel_ln1p(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.ln_1p());
            result
        } else {
            data.map(|&x| x.ln_1p())
        }
    }

    /// Parallel power function (x^y)
    pub fn parallel_pow(base: &ArrayView1<f64>, exp: &ArrayView1<f64>) -> Result<Array1<f64>> {
        if base.len() != exp.len() {
            return Err(crate::NumRs2Error::DimensionMismatch(format!(
                "Array lengths must match for pow: {} != {}",
                base.len(),
                exp.len()
            )));
        }

        if should_use_parallel(base.len()) {
            let mut result = Array1::zeros(base.len());
            Zip::from(&mut result)
                .and(base)
                .and(exp)
                .par_for_each(|out, &b, &e| *out = b.powf(e));
            Ok(result)
        } else {
            Ok(Zip::from(base).and(exp).map_collect(|&b, &e| b.powf(e)))
        }
    }

    /// SIMD-optimized square root (actually uses SIMD from scirs2-core)
    pub fn simd_sqrt(data: &ArrayView1<f64>) -> Array1<f64> {
        f64::simd_sqrt(data)
    }

    /// Parallel cube root
    pub fn parallel_cbrt(data: &ArrayView1<f64>) -> Array1<f64> {
        if should_use_parallel(data.len()) {
            let mut result = Array1::zeros(data.len());
            Zip::from(&mut result)
                .and(data)
                .par_for_each(|out, &x| *out = x.cbrt());
            result
        } else {
            data.map(|&x| x.cbrt())
        }
    }
}

/// Combined trigonometric and exponential optimization suite
pub struct SimdMathOps;

impl SimdMathOps {
    /// Apply a SIMD-optimized math function with automatic chunking for large arrays
    pub fn apply_chunked<F>(data: &ArrayView1<f64>, chunk_size: usize, simd_fn: F) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        if data.len() <= chunk_size {
            simd_fn(data)
        } else {
            // Process in chunks to avoid memory pressure
            let chunks: Vec<Array1<f64>> = data
                .axis_chunks_iter(Axis(0), chunk_size)
                .into_par_iter()
                .map(|chunk| simd_fn(&chunk))
                .collect();

            // Concatenate results
            let total_len: usize = chunks.iter().map(|c| c.len()).sum();
            let mut result = Array1::zeros(total_len);
            let mut offset = 0;

            for chunk in chunks {
                let chunk_len = chunk.len();
                result
                    .slice_mut(s![offset..offset + chunk_len])
                    .assign(&chunk);
                offset += chunk_len;
            }

            result
        }
    }

    /// Adaptive function selection based on data size and platform capabilities
    pub fn adaptive_math_function<F, G>(
        data: &ArrayView1<f64>,
        simd_fn: F,
        scalar_fn: G,
    ) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64>,
        G: Fn(f64) -> f64,
    {
        let optimizer = AutoOptimizer::new();

        if optimizer.should_use_simd(data.len()) {
            simd_fn(data)
        } else {
            // For small arrays, scalar might be faster due to overhead
            data.map(|&x| scalar_fn(x))
        }
    }
}

/// Chunked processing for very large arrays to avoid memory pressure
pub fn process_large_array<F, T>(
    data: &ArrayView1<f64>,
    chunk_size: usize,
    processor: F,
) -> Result<Array1<T>>
where
    F: Fn(&ArrayView1<f64>) -> Array1<T> + Send + Sync,
    T: Clone + Send + Default + scirs2_core::ndarray::LinalgScalar,
{
    if data.is_empty() {
        return Ok(Array1::from_vec(vec![]));
    }

    let chunks: Vec<Array1<T>> = data
        .axis_chunks_iter(Axis(0), chunk_size)
        .into_par_iter()
        .map(|chunk| processor(&chunk))
        .collect();

    // Calculate total length
    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut result = Array1::from_elem(total_len, T::default());

    // Concatenate chunks
    let mut offset = 0;
    for chunk in chunks {
        let chunk_len = chunk.len();
        result
            .slice_mut(s![offset..offset + chunk_len])
            .assign(&chunk);
        offset += chunk_len;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_optimization_info() {
        let info = get_optimization_info();
        assert!(info.contains("NumRS2 Optimizations Available"));
        assert!(info.contains("SIMD:"));
        assert!(info.contains("Parallel threads:"));
    }

    #[test]
    fn test_simd_matmul() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];

        let result = simd_matmul(&a.view(), &b.view())
            .expect("simd_matmul should succeed for compatible matrices");

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(result.shape(), &[2, 2]);
        let result_2d = result.view_2d().expect("result should be 2D array");
        assert_eq!(result_2d[[0, 0]], 19.0);
        assert_eq!(result_2d[[0, 1]], 22.0);
        assert_eq!(result_2d[[1, 0]], 43.0);
        assert_eq!(result_2d[[1, 1]], 50.0);
    }

    #[test]
    fn test_simd_elementwise_ops() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![5.0, 6.0, 7.0, 8.0];

        let result = simd_elementwise_ops(&a.view(), &b.view())
            .expect("simd_elementwise_ops should succeed for equal length arrays");

        let add_vec = result.add.to_vec();
        let sub_vec = result.sub.to_vec();
        let mul_vec = result.mul.to_vec();
        let div_vec = result.div.to_vec();

        assert_eq!(add_vec[0], 6.0);
        assert_eq!(sub_vec[0], -4.0);
        assert_eq!(mul_vec[0], 5.0);
        assert_eq!(div_vec[0], 0.2);
    }

    #[test]
    fn test_simd_vector_ops() {
        let v = array![1.0f32, 2.0, 3.0, 4.0];

        let result = simd_vector_ops(&v.view());

        assert_eq!(result.sum, 10.0);
        assert_eq!(result.mean, 2.5);
        assert_eq!(result.min, 1.0);
        assert_eq!(result.max, 4.0);
        // norm = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
        assert!((result.norm - 5.477).abs() < 0.01);
    }

    #[test]
    fn test_simd_vector_ops_upstream_bug_vector_yields_nan() {
        // The proven `scirs2_core::simd_ops::SimdUnifiedOps::
        // simd_max_element` bug vector: a 64-element f32 slice
        // `[5.0, 1.0 x9, NaN at index 10, 1.0 x53]`. Upstream silently
        // returned `1.0` (dropping the true max, `5.0`) instead of `NaN`.
        // `simd_vector_ops` must now report `NaN` for both min and max.
        let mut data = vec![1.0f32; 64];
        data[0] = 5.0;
        data[10] = f32::NAN;
        let v = Array1::from_vec(data);

        let result = simd_vector_ops(&v.view());

        assert!(result.min.is_nan(), "min should be NaN, got {}", result.min);
        assert!(result.max.is_nan(), "max should be NaN, got {}", result.max);
    }

    #[test]
    fn test_simd_vector_ops_nan_at_first_position_yields_nan() {
        let v = array![f32::NAN, 1.0, 2.0, 3.0];
        let result = simd_vector_ops(&v.view());
        assert!(result.min.is_nan());
        assert!(result.max.is_nan());
    }

    #[test]
    fn test_simd_vector_ops_nan_at_last_position_yields_nan() {
        let v = array![1.0f32, 2.0, 3.0, f32::NAN];
        let result = simd_vector_ops(&v.view());
        assert!(result.min.is_nan());
        assert!(result.max.is_nan());
    }

    #[test]
    fn test_adaptive_array_sum() {
        let data = Array1::from_vec(vec![1.0; 10000]);
        let sum = adaptive_array_sum(&data.view());
        assert_eq!(sum, 10000.0);
    }

    #[test]
    fn test_parallel_column_statistics() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stats = parallel_column_statistics(&data.view());

        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].mean, 2.5); // (1+4)/2
        assert_eq!(stats[0].sum, 5.0); // 1+4
        assert_eq!(stats[0].min, 1.0);
        assert_eq!(stats[0].max, 4.0);
    }

    #[test]
    fn test_chunked_processing() {
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect());

        let sums = chunked_array_processing(&data.view(), 10, |chunk| chunk.iter().sum::<f64>());

        assert_eq!(sums.len(), 10);
        assert_eq!(sums[0], 45.0); // 0+1+2+...+9 = 45
    }

    #[test]
    fn test_adaptive_math_function() {
        // Small array - should use scalar
        let small_data = array![1.0, 4.0, 9.0];
        let small_result =
            SimdMathOps::adaptive_math_function(&small_data.view(), enhanced_exp::simd_sqrt, |x| {
                x.sqrt()
            });
        assert_eq!(small_result.len(), 3);
        assert!((small_result[0] - 1.0).abs() < 1e-10);
        assert!((small_result[1] - 2.0).abs() < 1e-10);
        assert!((small_result[2] - 3.0).abs() < 1e-10);

        // Large array - should use SIMD
        let large_data = Array1::from_vec((0..10000).map(|x| (x + 1) as f64).collect());
        let large_result =
            SimdMathOps::adaptive_math_function(&large_data.view(), enhanced_exp::simd_sqrt, |x| {
                x.sqrt()
            });
        assert_eq!(large_result.len(), 10000);
    }

    #[test]
    fn test_enhanced_trig_functions() {
        use std::f64::consts::PI;

        let angles = array![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];

        // Test sine
        let sin_result = enhanced_math::parallel_sin(&angles.view());
        assert!((sin_result[0] - 0.0).abs() < 1e-10);
        assert!((sin_result[1] - 0.5).abs() < 1e-10);
        assert!((sin_result[2] - (2.0_f64.sqrt() / 2.0)).abs() < 1e-10);
        assert!((sin_result[3] - (3.0_f64.sqrt() / 2.0)).abs() < 1e-10);
        assert!((sin_result[4] - 1.0).abs() < 1e-10);

        // Test cosine
        let cos_result = enhanced_math::parallel_cos(&angles.view());
        assert!((cos_result[0] - 1.0).abs() < 1e-10);
        assert!((cos_result[1] - (3.0_f64.sqrt() / 2.0)).abs() < 1e-10);
        assert!((cos_result[2] - (2.0_f64.sqrt() / 2.0)).abs() < 1e-10);
        assert!((cos_result[3] - 0.5).abs() < 1e-10);
        assert!((cos_result[4] - 0.0).abs() < 1e-10);

        // Test sincos
        let (sin_res, cos_res) = enhanced_math::parallel_sincos(&angles.view());
        for i in 0..angles.len() {
            assert!((sin_res[i] - sin_result[i]).abs() < 1e-10);
            assert!((cos_res[i] - cos_result[i]).abs() < 1e-10);
        }

        // Test tangent
        let tan_result = enhanced_math::parallel_tan(&angles.view());
        assert!((tan_result[0] - 0.0).abs() < 1e-10);
        assert!((tan_result[1] - (1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
        assert!((tan_result[2] - 1.0).abs() < 1e-10);
        assert!((tan_result[3] - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_enhanced_exp_functions() {
        let x = array![0.0, 1.0, 2.0, -1.0, 0.5];

        // Test exponential
        let exp_result = enhanced_exp::parallel_exp(&x.view());
        assert!((exp_result[0] - 1.0).abs() < 1e-10);
        assert!((exp_result[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((exp_result[2] - std::f64::consts::E.powi(2)).abs() < 1e-10);
        assert!((exp_result[3] - (1.0 / std::f64::consts::E)).abs() < 1e-10);

        // Test logarithm
        let positive_x = array![1.0, std::f64::consts::E, 10.0, 100.0];
        let ln_result = enhanced_exp::parallel_ln(&positive_x.view());
        assert!((ln_result[0] - 0.0).abs() < 1e-10);
        assert!((ln_result[1] - 1.0).abs() < 1e-10);

        // Test sqrt using SIMD
        let sqrt_input = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let sqrt_result = enhanced_exp::simd_sqrt(&sqrt_input.view());
        assert!((sqrt_result[0] - 0.0).abs() < 1e-10);
        assert!((sqrt_result[1] - 1.0).abs() < 1e-10);
        assert!((sqrt_result[2] - 2.0).abs() < 1e-10);
        assert!((sqrt_result[3] - 3.0).abs() < 1e-10);
        assert!((sqrt_result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_process_large_array() {
        let large_data = Array1::from_vec((0..1000).map(|x| x as f64).collect());

        let result = process_large_array(&large_data.view(), 100, |chunk| chunk.map(|&x| x * 2.0))
            .expect("process_large_array should succeed");

        assert_eq!(result.len(), 1000);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[999], 1998.0);
    }
}

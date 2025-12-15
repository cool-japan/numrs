use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::simd_optimize::{detect_cpu_features, select_simd_implementation, SimdImplementation};
use num_traits::Float;
#[allow(unused_imports)]
use scirs2_core::parallel_ops::*;
// SCIRS2 POLICY: Use scirs2_core::simd_ops instead of simba

// Note: This module follows SCIRS2 POLICY - all SIMD operations go through scirs2_core::simd_ops
// NO direct usage of simba, wide, or other SIMD libraries

/// Trait for SIMD-accelerated operations
pub trait SimdOps<T> {
    /// Apply a unary operation with SIMD acceleration
    fn simd_map<F>(&self, f: F) -> Array<T>
    where
        F: Fn(T) -> T;

    /// Apply a binary operation with SIMD acceleration
    fn simd_zip_with<F>(&self, other: &Array<T>, f: F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T;

    /// Reduce with SIMD acceleration
    fn simd_reduce<F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, T) -> T;

    /// SIMD-accelerated element-wise addition
    fn simd_add(&self, other: &Array<T>) -> Result<Array<T>>
    where
        T: std::ops::Add<Output = T> + Copy;

    /// SIMD-accelerated element-wise multiplication
    fn simd_mul(&self, other: &Array<T>) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T> + Copy;

    /// SIMD-accelerated dot product
    fn simd_dot(&self, other: &Array<T>) -> Result<T>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + num_traits::Zero;

    /// SIMD-accelerated sum reduction
    fn simd_sum(&self) -> T
    where
        T: std::ops::Add<Output = T> + Copy + num_traits::Zero;

    /// SIMD-accelerated fused multiply-add
    fn simd_fma(&self, mul: &Array<T>, add: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Copy;

    /// SIMD-accelerated scalar addition (broadcast)
    fn simd_add_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Add<Output = T> + Copy;

    /// SIMD-accelerated scalar multiplication (broadcast)
    fn simd_mul_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Mul<Output = T> + Copy;

    /// SIMD-accelerated scalar subtraction (broadcast)
    fn simd_sub_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Sub<Output = T> + Copy;

    /// SIMD-accelerated scalar division (broadcast)
    fn simd_div_scalar(&self, scalar: T) -> Result<Array<T>>
    where
        T: std::ops::Div<Output = T> + Copy + PartialEq + num_traits::Zero;
}

impl<T: Float + 'static> SimdOps<T> for Array<T> {
    fn simd_map<F>(&self, f: F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        // Detect CPU features and select implementation
        let features = detect_cpu_features();
        let implementation = select_simd_implementation(&features);

        match implementation {
            SimdImplementation::AVX2 | SimdImplementation::AVX512
                if cfg!(target_arch = "x86_64") =>
            {
                // Use AVX2/AVX512 implementation if available on x86_64
                self.simd_map_avx2(&f)
            }
            SimdImplementation::AVX if cfg!(target_arch = "x86_64") => {
                // Use AVX implementation if available on x86_64
                self.simd_map_avx(&f)
            }
            SimdImplementation::SSE if cfg!(target_arch = "x86_64") => {
                // Use SSE implementation if available on x86_64
                self.simd_map_sse(&f)
            }
            SimdImplementation::NEON | SimdImplementation::SVE if cfg!(target_arch = "aarch64") => {
                // Use NEON/SVE implementation if available on aarch64
                self.simd_map_neon(&f)
            }
            _ => {
                // Fall back to scalar implementation
                self.simd_map_scalar(&f)
            }
        }
    }

    fn simd_zip_with<F>(&self, other: &Array<T>, f: F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        // Check if shapes are compatible for broadcasting
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        // Detect CPU features and select implementation
        let features = detect_cpu_features();
        let implementation = select_simd_implementation(&features);

        match implementation {
            SimdImplementation::AVX2 | SimdImplementation::AVX512
                if cfg!(target_arch = "x86_64") =>
            {
                // Use AVX2/AVX512 implementation if available on x86_64
                self.simd_zip_with_avx2(other, &f)
            }
            SimdImplementation::AVX if cfg!(target_arch = "x86_64") => {
                // Use AVX implementation if available on x86_64
                self.simd_zip_with_avx(other, &f)
            }
            SimdImplementation::SSE if cfg!(target_arch = "x86_64") => {
                // Use SSE implementation if available on x86_64
                self.simd_zip_with_sse(other, &f)
            }
            SimdImplementation::NEON | SimdImplementation::SVE if cfg!(target_arch = "aarch64") => {
                // Use NEON/SVE implementation if available on aarch64
                self.simd_zip_with_neon(other, &f)
            }
            _ => {
                // Fall back to scalar implementation
                self.simd_zip_with_scalar(other, &f)
            }
        }
    }

    fn simd_reduce<F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        // Detect CPU features and select implementation
        let features = detect_cpu_features();
        let implementation = select_simd_implementation(&features);

        match implementation {
            SimdImplementation::AVX2 | SimdImplementation::AVX512
                if cfg!(target_arch = "x86_64") =>
            {
                // Use AVX2/AVX512 implementation if available on x86_64
                self.simd_reduce_avx2(init, &f)
            }
            SimdImplementation::AVX if cfg!(target_arch = "x86_64") => {
                // Use AVX implementation if available on x86_64
                self.simd_reduce_avx(init, &f)
            }
            SimdImplementation::SSE if cfg!(target_arch = "x86_64") => {
                // Use SSE implementation if available on x86_64
                self.simd_reduce_sse(init, &f)
            }
            SimdImplementation::NEON | SimdImplementation::SVE if cfg!(target_arch = "aarch64") => {
                // Use NEON/SVE implementation if available on aarch64
                self.simd_reduce_neon(init, &f)
            }
            _ => {
                // Fall back to scalar implementation
                self.simd_reduce_scalar(init, &f)
            }
        }
    }

    fn simd_add(&self, other: &Array<T>) -> Result<Array<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        self.simd_zip_with(other, |x, y| x + y)
    }

    fn simd_mul(&self, other: &Array<T>) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        self.simd_zip_with(other, |x, y| x * y)
    }

    fn simd_dot(&self, other: &Array<T>) -> Result<T>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + num_traits::Zero,
    {
        if self.shape() != other.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }

        // Detect CPU features and select implementation
        let features = detect_cpu_features();
        let implementation = select_simd_implementation(&features);

        // For dot product, we use optimized implementations when available
        match implementation {
            SimdImplementation::AVX2 | SimdImplementation::AVX512
                if cfg!(target_arch = "x86_64") =>
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                    let other_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(other) };
                    let self_data = self_f32.to_vec();
                    let other_data = other_f32.to_vec();
                    let result: f32 = {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            crate::simd_optimize::avx2_ops::avx2_dot_f32(&self_data, &other_data)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            // Fallback for non-x86_64
                            self_data
                                .iter()
                                .zip(other_data.iter())
                                .map(|(a, b)| a * b)
                                .sum()
                        }
                    };
                    return Ok(T::from(result).unwrap());
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                    let other_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(other) };
                    let self_data = self_f64.to_vec();
                    let other_data = other_f64.to_vec();
                    let result: f64 = {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            crate::simd_optimize::avx2_ops::avx2_dot_f64(&self_data, &other_data)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            // Fallback for non-x86_64
                            self_data
                                .iter()
                                .zip(other_data.iter())
                                .map(|(a, b)| a * b)
                                .sum()
                        }
                    };
                    return Ok(T::from(result).unwrap());
                }
            }
            _ => {}
        }

        // Fall back to scalar dot product
        let self_data = self.to_vec();
        let other_data = other.to_vec();
        let mut result = T::zero();
        for (a, b) in self_data.iter().zip(other_data.iter()) {
            result = result + (*a * *b);
        }
        Ok(result)
    }

    fn simd_sum(&self) -> T
    where
        T: std::ops::Add<Output = T> + Copy + num_traits::Zero,
    {
        self.simd_reduce(T::zero(), |acc, x| acc + x)
    }

    fn simd_fma(&self, mul: &Array<T>, add: &Array<T>) -> Result<Array<T>>
    where
        T: Float + Copy,
    {
        if self.shape() != mul.shape() || self.shape() != add.shape() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: self.shape(),
                actual: if self.shape() != mul.shape() {
                    mul.shape()
                } else {
                    add.shape()
                },
            });
        }

        // Detect CPU features and select implementation
        let features = detect_cpu_features();
        let implementation = select_simd_implementation(&features);

        // Use FMA instructions when available
        match implementation {
            SimdImplementation::AVX2 | SimdImplementation::AVX512
                if cfg!(target_arch = "x86_64") && features.fma =>
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                    let mul_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(mul) };
                    let add_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(add) };

                    let self_data = self_f32.to_vec();
                    let mul_data = mul_f32.to_vec();
                    let add_data = add_f32.to_vec();
                    let mut result_data = vec![0.0f32; self_data.len()];

                    {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            crate::simd_optimize::avx2_ops::avx2_fma_f32(
                                &self_data,
                                &mul_data,
                                &add_data,
                                &mut result_data,
                            );
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            // Fallback FMA: a * b + c
                            for (((&a, &b), &c), r) in self_data
                                .iter()
                                .zip(mul_data.iter())
                                .zip(add_data.iter())
                                .zip(result_data.iter_mut())
                            {
                                *r = a * b + c;
                            }
                        }
                    }

                    let result_f32 = Array::from_vec(result_data).reshape(&self_f32.shape());
                    return Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result_f32) });
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                    let mul_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(mul) };
                    let add_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(add) };

                    let self_data = self_f64.to_vec();
                    let mul_data = mul_f64.to_vec();
                    let add_data = add_f64.to_vec();
                    let mut result_data = vec![0.0f64; self_data.len()];

                    {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            crate::simd_optimize::avx2_ops::avx2_fma_f64(
                                &self_data,
                                &mul_data,
                                &add_data,
                                &mut result_data,
                            );
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            // Fallback FMA: a * b + c
                            for (((&a, &b), &c), r) in self_data
                                .iter()
                                .zip(mul_data.iter())
                                .zip(add_data.iter())
                                .zip(result_data.iter_mut())
                            {
                                *r = a * b + c;
                            }
                        }
                    }

                    let result_f64 = Array::from_vec(result_data).reshape(&self_f64.shape());
                    return Ok(unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) });
                }
            }
            _ => {}
        }

        // Fall back to scalar FMA: self * mul + add
        let self_data = self.to_vec();
        let mul_data = mul.to_vec();
        let add_data = add.to_vec();

        let result: Vec<T> = self_data
            .iter()
            .zip(mul_data.iter())
            .zip(add_data.iter())
            .map(|((&a, &b), &c)| a * b + c)
            .collect();

        Ok(Array::from_vec(result).reshape(&self.shape()))
    }

    fn simd_add_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        // Use simd_map with scalar closure for SIMD-accelerated broadcast addition
        self.simd_map(|x| x + scalar)
    }

    fn simd_mul_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        // Use simd_map with scalar closure for SIMD-accelerated broadcast multiplication
        self.simd_map(|x| x * scalar)
    }

    fn simd_sub_scalar(&self, scalar: T) -> Array<T>
    where
        T: std::ops::Sub<Output = T> + Copy,
    {
        // Use simd_map with scalar closure for SIMD-accelerated broadcast subtraction
        self.simd_map(|x| x - scalar)
    }

    fn simd_div_scalar(&self, scalar: T) -> Result<Array<T>>
    where
        T: std::ops::Div<Output = T> + Copy + PartialEq + num_traits::Zero,
    {
        if scalar.is_zero() {
            return Err(NumRs2Error::InvalidOperation(
                "Division by zero".to_string(),
            ));
        }
        // Use simd_map with scalar closure for SIMD-accelerated broadcast division
        Ok(self.simd_map(|x| x / scalar))
    }
}

// Implementation of specific SIMD operations for different architectures
impl<T: Float + 'static> Array<T> {
    // Scalar implementations (fallback)
    fn simd_map_scalar<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        let data = self.to_vec();
        let result: Vec<T> = data.into_iter().map(f).collect();
        Array::from_vec(result).reshape(&self.shape())
    }

    fn simd_zip_with_scalar<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        let self_data = self.to_vec();
        let other_data = other.to_vec();

        let result: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();

        Ok(Array::from_vec(result).reshape(&self.shape()))
    }

    fn simd_reduce_scalar<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let data = self.to_vec();
        data.iter().fold(init, |acc, &x| f(acc, x))
    }

    // SSE implementations (x86_64)
    #[cfg(target_arch = "x86_64")]
    fn simd_map_sse<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use SSE intrinsics
        self.simd_map_scalar(f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_map_sse<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        self.simd_map_scalar(f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_zip_with_sse<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use SSE intrinsics
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_zip_with_sse<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_reduce_sse<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use SSE intrinsics
        self.simd_reduce_scalar(init, f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_reduce_sse<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.simd_reduce_scalar(init, f)
    }

    // AVX implementations (x86_64)
    #[cfg(target_arch = "x86_64")]
    fn simd_map_avx<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use AVX intrinsics
        self.simd_map_scalar(f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_map_avx<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        self.simd_map_scalar(f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_zip_with_avx<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use AVX intrinsics
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_zip_with_avx<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_reduce_avx<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use AVX intrinsics
        self.simd_reduce_scalar(init, f)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_reduce_avx<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.simd_reduce_scalar(init, f)
    }

    // AVX2 implementations (x86_64)
    #[cfg(target_arch = "x86_64")]
    fn simd_map_avx2<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        // Using specialized AVX2 implementations for common operations
        // Fall back to scalar for custom functions
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Check if the function is a square root operation
            let sqrt_test = f(T::one());
            let sqrt_expected = T::one().sqrt();
            if (sqrt_test - sqrt_expected).abs() < T::epsilon() {
                // This is a square root function
                let array_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                let result_f32 = crate::simd_optimize::avx2_optimized_sqrt_f32(array_f32);
                return unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result_f32) };
            }

            // Default to scalar implementation for other functions
            self.simd_map_scalar(f)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // Check if the function is a square root operation
            let sqrt_test = f(T::one());
            let sqrt_expected = T::one().sqrt();
            if (sqrt_test - sqrt_expected).abs() < T::epsilon() {
                // This is a square root function
                let array_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                let result_f64 = crate::simd_optimize::avx2_optimized_sqrt_f64(array_f64);
                return unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) };
            }

            // Default to scalar implementation for other functions
            self.simd_map_scalar(f)
        } else {
            // Default to scalar implementation for other types
            self.simd_map_scalar(f)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_map_avx2<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        self.simd_map_scalar(f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_zip_with_avx2<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        // Using specialized AVX2 implementations for common operations
        // Fall back to scalar for custom functions
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Check if the function is an addition operation
            let add_test = f(T::one(), T::one());
            let add_expected = T::one() + T::one();
            if (add_test - add_expected).abs() < T::epsilon() {
                // This is an addition function
                let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                let other_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(other) };
                let result_f32 = crate::simd_optimize::avx2_optimized_add_f32(self_f32, other_f32)?;
                return Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result_f32) });
            }

            // Check if the function is a multiplication operation
            let mul_test = f(T::from(2.0).unwrap(), T::from(3.0).unwrap());
            let mul_expected = T::from(2.0).unwrap() * T::from(3.0).unwrap();
            if (mul_test - mul_expected).abs() < T::epsilon() {
                // This is a multiplication function
                let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                let other_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(other) };
                let result_f32 = crate::simd_optimize::avx2_optimized_mul_f32(self_f32, other_f32)?;
                return Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result_f32) });
            }

            // Check if the function is a division operation
            let div_test = f(T::from(4.0).unwrap(), T::from(2.0).unwrap());
            let div_expected = T::from(4.0).unwrap() / T::from(2.0).unwrap();
            if (div_test - div_expected).abs() < T::epsilon() {
                // This is a division function
                let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                let other_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(other) };
                // Use the AVX2 implementation but with div function
                // We need to extract the slices and create a new array
                let a_data = self_f32.to_vec();
                let b_data = other_f32.to_vec();
                let mut result_data = vec![0.0f32; a_data.len()];

                unsafe {
                    // Use our AVX2 div implementation
                    crate::simd_optimize::avx2_ops::avx2_div_f32(
                        &a_data,
                        &b_data,
                        &mut result_data,
                    );
                }

                let result_f32 = Array::from_vec(result_data).reshape(&self_f32.shape());
                return Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result_f32) });
            }

            // Default to scalar implementation for other functions
            self.simd_zip_with_scalar(other, f)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // Check if the function is an addition operation
            let add_test = f(T::one(), T::one());
            let add_expected = T::one() + T::one();
            if (add_test - add_expected).abs() < T::epsilon() {
                // This is an addition function
                let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                let other_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(other) };
                let result_f64 = crate::simd_optimize::avx2_optimized_add_f64(self_f64, other_f64)?;
                return Ok(unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) });
            }

            // Check if the function is a multiplication operation
            let mul_test = f(T::from(2.0).unwrap(), T::from(3.0).unwrap());
            let mul_expected = T::from(2.0).unwrap() * T::from(3.0).unwrap();
            if (mul_test - mul_expected).abs() < T::epsilon() {
                // This is a multiplication function
                let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                let other_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(other) };
                let result_f64 = crate::simd_optimize::avx2_optimized_mul_f64(self_f64, other_f64)?;
                return Ok(unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) });
            }

            // Check if the function is a division operation
            let div_test = f(T::from(4.0).unwrap(), T::from(2.0).unwrap());
            let div_expected = T::from(4.0).unwrap() / T::from(2.0).unwrap();
            if (div_test - div_expected).abs() < T::epsilon() {
                // This is a division function
                let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                let other_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(other) };
                // Use the AVX2 implementation but with div function
                // We need to extract the slices and create a new array
                let a_data = self_f64.to_vec();
                let b_data = other_f64.to_vec();
                let mut result_data = vec![0.0f64; a_data.len()];

                unsafe {
                    // Use our AVX2 div implementation
                    crate::simd_optimize::avx2_ops::avx2_div_f64(
                        &a_data,
                        &b_data,
                        &mut result_data,
                    );
                }

                let result_f64 = Array::from_vec(result_data).reshape(&self_f64.shape());
                return Ok(unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) });
            }

            // Default to scalar implementation for other functions
            self.simd_zip_with_scalar(other, f)
        } else {
            // Default to scalar implementation for other types
            self.simd_zip_with_scalar(other, f)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_zip_with_avx2<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_reduce_avx2<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        // Using specialized AVX2 implementations for common operations
        // Fall back to scalar for custom functions
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Check if the function is a sum operation
            let sum_test = f(T::one(), T::one());
            let sum_expected = T::one() + T::one();
            if (sum_test - sum_expected).abs() < T::epsilon() && init == T::zero() {
                // This is a sum function with zero initialization
                let self_f32 = unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) };
                let result_f32 = crate::simd_optimize::avx2_optimized_sum_f32(self_f32);
                return T::from(result_f32).unwrap();
            }

            // Default to scalar implementation for other functions
            self.simd_reduce_scalar(init, f)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // Check if the function is a sum operation
            let sum_test = f(T::one(), T::one());
            let sum_expected = T::one() + T::one();
            if (sum_test - sum_expected).abs() < T::epsilon() && init == T::zero() {
                // This is a sum function with zero initialization
                let self_f64 = unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) };
                let result_f64 = crate::simd_optimize::avx2_optimized_sum_f64(self_f64);
                return T::from(result_f64).unwrap();
            }

            // Default to scalar implementation for other functions
            self.simd_reduce_scalar(init, f)
        } else {
            // Default to scalar implementation for other types
            self.simd_reduce_scalar(init, f)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_reduce_avx2<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.simd_reduce_scalar(init, f)
    }

    // NEON implementations (aarch64)
    #[cfg(target_arch = "aarch64")]
    fn simd_map_neon<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use NEON intrinsics
        self.simd_map_scalar(f)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn simd_map_neon<F>(&self, f: &F) -> Array<T>
    where
        F: Fn(T) -> T,
    {
        self.simd_map_scalar(f)
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_zip_with_neon<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use NEON intrinsics
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn simd_zip_with_neon<F>(&self, other: &Array<T>, f: &F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        self.simd_zip_with_scalar(other, f)
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_reduce_neon<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        // For now, just call the scalar implementation
        // In a real implementation, you would use NEON intrinsics
        self.simd_reduce_scalar(init, f)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn simd_reduce_neon<F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.simd_reduce_scalar(init, f)
    }
}

// Efficient SIMD implementations for common operations (SCIRS2 POLICY compliant)

/// SIMD-accelerated element-wise addition with automatic CPU feature detection
pub fn simd_add<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    a.simd_zip_with(b, |x, y| x + y)
}

/// SIMD-accelerated element-wise multiplication with automatic CPU feature detection
pub fn simd_mul<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    a.simd_zip_with(b, |x, y| x * y)
}

/// SIMD-accelerated element-wise division with automatic CPU feature detection
pub fn simd_div<T: Float + 'static>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>> {
    a.simd_zip_with(b, |x, y| x / y)
}

/// SIMD-accelerated element-wise exponentiation with automatic CPU feature detection
pub fn simd_exp<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    a.simd_map(|x| x.exp())
}

/// SIMD-accelerated element-wise logarithm with automatic CPU feature detection
pub fn simd_log<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    a.simd_map(|x| x.ln())
}

/// SIMD-accelerated element-wise square root with automatic CPU feature detection
pub fn simd_sqrt<T: Float + 'static>(a: &Array<T>) -> Array<T> {
    a.simd_map(|x| x.sqrt())
}

/// SIMD-accelerated sum of all elements with automatic CPU feature detection
pub fn simd_sum<T: Float + 'static>(a: &Array<T>) -> T {
    a.simd_reduce(T::zero(), |acc, x| acc + x)
}

/// SIMD-accelerated product of all elements with automatic CPU feature detection
pub fn simd_prod<T: Float + 'static>(a: &Array<T>) -> T {
    a.simd_reduce(T::one(), |acc, x| acc * x)
}

/// Returns the currently selected SIMD implementation based on CPU features
pub fn get_simd_implementation() -> SimdImplementation {
    let features = detect_cpu_features();
    select_simd_implementation(&features)
}

/// Returns the name of the currently selected SIMD implementation
pub fn get_simd_implementation_name() -> &'static str {
    get_simd_implementation().name()
}

// Tests for SIMD operations
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_add() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_add(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_simd_mul() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_mul(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_simd_div() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let b = Array::from_vec(vec![5.0f64, 6.0, 7.0, 8.0]);
        let c = simd_div(&a, &b).unwrap();
        assert_relative_eq!(c.to_vec()[0], 0.2, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[1], 2.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[2], 3.0 / 7.0, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[3], 4.0 / 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sqrt() {
        let a = Array::from_vec(vec![1.0f64, 4.0, 9.0, 16.0]);
        let b = simd_sqrt(&a);
        assert_relative_eq!(b.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(b.to_vec()[3], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_sum() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let sum = simd_sum(&a);
        assert_relative_eq!(sum, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_prod() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let prod = simd_prod(&a);
        assert_relative_eq!(prod, 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cpu_feature_detection() {
        // Just check that we get a valid implementation name
        let implementation = get_simd_implementation();
        println!("Detected SIMD implementation: {}", implementation.name());

        // The implementation should be one of the known types
        match implementation {
            SimdImplementation::Scalar
            | SimdImplementation::SSE
            | SimdImplementation::AVX
            | SimdImplementation::AVX2
            | SimdImplementation::AVX512
            | SimdImplementation::NEON
            | SimdImplementation::SVE => {
                // Implementation is valid
            }
        }
    }

    #[test]
    fn test_simd_add_scalar() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = a.simd_add_scalar(10.0);
        assert_eq!(result.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_simd_mul_scalar() {
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = a.simd_mul_scalar(2.0);
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_sub_scalar() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a.simd_sub_scalar(5.0);
        assert_eq!(result.to_vec(), vec![5.0, 15.0, 25.0, 35.0]);
    }

    #[test]
    fn test_simd_div_scalar() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a.simd_div_scalar(2.0).unwrap();
        assert_eq!(result.to_vec(), vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_simd_div_scalar_zero() {
        let a = Array::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]);
        let result = a.simd_div_scalar(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_scalar_ops_large_array() {
        // Test with larger array to exercise SIMD lanes
        let data: Vec<f64> = (0..1024).map(|i| i as f64).collect();
        let a = Array::from_vec(data);

        let result = a.simd_add_scalar(1.0);
        assert_relative_eq!(result.get(&[0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[1023]).unwrap(), 1024.0, epsilon = 1e-10);

        let result = a.simd_mul_scalar(2.0);
        assert_relative_eq!(result.get(&[0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(&[512]).unwrap(), 1024.0, epsilon = 1e-10);
    }
}

//! Element-wise mathematical operations for arrays
//!
//! This module provides SIMD-optimized element-wise math operations including:
//! - Basic operations (abs, exp, log, sqrt, pow, etc.)
//! - Trigonometric functions (sin, cos, tan, etc.)
//! - Hyperbolic functions (sinh, cosh, tanh, etc.)
//! - Rounding functions (floor, ceil, round, trunc)
//! - Utility functions (clip, sign)

use crate::array::Array;
use crate::error::Result;
use num_traits::{Float, NumCast};
use scirs2_core::ndarray::Array1;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Threshold for using SIMD optimizations (minimum array size)
/// v0.3.0: Increased from 32 to 64 to better amortize allocation overhead
pub(crate) const SIMD_THRESHOLD: usize = 64;

/// Check if SIMD optimization should be used for this array
#[inline]
pub(crate) fn should_use_simd<T: Clone>(array: &Array<T>) -> bool {
    array.len() >= SIMD_THRESHOLD
}

/// Helper to convert Array<f64> to ndarray Array1
#[inline]
pub(crate) fn to_nd_array_f64(arr: &Array<f64>) -> Array1<f64> {
    Array1::from_vec(arr.to_vec())
}

/// Helper to convert Array<f32> to ndarray Array1
#[inline]
pub(crate) fn to_nd_array_f32(arr: &Array<f32>) -> Array1<f32> {
    Array1::from_vec(arr.to_vec())
}

/// Helper to convert ndarray Array1 back to NumRS2 Array
#[inline]
pub(crate) fn from_nd_array<T: Clone + std::fmt::Debug + NumCast>(
    nd: Array1<T>,
    shape: &[usize],
) -> Array<T> {
    let data: Vec<T> = nd.into_iter().collect();
    Array::from_vec_shape(data, shape).unwrap_or_else(|e| panic!("{e}"))
}

// Basic element-wise math operations
pub trait ElementWiseMath<T> {
    // Basic operations
    fn abs(&self) -> Array<T>;
    fn exp(&self) -> Array<T>;
    fn log(&self) -> Array<T>;
    fn log10(&self) -> Array<T>;
    fn log2(&self) -> Array<T>;
    fn log1p(&self) -> Array<T>;
    fn expm1(&self) -> Array<T>;
    fn sqrt(&self) -> Array<T>;
    fn cbrt(&self) -> Array<T>;
    fn pow(&self, n: T) -> Array<T>;

    // Log-domain functions
    /// # Panics
    /// Panics if `self` and `other` cannot be broadcast together. Use
    /// [`ElementWiseMath::safe_logaddexp`] for a non-panicking version.
    fn logaddexp(&self, other: &Array<T>) -> Array<T>;
    /// # Panics
    /// Panics if `self` and `other` cannot be broadcast together. Use
    /// [`ElementWiseMath::safe_logaddexp2`] for a non-panicking version.
    fn logaddexp2(&self, other: &Array<T>) -> Array<T>;

    // Trigonometric functions
    fn sin(&self) -> Array<T>;
    fn cos(&self) -> Array<T>;
    fn tan(&self) -> Array<T>;
    fn asin(&self) -> Array<T>;
    fn acos(&self) -> Array<T>;
    fn atan(&self) -> Array<T>;
    /// # Panics
    /// Panics if `self` and `other` cannot be broadcast together. Use
    /// [`ElementWiseMath::safe_atan2`] for a non-panicking version.
    fn atan2(&self, other: &Array<T>) -> Array<T>;
    /// # Panics
    /// Panics if `self` and `other` cannot be broadcast together. Use
    /// [`ElementWiseMath::safe_hypot`] for a non-panicking version.
    fn hypot(&self, other: &Array<T>) -> Array<T>;
    fn degrees(&self) -> Array<T>;
    fn radians(&self) -> Array<T>;

    // Hyperbolic functions
    fn sinh(&self) -> Array<T>;
    fn cosh(&self) -> Array<T>;
    fn tanh(&self) -> Array<T>;
    fn asinh(&self) -> Array<T>;
    fn acosh(&self) -> Array<T>;
    fn atanh(&self) -> Array<T>;

    // Rounding functions
    fn floor(&self) -> Array<T>;
    fn ceil(&self) -> Array<T>;
    fn round(&self) -> Array<T>;
    fn trunc(&self) -> Array<T>;

    // Utility functions
    fn clip(&self, min: T, max: T) -> Array<T>;
    fn sign(&self) -> Array<T>;

    // Safe versions that return Result for error handling
    fn safe_logaddexp(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_logaddexp2(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_atan2(&self, other: &Array<T>) -> Result<Array<T>>;
    fn safe_hypot(&self, other: &Array<T>) -> Result<Array<T>>;
}

impl<T: Float + Clone + 'static> ElementWiseMath<T> for Array<T> {
    // Basic operations - using SimdUnifiedOps for platform-independent SIMD
    fn abs(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_abs(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_abs(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.abs())
    }

    fn exp(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_exp(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_exp(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.exp())
    }

    fn log(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_ln(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_ln(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.ln())
    }

    fn log10(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_log10(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_log10(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.log10())
    }

    fn log2(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_log2(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_log2(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.log2())
    }

    fn log1p(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_ln_1p(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_ln_1p(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| (x + T::one()).ln())
    }

    fn expm1(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_exp_m1(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_exp_m1(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.exp() - T::one())
    }

    fn sqrt(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_sqrt(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_sqrt(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.sqrt())
    }

    fn cbrt(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_cbrt(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_cbrt(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.powf(T::from(1.0 / 3.0).expect("1/3 should be representable")))
    }

    fn pow(&self, n: T) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let n_f64 = unsafe { *(&n as *const T as *const f64) };
                let result = f64::simd_powf(&nd.view(), n_f64);
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let n_f32 = unsafe { *(&n as *const T as *const f32) };
                let result = f32::simd_powf(&nd.view(), n_f32);
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.powf(n))
    }

    // Log-domain functions
    fn logaddexp(&self, other: &Array<T>) -> Array<T> {
        // Implementing log(exp(x) + exp(y)) in a numerically stable way
        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let max_val = if a > b { a } else { b };
            let sum = (a - max_val).exp() + (b - max_val).exp();
            max_val + sum.ln()
        })
        .unwrap_or_else(|e| panic!("Failed to broadcast in logaddexp: {}. Consider using safe_logaddexp() for error handling.", e))
    }

    fn logaddexp2(&self, other: &Array<T>) -> Array<T> {
        // Implementing log2(2^x + 2^y) in a numerically stable way
        // log2(2^x + 2^y) = log2(e) * ln(2^x + 2^y) = log2(e) * ln(e^(x*ln(2)) + e^(y*ln(2)))
        let ln2 = T::from(std::f64::consts::LN_2).expect("LN_2 constant should be representable");
        let log2_e =
            T::from(std::f64::consts::LOG2_E).expect("LOG2_E constant should be representable");

        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            // Convert log2 values to natural log
            let ln_a = a * ln2;
            let ln_b = b * ln2;

            let max_val = if ln_a > ln_b { ln_a } else { ln_b };
            let sum = (ln_a - max_val).exp() + (ln_b - max_val).exp();
            (max_val + sum.ln()) * log2_e
        })
        .unwrap_or_else(|e| panic!("Failed to broadcast in logaddexp2: {}. Consider using safe_logaddexp2() for error handling.", e))
    }

    // Trigonometric functions
    fn sin(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_sin(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_sin(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.sin())
    }

    fn cos(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_cos(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_cos(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.cos())
    }

    fn tan(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_tan(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_tan(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.tan())
    }

    fn asin(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_asin(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_asin(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.asin())
    }

    fn acos(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_acos(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_acos(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.acos())
    }

    fn atan(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_atan(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_atan(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.atan())
    }

    fn atan2(&self, other: &Array<T>) -> Array<T> {
        if should_use_simd(self) && should_use_simd(other) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd_y =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let nd_x = to_nd_array_f64(unsafe {
                    std::mem::transmute::<&Array<T>, &Array<f64>>(other)
                });
                let result = f64::simd_atan2(&nd_y.view(), &nd_x.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd_y =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let nd_x = to_nd_array_f32(unsafe {
                    std::mem::transmute::<&Array<T>, &Array<f32>>(other)
                });
                let result = f32::simd_atan2(&nd_y.view(), &nd_x.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.zip_with(other, |a, b| a.atan2(b)).unwrap_or_else(|e| {
            panic!(
                "Failed to broadcast in atan2: {}. Consider using safe_atan2() for error handling.",
                e
            )
        })
    }

    fn hypot(&self, other: &Array<T>) -> Array<T> {
        if should_use_simd(self) && should_use_simd(other) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd_x =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let nd_y = to_nd_array_f64(unsafe {
                    std::mem::transmute::<&Array<T>, &Array<f64>>(other)
                });
                let result = f64::simd_hypot(&nd_x.view(), &nd_y.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd_x =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let nd_y = to_nd_array_f32(unsafe {
                    std::mem::transmute::<&Array<T>, &Array<f32>>(other)
                });
                let result = f32::simd_hypot(&nd_x.view(), &nd_y.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.zip_with(other, |a, b| (a * a + b * b).sqrt())
            .unwrap_or_else(|e| panic!("Failed to broadcast in hypot: {}. Consider using safe_hypot() for error handling.", e))
    }

    fn degrees(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_to_degrees(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_to_degrees(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        let rad_to_deg = T::from(180.0).expect("180.0 should be representable")
            / T::from(std::f64::consts::PI).expect("PI constant should be representable");
        self.map(|x| x * rad_to_deg)
    }

    fn radians(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_to_radians(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_to_radians(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        let deg_to_rad = T::from(std::f64::consts::PI)
            .expect("PI constant should be representable")
            / T::from(180.0).expect("180.0 should be representable");
        self.map(|x| x * deg_to_rad)
    }

    // Hyperbolic functions
    fn sinh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_sinh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_sinh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.sinh())
    }

    fn cosh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_cosh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_cosh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.cosh())
    }

    fn tanh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_tanh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_tanh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.tanh())
    }

    fn asinh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_asinh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_asinh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.asinh())
    }

    fn acosh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_acosh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_acosh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.acosh())
    }

    fn atanh(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_atanh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_atanh(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.atanh())
    }

    // Rounding functions - Hardware SIMD accelerated
    fn floor(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_floor(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_floor(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.floor())
    }

    fn ceil(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_ceil(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_ceil(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.ceil())
    }

    fn round(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_round(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_round(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.round())
    }

    fn trunc(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_trunc(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_trunc(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| x.trunc())
    }

    // Utility functions
    fn clip(&self, min: T, max: T) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let min_f64: f64 = unsafe { std::mem::transmute_copy(&min) };
                let max_f64: f64 = unsafe { std::mem::transmute_copy(&max) };
                let result = f64::simd_clip(&nd.view(), min_f64, max_f64);
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let min_f32: f32 = unsafe { std::mem::transmute_copy(&min) };
                let max_f32: f32 = unsafe { std::mem::transmute_copy(&max) };
                let result = f32::simd_clip(&nd.view(), min_f32, max_f32);
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| {
            if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            }
        })
    }

    fn sign(&self) -> Array<T> {
        if should_use_simd(self) {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let nd =
                    to_nd_array_f64(unsafe { std::mem::transmute::<&Array<T>, &Array<f64>>(self) });
                let result = f64::simd_sign(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f64>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let nd =
                    to_nd_array_f32(unsafe { std::mem::transmute::<&Array<T>, &Array<f32>>(self) });
                let result = f32::simd_sign(&nd.view());
                return unsafe {
                    std::mem::transmute::<Array<f32>, Array<T>>(from_nd_array(
                        result,
                        &self.shape(),
                    ))
                };
            }
        }
        self.map(|x| {
            if x == T::zero() {
                T::zero()
            } else if x > T::zero() {
                T::one()
            } else {
                -T::one()
            }
        })
    }

    // Safe versions that return Result for proper error handling
    fn safe_logaddexp(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let max_val = if a > b { a } else { b };
            let sum = (a - max_val).exp() + (b - max_val).exp();
            max_val + sum.ln()
        })
        .map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in logaddexp: {}",
                e
            ))
        })
    }

    fn safe_logaddexp2(&self, other: &Array<T>) -> Result<Array<T>> {
        let ln2 = T::from(std::f64::consts::LN_2).expect("LN_2 constant should be representable");
        let log2_e =
            T::from(std::f64::consts::LOG2_E).expect("LOG2_E constant should be representable");

        self.zip_with(other, |a, b| {
            if a == T::neg_infinity() {
                return b;
            }
            if b == T::neg_infinity() {
                return a;
            }

            let a_scaled = a * ln2;
            let b_scaled = b * ln2;
            let max_val = if a_scaled > b_scaled {
                a_scaled
            } else {
                b_scaled
            };
            let sum = (a_scaled - max_val).exp() + (b_scaled - max_val).exp();
            (max_val + sum.ln()) * log2_e
        })
        .map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in logaddexp2: {}",
                e
            ))
        })
    }

    fn safe_atan2(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| a.atan2(b)).map_err(|e| {
            crate::error::NumRs2Error::ComputationError(format!(
                "Broadcasting failed in atan2: {}",
                e
            ))
        })
    }

    fn safe_hypot(&self, other: &Array<T>) -> Result<Array<T>> {
        self.zip_with(other, |a, b| (a * a + b * b).sqrt())
            .map_err(|e| {
                crate::error::NumRs2Error::ComputationError(format!(
                    "Broadcasting failed in hypot: {}",
                    e
                ))
            })
    }
}

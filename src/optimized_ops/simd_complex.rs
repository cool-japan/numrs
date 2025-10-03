//! SIMD-optimized operations for complex numbers
//!
//! This module provides SIMD-accelerated implementations for complex number
//! operations including arithmetic, transcendental functions, and FFT.

use crate::Result;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::Complex;
use std::f64::consts::PI;

/// SIMD-optimized complex number operations
pub struct SimdComplexOps;

impl SimdComplexOps {
    /// SIMD-optimized complex multiplication
    pub fn complex_multiply(
        a: &ArrayView1<Complex<f64>>,
        b: &ArrayView1<Complex<f64>>,
    ) -> Result<Array1<Complex<f64>>> {
        if a.len() != b.len() {
            return Err(crate::NumRs2Error::DimensionMismatch(format!(
                "Array lengths must match: {} != {}",
                a.len(),
                b.len()
            )));
        }

        // Extract real and imaginary parts
        let a_real: Array1<f64> = a.map(|c| c.re);
        let a_imag: Array1<f64> = a.map(|c| c.im);
        let b_real: Array1<f64> = b.map(|c| c.re);
        let b_imag: Array1<f64> = b.map(|c| c.im);

        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        // Real part: ac - bd
        let ac = f64::simd_mul(&a_real.view(), &b_real.view());
        let bd = f64::simd_mul(&a_imag.view(), &b_imag.view());
        let real_part = f64::simd_sub(&ac.view(), &bd.view());

        // Imaginary part: ad + bc
        let ad = f64::simd_mul(&a_real.view(), &b_imag.view());
        let bc = f64::simd_mul(&a_imag.view(), &b_real.view());
        let imag_part = f64::simd_add(&ad.view(), &bc.view());

        // Combine results
        let mut result = Array1::zeros(a.len());
        Zip::from(&mut result)
            .and(&real_part)
            .and(&imag_part)
            .for_each(|out, &re, &im| {
                *out = Complex::new(re, im);
            });

        Ok(result)
    }

    /// SIMD-optimized complex conjugate
    pub fn complex_conjugate(a: &ArrayView1<Complex<f64>>) -> Array1<Complex<f64>> {
        let real_part: Array1<f64> = a.map(|c| c.re);
        let imag_part: Array1<f64> = a.map(|c| c.im);

        // Negate imaginary part using SIMD multiplication by -1
        let neg_one = Array1::from_elem(imag_part.len(), -1.0);
        let neg_imag = f64::simd_mul(&imag_part.view(), &neg_one.view());

        // Combine results
        let mut result = Array1::zeros(a.len());
        Zip::from(&mut result)
            .and(&real_part)
            .and(&neg_imag)
            .for_each(|out, &re, &im| {
                *out = Complex::new(re, im);
            });

        result
    }

    /// SIMD-optimized complex magnitude (absolute value)
    pub fn complex_magnitude(a: &ArrayView1<Complex<f64>>) -> Array1<f64> {
        let real_part: Array1<f64> = a.map(|c| c.re);
        let imag_part: Array1<f64> = a.map(|c| c.im);

        // |a + bi| = sqrt(a² + b²)
        let real_squared = f64::simd_mul(&real_part.view(), &real_part.view());
        let imag_squared = f64::simd_mul(&imag_part.view(), &imag_part.view());
        let sum_squared = f64::simd_add(&real_squared.view(), &imag_squared.view());

        f64::simd_sqrt(&sum_squared.view())
    }

    /// SIMD-optimized complex phase (argument)
    pub fn complex_phase(a: &ArrayView1<Complex<f64>>) -> Array1<f64> {
        use crate::optimized_ops::enhanced_math::parallel_atan2;

        let real_part: Array1<f64> = a.map(|c| c.re);
        let imag_part: Array1<f64> = a.map(|c| c.im);

        // phase = atan2(imag, real)
        parallel_atan2(&imag_part.view(), &real_part.view())
            .unwrap_or_else(|_| Array1::zeros(a.len()))
    }

    /// SIMD-optimized complex exponential
    pub fn complex_exp(a: &ArrayView1<Complex<f64>>) -> Array1<Complex<f64>> {
        use crate::optimized_ops::enhanced_exp::parallel_exp;
        use crate::optimized_ops::enhanced_math::{parallel_cos, parallel_sin};

        let real_part: Array1<f64> = a.map(|c| c.re);
        let imag_part: Array1<f64> = a.map(|c| c.im);

        // e^(a + bi) = e^a * (cos(b) + i*sin(b))
        let exp_real = parallel_exp(&real_part.view());
        let cos_imag = parallel_cos(&imag_part.view());
        let sin_imag = parallel_sin(&imag_part.view());

        // Real part: e^a * cos(b)
        let result_real = f64::simd_mul(&exp_real.view(), &cos_imag.view());

        // Imaginary part: e^a * sin(b)
        let result_imag = f64::simd_mul(&exp_real.view(), &sin_imag.view());

        // Combine results
        let mut result = Array1::zeros(a.len());
        Zip::from(&mut result)
            .and(&result_real)
            .and(&result_imag)
            .for_each(|out, &re, &im| {
                *out = Complex::new(re, im);
            });

        result
    }
}

/// SIMD-optimized Fast Fourier Transform (FFT)
pub struct SimdFft;

impl SimdFft {
    /// Cooley-Tukey radix-2 FFT with SIMD optimization
    pub fn fft(input: &ArrayView1<Complex<f64>>) -> Result<Array1<Complex<f64>>> {
        let n = input.len();

        // Check if n is a power of 2
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(crate::NumRs2Error::InvalidOperation(
                "FFT input length must be a power of 2".to_string(),
            ));
        }

        let mut output = input.to_owned();
        Self::fft_recursive(&mut output.view_mut());
        Ok(output)
    }

    /// Recursive FFT implementation
    fn fft_recursive(data: &mut scirs2_core::ndarray::ArrayViewMut1<Complex<f64>>) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        // Divide
        let mut even = Array1::zeros(n / 2);
        let mut odd = Array1::zeros(n / 2);

        for i in 0..n / 2 {
            even[i] = data[2 * i];
            odd[i] = data[2 * i + 1];
        }

        // Conquer
        Self::fft_recursive(&mut even.view_mut());
        Self::fft_recursive(&mut odd.view_mut());

        // Combine using SIMD
        let angle_step = -2.0 * PI / n as f64;
        let angles: Array1<f64> =
            Array1::from_vec((0..n / 2).map(|k| angle_step * k as f64).collect());

        // Compute twiddle factors using SIMD
        let cos_angles = crate::optimized_ops::enhanced_math::parallel_cos(&angles.view());
        let sin_angles = crate::optimized_ops::enhanced_math::parallel_sin(&angles.view());

        let mut twiddle_factors = Array1::zeros(n / 2);
        Zip::from(&mut twiddle_factors)
            .and(&cos_angles)
            .and(&sin_angles)
            .for_each(|out, &c, &s| {
                *out = Complex::new(c, s);
            });

        // Apply twiddle factors using SIMD multiplication
        let twiddle_times_odd =
            SimdComplexOps::complex_multiply(&twiddle_factors.view(), &odd.view()).unwrap();

        // Combine results
        for k in 0..n / 2 {
            let t = twiddle_times_odd[k];
            data[k] = even[k] + t;
            data[k + n / 2] = even[k] - t;
        }
    }

    /// Inverse FFT using SIMD
    pub fn ifft(input: &ArrayView1<Complex<f64>>) -> Result<Array1<Complex<f64>>> {
        let n = input.len();

        // Take conjugate
        let conjugated = SimdComplexOps::complex_conjugate(input);

        // Apply FFT
        let fft_result = Self::fft(&conjugated.view())?;

        // Take conjugate again and scale
        let mut result = SimdComplexOps::complex_conjugate(&fft_result.view());
        let scale = 1.0 / n as f64;
        result.map_inplace(|c| *c *= scale);

        Ok(result)
    }

    /// 2D FFT with SIMD optimization
    pub fn fft2d(input: &ArrayView2<Complex<f64>>) -> Result<Array2<Complex<f64>>> {
        let (rows, cols) = input.dim();

        // Check dimensions are powers of 2
        if rows == 0 || (rows & (rows - 1)) != 0 || cols == 0 || (cols & (cols - 1)) != 0 {
            return Err(crate::NumRs2Error::InvalidOperation(
                "FFT2D dimensions must be powers of 2".to_string(),
            ));
        }

        let mut result = input.to_owned();

        // FFT along rows
        for i in 0..rows {
            let row_fft = Self::fft(&result.row(i))?;
            result.row_mut(i).assign(&row_fft);
        }

        // FFT along columns
        for j in 0..cols {
            let col_fft = Self::fft(&result.column(j))?;
            result.column_mut(j).assign(&col_fft);
        }

        Ok(result)
    }
}

/// SIMD-optimized convolution operations
pub struct SimdConvolution;

impl SimdConvolution {
    /// Fast convolution using FFT with SIMD
    pub fn convolve(signal: &ArrayView1<f64>, kernel: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let n = signal.len();
        let m = kernel.len();
        let output_len = n + m - 1;

        // Find next power of 2
        let fft_len = output_len.next_power_of_two();

        // Zero-pad inputs
        let mut padded_signal = Array1::zeros(fft_len);
        let mut padded_kernel = Array1::zeros(fft_len);

        padded_signal
            .slice_mut(scirs2_core::ndarray::s![..n])
            .assign(signal);
        padded_kernel
            .slice_mut(scirs2_core::ndarray::s![..m])
            .assign(kernel);

        // Convert to complex
        let complex_signal: Array1<Complex<f64>> = padded_signal.map(|&x| Complex::new(x, 0.0));
        let complex_kernel: Array1<Complex<f64>> = padded_kernel.map(|&x| Complex::new(x, 0.0));

        // FFT
        let signal_fft = SimdFft::fft(&complex_signal.view())?;
        let kernel_fft = SimdFft::fft(&complex_kernel.view())?;

        // Multiply in frequency domain
        let product = SimdComplexOps::complex_multiply(&signal_fft.view(), &kernel_fft.view())?;

        // Inverse FFT
        let result_complex = SimdFft::ifft(&product.view())?;

        // Extract real part and trim to output length
        let result_real: Array1<f64> = result_complex.map(|c| c.re);
        Ok(result_real
            .slice(scirs2_core::ndarray::s![..output_len])
            .to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_multiply() {
        let a = Array1::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let b = Array1::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);

        let result = SimdComplexOps::complex_multiply(&a.view(), &b.view()).unwrap();

        // (1 + 2i)(5 + 6i) = 5 + 6i + 10i + 12i² = 5 + 16i - 12 = -7 + 16i
        assert_relative_eq!(result[0].re, -7.0, epsilon = 1e-10);
        assert_relative_eq!(result[0].im, 16.0, epsilon = 1e-10);

        // (3 + 4i)(7 + 8i) = 21 + 24i + 28i + 32i² = 21 + 52i - 32 = -11 + 52i
        assert_relative_eq!(result[1].re, -11.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].im, 52.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_magnitude() {
        let a = Array1::from_vec(vec![Complex::new(3.0, 4.0), Complex::new(5.0, 12.0)]);

        let result = SimdComplexOps::complex_magnitude(&a.view());

        assert_relative_eq!(result[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 13.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fft_simple() {
        let input = Array1::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);

        let result = SimdFft::fft(&input.view()).unwrap();

        // DC component should be 4
        assert_relative_eq!(result[0].re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-10);

        // Other components should be 0
        for i in 1..4 {
            assert_relative_eq!(result[i].norm(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_inverse() {
        let input = Array1::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ]);

        let fft_result = SimdFft::fft(&input.view()).unwrap();
        let ifft_result = SimdFft::ifft(&fft_result.view()).unwrap();

        // Should recover original signal
        for i in 0..4 {
            assert_relative_eq!(ifft_result[i].re, input[i].re, epsilon = 1e-10);
            assert_relative_eq!(ifft_result[i].im, input[i].im, epsilon = 1e-10);
        }
    }
}

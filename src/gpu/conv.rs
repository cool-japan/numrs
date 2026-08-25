//! GPU 2-D convolution built on im2col plus GEMM.
//!
//! The convolution is lowered the same way cuDNN's `IMPLICIT_GEMM` and
//! PyTorch's CPU path lower it: a shader materialises the patch matrix
//! (im2col) and the existing tiled GEMM kernel multiplies the reshaped
//! weights by it. That reuses the fastest kernel the backend has instead of
//! writing a second, slower direct-convolution shader.
//!
//! ```text
//!   input   [N, Cin, H, W]      --im2col-->  col  [Cin*KH*KW, N*OH*OW]
//!   weights [Cout, Cin, KH, KW] --reshape--> w2d  [Cout, Cin*KH*KW]
//!   w2d * col                                out  [Cout, N*OH*OW]
//!   out     --permute + reshape-->           [N, Cout, OH, OW]
//! ```
//!
//! The im2col kernel copies raw 32-bit words and writes zero words for the
//! padded positions, so it is exact for every element type; the GEMM step is
//! f32/f64 only, which is what bounds [`conv2d`]'s element type.

use crate::error::{NumRs2Error, Result};
use crate::gpu::array::GpuArray;
use crate::gpu::kernel::{
    dispatch, linear_dispatch, meta_buffer, to_u32, words_per_element, Binding,
};
use crate::gpu::nd;

/// Geometry of a 2-D convolution.
///
/// Values are `(height, width)` pairs, matching the axis order of an NCHW
/// tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dParams {
    /// Step between two neighbouring output positions.
    pub stride: (usize, usize),
    /// Number of zero rows/columns added on *each* side of the input.
    pub padding: (usize, usize),
    /// Spacing between the sampled kernel taps (1 = dense kernel).
    pub dilation: (usize, usize),
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }
}

impl Conv2dParams {
    /// Convolution with the given stride and padding and a dense kernel.
    pub fn new(stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self {
            stride,
            padding,
            dilation: (1, 1),
        }
    }

    /// Returns a copy of these parameters with the given dilation.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Validates the parameters and returns the output spatial extent.
    ///
    /// Uses the standard floor formula
    /// `out = (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1`.
    pub fn output_size(
        &self,
        input_size: (usize, usize),
        kernel_size: (usize, usize),
    ) -> Result<(usize, usize)> {
        if self.stride.0 == 0 || self.stride.1 == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Convolution stride must be non-zero".to_string(),
            ));
        }
        if self.dilation.0 == 0 || self.dilation.1 == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Convolution dilation must be non-zero".to_string(),
            ));
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Convolution kernel must have a non-zero extent".to_string(),
            ));
        }

        let effective_h = self.dilation.0 * (kernel_size.0 - 1) + 1;
        let effective_w = self.dilation.1 * (kernel_size.1 - 1) + 1;
        let padded_h = input_size.0 + 2 * self.padding.0;
        let padded_w = input_size.1 + 2 * self.padding.1;

        if padded_h < effective_h || padded_w < effective_w {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Kernel {:?} (dilated to {}x{}) does not fit in the padded input {}x{}",
                kernel_size, effective_h, effective_w, padded_h, padded_w
            )));
        }

        Ok((
            (padded_h - effective_h) / self.stride.0 + 1,
            (padded_w - effective_w) / self.stride.1 + 1,
        ))
    }
}

/// Materialises the im2col patch matrix of an NCHW input on the GPU.
///
/// The result has shape `[channels * kernel_h * kernel_w, batch * out_h *
/// out_w]`: column `((n * out_h) + oh) * out_w + ow` holds the receptive field
/// of output position `(n, oh, ow)`, with zeros where the field falls outside
/// the padded input.
pub fn im2col<T: bytemuck::Pod + bytemuck::Zeroable>(
    input: &GpuArray<T>,
    kernel_size: (usize, usize),
    params: &Conv2dParams,
) -> Result<GpuArray<T>> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "im2col expects an NCHW input with 4 dimensions, got shape {:?}",
            shape
        )));
    }

    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let (out_h, out_w) = params.output_size((in_h, in_w), kernel_size)?;

    let rows = channels * kernel_size.0 * kernel_size.1;
    let cols = batch * out_h * out_w;
    let n_elements = rows * cols;
    if n_elements == 0 {
        return Err(NumRs2Error::InvalidOperation(format!(
            "im2col of input {:?} with kernel {:?} produces an empty matrix",
            shape, kernel_size
        )));
    }

    let context = input.context().clone();
    let result = GpuArray::<T>::new_with_shape(&[rows, cols], context.clone())?;
    let (groups_x, groups_y) = linear_dispatch(&context, n_elements)?;

    let meta = [
        to_u32(n_elements, "patch matrix size")?,
        words_per_element::<T>()?,
        groups_x,
        to_u32(batch, "batch size")?,
        to_u32(channels, "channel count")?,
        to_u32(in_h, "input height")?,
        to_u32(in_w, "input width")?,
        to_u32(kernel_size.0, "kernel height")?,
        to_u32(kernel_size.1, "kernel width")?,
        to_u32(out_h, "output height")?,
        to_u32(out_w, "output width")?,
        to_u32(params.stride.0, "vertical stride")?,
        to_u32(params.stride.1, "horizontal stride")?,
        to_u32(params.padding.0, "vertical padding")?,
        to_u32(params.padding.1, "horizontal padding")?,
        to_u32(params.dilation.0, "vertical dilation")?,
        to_u32(params.dilation.1, "horizontal dilation")?,
    ];
    let meta = meta_buffer(&context, "im2col Metadata", &meta);

    dispatch(
        &context,
        context.im2col_shader(),
        "im2col",
        "NumRS2 im2col",
        &[
            Binding::Storage(input.buffer()),
            Binding::StorageMut(result.buffer()),
            Binding::Storage(&meta),
        ],
        (groups_x, groups_y, 1),
    );

    Ok(result)
}

/// Runs a 2-D convolution (cross-correlation) on the GPU.
///
/// * `input` - NCHW tensor `[batch, in_channels, height, width]`
/// * `weights` - `[out_channels, in_channels, kernel_h, kernel_w]`
///
/// Returns `[batch, out_channels, out_height, out_width]`. Like PyTorch's
/// `conv2d` and every other deep-learning framework, this computes a
/// cross-correlation: the kernel is *not* flipped.
///
/// # Errors
///
/// Returns an error if the ranks or channel counts disagree, if the kernel
/// does not fit the padded input, or if the element type is neither f32 nor
/// f64 (the GEMM step has no other kernels).
pub fn conv2d<T: bytemuck::Pod + bytemuck::Zeroable>(
    input: &GpuArray<T>,
    weights: &GpuArray<T>,
    params: &Conv2dParams,
) -> Result<GpuArray<T>> {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let is_f64 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    if !is_f32 && !is_f64 {
        return Err(NumRs2Error::TypeCastError(
            "GPU convolution only supports f32 and f64 element types".to_string(),
        ));
    }

    let input_shape = input.shape();
    let weight_shape = weights.shape();
    if input_shape.len() != 4 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "conv2d expects an NCHW input with 4 dimensions, got shape {:?}",
            input_shape
        )));
    }
    if weight_shape.len() != 4 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "conv2d expects weights with 4 dimensions [out_channels, in_channels, kh, kw], got shape {:?}",
            weight_shape
        )));
    }
    if input_shape[1] != weight_shape[1] {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "conv2d input has {} channels but the weights expect {}",
            input_shape[1], weight_shape[1]
        )));
    }

    let batch = input_shape[0];
    let out_channels = weight_shape[0];
    let kernel_size = (weight_shape[2], weight_shape[3]);
    let (out_h, out_w) = params.output_size((input_shape[2], input_shape[3]), kernel_size)?;

    // [Cin*KH*KW, N*OH*OW]
    let col = im2col(input, kernel_size, params)?;
    let patch_len = weight_shape[1] * kernel_size.0 * kernel_size.1;

    // [Cout, Cin*KH*KW] * [Cin*KH*KW, N*OH*OW] = [Cout, N*OH*OW]
    let weights_2d = weights.reshape(&[out_channels, patch_len])?;
    let product = crate::gpu::ops::matmul(&weights_2d, &col)?;

    // [Cout, N, OH*OW] -> [N, Cout, OH*OW] -> [N, Cout, OH, OW]
    let grouped = product.reshape(&[out_channels, batch, out_h * out_w])?;
    let batched = nd::permute_axes(&grouped, &[1, 0, 2])?;
    batched.reshape(&[batch, out_channels, out_h, out_w])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_size_matches_reference_formula() {
        let params = Conv2dParams::default();
        assert_eq!(
            params.output_size((5, 5), (3, 3)).ok(),
            Some((3, 3)),
            "valid convolution shrinks by kernel - 1"
        );

        let padded = Conv2dParams::new((1, 1), (1, 1));
        assert_eq!(
            padded.output_size((5, 5), (3, 3)).ok(),
            Some((5, 5)),
            "same padding preserves the extent"
        );

        let strided = Conv2dParams::new((2, 2), (1, 1));
        assert_eq!(strided.output_size((7, 7), (3, 3)).ok(), Some((4, 4)));

        let dilated = Conv2dParams::default().with_dilation((2, 2));
        assert_eq!(dilated.output_size((7, 7), (3, 3)).ok(), Some((3, 3)));
    }

    #[test]
    fn test_output_size_rejects_impossible_geometry() {
        assert!(Conv2dParams::default().output_size((2, 2), (3, 3)).is_err());
        assert!(Conv2dParams::new((0, 1), (0, 0))
            .output_size((5, 5), (3, 3))
            .is_err());
        assert!(Conv2dParams::default()
            .with_dilation((0, 1))
            .output_size((5, 5), (3, 3))
            .is_err());
    }
}
